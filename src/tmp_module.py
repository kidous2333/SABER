"""
SAGA (Spatially Aggregated Global Attention) neural network module.

Custom pose estimation backbone components that replace standard C3k2 blocks
with attention-enhanced variants for improved keypoint detection on mice.

Architecture:
  - SAGA: Spatially Aggregated Global Attention block (core attention module)
    - PSA: Partitioned Self-Attention (intra-group)
    - GCCA: Group Cross-Correlation Attention (inter-group)
  - Bottleneck_SAGA: Bottleneck with embedded SAGA
  - SASA: C2f variant using Bottleneck_SAGA (drop-in replacement for C3k2)
"""

import math
import warnings
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import ultralytics.nn.modules as _ulm
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import Bottleneck, C2f, C3


# ------------------------------------------------------------------
#  Math / tensor helpers
# ------------------------------------------------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)
    with torch.no_grad():
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def exists(val):
    return val is not None


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def is_empty(t):
    return t.nelement() == 0


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def center_iter(x, means, buckets=None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means


def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


# ------------------------------------------------------------------
#  Sub-layers
# ------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class PSA(nn.Module):
    """Partitioned Self-Attention — intra-group attention over spatially partitioned tokens."""
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size

    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        gs = min(N, self.group_size)
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N

        paded_q = torch.cat((q, torch.flip(q[:, N - pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d", ng=ng, h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2, 2 * gs, gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2, 2 * gs, gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        out1 = F.scaled_dot_product_attention(paded_q, paded_k, paded_v)

        k_global = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
        v_global = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)

        out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]

        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)

        return out


class GCCA(nn.Module):
    """Group Cross-Correlation Attention — inter-group relational cross-attention."""
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, normed_x, x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
        else:
            x_global = x_means

        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k, v, x_global.detach()


# ------------------------------------------------------------------
#  Core SAGA block
# ------------------------------------------------------------------
class SAGA(nn.Module):
    """Spatially Aggregated Global Attention — groups spatial tokens and applies
    intra-group (PSA) and inter-group (GCCA) attention for global context modeling."""
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=3,
                 num_tokens=8, group_size=128, ema_decay=0.999):
        super().__init__()

        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN(dim, mlp_dim))
        self.gcca_attn = GCCA(dim, qk_dim, heads)
        self.psa_attn = PSA(dim, qk_dim, heads, group_size)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, 'b c h w->b (h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape

        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:, N - pad_n:N, :], dims=[-2])), dim=-2)
            x_means = torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c', cnt=self.num_tokens), dim=-2).detach()
        else:
            x_means = self.means.detach()

        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))

        k_global, v_global, x_means = self.gcca_attn(x, x_means)

        with torch.no_grad():
            x_scores = torch.einsum('b i c,j c->b i j',
                                    F.normalize(x, dim=-1),
                                    F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)

            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)

        y = self.psa_attn(x, idx_last, k_global, v_global)
        y = rearrange(y, 'b (h w) c->b c h w', h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c h w->b (h w) c')
        x = self.mlp(x, x_size=(h, w)) + x

        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    ema_inplace(self.means, new_means, self.ema_decay)

        return rearrange(x, 'b (h w) c->b c h w', h=h)


# ------------------------------------------------------------------
#  Bottleneck with embedded SAGA
# ------------------------------------------------------------------
class Bottleneck_SAGA(nn.Module):
    """Bottleneck with embedded SAGA for global context enhancement."""

    def __init__(
            self,
            c1: int,
            c2: int,
            shortcut: bool = True,
            g: int = 1,
            k: Tuple[int, int] = (3, 3),
            e: float = 0.5,
            saga_heads: int = 4,
            saga_mlp_ratio: float = 2.0,
            saga_num_tokens: int = 8,
            saga_group_size: int = 128,
            use_saga: bool = True
    ):
        super().__init__()

        c_ = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)

        self.use_saga = use_saga
        if self.use_saga:
            mlp_dim = int(c_ * saga_mlp_ratio)
            qk_dim = c_

            self.saga = SAGA(
                dim=c_,
                qk_dim=qk_dim,
                mlp_dim=mlp_dim,
                heads=saga_heads,
                num_tokens=saga_num_tokens,
                group_size=saga_group_size
            )
        else:
            self.saga = nn.Identity()

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv1 = self.cv1(x)

        if self.use_saga:
            x_saga = self.saga(x_conv1)
        else:
            x_saga = x_conv1

        x_conv2 = self.cv2(x_saga)

        if self.add:
            return x + x_conv2
        else:
            return x_conv2


# ------------------------------------------------------------------
#  SASA: Spatial Aggregated Self-Attention block (C2f variant)
# ------------------------------------------------------------------
class C3k_SAGA(C3):
    """C3k with Bottleneck_SAGA replacing standard bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck_SAGA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class SASA(C2f):
    """Spatial Aggregated Self-Attention — C2f variant with C3k_SAGA for the main branch.
    Drop-in replacement for C3k2 in backbone architectures."""

    def __init__(
        self, c1: int, c2: int, n: int = 1, c3k: bool = False, e: float = 0.5, g: int = 1, shortcut: bool = True
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_SAGA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
