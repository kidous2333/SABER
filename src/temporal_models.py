"""
temporal_models.py
Unified temporal model module: Transformer and Mamba (SSM).

All models accept per-frame probability distributions [T, n_classes] from the
first-layer model as input, and output refined per-frame probability
distributions [T, n_classes].

Supported model types (specified via temporal_model config):
  - "lgbm"        : Sliding window statistical features + LightGBM (handled in temporal_validator.py)
  - "bilstm"      : Bidirectional LSTM (defined in bilstm_temporal.py, called uniformly here)
  - "transformer" : Multi-head self-attention Transformer Encoder
  - "mamba"       : State space model (requires mamba-ssm library: pip install mamba-ssm)
"""

import logging
import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ── CUDA memory fragmentation protection ──
# expandable_segments:True lets PyTorch reuse large allocated memory segments
# rather than requesting new ones, avoiding "CUDA out of memory" caused by
# residual fragmentation between multi-stage training (LGBM -> BiLSTM).
# Only set on the first import of this module to avoid repeated setenv warnings.
if not os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ---------------------------------------------------------------------------
# Focal Loss (handles class imbalance)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights [n_classes] or float
        gamma: Focusing parameter; larger gamma focuses more on hard samples (default 2.0)
        reduction: "mean" | "sum"
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                alpha = torch.tensor([alpha])
            else:
                alpha = torch.as_tensor(alpha, dtype=torch.float)
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, C] unnormalized log probabilities
        targets: [N] integer labels
        """
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")  # [N]
        pt = torch.exp(-ce)  # p_t for the true class

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            if alpha.numel() == 1:
                alpha_t = alpha
            else:
                alpha_t = alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTemporalModel(nn.Module):
    """
    Transformer Encoder temporal refinement model.

    Architecture: Linear projection -> PositionalEncoding -> TransformerEncoder -> LayerNorm -> projection head
    Input: [batch, seq_len, input_dim]
    Output: [batch, seq_len, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = None,
        use_layer_norm: bool = True,
        use_deep_projection: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        if use_deep_projection:
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim or input_dim),
            )
        else:
            self.output_proj = nn.Linear(d_model, output_dim or input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.ln(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Mamba Model
# ---------------------------------------------------------------------------

class MambaTemporalModel(nn.Module):
    """
    Mamba (SSM) temporal refinement model.

    Depends on mamba-ssm library: pip install mamba-ssm
    Architecture: Linear projection -> N x Mamba blocks -> LayerNorm -> projection head
    Input: [batch, seq_len, input_dim]
    Output: [batch, seq_len, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = None,
        use_layer_norm: bool = True,
        use_deep_projection: bool = False,
    ):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError:
            raise ImportError(
                "Mamba model requires the mamba-ssm library. Install it first: pip install mamba-ssm\n"
                "Note: mamba-ssm requires a CUDA environment and may need to be compiled from source on Windows."
            )

        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            nn.Sequential(
                Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        if use_deep_projection:
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim or input_dim),
            )
        else:
            self.output_proj = nn.Linear(d_model, output_dim or input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = x + layer(x)  # residual connection
        x = self.final_ln(x)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Unified Training Entry Point
# ---------------------------------------------------------------------------

def train_sequence_model(
    model_type: str,
    proba_train: np.ndarray,
    y_train: np.ndarray,
    proba_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    cfg: dict,
    extra_train: "np.ndarray | None" = None,
    extra_val: "np.ndarray | None" = None,
    save_dir: str = "",
) -> tuple:
    """
    Unified training entry point, supporting bilstm / transformer / mamba.

    cfg corresponds to the temporal_validation.seq_model section in validation.yaml,
    along with top-level temporal_validation optimization toggles:
      - use_focal_loss / focal_alpha / focal_gamma
      - use_class_weights
      - seq_output_residual
      - seq_use_extra_features (used with extra_train / extra_val)

    save_dir: If non-empty, save the best model weights to this directory.

    Returns (model, history).
    """
    hidden_dim   = int(cfg.get("hidden_dim",   128))
    num_layers   = int(cfg.get("num_layers",   2))
    dropout      = float(cfg.get("dropout",    0.3))
    chunk_size   = int(cfg.get("chunk_size",   512))
    stride_train = int(cfg.get("stride_train", 256))
    stride_val   = int(cfg.get("stride_val",   512))
    batch_size   = int(cfg.get("batch_size",   32))
    epochs       = int(cfg.get("epochs",       20))
    lr           = float(cfg.get("lr",         1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    device_str   = cfg.get("device", "cuda")
    patience     = int(cfg.get("early_stopping_patience", 5))
    monitor_metric = cfg.get("monitor_metric", "val_acc")

    # ── Optimization toggles ──
    _use_focal          = bool(cfg.get("use_focal_loss",     False))
    _focal_alpha        = float(cfg.get("focal_alpha",       0.25))
    _focal_gamma        = float(cfg.get("focal_gamma",       2.0))
    _use_class_weights  = bool(cfg.get("use_class_weights",  False))
    # AMP mixed precision: float16 forward + backward, float32 weight updates, ~40-50% memory savings
    _use_amp_default    = (device_str == "cuda")  # enabled by default on GPU
    _use_amp            = bool(cfg.get("use_amp", _use_amp_default))
    _output_residual    = bool(cfg.get("seq_output_residual", False))
    _use_extra          = bool(cfg.get("seq_use_extra_features", False))

    # ── Architecture enhancement toggles ──
    _use_layer_norm     = bool(cfg.get("use_layer_norm",     True))
    _use_deep_proj      = bool(cfg.get("use_deep_projection", False))
    _use_attn_pool      = bool(cfg.get("use_attention_pooling", False))

    # ── Training strategy enhancement toggles ──
    _label_smoothing    = float(cfg.get("label_smoothing",   0.0))
    _use_cosine_warmup  = bool(cfg.get("use_cosine_warmup",  True))
    _warmup_epochs      = int(cfg.get("warmup_epochs",       10))
    _use_ema            = bool(cfg.get("use_ema",            False))
    _ema_decay          = float(cfg.get("ema_decay",         0.999))
    _use_balanced_samp  = bool(cfg.get("use_balanced_sampling", True))

    # ── Data augmentation toggles ──
    _use_mixup          = bool(cfg.get("use_mixup",          False))
    _mixup_alpha        = float(cfg.get("mixup_alpha",       0.2))
    _use_time_reversal  = bool(cfg.get("use_time_reversal",  False))

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Note: On new GPU / driver environments, cudnn.deterministic=True may
        # cause cuDNN to hang if it cannot find a deterministic algorithm.
        # Only enable when available.
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            logger.warning(f"[{model_type}] cudnn deterministic setup failed, continuing")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(f"[{model_type}] Training device: {device} ({torch.cuda.get_device_name(0)})")
        # ── Memory fragmentation cleanup ──
        # The preceding LGBM pipeline may leave fragmented GPU memory.
        # Force the PyTorch caching allocator to return all unused memory
        # to the CUDA driver, allowing subsequent BiLSTM large allocations
        # to find contiguous space.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        _alloc_gb = torch.cuda.memory_allocated() / (1024**3)
        _reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        logger.info(
            f"[{model_type}] GPU memory state (after cleanup): allocated={_alloc_gb:.2f}GiB, "
            f"reserved={_reserved_gb:.2f}GiB, "
            f"free={(torch.cuda.get_device_properties(0).total_memory/1024**3 - _reserved_gb):.2f}GiB"
        )
    else:
        logger.info(f"[{model_type}] Training device: CPU (GPU not available)")

    import time as _time
    logger.info(f"[{model_type}] Step 1/6: Preparing feature concatenation (proba_train={proba_train.shape} {proba_train.dtype}, proba_val={proba_val.shape} {proba_val.dtype})...")
    # ── Extra feature concatenation ──
    if _use_extra and extra_train is not None and extra_val is not None:
        _t0 = _time.time()
        logger.info(
            f"[{model_type}] seq_use_extra_features：extra_train={extra_train.shape} {extra_train.dtype}, "
            f"extra_val={extra_val.shape} {extra_val.dtype}"
        )
        # Optimization: pre-allocate with np.empty + copyto to avoid temporary arrays from hstack
        _extra_dim = extra_train.shape[1]
        _new_dim = n_classes + _extra_dim
        # train
        _t1 = _time.time()
        _new_train = np.empty((proba_train.shape[0], _new_dim), dtype=np.float32)
        _new_train[:, :n_classes] = proba_train
        _new_train[:, n_classes:] = extra_train
        proba_train = _new_train
        logger.info(f"[{model_type}] train concat done: elapsed {_time.time() - _t1:.2f}s, shape={proba_train.shape}")
        # val
        _t2 = _time.time()
        _new_val = np.empty((proba_val.shape[0], _new_dim), dtype=np.float32)
        _new_val[:, :n_classes] = proba_val
        _new_val[:, n_classes:] = extra_val
        proba_val = _new_val
        logger.info(f"[{model_type}] val concat done: elapsed {_time.time() - _t2:.2f}s, shape={proba_val.shape}")
        logger.info(
            f"[{model_type}] seq_use_extra_features total elapsed {_time.time() - _t0:.2f}s: "
            f"input dim after concat {_new_dim} (n_classes={n_classes} + extra={_extra_dim})"
        )
    else:
        _extra_dim = 0

    _input_dim = n_classes + _extra_dim

    # Build model
    logger.info(f"[{model_type}] Step 2/6: Building model (input_dim={_input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers})...")
    if model_type == "bilstm":
        from src.bilstm_temporal import BiLSTMTemporalModel
        _attn_nhead = int(cfg.get("nhead", 4))
        model = BiLSTMTemporalModel(
            input_dim=_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=n_classes,
            use_layer_norm=_use_layer_norm,
            use_deep_projection=_use_deep_proj,
            use_attention_pooling=_use_attn_pool,
            nhead=_attn_nhead,
        )
        _arch_flags = []
        if _use_layer_norm: _arch_flags.append("LayerNorm")
        if _use_deep_proj:  _arch_flags.append("DeepProj")
        if _use_attn_pool:  _arch_flags.append(f"AttnPool(nhead={_attn_nhead})")
        logger.info(f"[{model_type}] Architecture: {' + '.join(_arch_flags) if _arch_flags else 'vanilla'}")
        logger.info(f"[{model_type}] Step 3/6: Moving model to {device}...")
        model = model.to(device)
    elif model_type == "transformer":
        d_model = int(cfg.get("d_model", hidden_dim))
        nhead   = int(cfg.get("nhead", 4))
        while d_model % nhead != 0 and nhead > 1:
            nhead -= 1
        model = TransformerTemporalModel(
            input_dim=_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=n_classes,
            use_layer_norm=_use_layer_norm,
            use_deep_projection=_use_deep_proj,
        )
        _arch_flags = []
        if _use_layer_norm: _arch_flags.append("LayerNorm")
        if _use_deep_proj:  _arch_flags.append("DeepProj")
        logger.info(f"[{model_type}] Architecture: {' + '.join(_arch_flags) if _arch_flags else 'vanilla'}")
        logger.info(f"[{model_type}] Step 3/6: Moving model to {device}...")
        model = model.to(device)
        logger.info(f"[Transformer] d_model={d_model}, nhead={nhead}, num_layers={num_layers}, input_dim={_input_dim}")
    elif model_type == "mamba":
        d_model = int(cfg.get("d_model", hidden_dim))
        d_state = int(cfg.get("d_state", 16))
        d_conv  = int(cfg.get("d_conv",  4))
        expand  = int(cfg.get("expand",  2))
        model = MambaTemporalModel(
            input_dim=_input_dim,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=n_classes,
            use_layer_norm=_use_layer_norm,
            use_deep_projection=_use_deep_proj,
        )
        _arch_flags = []
        if _use_layer_norm: _arch_flags.append("LayerNorm")
        if _use_deep_proj:  _arch_flags.append("DeepProj")
        logger.info(f"[{model_type}] Architecture: {' + '.join(_arch_flags) if _arch_flags else 'vanilla'}")
        logger.info(f"[{model_type}] Step 3/6: Moving model to {device}...")
        model = model.to(device)
        logger.info(f"[Mamba] d_model={d_model}, d_state={d_state}, d_conv={d_conv}, expand={expand}, input_dim={_input_dim}")
    else:
        raise ValueError(f"Unknown temporal model type: {model_type}, supported: bilstm / transformer / mamba")

    # Sync CUDA to ensure model migration is complete (may hang asynchronously on new GPU drivers)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    logger.info(f"[{model_type}] Step 4/6: Creating dataset...")

    # Dataset
    from src.bilstm_temporal import ProbabilityDataset
    train_ds = ProbabilityDataset(proba_train, y_train, chunk_size, stride_train)
    val_ds   = ProbabilityDataset(proba_val,   y_val,   chunk_size, stride_val)

    # pin_memory may hang on some server NUMA / driver configurations; try first, fallback on failure
    logger.info(f"[{model_type}] Step 5/6: Creating DataLoader...")
    _sampler = None
    if _use_balanced_samp:
        # Weight by dominant class per chunk: weight = 1 / (class_freq + 1)
        _y_arr = np.asarray(y_train).astype(int)
        _class_counts = np.bincount(_y_arr, minlength=n_classes).astype(np.float32)
        _class_weights_sample = 1.0 / (_class_counts + 1.0)
        _class_weights_sample /= _class_weights_sample.sum()
        # Each chunk's weight = mean class weight of labels in that chunk
        _chunk_weights = np.zeros(len(train_ds), dtype=np.float64)
        for _ci, (_s, _e) in enumerate(train_ds.indices):
            _chunk_labels = _y_arr[_s:_e]
            _chunk_weights[_ci] = _class_weights_sample[_chunk_labels].mean()
        _chunk_weights /= _chunk_weights.sum()
        _sampler = torch.utils.data.WeightedRandomSampler(
            _chunk_weights, num_samples=len(train_ds), replacement=True,
        )
        logger.info(
            f"[{model_type}] balanced_sampling: class weights={_class_weights_sample.round(3).tolist()}"
        )
    try:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=_sampler,
            shuffle=(_sampler is None),
            num_workers=0, pin_memory=True,
        )
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    except (RuntimeError, SystemError) as e:
        logger.warning(f"[{model_type}] pin_memory=True failed ({e}), falling back to pin_memory=False")
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=_sampler,
            shuffle=(_sampler is None),
            num_workers=0, pin_memory=False,
        )
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    logger.info(f"[{model_type}] Step 6/6: Training set {len(train_ds)} chunks, validation set {len(val_ds)} chunks")

    # ── Loss function ──
    _ce_kwargs = {}
    if _label_smoothing > 0 and not _use_focal:
        _ce_kwargs["label_smoothing"] = _label_smoothing
        logger.info(f"[{model_type}] Label Smoothing: {_label_smoothing}")
    if _use_focal:
        if _label_smoothing > 0:
            logger.warning(
                f"[{model_type}] FocalLoss does not support label_smoothing, ignoring label_smoothing={_label_smoothing}"
            )
        if _use_class_weights:
            class_counts = np.bincount(y_train.astype(int), minlength=n_classes)
            class_weights = 1.0 / (class_counts + 1)
            class_weights = class_weights / class_weights.sum() * n_classes
            alpha_tensor = torch.as_tensor(class_weights, dtype=torch.float)
            logger.info(
                f"[{model_type}] FocalLoss + class_weights: gamma={_focal_gamma}, "
                f"alpha={class_weights.round(3).tolist()}"
            )
        else:
            alpha_tensor = torch.tensor([_focal_alpha], dtype=torch.float)
            logger.info(
                f"[{model_type}] FocalLoss: alpha={_focal_alpha}, gamma={_focal_gamma}"
            )
        criterion = FocalLoss(alpha=alpha_tensor, gamma=_focal_gamma, reduction="mean")
    elif _use_class_weights:
        class_counts = np.bincount(y_train.astype(int), minlength=n_classes)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * n_classes
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, **_ce_kwargs)
        logger.info(
            f"[{model_type}] CrossEntropyLoss + class_weights: {class_weights.round(3).tolist()}"
            + (f" + label_smoothing={_label_smoothing}" if _label_smoothing > 0 else "")
        )
    else:
        criterion = nn.CrossEntropyLoss(**_ce_kwargs)
        _ls_str = f" + label_smoothing={_label_smoothing}" if _label_smoothing > 0 else ""
        logger.info(f"[{model_type}] CrossEntropyLoss (unweighted){_ls_str}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── AMP (Automatic Mixed Precision) ──
    # Use float16 for forward and backward passes, float32 for weight updates,
    # saving ~40-50% GPU memory. Most beneficial for large models (input_dim>500 or hidden_dim>=512).
    _scaler = torch.amp.GradScaler("cuda", enabled=_use_amp) if device.type == "cuda" else None
    if _use_amp:
        logger.info(f"[{model_type}] AMP mixed precision enabled (float16 forward+backward, float32 weights)")
    else:
        logger.info(f"[{model_type}] AMP mixed precision disabled")

    # ── Learning rate scheduler ──
    if _use_cosine_warmup:
        # Linear warmup + cosine annealing
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        _warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                           total_iters=_warmup_epochs)
        _cosine = CosineAnnealingLR(optimizer, T_max=epochs - _warmup_epochs)
        scheduler = SequentialLR(optimizer,
                                 schedulers=[_warmup, _cosine],
                                 milestones=[_warmup_epochs])
        logger.info(
            f"[{model_type}] CosineWarmup: warmup={_warmup_epochs} epochs, "
            f"cosine T_max={epochs - _warmup_epochs}"
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.7, patience=8
        )
        logger.info(f"[{model_type}] ReduceLROnPlateau: mode=max, factor=0.7, patience=8")

    # ── EMA (Exponential Moving Average) ──
    ema_model = None
    if _use_ema:
        # Create EMA shadow model (deep copy of initial parameters)
        import copy as _copy
        ema_model = _copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        logger.info(f"[{model_type}] EMA: decay={_ema_decay}")

    history = {
        "train_loss": [], "val_loss": [], "val_acc": [],
        "val_balanced_acc": [], "val_macro_auc": [], "val_weighted_auc": [],
        "val_macro_f1": [], "val_weighted_f1": [], "val_per_class_acc": [],
    }
    best_val_acc = 0.0
    best_balanced_acc = 0.0
    best_macro_auc = 0.0
    best_weighted_auc = 0.0
    best_macro_f1 = 0.0
    best_weighted_f1 = 0.0
    best_monitor = 0.0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for proba_chunk, label_chunk in train_loader:
            proba_chunk = proba_chunk.to(device)   # [B, chunk_size, D]
            label_chunk = label_chunk.to(device)    # [B, chunk_size]

            # ── Data augmentation: time axis reversal (50% probability) ──
            if _use_time_reversal and torch.rand(1).item() < 0.5:
                proba_chunk = proba_chunk.flip(1)   # reverse time axis
                label_chunk = label_chunk.flip(1)

            # ── Data augmentation: Mixup ──
            if _use_mixup:
                _alpha = _mixup_alpha
                lam = float(np.random.beta(_alpha, _alpha)) if _alpha > 0 else 1.0
                if lam < 1.0 and proba_chunk.size(0) > 1:
                    _idx = torch.randperm(proba_chunk.size(0), device=device)
                    proba_chunk = lam * proba_chunk + (1 - lam) * proba_chunk[_idx]
                    # Use one-hot mix for labels
                    _y_oh = torch.nn.functional.one_hot(
                        label_chunk.reshape(-1), n_classes
                    ).float().reshape_as(proba_chunk[:, :, :n_classes])
                    _y_oh_mix = lam * _y_oh + (1 - lam) * _y_oh[_idx]
                    # Compute loss using mixed soft labels (requires special handling)
                    _labels_mixed = _y_oh_mix  # [B, chunk_size, n_classes]

            optimizer.zero_grad()

            # ── AMP autocast: float16 forward pass ──
            with torch.amp.autocast("cuda", enabled=_use_amp):
                logits = model(proba_chunk)
                # Output residual connection: output = input_proba (n_classes part only) + model(input)
                if _output_residual:
                    logits = proba_chunk[:, :, :n_classes] + logits

                if _use_mixup and lam < 1.0 and proba_chunk.size(0) > 1:
                    # Mixup loss: soft cross-entropy on logits per class
                    _log_probs = torch.log_softmax(logits.reshape(-1, n_classes), dim=-1)
                    loss = -(_labels_mixed.reshape(-1, n_classes) * _log_probs).sum(dim=-1).mean()
                else:
                    loss = criterion(logits.reshape(-1, n_classes), label_chunk.reshape(-1))

            # ── Backward pass (AMP: scale -> backward -> unscale -> clip -> step) ──
            if _use_amp and _scaler is not None:
                _scaler.scale(loss).backward()
                _scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                _scaler.step(optimizer)
                _scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # ── EMA update ──
            if ema_model is not None:
                with torch.no_grad():
                    for _p_ema, _p_model in zip(ema_model.parameters(), model.parameters()):
                        _p_ema.data.mul_(_ema_decay).add_(_p_model.data, alpha=1.0 - _ema_decay)

            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for proba_chunk, label_chunk in val_loader:
                proba_chunk = proba_chunk.to(device)
                label_chunk = label_chunk.to(device)
                with torch.amp.autocast("cuda", enabled=_use_amp):
                    logits = model(proba_chunk)
                    if _output_residual:
                        logits = proba_chunk[:, :, :n_classes] + logits
                val_loss += criterion(logits.reshape(-1, n_classes), label_chunk.reshape(-1)).item()
                pred = logits.argmax(dim=-1)
                correct += (pred == label_chunk).sum().item()
                total   += label_chunk.numel()
                all_preds.append(pred.reshape(-1).cpu().numpy())
                all_labels.append(label_chunk.reshape(-1).cpu().numpy())
                all_probs.append(
                    torch.softmax(logits, dim=-1).reshape(-1, n_classes).cpu().numpy()
                )
        val_loss /= len(val_loader)
        val_acc = correct / total

        y_pred_ep = np.concatenate(all_preds)
        y_true_ep = np.concatenate(all_labels)
        y_prob_ep = np.concatenate(all_probs, axis=0)
        classes_all = list(range(n_classes))
        val_balanced_acc = float(balanced_accuracy_score(y_true_ep, y_pred_ep))
        val_macro_f1 = float(f1_score(
            y_true_ep, y_pred_ep, labels=classes_all,
            average="macro", zero_division=0,
        ))
        val_weighted_f1 = float(f1_score(
            y_true_ep, y_pred_ep, labels=classes_all,
            average="weighted", zero_division=0,
        ))
        # Per-class val_acc (recall)
        per_class_acc = {}
        for c in classes_all:
            m = y_true_ep == c
            per_class_acc[c] = float((y_pred_ep[m] == c).mean()) if m.any() else float("nan")
        # OvR AUC (requires at least 2 classes present in val set)
        present = sorted(set(int(v) for v in y_true_ep))
        val_macro_auc = float("nan")
        val_weighted_auc = float("nan")
        if len(present) >= 2:
            prob_sub = y_prob_ep[:, present]
            row_sum = prob_sub.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum <= 0, 1.0, row_sum)
            prob_sub_n = prob_sub / row_sum
            try:
                val_macro_auc = float(roc_auc_score(
                    y_true_ep, prob_sub_n, multi_class="ovr",
                    average="macro", labels=present,
                ))
            except ValueError:
                pass
            try:
                val_weighted_auc = float(roc_auc_score(
                    y_true_ep, prob_sub_n, multi_class="ovr",
                    average="weighted", labels=present,
                ))
            except ValueError:
                pass

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_balanced_acc"].append(val_balanced_acc)
        history["val_macro_auc"].append(val_macro_auc)
        history["val_weighted_auc"].append(val_weighted_auc)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_weighted_f1"].append(val_weighted_f1)
        history["val_per_class_acc"].append(per_class_acc)
        _auc_str = f"{val_macro_auc:.4f}" if not np.isnan(val_macro_auc) else "N/A"
        _wauc_str = f"{val_weighted_auc:.4f}" if not np.isnan(val_weighted_auc) else "N/A"
        _mark_acc = " *" if val_acc > best_val_acc else ""
        _mark_bal = " *" if val_balanced_acc > best_balanced_acc else ""
        _mark_mauc = " *" if (not np.isnan(val_macro_auc) and val_macro_auc > best_macro_auc) else ""
        _mark_wauc = " *" if (not np.isnan(val_weighted_auc) and val_weighted_auc > best_weighted_auc) else ""
        _mark_mf1 = " *" if val_macro_f1 > best_macro_f1 else ""
        _mark_wf1 = " *" if val_weighted_f1 > best_weighted_f1 else ""
        logger.info(
            f"[{model_type}] Epoch {epoch+1}/{epochs}\n"
            f"    train_loss      = {train_loss:.4f}\n"
            f"    val_loss        = {val_loss:.4f}\n"
            f"    val_acc         = {val_acc:.4f}{_mark_acc}\n"
            f"    val_balanced_acc= {val_balanced_acc:.4f}{_mark_bal}\n"
            f"    macro_auc       = {_auc_str}{_mark_mauc}\n"
            f"    weighted_auc    = {_wauc_str}{_mark_wauc}\n"
            f"    macro_f1        = {val_macro_f1:.4f}{_mark_mf1}\n"
            f"    weighted_f1     = {val_weighted_f1:.4f}{_mark_wf1}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if val_balanced_acc > best_balanced_acc:
            best_balanced_acc = val_balanced_acc
        if not np.isnan(val_macro_auc) and val_macro_auc > best_macro_auc:
            best_macro_auc = val_macro_auc
        if not np.isnan(val_weighted_auc) and val_weighted_auc > best_weighted_auc:
            best_weighted_auc = val_weighted_auc
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
        if val_weighted_f1 > best_weighted_f1:
            best_weighted_f1 = val_weighted_f1

        _metric_map = {
            "val_acc": val_acc,
            "val_balanced_acc": val_balanced_acc,
            "macro_f1": val_macro_f1,
            "weighted_f1": val_weighted_f1,
            "macro_auc": val_macro_auc,
            "weighted_auc": val_weighted_auc,
        }
        _current_monitor = _metric_map.get(monitor_metric, val_acc)

        old_lr = optimizer.param_groups[0]["lr"]
        if _use_cosine_warmup:
            scheduler.step()  # epoch-based scheduler
        else:
            scheduler.step(_current_monitor)  # metric-based scheduler
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            _dir = "up" if new_lr > old_lr else "down"
            logger.info(f"[{model_type}] Learning rate change: {old_lr:.6f} -> {new_lr:.6f} {_dir}")

        if _current_monitor > best_monitor:
            best_monitor = _current_monitor
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[{model_type}] Early stopping at epoch {epoch+1} (best {monitor_metric}={best_monitor:.4f})")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(f"[{model_type}] Restored best model weights ({monitor_metric}={best_monitor:.4f})")

    # ── Use EMA weights (if EMA was enabled) ──
    if ema_model is not None:
        model.load_state_dict({k: v.clone() for k, v in ema_model.state_dict().items()})
        logger.info(f"[{model_type}] Switched to EMA weights (decay={_ema_decay})")
        del ema_model

    if save_dir:
        from pathlib import Path
        out = Path(save_dir) / "weights"
        out.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "n_classes": n_classes,
            "input_dim": _input_dim,
            "cfg": {k: v for k, v in cfg.items() if not callable(v)},
            "best_metrics": {
                "val_acc": best_val_acc,
                "val_balanced_acc": best_balanced_acc,
                "macro_auc": best_macro_auc,
                "weighted_auc": best_weighted_auc,
                "macro_f1": best_macro_f1,
                "weighted_f1": best_weighted_f1,
            },
            "history": {k: v for k, v in history.items() if k != "val_per_class_acc"},
        }
        save_path = out / f"{model_type}_best.pt"
        torch.save(ckpt, save_path)
        _bm = ckpt["best_metrics"]
        _mauc_str = f"{_bm['macro_auc']:.4f}" if not np.isnan(_bm['macro_auc']) else "N/A"
        _wauc_str = f"{_bm['weighted_auc']:.4f}" if not np.isnan(_bm['weighted_auc']) else "N/A"
        logger.info(
            f"[{model_type}] Best model saved: {save_path}\n"
            f"    Best {monitor_metric} = {best_monitor:.4f}\n"
            f"    val_acc          = {_bm['val_acc']:.4f}\n"
            f"    val_balanced_acc = {_bm['val_balanced_acc']:.4f}\n"
            f"    macro_auc        = {_mauc_str}\n"
            f"    weighted_auc     = {_wauc_str}\n"
            f"    macro_f1         = {_bm['macro_f1']:.4f}\n"
            f"    weighted_f1      = {_bm['weighted_f1']:.4f}"
        )

    # Summary of optimization toggles
    _flags = []
    if _use_layer_norm:     _flags.append("LayerNorm")
    if _use_deep_proj:      _flags.append("DeepProj")
    if _use_attn_pool:      _flags.append("AttnPool")
    if _label_smoothing > 0: _flags.append(f"LabelSmooth({_label_smoothing})")
    if _use_cosine_warmup:  _flags.append("CosineWarmup")
    if _use_ema:            _flags.append(f"EMA({_ema_decay})")
    if _use_balanced_samp:  _flags.append("BalancedSamp")
    if _use_mixup:          _flags.append(f"Mixup({_mixup_alpha})")
    if _use_time_reversal:  _flags.append("TimeRev")
    if _use_focal:          _flags.append("focal_loss")
    if _use_class_weights:  _flags.append("class_weights")
    if _output_residual:    _flags.append("output_residual")
    if _use_extra:          _flags.append(f"extra_features(dim={_extra_dim})")
    if _use_amp:            _flags.append("AMP")
    if _flags:
        logger.info(f"[{model_type}] Enabled optimizations: {', '.join(_flags)}")

    return model, history


# ---------------------------------------------------------------------------
# Hyperparameter Grid Search
# ---------------------------------------------------------------------------

def _generate_param_combinations(param_grid: dict, method: str, max_trials: int,
                                  model_type: str, rng: np.random.Generator) -> list:
    """Generate hyperparameter combinations. method="grid" returns full permutations, "random" returns random samples."""
    import itertools

    # Filter parameters relevant to the current model type
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    if method == "grid":
        combos = list(itertools.product(*values))
        combos = [dict(zip(keys, combo)) for combo in combos]
        logger.info(f"[GridSearch] grid method generated {len(combos)} combinations")
        return combos
    elif method == "random":
        # Total number of full permutations
        total = 1
        for v in values:
            total *= len(v)
        n = min(max_trials, total)
        # Random sampling
        indices = rng.choice(total, n, replace=False)
        combos = []
        for idx in indices:
            combo = {}
            remainder = int(idx)
            for k, v_list in zip(keys, values):
                divisor = 1
                for vv in values[values.index(v_list) + 1:]:
                    divisor *= len(vv)
                combo_idx = (remainder // divisor) % len(v_list)
                combo[k] = v_list[combo_idx]
            combos.append(combo)
        logger.info(f"[GridSearch] random method sampled {n} combinations from {total} total")
        return combos
    else:
        raise ValueError(f"Unknown search method: {method}, supported: grid / random")


def grid_search_sequence_model(
    model_type: str,
    proba_train: np.ndarray,
    y_train: np.ndarray,
    proba_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    cfg: dict,
    extra_train: "np.ndarray | None" = None,
    extra_val: "np.ndarray | None" = None,
    output_dir: str = "",
) -> dict:
    """
    Hyperparameter grid search: iterate over parameter combinations in param_grid,
    train a lightweight model for each, return best parameters and all trial results.

    cfg should contain a grid_search subsection. Fields outside grid_search are used
    as default configuration for each trial (overwritten by searched keys).

    Returns dict:
        best_params:  Best hyperparameter combination
        best_score:   Best score
        best_trial:   Best trial number
        monitor_metric: Metric used for evaluation
        trials:       List of all trial results (each with params / metrics / trial_id)
        report_path:  Report save path
    """
    gs_cfg = cfg.get("grid_search", {}) or {}
    if not gs_cfg.get("enabled", False):
        logger.info("[GridSearch] Not enabled, skipping")
        return {"best_params": {}, "best_score": 0.0, "trials": [], "skipped": True}

    param_grid = gs_cfg.get("param_grid", {})
    if not param_grid:
        logger.warning("[GridSearch] param_grid is empty, skipping")
        return {"best_params": {}, "best_score": 0.0, "trials": [], "skipped": True}

    method       = gs_cfg.get("method", "random")
    max_trials   = int(gs_cfg.get("max_trials", 30))
    gs_epochs    = int(gs_cfg.get("epochs", 60))
    gs_patience  = int(gs_cfg.get("early_stopping_patience", 10))
    monitor_key  = gs_cfg.get("monitor_metric", "val_balanced_acc")
    seed         = int(cfg.get("seed", 42))

    rng = np.random.default_rng(seed)

    # Build trial config template: copy original cfg and override epochs / patience
    trial_cfg_base = {k: v for k, v in cfg.items() if k != "grid_search"}
    trial_cfg_base["epochs"] = gs_epochs
    trial_cfg_base["early_stopping_patience"] = gs_patience
    # Disable data augmentation and EMA during search for fair comparison
    trial_cfg_base["use_mixup"] = False
    trial_cfg_base["use_time_reversal"] = False
    trial_cfg_base["use_ema"] = False
    trial_cfg_base["use_balanced_sampling"] = False

    # Generate all parameter combinations
    combos = _generate_param_combinations(param_grid, method, max_trials, model_type, rng)

    n_trials = len(combos)
    logger.info(
        f"[GridSearch] Starting hyperparameter search: model={model_type}, method={method}, "
        f"trials={n_trials}, epochs={gs_epochs}, patience={gs_patience}, "
        f"monitor={monitor_key}"
    )

    best_score = -1.0
    best_params = {}
    best_trial = -1
    all_trials = []

    import time as _time
    _t_start = _time.time()

    for trial_idx, combo in enumerate(combos):
        _t_trial_start = _time.time()

        # Build current trial config
        trial_cfg = dict(trial_cfg_base)
        for k, v in combo.items():
            trial_cfg[k] = v

        # Transformer: ensure d_model is divisible by nhead
        if model_type == "transformer":
            d_model_val = int(trial_cfg.get("d_model", trial_cfg_base.get("d_model", 128)))
            nhead_val   = int(trial_cfg.get("nhead", trial_cfg_base.get("nhead", 4)))
            while d_model_val % nhead_val != 0 and nhead_val > 1:
                nhead_val -= 1
            trial_cfg["nhead"] = nhead_val

        # Each trial uses its own seed sequence
        trial_cfg["seed"] = seed + trial_idx

        logger.info(
            f"[GridSearch] Trial {trial_idx + 1}/{n_trials}: {combo}"
        )

        try:
            # Note: train_sequence_model sets global seed internally; no extra handling here
            import torch as _torch
            _torch.manual_seed(trial_cfg["seed"])
            np.random.seed(trial_cfg["seed"])
            import random as _random
            _random.seed(trial_cfg["seed"])

            model, history = train_sequence_model(
                model_type=model_type,
                proba_train=proba_train,
                y_train=y_train,
                proba_val=proba_val,
                y_val=y_val,
                n_classes=n_classes,
                cfg=trial_cfg,
                extra_train=extra_train,
                extra_val=extra_val,
                save_dir="",  # do not save intermediate models
            )

            # Extract best value of the monitor metric
            _metric_map = {
                "val_acc":            history.get("val_acc", []),
                "val_balanced_acc":   history.get("val_balanced_acc", []),
                "macro_f1":           history.get("val_macro_f1", []),
                "weighted_f1":        history.get("val_weighted_f1", []),
                "macro_auc":          history.get("val_macro_auc", []),
                "weighted_auc":       history.get("val_weighted_auc", []),
            }
            metric_values = _metric_map.get(monitor_key, history.get("val_acc", []))
            # Filter NaN
            valid_vals = [v for v in metric_values if not (isinstance(v, float) and np.isnan(v))]
            trial_score = float(max(valid_vals)) if valid_vals else 0.0

        except Exception as e:
            logger.warning(f"[GridSearch] Trial {trial_idx + 1} failed: {e}")
            trial_score = 0.0
            history = {"error": str(e)}

        _t_trial_elapsed = _time.time() - _t_trial_start

        _val_accs = history.get("val_acc", [])
        _final_acc = float(_val_accs[-1]) if _val_accs else 0.0
        _final_bal = float(history.get("val_balanced_acc", [0])[-1]) if history.get("val_balanced_acc") else 0.0
        _final_f1  = float(history.get("val_macro_f1", [0])[-1]) if history.get("val_macro_f1") else 0.0

        trial_result = {
            "trial_id":  trial_idx + 1,
            "params":    combo,
            "score":     round(trial_score, 4),
            "monitor_metric": monitor_key,
            "final_val_acc": round(_final_acc, 4),
            "final_val_balanced_acc": round(_final_bal, 4),
            "final_val_macro_f1": round(_final_f1, 4),
            "elapsed_sec": round(_t_trial_elapsed, 1),
        }
        all_trials.append(trial_result)

        is_best = trial_score > best_score
        marker = " *" if is_best else ""
        logger.info(
            f"[GridSearch] Trial {trial_idx + 1}/{n_trials} complete: "
            f"score({monitor_key})={trial_score:.4f}{marker}  "
            f"val_acc={_final_acc:.4f}  val_balanced_acc={_final_bal:.4f}  "
            f"elapsed {_t_trial_elapsed:.0f}s"
        )

        if is_best:
            best_score = trial_score
            best_params = dict(combo)
            best_trial = trial_idx + 1

        # Memory cleanup
        del model, history
        import gc as _gc
        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    _t_total = _time.time() - _t_start
    logger.info(
        f"[GridSearch] Search complete: {n_trials} trials, total elapsed {_t_total:.0f}s, "
        f"best trial={best_trial}, {monitor_key}={best_score:.4f}, "
        f"best params: {best_params}"
    )

    # Save report
    report = {
        "model_type": model_type,
        "method": method,
        "n_trials": n_trials,
        "epochs_per_trial": gs_epochs,
        "monitor_metric": monitor_key,
        "best_trial": best_trial,
        "best_score": round(best_score, 4),
        "best_params": best_params,
        "total_elapsed_sec": round(_t_total, 1),
        "trials": all_trials,
    }

    report_path = ""
    if output_dir:
        import json
        from pathlib import Path as _Path
        out = _Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = str(out / "grid_search_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"[GridSearch] Report saved: {report_path}")

    return {
        "best_params": best_params,
        "best_score": round(best_score, 4),
        "best_trial": best_trial,
        "monitor_metric": monitor_key,
        "trials": all_trials,
        "report_path": report_path,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Unified Inference Entry Point
# ---------------------------------------------------------------------------

def predict_sequence_model(
    model: nn.Module,
    proba: np.ndarray,
    cfg: dict,
    extra: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Unified inference entry point, sliding window prediction with overlap averaging.

    proba: [T, n_classes] or [T, n_classes + extra_dim] (if extra is pre-concatenated)
    extra: [T, extra_dim] extra features; ignored if None

    Returns [T, n_classes] refined probability distribution.
    """
    import sys
    import time as _time

    def _log(msg):
        """Write to both logger and stdout for immediate visibility."""
        logger.info(msg)
        print(msg, flush=True)

    _log(f"[predict] Entry: proba={proba.shape} {proba.dtype}, extra={'yes' if extra is not None else 'no'}")
    _t_entry = _time.time()

    chunk_size = int(cfg.get("chunk_size", 512))
    stride     = int(cfg.get("stride_val", 512))
    batch_size = int(cfg.get("batch_size", 32))
    device_str = cfg.get("device", "cuda")
    _output_residual = bool(cfg.get("seq_output_residual", False))
    n_classes_orig = proba.shape[1]

    # Extra feature concatenation
    if extra is not None:
        _log(f"[predict] Concatenating extra: proba={proba.shape}, extra={extra.shape} {extra.dtype}")
        _t0 = _time.time()
        proba = np.hstack([proba, extra.astype(np.float32)])
        _log(f"[predict] hstack complete: shape={proba.shape}, elapsed {_time.time() - _t0:.1f}s")
    else:
        _log(f"[predict] No extra features")

    _log(f"[predict] Checking CUDA device (device_str={device_str})...")
    _t0 = _time.time()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    _log(f"[predict] Device={device}, elapsed {_time.time() - _t0:.1f}s")

    _log(f"[predict] Moving model to {device}...")
    _t0 = _time.time()
    model = model.to(device)
    model.eval()
    _log(f"[predict] Model migration complete, elapsed {_time.time() - _t0:.1f}s")

    T, input_dim = proba.shape
    _log(f"[predict] Creating tensors: T={T}, input_dim={input_dim}")
    _t0 = _time.time()
    proba_tensor = torch.from_numpy(proba).float()
    proba_sum = np.zeros((T, n_classes_orig), dtype=np.float32)
    count     = np.zeros(T, dtype=np.int32)
    _log(f"[predict] Tensor creation complete, elapsed {_time.time() - _t0:.1f}s")

    starts = list(range(0, T, stride))
    n_total = len(starts)
    n_batches = (n_total + batch_size - 1) // batch_size
    log_every = max(1, n_batches // 5)
    _log(f"[predict] Starting inference loop: {n_total} chunks -> {n_batches} batches (chunk={chunk_size}, stride={stride}, batch={batch_size})")

    # Pre-build full batch tensor: [n_total, chunk_size, input_dim]; only the last chunk needs padding
    _t0 = _time.time()
    _log(f"[predict] Pre-building batch tensor ({n_total}x{chunk_size}x{input_dim})...")
    batch_tensor = torch.zeros(n_total, chunk_size, input_dim, dtype=torch.float32)
    for idx in range(n_total):
        s = starts[idx]
        e = min(s + chunk_size, T)
        batch_tensor[idx, :e - s] = proba_tensor[s:e]
    _log(f"[predict] Batch tensor done: shape={batch_tensor.shape}, elapsed {_time.time() - _t0:.1f}s")

    with torch.no_grad():
        batch_idx = 0
        for batch_start in range(0, n_total, batch_size):
            batch_end = min(batch_start + batch_size, n_total)
            if batch_idx > 0 and batch_idx % log_every == 0:
                elapsed = _time.time() - _t_entry
                _log(f"[predict] Inference progress: batch {batch_idx}/{n_batches}, chunk {batch_end}/{n_total} ({100*batch_end/n_total:.0f}%), elapsed {elapsed:.0f}s")

            _batch = batch_tensor[batch_start:batch_end].to(device)  # [B, chunk_size, D]
            logits = model(_batch)  # [B, chunk_size, n_classes]
            # Output residual connection
            if _output_residual:
                logits = _batch[:, :, :n_classes_orig] + logits
            refined = torch.softmax(logits, dim=-1).cpu().numpy()  # [B, chunk_size, C]

            for j, idx in enumerate(range(batch_start, batch_end)):
                s = starts[idx]
                e = min(s + chunk_size, T)
                actual = e - s
                proba_sum[s:e] += refined[j, :actual]
                count[s:e] += 1

            batch_idx += 1

    del batch_tensor

    result = proba_sum / count[:, np.newaxis]
    _log(f"[predict] Inference complete: {T} frames -> {result.shape}, total elapsed {_time.time() - _t_entry:.0f}s")
    return result
