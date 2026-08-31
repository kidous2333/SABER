"""
temporal_validator.py
Temporal second-stage validation module.

Takes the per-frame probability distribution [T, n_classes] output from the
first-layer LightGBM (single-frame factor model), constructs sliding-window
temporal features, and trains a second-layer LightGBM multiclass classifier
to leverage behavioral continuity priors for improved accuracy.

Feature construction (per frame t, window size w):
  - Current frame probability distribution            (n_classes dim)
  - Window mean probability                           (n_classes dim)
  - Window std of probability                         (n_classes dim)
  - Window max probability                            (n_classes dim)
  - Current probability - window mean (trend)         (n_classes dim)
  - Window argmax class one-hot mode                  (n_classes dim)
  - Number of top-1 class changes in window (scalar)  (1 dim)
  - Current frame top-1 class ID (scalar)             (1 dim)
  - Current frame probability entropy (scalar)        (1 dim)
  Total: n_classes * 6 + 3 dims
"""

import csv
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np


def _write_csv(path: Path, rows: list, header: list) -> None:
    """Write CSV alongside a plot PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)
import lightgbm as lgb

from src.visualization import (
    plot_temporal_feature_breakdown,
    plot_proba_smoothing_effect,
    plot_transition_matrix,
    plot_viterbi_comparison,
)

# Decoders imported from independent package (previously inline definitions extracted to src/decoders/)
from src.decoders.base import SequenceDecoder
from src.decoders.merging import SegmentMerger
from src.decoders.viterbi import ViterbiDecoder, DurationViterbiDecoder
from src.decoders.crf import CRFDecoder
from src.decoders.pipeline import DecoderPipeline

logger = logging.getLogger(__name__)


def _build_temporal_features(proba: np.ndarray, window_size: int = 31) -> np.ndarray:
    """
    proba: [T, n_classes]  first-layer per-frame probabilities
    window_size: odd number, centered on current frame (default 31 frames, ~1 second)
    Returns: [T, feature_dim]
    """
    import gc
    T, n_classes = proba.shape
    half = window_size // 2

    # Boundary padding (edge frames replicate nearest frame)
    padded = np.concatenate([
        np.repeat(proba[:1], half, axis=0),
        proba,
        np.repeat(proba[-1:], half, axis=0),
    ], axis=0)  # [T + 2*half, n_classes]

    # cumsum sliding window mean
    cs = np.cumsum(padded, axis=0)
    cs_pad = np.empty((cs.shape[0] + 1, n_classes), dtype=cs.dtype)
    cs_pad[0] = 0.0
    cs_pad[1:] = cs
    win_sum  = cs_pad[window_size:window_size + T] - cs_pad[:T]
    win_mean = (win_sum / window_size).astype(np.float32)
    del cs, cs_pad, win_sum; gc.collect()

    # Std: E[x^2] - E[x]^2
    padded_sq = padded ** 2
    cs2 = np.cumsum(padded_sq, axis=0)
    del padded_sq
    cs2_pad = np.empty((cs2.shape[0] + 1, n_classes), dtype=cs2.dtype)
    cs2_pad[0] = 0.0
    cs2_pad[1:] = cs2
    win_sum2 = cs2_pad[window_size:window_size + T] - cs2_pad[:T]
    del cs2, cs2_pad
    win_var = win_sum2 / window_size - win_mean.astype(np.float64) ** 2
    del win_sum2
    win_std = np.sqrt(np.maximum(win_var, 0.0)).astype(np.float32)
    del win_var; gc.collect()

    # Window max (stride_tricks view, no data copy)
    from numpy.lib.stride_tricks import sliding_window_view
    win_view = sliding_window_view(padded, window_shape=window_size, axis=0)
    # win_view: [T, n_classes, window_size]
    win_max = win_view.max(axis=2).astype(np.float32)  # [T, n_classes]

    # Trend
    trend = (proba - win_mean).astype(np.float32)

    # Window argmax mode (vectorized, avoids per-frame for loop)
    win_argmax = win_view.argmax(axis=1).astype(np.int32)  # [T, window_size]
    del win_view; gc.collect()

    # Vectorized mode: count occurrences per class per frame, take argmax
    # Use one-hot accumulation instead of bincount loop
    argmax_onehot = np.zeros((T, n_classes), dtype=np.float32)
    for c in range(n_classes):
        argmax_onehot[:, c] = (win_argmax == c).sum(axis=1)
    mode_onehot = np.zeros((T, n_classes), dtype=np.float32)
    mode_onehot[np.arange(T), argmax_onehot.argmax(axis=1)] = 1.0
    del argmax_onehot

    # Number of top-1 class changes in window (normalized)
    changes = (win_argmax[:, 1:] != win_argmax[:, :-1]).sum(axis=1, keepdims=True).astype(np.float32)
    changes /= (window_size - 1)
    del win_argmax

    # Current frame top-1 class ID (normalized)
    top1_cur = proba.argmax(axis=1, keepdims=True).astype(np.float32) / n_classes

    # Current frame probability entropy (normalized)
    eps = 1e-9
    entropy = -(proba * np.log(proba + eps)).sum(axis=1, keepdims=True).astype(np.float32)
    entropy /= np.log(n_classes)

    features = np.concatenate([
        proba.astype(np.float32),
        win_mean,
        win_std,
        win_max,
        trend,
        mode_onehot,
        changes,
        top1_cur,
        entropy,
    ], axis=1)

    del win_mean, win_std, win_max, trend, mode_onehot, changes, top1_cur, entropy
    gc.collect()
    return features


def _build_temporal_features_multiscale(proba: np.ndarray, window_sizes: list) -> np.ndarray:
    """
    Multi-scale temporal feature construction: calls _build_temporal_features for each
    window_size, concatenates all window-scale features into one large feature matrix.

    proba: [T, n_classes]
    window_sizes: e.g. [7, 15, 31, 63]
    Returns: [T, total_feature_dim]
    """
    import gc
    feats = []
    for ws in window_sizes:
        f = _build_temporal_features(proba, window_size=ws)
        feats.append(f)
        gc.collect()
    result = np.concatenate(feats, axis=1)
    n_classes = proba.shape[1]
    dim_per_window = n_classes * 6 + 3
    del feats; gc.collect()
    logger.info(
        f"Multi-scale temporal features: window_sizes={window_sizes}, "
        f"dim per window={dim_per_window}, total dim={result.shape[1]}"
    )
    return result


def _build_enhanced_temporal_features(proba: np.ndarray) -> np.ndarray:
    """
    Enhanced temporal features: differences, EMA, cross-category features.

    Appended on top of base features:
      - First-order probability difference (adjacent frame change)   (n_classes dim)
      - Second-order probability difference (acceleration)           (n_classes dim)
      - EMA-smoothed probability (half-life 3/10/30 frames)         (n_classes * 3 dim)
      - top1-top2 probability margin                                 (1 dim)
      - Probability Gini coefficient (concentration)                 (1 dim)
      Total: n_classes * 5 + 2 dims
    """
    import gc
    T, n_classes = proba.shape
    proba = proba.astype(np.float32)

    parts = []

    # First-order difference (first frame filled with 0)
    diff1 = np.zeros_like(proba)
    diff1[1:] = proba[1:] - proba[:-1]
    parts.append(diff1)

    # Second-order difference (first two frames filled with 0)
    diff2 = np.zeros_like(proba)
    diff2[2:] = diff1[2:] - diff1[1:-1]
    parts.append(diff2)

    # EMA smoothing (half-life 3/10/30 frames)
    for halflife in [3, 10, 30]:
        alpha = np.exp(np.log(0.5) / halflife)
        ema = np.zeros_like(proba)
        ema[0] = proba[0]
        for t in range(1, T):
            ema[t] = alpha * ema[t - 1] + (1 - alpha) * proba[t]
        parts.append(ema.astype(np.float32))

    # top1-top2 probability margin
    top2 = -np.partition(-proba, 1, axis=1)[:, :2]
    margin = (top2[:, 0] - top2[:, 1]).reshape(-1, 1).astype(np.float32)
    parts.append(margin)

    # Probability Gini coefficient (concentration, 0=uniform, 1=fully concentrated)
    proba_sorted = np.sort(proba, axis=1)
    n = n_classes
    index = np.arange(1, n + 1, dtype=np.float32)
    gini = (2 * (proba_sorted * index).sum(axis=1) - (n + 1) * proba_sorted.sum(axis=1)) / n
    gini = np.clip(gini / (n - 1) * n, 0.0, 1.0).reshape(-1, 1).astype(np.float32)
    parts.append(gini)

    result = np.concatenate(parts, axis=1)
    del parts, proba, diff1, diff2, ema, margin, gini; gc.collect()
    logger.info(
        f"Enhanced temporal features: dim={result.shape[1]}"
        f" (diff1={n_classes} + diff2={n_classes}"
        f" + ema*3={n_classes * 3} + margin=1 + gini=1)"
    )
    return result



def _build_autocorr_features(proba: np.ndarray, max_lag: int = 3) -> np.ndarray:
    """
    Short-term autocorrelation features for probability sequences.

    Computes lag=1..max_lag autocorrelation coefficients for each class
    (computed within a sliding window).
    Returns: [T, n_classes * max_lag]
    """
    import gc
    T, n_classes = proba.shape
    proba = proba.astype(np.float32)
    window = 31  # Autocorrelation computation window
    half = window // 2

    parts = []
    for lag in range(1, max_lag + 1):
        # Compute lag-k autocorrelation for each class
        # r_k[t] = E[(x_t - mean_t)(x_{t-k} - mean_t)] / (std_t * std_{t-k})
        autocorr = np.zeros((T, n_classes), dtype=np.float32)
        for c in range(n_classes):
            series = proba[:, c]
            # Compute autocorrelation within sliding window
            for t in range(T):
                lo = max(0, t - half)
                hi = min(T, t + half + 1)
                if hi - lo <= lag:
                    autocorr[t, c] = 0.0
                    continue
                win = series[lo:hi]
                mu = win.mean()
                sig = win.std() + 1e-9
                # lag-k autocorrelation
                if len(win) > lag:
                    ac = ((win[lag:] - mu) * (win[:-lag] - mu)).mean() / (sig * sig)
                    autocorr[t, c] = float(np.clip(ac, -1.0, 1.0))
        parts.append(autocorr)

    result = np.concatenate(parts, axis=1)
    del parts, autocorr; gc.collect()
    logger.info(f"Autocorrelation features: lag={list(range(1, max_lag + 1))}, dim={result.shape[1]}")
    return result


def _build_distribution_shape_features(proba: np.ndarray) -> np.ndarray:
    """
    Probability distribution shape features: skewness and kurtosis.

    Computes skewness and kurtosis of the probability distribution within a
    sliding window for each frame, providing finer-grained concentration
    information beyond entropy.
    Returns: [T, n_classes * 2] (skewness + kurtosis per class)
    """
    import gc
    T, n_classes = proba.shape
    proba = proba.astype(np.float32)
    window = 31
    half = window // 2

    skew = np.zeros((T, n_classes), dtype=np.float32)
    kurt = np.zeros((T, n_classes), dtype=np.float32)

    for c in range(n_classes):
        series = proba[:, c]
        for t in range(T):
            lo = max(0, t - half)
            hi = min(T, t + half + 1)
            win = series[lo:hi]
            mu = win.mean()
            sig = win.std() + 1e-9
            z = (win - mu) / sig
            skew[t, c] = float((z ** 3).mean())
            kurt[t, c] = float((z ** 4).mean() - 3.0)  # excess kurtosis

    # Clip extremes
    skew = np.clip(skew, -5.0, 5.0)
    kurt = np.clip(kurt, -5.0, 15.0)

    result = np.concatenate([skew, kurt], axis=1)
    del skew, kurt; gc.collect()
    logger.info(f"Distribution shape features: skew({n_classes}D) + kurt({n_classes}D), total dim={result.shape[1]}")
    return result


def _build_cross_scale_features(
    proba: np.ndarray,
    short_ws: int = 7,
    long_ws: int = 31,
) -> np.ndarray:
    """
    Multi-scale interaction features: differences between short and long window statistics.

    Returns: [T, n_classes * 3]
      - mean_short - mean_long (trend divergence)
      - std_short / std_long (volatility ratio)
      - max_short - max_long (peak difference)
    """
    import gc
    T, n_classes = proba.shape
    proba = proba.astype(np.float32)

    def _win_stats(p, ws):
        half = ws // 2
        padded = np.concatenate([
            np.repeat(p[:1], half, axis=0),
            p,
            np.repeat(p[-1:], half, axis=0),
        ], axis=0)
        from numpy.lib.stride_tricks import sliding_window_view
        wv = sliding_window_view(padded, window_shape=ws, axis=0)  # [T, C, ws]
        return wv.mean(axis=2).astype(np.float32), wv.std(axis=2).astype(np.float32), wv.max(axis=2).astype(np.float32)

    mean_s, std_s, max_s = _win_stats(proba, short_ws)
    mean_l, std_l, max_l = _win_stats(proba, long_ws)

    trend_div = mean_s - mean_l
    vol_ratio = std_s / (std_l + 1e-9)
    peak_diff = max_s - max_l

    # Clip ratio
    vol_ratio = np.clip(vol_ratio, 0.1, 10.0)
    peak_diff = np.clip(peak_diff, -1.0, 1.0)

    result = np.concatenate([trend_div, vol_ratio, peak_diff], axis=1)
    del mean_s, std_s, max_s, mean_l, std_l, max_l; gc.collect()
    logger.info(
        f"Cross-scale interaction features: short={short_ws}, long={long_ws}, "
        f"trend_div+vol_ratio+peak_diff, total dim={result.shape[1]}"
    )
    return result


def _merge_short_segments(seq: np.ndarray, min_segment: int) -> np.ndarray:
    """Merge continuous segments shorter than min_segment into the adjacent longest segment."""
    seq = seq.copy()
    T = len(seq)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < T:
            c = seq[i]
            j = i
            while j < T and seq[j] == c:
                j += 1
            seg_len = j - i
            if seg_len < min_segment:
                # Find lengths of left/right neighbor segments
                left_len = i  # Rough distance to 0 on left
                right_len = T - j
                if left_len == 0 and right_len == 0:
                    break
                if left_len == 0:
                    fill = seq[j] if j < T else seq[i - 1]
                elif right_len == 0:
                    fill = seq[i - 1]
                else:
                    # Take labels from adjacent segments (one frame each side)
                    left_label = seq[i - 1]
                    right_label = seq[j] if j < T else left_label
                    # Choose the longer side of adjacent segments
                    # Simple strategy: take left label (maintain temporal continuity)
                    fill = left_label
                seq[i:j] = fill
                changed = True
            i = j
    return seq


def _merge_short_segments_proba(
    seq: np.ndarray,
    proba: np.ndarray,
    min_segment: int,
) -> np.ndarray:
    """
    Probability-aware short segment merging.

    Unlike _merge_short_segments, when left and right neighbor classes differ,
    uses proba scores (rather than simply taking the left) to decide merge direction:

    - Left and right neighbors same class -> merge directly into that class
    - Left and right neighbors different -> compare average probability of the
      short segment on both classes, take the higher one
    - Only one side has a neighbor -> take that neighbor's class
    """
    seq = seq.copy()
    T = len(seq)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < T:
            c = seq[i]
            j = i
            while j < T and seq[j] == c:
                j += 1
            seg_len = j - i
            if seg_len < min_segment:
                left_label = seq[i - 1] if i > 0 else None
                right_label = seq[j] if j < T else None

                if left_label is None and right_label is None:
                    break

                if left_label is None:
                    fill = right_label
                elif right_label is None:
                    fill = left_label
                elif left_label == right_label:
                    # Left and right neighbors agree, high-confidence merge
                    fill = left_label
                else:
                    # Left and right neighbors differ -> decide with probabilities
                    seg_proba = proba[i:j]  # [seg_len, C]
                    left_score = seg_proba[:, int(left_label)].mean()
                    right_score = seg_proba[:, int(right_label)].mean()
                    fill = left_label if left_score >= right_score else right_label

                seq[i:j] = fill
                changed = True
            i = j
    return seq


def _estimate_per_class_durations(y: np.ndarray, n_classes: int) -> dict:
    """
    Estimate duration frame distribution for each class from label sequence.

    Returns dict:
        per_class_mean: [C] mean duration frames per class
        per_class_dur_logprob: [C, max_dur+1] log probability that class duration >= d
    """
    C = n_classes
    segments = {c: [] for c in range(C)}
    i = 0
    T = len(y)
    while i < T:
        c = int(y[i])
        j = i
        while j < T and int(y[j]) == c:
            j += 1
        dur = j - i
        segments[c].append(dur)
        i = j

    per_class_mean = np.ones(C, dtype=np.float64) * 30  # default
    max_dur = 120  # cap at 120 frames (4 sec @ 30fps)

    per_class_dur_logprob = np.zeros((C, max_dur + 1), dtype=np.float64)
    for c in range(C):
        if segments[c]:
            per_class_mean[c] = max(1.0, np.mean(segments[c]))
            # Build empirical CDF: P(duration >= d)
            durs = np.array(segments[c])
            for d in range(1, max_dur + 1):
                p = np.mean(durs >= d)
                per_class_dur_logprob[c, d] = np.log(max(p, 1e-6))
        else:
            per_class_dur_logprob[c, :] = 0.0  # no data, no penalty

    logger.info(
        f"Per-class duration estimates: {dict(zip(range(C), per_class_mean.round(1).tolist()))}"
    )
    return {
        "per_class_mean": per_class_mean,
        "per_class_dur_logprob": per_class_dur_logprob,
    }


def _fit_transitions_asymmetric(
    y_train: np.ndarray,
    n_classes: int,
    self_boost: float = 30.0,
    cross_smoothing: float = 1.0,
    per_class: bool = False,
) -> tuple:
    """
    Asymmetric smoothed transition matrix estimation.

    Encodes the structural prior that "behavioral segments last at least min_segment frames":
    - Self-transition pseudo-count = self_boost (much larger than 1, strongly prefers staying)
    - Cross-class transition pseudo-count = cross_smoothing (standard Laplace smoothing)
    - per_class=True: self_boost is independently computed per class based on mean duration

    Returns (log_trans [C,C], log_pi [C]).
    """
    C = n_classes

    if per_class:
        dur_info = _estimate_per_class_durations(y_train, n_classes)
        per_class_mean = dur_info["per_class_mean"]
        # P(stay|c) = 1 - 1/mean_duration -> self_boost approximately mean_duration - 1
        per_class_boost = np.maximum(per_class_mean - 1, 1.0)
        trans = np.full((C, C), cross_smoothing, dtype=np.float64)
        for c in range(C):
            trans[c, c] = per_class_boost[c]
        for t in range(len(y_train) - 1):
            trans[int(y_train[t]), int(y_train[t + 1])] += 1.0
        trans /= trans.sum(axis=1, keepdims=True)
    else:
        trans = np.full((C, C), cross_smoothing, dtype=np.float64)
        np.fill_diagonal(trans, self_boost)
        for t in range(len(y_train) - 1):
            trans[int(y_train[t]), int(y_train[t + 1])] += 1.0
        trans /= trans.sum(axis=1, keepdims=True)

    pi = np.full(C, cross_smoothing, dtype=np.float64)
    pi[int(y_train[0])] += 1.0
    pi /= pi.sum()

    log_trans = np.log(trans + 1e-300)
    log_pi = np.log(pi + 1e-300)
    logger.info(
        f"Asymmetric transition matrix: self_boost={self_boost}, cross_smoothing={cross_smoothing}, "
        f"per_class={per_class}, "
        f"self-transition probabilities={np.diag(trans).round(4).tolist()}"
    )
    return log_trans, log_pi


# ViterbiDecoder and DurationViterbiDecoder have been extracted to src/decoders/viterbi.py
# Import retained at top of file for backward compatibility


# CRFDecoder has been extracted to src/decoders/crf.py
# Import retained at top of file for backward compatibility


def _enrich_crf_features(proba: np.ndarray) -> np.ndarray:
    """
    Append first-order temporal features to CRF input, enhancing its perception
    of probability change trends.

    proba: [T, C] probability matrix
    Returns: [T, C*2 + 2] augmented feature matrix (raw proba + first-order diff + entropy + top1-top2 margin)
    """
    T, C = proba.shape
    proba = proba.astype(np.float32)

    # First-order difference (first frame filled with 0)
    delta = np.zeros_like(proba)
    delta[1:] = proba[1:] - proba[:-1]

    # Probability distribution entropy (normalized to [0, 1])
    eps = 1e-9
    entropy = -(proba * np.log(proba + eps)).sum(axis=1, keepdims=True)
    entropy /= max(np.log(C), eps)

    # top1-top2 probability margin
    top2 = -np.partition(-proba, 1, axis=1)[:, :2]
    margin = (top2[:, 0] - top2[:, 1]).reshape(-1, 1)

    result = np.hstack([proba, delta, entropy, margin]).astype(np.float32)
    logger.info(
        f"[CRF enrich] proba {proba.shape} -> enriched {result.shape}"
        f" (+delta({C}D) +entropy(1D) +margin(1D))"
    )
    return result


def _tune_crf_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bin_edges: np.ndarray,
    n_bins: int,
    algorithm: str = "lbfgs",
    max_iterations: int = 100,
    val_ratio: float = 0.2,
    chunk_size: int = 2000,
) -> tuple:
    """
    CRF hyperparameter grid search: c1 (L1) x c2 (L2).

    Splits training data into multiple sequences by chunk_size, reserves val_ratio
    as validation set, iterates over c1/c2 combinations, and returns the (c1, c2)
    yielding the highest validation accuracy.

    Returns (best_c1, best_c2, best_acc).
    """
    import sklearn_crfsuite
    import gc

    T = len(X_train)
    # Split into multiple short sequences for validation
    X_seqs = []
    y_seqs = []
    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)
        if end - i < 2:
            continue
        X_seqs.append(CRFDecoder._to_crf_seq(X_train[i:end], bin_edges))
        y_seqs.append([str(int(v)) for v in y_train[i:end]])

    n_total = len(X_seqs)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val

    if n_train < 1 or n_val < 1:
        logger.warning("[CRF tune] Insufficient data to split train/validation, skipping hyperparameter search")
        return 0.1, 0.1, 0.0

    rng = np.random.default_rng(42)
    idx = rng.permutation(n_total)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train_tune = [X_seqs[i] for i in train_idx]
    y_train_tune = [y_seqs[i] for i in train_idx]
    X_val_tune = [X_seqs[i] for i in val_idx]
    y_val_tune = [y_seqs[i] for i in val_idx]

    c1_values = [0.01, 0.05, 0.1, 0.5]
    c2_values = [0.01, 0.05, 0.1, 0.5]
    n_combos = len(c1_values) * len(c2_values)

    logger.info(
        f"[CRF tune] Hyperparameter search: c1={c1_values}, c2={c2_values} ({n_combos} combinations), "
        f"train seqs={n_train}, val seqs={n_val}"
    )

    best_c1, best_c2, best_acc = 0.1, 0.1, -1.0
    combo = 0
    for c1 in c1_values:
        for c2 in c2_values:
            combo += 1
            crf = sklearn_crfsuite.CRF(
                algorithm=algorithm,
                c1=c1, c2=c2,
                max_iterations=max_iterations,
                all_possible_transitions=True,
                all_possible_states=True,
            )
            crf.fit(X_train_tune, y_train_tune)

            # Evaluate on validation sequences
            correct = total = 0
            for x_seq, y_seq in zip(X_val_tune, y_val_tune):
                pred = crf.predict([x_seq])[0]
                correct += sum(p == t for p, t in zip(pred, y_seq))
                total += len(y_seq)
            acc = correct / total if total > 0 else 0.0

            marker = " *" if acc > best_acc else ""
            logger.info(f"  [{combo}/{n_combos}] c1={c1}, c2={c2} -> val_acc={acc:.4f}{marker}")
            if acc > best_acc:
                best_acc = acc
                best_c1 = c1
                best_c2 = c2
            gc.collect()

    logger.info(
        f"[CRF tune] Best hyperparameters: c1={best_c1}, c2={best_c2} (val_acc={best_acc:.4f})"
    )
    return best_c1, best_c2, best_acc


def _eval_sequence_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            classes_sorted: list, id_to_name: dict,
                            proba: np.ndarray = None) -> dict:
    """Compute post-decoding metrics (accuracy / balanced_accuracy / macro_f1 / per_class / AUC)."""
    from sklearn.metrics import roc_auc_score
    acc = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes_sorted,
                               average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=classes_sorted,
                                  average="weighted", zero_division=0))
    report = classification_report(
        y_true, y_pred, labels=classes_sorted,
        target_names=[id_to_name[c] for c in classes_sorted],
        output_dict=True, zero_division=0,
    )

    macro_auc, weighted_auc = 0.0, 0.0
    per_class_auc = {id_to_name[c]: None for c in classes_sorted}
    if proba is not None:
        present = sorted(set(int(v) for v in y_true))
        if len(present) >= 2:
            idx_present = [classes_sorted.index(c) for c in present]
            proba_sub = proba[:, idx_present]
            row_sum = proba_sub.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum <= 0, 1.0, row_sum)
            proba_sub_n = proba_sub / row_sum
            try:
                macro_auc = float(roc_auc_score(y_true, proba_sub_n, multi_class="ovr",
                                                average="macro", labels=present))
            except ValueError:
                pass
            try:
                weighted_auc = float(roc_auc_score(y_true, proba_sub_n, multi_class="ovr",
                                                   average="weighted", labels=present))
            except ValueError:
                pass
            for c in present:
                cname = id_to_name[c]
                y_bin = (y_true == c).astype(int)
                col = classes_sorted.index(c)
                try:
                    per_class_auc[cname] = float(roc_auc_score(y_bin, proba[:, col]))
                except ValueError:
                    per_class_auc[cname] = None

    per_class = {}
    for c in classes_sorted:
        cname = id_to_name[c]
        r = report.get(cname, {})
        per_class[cname] = {
            "precision": round(float(r.get("precision", 0.0)), 4),
            "recall":    round(float(r.get("recall",    0.0)), 4),
            "f1":        round(float(r.get("f1-score",  0.0)), 4),
            "support":   int(r.get("support", 0)),
            "auc":       round(per_class_auc[cname], 4) if per_class_auc[cname] is not None else None,
        }
    return {
        "accuracy":          round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1":          round(macro_f1, 4),
        "weighted_f1":       round(weighted_f1, 4),
        "macro_auc":         round(macro_auc, 4),
        "weighted_auc":      round(weighted_auc, 4),
        "per_class":         per_class,
    }


class TemporalValidator:
    """
    Temporal second-stage validator.

    Input: per-frame probability matrices [T, n_classes] from first-layer model on train/val
    Output: result dict consistent with SynthValidator.train_and_eval format
    """

    DEFAULT_LGBM_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, cfg: dict):
        synth_cfg = cfg.get("synth_validation", {}) or {}
        temporal_cfg = cfg.get("temporal_validation", {}) or {}
        self._temporal_cfg = temporal_cfg  # Keep raw config for _build_decoders
        base_params = {**self.DEFAULT_LGBM_PARAMS, **synth_cfg.get("lgbm_params", {})}
        self.lgbm_params = {**base_params, **temporal_cfg.get("lgbm_params", {})}
        self.class_weight = temporal_cfg.get("class_weight", synth_cfg.get("class_weight", "balanced"))
        self.early_stopping_rounds = int(temporal_cfg.get("early_stopping_rounds",
                                                           synth_cfg.get("early_stopping_rounds", 50)))
        self.window_size = int(temporal_cfg.get("window_size", 31))
        self.top_k = int(synth_cfg.get("top_k", 3))
        self.normalize_cm = synth_cfg.get("normalize_cm", True)

        # Temporal model selection: lgbm | bilstm | transformer | mamba
        # Legacy config compatibility: use_bilstm: true -> temporal_model: bilstm
        _use_bilstm_legacy = bool(temporal_cfg.get("use_bilstm", False))
        _default_model = "bilstm" if _use_bilstm_legacy else "lgbm"
        self.temporal_model = temporal_cfg.get("temporal_model", _default_model)

        # Neural network temporal model common parameters (seq_model section,
        # or legacy bilstm_xxx flat keys for compatibility)
        _bilstm_cfg = temporal_cfg.get("bilstm", {}) or {}
        _seq_defaults = {
            "hidden_dim":               int(temporal_cfg.get("bilstm_hidden_dim",               _bilstm_cfg.get("hidden_dim",               128))),
            "num_layers":               int(temporal_cfg.get("bilstm_num_layers",               _bilstm_cfg.get("num_layers",               2))),
            "dropout":                  float(temporal_cfg.get("bilstm_dropout",                _bilstm_cfg.get("dropout",                  0.3))),
            "chunk_size":               int(temporal_cfg.get("bilstm_chunk_size",               _bilstm_cfg.get("chunk_size",               512))),
            "stride_train":             int(temporal_cfg.get("bilstm_stride_train",             _bilstm_cfg.get("stride_train",             256))),
            "stride_val":               int(temporal_cfg.get("bilstm_stride_val",               _bilstm_cfg.get("stride_val",               512))),
            "batch_size":               int(temporal_cfg.get("bilstm_batch_size",               _bilstm_cfg.get("batch_size",               32))),
            "epochs":                   int(temporal_cfg.get("bilstm_epochs",                   _bilstm_cfg.get("epochs",                   20))),
            "lr":                       float(temporal_cfg.get("bilstm_lr",                     _bilstm_cfg.get("lr",                       1e-3))),
            "weight_decay":             float(temporal_cfg.get("bilstm_weight_decay",           _bilstm_cfg.get("weight_decay",             1e-4))),
            "device":                   temporal_cfg.get("bilstm_device",                       _bilstm_cfg.get("device",                   "cuda")),
            "early_stopping_patience":  int(temporal_cfg.get("bilstm_early_stopping_patience",  _bilstm_cfg.get("early_stopping_patience",  5))),
        }
        # seq_model section values override legacy flat keys
        _seq_cfg_node = temporal_cfg.get("seq_model", {}) or {}
        self.seq_model_cfg = {**_seq_defaults, **_seq_cfg_node}
        self.monitor_metric = temporal_cfg.get("monitor_metric", "val_acc")
        self.seq_model_cfg["monitor_metric"] = self.monitor_metric

        # -- Optimization flags --
        # Neural network path
        self.use_focal_loss        = bool(temporal_cfg.get("use_focal_loss",         False))
        self.focal_alpha           = float(temporal_cfg.get("focal_alpha",           0.25))
        self.focal_gamma           = float(temporal_cfg.get("focal_gamma",           2.0))
        self.use_class_weights     = bool(temporal_cfg.get("use_class_weights",      False))
        self.seq_output_residual   = bool(temporal_cfg.get("seq_output_residual",    False))
        self.seq_use_extra_features = bool(temporal_cfg.get("seq_use_extra_features", False))
        # LGBM path
        self.multi_scale_windows   = temporal_cfg.get("multi_scale_windows", [])
        if isinstance(self.multi_scale_windows, (int, float)):
            self.multi_scale_windows = [int(self.multi_scale_windows)]
        if not isinstance(self.multi_scale_windows, list):
            self.multi_scale_windows = []
        self.multi_scale_windows = [int(w) for w in self.multi_scale_windows if int(w) > 0]
        self.enhanced_features     = bool(temporal_cfg.get("enhanced_features",      False))
        self.autocorr_features     = bool(temporal_cfg.get("autocorr_features",      False))
        self.cross_scale_features  = bool(temporal_cfg.get("cross_scale_features",   False))
        self.distribution_shape    = bool(temporal_cfg.get("distribution_shape",     False))

        # -- Sequence decoding optimization --
        self.min_segment         = int(temporal_cfg.get("min_segment",         30))
        self.self_trans_boost    = float(temporal_cfg.get("self_trans_boost",  30.0))
        self.duration_penalty    = float(temporal_cfg.get("duration_penalty",  5.0))
        self.prob_aware_merge    = bool(temporal_cfg.get("prob_aware_merge",   True))
        self.seq_decoder_mode    = temporal_cfg.get("seq_decoder_mode", "viterbi_duration")
        self.per_class_duration  = bool(temporal_cfg.get("per_class_duration", False))
        self.per_class_transition = bool(temporal_cfg.get("per_class_transition", False))
        # CRF control
        self.skip_crf            = bool(temporal_cfg.get("skip_crf",           False))
        self.crf_max_train_samples = int(temporal_cfg.get("crf_max_train_samples", 50000))
        self.tune_crf_hyperparams  = bool(temporal_cfg.get("tune_crf_hyperparams",  False))
        # Mode normalization: remove spaces, lowercase
        self.seq_decoder_mode = self.seq_decoder_mode.lower().replace(" ", "")

        # Inject optimization flags into seq_model_cfg (for temporal_models.py)
        self.seq_model_cfg["use_focal_loss"]         = self.use_focal_loss
        self.seq_model_cfg["focal_alpha"]            = self.focal_alpha
        self.seq_model_cfg["focal_gamma"]            = self.focal_gamma
        self.seq_model_cfg["use_class_weights"]      = self.use_class_weights
        self.seq_model_cfg["seq_output_residual"]    = self.seq_output_residual
        self.seq_model_cfg["seq_use_extra_features"] = self.seq_use_extra_features

        if self.temporal_model != "lgbm":
            _flags = []
            if self.use_focal_loss:       _flags.append("focal_loss")
            if self.use_class_weights:    _flags.append("class_weights")
            if self.seq_output_residual:  _flags.append("output_residual")
            if self.seq_use_extra_features: _flags.append("extra_features")
            # New training enhancements
            _seq = self.seq_model_cfg
            if _seq.get("use_layer_norm"):     _flags.append("LayerNorm")
            if _seq.get("use_deep_projection"): _flags.append("DeepProj")
            if _seq.get("use_attention_pooling"): _flags.append("AttnPool")
            if _seq.get("label_smoothing", 0) > 0: _flags.append(f"LabelSmooth({_seq['label_smoothing']})")
            if _seq.get("use_cosine_warmup"): _flags.append("CosineWarmup")
            if _seq.get("use_ema"):         _flags.append(f"EMA({_seq.get('ema_decay', 0.999)})")
            if _seq.get("use_balanced_sampling"): _flags.append("BalancedSamp")
            if _seq.get("use_mixup"):       _flags.append(f"Mixup({_seq.get('mixup_alpha', 0.2)})")
            if _seq.get("use_time_reversal"): _flags.append("TimeRev")
            _opt_str = f" opts={' '.join(_flags)}" if _flags else ""
            logger.info(
                f"Temporal model: {self.temporal_model} (hidden_dim={self.seq_model_cfg['hidden_dim']}, "
                f"num_layers={self.seq_model_cfg['num_layers']}, device={self.seq_model_cfg['device']}{_opt_str})"
            )
        else:
            _flags = []
            if self.multi_scale_windows: _flags.append(f"multi_scale={self.multi_scale_windows}")
            if self.enhanced_features:   _flags.append("enhanced")
            if self.autocorr_features:   _flags.append("autocorr")
            if self.cross_scale_features: _flags.append("cross_scale")
            if self.distribution_shape:  _flags.append("dist_shape")
            # Decoder optimization flags (shared between LGBM and NN paths)
            _dflags = []
            if self.self_trans_boost > 0:   _dflags.append(f"self_boost={self.self_trans_boost}")
            if self.duration_penalty > 0:    _dflags.append(f"dur_pen={self.duration_penalty}")
            if self.prob_aware_merge:        _dflags.append("prob_merge")
            if self.per_class_duration:      _dflags.append("per_class_dur")
            if self.per_class_transition:    _dflags.append("per_class_trans")
            _dflags.append(f"decoder={self.seq_decoder_mode}")
            _flags.extend(_dflags)
            _opt_str = f" opts={' '.join(_flags)}" if _flags else ""
            if _opt_str:
                logger.info(f"Temporal model: lgbm{_opt_str})")

        # Decoder optimization log (NN path also prints)
        _dlog_parts = []
        if self.self_trans_boost > 0:
            _dlog_parts.append(f"self_trans_boost={self.self_trans_boost}")
        if self.duration_penalty > 0:
            _dlog_parts.append(f"duration_penalty={self.duration_penalty}")
        if self.prob_aware_merge:
            _dlog_parts.append("prob_aware_merge")
        if self.per_class_duration:
            _dlog_parts.append("per_class_duration")
        if self.per_class_transition:
            _dlog_parts.append("per_class_transition")
        _dlog_parts.append(f"decoder={self.seq_decoder_mode}")
        if _dlog_parts:
            logger.info(f"[Decoder] Sequence decoding optimization: {', '.join(_dlog_parts)}")

        # CRF optimization status (printed for non-LGBM paths; LGBM path also applies)
        _crf_flags = []
        if self.skip_crf:
            _crf_flags.append("skip")
        else:
            _crf_flags.append(f"max_train_samples={self.crf_max_train_samples}")
            if self.tune_crf_hyperparams:
                _crf_flags.append("tune_hyperparams")
            if self.temporal_model != "lgbm":
                _crf_flags.append("enriched_features(delta+entropy+margin)")
        if _crf_flags:
            logger.info(f"[CRF] Config: {', '.join(_crf_flags)}")

    def _build_decoders(self) -> list:
        """Build decoder list.

        Prefers the new-format decoders list config; falls back to parsing the
        old seq_decoder_mode string.

        New format example:
            decoders:
              - name: viterbi_duration
                config: {min_segment: 30, ...}
              - name: crf
                config: {max_train_samples: 50000, ...}
              - name: hmm
                config: {n_iter: 100, ...}
              - name: rl
                config: {n_episodes: 1000, ...}

        Old format (backward compatible):
            seq_decoder_mode: "viterbi_duration+crf"
        """
        # -- Try new format --
        new_decoders = self._temporal_cfg.get("decoders", None)
        if new_decoders is not None:
            return self._build_decoders_from_list(new_decoders)

        # -- Old format fallback --
        decoders = []
        mode = self.seq_decoder_mode
        use_duration = "duration" in mode
        use_crf = "crf" in mode and not self.skip_crf

        if "viterbi" in mode:
            if use_duration:
                dec = DurationViterbiDecoder(
                    min_duration=self.min_segment,
                    duration_penalty=self.duration_penalty,
                    self_boost=self.self_trans_boost,
                    cross_smoothing=1.0,
                    prob_aware_merge=self.prob_aware_merge,
                    per_class_duration=self.per_class_duration,
                    per_class_transition=self.per_class_transition,
                )
                pc_flags = []
                if self.per_class_duration: pc_flags.append("per_class_duration")
                if self.per_class_transition: pc_flags.append("per_class_transition")
                pc_str = f", {', '.join(pc_flags)}" if pc_flags else ""
                logger.info(
                    f"[Decoder] Using DurationViterbiDecoder: min_duration={self.min_segment}, "
                    f"duration_penalty={self.duration_penalty}, self_boost={self.self_trans_boost}, "
                    f"prob_aware_merge={self.prob_aware_merge}{pc_str}"
                )
            else:
                dec = ViterbiDecoder(
                    min_segment=self.min_segment,
                    self_boost=self.self_trans_boost,
                    cross_smoothing=1.0,
                    prob_aware_merge=self.prob_aware_merge,
                    per_class_transition=self.per_class_transition,
                )
                logger.info(
                    f"[Decoder] Using ViterbiDecoder: min_segment={self.min_segment}, "
                    f"self_boost={self.self_trans_boost}, prob_aware_merge={self.prob_aware_merge}"
                )
            decoders.append(dec)

        if use_crf:
            decoders.append(CRFDecoder(
                max_train_samples=self.crf_max_train_samples,
                tune_hyperparams=self.tune_crf_hyperparams,
            ))

        if not decoders:
            logger.warning("[Decoder] No decoder matched; skipping sequence decoding")

        return decoders

    def _build_decoders_from_list(self, decoder_specs: list) -> list:
        """Build decoder instances from list-style config."""
        decoders = []
        for spec in decoder_specs:
            name = spec.get("name", "").lower().replace(" ", "")
            cfg = spec.get("config", {}) or {}

            if name in ("viterbi", "viterbi_duration"):
                if "duration" in name:
                    dec = DurationViterbiDecoder(
                        min_duration=cfg.get("min_segment", self.min_segment),
                        duration_penalty=cfg.get("duration_penalty", self.duration_penalty),
                        self_boost=cfg.get("self_trans_boost", self.self_trans_boost),
                        cross_smoothing=cfg.get("cross_smoothing", 1.0),
                        prob_aware_merge=cfg.get("prob_aware_merge", self.prob_aware_merge),
                        per_class_duration=cfg.get("per_class_duration", self.per_class_duration),
                        per_class_transition=cfg.get("per_class_transition", self.per_class_transition),
                    )
                else:
                    dec = ViterbiDecoder(
                        min_segment=cfg.get("min_segment", self.min_segment),
                        self_boost=cfg.get("self_trans_boost", self.self_trans_boost),
                        cross_smoothing=cfg.get("cross_smoothing", 1.0),
                        prob_aware_merge=cfg.get("prob_aware_merge", self.prob_aware_merge),
                        per_class_transition=cfg.get("per_class_transition", self.per_class_transition),
                    )
                logger.info(f"[Decoder] [{name}] Created from list config")
                decoders.append(dec)

            elif name == "crf":
                if cfg.get("skip", self.skip_crf):
                    logger.info(f"[Decoder] [crf] skip=true, skipping")
                    continue
                decoders.append(CRFDecoder(
                    c1=cfg.get("c1", 0.1),
                    c2=cfg.get("c2", 0.1),
                    max_iterations=cfg.get("max_iterations", 100),
                    n_bins=cfg.get("n_bins", 10),
                    max_seq_len=cfg.get("max_seq_len", 0),
                    use_enriched_features=cfg.get("use_enriched_features", True),
                    max_train_samples=cfg.get("max_train_samples", self.crf_max_train_samples),
                    tune_hyperparams=cfg.get("tune_hyperparams", self.tune_crf_hyperparams),
                ))
                logger.info(f"[Decoder] [crf] Created from list config")

            elif name == "hmm":
                from src.decoders.hmm import HMMDecoder
                decoders.append(HMMDecoder(
                    n_components=cfg.get("n_components"),
                    covariance_type=cfg.get("covariance_type", "diag"),
                    n_iter=cfg.get("n_iter", 100),
                    min_segment=cfg.get("min_segment", self.min_segment),
                    prob_aware_merge=cfg.get("prob_aware_merge", self.prob_aware_merge),
                    supervised_init=cfg.get("supervised_init", True),
                    refine_em=cfg.get("refine_em", False),
                ))
                logger.info(f"[Decoder] [hmm] Created from list config")

            elif name == "rl":
                from src.decoders.rl import RLDecoder
                decoders.append(RLDecoder(
                    min_segment=cfg.get("min_segment", self.min_segment),
                    prob_aware_merge=cfg.get("prob_aware_merge", self.prob_aware_merge),
                    learning_rate=cfg.get("learning_rate", 0.01),
                    n_episodes=cfg.get("n_episodes", 1000),
                    epsilon=cfg.get("epsilon", 0.1),
                    gamma=cfg.get("gamma", 0.9),
                    epsilon_min=cfg.get("epsilon_min", 0.01),
                    epsilon_decay=cfg.get("epsilon_decay", 0.995),
                ))
                logger.info(f"[Decoder] [rl] Created from list config")

            elif name == "rule_correct":
                from src.decoders.rules import RuleCorrector
                decoders.append(RuleCorrector(
                    max_flicker=cfg.get("max_flicker", 3),
                    flicker_prob_threshold=cfg.get("flicker_prob_threshold", 0.05),
                    window_size=cfg.get("window_size", 31),
                    proba_consistency_threshold=cfg.get("proba_consistency_threshold", 0.20),
                    min_consecutive=cfg.get("min_consecutive", 5),
                    boundary_refine=cfg.get("boundary_refine", True),
                    confidence_margin=cfg.get("confidence_margin", 0.3),
                    smooth_transitions=cfg.get("smooth_transitions", True),
                ))
                logger.info(f"[Decoder] [rule_correct] Created from list config")

            else:
                logger.warning(f"[Decoder] Unknown decoder type: {name}, skipping")

        if not decoders:
            logger.warning("[Decoder] List config produced no decoders")
        return decoders

    def _create_viterbi_decoder(self):
        """Create Viterbi decoder based on seq_decoder_mode (kept for NN path visualization)."""
        decoders = self._build_decoders()
        for d in decoders:
            if d.name in ("viterbi", "viterbi_duration"):
                return d
        return None

    def build_features(
        self,
        proba_train: np.ndarray,
        proba_val: np.ndarray,
    ) -> tuple:
        """Construct temporal feature matrices, returns (X_train_feat, X_val_feat)."""
        import gc

        if self.multi_scale_windows:
            logger.info(
                f"Constructing multi-scale temporal features: window_sizes={self.multi_scale_windows}, "
                f"train={proba_train.shape}, val={proba_val.shape}"
            )
            X_train_feat = _build_temporal_features_multiscale(proba_train, self.multi_scale_windows)
            X_val_feat   = _build_temporal_features_multiscale(proba_val,   self.multi_scale_windows)
        else:
            logger.info(
                f"Constructing temporal features: window_size={self.window_size}, "
                f"train={proba_train.shape}, val={proba_val.shape}"
            )
            X_train_feat = _build_temporal_features(proba_train, self.window_size)
            X_val_feat   = _build_temporal_features(proba_val,   self.window_size)

        # Enhanced temporal features (appended, not replacing)
        if self.enhanced_features:
            logger.info("Appending enhanced temporal features (diff/EMA/cross-category)")
            enh_train = _build_enhanced_temporal_features(proba_train)
            enh_val   = _build_enhanced_temporal_features(proba_val)
            X_train_feat = np.hstack([X_train_feat, enh_train])
            X_val_feat   = np.hstack([X_val_feat,   enh_val])
            del enh_train, enh_val; gc.collect()

        # Autocorrelation features (appended)
        if self.autocorr_features:
            logger.info("Appending autocorrelation features (lag 1/2/3)")
            ac_train = _build_autocorr_features(proba_train)
            ac_val   = _build_autocorr_features(proba_val)
            X_train_feat = np.hstack([X_train_feat, ac_train])
            X_val_feat   = np.hstack([X_val_feat,   ac_val])
            del ac_train, ac_val; gc.collect()

        # Distribution shape features (appended)
        if self.distribution_shape:
            logger.info("Appending distribution shape features (skewness/kurtosis)")
            shape_train = _build_distribution_shape_features(proba_train)
            shape_val   = _build_distribution_shape_features(proba_val)
            X_train_feat = np.hstack([X_train_feat, shape_train])
            X_val_feat   = np.hstack([X_val_feat,   shape_val])
            del shape_train, shape_val; gc.collect()

        # Cross-scale interaction features (appended)
        if self.cross_scale_features and self.multi_scale_windows:
            logger.info("Appending cross-scale interaction features (short-long trend/vol/peak diff)")
            short_ws = min(self.multi_scale_windows)
            long_ws = max(self.multi_scale_windows)
            cs_train = _build_cross_scale_features(proba_train, short_ws=short_ws, long_ws=long_ws)
            cs_val   = _build_cross_scale_features(proba_val,   short_ws=short_ws, long_ws=long_ws)
            X_train_feat = np.hstack([X_train_feat, cs_train])
            X_val_feat   = np.hstack([X_val_feat,   cs_val])
            del cs_train, cs_val; gc.collect()

        logger.info(f"Temporal feature dimension: {X_val_feat.shape[1]}")
        return X_train_feat, X_val_feat

    def train_and_eval(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        label_map: dict,
    ) -> tuple:
        """
        Train second-layer LightGBM and evaluate.
        Returns (result_dict, cm, cm_norm, class_names), consistent with
        SynthValidator.train_and_eval format.
        """
        y_train = np.asarray(y_train).astype(int)
        y_val = np.asarray(y_val).astype(int)

        classes_sorted = sorted(set(int(v) for v in label_map.values()))
        id_to_name = {int(v): k for k, v in label_map.items()}
        class_names = [id_to_name[c] for c in classes_sorted]
        n_classes = len(classes_sorted)

        logger.info(
            f"[Temporal] Training second-order LGBM: n_classes={n_classes}, "
            f"feature_dim={X_train.shape[1]}, "
            f"train={len(y_train)}, val={len(y_val)}"
        )
        logger.info(f"[Temporal] Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        logger.info(f"[Temporal] Val label distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            class_weight=self.class_weight if self.class_weight else None,
            verbose=-1,
            **self.lgbm_params,
        )
        fit_kwargs = {}
        if self.early_stopping_rounds > 0:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=10),
            ]
        model.fit(X_train, y_train, **fit_kwargs)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        model_classes = list(model.classes_)

        full_proba = np.zeros((len(y_val), n_classes), dtype=np.float32)
        for i, c in enumerate(model_classes):
            if c in classes_sorted:
                full_proba[:, classes_sorted.index(c)] = y_proba[:, i]

        cm = confusion_matrix(y_val, y_pred, labels=classes_sorted)
        cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        # top-k accuracy
        top_k_acc = None
        top_k_per_class = {}
        if self.top_k >= 2:
            k = min(self.top_k, n_classes)
            top_k_cols = np.argsort(full_proba, axis=1)[:, -k:]
            top_k_class_ids = np.array([[classes_sorted[j] for j in row] for row in top_k_cols])
            hit = np.array([int(y_val[i]) in top_k_class_ids[i] for i in range(len(y_val))])
            top_k_acc = float(hit.mean())
            for c in classes_sorted:
                mask = y_val == c
                top_k_per_class[id_to_name[c]] = (
                    round(float(hit[mask].mean()), 4) if mask.any() else None
                )
            logger.info(f"[Temporal] top-{k} accuracy: {top_k_acc:.4f}")

        acc = float(accuracy_score(y_val, y_pred))
        balanced_acc = float(balanced_accuracy_score(y_val, y_pred))
        macro_f1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="weighted", zero_division=0))
        per_class_f1 = f1_score(y_val, y_pred, labels=classes_sorted, average=None, zero_division=0)

        present = sorted(set(int(v) for v in y_val))
        macro_auc, weighted_auc = 0.0, 0.0
        per_class_auc = {cname: None for cname in class_names}
        if len(present) >= 2:
            idx_present = [classes_sorted.index(c) for c in present]
            proba_sub = full_proba[:, idx_present]
            row_sum = proba_sub.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum <= 0, 1.0, row_sum)
            proba_sub_n = proba_sub / row_sum
            try:
                macro_auc = float(roc_auc_score(y_val, proba_sub_n, multi_class="ovr",
                                                 average="macro", labels=present))
            except ValueError as e:
                logger.warning(f"[Temporal] macro AUC computation failed: {e}")
            try:
                weighted_auc = float(roc_auc_score(y_val, proba_sub_n, multi_class="ovr",
                                                    average="weighted", labels=present))
            except ValueError as e:
                logger.warning(f"[Temporal] weighted AUC computation failed: {e}")
            for c in present:
                cname = id_to_name[c]
                y_bin = (y_val == c).astype(int)
                col = classes_sorted.index(c)
                try:
                    per_class_auc[cname] = float(roc_auc_score(y_bin, full_proba[:, col]))
                except ValueError:
                    per_class_auc[cname] = None

        report = classification_report(
            y_val, y_pred, labels=classes_sorted, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        per_class_metrics = {}
        for i, c in enumerate(classes_sorted):
            cname = class_names[i]
            r = report.get(cname, {})
            per_class_metrics[cname] = {
                "precision": round(float(r.get("precision", 0.0)), 4),
                "recall": round(float(r.get("recall", 0.0)), 4),
                "f1": round(float(per_class_f1[i]), 4),
                "support": int(r.get("support", 0)),
                "auc": round(per_class_auc[cname], 4) if per_class_auc[cname] is not None else None,
            }

        result = {
            "metrics": {
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(balanced_acc, 4),
                "top_k_accuracy": round(top_k_acc, 4) if top_k_acc is not None else None,
                "top_k": self.top_k if top_k_acc is not None else None,
                "top_k_per_class": top_k_per_class if top_k_acc is not None else {},
                "macro_auc": round(macro_auc, 4),
                "weighted_auc": round(weighted_auc, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_f1": round(weighted_f1, 4),
                "per_class": per_class_metrics,
            },
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "class_ids_in_order": classes_sorted,
            "class_names_in_order": class_names,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "window_size": self.window_size,
            "feature_dim": int(X_train.shape[1]),
        }
        # Temporal LGBM overall metrics summary
        logger.info(
            f"[Temporal][LGBM] accuracy={acc:.4f}  "
            f"balanced_acc(per_class_acc_mean)={balanced_acc:.4f}  "
            f"macro_auc={macro_auc:.4f}  weighted_auc={weighted_auc:.4f}  "
            f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}  "
            f"top_k_accuracy={top_k_acc if top_k_acc is not None else 'N/A'}"
        )
        for cname, m in per_class_metrics.items():
            logger.info(
                f"[Temporal][LGBM][per_class] {cname}: "
                f"precision={m['precision']}  recall={m['recall']}  "
                f"f1={m['f1']}  auc={m['auc']}  support={m['support']}"
            )
        return result, cm, cm_norm, class_names, full_proba, model

    def save_report(
        self,
        result: dict,
        cm: np.ndarray,
        cm_norm: np.ndarray,
        class_names: list,
        output_dir: str,
    ) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "temporal_report.json"
        # Filter non-JSON-serializable values (ndarray etc.), skip keys starting with _
        _json_safe = {}
        for _k, _v in result.items():
            if _k.startswith("_"):
                continue
            if isinstance(_v, np.ndarray):
                _json_safe[_k] = _v.tolist()
            elif isinstance(_v, dict):
                _json_safe[_k] = {
                    _dk: (_dv.tolist() if isinstance(_dv, np.ndarray) else _dv)
                    for _dk, _dv in _v.items()
                }
            else:
                _json_safe[_k] = _v
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe, f, ensure_ascii=False, indent=2)
        logger.info(f"[Temporal] Report saved: {json_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping confusion matrix image output.")
            return {"json": str(json_path)}

        def _plot_cm(mat, title, fname, fmt):
            fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.9),
                                             max(5, len(class_names) * 0.8)))
            im = ax.imshow(mat, interpolation="nearest", cmap="Greens")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            thresh = mat.max() / 2.0 if mat.size else 0
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, format(mat[i, j], fmt),
                            ha="center", va="center",
                            color="white" if mat[i, j] > thresh else "black",
                            fontsize=8)
            plt.tight_layout()
            path = out / fname
            plt.savefig(path, dpi=150)
            plt.close(fig)
            logger.info(f"[Temporal] Confusion matrix saved: {path}")
            return str(path)

        raw_png = _plot_cm(cm, "Temporal CM (raw)", "temporal_confusion_matrix_raw.png", "d")
        # CSV: raw confusion matrix
        _cm_csv_rows = [[class_names[i]] + [str(cm[i, j]) for j in range(len(class_names))]
                        for i in range(len(class_names))]
        _write_csv(out / "temporal_confusion_matrix_raw.csv", _cm_csv_rows,
                   ["true_class"] + [f"pred_{c}" for c in class_names])

        norm_png = ""
        if self.normalize_cm:
            norm_png = _plot_cm(cm_norm, "Temporal CM (row-normalized)",
                                "temporal_confusion_matrix.png", ".2f")
            # CSV: normalized confusion matrix
            _cm_norm_csv_rows = [[class_names[i]] + [f"{cm_norm[i, j]:.4f}" for j in range(len(class_names))]
                                 for i in range(len(class_names))]
            _write_csv(out / "temporal_confusion_matrix.csv", _cm_norm_csv_rows,
                       ["true_class"] + [f"pred_{c}" for c in class_names])

        return {"json": str(json_path), "cm_raw_png": raw_png, "cm_norm_png": norm_png}

    def run(
        self,
        proba_train: np.ndarray,
        y_train: np.ndarray,
        proba_val: np.ndarray,
        y_val: np.ndarray,
        label_map: dict,
        output_dir: str,
        min_segment: int = 30,
        extra_train: "np.ndarray | None" = None,
        extra_val: "np.ndarray | None" = None,
        viz_dir: str = "",
    ) -> dict:
        """End-to-end entry: feature construction -> training -> evaluation -> sequence decoding (Viterbi + CRF) -> save report.

        extra_train / extra_val: optional extra feature matrices [T, D_extra].
            - lgbm path: concatenated at the end of temporal feature matrix after construction
            - NN path (seq_use_extra_features=true): concatenated with proba as model input
        """

        # ---- Neural network temporal model path (bilstm / transformer / mamba) ----
        if self.temporal_model != "lgbm":
            logger.info(f"Using neural network temporal model: {self.temporal_model}")
            from src.temporal_models import train_sequence_model, predict_sequence_model, grid_search_sequence_model

            classes_sorted = sorted(set(int(v) for v in label_map.values()))
            n_classes = len(classes_sorted)
            id_to_name = {int(v): k for k, v in label_map.items()}
            class_names = [id_to_name[c] for c in classes_sorted]

            # -- Hyperparameter grid search --
            grid_search_result = None
            if self.seq_model_cfg.get("grid_search", {}).get("enabled", False):
                logger.info(
                    f"[{self.temporal_model}] === Starting hyperparameter grid search ==="
                )
                grid_search_result = grid_search_sequence_model(
                    model_type=self.temporal_model,
                    proba_train=proba_train,
                    y_train=np.asarray(y_train).astype(int),
                    proba_val=proba_val,
                    y_val=np.asarray(y_val).astype(int),
                    n_classes=n_classes,
                    cfg=self.seq_model_cfg,
                    extra_train=extra_train if self.seq_use_extra_features else None,
                    extra_val=extra_val if self.seq_use_extra_features else None,
                    output_dir=output_dir,
                )
                if not grid_search_result.get("skipped") and grid_search_result.get("best_params"):
                    # Merge best params into seq_model_cfg (override defaults)
                    best_p = grid_search_result["best_params"]
                    logger.info(
                        f"[{self.temporal_model}] Grid search done, applying best params: {best_p}"
                    )
                    for k, v in best_p.items():
                        self.seq_model_cfg[k] = v

            # -- Train final model with (potentially grid-search-optimized) config --
            model, _ = train_sequence_model(
                model_type=self.temporal_model,
                proba_train=proba_train,
                y_train=np.asarray(y_train).astype(int),
                proba_val=proba_val,
                y_val=np.asarray(y_val).astype(int),
                n_classes=n_classes,
                cfg=self.seq_model_cfg,
                extra_train=extra_train if self.seq_use_extra_features else None,
                extra_val=extra_val if self.seq_use_extra_features else None,
                save_dir=output_dir,
            )

            proba_val_refined = predict_sequence_model(
                model=model,
                proba=proba_val,
                cfg=self.seq_model_cfg,
                extra=extra_val if self.seq_use_extra_features else None,
            )

            y_val_arr = np.asarray(y_val).astype(int)
            y_pred = np.array([classes_sorted[i] for i in proba_val_refined.argmax(axis=1)])

            cm = confusion_matrix(y_val_arr, y_pred, labels=classes_sorted)
            cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

            acc = float(accuracy_score(y_val_arr, y_pred))
            balanced_acc = float(balanced_accuracy_score(y_val_arr, y_pred))
            macro_f1 = float(f1_score(y_val_arr, y_pred, labels=classes_sorted, average="macro", zero_division=0))
            weighted_f1 = float(f1_score(y_val_arr, y_pred, labels=classes_sorted, average="weighted", zero_division=0))
            per_class_f1_arr = f1_score(y_val_arr, y_pred, labels=classes_sorted, average=None, zero_division=0)

            present = sorted(set(int(v) for v in y_val_arr))
            macro_auc, weighted_auc = 0.0, 0.0
            per_class_auc = {cname: None for cname in class_names}
            if len(present) >= 2:
                idx_present = [classes_sorted.index(c) for c in present]
                proba_sub = proba_val_refined[:, idx_present]
                row_sum = proba_sub.sum(axis=1, keepdims=True)
                row_sum = np.where(row_sum <= 0, 1.0, row_sum)
                proba_sub_n = proba_sub / row_sum
                try:
                    macro_auc = float(roc_auc_score(y_val_arr, proba_sub_n, multi_class="ovr",
                                                     average="macro", labels=present))
                except ValueError:
                    pass
                try:
                    weighted_auc = float(roc_auc_score(y_val_arr, proba_sub_n, multi_class="ovr",
                                                        average="weighted", labels=present))
                except ValueError:
                    pass
                for c in present:
                    cname = id_to_name[c]
                    y_bin = (y_val_arr == c).astype(int)
                    col = classes_sorted.index(c)
                    try:
                        per_class_auc[cname] = float(roc_auc_score(y_bin, proba_val_refined[:, col]))
                    except ValueError:
                        per_class_auc[cname] = None

            report_dict = classification_report(
                y_val_arr, y_pred, labels=classes_sorted, target_names=class_names,
                output_dict=True, zero_division=0,
            )
            per_class_metrics = {}
            for i, c in enumerate(classes_sorted):
                cname = class_names[i]
                r = report_dict.get(cname, {})
                per_class_metrics[cname] = {
                    "precision": round(float(r.get("precision", 0.0)), 4),
                    "recall":    round(float(r.get("recall",    0.0)), 4),
                    "f1":        round(float(per_class_f1_arr[i]),     4),
                    "support":   int(r.get("support", 0)),
                    "auc":       round(per_class_auc[cname], 4) if per_class_auc[cname] is not None else None,
                }

            result = {
                "metrics": {
                    "accuracy":    round(acc,          4),
                    "balanced_accuracy": round(balanced_acc, 4),
                    "macro_auc":   round(macro_auc,    4),
                    "weighted_auc": round(weighted_auc, 4),
                    "macro_f1":    round(macro_f1,     4),
                    "weighted_f1": round(weighted_f1,  4),
                    "per_class":   per_class_metrics,
                },
                "confusion_matrix":            cm.tolist(),
                "confusion_matrix_normalized": cm_norm.tolist(),
                "class_ids_in_order":          classes_sorted,
                "class_names_in_order":        class_names,
                "n_train":    int(len(y_train)),
                "n_val":      int(len(y_val)),
                "window_size": self.window_size,
                "feature_dim": int(proba_train.shape[1]),
                "model":       self.temporal_model,
                "timestamp":   datetime.now().isoformat(),
                "optimizations": {
                    "focal_loss":       self.use_focal_loss,
                    "class_weights":    self.use_class_weights,
                    "output_residual":  self.seq_output_residual,
                    "extra_features":   self.seq_use_extra_features,
                    "grid_search":      {
                        "enabled": self.seq_model_cfg.get("grid_search", {}).get("enabled", False),
                        "best_params": grid_search_result.get("best_params", {}) if grid_search_result else {},
                        "best_score": grid_search_result.get("best_score", 0.0) if grid_search_result else 0.0,
                        "report_path": grid_search_result.get("report_path", "") if grid_search_result else "",
                    } if grid_search_result and not grid_search_result.get("skipped") else None,
                    "decoder": {
                        "mode":              self.seq_decoder_mode,
                        "min_segment":       self.min_segment,
                        "self_trans_boost":  self.self_trans_boost,
                        "duration_penalty":  self.duration_penalty,
                        "prob_aware_merge":  self.prob_aware_merge,
                        "per_class_duration": self.per_class_duration,
                        "per_class_transition": self.per_class_transition,
                    },
                },
            }
            # Temporal neural network model overall metrics summary
            logger.info(
                f"[Temporal][{self.temporal_model}] accuracy={acc:.4f}  "
                f"balanced_acc(per_class_acc_mean)={balanced_acc:.4f}  "
                f"macro_auc={macro_auc:.4f}  weighted_auc={weighted_auc:.4f}  "
                f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}"
            )
            for cname, m in per_class_metrics.items():
                logger.info(
                    f"[Temporal][{self.temporal_model}][per_class] {cname}: "
                    f"precision={m['precision']}  recall={m['recall']}  "
                    f"f1={m['f1']}  auc={m['auc']}  support={m['support']}"
                )

            # ---- Sequence decoding (pipeline orchestration) ----
            # Step 1: Prepare training data (sampling + NN inference for CRF/HMM/RL)
            _y_train_arr = np.asarray(y_train).astype(int)
            _extra_train = extra_train if self.seq_use_extra_features else None
            _proba_train_for_decoders = proba_train
            _y_train_for_decoders = _y_train_arr
            _extra_train_for_decoders = _extra_train
            if self.crf_max_train_samples > 0 and proba_train.shape[0] > self.crf_max_train_samples:
                _rng = np.random.default_rng(42)
                _idx = _rng.choice(proba_train.shape[0], self.crf_max_train_samples, replace=False)
                _idx.sort()
                _proba_train_for_decoders = proba_train[_idx]
                _y_train_for_decoders = _y_train_arr[_idx]
                if _extra_train is not None:
                    _extra_train_for_decoders = _extra_train[_idx]
                logger.info(
                    f"[{self.temporal_model}][Decoder] Training data sampling: "
                    f"{proba_train.shape[0]} -> {self.crf_max_train_samples} frames"
                )

            # NN inference on training set (for CRF/HMM/RL to use refined probabilities)
            logger.info(
                f"[{self.temporal_model}][Decoder] Training set NN inference -- "
                f"proba_train shape={_proba_train_for_decoders.shape}"
            )
            proba_train_refined = predict_sequence_model(
                model=model, proba=_proba_train_for_decoders, cfg=self.seq_model_cfg,
                extra=_extra_train_for_decoders,
            )

            # Step 2: Build pipeline and run
            decoders = self._build_decoders()
            pipeline = DecoderPipeline(
                decoders=decoders,
                classes_sorted=classes_sorted,
                id_to_name=id_to_name,
                output_dir=output_dir,
                viz_dir=viz_dir,
            )
            seq_decoding: dict = pipeline.run(
                proba_val=proba_val_refined,
                y_val=y_val_arr,
                proba_train=proba_train_refined,
                y_train=_y_train_for_decoders,
                feat_train=None,   # NN path: CRF builds enriched features from proba
                feat_val=None,
            )

            # Step 3: Viterbi visualization (using fitted decoders from pipeline)
            if viz_dir:
                viterbi = next((d for d in decoders if d.name in ("viterbi", "viterbi_duration")), None)
                if viterbi is not None:
                    try:
                        if getattr(viterbi, "log_trans", None) is not None:
                            plot_transition_matrix(
                                log_trans=viterbi.log_trans,
                                class_names=class_names,
                                viz_dir=viz_dir,
                            )
                        y_pred_raw = np.array([classes_sorted[i] for i in proba_val_refined.argmax(axis=1)])
                        y_viterbi = viterbi.decode(proba_val_refined)
                        plot_viterbi_comparison(
                            y_true=y_val_arr, y_pred_raw=y_pred_raw,
                            y_pred_viterbi=y_viterbi,
                            class_names=class_names,
                            viz_dir=viz_dir,
                        )
                        plot_proba_smoothing_effect(
                            proba_raw=proba_val_refined, labels=y_val_arr,
                            class_names=class_names,
                            window_size=self.window_size,
                            viz_dir=viz_dir,
                        )
                    except Exception as _e:
                        logger.warning(f"[Viz] Decoding visualization failed: {_e}")

            result["sequence_decoding"] = seq_decoding
            result["_proba_val"] = proba_val_refined  # For post-processing
            files = self.save_report(result, cm, cm_norm, class_names, output_dir)
            result["output_files"] = files
            return result

        # ---- LGBM path (original logic) ----
        X_train_feat, X_val_feat = self.build_features(proba_train, proba_val)

        # -- Visualization: temporal feature breakdown + probability smoothing effect --
        if viz_dir:
            try:
                n_classes = proba_train.shape[1]
                plot_temporal_feature_breakdown(
                    n_classes=n_classes,
                    multi_scale_windows=self.multi_scale_windows,
                    enhanced_features=self.enhanced_features,
                    autocorr_features=self.autocorr_features,
                    cross_scale_features=self.cross_scale_features,
                    distribution_shape=self.distribution_shape,
                    window_size=self.window_size,
                    viz_dir=viz_dir,
                )
            except Exception as _e:
                logger.warning(f"[Viz] Temporal feature breakdown visualization failed: {_e}")

        # Residual connection (Stage 2): concat meta_X directly at the end of temporal features
        if extra_train is not None and extra_val is not None:
            X_train_feat = np.hstack([X_train_feat, extra_train.astype(np.float32)])
            X_val_feat   = np.hstack([X_val_feat,   extra_val.astype(np.float32)])
            logger.info(
                f"[Temporal][Residual-Stage2] extra features concat'd shape={X_train_feat.shape}"
            )

        result, cm, cm_norm, class_names, lgbm_proba_val, lgbm_model = self.train_and_eval(
            X_train_feat, y_train, X_val_feat, y_val, label_map
        )
        # Save LGBM model weights
        try:
            _lgbm_out = Path(output_dir) / "weights"
            _lgbm_out.mkdir(parents=True, exist_ok=True)
            _lgbm_path = _lgbm_out / "temporal_lgbm_model.txt"
            lgbm_model.booster_.save_model(str(_lgbm_path))
            logger.info(f"[Temporal][LGBM] Model saved: {_lgbm_path}")
        except Exception as _e:
            logger.warning(f"[Temporal][LGBM] Model save failed: {_e}")
        result["timestamp"] = datetime.now().isoformat()
        result["model"] = "lgbm"
        result["optimizations"] = {
            "multi_scale_windows": self.multi_scale_windows if self.multi_scale_windows else [self.window_size],
            "enhanced_features":   self.enhanced_features,
            "autocorr_features":   self.autocorr_features,
            "cross_scale_features": self.cross_scale_features,
            "distribution_shape":  self.distribution_shape,
            "decoder": {
                "mode":              self.seq_decoder_mode,
                "min_segment":       self.min_segment,
                "self_trans_boost":  self.self_trans_boost,
                "duration_penalty":  self.duration_penalty,
                "prob_aware_merge":  self.prob_aware_merge,
                "per_class_duration": self.per_class_duration,
                "per_class_transition": self.per_class_transition,
            },
        }

        classes_sorted = sorted(set(int(v) for v in label_map.values()))
        id_to_name = {int(v): k for k, v in label_map.items()}
        n_classes = len(classes_sorted)

        # ---- Sequence decoding (pipeline orchestration) ----
        # LGBM path: CRF uses temporal feature matrices feat_train/feat_val;
        # Viterbi uses proba_train/proba_val (full y_train for transition matrix estimation).
        # CRF max_train_samples sampling is handled internally by CRFDecoder.
        decoders = self._build_decoders()
        pipeline = DecoderPipeline(
            decoders=decoders,
            classes_sorted=classes_sorted,
            id_to_name=id_to_name,
            output_dir=output_dir,
            viz_dir=viz_dir,
        )
        seq_decoding: dict = pipeline.run(
            proba_val=proba_val,
            y_val=y_val,
            proba_train=proba_train,
            y_train=np.asarray(y_train).astype(int),
            feat_train=X_train_feat,
            feat_val=X_val_feat,
        )

        # ---- Viterbi visualization (using fitted decoders from pipeline) ----
        if viz_dir:
            viterbi = next((d for d in decoders if d.name in ("viterbi", "viterbi_duration")), None)
            if viterbi is not None:
                try:
                    if getattr(viterbi, "log_trans", None) is not None:
                        plot_transition_matrix(
                            log_trans=viterbi.log_trans,
                            class_names=class_names,
                            viz_dir=viz_dir,
                        )
                    y_viterbi = viterbi.decode(proba_val)
                    y_pred_raw = np.array([classes_sorted[i] for i in lgbm_proba_val.argmax(axis=1)])
                    plot_viterbi_comparison(
                        y_true=y_val, y_pred_raw=y_pred_raw,
                        y_pred_viterbi=y_viterbi,
                        class_names=class_names,
                        viz_dir=viz_dir,
                    )
                    plot_proba_smoothing_effect(
                        proba_raw=proba_val, labels=y_val,
                        class_names=class_names,
                        window_size=self.window_size,
                        viz_dir=viz_dir,
                    )
                except Exception as _e:
                    logger.warning(f"[Viz] Decoding visualization failed: {_e}")

        result["sequence_decoding"] = seq_decoding
        result["_proba_val"] = lgbm_proba_val  # For post-processing

        files = self.save_report(result, cm, cm_norm, class_names, output_dir)
        result["output_files"] = files
        return result


# =========================================================================
# PostCalibrator: Posterior probability calibration
# =========================================================================

class PostCalibrator:
    """Posterior probability calibration: corrects probabilities before argmax.

    Three optional calibration methods (applied in sequence):
      1. Stage mixing: alpha[c] * BiLSTM[c] + (1-alpha[c]) * meta-LGBM[c] (per-class)
      2. Temperature scaling: softmax(log(p) / T)
      3. Per-class bias: additive bias in logit space
    """

    def __init__(self, temperature: float = 1.0, mix_alpha = 1.0,
                 biases: "np.ndarray | None" = None):
        self.temperature = temperature
        self.mix_alpha = mix_alpha      # float or [C] ndarray
        self.biases = biases            # [C] or None

    def calibrate(self, proba_bilstm, proba_meta=None):
        proba = proba_bilstm.astype(np.float64).copy()

        # 1. Stage mixing (per-class alpha)
        if proba_meta is not None:
            alpha = np.asarray(self.mix_alpha, dtype=np.float64)
            if alpha.ndim == 0:
                if alpha < 1.0 - 1e-8:
                    proba = float(alpha) * proba + (1.0 - float(alpha)) * proba_meta.astype(np.float64)
            else:
                # per-class: alpha[c] * BiLSTM[c] + (1-alpha[c]) * meta[c]
                meta = proba_meta.astype(np.float64)
                for c in range(min(len(alpha), proba.shape[1])):
                    a = alpha[c]
                    if a < 1.0 - 1e-8:
                        proba[:, c] = a * proba[:, c] + (1.0 - a) * meta[:, c]

        # 2. Temperature scaling
        if abs(self.temperature - 1.0) > 1e-6:
            log_p = np.log(np.clip(proba, 1e-300, 1.0))
            proba = np.exp(log_p / self.temperature)
            proba /= proba.sum(axis=1, keepdims=True)

        # 3. Per-class bias
        if self.biases is not None:
            log_p = np.log(np.clip(proba, 1e-300, 1.0))
            log_p += self.biases.astype(np.float64)
            proba = np.exp(log_p)
            proba /= proba.sum(axis=1, keepdims=True)

        return proba.astype(np.float32)


# =========================================================================
# tune_pipeline: Multi-stage search for optimal PostCalibrator + RuleCorrector parameters
# =========================================================================

def _tune_eval(proba, labels, y_val_arr, cls_sorted, id_to_name):
    """Quick evaluation of a set of probability outputs."""
    pred = proba.argmax(axis=1)
    return _eval_sequence_metrics(y_val_arr, pred, cls_sorted, id_to_name, proba=proba)


def tune_pipeline(proba_val, y_val, proba_meta, label_map, output_dir, cfg):
    """Multi-stage search for optimal PostCalibrator + RuleCorrector parameters.

    Search strategy (5 phases total):
      Phase 1: mix_alpha scalar coarse search -> per-class search -> refinement
      Phase 2: Multi-round per-class bias search (6 rounds with increasing range)
      Phase 3: Joint random search (alpha + bias + temperature, 500 rounds)
      Phase 4: Temperature fine search (0.75~1.35, step 0.01)
      Phase 5: RuleCorrector grid search (384 combinations)

    Returns:
        (best_all, best_acc) — best_all contains "acc", "calib", "rule_correct" keys
    """
    from src.decoders.rules import RuleCorrector
    import itertools

    classes_sorted = sorted(set(int(v) for v in label_map.values()))
    id_to_name = {int(v): k for k, v in label_map.items()}
    C = proba_val.shape[1]
    y_val_arr = np.asarray(y_val).astype(int)
    best_all = {"acc": 0.0, "calib": {"temperature": 1.0, "mix_alpha": 1.0, "biases": None},
                "rule_correct": {}}

    def _update(msg, calib_p, rc_p=None):
        cal_proba = PostCalibrator(**calib_p).calibrate(proba_val, proba_meta)
        if rc_p:
            rc = RuleCorrector(**rc_p)
            rc.fit(np.zeros(1, dtype=int), C)
            pred = rc.decode(cal_proba)
        else:
            pred = cal_proba.argmax(axis=1)
        m = _tune_eval(cal_proba, pred, y_val_arr, classes_sorted, id_to_name)
        if m["accuracy"] > best_all["acc"]:
            best_all["acc"] = m["accuracy"]
            best_all["calib"] = calib_p
            if rc_p:
                best_all["rule_correct"] = rc_p
            logger.info(f"  [Tune] {msg}: acc={m['accuracy']:.4f}  f1={m['macro_f1']:.4f}  *")
        return m

    # Baseline
    base_pred = proba_val.argmax(axis=1)
    base_m = _tune_eval(proba_val, base_pred, y_val_arr, classes_sorted, id_to_name)
    best_all["acc"] = base_m["accuracy"]
    logger.info(f"[Tune] Baseline (argmax): acc={base_m['accuracy']:.4f}  macro_f1={base_m['macro_f1']:.4f}")

    # ====== Phase 1: Search mix_alpha (scalar first, then per-class) ======
    logger.info("[Tune] Phase 1a: Coarse scalar mix_alpha search ...")
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        _update(f"alpha={alpha:.2f}", {"temperature": 1.0, "mix_alpha": alpha, "biases": None})

    # Find best scalar alpha, use as starting point for per-class alpha search
    best_scalar = best_all["calib"]["mix_alpha"]
    logger.info(f"[Tune] Phase 1b: Searching per-class mix_alpha (start={best_scalar:.2f}) ...")
    per_class_alpha = np.full(C, best_scalar, dtype=np.float64)
    for c in range(C):
        best_c_acc = 0.0
        best_c_alpha = per_class_alpha[c]
        for a in np.arange(max(0.0, best_scalar - 0.3), min(1.0, best_scalar + 0.35), 0.05):
            per_class_alpha[c] = a
            m = _update(
                f"  alpha[{c}]={a:.2f}",
                {"temperature": 1.0, "mix_alpha": per_class_alpha.copy(), "biases": None}
            )
            if m["accuracy"] > best_c_acc:
                best_c_acc = m["accuracy"]
                best_c_alpha = a
        per_class_alpha[c] = best_c_alpha
    logger.info(f"[Tune] Phase 1b done: per_class_alpha={per_class_alpha.round(3).tolist()}")
    logger.info(f"[Tune] Phase 1 best: acc={best_all['acc']:.4f}")

    # ====== Phase 1c: per-class alpha refinement (step 0.02) ======
    logger.info("[Tune] Phase 1c: Refining per-class alpha (step 0.02) ...")
    per_class_alpha = best_all["calib"]["mix_alpha"].copy()
    for c in range(C):
        best_c_alpha = per_class_alpha[c]
        center = best_c_alpha
        for a in np.arange(max(0.0, center - 0.10), min(1.01, center + 0.12), 0.02):
            per_class_alpha[c] = a
            m = _update(f"  alpha[{c}]={a:.2f}",
                        {"temperature": 1.0, "mix_alpha": per_class_alpha.copy(), "biases": None})
            if m["accuracy"] > best_all["acc"] or a == center:
                best_c_alpha = a
        per_class_alpha[c] = best_c_alpha
    logger.info(f"[Tune] Phase 1c done: alpha={per_class_alpha.round(3).tolist()}  acc={best_all['acc']:.4f}")

    # ====== Phase 2: Multi-round bias refinement ======
    logger.info("[Tune] Phase 2: Multi-round per-class bias search ...")
    best_alpha = best_all["calib"]["mix_alpha"]

    for round_idx, (bias_range, n_steps) in enumerate([
        (0.05, 7), (0.10, 9), (0.15, 11), (0.20, 13), (0.30, 15), (0.40, 17),
    ]):
        logger.info(f"  Round {round_idx+1}: bias_range={bias_range}, n_steps={n_steps}")
        best_biases = (best_all["calib"].get("biases")
                       if best_all["calib"].get("biases") is not None
                       else np.zeros(C, dtype=np.float64))

        for c in range(C):
            best_c_acc = 0.0
            best_c_bias = best_biases[c]
            candidates = np.linspace(-bias_range, bias_range, n_steps)
            for b in candidates:
                biases = best_biases.copy()
                biases[c] = b
                m = _update(
                    f"  bias[{c}]={b:.3f}",
                    {"temperature": 1.0, "mix_alpha": best_alpha, "biases": biases}
                )
                if m["accuracy"] > best_c_acc:
                    best_c_acc = m["accuracy"]
                    best_c_bias = b
            best_biases[c] = best_c_bias
        logger.info(f"  Round {round_idx+1} done: acc={best_all['acc']:.4f}")

    logger.info(f"[Tune] Phase 2 best: acc={best_all['acc']:.4f}")

    # ====== Phase 3: Joint random search (alpha + bias + temperature) ======
    logger.info("[Tune] Phase 3: Joint random perturbation (500 rounds) ...")
    np.random.seed(42)
    best_alpha = best_all["calib"]["mix_alpha"].copy()
    best_biases = (best_all["calib"]["biases"].copy()
                   if best_all["calib"]["biases"] is not None
                   else np.zeros(C, dtype=np.float64))
    best_temp = best_all["calib"]["temperature"]
    n_improved = 0

    for i in range(500):
        # Random perturbation strength: small (fine-tuning) vs large (exploration)
        if np.random.random() < 0.7:
            # Small perturbation: fine-tune near current best
            alpha_noise = np.random.randn(C) * 0.03
            bias_noise = np.random.randn(C) * 0.03
            temp_noise = np.random.randn() * 0.03
        else:
            # Large perturbation: escape local optimum
            alpha_noise = np.random.randn(C) * 0.10
            bias_noise = np.random.randn(C) * 0.10
            temp_noise = np.random.randn() * 0.10

        new_alpha = np.clip(best_alpha + alpha_noise, 0.1, 1.0)
        new_biases = best_biases + bias_noise
        new_temp = np.clip(best_temp + temp_noise, 0.7, 1.4)

        m = _update(
            f"joint_{i}",
            {"temperature": new_temp, "mix_alpha": new_alpha, "biases": new_biases}
        )
        if m["accuracy"] > best_all["acc"]:
            n_improved += 1
            best_alpha = new_alpha.copy()
            best_biases = new_biases.copy()
            best_temp = new_temp
            if n_improved % 5 == 0:
                logger.info(f"  [Tune] +{n_improved} improvements at iter {i}")

    logger.info(f"[Tune] Phase 3 done: {n_improved} improvements, acc={best_all['acc']:.4f}")

    # ====== Phase 4: Fine temperature search ======
    logger.info("[Tune] Phase 4: Fine temperature search (0.75~1.35, step 0.01) ...")
    best_biases = best_all["calib"]["biases"]
    best_alpha = best_all["calib"]["mix_alpha"]
    for temp in np.arange(0.75, 1.36, 0.01):
        _update(f"  T={temp:.2f}", {"temperature": float(temp), "mix_alpha": best_alpha,
                                      "biases": best_biases})
    logger.info(f"[Tune] Phase 4 done: acc={best_all['acc']:.4f}")

    # ====== Phase 5: Joint search with RuleCorrector ======
    logger.info("[Tune] Phase 5: Joint search with RuleCorrector ...")
    rc_grid = {
        "max_flicker": [2, 3, 4],
        "flicker_prob_threshold": [0.02, 0.05, 0.08, 0.12],
        "confidence_margin": [0.15, 0.25, 0.35, 0.50],
        "min_consecutive": [3, 5, 8],
        "smooth_transitions": [True, False],
    }
    rc_keys = list(rc_grid.keys())
    rc_values = list(rc_grid.values())
    n_rc = 1
    for v in rc_values:
        n_rc *= len(v)
    for n_done, rc_combo in enumerate(itertools.product(*rc_values)):
        rc_kw = dict(zip(rc_keys, rc_combo))
        rc_kw.update({"window_size": 31, "boundary_refine": True,
                       "proba_consistency_threshold": 0.20})
        if n_done % 30 == 0:
            logger.info(f"  [Tune] Phase 5: {n_done}/{n_rc}")
        _update(f"rc_{n_done}", best_all["calib"], rc_kw)

    logger.info(f"[Tune] === Best (acc={best_all['acc']:.4f}) ===")
    logger.info(f"[Tune]   calib: {best_all['calib']}")
    logger.info(f"[Tune]   rule_correct: {best_all['rule_correct']}")
    return best_all, best_all["acc"]
