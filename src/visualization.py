"""
visualization.py
Pipeline visualization utilities.

All functions save plots to subdirectories of a viz_dir root.
Each function is independent and handles missing dependencies gracefully.
"""

import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _write_csv(path: Path, rows: list, header: list) -> None:
    """Write CSV alongside a plot PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    logger.info(f"  Table saved: {path}")

_MPL_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib not available, visualizations disabled.")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ═══════════════════════════════════════════════════════════════
# 1) Data loading diagnostics
# ═══════════════════════════════════════════════════════════════

def plot_label_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    class_names: list,
    viz_dir: str,
):
    """Bar chart of per-class frame counts for train / val."""
    if not _MPL_AVAILABLE:
        return
    y_tr = np.asarray(y_train).astype(int)
    y_vl = np.asarray(y_val).astype(int)
    n_classes = len(class_names)
    tr_counts = [(y_tr == i).sum() for i in range(n_classes)]
    vl_counts = [(y_vl == i).sum() for i in range(n_classes)]

    out = _ensure_dir(Path(viz_dir) / "01_data")
    x = np.arange(n_classes)
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.7), 5))
    ax.bar(x - w / 2, tr_counts, w, label="Train", color="#4C72B0", edgecolor="white")
    ax.bar(x + w / 2, vl_counts, w, label="Val",   color="#DD8452", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Frame count")
    ax.set_title("Label Distribution (Train vs Val)")
    ax.legend()
    for i in range(n_classes):
        ax.text(i - w / 2, tr_counts[i] + max(tr_counts) * 0.01, str(tr_counts[i]),
                ha="center", fontsize=7)
        ax.text(i + w / 2, vl_counts[i] + max(vl_counts) * 0.01, str(vl_counts[i]),
                ha="center", fontsize=7)
    plt.tight_layout()
    path = out / "label_distribution.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV
    csv_rows = [[class_names[i], str(tr_counts[i]), str(vl_counts[i])] for i in range(n_classes)]
    _write_csv(out / "label_distribution.csv", csv_rows, ["class", "train_count", "val_count"])
    logger.info(f"[Viz] Label distribution saved: {path}")


def plot_keypoint_trajectory(
    train_kp: np.ndarray,
    train_lb: np.ndarray,
    flat_attributes: list,
    class_names: list,
    viz_dir: str,
    n_samples: int = 3,
    segment_len: int = 200,
):
    """
    Sample random segments and plot keypoint trajectories (x-y),
    colored by behavior class.
    """
    if not _MPL_AVAILABLE:
        return
    kp = np.asarray(train_kp).astype(np.float32)
    lb = np.asarray(train_lb).astype(int)
    T, D = kp.shape
    n_classes = len(class_names)

    # Pick body-center keypoints (Self_kp{0..N}_x / _y pairs)
    kp_indices = []
    for i in range(20):
        x_name = f"Self_kp{i}_x"
        y_name = f"Self_kp{i}_y"
        try:
            ix = flat_attributes.index(x_name)
            iy = flat_attributes.index(y_name)
            kp_indices.append((ix, iy))
        except ValueError:
            break
    if not kp_indices:
        logger.warning("[Viz] Self_kp keypoint columns not found, skipping trajectory plot.")
        return

    out = _ensure_dir(Path(viz_dir) / "01_data")
    rng = np.random.default_rng(42)
    n_classes_avail = len(np.unique(lb))
    cmap = plt.get_cmap("tab20" if n_classes_avail <= 20 else "hsv", n_classes)

    n_samples = min(n_samples, 5)
    fig, axes = plt.subplots(1, n_samples, figsize=(4.5 * n_samples, 4), squeeze=False)
    csv_rows = []
    for idx, ax in enumerate(axes.flat):
        # pick random start
        start = int(rng.integers(0, max(1, T - segment_len)))
        end = min(start + segment_len, T)
        seg_kp = kp[start:end]
        seg_lb = lb[start:end]
        frame_idx = np.arange(end - start)

        # Compute centroid trajectory from all available keypoints
        cx = np.mean([seg_kp[:, ix] for ix, _ in kp_indices], axis=0)
        cy = np.mean([seg_kp[:, iy] for _, iy in kp_indices], axis=0)

        # Collect CSV raw data for this segment
        for fi in range(len(frame_idx)):
            lid = int(seg_lb[fi])
            lname = class_names[lid] if lid < len(class_names) else f"class_{lid}"
            csv_rows.append([str(idx), str(int(start + fi)), f"{cx[fi]:.6f}", f"{cy[fi]:.6f}", str(lid), lname])

        for c in range(n_classes):
            mask = seg_lb == c
            if not mask.any():
                continue
            ax.scatter(cx[mask], cy[mask], c=[cmap(c)], s=3, alpha=0.6,
                       label=class_names[c] if idx == 0 else None)
        # Connect with thin line
        ax.plot(cx, cy, color="gray", alpha=0.3, linewidth=0.5)
        ax.set_title(f"Frames {start}–{end}", fontsize=9)
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")
        ax.invert_yaxis()
        ax.set_aspect("equal", adjustable="datalim")

    if n_classes <= 12:
        fig.legend(loc="lower center", ncol=min(n_classes, 6), fontsize=7,
                   bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.suptitle("Keypoint Trajectory Samples (centroid, colored by behavior)", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    path = out / "keypoint_trajectory_sample.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    _write_csv(out / "keypoint_trajectory_sample.csv", csv_rows,
               ["segment", "frame", "centroid_x", "centroid_y", "label_id", "label_name"])
    logger.info(f"[Viz] Keypoint trajectory saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 2) Factor matrix diagnostics
# ═══════════════════════════════════════════════════════════════

def plot_factor_valid_ratio(
    used_names: list,
    dropped_names: list,
    X: np.ndarray,
    viz_dir: str,
    group_name: str = "",
):
    """Bar chart: NaN ratio per factor, with dropped factors shown."""
    if not _MPL_AVAILABLE:
        return
    tag = f"_{group_name}" if group_name else ""
    out = _ensure_dir(Path(viz_dir) / "02_factors")
    X = np.asarray(X).astype(np.float32)

    all_names = used_names + list(dropped_names)
    if not all_names:
        return
    nan_ratios = []
    for i, name in enumerate(used_names):
        nan_ratios.append(float(np.isnan(X[:, i]).mean()))
    for name in dropped_names:
        nan_ratios.append(1.0)
    colors = ["#55A868" if i < len(used_names) else "#C44E52"
              for i in range(len(all_names))]

    # Sort by ratio descending
    order = np.argsort(nan_ratios)[::-1]
    names_sorted = [all_names[i] for i in order]
    ratios_sorted = [nan_ratios[i] for i in order]
    colors_sorted = [colors[i] for i in order]

    n = len(all_names)
    # Limit display to avoid enormous images when factor count is high
    max_display = 60
    if n > max_display:
        # Keep worst (highest NaN) + best (lowest NaN) factors
        keep = max_display // 2
        names_sorted = list(names_sorted[:keep]) + list(names_sorted[-keep:])
        ratios_sorted = list(ratios_sorted[:keep]) + list(ratios_sorted[-keep:])
        colors_sorted = list(colors_sorted[:keep]) + list(colors_sorted[-keep:])
        n = len(names_sorted)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.22), 5))
    bars = ax.bar(range(n), ratios_sorted, color=colors_sorted, edgecolor="white", linewidth=0.3)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names_sorted, rotation=80, ha="right", fontsize=6)
    ax.set_ylabel("NaN ratio")
    ax.set_title(f"Factor Valid Ratio{tag} (green=used, red=dropped)")
    ax.set_ylim(0, 1.05)
    green_patch = mpatches.Patch(color="#55A868", label=f"Used ({len(used_names)})")
    red_patch   = mpatches.Patch(color="#C44E52", label=f"Dropped ({len(dropped_names)})")
    ax.legend(handles=[green_patch, red_patch], fontsize=8)
    plt.tight_layout()
    path = out / f"factor_valid_ratio{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV
    vrows = [[names_sorted[i], f"{ratios_sorted[i]:.4f}",
              "used" if colors_sorted[i] == "#55A868" else "dropped"]
             for i in range(n)]
    _write_csv(out / f"factor_valid_ratio{tag}.csv", vrows, ["factor_name", "nan_ratio", "status"])
    logger.info(f"[Viz] Factor valid ratio saved: {path}")


def plot_factor_distribution(
    X: np.ndarray,
    used_names: list,
    viz_dir: str,
    group_name: str = "",
    max_factors: int = 20,
):
    """Histogram grid of randomly sampled factor value distributions."""
    if not _MPL_AVAILABLE:
        return
    tag = f"_{group_name}" if group_name else ""
    out = _ensure_dir(Path(viz_dir) / "02_factors")
    X = np.asarray(X).astype(np.float32)
    K = X.shape[1]
    if K == 0:
        return
    n_show = min(K, max_factors)
    rng = np.random.default_rng(42)
    indices = sorted(rng.choice(K, n_show, replace=False))

    cols = min(5, n_show)
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.2 * rows), squeeze=False)
    for ax_idx, ki in enumerate(indices):
        ax = axes.flat[ax_idx]
        vals = X[:, ki]
        vals_finite = vals[np.isfinite(vals)]
        if len(vals_finite) > 0:
            ax.hist(vals_finite, bins=50, color="#4C72B0", alpha=0.8, edgecolor="white",
                    linewidth=0.2)
            ax.axvline(np.nanmean(vals), color="#C44E52", linestyle="--", linewidth=1)
        name = used_names[ki] if ki < len(used_names) else f"col_{ki}"
        # Truncate long names
        if len(name) > 30:
            name = name[:27] + "..."
        ax.set_title(name, fontsize=7)
        ax.tick_params(labelsize=6)
    # Hide unused axes
    for ax_idx in range(n_show, rows * cols):
        axes.flat[ax_idx].set_visible(False)
    plt.suptitle(f"Factor Value Distributions{tag} (red=mean)", fontsize=11)
    plt.tight_layout()
    path = out / f"factor_distribution{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV: summary stats per sampled factor
    drows = []
    for ki in indices:
        vals = X[:, ki]
        vals_finite = vals[np.isfinite(vals)]
        name = used_names[ki] if ki < len(used_names) else f"col_{ki}"
        if len(vals_finite) > 0:
            drows.append([name,
                          f"{np.nanmean(vals):.6f}", f"{np.nanstd(vals):.6f}",
                          f"{np.nanmin(vals):.6f}", f"{np.percentile(vals_finite, 25):.6f}",
                          f"{np.percentile(vals_finite, 50):.6f}", f"{np.percentile(vals_finite, 75):.6f}",
                          f"{np.nanmax(vals):.6f}", f"{np.isnan(vals).mean():.4f}"])
        else:
            drows.append([name, "", "", "", "", "", "", "", "1.0000"])
    _write_csv(out / f"factor_distribution{tag}.csv", drows,
               ["factor_name", "mean", "std", "min", "p25", "p50", "p75", "max", "nan_ratio"])
    logger.info(f"[Viz] Factor distribution saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 3) Standardization diagnostics
# ═══════════════════════════════════════════════════════════════

def plot_standardize_effect(
    pre_mean: np.ndarray,
    pre_std: np.ndarray,
    post_mean: np.ndarray,
    post_std: np.ndarray,
    fill_values: np.ndarray,
    viz_dir: str,
    split_name: str = "",
):
    """Scatter: per-feature mean/std before vs after standardization + fill value histogram."""
    if not _MPL_AVAILABLE:
        return
    tag = f"_{split_name}" if split_name else ""
    out = _ensure_dir(Path(viz_dir) / "03_standardize")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Mean before vs after
    ax = axes[0]
    ax.scatter(pre_mean, post_mean, s=8, alpha=0.5, color="#4C72B0", edgecolors="none")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean (before standardization)")
    ax.set_ylabel("Mean (after standardization)")
    ax.set_title("Feature Mean: Before vs After")

    # Std before vs after
    ax = axes[1]
    ax.scatter(pre_std, post_std, s=8, alpha=0.5, color="#DD8452", edgecolors="none")
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Std (before standardization)")
    ax.set_ylabel("Std (after standardization)")
    ax.set_title("Feature Std: Before vs After")

    # Fill value distribution
    ax = axes[2]
    fv = np.asarray(fill_values).ravel()
    fv = fv[np.isfinite(fv)]
    ax.hist(fv, bins=50, color="#55A868", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Fill value (mean of column)")
    ax.set_title("NaN Fill Value Distribution")

    plt.suptitle(f"Standardization Effect{tag}", fontsize=12)
    plt.tight_layout()
    path = out / f"standardize_effect{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV
    n_feats = len(pre_mean)
    fv_flat = np.asarray(fill_values).ravel()
    srows = [[str(i),
              f"{pre_mean[i]:.6f}", f"{pre_std[i]:.6f}",
              f"{post_mean[i]:.6f}", f"{post_std[i]:.6f}",
              f"{fv_flat[i]:.6f}" if i < len(fv_flat) else ""]
             for i in range(n_feats)]
    _write_csv(out / f"standardize_effect{tag}.csv", srows,
               ["feature_index", "pre_mean", "pre_std", "post_mean", "post_std", "fill_value"])
    logger.info(f"[Viz] Standardization effect saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 4) Group-level LGBM diagnostics
# ═══════════════════════════════════════════════════════════════

def plot_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    used_names: list,
    lgbm_params: dict,
    class_weight,
    early_stopping_rounds: int,
    viz_dir: str,
    group_name: str = "",
    top_k: int = 20,
):
    """Fit a quick LGBM and plot top-k feature importances."""
    if not _MPL_AVAILABLE:
        return
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("[Viz] lightgbm not available, skipping feature importance.")
        return
    tag = f"_{group_name}" if group_name else ""
    out = _ensure_dir(Path(viz_dir) / "04_group_lgbm")

    y_tr = np.asarray(y_train).astype(int)
    y_vl = np.asarray(y_val).astype(int)
    n_classes = len(np.unique(np.concatenate([y_tr, y_vl])))
    n_features = X_train.shape[1]
    top_k = min(top_k, n_features)

    safe_params = dict(lgbm_params)
    min_leaf = max(1, len(y_tr) // (n_classes * 20))
    safe_params.setdefault("min_data_in_leaf", min_leaf)
    safe_params["min_data_in_leaf"] = min(safe_params["min_data_in_leaf"], min_leaf)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        class_weight=class_weight if class_weight else None,
        verbose=-1,
        **safe_params,
    )
    fit_kw = {}
    if early_stopping_rounds > 0:
        fit_kw["eval_set"] = [(X_val, y_vl)]
        fit_kw["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    model.fit(X_train, y_tr, **fit_kw)

    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top_k]
    top_imp = imp[idx]
    top_names = [used_names[i] if i < len(used_names) else f"f{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(max(6, top_k * 0.25), 5))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, top_k))
    ax.barh(range(top_k), top_imp[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([top_names[i] for i in range(top_k)][::-1], fontsize=8)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"Top-{top_k} Feature Importance{tag}")
    plt.tight_layout()
    path = out / f"feature_importance{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV: all feature importances sorted
    all_idx = np.argsort(imp)[::-1]
    irows = [[used_names[i] if i < len(used_names) else f"f{i}", f"{imp[i]:.6f}"] for i in all_idx]
    _write_csv(out / f"feature_importance{tag}.csv", irows, ["factor_name", "importance"])
    import gc
    del model; gc.collect()
    logger.info(f"[Viz] Feature importance saved: {path}")


def plot_proba_distribution(
    proba: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    viz_dir: str,
    group_name: str = "",
):
    """Violin / box plot of predicted probability per true class."""
    if not _MPL_AVAILABLE:
        return
    tag = f"_{group_name}" if group_name else ""
    out = _ensure_dir(Path(viz_dir) / "04_group_lgbm")

    proba = np.asarray(proba).astype(np.float32)
    labels = np.asarray(labels).astype(int)
    n_classes = len(class_names)
    present = sorted(set(int(v) for v in labels))

    fig, axes = plt.subplots(1, min(2, len(present)), figsize=(7 * min(2, len(present)), 5),
                              squeeze=False)
    for ax_i, (target_c, ax) in enumerate(zip(present[:2], axes.flat)):
        mask = labels == target_c
        data = [proba[mask, c] for c in range(n_classes)]
        parts = ax.violinplot(data, positions=range(n_classes), showmeans=True,
                               showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
        ax.set_xticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Predicted probability")
        ax.set_title(f"Probability Distribution when True={class_names[target_c]}")

    plt.suptitle(f"Group LGBM Probability Distribution{tag}", fontsize=11)
    plt.tight_layout()
    path = out / f"proba_distribution{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV: per-class mean/std of predicted probas for each true class
    prows = []
    for target_c in present:
        mask = labels == target_c
        tc_name = class_names[target_c] if target_c < len(class_names) else f"class_{target_c}"
        for c in range(n_classes):
            pc_name = class_names[c] if c < len(class_names) else f"class_{c}"
            pvals = proba[mask, c]
            prows.append([tc_name, pc_name,
                          f"{np.mean(pvals):.6f}", f"{np.std(pvals):.6f}",
                          f"{np.min(pvals):.6f}", f"{np.max(pvals):.6f}"])
    _write_csv(out / f"proba_distribution{tag}.csv", prows,
               ["true_class", "predicted_class", "mean_proba", "std_proba", "min_proba", "max_proba"])
    logger.info(f"[Viz] Probability distribution saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 5) Meta feature correlation
# ═══════════════════════════════════════════════════════════════

def plot_meta_correlation(
    meta_X: np.ndarray,
    group_names: list,
    n_classes: int,
    viz_dir: str,
):
    """Heatmap of correlations among meta features (group probas)."""
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "05_meta")
    X = np.asarray(meta_X).astype(np.float32)
    # Sample if too large
    if X.shape[0] > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], 10000, replace=False)
        X = X[idx]

    # Build short labels: group_name + class_i
    n_groups = len(group_names)
    labels = []
    for gname in group_names:
        for c in range(n_classes):
            labels.append(f"{gname[:4]}_c{c}")

    if X.shape[1] != len(labels):
        labels = [f"f{i}" for i in range(X.shape[1])]

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    # Truncate long labels for display
    display_labels = [l if len(l) <= 10 else l[:9] + "." for l in labels]

    fig_size = min(24, max(6, X.shape[1] * 0.25))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr, interpolation="nearest", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=90, fontsize=5)
    ax.set_yticklabels(display_labels, fontsize=5)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Meta Feature Correlation (group probas)", fontsize=10)
    plt.tight_layout()
    path = out / "meta_feature_correlation.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV: full correlation matrix with labels
    mrows = [[labels[i]] + [f"{corr[i, j]:.6f}" for j in range(len(labels))] for i in range(len(labels))]
    _write_csv(out / "meta_feature_correlation.csv", mrows, ["feature"] + labels)
    logger.info(f"[Viz] Meta feature correlation saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 6) Temporal feature diagnostics
# ═══════════════════════════════════════════════════════════════

def plot_temporal_feature_breakdown(
    n_classes: int,
    multi_scale_windows: list,
    enhanced_features: bool,
    autocorr_features: bool,
    cross_scale_features: bool,
    distribution_shape: bool,
    window_size: int,
    viz_dir: str,
):
    """Pie/bar chart showing temporal feature dimension composition."""
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "06_temporal")
    use_ms = bool(multi_scale_windows)
    ws_list = multi_scale_windows if use_ms else [window_size]
    n_scales = len(ws_list)

    dims = {}
    # Base features (per window scale): n_classes * 6 + 3
    base_dim = (n_classes * 6 + 3) * n_scales
    dims[f"Base ({n_scales} scale{'s' if n_scales > 1 else ''})"] = base_dim
    total = base_dim

    if enhanced_features:
        enh_dim = (n_classes * 5 + 2)  # diff1+diff2+ema*3+margin+gini
        dims["Enhanced (diff/EMA)"] = enh_dim
        total += enh_dim
    if autocorr_features:
        ac_dim = n_classes * 3  # lag 1/2/3
        dims["Autocorr (lag1-3)"] = ac_dim
        total += ac_dim
    if distribution_shape:
        ds_dim = n_classes * 2  # skew + kurt
        dims["Dist Shape (skew/kurt)"] = ds_dim
        total += ds_dim
    if cross_scale_features and use_ms:
        cs_dim = n_classes * 3  # trend_div + vol_ratio + peak_diff
        dims["Cross-Scale"] = cs_dim
        total += cs_dim

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(dims)))
    wedges, texts, autotexts = ax1.pie(
        dims.values(), labels=None, autopct="%1.1f%%",
        colors=colors, startangle=90,
    )
    ax1.set_title(f"Temporal Feature Composition\n(total={total} dims)")

    ax2.barh(list(dims.keys()), list(dims.values()), color=colors, edgecolor="white")
    ax2.set_xlabel("Dimension count")
    for i, (k, v) in enumerate(dims.items()):
        ax2.text(v + total * 0.02, i, str(v), va="center", fontsize=9)

    ax1.legend(wedges, dims.keys(), loc="lower center", fontsize=7,
               bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    plt.suptitle("Temporal Feature Dimension Breakdown", fontsize=11)
    plt.tight_layout()
    path = out / "temporal_feature_breakdown.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV
    trows = [[k, str(v), f"{v / total * 100:.1f}"] for k, v in dims.items()]
    _write_csv(out / "temporal_feature_breakdown.csv", trows, ["component", "dimension_count", "percentage"])
    logger.info(f"[Viz] Temporal feature breakdown saved: {path}")


def plot_proba_smoothing_effect(
    proba_raw: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    window_size: int,
    viz_dir: str,
    n_frames: int = 500,
):
    """Show raw proba vs smoothed (EMA) probability for one class over time."""
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "06_temporal")
    proba = np.asarray(proba_raw).astype(np.float32)
    labels = np.asarray(labels).astype(int)
    T, C = proba.shape

    # pick a class that appears frequently
    class_counts = [(labels == c).sum() for c in range(C)]
    best_c = int(np.argmax(class_counts))

    n_frames = min(n_frames, T)
    rng = np.random.default_rng(42)
    start = int(rng.integers(0, max(1, T - n_frames)))
    end = start + n_frames

    seg_proba = proba[start:end, best_c]
    seg_label = labels[start:end]
    seg_label_bin = (seg_label == best_c).astype(float)

    # EMA smoothing
    halflife = 10
    alpha = np.exp(np.log(0.5) / halflife)
    ema = np.zeros_like(seg_proba)
    ema[0] = seg_proba[0]
    for t in range(1, len(seg_proba)):
        ema[t] = alpha * ema[t - 1] + (1 - alpha) * seg_proba[t]

    # Moving average
    half = window_size // 2
    padded = np.concatenate([
        np.full(half, seg_proba[0]), seg_proba, np.full(half, seg_proba[-1])
    ])
    ma = np.convolve(padded, np.ones(window_size) / window_size, mode="same")[half:half + len(seg_proba)]

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(seg_proba))
    ax.plot(x, seg_proba, alpha=0.3, linewidth=0.5, color="gray", label="Raw proba")
    ax.plot(x, ma, linewidth=1.5, color="#4C72B0", label=f"MA (w={window_size})")
    ax.plot(x, ema, linewidth=1.5, color="#DD8452", label=f"EMA (hl={halflife})")
    # Ground truth regions
    for c_val, color in [(1, "#55A868"), (0, "#C44E52")]:
        mask = seg_label_bin == c_val
        if mask.any():
            ax.fill_between(x, 0, 1, where=mask, alpha=0.08, color=color,
                           label=f"True={class_names[best_c]}" if c_val == 1 else f"True≠{class_names[best_c]}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Probability")
    ax.set_title(f"Probability Smoothing Effect — class: {class_names[best_c]} (frames {start}–{end})")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    path = out / "proba_smoothing_effect.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV
    srows = []
    for fi in range(len(seg_proba)):
        srows.append([str(int(start + fi)),
                      f"{seg_proba[fi]:.6f}", f"{ma[fi]:.6f}", f"{ema[fi]:.6f}",
                      str(int(seg_label_bin[fi]))])
    _write_csv(out / "proba_smoothing_effect.csv", srows,
               ["frame", "raw_proba", f"MA_w{window_size}", "EMA_hl10", "ground_truth"])
    logger.info(f"[Viz] Probability smoothing effect saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 7) Decoding diagnostics
# ═══════════════════════════════════════════════════════════════

def plot_transition_matrix(
    log_trans: np.ndarray,
    class_names: list,
    viz_dir: str,
):
    """Heatmap of transition probability matrix (exp of log_trans)."""
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "07_decoding")
    trans = np.exp(np.asarray(log_trans))
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))
    im = ax.imshow(trans, interpolation="nearest", cmap="YlOrRd")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title("Transition Probability Matrix")
    thresh = trans.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{trans[i, j]:.3f}", ha="center", va="center",
                    color="white" if trans[i, j] > thresh else "black", fontsize=7)
    plt.tight_layout()
    path = out / "transition_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV
    trows = []
    for i in range(n):
        for j in range(n):
            trows.append([class_names[i], class_names[j], f"{trans[i, j]:.6f}"])
    _write_csv(out / "transition_matrix.csv", trows, ["from_class", "to_class", "probability"])
    logger.info(f"[Viz] Transition matrix saved: {path}")


def plot_viterbi_comparison(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred_viterbi: np.ndarray,
    class_names: list,
    viz_dir: str,
    n_frames: int = 500,
    seed: int = None,
):
    """Three-row raster: Ground Truth / LGBM raw / Viterbi decoded."""
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "07_decoding")
    y_true = np.asarray(y_true).astype(int)
    y_raw = np.asarray(y_pred_raw).astype(int)
    y_vit = np.asarray(y_pred_viterbi).astype(int)
    T = len(y_true)
    n_frames = min(n_frames, T)
    n_classes = len(class_names)

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, T - n_frames)))
    end = start + n_frames

    segs = [y_true[start:end], y_raw[start:end], y_vit[start:end]]
    titles = ["Ground Truth", "LGBM Raw", "Viterbi Decoded"]

    cmap = plt.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)
    class_to_color = {i: cmap(i) for i in range(n_classes)}

    fig, axes = plt.subplots(3, 1, figsize=(min(20, n_frames / 30), 4), sharex=True)
    fig.subplots_adjust(hspace=0.1)

    for ax, seg, title in zip(axes, segs, titles):
        color_img = np.array([[class_to_color[int(c)][:3] for c in seg]])
        ax.imshow(color_img, aspect="auto", interpolation="nearest",
                  extent=[0, n_frames, 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=9, rotation=0, labelpad=70, va="center")
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[-1].set_xlabel(f"Frame offset: {start}–{end}", fontsize=8)
    patches = [mpatches.Patch(color=class_to_color[i], label=class_names[i])
               for i in range(n_classes)]
    fig.legend(handles=patches, loc="lower center", ncol=min(n_classes, 6),
               fontsize=7, bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.suptitle("Viterbi Decoding Comparison", fontsize=10, y=1.02)
    plt.tight_layout()
    path = out / "viterbi_comparison_raster.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    # CSV
    vrows = []
    for fi in range(len(segs[0])):
        vrows.append([str(int(start + fi)),
                      class_names[int(segs[0][fi])] if int(segs[0][fi]) < n_classes else f"class_{int(segs[0][fi])}",
                      class_names[int(segs[1][fi])] if int(segs[1][fi]) < n_classes else f"class_{int(segs[1][fi])}",
                      class_names[int(segs[2][fi])] if int(segs[2][fi]) < n_classes else f"class_{int(segs[2][fi])}"])
    _write_csv(out / "viterbi_comparison_raster.csv", vrows,
               ["frame", "ground_truth", "lgbm_raw", "viterbi_decoded"])
    logger.info(f"[Viz] Viterbi comparison raster saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 8) Pipeline summary
# ═══════════════════════════════════════════════════════════════

def plot_pipeline_comparison(
    group_metrics: dict,
    meta_metrics: dict,
    temporal_metrics: dict,
    seq_decoding: dict,
    monitor_metric_key: str = "balanced_accuracy",
    viz_dir: str = "",
):
    """Grouped bar chart of accuracy / macro_f1 / weighted_f1 across all stages."""
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "08_summary")
    _key = monitor_metric_key

    # Build stage data
    stages = []
    vals_acc, vals_f1, vals_wf1, vals_key = [], [], [], []
    stage_labels = []

    for gname, gm in group_metrics.items():
        stages.append(("group", gname, gm))
    if meta_metrics and "accuracy" in meta_metrics:
        stages.append(("meta", "meta", meta_metrics))
    if temporal_metrics and "accuracy" in temporal_metrics:
        tm_name = temporal_metrics.get("model", "lgbm")
        stages.append(("temporal", f"temporal_{tm_name}", temporal_metrics))
    for dec_key, dm in seq_decoding.items():
        if dm and "accuracy" in dm:
            stages.append(("decoding", dec_key, dm))

    if len(stages) <= 1:
        return

    stage_labels = [s[1] for s in stages]
    vals_acc = [s[2].get("accuracy", 0) for s in stages]
    vals_f1  = [s[2].get("macro_f1", 0) for s in stages]
    vals_wf1 = [s[2].get("weighted_f1", 0) for s in stages]
    vals_key = [s[2].get(_key, 0) for s in stages]

    n = len(stages)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), 5))
    ax.bar(x - w, vals_acc, w, label="Accuracy", color="#4C72B0", edgecolor="white")
    ax.bar(x,      vals_f1,  w, label="Macro F1", color="#55A868", edgecolor="white")
    ax.bar(x + w,  vals_wf1, w, label="Weighted F1", color="#DD8452", edgecolor="white")

    for i in range(n):
        ax.text(i - w, vals_acc[i] + 0.01, f"{vals_acc[i]:.3f}", ha="center", fontsize=7)
        ax.text(i,      vals_f1[i]  + 0.01, f"{vals_f1[i]:.3f}",  ha="center", fontsize=7)
        ax.text(i + w,  vals_wf1[i] + 0.01, f"{vals_wf1[i]:.3f}", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Pipeline Stage Comparison")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = out / "pipeline_stage_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV
    prows = [[stage_labels[i], f"{vals_acc[i]:.4f}", f"{vals_f1[i]:.4f}",
              f"{vals_wf1[i]:.4f}", f"{vals_key[i]:.4f}"] for i in range(n)]
    _write_csv(out / "pipeline_stage_comparison.csv", prows,
               ["stage", "accuracy", "macro_f1", "weighted_f1", _key])
    logger.info(f"[Viz] Pipeline stage comparison saved: {path}")


def plot_per_class_f1_evolution(
    group_metrics: dict,
    meta_metrics: dict,
    temporal_metrics: dict,
    seq_decoding: dict,
    class_names: list,
    viz_dir: str,
):
    """Line plot: per-class F1 across pipeline stages."""
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "08_summary")

    # Collect per_class metrics from each stage
    stage_data = []
    stage_labels = []

    # From group metrics: take the best group per class, or use combined
    # For simplicity, use first group or aggregate
    for gname, gm in group_metrics.items():
        # Use the group with most factors as representative
        pass  # We'll use meta_metrics or temporal_metrics instead

    # Build ordered list of stages that have per_class
    if meta_metrics and "per_class" in meta_metrics:
        stage_data.append(meta_metrics["per_class"])
        stage_labels.append("Meta")
    elif group_metrics:
        # fall back to first group
        first_gm = list(group_metrics.values())[0]
        # group metrics dont have per_class in the same format...
        # Actually they don't, group_metrics has group-level metrics only
        pass

    # Actually, group_metrics don't have per_class. Let me use:
    # temporal then viterbi then crf
    if temporal_metrics and "per_class" in temporal_metrics:
        tm_name = temporal_metrics.get("model", "lgbm")
        stage_data.append(temporal_metrics["per_class"])
        stage_labels.append(f"Temporal ({tm_name})")

    for dec_name, dm in seq_decoding.items():
        if dm and "per_class" in dm:
            stage_data.append(dm["per_class"])
            stage_labels.append(dec_name)

    if len(stage_data) <= 1:
        return

    n_stages = len(stage_data)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.6), 5))
    colors = plt.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)

    for c in range(n_classes):
        cname = class_names[c]
        f1_vals = []
        for sd in stage_data:
            pc = sd.get(cname, {})
            f1_vals.append(pc.get("f1", 0) if pc else 0)
        ax.plot(range(n_stages), f1_vals, marker="o", color=colors(c),
                label=cname, linewidth=1.5, markersize=4)

    ax.set_xticks(range(n_stages))
    ax.set_xticklabels(stage_labels, fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Evolution Across Pipeline Stages")
    if n_classes <= 12:
        ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    path = out / "per_class_f1_evolution.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV
    evo_rows = []
    for si, sd in enumerate(stage_data):
        stage_name = stage_labels[si]
        for c in range(n_classes):
            cname = class_names[c] if c < n_classes else f"class_{c}"
            pc = sd.get(cname, {})
            f1_val = pc.get("f1", 0) if pc else 0
            evo_rows.append([cname, stage_name, f"{f1_val:.4f}" if f1_val else ""])
    _write_csv(out / "per_class_f1_evolution.csv", evo_rows, ["class", "stage", "f1"])
    logger.info(f"[Viz] Per-class F1 evolution saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 9) Best model detailed metrics table
# ═══════════════════════════════════════════════════════════════

def _draw_metric_table(
    ax,
    cell_text: list,
    col_labels: list,
    title: str,
    col_widths: list = None,
    header_color: str = "#3A7CA5",
    row_colors: list = None,
    value_cols: list = None,
    fontsize: int = 9,
):
    """Draw a styled matplotlib table on the given axis. Returns the table object."""
    n_rows = len(cell_text)
    n_cols = len(col_labels)

    if row_colors is None:
        row_colors = [["#F5F5F5", "#FFFFFF"][i % 2] for i in range(n_rows)]

    if value_cols is None:
        value_cols = list(range(n_cols))

    # Build cellColours with value-based coloring for metric columns
    cell_colours = []
    for i in range(n_rows):
        row_c = []
        for j in range(n_cols):
            if j == 0:
                row_c.append(row_colors[i])
            elif j in value_cols:
                val = cell_text[i][j]
                if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                    intensity = min(max(float(val), 0.0), 1.0)
                    r, g, b = 0.95 * (1 - intensity), 0.85 + 0.1 * intensity, 1 - intensity * 0.6
                    row_c.append((r, g, b, 0.35))
                else:
                    row_c.append(row_colors[i])
            else:
                row_c.append(row_colors[i])
        cell_colours.append(row_c)

    # Build formatted cell text (stringified)
    cell_text_str = []
    for i, row in enumerate(cell_text):
        str_row = []
        for j, val in enumerate(row):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                str_row.append("  —  ")
            elif isinstance(val, float):
                str_row.append(f"{val:.4f}")
            elif isinstance(val, int) and j > 0:
                str_row.append(f"{val:,}")
            else:
                str_row.append(str(val))
        cell_text_str.append(str_row)

    table = ax.table(
        cellText=cell_text_str,
        colLabels=col_labels,
        cellColours=cell_colours,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.35)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold", fontsize=fontsize)

    # Style row labels
    for i in range(n_rows):
        cell = table[i + 1, 0]
        cell.set_text_props(fontweight="bold", fontsize=fontsize)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")


def plot_best_model_metrics_table(
    metrics: dict,
    model_name: str,
    viz_dir: str,
):
    """Render the best model's aggregate + per-class metrics as a detailed table PNG."""
    if not _MPL_AVAILABLE or not metrics:
        return
    out = _ensure_dir(Path(viz_dir) / "09_best_model")

    per_class = metrics.get("per_class", {})
    if not per_class:
        return

    n_classes = len(per_class)
    class_names = list(per_class.keys())

    # Build per-class table rows
    rows = []
    for cname in class_names:
        pc = per_class[cname]
        rows.append([
            cname,
            pc.get("precision"),
            pc.get("recall"),
            pc.get("f1"),
            pc.get("auc"),
            pc.get("support", 0),
        ])

    # Add macro/weighted averages row
    macro_prec = np.mean([pc.get("precision", 0) or 0 for pc in per_class.values()])
    macro_rec  = np.mean([pc.get("recall", 0) or 0 for pc in per_class.values()])
    macro_f1v  = np.mean([pc.get("f1", 0) or 0 for pc in per_class.values()])
    aucs = [pc.get("auc") for pc in per_class.values() if pc.get("auc") is not None]
    macro_auc  = np.mean(aucs) if aucs else None
    rows.append([
        "macro_avg",
        round(macro_prec, 4),
        round(macro_rec, 4),
        round(macro_f1v, 4),
        round(macro_auc, 4) if macro_auc is not None else None,
        sum(pc.get("support", 0) for pc in per_class.values()),
    ])

    col_labels = ["Class", "Precision", "Recall", "F1", "AUC", "Support"]
    value_cols = [1, 2, 3, 4]

    # Determine figure size
    n_rows_total = len(rows)
    fig_height = max(5.5, n_rows_total * 0.45 + 3.0)
    fig_width = max(10, len(col_labels) * 1.8)

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)
    _draw_metric_table(
        ax=ax,
        cell_text=rows,
        col_labels=col_labels,
        title=f"Best Model: {model_name} — Per-Class Metrics",
        value_cols=value_cols,
        fontsize=9,
    )

    # ---- Aggregate metrics text box at top ----
    agg_lines = []
    acc  = metrics.get("accuracy")
    bacc = metrics.get("balanced_accuracy")
    mauc = metrics.get("macro_auc")
    wauc = metrics.get("weighted_auc")
    mf1  = metrics.get("macro_f1")
    wf1  = metrics.get("weighted_f1")
    topk = metrics.get("top_k_accuracy")
    topk_k = metrics.get("top_k")

    if acc is not None:
        agg_lines.append(f"Accuracy: {acc:.4f}")
    if bacc is not None:
        agg_lines.append(f"Balanced Acc: {bacc:.4f}")
    if mf1 is not None:
        agg_lines.append(f"Macro F1: {mf1:.4f}")
    if wf1 is not None:
        agg_lines.append(f"Weighted F1: {wf1:.4f}")
    if mauc is not None:
        agg_lines.append(f"Macro AUC: {mauc:.4f}")
    if wauc is not None:
        agg_lines.append(f"Weighted AUC: {wauc:.4f}")
    if topk is not None and topk_k is not None:
        agg_lines.append(f"Top-{topk_k} Acc: {topk:.4f}")

    agg_text = " | ".join(agg_lines)
    fig.text(0.5, 0.96, agg_text, ha="center", va="top", fontsize=10,
             fontweight="bold", bbox=dict(boxstyle="round,pad=0.4",
             facecolor="#E8F4FD", edgecolor="#3A7CA5", alpha=0.9))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    safe_name = model_name.replace(" ", "_").replace("/", "_").replace("+", "_")
    path = out / f"best_model_metrics_{safe_name}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV: reuse rows already built
    brows = []
    for row in rows:
        brows.append([str(v) if v is not None else "" for v in row])
    _write_csv(out / f"best_model_metrics_{safe_name}.csv", brows, col_labels)
    logger.info(f"[Viz] Best model metrics table saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 10) OvR binary classification results table
# ═══════════════════════════════════════════════════════════════

def plot_ovr_results_table(
    ovr_results: dict,
    viz_dir: str,
):
    """Render the OvR binary classification results as a detailed table PNG."""
    if not _MPL_AVAILABLE or not ovr_results:
        return
    out = _ensure_dir(Path(viz_dir) / "10_ovr")

    macro_avg = ovr_results.pop("macro_avg", {})
    class_names = list(ovr_results.keys())

    # Build rows
    rows = []
    for cname in class_names:
        v = ovr_results[cname]
        rows.append([
            cname,
            v.get("acc"),
            v.get("f1"),
            v.get("auc"),
            v.get("n_pos_val", 0),
        ])

    # macro_avg row
    rows.append([
        "macro_avg",
        macro_avg.get("acc"),
        macro_avg.get("f1"),
        macro_avg.get("auc"),
        sum(v.get("n_pos_val", 0) for v in ovr_results.values()),
    ])

    # Restore macro_avg
    ovr_results["macro_avg"] = macro_avg

    col_labels = ["Class", "ACC", "F1", "AUC", "Val Pos/Neg"]
    value_cols = [1, 2, 3]

    n_rows_total = len(rows)
    fig_height = max(4.5, n_rows_total * 0.45 + 2.5)
    fig_width = max(9, len(col_labels) * 1.8)

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_subplot(111)

    # Special row colors: highlight macro_avg
    row_colors = []
    for i in range(n_rows_total - 1):
        row_colors.append(["#F5F5F5", "#FFFFFF"][i % 2])
    row_colors.append("#E8F4FD")  # macro_avg highlighted

    _draw_metric_table(
        ax=ax,
        cell_text=rows,
        col_labels=col_labels,
        title="OvR Binary Classification Results (One-vs-Rest, 1:1 undersampled)",
        value_cols=value_cols,
        row_colors=row_colors,
        fontsize=9,
    )

    # Note about undersampling
    fig.text(0.5, 0.02, "Note: train/val both undersampled to 1:1 (positive:negative). "
             "ACC = accuracy on balanced val set.", ha="center", fontsize=8,
             color="gray", fontstyle="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = out / "ovr_results_table.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)
    # CSV
    orows = []
    for row in rows:
        orows.append([str(v) if v is not None else "" for v in row])
    _write_csv(out / "ovr_results_table.csv", orows, col_labels)
    logger.info(f"[Viz] OvR results table saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 11) Factor space UMAP dimensionality reduction visualization
# ═══════════════════════════════════════════════════════════════

def plot_umap_factor_space(
    embedding: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    group_embeddings: "dict | None" = None,
    viz_dir: str = "visualizations",
):
    """UMAP 2D scatter plot, colored by behavior class; optional per-group UMAP subplots.

    Parameters
    ----------
    embedding: [N, 2] UMAP embedding coordinates of the entire factor space
    labels: [N] Behavior labels (int), aligned frame-by-frame with embedding
    class_names: List of readable class names
    group_embeddings: dict {group_name: (embedding [N_g, 2], labels [N_g])} — per-group independent UMAP
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "02_factors_umap")

    n_classes = len(class_names)
    labels = np.asarray(labels).astype(int)
    cmap = plt.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)

    # --- Color by behavior class (full factor space) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cname in enumerate(class_names):
        mask = labels == i
        if mask.any():
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[cmap(i)], label=cname, s=1, alpha=0.6, rasterized=True)
    ax.set_title("Factor Space UMAP — All Groups Combined, by Behavior Class", fontsize=13)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=8, fontsize=8, loc="lower right", ncol=2 if n_classes > 5 else 1)
    plt.tight_layout()
    plt.savefig(out / "umap_by_behavior.png", dpi=300)
    plt.close(fig)
    logger.info(f"[Viz] UMAP behavior class scatter plot saved: {out / 'umap_by_behavior.png'}")

    # --- Per-group UMAP (if provided) ---
    if group_embeddings:
        n_groups = len(group_embeddings)
        gnames = list(group_embeddings.keys())
        fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))
        if n_groups == 1:
            axes = [axes]
        for gi, gname in enumerate(gnames):
            ax = axes[gi]
            g_embed, g_labels = group_embeddings[gname]
            g_labels = np.asarray(g_labels).astype(int)
            n_frames = len(g_embed)
            for i, cname in enumerate(class_names):
                cm = g_labels == i
                if cm.any():
                    ax.scatter(g_embed[cm, 0], g_embed[cm, 1],
                               c=[cmap(i)], s=1, alpha=0.5, rasterized=True)
            ax.set_title(f"{gname} ({n_frames} frames)", fontsize=11)
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        plt.tight_layout()
        plt.savefig(out / "umap_by_group.png", dpi=300)
        plt.close(fig)
        logger.info(f"[Viz] UMAP group scatter plot saved: {out / 'umap_by_group.png'}")


# ═══════════════════════════════════════════════════════════════
# 12) Factor-behavior AUC correlation heatmap
# ═══════════════════════════════════════════════════════════════

def plot_factor_behavior_heatmap(
    auc_matrix: np.ndarray,
    factor_names: list,
    class_ids_sorted: list,
    class_names_readable: list,
    top_k_per_class: int = 20,
    viz_dir: str = "visualizations",
):
    """Factor x behavior AUC heatmap, rows sorted by hierarchical clustering, showing only top-k factors per class.

    Parameters
    ----------
    auc_matrix: [N_factors, N_classes] AUC matrix (may contain NaN)
    factor_names: List of factor names
    class_ids_sorted: List of class IDs
    class_names_readable: List of readable class names
    top_k_per_class: Number of top factors to show per class
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "02_factors")

    # Select top-k factors per class
    selected_indices: Set[int] = set()
    for j in range(len(class_ids_sorted)):
        col = auc_matrix[:, j]
        valid = ~np.isnan(col)
        if valid.any():
            top_idx = np.argsort(col[valid])[::-1][:top_k_per_class]
            for idx in np.where(valid)[0][top_idx]:
                selected_indices.add(int(idx))

    if len(selected_indices) < 2:
        logger.warning("[Viz] Factor-behavior heatmap: insufficient valid factors, skipping.")
        return

    idx_list = sorted(selected_indices)
    sub_matrix = auc_matrix[idx_list]
    sub_names = [factor_names[i] for i in idx_list]

    # Hierarchical clustering to order rows
    from scipy.cluster.hierarchy import linkage, leaves_list
    fill_mat = np.where(np.isfinite(sub_matrix), sub_matrix, 0.0)
    if len(idx_list) >= 2:
        order = leaves_list(linkage(fill_mat, method="average"))
        sub_matrix = sub_matrix[order]
        sub_names = [sub_names[i] for i in order]

    # Plot
    n_fac = len(idx_list)
    n_cls = len(class_ids_sorted)
    fig_h = max(6, n_fac * 0.22)
    fig_w = max(10, n_cls * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(sub_matrix, aspect="auto", cmap="YlOrRd", vmin=0.45, vmax=1.0)
    ax.set_xticks(range(n_cls))
    ax.set_xticklabels(class_names_readable, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_fac))
    ax.set_yticklabels(sub_names, fontsize=5)
    ax.set_xlabel("Behavior Class")
    ax.set_ylabel("Factor")
    ax.set_title(f"Factor-Behavior AUC Heatmap (top-{top_k_per_class} per class, {n_fac} factors)", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("AUC", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "factor_behavior_auc_heatmap.png", dpi=300)
    plt.close(fig)
    logger.info(f"[Viz] Factor-behavior heatmap saved: {out / 'factor_behavior_auc_heatmap.png'}")


# ═══════════════════════════════════════════════════════════════
# 13) Feature utilization matrix heatmap
# ═══════════════════════════════════════════════════════════════

def plot_feature_utilization(
    features: list,
    aggregators: list,
    matrix: np.ndarray,
    top_n_bar: int = 30,
    viz_dir: str = "visualizations",
):
    """Feature utilization analysis: heatmap (feature x aggregator) + feature frequency bar chart.

    Parameters
    ----------
    features: List of feature names
    aggregators: List of aggregator names
    matrix: [F, A] Usage frequency matrix
    top_n_bar: Number of top features to show in bar chart
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "02_factors")

    # -- Heatmap --
    fig_w = max(8, len(aggregators) * 1.2)
    fig_h = max(8, len(features) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(aggregators)))
    ax.set_xticklabels(aggregators, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=6)
    ax.set_xlabel("Aggregator"); ax.set_ylabel("Feature")
    ax.set_title(f"Feature × Aggregator Utilization ({len(features)} features, {len(aggregators)} aggs)", fontsize=12)
    for i in range(len(features)):
        for j in range(len(aggregators)):
            v = matrix[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=5)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Usage count", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "feature_utilization_heatmap.png", dpi=300)
    plt.close(fig)
    logger.info(f"[Viz] Feature utilization heatmap saved: {out / 'feature_utilization_heatmap.png'}")

    # -- Bar chart: top feature usage --
    feat_usage = matrix.sum(axis=1)
    order = np.argsort(feat_usage)[::-1][:top_n_bar]
    top_feats = [features[i] for i in order]
    top_counts = [feat_usage[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n_bar * 0.3)))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(top_feats)))
    ax.barh(range(len(top_feats)), top_counts, color=colors, edgecolor="white")
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of factors using this feature")
    ax.set_title(f"Top-{top_n_bar} Most Utilized Features in Discovered Factors", fontsize=12)
    for i, v in enumerate(top_counts):
        ax.text(v + max(top_counts) * 0.01, i, str(v), va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "feature_usage_ranking.png", dpi=300)
    plt.close(fig)
    logger.info(f"[Viz] Feature usage ranking saved: {out / 'feature_usage_ranking.png'}")


# ═══════════════════════════════════════════════════════════════
# 14) Factor metadata overview panel
# ═══════════════════════════════════════════════════════════════

def plot_factor_overview(
    meta: dict,
    viz_dir: str = "visualizations",
):
    """Factor metadata overview — 2x2 panel: target distribution, seq_length distribution, AUC distribution, complexity.

    Parameters
    ----------
    meta: dict returned by extract_factor_metadata()
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "02_factors")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Factors per target behavior
    ax = axes[0, 0]
    targets = meta.get("target_names", [])
    counts = [meta["target_counts"].get(t, 0) for t in targets]
    colors_a = plt.get_cmap("tab20")(np.linspace(0, 1, len(targets))) if targets else []
    bars = ax.bar(range(len(targets)), counts, color=colors_a, edgecolor="white")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of Factors")
    ax.set_title("(a) Factors per Target Behavior", fontsize=11)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(counts) * 0.02, str(v),
                ha="center", fontsize=8)

    # (b) seq_length distribution
    ax = axes[0, 1]
    sl_dist = meta.get("seq_length_distribution", {})
    sl_keys = sorted(sl_dist.keys(), key=lambda x: int(x))
    sl_vals = [sl_dist[k] for k in sl_keys]
    colors_b = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"][:len(sl_keys)]
    ax.bar(range(len(sl_keys)), sl_vals, color=colors_b, edgecolor="white")
    ax.set_xticks(range(len(sl_keys)))
    ax.set_xticklabels([f"seq={k}" for k in sl_keys], fontsize=9)
    ax.set_ylabel("Number of Factors")
    ax.set_title("(b) Sequence Length Distribution", fontsize=11)
    for i, v in enumerate(sl_vals):
        ax.text(i, v + max(sl_vals) * 0.02, str(v), ha="center", fontsize=8)

    # (c) AUC distribution
    ax = axes[1, 0]
    aucs = meta.get("auc_distribution", [])
    if aucs:
        ax.hist(aucs, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(aucs), color="#DD8452", linestyle="--", linewidth=1.5,
                   label=f"Mean={np.mean(aucs):.3f}")
        ax.axvline(np.median(aucs), color="#55A868", linestyle="--", linewidth=1.5,
                   label=f"Median={np.median(aucs):.3f}")
        ax.legend(fontsize=8)
        ax.set_xlabel("Best AUC per Factor")
        ax.set_ylabel("Count")
    else:
        ax.text(0.5, 0.5, "No AUC data available\n(factors not individually validated)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_title("(c) AUC Distribution", fontsize=11)

    # (d) Feature count vs complexity (tree size)
    ax = axes[1, 1]
    feat_counts = meta.get("features_per_factor", [])
    tree_sizes = meta.get("tree_sizes", [])
    if feat_counts and tree_sizes and len(feat_counts) == len(tree_sizes):
        ax.scatter(feat_counts, tree_sizes, s=3, alpha=0.4, c="#4C72B0", rasterized=True)
        ax.set_xlabel("Number of Features per Factor")
        ax.set_ylabel("Tree Size (DEAP GP)")
        ax.set_title("(d) Factor Complexity: Features vs Tree Size", fontsize=11)
    elif feat_counts:
        ax.hist(feat_counts, bins=min(30, max(feat_counts) + 1), color="#4C72B0", edgecolor="white")
        ax.set_xlabel("Number of Features per Factor")
        ax.set_ylabel("Count")
        ax.set_title("(d) Features per Factor Distribution", fontsize=11)
    else:
        ax.text(0.5, 0.5, "No complexity data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("(d) Factor Complexity", fontsize=11)

    plt.tight_layout()
    plt.savefig(out / "factor_overview.png", dpi=300)
    plt.close(fig)
    logger.info(f"[Viz] Factor overview panel saved: {out / 'factor_overview.png'}")


# ═══════════════════════════════════════════════════════════════
# 15) Per-group confusion matrix
# ═══════════════════════════════════════════════════════════════

def plot_group_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    viz_dir: str,
    group_name: str = "",
):
    """Single-group confusion matrix heatmap (raw count + row-normalized), with CSV.

    Parameters
    ----------
    cm: [n_classes, n_classes] Confusion matrix (raw counts)
    class_names: List of class names
    group_name: Group name identifier (short/medium/long)
    """
    if not _MPL_AVAILABLE:
        return
    tag = f"_{group_name}" if group_name else ""
    out = _ensure_dir(Path(viz_dir) / "04_group_lgbm")
    n = len(class_names)

    cm = np.asarray(cm).astype(np.float64)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    for mat, suffix, fmt in [
        (cm, "raw", "d"),
        (cm_norm, "norm", ".3f"),
    ]:
        fig, ax = plt.subplots(figsize=(max(7, n * 0.9), max(5.5, n * 0.75)))
        im = ax.imshow(mat, interpolation="nearest", cmap="Greens" if suffix == "raw" else "YlOrRd")
        ax.set_title(f"Confusion Matrix{tag} ({'Raw Count' if suffix == 'raw' else 'Row-Normalized'})", fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        thresh = mat.max() / 2.0 if mat.size else 0
        for i in range(n):
            for j in range(n):
                val_str = f"{int(mat[i, j])}" if suffix == "raw" else f"{mat[i, j]:.3f}"
                ax.text(j, i, val_str, ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black", fontsize=8)
        plt.tight_layout()
        fname = f"confusion_matrix{tag}_{suffix}.png"
        plt.savefig(out / fname, dpi=300)
        plt.close(fig)

        # CSV
        csv_name = f"confusion_matrix{tag}_{suffix}.csv"
        rows = [[class_names[i]] + [f"{mat[i, j]:.4f}" if suffix == "norm" else str(int(mat[i, j]))
                                    for j in range(n)] for i in range(n)]
        _write_csv(out / csv_name, rows, ["true_class"] + [f"pred_{c}" for c in class_names])

    logger.info(f"[Viz] Group confusion matrix saved{tag}: {out}")


# ═══════════════════════════════════════════════════════════════
# 16) Per-class metrics comparison bar chart by group
# ═══════════════════════════════════════════════════════════════

def plot_per_class_metrics_comparison(
    group_per_class: dict,
    class_names: list,
    viz_dir: str,
):
    """Grouped bar chart of Precision/Recall/F1 per class per group.

    Parameters
    ----------
    group_per_class: {group_name: {class_name: {"precision":, "recall":, "f1":}}}
    class_names: List of class names
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "04_group_lgbm")
    groups = list(group_per_class.keys())
    if len(groups) <= 1:
        return
    n_groups = len(groups)
    n_classes = len(class_names)

    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    group_colors = {"short": "#4C72B0", "medium": "#DD8452", "long": "#55A868"}
    default_colors = plt.get_cmap("Set2")(np.linspace(0, 1, n_groups))

    fig, axes = plt.subplots(1, 3, figsize=(max(15, n_classes * 1.8), 5.5), squeeze=False)
    w = 0.8 / n_groups
    x = np.arange(n_classes)

    for mi, (metric_key, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[0, mi]
        for gi, gname in enumerate(groups):
            vals = []
            for cname in class_names:
                pc = group_per_class[gname].get(cname, {})
                vals.append(pc.get(metric_key, 0) or 0)
            offset = (gi - (n_groups - 1) / 2) * w
            color = group_colors.get(gname, default_colors[gi])
            ax.bar(x + offset, vals, w, label=gname, color=color, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(metric_label)
        ax.set_title(f"Per-Class {metric_label} by Group", fontsize=11)
        ax.set_ylim(0, 1.05)
        if mi == 0:
            ax.legend(fontsize=8, ncol=n_groups, loc="upper right")

    plt.suptitle("Group-Level Per-Class Metrics Comparison", fontsize=13)
    plt.tight_layout()
    path = out / "per_class_metrics_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # CSV
    rows = []
    for gname in groups:
        for cname in class_names:
            pc = group_per_class[gname].get(cname, {})
            rows.append([gname, cname,
                         f"{pc.get('precision', 0) or 0:.4f}",
                         f"{pc.get('recall', 0) or 0:.4f}",
                         f"{pc.get('f1', 0) or 0:.4f}"])
    _write_csv(out / "per_class_metrics_comparison.csv", rows,
               ["group", "class", "precision", "recall", "f1"])
    logger.info(f"[Viz] Per-class metrics comparison by group saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 17) Prediction confidence vs correctness distribution
# ═══════════════════════════════════════════════════════════════

def plot_confidence_vs_correctness(
    proba: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    viz_dir: str,
    stage_name: str = "",
):
    """Prediction confidence (max proba) histogram: correct vs incorrect, with CSV.

    Parameters
    ----------
    proba: [N, C] Prediction probability matrix
    labels: [N] Ground truth labels (int)
    stage_name: Stage name identifier
    """
    if not _MPL_AVAILABLE:
        return
    tag = f"_{stage_name}" if stage_name else ""
    out = _ensure_dir(Path(viz_dir) / "08_summary")

    proba = np.asarray(proba).astype(np.float32)
    labels = np.asarray(labels).astype(int)
    n_classes = proba.shape[1]

    pred = proba.argmax(axis=1)
    max_proba = proba.max(axis=1)
    correct_mask = pred == labels
    incorrect_mask = ~correct_mask

    conf_correct = max_proba[correct_mask]
    conf_incorrect = max_proba[incorrect_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Dual histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    if len(conf_correct) > 0:
        ax.hist(conf_correct, bins=bins, alpha=0.7, color="#55A868", label=f"Correct (n={len(conf_correct)})",
                edgecolor="white", linewidth=0.3)
    if len(conf_incorrect) > 0:
        ax.hist(conf_incorrect, bins=bins, alpha=0.7, color="#C44E52", label=f"Incorrect (n={len(conf_incorrect)})",
                edgecolor="white", linewidth=0.3)
    ax.axvline(x=1.0 / n_classes, color="gray", linestyle="--", linewidth=0.8, label=f"Chance={1.0/n_classes:.2f}")
    ax.set_xlabel("Max Predicted Probability")
    ax.set_ylabel("Frame Count")
    ax.set_title(f"Prediction Confidence Distribution{tag}")
    ax.legend(fontsize=9)

    # (b) Per-class mean confidence (correct vs incorrect)
    ax = axes[1]
    x = np.arange(n_classes)
    w = 0.35
    per_class_correct_mean = []
    per_class_incorrect_mean = []
    per_class_correct_count = []
    per_class_incorrect_count = []
    for c in range(n_classes):
        cmask = labels == c
        c_correct = (pred == c) & cmask
        c_incorrect = (pred != c) & cmask
        per_class_correct_mean.append(float(max_proba[c_correct].mean()) if c_correct.any() else 0)
        per_class_incorrect_mean.append(float(max_proba[c_incorrect].mean()) if c_incorrect.any() else 0)
        per_class_correct_count.append(int(c_correct.sum()))
        per_class_incorrect_count.append(int(c_incorrect.sum()))

    ax.bar(x - w / 2, per_class_correct_mean, w, color="#55A868", label="When Correct", edgecolor="white")
    ax.bar(x + w / 2, per_class_incorrect_mean, w, color="#C44E52", label="When Incorrect", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean Max Probability")
    ax.set_title("Per-Class Mean Confidence (Correct vs Incorrect)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.suptitle(f"Model Confidence Analysis{tag}", fontsize=12)
    plt.tight_layout()
    path = out / f"confidence_vs_correctness{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    # CSV
    rows = []
    for c in range(n_classes):
        rows.append([class_names[c],
                     f"{per_class_correct_mean[c]:.4f}", str(per_class_correct_count[c]),
                     f"{per_class_incorrect_mean[c]:.4f}", str(per_class_incorrect_count[c])])
    rows.append(["overall",
                 f"{conf_correct.mean():.4f}" if len(conf_correct) > 0 else "0",
                 str(len(conf_correct)),
                 f"{conf_incorrect.mean():.4f}" if len(conf_incorrect) > 0 else "0",
                 str(len(conf_incorrect))])
    _write_csv(out / f"confidence_vs_correctness{tag}.csv", rows,
               ["class", "mean_conf_correct", "n_correct", "mean_conf_incorrect", "n_incorrect"])
    logger.info(f"[Viz] Confidence analysis saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 18) Confusion matrix stage delta heatmap
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix_delta(
    cm_stages: dict,
    class_names: list,
    viz_dir: str,
):
    """Confusion matrix cross-stage comparison: row-normalized CM per stage + delta heatmap between adjacent stages.

    Parameters
    ----------
    cm_stages: {stage_label: cm_array [n_classes, n_classes]}, ordered by stage
    class_names: List of class names
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "08_summary")
    stage_labels = list(cm_stages.keys())
    if len(stage_labels) <= 1:
        return

    n_stages = len(stage_labels)
    n_classes = len(class_names)

    # Row-normalize each CM
    cm_norms = {}
    for label, cm in cm_stages.items():
        cm = np.asarray(cm).astype(np.float64)
        cm_norms[label] = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    # Layout: top row = CMs per stage, bottom row = adjacent stage delta
    n_cols = n_stages
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(4 * n_cols, n_classes * n_cols * 0.8),
                                      max(9, n_classes * 1.2)),
                             squeeze=False)

    for si, label in enumerate(stage_labels):
        # Top row: CM heatmap
        ax = axes[0, si]
        mat = cm_norms[label]
        im = ax.imshow(mat, interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(f"{label}\nCM (row-norm)", fontsize=9)
        ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
        if si == 0:
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
        else:
            ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.set_xlabel("Pred" if si == n_stages // 2 else "")
        ax.set_ylabel("True" if si == 0 else "")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Bottom row: delta heatmap (relative to previous stage)
        ax2 = axes[1, si]
        if si == 0:
            ax2.text(0.5, 0.5, "(baseline)", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=11, color="gray")
            ax2.set_title("—", fontsize=9)
            ax2.axis("off")
        else:
            delta = cm_norms[label] - cm_norms[stage_labels[si - 1]]
            vmax = max(abs(delta).max(), 0.01)
            im2 = ax2.imshow(delta, interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax2.set_title(f"Δ ({label} − {stage_labels[si - 1]})", fontsize=9)
            ax2.set_xticks(range(n_classes)); ax2.set_yticks(range(n_classes))
            if si == 1:
                ax2.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
                ax2.set_yticklabels(class_names, fontsize=7)
            else:
                ax2.set_xticklabels([]); ax2.set_yticklabels([])
            ax2.set_xlabel("Pred" if si == n_stages // 2 else "")
            ax2.set_ylabel("True" if si == 0 else "")
            # Annotate max improvement / degradation
            max_imp = np.unravel_index(np.argmax(delta), delta.shape)
            max_deg = np.unravel_index(np.argmin(delta), delta.shape)
            ax2.annotate(f"↑{delta[max_imp]:.3f}", xy=(max_imp[1], max_imp[0]),
                         fontsize=6, color="blue", ha="center", va="bottom")
            ax2.annotate(f"↓{delta[max_deg]:.3f}", xy=(max_deg[1], max_deg[0]),
                         fontsize=6, color="red", ha="center", va="top")
            plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle("Confusion Matrix Evolution Across Pipeline Stages", fontsize=13, y=1.01)
    plt.tight_layout()
    path = out / "confusion_matrix_delta.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # CSV: all CMs + deltas
    rows = []
    for si, label in enumerate(stage_labels):
        mat = cm_norms[label]
        for i in range(n_classes):
            rows.append([label, "cm_norm", class_names[i]] +
                        [f"{mat[i, j]:.4f}" for j in range(n_classes)])
        if si > 0:
            delta = cm_norms[label] - cm_norms[stage_labels[si - 1]]
            for i in range(n_classes):
                rows.append([f"{label}_delta", "delta", class_names[i]] +
                            [f"{delta[i, j]:.4f}" for j in range(n_classes)])
    _write_csv(out / "confusion_matrix_delta.csv", rows,
               ["stage", "type", "true_class"] + [f"pred_{c}" for c in class_names])
    logger.info(f"[Viz] Confusion matrix stage delta saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 19) Per-class metrics cross-stage heatmap
# ═══════════════════════════════════════════════════════════════

def plot_per_class_metrics_heatmap(
    per_class_by_stage: dict,
    class_names: list,
    viz_dir: str,
):
    """Cross-stage heatmap of Precision/Recall/F1 per class.

    Parameters
    ----------
    per_class_by_stage: {stage_label: {class_name: {"precision":, "recall":, "f1":}}}
    class_names: List of class names
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "08_summary")
    stage_labels = list(per_class_by_stage.keys())
    if len(stage_labels) <= 1:
        return

    n_classes = len(class_names)
    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]

    # Build matrix: [n_classes * 3, n_stages], rows = (class, metric)
    n_rows = n_classes * 3
    heatmap_data = np.zeros((n_rows, len(stage_labels)))
    row_labels = []
    for mi, (mkey, mlabel) in enumerate(zip(metrics, metric_labels)):
        for ci, cname in enumerate(class_names):
            row_idx = mi * n_classes + ci
            for si, slabel in enumerate(stage_labels):
                pc = per_class_by_stage[slabel].get(cname, {})
                heatmap_data[row_idx, si] = pc.get(mkey, 0) or 0
            row_labels.append(f"{cname[:12]} {mlabel[:4]}")

    fig, ax = plt.subplots(figsize=(max(8, len(stage_labels) * 2),
                                    max(10, n_rows * 0.32)))
    im = ax.imshow(heatmap_data, interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7)

    # Separator lines between the three metric groups
    for mi in range(1, 3):
        ax.axhline(y=mi * n_classes - 0.5, color="gray", linestyle="-", linewidth=1.2)

    ax.set_xlabel("Pipeline Stage")
    ax.set_title("Per-Class Metrics Across Pipeline Stages", fontsize=13)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score", fontsize=9)

    # Annotate values (small font)
    for i in range(n_rows):
        for j in range(len(stage_labels)):
            v = heatmap_data[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if v > 0.5 else "black")

    plt.tight_layout()
    path = out / "per_class_metrics_heatmap.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    # CSV
    rows = []
    for si, slabel in enumerate(stage_labels):
        pc = per_class_by_stage[slabel]
        for cname in class_names:
            entry = pc.get(cname, {})
            rows.append([slabel, cname,
                         f"{entry.get('precision', 0) or 0:.4f}",
                         f"{entry.get('recall', 0) or 0:.4f}",
                         f"{entry.get('f1', 0) or 0:.4f}",
                         f"{entry.get('auc') if entry.get('auc') is not None else ''}"])
    _write_csv(out / "per_class_metrics_heatmap.csv", rows,
               ["stage", "class", "precision", "recall", "f1", "auc"])
    logger.info(f"[Viz] Per-class metrics cross-stage heatmap saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 20) LGBM training learning curve
# ═══════════════════════════════════════════════════════════════

def plot_learning_curve(
    evals_result: dict,
    viz_dir: str,
    group_name: str = "",
):
    """LGBM train/val multi_logloss learning curve.

    Parameters
    ----------
    evals_result: LGBM evals_result_ dict, e.g.
        {"training": {"multi_logloss": [...]}, "valid_1": {"multi_logloss": [...]}}
    group_name: Group name identifier
    """
    if not _MPL_AVAILABLE:
        return
    tag = f"_{group_name}" if group_name else ""
    out = _ensure_dir(Path(viz_dir) / "04_group_lgbm")

    fig, ax = plt.subplots(figsize=(8, 5))

    has_data = False
    for ds_name, metrics in evals_result.items():
        for metric_name, values in metrics.items():
            if values:
                has_data = True
                label = f"{ds_name} {metric_name}"
                ax.plot(range(len(values)), values, linewidth=1.5, label=label)

    if not has_data:
        plt.close(fig)
        return

    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Multi-LogLoss")
    ax.set_title(f"LGBM Learning Curve{tag}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_xlim(left=0)
    plt.tight_layout()
    path = out / f"learning_curve{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    # CSV: collect all metrics into rows
    rows = []
    max_len = 0
    for ds_name, metrics in evals_result.items():
        for metric_name, values in metrics.items():
            max_len = max(max_len, len(values))
    for round_idx in range(max_len):
        row = [str(round_idx)]
        for ds_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                row.append(f"{values[round_idx]:.6f}" if round_idx < len(values) else "")
        rows.append(row)

    header = ["round"]
    for ds_name, metrics in evals_result.items():
        for metric_name in metrics:
            header.append(f"{ds_name}_{metric_name}")
    _write_csv(out / f"learning_curve{tag}.csv", rows, header)
    logger.info(f"[Viz] Learning curve saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 21) Top-K accuracy curve
# ═══════════════════════════════════════════════════════════════

def plot_topk_accuracy_curve(
    proba_by_stage: dict,
    labels: np.ndarray,
    class_names: list,
    viz_dir: str,
):
    """Top-K accuracy vs K curve, comparing across stages.

    Parameters
    ----------
    proba_by_stage: {stage_label: proba_matrix [N, C]}
    labels: [N] Ground truth labels (int)
    class_names: List of class names
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "08_summary")
    labels = np.asarray(labels).astype(int)
    stage_labels = list(proba_by_stage.keys())
    if not stage_labels:
        return

    n_classes = len(class_names)
    ks = list(range(1, n_classes + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(stage_labels)))
    csv_rows = [["stage", "K", "topk_accuracy"]]

    for si, (slabel, proba) in enumerate(proba_by_stage.items()):
        proba = np.asarray(proba).astype(np.float32)
        # top-k indices per sample
        topk_indices = np.argsort(proba, axis=1)[:, ::-1]  # [N, C], descending
        topk_vals = []
        for k in ks:
            # Check if true label is in top-k
            hit = np.array([labels[i] in topk_indices[i, :k] for i in range(len(labels))])
            topk_vals.append(float(hit.mean()))
        ax.plot(ks, topk_vals, marker="o", linewidth=1.5, color=colors[si], label=slabel, markersize=4)
        for k, v in zip(ks, topk_vals):
            csv_rows.append([slabel, str(k), f"{v:.4f}"])

    ax.set_xlabel("K (Top-K)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-K Accuracy Across Pipeline Stages")
    ax.legend(fontsize=9)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    path = out / "topk_accuracy_curve.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    _write_csv(out / "topk_accuracy_curve.csv", csv_rows, ["stage", "K", "topk_accuracy"])
    logger.info(f"[Viz] Top-K accuracy curve saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 22) Behavior segment duration vs classification error rate
# ═══════════════════════════════════════════════════════════════

def plot_error_vs_duration(
    y_true: np.ndarray,
    y_pred_by_stage: dict,
    class_names: list,
    viz_dir: str,
    n_bins: int = 8,
):
    """Behavior segment duration vs classification error rate scatter/line chart.

    Parameters
    ----------
    y_true: [T] Ground truth label sequence
    y_pred_by_stage: {stage_label: y_pred_array [T]}
    class_names: List of class names
    n_bins: Number of duration bins
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "08_summary")
    y_true = np.asarray(y_true).astype(int)
    stage_labels = list(y_pred_by_stage.keys())
    if not stage_labels:
        return

    # Compute each contiguous behavior segment (start, end, class, duration)
    segments = []
    T = len(y_true)
    if T == 0:
        return
    run_start = 0
    for t in range(1, T):
        if y_true[t] != y_true[t - 1]:
            segments.append((run_start, t, int(y_true[run_start]), t - run_start))
            run_start = t
    segments.append((run_start, T, int(y_true[run_start]), T - run_start))

    if not segments:
        return

    durations = np.array([s[3] for s in segments])
    # Log-scale binning
    if durations.max() <= 1:
        return
    log_min = np.log10(max(1, durations.min()))
    log_max = np.log10(durations.max() + 1)
    bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(stage_labels)))
    csv_rows = []

    for si, (slabel, y_pred) in enumerate(y_pred_by_stage.items()):
        y_pred = np.asarray(y_pred).astype(int)
        bin_errors = []
        bin_stds = []
        bin_counts = []
        for bi in range(n_bins):
            lo, hi = bin_edges[bi], bin_edges[bi + 1]
            mask = (durations >= lo) & (durations < hi) if bi < n_bins - 1 else (durations >= lo)
            if mask.sum() == 0:
                bin_errors.append(np.nan)
                bin_stds.append(np.nan)
                bin_counts.append(0)
                continue
            # Compute frame-level error rate for segments in this bin
            seg_errors = []
            for seg_i in np.where(mask)[0]:
                seg = segments[int(seg_i)]
                seg_pred = y_pred[seg[0]:seg[1]]
                seg_true = y_true[seg[0]:seg[1]]
                err = 1.0 - (seg_pred == seg_true).mean()
                seg_errors.append(err)
            seg_errors = np.array(seg_errors)
            bin_errors.append(float(seg_errors.mean()))
            bin_stds.append(float(seg_errors.std()))
            bin_counts.append(int(mask.sum()))

        valid = ~np.isnan(bin_errors)
        if valid.any():
            x_plot = bin_centers[valid]
            y_plot = np.array(bin_errors)[valid]
            ax.errorbar(x_plot, y_plot,
                        yerr=np.array(bin_stds)[valid] / np.sqrt(np.maximum(np.array(bin_counts)[valid], 1)),
                        marker="o", linewidth=1.5, color=colors[si], label=slabel, markersize=5, capsize=3)
            for bi in range(n_bins):
                csv_rows.append([slabel, str(int(bi)),
                                 f"{bin_centers[bi]:.1f}", f"{bin_edges[bi]:.1f}-{bin_edges[bi+1]:.1f}",
                                 f"{bin_errors[bi]:.4f}" if not np.isnan(bin_errors[bi]) else "",
                                 str(bin_counts[bi])])

    ax.set_xscale("log")
    ax.set_xlabel("Behavior Segment Duration (frames, log scale)")
    ax.set_ylabel("Frame-Level Error Rate")
    ax.set_title("Error Rate vs Behavior Segment Duration")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_ylim(bottom=-0.02, top=1.02)
    plt.tight_layout()
    path = out / "error_vs_duration.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    _write_csv(out / "error_vs_duration.csv", csv_rows,
               ["stage", "bin", "bin_center", "bin_range", "mean_error", "n_segments"])
    logger.info(f"[Viz] Error rate vs segment duration saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 23) Prediction flip analysis (prediction changes between stages)
# ═══════════════════════════════════════════════════════════════

def plot_prediction_flip_flow(
    y_pred_by_stage: dict,
    class_names: list,
    viz_dir: str,
):
    """Prediction flip matrix and net gain bar chart between stages.

    Parameters
    ----------
    y_pred_by_stage: Ordered dict {stage_label: y_pred_array [T]}
    class_names: List of class names
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "08_summary")
    stage_labels = list(y_pred_by_stage.keys())
    if len(stage_labels) <= 1:
        return

    # Compare adjacent stages
    n_pairs = len(stage_labels) - 1
    fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 4.5), squeeze=False)

    all_csv_rows = []
    for pi in range(n_pairs):
        prev_label = stage_labels[pi]
        next_label = stage_labels[pi + 1]
        y_prev = np.asarray(y_pred_by_stage[prev_label]).astype(int)
        y_next = np.asarray(y_pred_by_stage[next_label]).astype(int)

        # Flip matrix: from prev prediction to next prediction
        n = len(class_names)
        flip_matrix = np.zeros((n, n), dtype=int)
        for i in range(len(y_prev)):
            flip_matrix[y_prev[i], y_next[i]] += 1
        # Exclude diagonal (unchanged predictions)

        # Net gain: corrections minus errors per class
        y_true = None  # We need ground truth — passed separately? Actually we don't have it.
        # Let's use a simpler approach: count flips only
        total_flips = flip_matrix.sum() - np.trace(flip_matrix)
        flip_rate = total_flips / len(y_prev) if len(y_prev) > 0 else 0

        ax = axes[0, pi]
        # Show only off-diagonal elements
        off_diag = flip_matrix.copy()
        np.fill_diagonal(off_diag, 0)
        if off_diag.max() > 0:
            im = ax.imshow(off_diag, interpolation="nearest", cmap="YlOrRd", aspect="auto")
            plt.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_xlabel(f"Predicted ({next_label})", fontsize=8)
        ax.set_ylabel(f"Predicted ({prev_label})", fontsize=8)
        ax.set_title(f"{prev_label} → {next_label}\n({total_flips} flips, {flip_rate:.1%} of frames)", fontsize=9)

        # Annotate non-zero flips
        for i in range(n):
            for j in range(n):
                if i != j and flip_matrix[i, j] > 0:
                    ax.text(j, i, str(flip_matrix[i, j]), ha="center", va="center", fontsize=6,
                            color="white" if flip_matrix[i, j] > off_diag.max() / 2 else "black")

        for i in range(n):
            for j in range(n):
                all_csv_rows.append([f"{prev_label}→{next_label}", class_names[i], class_names[j],
                                     str(flip_matrix[i, j])])

    plt.suptitle("Prediction Flips Between Pipeline Stages", fontsize=12)
    plt.tight_layout()
    path = out / "prediction_flip_flow.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_csv(out / "prediction_flip_flow.csv", all_csv_rows,
               ["transition", "from_pred", "to_pred", "count"])
    logger.info(f"[Viz] Prediction flip analysis saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 24) Model calibration curve (Reliability Diagram)
# ═══════════════════════════════════════════════════════════════

def plot_calibration_curve(
    proba: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    viz_dir: str,
    stage_name: str = "",
    n_bins: int = 10,
):
    """Per-class calibration curve (Reliability Diagram), with CSV.

    Parameters
    ----------
    proba: [N, C] Prediction probability matrix
    labels: [N] Ground truth labels (int)
    class_names: List of class names
    stage_name: Stage name identifier
    n_bins: Number of confidence bins
    """
    if not _MPL_AVAILABLE:
        return
    tag = f"_{stage_name}" if stage_name else ""
    out = _ensure_dir(Path(viz_dir) / "09_best_model")
    proba = np.asarray(proba).astype(np.float32)
    labels = np.asarray(labels).astype(int)
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, min(2, n_classes), figsize=(7 * min(2, n_classes), 5.5),
                              squeeze=False)
    if n_classes == 1:
        axes = np.array([[axes[0, 0]]])

    colors = plt.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)
    csv_rows = []

    for ax_i in range(min(2, n_classes)):
        ax = axes[0, ax_i]
        c = ax_i
        y_bin = (labels == c).astype(int)
        proba_c = proba[:, c]

        # Binned calibration computation
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_acc = []
        bin_conf = []
        bin_counts = []
        for bi in range(n_bins):
            mask = (proba_c >= bin_edges[bi]) & (proba_c < bin_edges[bi + 1])
            if bi == n_bins - 1:
                mask = (proba_c >= bin_edges[bi]) & (proba_c <= bin_edges[bi + 1])
            if mask.sum() > 0:
                bin_acc.append(float(y_bin[mask].mean()))
                bin_conf.append(float(proba_c[mask].mean()))
                bin_counts.append(int(mask.sum()))
            else:
                bin_acc.append(np.nan)
                bin_conf.append(np.nan)
                bin_counts.append(0)

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfectly calibrated")
        valid = ~np.isnan(bin_acc)
        if valid.any():
            ax.plot(np.array(bin_conf)[valid], np.array(bin_acc)[valid], marker="o",
                    linewidth=2, color=colors(c), label=f"Class: {class_names[c]}", markersize=6)
            # Bar chart showing sample count per bin
            ax2 = ax.twinx()
            ax2.bar(bin_centers, bin_counts, width=0.08, alpha=0.2, color="gray", edgecolor="none")
            ax2.set_ylabel("Frame count", fontsize=8, alpha=0.6)

        ax.set_xlabel("Mean Predicted Probability (Confidence)")
        ax.set_ylabel("Fraction of Positives (Accuracy)")
        ax.set_title(f"Calibration Curve — {class_names[c]}{tag}")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3, linewidth=0.5)

        for bi in range(n_bins):
            csv_rows.append([class_names[c], stage_name, str(bi),
                             f"{bin_edges[bi]:.2f}-{bin_edges[bi+1]:.2f}",
                             f"{bin_conf[bi]:.4f}" if not np.isnan(bin_conf[bi]) else "",
                             f"{bin_acc[bi]:.4f}" if not np.isnan(bin_acc[bi]) else "",
                             str(bin_counts[bi])])

    # ECE annotation
    ece_total = 0.0
    total_n = len(labels)
    for c in range(n_classes):
        y_bin = (labels == c).astype(int)
        proba_c = proba[:, c]
        ece_c = 0.0
        for bi in range(n_bins):
            mask = (proba_c >= bin_edges[bi]) & (proba_c < bin_edges[bi + 1])
            if bi == n_bins - 1:
                mask = (proba_c >= bin_edges[bi]) & (proba_c <= bin_edges[bi + 1])
            n_b = mask.sum()
            if n_b > 0:
                acc_b = y_bin[mask].mean()
                conf_b = proba_c[mask].mean()
                ece_c += (n_b / total_n) * abs(acc_b - conf_b) if total_n > 0 else 0
        ece_total += ece_c / n_classes if n_classes > 0 else 0

    fig.suptitle(f"Reliability Diagrams{tag}  (ECE={ece_total:.4f})", fontsize=12)
    plt.tight_layout()
    path = out / f"calibration_curve{tag}.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    csv_rows.append(["ECE", stage_name, "", "", "", f"{ece_total:.4f}", ""])
    _write_csv(out / f"calibration_curve{tag}.csv", csv_rows,
               ["class", "stage", "bin", "conf_range", "mean_conf", "mean_acc", "count"])
    logger.info(f"[Viz] Calibration curve saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 25) Temporal window size sensitivity analysis (based on multi-window run results)
# ═══════════════════════════════════════════════════════════════

def plot_window_size_sensitivity(
    window_results: dict,
    viz_dir: str,
):
    """Temporal window size vs metric line chart (requires multiple runs to collect data).

    Parameters
    ----------
    window_results: {window_size: {"accuracy":, "macro_f1":, "macro_auc":, "weighted_f1":}}
    viz_dir: Output directory
    """
    if not _MPL_AVAILABLE or not window_results:
        return
    out = _ensure_dir(Path(viz_dir) / "06_temporal")
    ws_list = sorted(window_results.keys())

    metrics = ["accuracy", "macro_f1", "macro_auc", "weighted_f1"]
    metric_labels = ["Accuracy", "Macro F1", "Macro AUC", "Weighted F1"]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    csv_rows = []
    for mi, (mkey, mlabel) in enumerate(zip(metrics, metric_labels)):
        vals = []
        for ws in ws_list:
            v = window_results[ws].get(mkey, 0)
            vals.append(v if v is not None else 0)
            csv_rows.append([str(ws), mkey, f"{window_results[ws].get(mkey, 0):.4f}"])
        ax.plot(ws_list, vals, marker="o", linewidth=1.5, color=colors[mi], label=mlabel, markersize=6)

    ax.set_xlabel("Temporal Window Size (frames)")
    ax.set_ylabel("Score")
    ax.set_title("Temporal Window Size Sensitivity")
    ax.legend(fontsize=9)
    ax.set_xticks(ws_list)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = out / "window_size_sensitivity.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    _write_csv(out / "window_size_sensitivity.csv", csv_rows, ["window_size", "metric", "value"])
    logger.info(f"[Viz] Window size sensitivity analysis saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 26) Per-class AUC cross-stage comparison bar chart
# ═══════════════════════════════════════════════════════════════

def plot_per_class_auc_comparison(
    per_class_by_stage: dict,
    class_names: list,
    viz_dir: str,
):
    """Cross-stage grouped bar chart of AUC per class.

    Parameters
    ----------
    per_class_by_stage: {stage_label: {class_name: {"auc": float or None}}}
    class_names: List of class names
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "08_summary")
    stage_labels = list(per_class_by_stage.keys())
    if len(stage_labels) <= 1:
        return

    n_classes = len(class_names)
    n_stages = len(stage_labels)
    w = 0.8 / n_stages
    x = np.arange(n_classes)
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, n_stages))

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.4), 5.5))
    for si, slabel in enumerate(stage_labels):
        vals = []
        for cname in class_names:
            v = per_class_by_stage[slabel].get(cname, {}).get("auc")
            vals.append(v if v is not None else 0)
        offset = (si - (n_stages - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=slabel, color=colors[si], edgecolor="white", linewidth=0.3)
        for i, v in enumerate(vals):
            if v > 0.01:
                ax.text(x[i] + offset, v + 0.01, f"{v:.2f}", ha="center", fontsize=5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_title("Per-Class AUC Across Pipeline Stages")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    path = out / "per_class_auc_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    # CSV
    rows = []
    for slabel in stage_labels:
        for cname in class_names:
            v = per_class_by_stage[slabel].get(cname, {}).get("auc")
            rows.append([slabel, cname, f"{v:.4f}" if v is not None else ""])
    _write_csv(out / "per_class_auc_comparison.csv", rows, ["stage", "class", "auc"])
    logger.info(f"[Viz] Per-class AUC comparison saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 27) Behavior segment length distribution + per-class segment statistics
# ═══════════════════════════════════════════════════════════════

def plot_segment_duration_analysis(
    y_true: np.ndarray,
    class_names: list,
    viz_dir: str,
):
    """Behavior segment duration distribution: overall histogram + per-class box plot.

    Parameters
    ----------
    y_true: [T] Ground truth label sequence
    class_names: List of class names
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "01_data")
    y_true = np.asarray(y_true).astype(int)
    n_classes = len(class_names)

    # Extract all contiguous segments
    per_class_durations = {c: [] for c in range(n_classes)}
    all_durations = []
    T = len(y_true)
    if T == 0:
        return
    run_start = 0
    for t in range(1, T):
        if y_true[t] != y_true[t - 1]:
            dur = t - run_start
            c = int(y_true[run_start])
            if c < n_classes:
                per_class_durations[c].append(dur)
            all_durations.append(dur)
            run_start = t
    dur = T - run_start
    c = int(y_true[run_start])
    if c < n_classes:
        per_class_durations[c].append(dur)
    all_durations.append(dur)

    all_durations = np.array(all_durations)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Overall segment duration histogram
    ax = axes[0]
    log_max = np.log10(max(1, all_durations.max()))
    bins = np.logspace(0, log_max + 0.2, 30)
    ax.hist(all_durations, bins=bins, color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Segment Duration (frames, log scale)")
    ax.set_ylabel("Count")
    ax.set_title(f"Behavior Segment Duration Distribution\n({len(all_durations)} segments, "
                 f"mean={all_durations.mean():.1f}, median={np.median(all_durations):.1f})")

    # (b) Per-class box plot
    ax = axes[1]
    data_to_plot = []
    labels_to_plot = []
    stats_rows = []
    for c in range(n_classes):
        if per_class_durations[c]:
            data_to_plot.append(np.array(per_class_durations[c]))
            labels_to_plot.append(class_names[c])
            d = np.array(per_class_durations[c])
            stats_rows.append([class_names[c], str(len(d)),
                               f"{d.mean():.1f}", f"{np.median(d):.1f}",
                               f"{d.min()}", f"{d.max()}",
                               f"{d.std():.1f}"])

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True, vert=True)
        colors = plt.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)
        for patch_i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors(patch_i))
            patch.set_alpha(0.6)
        ax.set_ylabel("Segment Duration (frames)")
        ax.set_title("Per-Class Segment Duration Distribution")
        ax.tick_params(axis="x", rotation=35, labelsize=8)

    plt.suptitle("Behavior Segment Duration Analysis", fontsize=13)
    plt.tight_layout()
    path = out / "segment_duration_analysis.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    _write_csv(out / "segment_duration_analysis.csv", stats_rows,
               ["class", "n_segments", "mean_dur", "median_dur", "min_dur", "max_dur", "std_dur"])
    logger.info(f"[Viz] Segment duration analysis saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 28) Macro-average metrics radar chart by stage
# ═══════════════════════════════════════════════════════════════

def plot_stage_radar(
    stage_metrics: dict,
    viz_dir: str,
):
    """Macro-average metrics radar chart by stage (accuracy, balanced_acc, macro_f1, macro_auc, weighted_f1, weighted_auc).

    Parameters
    ----------
    stage_metrics: {stage_label: {"accuracy":, "balanced_accuracy":, "macro_f1":, "macro_auc":, "weighted_f1":, "weighted_auc":}}
    """
    if not _MPL_AVAILABLE:
        return
    out = _ensure_dir(Path(viz_dir) / "08_summary")
    stage_labels = list(stage_metrics.keys())
    if len(stage_labels) <= 1:
        return

    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "weighted_f1", "weighted_auc"]
    metric_labels = ["Accuracy", "Balanced Acc", "Macro F1", "Macro AUC", "Weighted F1", "Weighted AUC"]
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(stage_labels)))

    csv_rows = []
    for si, slabel in enumerate(stage_labels):
        vals = [stage_metrics[slabel].get(m, 0) or 0 for m in metrics]
        vals += vals[:1]
        ax.fill(angles, vals, alpha=0.1, color=colors[si])
        ax.plot(angles, vals, linewidth=1.5, color=colors[si], label=slabel, marker="o", markersize=4)
        for mi, m in enumerate(metrics):
            csv_rows.append([slabel, metric_labels[mi], f"{stage_metrics[slabel].get(m, 0):.4f}"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title("Pipeline Stage Metrics Radar", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    path = out / "stage_radar.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    _write_csv(out / "stage_radar.csv", csv_rows, ["stage", "metric", "value"])
    logger.info(f"[Viz] Stage radar chart saved: {path}")
