"""
correlation.py -- Factor correlation analysis tool

Features:
  1. Load valid_factors.json, group by seq_length
  2. Optionally select top-k factors by AUC per group for analysis
  3. Compute factor values on the training set on the fly
  4. Compute Pearson correlation matrix within each group
  5. Output heatmap (PNG) + CSV (raw correlation matrix data) + correlation report (JSON)
  6. Greedy redundancy removal: when correlation exceeds threshold, keep the factor with stronger predictive power
  7. The recommended_factors in the report can be passed directly to train_behavior.py (via filter_corr field in validation.yaml)

Usage:
  python correlation.py
  python correlation.py --threshold 0.85
  python correlation.py --threshold 0.85 --keep-by best_auc
  python correlation.py --max-samples 50000  # quick debug
  python correlation.py --seq 1 3            # only analyze specified seq_length
  python correlation.py --no-plot            # skip heatmap generation
  python correlation.py --top-k 20           # only select top 20 factors by AUC per group
  python correlation.py --config-common config/seq/1.yaml --config-validation config/validation.yaml
"""


import sys
from pathlib import Path
# Allow running as a script from a subdirectory (e.g. `python mining/discovery.py`)
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    import io
    if getattr(sys.stdout, "encoding", None) != "utf-8" and getattr(sys.stdout, "buffer", None) is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# --------------------------- logging ----------------------------

def _setup_logging() -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    h.flush = sys.stdout.flush  # Ensure no buffering on Windows

    # Configure root logger to ensure all modules (synth_validator, factor_parallel, etc.) output logs
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        root.addHandler(h)

    log = logging.getLogger("factor_corr")
    log.setLevel(logging.INFO)
    return log


# --------------------------- factor matrix ----------------------

def _compute_factor_values(
    factors: list,
    kp_full: np.ndarray,
    flat_attributes: list,
    labels: np.ndarray,
    purity_mode: str,
    num_workers: int,
    num_chunks: int,
    log: logging.Logger,
) -> tuple:
    """
    Compute factor matrix, returns (X [T, K], used_names, dropped_names).
    Uses multi-resolution mode: each factor builds centered windows at its own seq_length.
    """
    from src.synth_validator import SynthValidator

    # Build a minimal cfg, only needs factor_engine section
    cfg = {"factor_engine": {"max_error_ratio": 0.5, "min_valid_ratio": 0.1}}
    sv = SynthValidator(cfg)

    # Statistics by seq_length
    from collections import Counter
    sl_dist = Counter(f.get("seq_length", 1) for f in factors)
    log.info(f"  Factor seq_length distribution: {dict(sorted(sl_dist.items()))}")
    log.info(f"  Mode distribution: row={sum(1 for f in factors if f.get('mode','row')!='batch')}, "
             f"batch={sum(1 for f in factors if f.get('mode','row')=='batch')}")

    t0 = time.time()
    if num_workers > 1:
        log.info(f"  Using multiprocess mode ({num_workers} workers, {num_chunks} chunks)...")
        X, labels_out, used, dropped = sv.compute_factor_matrix_multiresolution_fast(
            factors=factors,
            kp_full=kp_full,
            labels=labels,
            flat_attributes=flat_attributes,
            purity_mode=purity_mode,
            num_workers=num_workers,
            num_chunks=num_chunks,
            split_name="corr",
        )
    else:
        log.info(f"  Using single-process mode...")
        X, labels_out, used, dropped = sv.compute_factor_matrix_multiresolution(
            factors=factors,
            kp_full=kp_full,
            labels=labels,
            flat_attributes=flat_attributes,
            purity_mode=purity_mode,
            split_name="corr",
        )

    elapsed = time.time() - t0
    nan_ratio = np.isnan(X).sum() / X.size * 100 if X.size > 0 else 0
    log.info(f"  Factor matrix: {X.shape}, used {len(used)}, dropped {len(dropped)}, "
             f"NaN ratio {nan_ratio:.1f}%, time {elapsed:.1f}s")
    return X, used, dropped


# --------------------------- correlation ------------------------

def _pearson_corr_matrix(X: np.ndarray, log: logging.Logger) -> np.ndarray:
    """Compute column-wise Pearson correlation matrix, fill NaN columns with mean."""
    T, K = X.shape
    log.info(f"  Correlation matrix computation: {T} samples x {K} factors")

    t0 = time.time()
    col_means = np.nanmean(X, axis=0)
    nan_counts = np.isnan(X).sum(axis=0)
    if nan_counts.any():
        nan_cols = (nan_counts > 0).sum()
        log.info(f"  {nan_cols}/{K} columns contain NaN, filling with mean...")
    X_filled = np.where(np.isnan(X), col_means[np.newaxis, :], X)

    stds = X_filled.std(axis=0)
    zero_var = stds == 0
    if zero_var.any():
        n_zero = zero_var.sum()
        log.info(f"  {n_zero} columns have zero variance, setting to zero")
        X_filled[:, zero_var] = 0.0

    corr = np.corrcoef(X_filled.T)
    corr = np.nan_to_num(corr, nan=0.0)
    elapsed = time.time() - t0
    log.info(f"  Correlation matrix computation complete, time {elapsed:.1f}s")
    return corr.astype(np.float32)


def _greedy_dedup(
    factor_names: list,
    factors_meta: dict,
    corr: np.ndarray,
    threshold: float,
    keep_by: str,
    log: logging.Logger,
) -> tuple:
    """
    Greedy redundancy removal: sort by keep_by descending, add to keep set one by one;
    discard if correlation with any kept factor >= threshold.

    Returns (kept_names, removed_pairs)
    removed_pairs: list of {"removed": name, "kept_instead": name, "corr": float}
    """
    t0 = time.time()

    def score(name):
        meta = factors_meta.get(name, {})
        if keep_by == "best_auc":
            return meta.get("best_auc", 0.0)
        if keep_by == "weighted_auc":
            vcs = meta.get("valid_classes", [])
            if not vcs:
                return meta.get("best_auc", 0.0)
            return sum(v["auc"] for v in vcs) / len(vcs)
        if keep_by == "seq_length_asc":
            return -meta.get("seq_length", 1)
        return meta.get("best_auc", 0.0)

    n = len(factor_names)
    order = sorted(range(n), key=lambda i: score(factor_names[i]), reverse=True)

    abs_corr = np.abs(corr)
    kept_idx = []
    removed_pairs = []

    for i in order:
        if kept_idx:
            max_corr_with_kept = abs_corr[i, kept_idx].max()
            if max_corr_with_kept >= threshold:
                j = kept_idx[int(abs_corr[i, kept_idx].argmax())]
                removed_pairs.append({
                    "removed": factor_names[i],
                    "kept_instead": factor_names[j],
                    "corr": round(float(abs_corr[i, j]), 4),
                })
                continue
        kept_idx.append(i)

    kept_names = [factor_names[i] for i in kept_idx]
    elapsed = time.time() - t0
    log.info(f"  Greedy dedup time {elapsed:.2f}s")
    return kept_names, removed_pairs


# --------------------------- plot -------------------------------

def _save_corr_csv(corr: np.ndarray, names: list, csv_path: str) -> bool:
    """Save correlation matrix as CSV file with row and column factor names."""
    try:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([""] + names)
            for i, name in enumerate(names):
                writer.writerow([name] + [round(float(v), 6) for v in corr[i]])
        return True
    except Exception:
        return False


def _plot_heatmap(corr: np.ndarray, names: list, out_path: str, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        n = len(names)
        fig_size = max(6, min(n * 0.35 + 2, 40))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        cmap = plt.cm.RdBu_r
        im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        tick_labels = [n[:20] for n in names]
        if n <= 60:
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=max(4, 8 - n // 20))
            ax.set_yticklabels(tick_labels, fontsize=max(4, 8 - n // 20))
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        return False


# --------------------------- main logic -------------------------

def run_correlation_analysis(
    cfg: dict,
    cfg_dir: Path,
    factors_path: str,
    output_dir: str,
    threshold: float,
    keep_by: str,
    max_samples: int,
    num_workers: int,
    num_chunks: int,
    purity_mode: str,
    seq_filter: list,
    top_k: int,
    no_plot: bool,
    output_factors_path: str = "",
    log: logging.Logger = None,
):
    from mining.discovery import load_dataset_config, build_raw_frame_data

    total_t0 = time.time()
    print("\n" + "=" * 60)
    print("Factor correlation analysis")
    print(f"  Threshold: {threshold}  Strategy: {keep_by}  Purity mode: {purity_mode}")
    print(f"  Parallel: {num_workers} workers x {num_chunks} chunks")
    print("=" * 60)

    ds_cfg_file = cfg.get("dataset_config_file", "")
    if not ds_cfg_file:
        raise ValueError("common.yaml is missing the dataset_config_file field.")
    ds_cfg_path = Path(ds_cfg_file)
    if not ds_cfg_path.is_absolute():
        ds_cfg_path = cfg_dir / ds_cfg_path

    ds_info = load_dataset_config(cfg, str(ds_cfg_path), log)

    t0 = time.time()
    log.info("Loading raw frame data (training set)...")
    train_data, _, flat_attributes, _video_lengths = build_raw_frame_data(ds_info, log, cfg=cfg)
    kp_full, labels = train_data  # [T, D], [T]
    log.info(f"Data loading complete, time {time.time() - t0:.1f}s")

    if max_samples > 0 and len(labels) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(labels), max_samples, replace=False)
        idx.sort()
        kp_full = kp_full[idx]
        labels = labels[idx]
        log.info(f"Sampled {max_samples} frames for correlation computation")

    log.info(f"Data size: {kp_full.shape[0]} frames, D={kp_full.shape[1]}, "
             f"class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Load factors
    with open(factors_path, "r", encoding="utf-8") as f:
        all_factors = json.load(f)
    log.info(f"Factor library: {len(all_factors)} factors (source: {factors_path})")

    if seq_filter:
        all_factors = [fac for fac in all_factors if fac.get("seq_length") in seq_filter]
        log.info(f"After seq_length filter: {len(all_factors)} factors (keeping seq={seq_filter})")

    if not all_factors:
        log.error("No factors after filtering, exiting.")
        return

    # Group by seq_length
    by_seq: dict = defaultdict(list)
    for fac in all_factors:
        by_seq[fac.get("seq_length", 1)].append(fac)

    log.info(f"Grouping: {', '.join(f'seq={k}({len(v)})' for k, v in sorted(by_seq.items()))}")

    # Top-k filtering per group (by best_auc descending)
    if top_k > 0:
        for seq_len in by_seq:
            group = by_seq[seq_len]
            group.sort(key=lambda f: f.get("best_auc", 0.0), reverse=True)
            original = len(group)
            by_seq[seq_len] = group[:top_k]
            if len(group) > top_k:
                log.info(f"  seq={seq_len}: top-{top_k} filter, {original} -> {len(by_seq[seq_len])}")

    factors_meta = {fac["name"]: fac for fac in all_factors}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global result container
    report = {
        "threshold": threshold,
        "keep_by": keep_by,
        "purity_mode": purity_mode,
        "total_factors": len(all_factors),
        "groups": {},
        "recommended_factors": [],
        "removed_pairs": [],
    }

    all_kept = []
    group_count = len(by_seq)

    for gi, seq_len in enumerate(sorted(by_seq.keys()), 1):
        group_factors = by_seq[seq_len]
        gname = f"seq_{seq_len}"
        group_t0 = time.time()
        log.info(f"\n{'='*55}")
        log.info(f"[{gi}/{group_count}] Processing group {gname}: {len(group_factors)} factors")
        log.info(f"{'='*55}")

        # Compute factor matrix
        t0 = time.time()
        log.info(f"  Computing factor matrix...")
        X, used_names, dropped_names = _compute_factor_values(
            factors=group_factors,
            kp_full=kp_full,
            flat_attributes=flat_attributes,
            labels=labels,
            purity_mode=purity_mode,
            num_workers=num_workers,
            num_chunks=num_chunks,
            log=log,
        )
        log.info(f"  Factor computation time {time.time() - t0:.1f}s")

        if len(used_names) == 0:
            log.warning(f"  Group {gname} has no valid factors, skipping.")
            report["groups"][gname] = {"n_factors": 0, "n_used": 0, "n_dropped": len(dropped_names)}
            continue

        # Compute correlation matrix
        corr = _pearson_corr_matrix(X, log)

        # Count high-correlation pairs (vectorized)
        t0 = time.time()
        n = len(used_names)
        abs_corr_upper = np.abs(np.triu(corr, k=1))
        high_mask = abs_corr_upper >= threshold
        high_count = high_mask.sum()
        log.info(f"  High-correlation pairs (|r|>={threshold}): {high_count} pairs")

        high_corr_pairs = []
        if high_count > 0:
            rows, cols = np.where(high_mask)
            corr_vals = abs_corr_upper[rows, cols]
            sort_idx = np.argsort(-corr_vals)
            for si in sort_idx[:200]:
                i, j = int(rows[si]), int(cols[si])
                high_corr_pairs.append({
                    "factor_a": used_names[i],
                    "factor_b": used_names[j],
                    "corr": round(float(corr_vals[si]), 4),
                })
            if high_count > 5:
                top5 = high_corr_pairs[:5]
                log.info(f"  Top-5 high-correlation pairs:")
                for p in top5:
                    log.info(f"    {p['factor_a'][:30]} <-> {p['factor_b'][:30]} : r={p['corr']}")

        # Greedy redundancy removal
        kept, removed = _greedy_dedup(
            factor_names=used_names,
            factors_meta=factors_meta,
            corr=corr,
            threshold=threshold,
            keep_by=keep_by,
            log=log,
        )
        log.info(f"  Dedup result: {len(used_names)} -> {len(kept)} (removed {len(removed)})")

        all_kept.extend(kept)
        report["removed_pairs"].extend(removed)

        # Sparse correlation matrix storage (vectorized filtering)
        abs_upper = np.abs(np.triu(corr, k=1))
        sig_mask = abs_upper >= 0.3
        corr_dict = {}
        if sig_mask.any():
            sig_rows, sig_cols = np.where(sig_mask)
            for r, c in zip(sig_rows, sig_cols):
                corr_dict[f"{used_names[r]}|{used_names[c]}"] = round(float(corr[r, c]), 4)

        report["groups"][gname] = {
            "seq_length": seq_len,
            "n_factors": len(group_factors),
            "n_used": len(used_names),
            "n_dropped_compute": len(dropped_names),
            "n_high_corr_pairs": int(high_count),
            "n_kept_after_dedup": len(kept),
            "n_removed_by_dedup": len(removed),
            "used_factor_names": used_names,
            "dropped_factor_names": dropped_names,
            "kept_factor_names": kept,
            "high_corr_pairs": high_corr_pairs[:200],
            "corr_sparse": corr_dict,
        }

        # Heatmap + CSV
        if len(used_names) >= 2:
            # CSV: save raw correlation matrix
            csv_path = str(out_dir / f"corr_heatmap_{gname}.csv")
            _save_corr_csv(corr, used_names, csv_path)
            log.info(f"  Correlation matrix CSV saved: {csv_path}")

            if not no_plot:
                plot_path = str(out_dir / f"corr_heatmap_{gname}.png")
                ok = _plot_heatmap(
                    corr=corr,
                    names=used_names,
                    out_path=plot_path,
                    title=f"Factor Correlation -- seq_length={seq_len} ({len(used_names)} factors)",
                )
                if ok:
                    log.info(f"  Heatmap saved: {plot_path}")
                else:
                    log.warning(f"  Heatmap generation failed (matplotlib unavailable?)")

        group_elapsed = time.time() - group_t0
        log.info(f"  Group {gname} total time {group_elapsed:.1f}s")

    report["recommended_factors"] = all_kept
    report["n_recommended"] = len(all_kept)
    report["n_total_removed"] = len(report["removed_pairs"])

    # Save JSON report
    report_path = out_dir / "factor_correlation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Save deduplicated factor JSON (optional)
    if output_factors_path:
        name_to_factor = {fac["name"]: fac for fac in all_factors}
        kept_factors = []
        missing = 0
        for name in all_kept:
            fac = name_to_factor.get(name)
            if fac:
                kept_factors.append(fac)
            else:
                missing += 1
        if missing:
            log.warning(f"{missing} recommended factors not found in original list (skipped)")

        out_path = Path(output_factors_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kept_factors, f, ensure_ascii=False, indent=2)
        log.info(f"Deduplicated factors saved: {out_path} ({len(kept_factors)})")

    total_elapsed = time.time() - total_t0

    # Console summary
    print("\n" + "=" * 60)
    print(f"Factor correlation analysis complete (threshold={threshold}, keep_by={keep_by})")
    print(f"Total time: {total_elapsed:.1f}s")
    print(f"{'Group':<15} {'Total':>6} {'Valid':>6} {'HighCorr':>8} {'Kept':>6} {'Removed':>6}")
    print("-" * 60)
    for gname, g in report["groups"].items():
        print(f"{gname:<15} {g['n_factors']:>6} {g['n_used']:>6} "
              f"{g['n_high_corr_pairs']:>8} {g['n_kept_after_dedup']:>6} {g['n_removed_by_dedup']:>6}")
    print("=" * 60)
    print(f"Recommended factors: {report['n_recommended']} / {report['total_factors']}")
    print(f"Report path: {report_path}")
    if output_factors_path:
        print(f"Deduplicated factors: {output_factors_path} ({len(kept_factors)})")
    print("=" * 60)
    print("\nRun synthetic validation with recommended factors:")
    print(f"  Set run.filter_corr: \"{report_path}\" in config/validation.yaml")
    print(f"  Then run: python train_behavior.py")


# --------------------------- CLI --------------------------------

def main():
    from mining.discovery import _load_merged_config

    p = argparse.ArgumentParser(
        description="Factor correlation analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config-common", default="config/seq/1.yaml", help="Common configuration file")
    p.add_argument("--config-validation", default="config/validation.yaml", help="Synthetic validation configuration file")
    p.add_argument("--config", default=None, help="Single config file path (legacy usage)")
    p.add_argument("--factors", default=None,
                   help="Valid factor JSON path (default: read from validation.yaml run.factors)")
    p.add_argument("--output-dir", default=None,
                   help="Report and heatmap output directory (default: read from validation.yaml run.output_dir)")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Correlation dedup threshold, absolute value >= this is considered high correlation (default 0.85)")
    p.add_argument("--keep-by", default="best_auc",
                   choices=["best_auc", "weighted_auc", "seq_length_asc"],
                   help="Strategy for which factor to keep during dedup (default best_auc)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Maximum sample frames (0=unlimited, default: read from validation.yaml run.max_samples)")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Number of parallel processes (default: read from validation.yaml run.num_workers)")
    p.add_argument("--num-chunks", type=int, default=None,
                   help="Number of window chunks (default: read from validation.yaml run.num_chunks)")
    p.add_argument("--purity-mode", default=None,
                   choices=["nan_boundary", "none", "strict"],
                   help="Boundary purity strategy (default: read from validation.yaml run.purity_mode)")
    p.add_argument("--seq", nargs="+", type=int, metavar="N",
                   help="Only analyze factors with specified seq_length (multiple allowed, default: all)")
    p.add_argument("--top-k", type=int, default=0, metavar="N",
                   help="Only select top N factors by AUC per group for analysis (0=unlimited)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip heatmap generation (save time)")
    p.add_argument("--output-factors", default="",
                   help="Deduplicated factor JSON output path (empty=don't save, e.g., memory/valid_factors_deduped.json)")
    args = p.parse_args()

    if args.config is not None:
        cfg = _load_merged_config(args.config)
        cfg_dir = Path(args.config).resolve().parent
    else:
        cfg = _load_merged_config(args.config_common, args.config_validation)
        cfg_dir = Path(args.config_common).resolve().parent

    run_cfg = cfg.get("run", {})
    factors_path = args.factors or run_cfg.get("factors", "memory/valid_factors.json")
    output_dir = args.output_dir or run_cfg.get("output_dir", "memory")
    max_samples = args.max_samples if args.max_samples is not None else run_cfg.get("max_samples", 0)
    num_workers = args.num_workers if args.num_workers is not None else run_cfg.get("num_workers", 8)
    num_chunks = args.num_chunks if args.num_chunks is not None else run_cfg.get("num_chunks", 256)
    purity_mode = args.purity_mode or run_cfg.get("purity_mode", "nan_boundary")

    log = _setup_logging()
    run_correlation_analysis(
        cfg=cfg,
        cfg_dir=cfg_dir,
        factors_path=factors_path,
        output_dir=output_dir,
        threshold=args.threshold,
        keep_by=args.keep_by,
        max_samples=max_samples,
        num_workers=num_workers,
        num_chunks=num_chunks,
        purity_mode=purity_mode,
        seq_filter=args.seq or [],
        top_k=args.top_k,
        no_plot=args.no_plot,
        output_factors_path=args.output_factors,
        log=log,
    )


if __name__ == "__main__":
    main()
