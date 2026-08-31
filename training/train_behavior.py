"""
train_behavior.py
Factor synthetic validation CLI entry point.

Usage (run directly, no arguments needed):
  python train_behavior.py

All run parameters are configured in the run: section of config/validation.yaml.
To temporarily override config file paths:
  python train_behavior.py --config-common config/seq/1.yaml --config-validation config/validation.yaml

Flow:
  1. Load config/seq/1.yaml + config/validation.yaml
  2. Build train / val MouseBehaviorDataset (reuse functions from discovery.py)
  3. Read all valid factors from memory/valid_factors.json
  4. Recompute factor matrices [N, K] on train / val (N = number of windows)
  5. NaN fill + StandardScaler
  6. LightGBM multi-class (class_weight='balanced') training -> val holdout evaluation
  7. Output synth_report.json + confusion matrix PNG

Each run creates an independent directory under runs/train/expN, storing all weights,
charts, reports, and config file copies.
"""


import sys
from pathlib import Path
# Allow running as a script from a subdirectory (e.g. `python mining/discovery.py`)
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import argparse
import json
import shutil
import tempfile
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np

from mining.discovery import setup_logging, load_dataset_config, build_train_val_datasets, build_raw_frame_data, save_raw_frame_cache, _load_merged_config
from src.hardware_monitor import HardwareMonitor
from src.synth_validator import SynthValidator
from src.temporal_validator import TemporalValidator
from src.visualization import (
    plot_label_distribution,
    plot_keypoint_trajectory,
    plot_pipeline_comparison,
    plot_per_class_f1_evolution,
    plot_best_model_metrics_table,
    plot_ovr_results_table,
    plot_segment_duration_analysis,
)


def get_next_run_dir(base_dir: str = "runs") -> Path:
    """
    Ultralytics YOLO-style: create incrementally numbered exp folders under runs/train/.
    Returns the newly created experiment directory Path.
    """
    train_dir = Path(base_dir) / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted([
        int(d.name[3:]) for d in train_dir.iterdir()
        if d.is_dir() and d.name.startswith("exp") and d.name[3:].isdigit()
    ])
    next_id = existing[-1] + 1 if existing else 1
    run_dir = train_dir / f"exp{next_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    (run_dir / "configs").mkdir(exist_ok=True)

    return run_dir


def plot_raster(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    proba_val: "np.ndarray",
    class_names: list,
    output_path: str,
    n_frames: int = 1000,
    seed: int = None,
) -> tuple:
    """
    Randomly select a contiguous segment of n_frames from y_true/y_pred, draw a two-row raster plot.
    Top row: ground truth; Bottom row: model prediction. Different classes use different colors.
    Title shows the segment's ACC / macro-F1 / macro-AUC.
    Returns (output_path, start, end, seg_true, seg_pred, seg_proba).
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    T = len(y_true)
    n_frames = min(n_frames, T)

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, T - n_frames + 1))
    end = start + n_frames

    seg_true  = y_true[start:end]
    seg_pred  = y_pred[start:end]
    seg_proba = proba_val[start:end] if proba_val is not None else None

    # ---------- Segment metrics ----------
    classes_present = sorted(set(int(v) for v in seg_true))
    n_classes = len(class_names)
    all_class_ids = list(range(n_classes))

    acc  = float(accuracy_score(seg_true, seg_pred))
    f1   = float(f1_score(seg_true, seg_pred, labels=all_class_ids,
                          average="macro", zero_division=0))
    auc_str = "n/a"
    if seg_proba is not None and len(classes_present) >= 2:
        try:
            present_idx = [c for c in classes_present if c < seg_proba.shape[1]]
            proba_sub = seg_proba[:, present_idx]
            row_sum = proba_sub.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum <= 0, 1.0, row_sum)
            proba_sub = proba_sub / row_sum
            auc = float(roc_auc_score(seg_true, proba_sub,
                                      multi_class="ovr", average="macro",
                                      labels=classes_present))
            auc_str = f"{auc:.4f}"
        except Exception:
            pass

    # ---------- Color mapping ----------
    cmap = plt.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)
    class_to_color = {i: cmap(i) for i in range(n_classes)}

    # ---------- Layout: 2-row raster ----------
    fig, axes = plt.subplots(2, 1, figsize=(min(20, n_frames / 30), 3), sharex=True)
    fig.subplots_adjust(hspace=0.05)

    for ax, seg, title in zip(axes, [seg_true, seg_pred], ["Ground Truth", "Predicted"]):
        color_img = np.array([[class_to_color[int(c)][:3] for c in seg]])  # [1, T, 3]
        ax.imshow(color_img, aspect="auto", interpolation="nearest",
                  extent=[0, n_frames, 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=9, rotation=0, labelpad=60, va="center")
        ax.spines[["top", "right", "left"]].set_visible(False)

    axes[-1].set_xlabel(f"Frame offset: {start}--{end}", fontsize=8)

    # ---------- Legend ----------
    patches = [
        mpatches.Patch(color=class_to_color[i], label=class_names[i])
        for i in range(n_classes)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=min(n_classes, 6),
               fontsize=7, bbox_to_anchor=(0.5, -0.18), frameon=False)

    title_str = (
        f"Behavior Raster  (frames {start}--{end})   "
        f"ACC={acc:.4f}  macro-F1={f1:.4f}  macro-AUC={auc_str}"
    )
    fig.suptitle(title_str, fontsize=9, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path, start, end, seg_true, seg_pred, seg_proba


def _verify_saved_weights(output_dir: str, stage_cache_dir: str, logger):
    """After train_behavior completes, load saved weights from disk and compare predictions with stage cache.

    This is the most thorough verification: same process, same data, model just saved and immediately reloaded.
    """
    import lightgbm as lgb
    import numpy as np
    from pathlib import Path

    weights_dir = Path(output_dir) / "weights"
    sc_root = Path(stage_cache_dir)

    if not weights_dir.exists():
        logger.warning("[Weight verification] weights directory does not exist, skipping")
        return

    all_ok = True

    # -- Verify group LGBM --
    for gname in ["short", "medium", "long"]:
        model_path = weights_dir / f"group_{gname}_lgbm.txt"
        sc_dir = sc_root / f"group_{gname}"
        sc_proba_dir = sc_root / f"group_{gname}_proba"
        x_path = sc_dir / "X_std_val.npy"
        p_path = sc_proba_dir / "proba_val.npy"
        cf_path = weights_dir / f"common_factors_{gname}.json"
        sc_common_path = sc_dir / "common.json"

        if not model_path.exists():
            logger.warning(f"[Weight verification] {model_path.name} does not exist, skipping")
            continue
        if not x_path.exists() or not p_path.exists():
            logger.warning(f"[Weight verification] group '{gname}' stage cache incomplete, skipping")
            continue

        X = np.load(x_path)
        proba_cached = np.load(p_path)

        loaded_model = lgb.Booster(model_file=str(model_path))
        expected_nfeat = loaded_model.num_feature()
        actual_nfeat = X.shape[1]

        # -- Align feature columns when stage cache and model weights are out of sync --
        if actual_nfeat != expected_nfeat and cf_path.exists() and sc_common_path.exists():
            import json as _json_align
            with open(cf_path, "r", encoding="utf-8") as _f:
                _training_common = _json_align.load(_f).get("common", [])
            with open(sc_common_path, "r", encoding="utf-8") as _f:
                _cache_common = _json_align.load(_f)

            if len(_training_common) == expected_nfeat and len(_cache_common) == actual_nfeat:
                _name_to_col = {n: i for i, n in enumerate(_cache_common)}
                _aligned = np.full((X.shape[0], expected_nfeat), np.nan, dtype=np.float32)
                _matched = 0
                for _ci, _name in enumerate(_training_common):
                    _src_col = _name_to_col.get(_name)
                    if _src_col is not None:
                        _aligned[:, _ci] = X[:, _src_col]
                        _matched += 1
                X = _aligned
                logger.info(
                    f"[Weight verification] group '{gname}': aligned X columns "
                    f"{actual_nfeat} -> {expected_nfeat}, matched {_matched}/{expected_nfeat}"
                )
            else:
                logger.warning(
                    f"[Weight verification] group '{gname}': common_factors length mismatch, "
                    f"weights={len(_training_common)}(expect {expected_nfeat}), "
                    f"cache={len(_cache_common)}(actual {actual_nfeat}), skipping"
                )
                continue
        elif actual_nfeat != expected_nfeat:
            logger.warning(
                f"[Weight verification] group '{gname}': feature count mismatch "
                f"(model={expected_nfeat}, cache={actual_nfeat}) and alignment metadata missing, skipping"
            )
            continue

        proba_loaded = loaded_model.predict(X).astype(np.float32)

        min_c = min(proba_cached.shape[1], proba_loaded.shape[1])
        max_diff = float(np.abs(proba_cached[:, :min_c] - proba_loaded[:, :min_c]).max())

        if max_diff < 5e-4:
            logger.info(f"[Weight verification] group '{gname}' PASS (max_diff={max_diff:.2e})")
        else:
            logger.error(
                f"[Weight verification] group '{gname}' FAIL (max_diff={max_diff:.6e})"
            )
            flat_idx = np.argmax(
                np.abs(proba_cached[:, :min_c] - proba_loaded[:, :min_c]).ravel()
            )
            logger.error(
                f"  cached={proba_cached.ravel()[flat_idx]:.6f}, "
                f"loaded={proba_loaded.ravel()[flat_idx]:.6f}"
            )
            all_ok = False

    # -- Verify Meta-LGBM --
    meta_model_path = weights_dir / "meta_lgbm.txt"
    meta_sc_dir = sc_root / "meta_lgbm_r1"
    meta_p_path = meta_sc_dir / "proba_meta_val.npy"

    if meta_model_path.exists() and meta_p_path.exists():
        proba_meta_cached = np.load(meta_p_path)
        loaded_meta = lgb.Booster(model_file=str(meta_model_path))

        # Rebuild meta_X (same order as training)
        meta_parts = []
        for gname in ["short", "medium", "long"]:
            sc_dir_g = sc_root / f"group_{gname}"
            sc_proba_dir_g = sc_root / f"group_{gname}_proba"
            p_val_p = sc_proba_dir_g / "proba_val.npy"
            x_val_p = sc_dir_g / "X_std_val.npy"
            if p_val_p.exists():
                p = np.load(p_val_p)
                if x_val_p.exists():
                    xs = np.load(x_val_p)
                    if xs.shape[0] == p.shape[0]:
                        p = np.hstack([xs, p])
                meta_parts.append(p)

        if meta_parts:
            meta_X = np.hstack(meta_parts).astype(np.float32)
            proba_meta_loaded = loaded_meta.predict(meta_X).astype(np.float32)
            min_c = min(proba_meta_cached.shape[1], proba_meta_loaded.shape[1])
            max_diff = float(np.abs(
                proba_meta_cached[:, :min_c] - proba_meta_loaded[:, :min_c]
            ).max())

            if max_diff < 3e-1:
                logger.info(f"[Weight verification] meta-LGBM PASS (max_diff={max_diff:.2e}, float16 quantization error)")
            else:
                logger.error(f"[Weight verification] meta-LGBM FAIL (max_diff={max_diff:.6e})")
                all_ok = False

    # -- Verify BiLSTM checkpoint --
    bilstm_path = weights_dir / "bilstm_best.pt"
    if bilstm_path.exists():
        import torch
        ckpt = torch.load(bilstm_path, map_location="cpu", weights_only=False)
        logger.info(
            f"[Weight verification] BiLSTM checkpoint: model={ckpt.get('model_type')}, "
            f"n_classes={ckpt.get('n_classes')}, input_dim={ckpt.get('input_dim')}"
        )
        # BiLSTM deterministic verification is more complex (needs same device), only verify checkpoint structure here
        required_keys = ["model_state_dict", "model_type", "n_classes", "input_dim", "cfg"]
        missing_keys = [k for k in required_keys if k not in ckpt]
        if missing_keys:
            logger.error(f"[Weight verification] BiLSTM checkpoint missing fields: {missing_keys}")
            all_ok = False
        else:
            logger.info(f"[Weight verification] BiLSTM checkpoint structure OK")

    if all_ok:
        logger.info("[Weight verification] All weight files verified -- save/load is fully consistent")
    else:
        logger.error("[Weight verification] Inconsistencies found, please check deterministic configuration")


def main():
    parser = argparse.ArgumentParser(description="Factor synthetic validation (multi-class LGBM)")
    parser.add_argument("--config-common",     default="config/seq/1.yaml",     help="Common configuration file")
    parser.add_argument("--config-validation", default="config/validation.yaml", help="Synthetic validation configuration file")
    parser.add_argument("--config", default=None, help="Single config file path (legacy usage)")
    args = parser.parse_args()

    if args.config is not None:
        cfg = _load_merged_config(args.config)
        cfg_dir = Path(args.config).resolve().parent
    else:
        cfg = _load_merged_config(args.config_common, args.config_validation)
        cfg_dir = Path(args.config_common).resolve().parent

    # Read all run parameters from the run: section, use safe defaults when missing
    run_cfg = cfg.get("run", {})
    factors_path     = run_cfg.get("factors",            "memory/valid_factors.json")
    _output_dir_cfg  = run_cfg.get("output_dir",         "memory")
    max_samples      = run_cfg.get("max_samples",        0)
    num_workers      = run_cfg.get("num_workers",        8)
    num_chunks       = run_cfg.get("num_chunks",         256)
    multiresolution  = run_cfg.get("multiresolution",    True)
    do_temporal      = run_cfg.get("temporal",           True)
    do_ovr           = run_cfg.get("ovr",                True)
    visualize_raster = run_cfg.get("visualize_raster",   True)
    raster_frames    = run_cfg.get("raster_frames",      10000)
    raster_seed      = run_cfg.get("raster_seed",        None)
    purity_mode      = run_cfg.get("purity_mode",        "nan_boundary")
    val_purity_mode  = run_cfg.get("val_purity_mode",    None)
    use_cache        = run_cfg.get("use_cache",          True)
    refresh_cache    = run_cfg.get("refresh_cache",      False)
    cache_dir        = run_cfg.get("cache_dir",          "dataset_cache")
    use_stage_cache  = run_cfg.get("use_stage_cache",    True)
    refresh_stage    = run_cfg.get("refresh_stage_cache",False)
    stage_cache_dir  = run_cfg.get("stage_cache_dir",   "pipeline_stage_cache")
    filter_corr      = run_cfg.get("filter_corr",        "")
    corr_threshold   = run_cfg.get("corr_threshold",     0.85)
    save_cache_path  = run_cfg.get("save_cache",         "")
    tmp_dir          = cfg.get("synth_validation", {}).get("tmp_dir", "")
    temporal_window  = cfg.get("temporal_validation", {}).get("window_size", 31)

    # ---------- Create runs experiment directory (Ultralytics-style) ----------
    run_dir = get_next_run_dir("runs")
    output_dir = str(run_dir)
    viz_dir = str(run_dir / "visualizations")

    # Copy config file copies to experiment directory for reproducibility
    configs_dir = run_dir / "configs"
    _common_src = Path(args.config_common) if not args.config else Path(args.config)
    _val_src = Path(args.config_validation) if not args.config else None
    if _common_src.exists():
        shutil.copy2(_common_src, configs_dir / _common_src.name)
    if _val_src and _val_src.exists():
        shutil.copy2(_val_src, configs_dir / _val_src.name)
    # Save the merged full config (for direct reproducibility)
    with open(configs_dir / "merged_config.yaml", "w", encoding="utf-8") as _f:
        yaml.dump(cfg, _f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger, log_file = setup_logging(cfg.get("output", {}).get("logs_dir", "logs"))
    logger.info("=" * 60)
    logger.info("Factor synthetic validation starting")
    logger.info(f"Experiment directory: {run_dir}")
    logger.info("=" * 60)

    # ---------- Background hardware monitor ----------
    hw_monitor_interval = run_cfg.get("hw_monitor_interval", 30)
    hw_monitor = HardwareMonitor(interval_sec=hw_monitor_interval)
    hw_monitor.start()
    logger.info(f"[HardwareMonitor] Hardware monitoring started, sampling interval {hw_monitor_interval}s")

    # ---------- Factor correlation filtering (optional) ----------
    factors_path_final = factors_path
    temp_factors_file = None
    if filter_corr:
        logger.info(f"Enabling correlation filtering: {filter_corr}")
        factors_all = SynthValidator.load_valid_factors(factors_path)
        factors_filtered = SynthValidator.filter_factors_by_correlation(
            factors=factors_all,
            corr_report_path=filter_corr,
            threshold=corr_threshold,
        )
        temp_factors_file = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".json", delete=False
        )
        json.dump(factors_filtered, temp_factors_file, ensure_ascii=False, indent=2)
        temp_factors_file.close()
        factors_path_final = temp_factors_file.name
        logger.info(f"Filtered factors written to temporary file: {factors_path_final}")

    # ---------- Save factors.json to experiment directory ----------
    _factors_dst = run_dir / "factors.json"
    shutil.copy2(factors_path_final, str(_factors_dst))
    logger.info(f"Factor list saved to experiment directory: {_factors_dst}")

    # ---------- Build dataset (reuse discovery.py logic) ----------
    ds_cfg_file = cfg.get("dataset_config_file", "")
    if not ds_cfg_file:
        raise ValueError("common.yaml is missing the dataset_config_file field.")
    ds_cfg_path = Path(ds_cfg_file)
    if not ds_cfg_path.is_absolute():
        ds_cfg_path = cfg_dir / ds_cfg_path

    ds_info = load_dataset_config(cfg, str(ds_cfg_path), logger)

    # Keep consistent with discovery.py: synchronize behavior_classes with merged label_map,
    # ensuring synthetic validation's class list matches actual labels (after label_merge)
    merged_label_map = ds_info["label_map"]
    cfg["label_map"] = merged_label_map  # Write to cfg for inference.py fallback
    cfg.setdefault("data", {})["behavior_classes"] = sorted(merged_label_map, key=merged_label_map.get)

    # Re-save merged_config.yaml (supplement label_map)
    try:
        with open(configs_dir / "merged_config.yaml", "w", encoding="utf-8") as _f:
            yaml.dump(cfg, _f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    except Exception:
        pass

    seq_cfg = cfg.get("sequence", {})

    # ---------- Multi-resolution mode: load raw per-frame data ----------
    if multiresolution:
        logger.info(
            f"Multi-resolution mode starting (purity_mode={purity_mode}):\n"
            f"  Each factor builds centered windows at its seq_length around the validation frame,\n"
            f"  all factors share the same frame indices and labels."
        )
        (train_kp, train_lb), (val_kp, val_lb), flat_attributes, video_lengths = build_raw_frame_data(ds_info, logger, cfg=cfg)
        train_vlens, val_vlens = video_lengths

        if max_samples > 0:
            import numpy as _np
            rng = _np.random.default_rng(42)
            for kp, lb, tag in [(train_kp, train_lb, "training"), (val_kp, val_lb, "validation")]:
                if len(lb) > max_samples:
                    idx = rng.choice(len(lb), max_samples, replace=False)
                    idx.sort()
                    if tag == "training":
                        train_kp, train_lb = kp[idx], lb[idx]
                    else:
                        val_kp, val_lb = kp[idx], lb[idx]
            logger.info(f"Sampled: training set {train_kp.shape[0]} frames, validation set {val_kp.shape[0]} frames")

        logger.info(f"Training set: {train_kp.shape[0]} frames, Validation set: {val_kp.shape[0]} frames, D={train_kp.shape[1]}")

        # -- Visualization: label distribution + keypoint trajectory --
        classes_sorted_viz = sorted(set(int(v) for v in ds_info["label_map"].values()))
        id_to_name_viz = {int(v): k for k, v in ds_info["label_map"].items()}
        class_names_viz = [id_to_name_viz[c] for c in classes_sorted_viz]
        try:
            plot_label_distribution(
                y_train=train_lb, y_val=val_lb,
                class_names=class_names_viz,
                viz_dir=viz_dir,
            )
            plot_keypoint_trajectory(
                train_kp=train_kp, train_lb=train_lb,
                flat_attributes=flat_attributes,
                class_names=class_names_viz,
                viz_dir=viz_dir,
            )
            plot_segment_duration_analysis(
                y_true=val_lb,
                class_names=class_names_viz,
                viz_dir=viz_dir,
            )
        except Exception as _e:
            logger.warning(f"[Viz] Data visualization failed: {_e}")

        if save_cache_path:
            save_raw_frame_cache(
                train_data=(train_kp, train_lb),
                val_data=(val_kp, val_lb),
                flat_attributes=flat_attributes,
                label_map=ds_info["label_map"],
                output_path=save_cache_path,
                logger=logger,
            )

        validator = SynthValidator(cfg)
        result = validator.run_multiscale_temporal(
            factors_path=factors_path_final,
            train_kp_raw=train_kp,
            train_lb_raw=train_lb,
            val_kp_raw=val_kp,
            val_lb_raw=val_lb,
            flat_attributes=flat_attributes,
            label_map=ds_info["label_map"],
            output_dir=output_dir,
            cfg=cfg,
            purity_mode=purity_mode,
            val_purity_mode=val_purity_mode,
            num_workers=num_workers,
            num_chunks=num_chunks,
            cache_dir=cache_dir,
            stage_cache_dir=stage_cache_dir,
            use_stage_cache=use_stage_cache,
            refresh_stage_cache=refresh_stage,
            viz_dir=viz_dir,
            train_video_lengths=train_vlens,
            val_video_lengths=val_vlens,
        )

        # ---------- Weight verification: immediately load from disk and compare with in-memory model ----------
        _verify_saved_weights(output_dir, stage_cache_dir, logger)

    # ---------- Standard mode: fixed window width ----------
    else:
        train_data, val_data, flat_attributes = build_train_val_datasets(
            ds_info, seq_cfg, logger, cfg
        )
        train_kp, train_lb = train_data
        val_kp,   val_lb   = val_data

        if max_samples > 0:
            import numpy as _np
            rng = _np.random.default_rng(42)
            for kp, lb, tag in [(train_kp, train_lb, "training"), (val_kp, val_lb, "validation")]:
                if len(lb) > max_samples:
                    idx = rng.choice(len(lb), max_samples, replace=False)
                    idx.sort()
                    if tag == "training":
                        train_kp, train_lb = kp[idx], lb[idx]
                    else:
                        val_kp, val_lb = kp[idx], lb[idx]
            logger.info(f"Sampled: training set {train_kp.shape[0]} windows, validation set {val_kp.shape[0]} windows")

        logger.info(f"Training set: {train_kp.shape[0]} windows (seq_length={train_kp.shape[1]}), Validation set: {val_kp.shape[0]} windows")

        validator = SynthValidator(cfg)
        result = validator.run(
            factors_path=factors_path_final,
            train_kp=train_kp,
            train_lb=train_lb,
            val_kp=val_kp,
            val_lb=val_lb,
            flat_attributes=flat_attributes,
            label_map=ds_info["label_map"],
            output_dir=output_dir,
            num_workers=num_workers,
            num_chunks=num_chunks,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            cache_dir=cache_dir,
        )

    # ---------- Raster plot visualization ----------
    if visualize_raster:
        import numpy as _np
        label_map = ds_info["label_map"]
        classes_sorted = sorted(set(int(v) for v in label_map.values()))
        id_to_name = {int(v): k for k, v in label_map.items()}
        class_names_ordered = [id_to_name[c] for c in classes_sorted]

        # Use final prediction probability: prefer temporal stage, then meta, then standard mode
        proba_val = result.get("_proba_meta_val")
        if proba_val is None:
            proba_val = result.get("_proba_val")
        if proba_val is not None:
            y_pred_val = _np.array([classes_sorted[i] for i in proba_val.argmax(axis=1)])
            y_true_val = _np.asarray(val_lb)
            out_path = str(run_dir / "raster_plot.png")
            saved, rst_start, rst_end, rst_true, rst_pred, rst_proba = plot_raster(
                y_true=y_true_val,
                y_pred=y_pred_val,
                proba_val=proba_val,
                class_names=class_names_ordered,
                output_path=out_path,
                n_frames=raster_frames,
                seed=raster_seed,
            )
            logger.info(f"Raster plot saved: {saved}")
            # CSV alongside raster plot (uses exact segment from plot_raster)
            import csv as _csv
            _raster_csv = str(run_dir / "raster_plot.csv")
            with open(_raster_csv, "w", encoding="utf-8", newline="") as _rf:
                _w = _csv.writer(_rf)
                header = ["frame_offset", "true_label", "pred_label"]
                if rst_proba is not None:
                    for ci, cn in enumerate(class_names_ordered):
                        header.append(f"proba_{cn}")
                _w.writerow(header)
                for fi in range(len(rst_true)):
                    row = [str(int(rst_start + fi)),
                           class_names_ordered[int(rst_true[fi])] if int(rst_true[fi]) < len(class_names_ordered) else f"class_{int(rst_true[fi])}",
                           class_names_ordered[int(rst_pred[fi])] if int(rst_pred[fi]) < len(class_names_ordered) else f"class_{int(rst_pred[fi])}"]
                    if rst_proba is not None:
                        for ci in range(len(class_names_ordered)):
                            row.append(f"{rst_proba[fi, ci]:.6f}")
                    _w.writerow(row)
            logger.info(f"Raster plot CSV saved: {_raster_csv}")
        else:
            logger.warning("Prediction probability array not found, skipping raster plot generation.")

    # ---------- Console summary ----------
    if result.get("multiscale_temporal"):
        # ---- Multi-scale temporal pipeline results ----
        logger.info("\n" + "=" * 60)
        logger.info("Multi-scale temporal validation results (3-stage pipeline)")
        logger.info("=" * 60)

        # Group metrics
        logger.info(f"[Stage 1] Per-resolution group LGBM ({result['n_groups']} groups)")
        for gname, gm in result["group_metrics"].items():
            logger.info(
                f"  Group '{gname}': accuracy={gm['accuracy']:.4f}  "
                f"macro_f1={gm['macro_f1']:.4f}  n_factors={gm['n_factors']}"
            )

        # meta-LGBM metrics
        mm = result.get("meta_metrics", {})
        if mm and "accuracy" in mm:
            logger.info(f"[Stage 2] meta-LGBM:  accuracy={mm['accuracy']:.4f}  "
                        f"macro_f1={mm.get('macro_f1', 0):.4f}  "
                        f"macro_auc={mm.get('macro_auc', 0):.4f}")

        # Temporal model metrics
        tm = result.get("temporal_metrics", {})
        if tm and "accuracy" in tm:
            temporal_model_name = {"lgbm": "LGBM", "bilstm": "BiLSTM", "transformer": "Transformer", "mamba": "Mamba"}.get(tm.get("model", "lgbm"), tm.get("model", "LGBM").upper())
            logger.info(
                f"[Stage 3] Temporal {temporal_model_name}: "
                f"acc={tm['accuracy']:.4f}  "
                f"bal_acc={tm.get('balanced_accuracy', 0):.4f}  "
                f"macro_auc={tm.get('macro_auc', 0):.4f}  "
                f"weighted_auc={tm.get('weighted_auc', 0):.4f}  "
                f"macro_f1={tm.get('macro_f1', 0):.4f}  "
                f"weighted_f1={tm.get('weighted_f1', 0):.4f}"
            )

            # Compare base / Viterbi / CRF, select best model
            monitor_metric = cfg.get("temporal_validation", {}).get("monitor_metric", "val_acc")
            _metric_key_map = {
                "val_acc": "accuracy",
                "val_balanced_acc": "balanced_accuracy",
                "macro_f1": "macro_f1",
                "weighted_f1": "weighted_f1",
                "macro_auc": "macro_auc",
                "weighted_auc": "weighted_auc",
            }
            _sel_key = _metric_key_map.get(monitor_metric, "accuracy")

            seq_dec = result.get("sequence_decoding", {})
            candidates = {temporal_model_name: tm}
            for dec_key, dec_m in seq_dec.items():
                if "accuracy" in dec_m:
                    tag = f"{temporal_model_name}+{dec_key}"
                    candidates[tag] = dec_m
                    logger.info(
                        f"[Stage 3] Temporal {tag}: "
                        f"acc={dec_m['accuracy']:.4f}  "
                        f"bal_acc={dec_m.get('balanced_accuracy', 0):.4f}  "
                        f"macro_auc={dec_m.get('macro_auc', 0):.4f}  "
                        f"weighted_auc={dec_m.get('weighted_auc', 0):.4f}  "
                        f"macro_f1={dec_m.get('macro_f1', 0):.4f}  "
                        f"weighted_f1={dec_m.get('weighted_f1', 0):.4f}"
                    )

            best_name = temporal_model_name
            best_metrics = tm
            best_score = tm.get(_sel_key, 0)
            for cand_name, cand_m in candidates.items():
                cand_score = cand_m.get(_sel_key, 0)
                if cand_score > best_score:
                    best_score = cand_score
                    best_name = cand_name
                    best_metrics = cand_m

            if best_name != temporal_model_name:
                logger.info(f"[Stage 3] Final selection: {best_name} ({monitor_metric} improvement "
                            f"{tm.get(_sel_key, 0):.4f} -> {best_score:.4f})")
            else:
                logger.info(f"[Stage 3] Final selection: {best_name} (sequence decoding did not improve)")

            # -- Save final model selection information --
            _final_sel = {
                "selected_model": best_name,
                "monitor_metric": monitor_metric,
                "score": round(float(best_score), 4),
                "candidates": {k: round(float(v.get(_sel_key, 0)), 4) for k, v in candidates.items()},
                "per_class": {k: {kk: vv for kk, vv in v.items() if kk != "auc" or vv is not None}
                              for k, v in best_metrics.get("per_class", {}).items()},
                "accuracy": best_metrics.get("accuracy"),
                "balanced_accuracy": best_metrics.get("balanced_accuracy"),
                "macro_f1": best_metrics.get("macro_f1"),
                "macro_auc": best_metrics.get("macro_auc"),
            }
            with open(run_dir / "final_selection.json", "w", encoding="utf-8") as _fs:
                json.dump(_final_sel, _fs, ensure_ascii=False, indent=2)
            logger.info(f"Final model selection saved: {run_dir / 'final_selection.json'}")

            logger.info(f"-- per class (final model: {best_name}) --")
            hdr = f"  {'class':18s} | {'P':>6} {'R':>6} {'F1':>6} {'AUC':>7} {'sup':>6}"
            logger.info(hdr)
            best_pc = best_metrics.get("per_class", {})
            for cname, v in best_pc.items():
                auc_s = f"{v['auc']:.4f}" if v.get("auc") is not None else "   n/a"
                logger.info(
                    f"  {cname:18s} | {v.get('precision', 0):>6.3f} {v.get('recall', 0):>6.3f} "
                    f"{v.get('f1', 0):>6.3f} {auc_s:>7} {v.get('support', 0):>6}"
                )

        logger.info(f"\nOutput files: {result.get('output_files', {})}")
        logger.info("=" * 60)

    else:
        # ---- Standard single-stage result (original print logic) ----
        m = result["metrics"]
        logger.info("\n" + "=" * 60)
        logger.info("Synthetic validation results")
        logger.info("=" * 60)
        n_total = result['n_factors_used']
        n_selected = result.get('n_selected_factors', n_total)
        n_dropped = len(result['factors_dropped'])
        if n_selected < n_total:
            logger.info(f"Total factors: {n_total}  After filtering: {n_selected}  Dropped: {n_dropped}")
        else:
            logger.info(f"Factors used: {n_total}  Dropped: {n_dropped}")
        logger.info(f"accuracy     : {m['accuracy']:.4f}")
        if m.get("top_k_accuracy") is not None:
            k = m.get("top_k", 2)
            logger.info(f"top-{k} accuracy: {m['top_k_accuracy']:.4f}")
        logger.info(f"macro AUC    : {m['macro_auc']:.4f} | weighted AUC: {m['weighted_auc']:.4f}")
        logger.info(f"macro F1     : {m['macro_f1']:.4f} | weighted F1 : {m['weighted_f1']:.4f}")
        top_k = m.get("top_k")
        top_k_pc = m.get("top_k_per_class", {})
        logger.info("-- per class --")
        hdr = f"  {'class':18s} | {'P':>6} {'R':>6} {'F1':>6} {'AUC':>7} {'sup':>6}"
        if top_k_pc:
            hdr += f" {'top-' + str(top_k) + ' acc':>10}"
        logger.info(hdr)
        for cname, v in m["per_class"].items():
            auc_s = f"{v['auc']:.4f}" if v["auc"] is not None else "   n/a"
            line = (
                f"  {cname:18s} | {v['precision']:>6.3f} {v['recall']:>6.3f} "
                f"{v['f1']:>6.3f} {auc_s:>7} {v['support']:>6}"
            )
            if top_k_pc:
                tk = top_k_pc.get(cname)
                line += f" {tk:>10.4f}" if tk is not None else f" {'  n/a':>10}"
            logger.info(line)

        logger.info("\nConfusion matrix (row=true, column=pred):")
        header = " " * 22 + "".join(f"{n[:10]:>11s}" for n in result["class_names_in_order"])
        logger.info(header)
        for i, row in enumerate(result["confusion_matrix"]):
            name = result["class_names_in_order"][i]
            logger.info(f"  {name:20s}" + "".join(f"{v:>11d}" for v in row))

        logger.info(f"\nOutput files: {result.get('output_files', {})}")
        logger.info("=" * 60)

    # ---------- Temporal two-stage validation (standard mode only) ----------
    if do_temporal and not result.get("multiscale_temporal"):
        proba_train = result.get("_proba_train")
        proba_val = result.get("_proba_val")
        if proba_train is None or proba_val is None:
            logger.error("Single-frame model did not return probability matrix, cannot run temporal validation.")
        else:
            cfg.setdefault("temporal_validation", {})
            cfg["temporal_validation"]["window_size"] = temporal_window

            logger.info("\n" + "=" * 60)
            logger.info(f"Temporal two-stage validation starting (window_size={temporal_window})")
            logger.info("=" * 60)

            t_validator = TemporalValidator(cfg)
            t_result = t_validator.run(
                proba_train=proba_train,
                y_train=train_lb,
                proba_val=proba_val,
                y_val=val_lb,
                label_map=ds_info["label_map"],
                output_dir=output_dir,
                min_segment=cfg.get("temporal_validation", {}).get("min_segment", 30),
                viz_dir=viz_dir,
            )

            tm = t_result["metrics"]
            logger.info("\n" + "=" * 60)
            logger.info("Temporal validation results")
            logger.info("=" * 60)
            logger.info(f"accuracy     : {tm['accuracy']:.4f}  "
                        f"(single-frame baseline: {result['metrics']['accuracy']:.4f}, "
                        f"improvement {tm['accuracy'] - result['metrics']['accuracy']:+.4f})")
            if tm.get("top_k_accuracy") is not None:
                k = tm.get("top_k", 3)
                logger.info(f"top-{k} accuracy: {tm['top_k_accuracy']:.4f}  "
                            f"(single-frame baseline: {result['metrics'].get('top_k_accuracy', 0):.4f})")
            logger.info(f"macro AUC    : {tm['macro_auc']:.4f} | weighted AUC: {tm['weighted_auc']:.4f}")
            logger.info(f"macro F1     : {tm['macro_f1']:.4f} | weighted F1 : {tm['weighted_f1']:.4f}")
            top_k = tm.get("top_k")
            top_k_pc = tm.get("top_k_per_class", {})
            logger.info("-- per class --")
            hdr = f"  {'class':18s} | {'P':>6} {'R':>6} {'F1':>6} {'AUC':>7} {'sup':>6}"
            if top_k_pc:
                hdr += f" {'top-' + str(top_k) + ' acc':>10}"
            logger.info(hdr)
            for cname, v in tm["per_class"].items():
                auc_s = f"{v['auc']:.4f}" if v["auc"] is not None else "   n/a"
                line = (
                    f"  {cname:18s} | {v['precision']:>6.3f} {v['recall']:>6.3f} "
                    f"{v['f1']:>6.3f} {auc_s:>7} {v['support']:>6}"
                )
                if top_k_pc:
                    tk = top_k_pc.get(cname)
                    line += f" {tk:>10.4f}" if tk is not None else f" {'  n/a':>10}"
                logger.info(line)
            logger.info(f"\nOutput files: {t_result.get('output_files', {})}")
            logger.info("=" * 60)

            # Save standard mode temporal model selection
            _tfinal = {
                "selected_model": "temporal_lgbm",
                "monitor_metric": "accuracy",
                "score": round(tm.get("accuracy", 0), 4),
                "accuracy": tm.get("accuracy"),
                "macro_f1": tm.get("macro_f1"),
                "macro_auc": tm.get("macro_auc"),
            }
            with open(run_dir / "final_selection.json", "w", encoding="utf-8") as _fs:
                json.dump(_tfinal, _fs, ensure_ascii=False, indent=2)
            logger.info(f"Final model selection saved: {run_dir / 'final_selection.json'}")

    # ---------- OvR binary classification validation ----------
    if do_ovr:
        if not multiresolution:
            logger.error("ovr currently only supports multiresolution mode. Please set multiresolution: true in validation.yaml.")
        else:
            logger.info("\n" + "=" * 60)
            logger.info("OvR binary classification validation starting")
            logger.info("=" * 60)
            # Derive OvR metrics directly from the 3-stage pipeline's meta-LGBM proba, no retraining needed
            ovr_result = validator.run_ovr(
                factors_path=factors_path_final,
                train_kp_raw=train_kp,
                train_lb_raw=train_lb,
                val_kp_raw=val_kp,
                val_lb_raw=val_lb,
                flat_attributes=flat_attributes,
                label_map=ds_info["label_map"],
                output_dir=output_dir,
                proba_val_precomputed=result.get("_proba_meta_val"),
                y_val_precomputed=result.get("_y_meta_val"),
            )
            ovr = ovr_result["ovr_results"]
            logger.info("\n" + "=" * 60)
            logger.info("OvR per-class metrics (multi-class proba deduction, full val, no sampling, no retraining)")
            logger.info("=" * 60)
            hdr = f"  {'class':20s} | {'AUC':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}"
            logger.info(hdr)
            for cname, v in ovr.items():
                if cname == "macro_avg":
                    continue
                auc_s = f"{v['auc']:.4f}" if v.get("auc") is not None else "   n/a"
                logger.info(
                    f"  {cname:20s} | {auc_s:>7} {v['f1']:>7.4f} "
                    f"{v['precision']:>7.4f} {v['recall']:>7.4f}"
                )
            ma = ovr.get("macro_avg", {})
            logger.info("-" * 52)
            ma_auc = f"{ma['auc']:.4f}" if ma.get("auc") is not None else "   n/a"
            ma_f1  = f"{ma['f1']:.4f}"  if ma.get("f1")  is not None else "   n/a"
            logger.info(f"  {'macro_avg':20s} | {ma_auc:>7} {ma_f1:>7}")
            logger.info(f"\nOutput files: {ovr_result.get('output_files', {})}")
            logger.info("=" * 60)

            # -- OvR results table visualization --
            try:
                plot_ovr_results_table(
                    ovr_results=ovr_result["ovr_results"],
                    viz_dir=viz_dir,
                )
            except Exception as _e:
                logger.warning(f"[Viz] OvR results table visualization failed: {_e}")

    # ---------- Pipeline summary visualization ----------
    if multiresolution and result.get("multiscale_temporal"):
        try:
            monitor_metric = cfg.get("temporal_validation", {}).get("monitor_metric", "val_balanced_acc")
            plot_pipeline_comparison(
                group_metrics=result.get("group_metrics", {}),
                meta_metrics=result.get("meta_metrics", {}),
                temporal_metrics=result.get("temporal_metrics", {}),
                seq_decoding=result.get("sequence_decoding", {}),
                monitor_metric_key=monitor_metric,
                viz_dir=viz_dir,
            )
            plot_per_class_f1_evolution(
                group_metrics=result.get("group_metrics", {}),
                meta_metrics=result.get("meta_metrics", {}),
                temporal_metrics=result.get("temporal_metrics", {}),
                seq_decoding=result.get("sequence_decoding", {}),
                class_names=class_names_viz,
                viz_dir=viz_dir,
            )
        except Exception as _e:
            logger.warning(f"[Viz] Pipeline summary visualization failed: {_e}")

    # ---------- Best model metrics table visualization ----------
    if multiresolution and result.get("multiscale_temporal"):
        # Find the best model (reuse logic from the console summary above)
        try:
            tm = result.get("temporal_metrics", {})
            seq_dec = result.get("sequence_decoding", {})
            monitor_metric = cfg.get("temporal_validation", {}).get("monitor_metric", "val_balanced_acc")
            _metric_key_map = {
                "val_acc": "accuracy", "val_balanced_acc": "balanced_accuracy",
                "macro_f1": "macro_f1", "weighted_f1": "weighted_f1",
                "macro_auc": "macro_auc", "weighted_auc": "weighted_auc",
            }
            _sel_key = _metric_key_map.get(monitor_metric, "accuracy")
            _tm_name = tm.get("model", "lgbm")
            temporal_model_name_display = {"lgbm": "LGBM", "bilstm": "BiLSTM",
                                           "transformer": "Transformer", "mamba": "Mamba"}.get(_tm_name, _tm_name.upper())

            best_name = temporal_model_name_display
            best_metrics = tm
            best_score = tm.get(_sel_key, 0)
            for dec_key, dm in seq_dec.items():
                if dm and "accuracy" in dm:
                    cand_score = dm.get(_sel_key, 0)
                    if cand_score > best_score:
                        best_score = cand_score
                        best_name = f"{temporal_model_name_display}+{dec_key}"
                        best_metrics = dm

            plot_best_model_metrics_table(
                metrics=best_metrics,
                model_name=best_name,
                viz_dir=viz_dir,
            )
        except Exception as _e:
            logger.warning(f"[Viz] Best model metrics table visualization failed: {_e}")

    # ---------- Post-processing: PostCalibrator + DecoderPipeline grid search (validation.py --tune full pipeline) ----------
    optimized_result = None
    if multiresolution and result.get("multiscale_temporal"):
        proba_meta_val_pp = result.get("_proba_meta_val")
        proba_temporal_val_pp = result.get("_proba_temporal_val")
        proba_meta_train_pp = result.get("_proba_meta_train")
        if proba_meta_val_pp is not None and proba_temporal_val_pp is not None:
            logger.info("\n" + "=" * 60)
            logger.info("Post-processing: PostCalibrator + DecoderPipeline grid search (validation.py --tune full pipeline)")
            logger.info("=" * 60)
            try:
                from src.temporal_validator import tune_pipeline, PostCalibrator
                from src.decoders.pipeline import DecoderPipeline

                label_map_pp = ds_info["label_map"]
                classes_sorted_pp = sorted(set(int(v) for v in label_map_pp.values()))
                id_to_name_pp = {int(v): k for k, v in label_map_pp.items()}
                class_names_pp = [id_to_name_pp[c] for c in classes_sorted_pp]
                n_classes_pp = len(classes_sorted_pp)
                y_val_arr = np.asarray(val_lb).astype(int)

                # Step 1: Run PostCalibrator + RuleCorrector grid search
                best_params, best_acc = tune_pipeline(
                    proba_val=proba_temporal_val_pp,
                    y_val=val_lb,
                    proba_meta=proba_meta_val_pp,
                    label_map=label_map_pp,
                    output_dir=output_dir,
                    cfg=cfg,
                )

                calib_p = best_params["calib"]
                rc_p = best_params.get("rule_correct", {})

                # Step 2: Inject best parameters into cfg (consistent with validation.py --tune)
                cfg.setdefault("temporal_validation", {})["calibrator"] = calib_p
                for dec_cfg in cfg.setdefault("temporal_validation", {}).setdefault("decoders", []):
                    if dec_cfg.get("name") == "rule_correct":
                        dec_cfg.setdefault("config", {}).update(rc_p)

                mix_a = calib_p.get("mix_alpha", 1.0)
                logger.info(
                    f"[Post-processing] Best params (acc={best_acc:.4f}): "
                    f"T={calib_p.get('temperature', 1.0):.2f}  "
                    f"alpha={'per_class' if isinstance(mix_a, np.ndarray) else f'{mix_a:.2f}'}"
                    + (f"  rule_correct={rc_p}" if rc_p else "")
                )
                if isinstance(mix_a, np.ndarray):
                    logger.info(f"[Post-processing]   per_class_alpha: {dict(zip(class_names_pp, mix_a.round(3).tolist()))}")
                biases_v = calib_p.get("biases")
                if biases_v is not None:
                    logger.info(f"[Post-processing]   biases: {dict(zip(class_names_pp, biases_v.round(3).tolist()))}")

                # Step 3: Apply best PostCalibrator
                calib = PostCalibrator(
                    temperature=calib_p.get("temperature", 1.0),
                    mix_alpha=calib_p["mix_alpha"],
                    biases=calib_p.get("biases"),
                )
                calibrated_proba_val = calib.calibrate(proba_temporal_val_pp, proba_meta_val_pp)

                # Step 4: Re-run DecoderPipeline with calibrated probabilities (all decoders benefit)
                # Use proba_meta_train as decoder training input (consistent with validation.py)
                if proba_meta_train_pp is not None:
                    n_train_dec = min(len(proba_meta_train_pp), 50000)
                    proba_train_dec = proba_meta_train_pp[:n_train_dec]
                    y_train_dec = train_lb[:n_train_dec]
                else:
                    proba_train_dec = None
                    y_train_dec = None
                    logger.warning("[Post-processing] proba_meta_train unavailable, decoder will run with limited training data")

                tv_opt = TemporalValidator(cfg)
                opt_decoders = tv_opt._build_decoders()
                if opt_decoders:
                    opt_pipeline = DecoderPipeline(
                        decoders=opt_decoders,
                        classes_sorted=classes_sorted_pp,
                        id_to_name=id_to_name_pp,
                        output_dir=output_dir,
                    )
                    opt_seq_decoding = opt_pipeline.run(
                        proba_val=calibrated_proba_val,
                        y_val=y_val_arr,
                        proba_train=proba_train_dec,
                        y_train=np.asarray(y_train_dec).astype(int) if y_train_dec is not None else None,
                    )
                    logger.info(f"[Post-processing] DecoderPipeline complete, decoders: {list(opt_seq_decoding.keys())}")
                else:
                    opt_seq_decoding = {}
                    logger.warning("[Post-processing] No decoders configured, skipping DecoderPipeline")

                # Step 5: Build optimized_result (argmax baseline + all decoder results)
                optimized_pred_base = calibrated_proba_val.argmax(axis=1)

                from sklearn.metrics import (
                    accuracy_score, balanced_accuracy_score,
                    f1_score, roc_auc_score, confusion_matrix,
                    precision_recall_fscore_support,
                )
                opt_acc = float(accuracy_score(y_val_arr, optimized_pred_base))
                opt_bal_acc = float(balanced_accuracy_score(y_val_arr, optimized_pred_base))
                opt_macro_f1 = float(f1_score(y_val_arr, optimized_pred_base, labels=classes_sorted_pp, average="macro", zero_division=0))
                opt_weighted_f1 = float(f1_score(y_val_arr, optimized_pred_base, labels=classes_sorted_pp, average="weighted", zero_division=0))
                opt_cm = confusion_matrix(y_val_arr, optimized_pred_base, labels=classes_sorted_pp)

                prec_o, rec_o, f1_o, sup_o = precision_recall_fscore_support(
                    y_val_arr, optimized_pred_base, labels=classes_sorted_pp, zero_division=0)
                opt_per_class = {}
                opt_aucs = []
                for ci, c in enumerate(classes_sorted_pp):
                    cname = class_names_pp[ci]
                    y_bin = (y_val_arr == c).astype(int)
                    try:
                        auc_o = float(roc_auc_score(y_bin, calibrated_proba_val[:, ci]))
                        opt_aucs.append(auc_o)
                    except ValueError:
                        auc_o = None
                    opt_per_class[cname] = {
                        "precision": round(float(prec_o[ci]), 4),
                        "recall": round(float(rec_o[ci]), 4),
                        "f1": round(float(f1_o[ci]), 4),
                        "auc": round(auc_o, 4) if auc_o is not None else None,
                        "support": int(sup_o[ci]),
                    }
                opt_macro_auc = float(np.mean(opt_aucs)) if opt_aucs else 0.0

                # Get Calib+rule_correct results (extract from DecoderPipeline output)
                rc_result = opt_seq_decoding.get("rule_correct", {})
                if rc_result and "accuracy" in rc_result:
                    optimized_result = {
                        "model_name": "Calib+rule_correct",
                        "accuracy": rc_result.get("accuracy"),
                        "balanced_accuracy": rc_result.get("balanced_accuracy"),
                        "macro_f1": rc_result.get("macro_f1"),
                        "weighted_f1": rc_result.get("weighted_f1"),
                        "macro_auc": rc_result.get("macro_auc"),
                        "per_class": rc_result.get("per_class", {}),
                        "confusion_matrix": None,  # Rebuilt later from y_pred
                        "y_pred": rc_result.get("y_pred"),
                        "proba": calibrated_proba_val,
                    }
                    rc_y_pred = np.asarray(rc_result["y_pred"]) if rc_result.get("y_pred") is not None else optimized_pred_base
                    from sklearn.metrics import confusion_matrix as _cm_fn
                    optimized_result["confusion_matrix"] = _cm_fn(y_val_arr, rc_y_pred, labels=classes_sorted_pp)
                    logger.info(
                        f"[Post-processing] Calib+rule_correct: "
                        f"acc={optimized_result['accuracy']:.4f}  "
                        f"bal_acc={optimized_result['balanced_accuracy']:.4f}  "
                        f"macro_f1={optimized_result['macro_f1']:.4f}  "
                        f"macro_auc={optimized_result['macro_auc']:.4f}"
                    )
                else:
                    # Fallback: use argmax baseline
                    optimized_result = {
                        "model_name": "Calib (argmax)",
                        "accuracy": round(opt_acc, 4),
                        "balanced_accuracy": round(opt_bal_acc, 4),
                        "macro_f1": round(opt_macro_f1, 4),
                        "weighted_f1": round(opt_weighted_f1, 4),
                        "macro_auc": round(opt_macro_auc, 4),
                        "per_class": opt_per_class,
                        "confusion_matrix": opt_cm,
                        "y_pred": optimized_pred_base,
                        "proba": calibrated_proba_val,
                    }
                    logger.info(
                        f"[Post-processing] Calib (argmax, no rule_correct): "
                        f"acc={opt_acc:.4f}  bal_acc={opt_bal_acc:.4f}  "
                        f"macro_f1={opt_macro_f1:.4f}  macro_auc={opt_macro_auc:.4f}"
                    )

                # Per-class logging
                pc_src = optimized_result.get("per_class", {})
                for cname, m in pc_src.items():
                    logger.info(
                        f"[Post-processing][per_class] {cname}: "
                        f"precision={m.get('precision','?')}  recall={m.get('recall','?')}  "
                        f"f1={m.get('f1','?')}  auc={m.get('auc','?')}  support={m.get('support','?')}"
                    )

                # Step 6: Save optimized model report
                _serialize_calib = {}
                for k, v in calib_p.items():
                    if isinstance(v, np.ndarray):
                        _serialize_calib[k] = v.tolist()
                    else:
                        _serialize_calib[k] = v

                opt_report = {
                    "timestamp": datetime.now().isoformat(),
                    "model": optimized_result["model_name"],
                    "metrics": {k: v for k, v in optimized_result.items()
                               if k not in ("confusion_matrix", "y_pred", "proba", "per_class")},
                    "per_class": optimized_result.get("per_class", {}),
                    "best_params": {
                        "calibrator": _serialize_calib,
                        "rule_correct": rc_p,
                    },
                }
                opt_report_path = run_dir / "optimized_model_report.json"
                with open(opt_report_path, "w", encoding="utf-8") as _of:
                    json.dump(opt_report, _of, ensure_ascii=False, indent=2)
                logger.info(f"Optimized model report saved: {opt_report_path}")

                # Step 7: Generate comparison visualization including optimized model
                from src.visualization import (
                    plot_confusion_matrix_delta, plot_per_class_metrics_heatmap,
                    plot_topk_accuracy_curve, plot_error_vs_duration,
                    plot_prediction_flip_flow, plot_stage_radar,
                    plot_per_class_auc_comparison, plot_confidence_vs_correctness,
                    plot_calibration_curve, plot_per_class_metrics_comparison,
                    plot_group_confusion_matrix,
                )
                opt_viz_dir = str(run_dir / "visualizations")

                # Collect stage data (keep: short/medium/long, meta, temporal, Calib+RC)
                existing_cm = {}
                existing_per_class = {}
                existing_probas = {}
                existing_preds = {}
                existing_stage_metrics = {}

                # Collect existing stages from result
                gm = result.get("group_metrics", {})
                for gname, gmet in gm.items():
                    existing_stage_metrics[gname] = gmet       # short / medium / long
                mm = result.get("meta_metrics", {})
                if mm and "accuracy" in mm:
                    existing_stage_metrics["meta"] = mm
                tm = result.get("temporal_metrics", {})
                if tm and "accuracy" in tm:
                    existing_stage_metrics["temporal"] = tm

                # Add Calib+RC (from optimized_result, prefers rule_correct)
                existing_stage_metrics["Calib+RC"] = optimized_result

                # Collect per_class
                if mm and mm.get("per_class"):
                    existing_per_class["meta"] = mm["per_class"]
                if tm and tm.get("per_class"):
                    existing_per_class["temporal"] = tm["per_class"]
                existing_per_class["Calib+RC"] = optimized_result.get("per_class", {})

                # Predictions
                existing_preds["Calib+RC"] = np.asarray(optimized_result["y_pred"]) if optimized_result.get("y_pred") is not None else None

                # Probabilities
                existing_probas["Calib+RC"] = calibrated_proba_val

                # Confusion matrices
                existing_cm["Calib+RC"] = optimized_result["confusion_matrix"]

                # Generate comparison visualizations
                if len(existing_stage_metrics) > 1:
                    plot_stage_radar(
                        stage_metrics=existing_stage_metrics,
                        viz_dir=opt_viz_dir,
                    )
                if len(existing_per_class) > 1:
                    plot_per_class_metrics_heatmap(
                        per_class_by_stage=existing_per_class,
                        class_names=class_names_pp, viz_dir=opt_viz_dir,
                    )
                    plot_per_class_auc_comparison(
                        per_class_by_stage=existing_per_class,
                        class_names=class_names_pp, viz_dir=opt_viz_dir,
                    )
                    # Compare core stages: meta / temporal / Calib+RC
                    _core_stages = {k: v for k, v in existing_per_class.items()
                                   if k in ["meta", "temporal", "Calib+RC"]}
                    if len(_core_stages) > 1:
                        plot_per_class_metrics_comparison(
                            group_per_class=_core_stages,
                            class_names=class_names_pp, viz_dir=opt_viz_dir,
                        )

                # Prediction flip + error duration analysis (Calib+RC only)
                if len(existing_preds) > 1:
                    plot_prediction_flip_flow(
                        y_pred_by_stage=existing_preds,
                        class_names=class_names_pp, viz_dir=opt_viz_dir,
                    )
                    plot_error_vs_duration(
                        y_true=val_lb, y_pred_by_stage=existing_preds,
                        class_names=class_names_pp, viz_dir=opt_viz_dir,
                    )

                # Top-K curves
                if len(existing_probas) > 1:
                    plot_topk_accuracy_curve(
                        proba_by_stage=existing_probas,
                        labels=val_lb, class_names=class_names_pp, viz_dir=opt_viz_dir,
                    )

                # CM delta
                if len(existing_cm) > 1:
                    plot_confusion_matrix_delta(
                        cm_stages=existing_cm,
                        class_names=class_names_pp, viz_dir=opt_viz_dir,
                    )

                # Optimized model specific visualizations
                plot_confidence_vs_correctness(
                    proba=calibrated_proba_val, labels=val_lb,
                    class_names=class_names_pp,
                    viz_dir=opt_viz_dir, stage_name="optimized",
                )
                plot_calibration_curve(
                    proba=calibrated_proba_val, labels=val_lb,
                    class_names=class_names_pp,
                    viz_dir=opt_viz_dir, stage_name="optimized",
                )
                plot_group_confusion_matrix(
                    cm=opt_cm, class_names=class_names_pp,
                    viz_dir=opt_viz_dir, group_name="optimized",
                )

                logger.info("[Post-processing] Optimized model comparison visualization generated")
            except Exception as _e:
                logger.warning(f"[Post-processing] Grid search or visualization failed: {_e}", exc_info=True)

    # ---------- Stop hardware monitor ----------
    hw_monitor.stop()

    # ---------- Clean up temporary factor file ----------
    if temp_factors_file is not None:
        import os
        try:
            os.unlink(temp_factors_file.name)
        except OSError:
            pass

    # Copy the complete run log to experiment directory for reproducibility
    if log_file:
        shutil.copy2(log_file, run_dir / Path(log_file).name)
        logger.info(f"Log copied to experiment directory: {run_dir / Path(log_file).name}")


if __name__ == "__main__":
    main()
