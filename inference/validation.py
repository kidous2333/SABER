#!/usr/bin/env python
"""
validation.py -- Validation script based on saved weights.

Unlike train_behavior.py, this script does not retrain temporal models (BiLSTM/Transformer/Mamba),
but directly loads the saved best weights from runs/train/expN/weights/ for inference,
then applies post-processing decoders from the current configuration.

Usage:
  # Use config from training experiment directory + decoder config from current validation.yaml
  python validation.py --run runs/train/exp4

  # Use external config file (decoder config read from it)
  python validation.py --run runs/train/exp4 --config config/validation.yaml

  # Separate common + validation config
  python validation.py --run runs/train/exp4 --config-common config/seq/1.yaml --config-validation config/validation.yaml

Workflow:
  1. Load merged_config.yaml from run directory (dataset paths, factor list, etc.)
  2. Load dataset + factors
  3. Run Stage1+2 (factor group LGBM + meta-LGBM, sub-second if stage cache hit)
  4. Load temporal model weights from weights/ -> inference -> refined probability matrix
  5. Apply post-processing decoders (read from current config's decoder list)
  6. Output metric comparison
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
import shutil
import sys
import time as _time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# ---- Reuse utility functions from train_behavior ----
from mining.discovery import setup_logging, load_dataset_config, build_raw_frame_data, _load_merged_config
from training.train_behavior import get_next_run_dir
from src.synth_validator import SynthValidator
from src.temporal_validator import TemporalValidator, _eval_sequence_metrics, PostCalibrator, tune_pipeline
from src.temporal_models import predict_sequence_model
from src.visualization import (
    plot_label_distribution,
    plot_keypoint_trajectory,
)

logger = logging.getLogger("validate")


def load_model_from_checkpoint(ckpt_path: Path, device: str = "cuda"):
    """Rebuild temporal model from checkpoint and load weights.

    Returns:
        model (nn.Module): model with loaded weights set to eval mode
        ckpt (dict): checkpoint metadata (model_type, cfg, n_classes, etc.)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    n_classes  = ckpt["n_classes"]
    input_dim  = ckpt["input_dim"]
    cfg_saved  = ckpt["cfg"]

    logger.info(f"[validate] checkpoint: model={model_type}, input_dim={input_dim}, n_classes={n_classes}")

    hidden_dim = int(cfg_saved.get("hidden_dim", 128))
    num_layers = int(cfg_saved.get("num_layers", 2))
    dropout    = float(cfg_saved.get("dropout", 0.3))
    use_ln     = bool(cfg_saved.get("use_layer_norm", False))
    use_dp     = bool(cfg_saved.get("use_deep_projection", False))
    use_ap     = bool(cfg_saved.get("use_attention_pooling", False))
    nhead      = int(cfg_saved.get("nhead", 4))
    d_model    = int(cfg_saved.get("d_model", hidden_dim))

    if model_type == "bilstm":
        from src.bilstm_temporal import BiLSTMTemporalModel
        model = BiLSTMTemporalModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=n_classes,
            use_layer_norm=use_ln,
            use_deep_projection=use_dp,
            use_attention_pooling=use_ap,
            nhead=nhead,
        )
    elif model_type == "transformer":
        from src.temporal_models import TransformerTemporalModel
        # Ensure d_model is divisible by nhead
        while d_model % nhead != 0 and nhead > 1:
            nhead -= 1
        model = TransformerTemporalModel(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout, output_dim=n_classes,
            use_layer_norm=use_ln, use_deep_projection=use_dp,
        )
    elif model_type == "mamba":
        from src.temporal_models import MambaTemporalModel
        d_state = int(cfg_saved.get("d_state", 16))
        d_conv  = int(cfg_saved.get("d_conv", 4))
        expand  = int(cfg_saved.get("expand", 2))
        model = MambaTemporalModel(
            input_dim=input_dim, d_model=d_model, d_state=d_state, d_conv=d_conv,
            expand=expand, num_layers=num_layers, dropout=dropout, output_dim=n_classes,
            use_layer_norm=use_ln, use_deep_projection=use_dp,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(ckpt["model_state_dict"])
    _dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(_dev)
    model.eval()
    logger.info(f"[validate] Model loaded on {_dev}: {model_type}")

    return model, ckpt


def run_stages_1_2(cfg, factors_path, train_kp, train_lb, val_kp, val_lb,
                   flat_attributes, label_map, output_dir, run_cfg):
    """Run Stage1+2 (factor grouping + meta-LGBM), reusing stage cache.

    Returns:
        proba_meta_train: [T_train, C]
        proba_meta_val:   [T_val, C]
        train_lb, val_lb: label arrays
        label_map:        class mapping
    """
    logger.info("[validate] === Stage1+2: factor group LGBM + meta-LGBM ===")
    validator = SynthValidator(cfg)

    multiresolution = run_cfg.get("multiresolution", True)
    num_workers      = run_cfg.get("num_workers", 0)
    num_chunks       = run_cfg.get("num_chunks", 256)
    cache_dir        = run_cfg.get("cache_dir", "dataset_cache")
    stage_cache_dir  = run_cfg.get("stage_cache_dir", "pipeline_stage_cache")
    use_stage_cache  = run_cfg.get("use_stage_cache", True)
    refresh_stage    = run_cfg.get("refresh_stage_cache", False)
    purity_mode      = run_cfg.get("purity_mode", "nan_boundary")
    val_purity_mode  = run_cfg.get("val_purity_mode", None)

    if not multiresolution:
        raise ValueError("validation.py only supports multiresolution mode")

    result = validator.run_multiscale_temporal(
        factors_path=factors_path,
        train_kp_raw=train_kp,
        train_lb_raw=train_lb,
        val_kp_raw=val_kp,
        val_lb_raw=val_lb,
        flat_attributes=flat_attributes,
        label_map=label_map,
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
        viz_dir="",
        skip_temporal=True,  # Skip temporal training, we use our own model weights
    )

    # Extract meta proba + extra features from result
    proba_meta_train = result.get("_proba_meta_train")
    proba_meta_val   = result.get("_proba_meta_val")
    extra_train      = result.get("_extra_train")
    extra_val        = result.get("_extra_val")

    if proba_meta_train is None or proba_meta_val is None:
        raise RuntimeError(
            "Stage1+2 did not produce meta proba. Please check if stage cache is available, "
            "or run train_behavior.py once fully to generate cache."
        )

    logger.info(
        f"[validate] Stage1+2 complete: "
        f"meta_train={proba_meta_train.shape}, meta_val={proba_meta_val.shape}"
        + (f", extra_train={extra_train.shape}, extra_val={extra_val.shape}"
           if extra_train is not None else ", extra=none")
    )
    return proba_meta_train, proba_meta_val, extra_train, extra_val, train_lb, val_lb, label_map


def run_post_processing(proba_val, y_val, proba_train, y_train,
                        label_map, output_dir, cfg, extra_train=None, extra_val=None):
    """Apply post-processing decoders from current configuration.

    Returns:
        seq_decoding: {decoder_name: metrics_dict}
    """
    temporal_cfg = cfg.get("temporal_validation", {})
    tv = TemporalValidator(cfg)
    decoders = tv._build_decoders()

    if not decoders:
        logger.info("[validate] No decoders configured, skipping post-processing")
        return {}, None

    classes_sorted = sorted(set(int(v) for v in label_map.values()))
    id_to_name = {int(v): k for k, v in label_map.items()}

    from src.decoders.pipeline import DecoderPipeline
    pipeline = DecoderPipeline(
        decoders=decoders,
        classes_sorted=classes_sorted,
        id_to_name=id_to_name,
        output_dir=output_dir,
    )

    seq_decoding = pipeline.run(
        proba_val=proba_val,
        y_val=np.asarray(y_val).astype(int),
        proba_train=proba_train,
        y_train=np.asarray(y_train).astype(int) if y_train is not None else None,
        feat_train=extra_train,
        feat_val=extra_val,
    )
    return seq_decoding, classes_sorted, id_to_name


def _merge_validation_sections(cfg: dict, ext_cfg: dict) -> None:
    """Merge validation-related sections from external config into cfg."""
    for section in ["temporal_validation", "synth_validation"]:
        if section in ext_cfg:
            cfg.setdefault(section, {}).update(ext_cfg[section])


def main():
    parser = argparse.ArgumentParser(description="Validation script based on saved weights")
    parser.add_argument("--run", type=str, required=True,
                        help="Training experiment directory, e.g. runs/train/exp4")
    parser.add_argument("--config", type=str, default=None,
                        help="External validation config file (overrides decoder config in run directory)")
    parser.add_argument("--config-common", type=str, default=None,
                        help="External common config file")
    parser.add_argument("--config-validation", type=str, default=None,
                        help="External validation config file")
    parser.add_argument("--tune", action="store_true",
                        help="Grid search for optimal RuleCorrector parameters")

    args = parser.parse_args()
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"Error: directory does not exist -- {run_dir}")
        sys.exit(1)

    # -- Step 1: Load config --
    merged_path = run_dir / "configs" / "merged_config.yaml"
    if not merged_path.exists():
        print(f"Error: merged_config.yaml not found -- {merged_path}")
        sys.exit(1)

    with open(merged_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # If external config specified, merge it (post-processing sections use external)
    if args.config:
        # --config: complete validation config file
        with open(args.config, "r", encoding="utf-8") as f:
            ext_cfg = yaml.safe_load(f)
        _merge_validation_sections(cfg, ext_cfg)
    elif args.config_common and args.config_validation:
        # --config-common + --config-validation: replace both
        cfg = _load_merged_config(args.config_common, args.config_validation)
    elif args.config_validation:
        # --config-validation only: override validation sections, common uses run directory's
        with open(args.config_validation, "r", encoding="utf-8") as f:
            ext_cfg = yaml.safe_load(f)
        _merge_validation_sections(cfg, ext_cfg)

    run_cfg = cfg.get("run", {})

    # -- Create output directory --
    val_dir = get_next_run_dir("runs/val")
    output_dir = str(val_dir)
    viz_dir = str(val_dir / "visualizations")
    configs_dir = val_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(merged_path, configs_dir / "merged_config.yaml")
    if args.config:
        shutil.copy2(args.config, configs_dir / Path(args.config).name)
    if args.config_validation:
        shutil.copy2(args.config_validation, configs_dir / Path(args.config_validation).name)

    # -- Logging --
    setup_logging(cfg.get("output", {}).get("logs_dir", "logs"))
    logger.info("=" * 60)
    logger.info(f"Validation mode started -- source experiment: {run_dir}")
    logger.info(f"Output directory: {val_dir}")
    logger.info("=" * 60)

    # -- Step 2: Load dataset --
    ds_cfg_file = cfg.get("dataset_config_file", "")
    if not ds_cfg_file:
        raise ValueError("common.yaml is missing the dataset_config_file field.")

    # merged_config may contain absolute paths from training machine; prefer local files with same name
    ds_cfg_name = Path(ds_cfg_file).name
    searched = []
    ds_cfg_path = None
    for candidate in [
        Path(".") / ds_cfg_name,
        Path("config") / ds_cfg_name,
        Path(".") / "config" / ds_cfg_name,
    ]:
        searched.append(str(candidate))
        if candidate.exists():
            ds_cfg_path = candidate.resolve()
            break

    if ds_cfg_path is None:
        raise FileNotFoundError(
            f"Dataset config file {ds_cfg_name} not found in: {searched}"
        )
    logger.info(f"[validate] Dataset config: {ds_cfg_path}")

    ds_info = load_dataset_config(cfg, str(ds_cfg_path), logger)
    merged_label_map = ds_info["label_map"]
    cfg.setdefault("data", {})["behavior_classes"] = sorted(merged_label_map, key=merged_label_map.get)

    logger.info("[validate] Loading raw per-frame data...")
    train_data, val_data, flat_attributes, video_lengths = build_raw_frame_data(ds_info, logger, cfg=cfg)
    train_kp, train_lb = train_data
    val_kp,   val_lb   = val_data
    logger.info(f"[validate] Training set: {train_kp.shape[0]} frames, Validation set: {val_kp.shape[0]} frames, D={train_kp.shape[1]}")

    # -- Step 3: Load factors (prefer factors.json in exp folder) --
    factors_path = run_dir / "factors.json"
    if factors_path.exists():
        logger.info(f"[validate] Factor file (from exp folder): {factors_path}")
    else:
        factors_fname = Path(run_cfg.get("factors", "memory/valid_factors.json")).name
        factors_path = None
        for candidate in [
            Path(".") / factors_fname,
            Path("memory_before") / factors_fname,
        ]:
            if candidate.exists():
                factors_path = candidate.resolve()
                break
        if factors_path is None:
            raise FileNotFoundError(
                f"Factor file not found: {run_dir / 'factors.json'} or {factors_fname}"
            )
        logger.info(f"[validate] Factor file (fallback): {factors_path}")

    # -- Step 4: Stage1+2 (factor grouping + meta-LGBM) --
    t0 = _time.time()
    proba_meta_train, proba_meta_val, extra_train, extra_val, train_lb, val_lb, label_map = run_stages_1_2(
        cfg, str(factors_path), train_kp, train_lb, val_kp, val_lb,
        flat_attributes, ds_info["label_map"], output_dir, run_cfg,
    )
    logger.info(f"[validate] Stage1+2 time: {_time.time() - t0:.1f}s")

    # -- Step 5: Load temporal model + inference --
    temporal_cfg = cfg.get("temporal_validation", {})
    temporal_model_type = temporal_cfg.get("temporal_model", "bilstm")
    ckpt_path = run_dir / "weights" / f"{temporal_model_type}_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {ckpt_path}. "
            f"Please ensure train_behavior.py has completed training and saved weights."
        )

    logger.info(f"[validate] Loading temporal model: {ckpt_path}")
    model, ckpt = load_model_from_checkpoint(ckpt_path, device=temporal_cfg.get("device", "cuda"))
    logger.info(
        f"[validate] Checkpoint metrics: "
        f"val_acc={ckpt['best_metrics']['val_acc']:.4f}, "
        f"val_balanced_acc={ckpt['best_metrics']['val_balanced_acc']:.4f}, "
        f"macro_f1={ckpt['best_metrics']['macro_f1']:.4f}"
    )

    # Prepare inference input
    seq_model_cfg = temporal_cfg.get("seq_model", {})
    # Supplement architecture parameters from checkpoint cfg (ensure consistency with training)
    saved_cfg = ckpt["cfg"]
    for k in ["hidden_dim", "num_layers", "dropout", "chunk_size", "stride_val",
              "batch_size", "seq_output_residual", "seq_use_extra_features",
              "use_layer_norm", "use_deep_projection", "use_attention_pooling"]:
        if k in saved_cfg and k not in seq_model_cfg:
            seq_model_cfg[k] = saved_cfg[k]

    # Inference validation set config
    seq_model_cfg["device"] = temporal_cfg.get("device", "cuda")
    seq_model_cfg.setdefault("chunk_size", 512)
    seq_model_cfg.setdefault("stride_val", 256)
    seq_model_cfg.setdefault("batch_size", 32)
    seq_model_cfg.setdefault("seq_output_residual", False)

    # Resolve extra features
    seq_use_extra = temporal_cfg.get("seq_use_extra_features", False)
    extra_val_for_inference = None
    if seq_use_extra and extra_val is not None:
        extra_val_for_inference = extra_val.astype(np.float32)
        logger.info(f"[validate] Using extra features: val={extra_val_for_inference.shape}")
    elif seq_use_extra:
        logger.warning("[validate] seq_use_extra_features=true, but extra features not available")

    logger.info(f"[validate] Starting inference...")
    t1 = _time.time()
    proba_val_refined = predict_sequence_model(
        model=model, proba=proba_meta_val, cfg=seq_model_cfg,
        extra=extra_val_for_inference,
    )
    logger.info(f"[validate] Inference complete, time {_time.time() - t1:.1f}s")
    logger.info(f"[validate] proba_val_refined shape: {proba_val_refined.shape}")

    # -- Step 6: Baseline metrics (BiLSTM argmax) --
    classes_sorted = sorted(set(int(v) for v in label_map.values()))
    id_to_name = {int(v): k for k, v in label_map.items()}
    class_names_display = [id_to_name[c] for c in classes_sorted]
    n_classes = len(classes_sorted)

    y_pred_base = proba_val_refined.argmax(axis=1)
    base_metrics = _eval_sequence_metrics(
        np.asarray(val_lb).astype(int), y_pred_base,
        classes_sorted, id_to_name, proba=proba_val_refined,
    )
    logger.info(
        f"[validate] BiLSTM (no post-processing): "
        f"acc={base_metrics['accuracy']:.4f}  "
        f"bal_acc={base_metrics['balanced_accuracy']:.4f}  "
        f"macro_auc={base_metrics['macro_auc']:.4f}  "
        f"macro_f1={base_metrics['macro_f1']:.4f}  "
        f"weighted_f1={base_metrics['weighted_f1']:.4f}"
    )

    # -- Step 7: Post-processing --
    calibrated_proba = proba_val_refined  # default: no calibration
    if args.tune:
        logger.info("[validate] === Grid search Calibrator + RuleCorrector parameters ===")
        best_params, best_acc = tune_pipeline(
            proba_val=proba_val_refined,
            y_val=val_lb,
            proba_meta=proba_meta_val,
            label_map=label_map,
            output_dir=output_dir,
            cfg=cfg,
        )
        calib_p = best_params["calib"]
        rc_p = best_params["rule_correct"]

        print(f"\n  Best params (acc={best_acc:.4f}):")
        mix_a = calib_p['mix_alpha']
        if isinstance(mix_a, np.ndarray):
            class_names_display = [id_to_name[c] for c in classes_sorted]
            print(f"    Calibrator:  temperature={calib_p['temperature']:.2f}")
            print(f"      per_class_alpha: {dict(zip(class_names_display, mix_a.round(3).tolist()))}")
        else:
            print(f"    Calibrator:  temperature={calib_p['temperature']:.2f}  mix_alpha={mix_a:.2f}")
        biases_v = calib_p.get('biases')
        if biases_v is not None:
            print(f"      biases: {dict(zip(class_names_display, biases_v.round(3).tolist()))}")
        if rc_p:
            print(f"    RuleCorrector: {rc_p}")

        # Inject best parameters into config
        for dec_cfg in cfg.setdefault("temporal_validation", {}).setdefault("decoders", []):
            if dec_cfg.get("name") == "rule_correct":
                dec_cfg.setdefault("config", {}).update(rc_p)
        # Save calibrator config
        cfg.setdefault("temporal_validation", {})["calibrator"] = calib_p
        logger.info("[validate] Updated config with best parameters, running formal post-processing...")

    # Apply Calibrator (if configured)
    calib_cfg = temporal_cfg.get("calibrator", {})
    if calib_cfg:
        mix_a = calib_cfg.get("mix_alpha", 1.0)
        calib = PostCalibrator(
            temperature=calib_cfg.get("temperature", 1.0),
            mix_alpha=np.array(mix_a) if isinstance(mix_a, list) else mix_a,
            biases=np.array(calib_cfg["biases"]) if calib_cfg.get("biases") is not None else None,
        )
        calibrated_proba = calib.calibrate(proba_val_refined, proba_meta_val)
        if isinstance(calib.mix_alpha, np.ndarray):
            logger.info(
                f"[validate] Applied Calibrator: T={calib.temperature:.2f}, "
                f"per_class_alpha={calib.mix_alpha.round(3).tolist()}"
            )
        else:
            logger.info(
                f"[validate] Applied Calibrator: T={calib.temperature:.2f}, "
                f"alpha={calib.mix_alpha:.2f}"
            )

        # Print calibrated baseline
        calib_pred = calibrated_proba.argmax(axis=1)
        calib_base = _eval_sequence_metrics(
            np.asarray(val_lb).astype(int), calib_pred,
            classes_sorted, id_to_name, proba=calibrated_proba,
        )
        logger.info(
            f"[validate] BiLSTM+Calibrator: "
            f"acc={calib_base['accuracy']:.4f}  "
            f"macro_f1={calib_base['macro_f1']:.4f}"
        )

    seq_decoding, _, _ = run_post_processing(
        proba_val=calibrated_proba,
        y_val=val_lb,
        proba_train=proba_meta_train[:min(len(proba_meta_train), 50000)],
        y_train=train_lb[:min(len(train_lb), 50000)],
        label_map=label_map,
        output_dir=output_dir,
        cfg=cfg,
    )

    # -- Step 8: Output comparison --
    _name = temporal_model_type.upper()

    rows = []
    rows.append((f"{_name} (base)", base_metrics))
    for dec_name, dm in seq_decoding.items():
        if "accuracy" in dm:
            rows.append((f"{_name}+{dec_name}", dm))

    cols = [
        ("acc",       "accuracy"),
        ("bal_acc",   "balanced_accuracy"),
        ("macro_f1",  "macro_f1"),
        ("macro_auc", "macro_auc"),
        ("w_f1",      "weighted_f1"),
    ]

    name_w = 30
    sep = "=" * 85

    print("\n" + sep)
    print(f"  Validate: {run_dir.name}")
    print(sep)
    header = f"  {'model':<{name_w}}"
    for cname, _ in cols:
        header += f"{cname:>10}"
    print(header)
    print("  " + "-" * (name_w + len(cols) * 10))
    for tag, m in rows:
        line = f"  {tag:<{name_w}}"
        for _, key in cols:
            v = m.get(key)
            line += f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
        print(line)
    print(sep)

    # Delta relative to baseline
    if len(rows) > 1:
        print()
        for tag, m in rows[1:]:
            parts = []
            for _, key in cols:
                delta = m.get(key, 0) - rows[0][1].get(key, 0)
                parts.append(f"{key} {delta:+.4f}")
            print(f"  {tag}: {', '.join(parts)}")

    # Save complete result JSON
    report = {
        "source_run": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "temporal_model": temporal_model_type,
        "base_metrics": base_metrics,
        "sequence_decoding": seq_decoding,
    }
    report_path = val_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"[validate] Report saved: {report_path}")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
