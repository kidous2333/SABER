"""
TrainBehaviorWorker — runs the synthetic validation pipeline in background.

Wraps SynthValidator.run_multiscale_temporal() with real data loading,
factor computation, and multi-resolution evaluation.

All config is built directly from GUI params — no external YAML config files needed.
"""

import json
import logging
import sys
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker
from gui.utils.logging_handler import ModuleLogRedirector


# ── Helper: parse comma-separated ints ──
def _parse_int_list(val):
    if isinstance(val, list):
        return [int(v) for v in val]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return []


# ── Config builders ──

def _build_label_map(params: dict) -> dict:
    """Build label_map dict from GUI label map editor or fallback to individual IDs."""
    pairs = params.get("label_map_pairs", {})
    if pairs and len(pairs) >= 2:
        return {k: int(v) for k, v in pairs.items() if k.strip() and v.strip()}
    # Fallback to individual {name}_id params
    return {
        "explore_object": int(params.get("explore_object_id", 0)),
        "climb": int(params.get("climb_id", 1)),
        "self_grooming": int(params.get("self_grooming_id", 2)),
        "stand": int(params.get("stand_id", 3)),
        "blank": int(params.get("blank_id", 4)),
        "positive_sniffs": int(params.get("positive_sniffs_id", 5)),
        "approach": int(params.get("approach_id", 6)),
    }


def _build_dataset_config(params: dict) -> dict:
    """Build train/val dataset config dicts from GUI Data group params."""
    train_config = [{
        "Mouse_key_point_file": list(params.get("train_mouse_dirs", [])),
        "Tail_key_point_file": list(params.get("train_tail_dirs", [])),
        "behavior_file_mouse1": list(params.get("train_behavior_m1_files", [])),
        "behavior_file_mouse2": list(params.get("train_behavior_m2_files", [])),
        "max_instances_num": int(params.get("max_instances", 2)),
    }]
    val_config = [{
        "Mouse_key_point_file": list(params.get("val_mouse_dirs", [])),
        "Tail_key_point_file": list(params.get("val_tail_dirs", [])),
        "behavior_file_mouse1": list(params.get("val_behavior_m1_files", [])),
        "behavior_file_mouse2": list(params.get("val_behavior_m2_files", [])),
        "max_instances_num": int(params.get("max_instances", 2)),
    }]
    return {"train_dataset_config": train_config, "val_dataset_config": val_config}


def _build_cfg(params: dict, label_map: dict) -> dict:
    """Build the full cfg dict from GUI params, covering all validation.yaml keys."""
    behavior_classes = sorted(label_map, key=label_map.get)

    # Parse multi_scale_windows
    msw = _parse_int_list(params.get("multi_scale_windows", "7,15,31,63"))

    # Parse val_purity_mode (null → None)
    vpm = params.get("val_purity_mode", "null")
    if vpm == "null" or vpm is None:
        val_purity_mode = None
    else:
        val_purity_mode = str(vpm)

    # Build decoders list
    decoders = []
    if params.get("rule_correct_enabled", True):
        decoders.append({
            "name": "rule_correct",
            "config": {
                "max_flicker": 3,
                "flicker_prob_threshold": 0.05,
                "window_size": 31,
                "proba_consistency_threshold": 0.20,
                "min_consecutive": 5,
                "boundary_refine": True,
                "confidence_margin": 0.3,
                "smooth_transitions": True,
            }
        })

    # Force cache
    force_cache = {}
    fc_raw = params.get("force_cache_raw", "").strip()
    if fc_raw:
        force_cache["raw_frames"] = fc_raw

    cfg = {
        "label_map": label_map,
        "data": {"behavior_classes": behavior_classes},

        "sequence": {
            "seq_length": int(params.get("seq_length", 1)),
            "stride": int(params.get("stride", 1)),
            "frame_interval": int(params.get("frame_interval", 1)),
            "purity_threshold": float(params.get("purity_threshold", 1.0)),
            "boundary_margin": int(params.get("boundary_margin", 10)),
        },

        "preprocessing": {
            "scale_normalize": bool(params.get("scale_normalize", False)),
            "rotation_align": bool(params.get("rotation_align", False)),
            "bidirectional_impute": bool(params.get("bidirectional_impute", True)),
            "soft_boundary": bool(params.get("soft_boundary", True)),
            "engineered_features": {
                "relative_orientation": False, "nose_nose_dist": False,
                "relative_velocity_dot": False, "relative_velocity_cross": False,
                "tail_energy": False, "body_elongation": False,
                "stillness": False, "engineered_derivatives": False,
            },
            "augmentation": {
                "enabled": True, "horizontal_flip": True,
                "temporal_jitter": 2, "rotation_jitter_deg": 5.0,
            },
        },

        "imputation": {"short_gap_max": 10},

        "run": {
            "output_dir": "",  # set at runtime
            "purity_mode": str(params.get("purity_mode", "nan_boundary")),
            "val_purity_mode": val_purity_mode,
            "multiresolution": True,
            "temporal": bool(params.get("do_temporal", True)),
            "ovr": bool(params.get("do_ovr", True)),
            "benchmark": bool(params.get("benchmark", False)),
            "visualize_raster": bool(params.get("visualize_raster", True)),
            "raster_frames": int(params.get("raster_frames", 5000)),
            "raster_seed": None,
            "use_cache": bool(params.get("use_cache", True)),
            "refresh_cache": bool(params.get("refresh_cache", False)),
            "use_stage_cache": bool(params.get("use_stage_cache", True)),
            "refresh_stage_cache": bool(params.get("refresh_stage_cache", False)),
            "cache_dir": "dataset_cache",
            "stage_cache_dir": "pipeline_stage_cache",
            "num_workers": int(params.get("num_workers", 8)),
            "num_chunks": int(params.get("num_chunks", 256)),
            "max_samples": int(params.get("max_samples", 0)),
            "filter_corr": str(params.get("filter_corr", "")).strip(),
            "corr_threshold": float(params.get("corr_threshold", 0.85)),
            "save_cache": str(params.get("save_cache", "")).strip() or "dataset_cache/raw_frames_cache.pt",
        },

        "synth_validation": {
            "use_gpu": bool(params.get("use_gpu", False)),
            "tmp_dir": "pipeline_stage_cache",
            "nan_fill": str(params.get("nan_fill", "mean")),
            "normalize_cm": bool(params.get("normalize_cm", True)),
            "class_weight": str(params.get("class_weight", "balanced")),
            "early_stopping_rounds": int(params.get("early_stopping", 50)),
            "top_k": int(params.get("top_k", 3)),
            "feature_selection_top_k": int(params.get("feature_selection_top_k", 0)),
            "top_k_per_group": {
                "short": int(params.get("top_k_short", 0)),
                "medium": int(params.get("top_k_medium", 0)),
                "long": int(params.get("top_k_long", 0)),
            },
            "use_focal_loss": bool(params.get("use_focal_loss", True)),
            "focal_alpha": float(params.get("focal_alpha", 0.25)),
            "focal_gamma": float(params.get("focal_gamma", 2.0)),
            "lgbm_params": {
                "n_estimators": int(params.get("lgbm_estimators", 500)),
                "max_depth": int(params.get("lgbm_depth", 6)),
                "learning_rate": float(params.get("lgbm_lr", 0.05)),
                "num_leaves": int(params.get("num_leaves", 63)),
                "random_state": 42,
                "n_jobs": 1,
                "deterministic": True,
            },
            "residual_stage1": bool(params.get("residual_stage1", True)),
            "residual_stage2": bool(params.get("residual_stage2", True)),
        },

        "temporal_validation": {
            "window_size": int(params.get("temporal_window", 31)),
            "monitor_metric": str(params.get("monitor_metric", "val_balanced_acc")),
            "temporal_model": str(params.get("temporal_model", "bilstm")),
            "use_focal_loss": bool(params.get("t_use_focal_loss", True)),
            "focal_alpha": float(params.get("t_focal_alpha", 0.25)),
            "focal_gamma": float(params.get("t_focal_gamma", 2.0)),
            "use_class_weights": bool(params.get("use_class_weights", True)),
            "seq_output_residual": bool(params.get("seq_output_residual", True)),
            "seq_use_extra_features": bool(params.get("seq_use_extra_features", True)),
            "multi_scale_windows": msw,
            "enhanced_features": bool(params.get("enhanced_features", True)),
            "autocorr_features": bool(params.get("autocorr_features", True)),
            "cross_scale_features": bool(params.get("cross_scale_features", True)),
            "distribution_shape": bool(params.get("distribution_shape", True)),
            "decoders": decoders,
            "seq_model": {
                "hidden_dim": int(params.get("hidden_dim", 512)),
                "num_layers": int(params.get("num_layers", 3)),
                "dropout": float(params.get("dropout", 0.3)),
                "epochs": int(params.get("epochs", 200)),
                "lr": float(params.get("lr", 0.005)),
                "weight_decay": float(params.get("weight_decay", 0.0005)),
                "chunk_size": 512,
                "stride_train": 512,
                "stride_val": 256,
                "batch_size": int(params.get("batch_size", 32)),
                "device": str(params.get("device", "cuda")),
                "use_amp": bool(params.get("use_amp", True)),
                "early_stopping_patience": int(params.get("early_stopping_patience", 20)),
                "seed": 42,
                "d_model": 128,
                "nhead": 4,
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "use_layer_norm": bool(params.get("use_layer_norm", True)),
                "use_deep_projection": False,
                "use_attention_pooling": False,
                "label_smoothing": float(params.get("label_smoothing", 0.1)),
                "use_cosine_warmup": bool(params.get("use_cosine_warmup", True)),
                "warmup_epochs": int(params.get("warmup_epochs", 10)),
                "use_ema": False,
                "ema_decay": 0.999,
                "use_balanced_sampling": bool(params.get("use_balanced_sampling", True)),
                "use_mixup": False,
                "mixup_alpha": 0.2,
                "use_time_reversal": False,
                "grid_search": {"enabled": False},
            },
        },

        "force_cache": force_cache,
    }
    return cfg


class TrainBehaviorWorker(BaseWorker):
    """Worker that runs the full synthetic validation pipeline."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    @Slot()
    def run(self):
        redirector = None
        try:
            self.log("=" * 50, 20)
            self.log("Training Behavior Pipeline Starting", 20)
            self.log("=" * 50, 20)
            self.set_progress(0, "Building configuration...")

            project_root = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(project_root))

            from mining.discovery import build_raw_frame_data

            params = self._params

            # ---- Build config directly from GUI params (no YAML files) ----
            label_map = _build_label_map(params)
            self.log(f"Label map built from GUI: {len(label_map)} classes", 20)

            label_merge_enabled = params.get("label_merge_enabled", True)
            label_merge_config = params.get("label_merge_config", "")
            if label_merge_enabled and label_merge_config.strip():
                from mining.discovery import _apply_label_merge
                groups = []
                for rule in label_merge_config.split(","):
                    rule = rule.strip()
                    if ":" in rule:
                        src, tgt = rule.split(":", 1)
                        groups.append({"target": tgt.strip(), "sources": [src.strip()]})
                if groups:
                    label_merge_cfg = {"enabled": True, "groups": groups}
                    label_map = _apply_label_merge(label_map, label_merge_cfg,
                                                   logging.getLogger("train_behavior"))
                    self.log(f"Label merge applied: {label_merge_config}", 20)

            ds_info = _build_dataset_config(params)
            ds_info["label_map"] = label_map
            self.log("Dataset config built from GUI params", 20)

            cfg = _build_cfg(params, label_map)
            # output_dir is set at runtime after run_dir is created
            cfg.setdefault("run", {})["output_dir"] = ""  # placeholder, filled below
            self.log("Full config built from GUI params", 20)

            # Confirm mode settings
            mode = params.get("mode", "multiresolution")
            do_temporal = params.get("do_temporal", True)
            do_ovr = params.get("do_ovr", True)
            self.log(f"Mode: {mode} | Temporal: {do_temporal} | OvR: {do_ovr}", 20)
            self.log(f"Temporal model: {params.get('temporal_model', 'bilstm')} | "
                     f"Device: {params.get('device', 'cuda')}", 20)

            # ---- Load dataset ----
            self.set_progress(5, "Loading dataset...")
            self.log("Building train/val raw frame data...", 20)

            train_data, val_data, flat_attributes, video_lengths = build_raw_frame_data(
                ds_info, logging.getLogger("train_behavior"), cfg=cfg
            )
            train_kp, train_lb = train_data
            val_kp, val_lb = val_data
            train_vlens, val_vlens = video_lengths

            max_samples = int(params.get("max_samples", 0))
            if max_samples > 0:
                rng = np.random.default_rng(42)
                for kp, lb, tag in [(train_kp, train_lb, "training"),
                                     (val_kp, val_lb, "validation")]:
                    n = len(lb)
                    if n <= max_samples:
                        continue
                    idx = rng.choice(n, max_samples, replace=False)
                    idx.sort()
                    kp[:] = kp[idx]
                    lb[:] = lb[idx]
                self.log(f"Sampled to {max_samples} frames per set", 20)

            self.log(
                f"Training: {train_kp.shape[0]} frames, "
                f"Validation: {val_kp.shape[0]} frames, D={train_kp.shape[1]}",
                20,
            )

            # ---- Factor loading ----
            self.set_progress(15, "Loading factors...")
            factors_path = params.get("factors_path", "").strip()
            if not factors_path:
                factors_path = "memory/valid_factors.json"
            factors_path = str(Path(factors_path) if Path(factors_path).is_absolute()
                               else project_root / factors_path)
            from src.synth_validator import SynthValidator
            factors = SynthValidator.load_valid_factors(factors_path)
            self.log(f"Loaded {len(factors)} factors from {factors_path}", 20)
            if not factors:
                self.error.emit(f"No factors found in {factors_path}")
                self.finished.emit()
                return

            # ---- Set up redirector ----
            redirector = ModuleLogRedirector()
            redirector.log_signal.connect(self._on_worker_log)
            redirector.install()

            # Enable INFO-level pipeline logs; suppress noisy third-party loggers
            for _name in ["src.synth_validator", "src.temporal_validator",
                          "src.validator", "src.factor_engine", "train_behavior"]:
                logging.getLogger(_name).setLevel(logging.INFO)
            logging.root.setLevel(logging.INFO)
            for _noisy in ["PySide6", "PySide6.QtCore", "matplotlib", "PIL",
                           "lightgbm", "sklearn", "numexpr", "PIL.Image"]:
                logging.getLogger(_noisy).setLevel(logging.WARNING)

            # ---- Output dir ----
            from training.train_behavior import get_next_run_dir
            run_dir = get_next_run_dir(str(project_root / "runs")).resolve()
            output_dir = str(run_dir)
            viz_dir = str(run_dir / "visualizations")
            Path(viz_dir).mkdir(parents=True, exist_ok=True)
            cfg["run"]["output_dir"] = output_dir  # Inject actual path

            # ---- Data visualizations (01_data) ----
            from src.visualization import (
                plot_label_distribution, plot_segment_duration_analysis,
            )
            classes_sorted_viz = sorted(set(int(v) for v in label_map.values()))
            id_to_name_viz = {int(v): k for k, v in label_map.items()}
            class_names_viz = [id_to_name_viz[c] for c in classes_sorted_viz]
            try:
                plot_label_distribution(
                    y_train=train_lb, y_val=val_lb,
                    class_names=class_names_viz, viz_dir=viz_dir,
                )
                plot_segment_duration_analysis(
                    y_true=val_lb, class_names=class_names_viz, viz_dir=viz_dir,
                )
                self.log("Data visualizations (01_data) generated", 20)
            except Exception as _e:
                self.log(f"Data visualization skipped: {_e}", 30)

            configs_dir = run_dir / "configs"
            configs_dir.mkdir(parents=True, exist_ok=True)
            with open(configs_dir / "gui_params.json", "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2, default=str)
            with open(configs_dir / "built_config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

            # ---------- Save factors.json to experiment directory ----------
            _factors_dst = run_dir / "factors.json"
            with open(_factors_dst, "w", encoding="utf-8") as f:
                json.dump(factors, f, ensure_ascii=False, indent=2)
            self.log(f"Factor list saved to experiment directory: {_factors_dst}", 20)

            self.log(f"Experiment directory: {run_dir}", 20)

            # Emit run_dir early so GUI can start polling for images
            self.partial_result.emit({"type": "run_dir", "run_dir": str(run_dir)})

            # ---- Run pipeline ----
            purity_mode = cfg.get("run", {}).get("purity_mode", "nan_boundary")
            val_purity_mode = cfg.get("run", {}).get("val_purity_mode", None)  # Read from cfg (built from params)
            num_workers = int(params.get("num_workers", 8))
            num_chunks = int(params.get("num_chunks", 256))
            use_stage_cache = params.get("use_stage_cache", True)
            refresh_stage = params.get("refresh_stage_cache", False)
            refresh_cache = params.get("refresh_cache", False)

            # If refresh enabled, clear relevant caches to force full recomputation
            if refresh_cache:
                import shutil
                cache_p = project_root / "dataset_cache"
                for pat in ["factor_matrix_*", "cached_data_*", "raw_frames_cache*"]:
                    for f in cache_p.glob(pat):
                        if f.is_dir():
                            shutil.rmtree(f, ignore_errors=True)
                        else:
                            f.unlink(missing_ok=True)
                self.log("Factor cache cleared for full refresh", 20)
            if refresh_stage:
                stage_p = project_root / "pipeline_stage_cache"
                if stage_p.exists():
                    import shutil
                    shutil.rmtree(stage_p, ignore_errors=True)
                self.log("Stage cache cleared for full refresh", 20)

            self.set_progress(20, "Stage 1/3: Computing factor matrix + group LGBM...")

            # ── Progress callback: map pipeline 0→1 fraction to worker 20→85 range ──
            def _pipeline_progress(frac: float, status: str):
                pct = 20 + int(frac * 65)
                self.set_progress(pct, status)

            validator = SynthValidator(cfg)
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
                cache_dir=str(project_root / "dataset_cache"),
                stage_cache_dir=str(project_root / "pipeline_stage_cache"),
                use_stage_cache=use_stage_cache,
                refresh_stage_cache=refresh_stage,
                viz_dir=viz_dir,
                progress_callback=_pipeline_progress,
            )

            self.set_progress(85, "Building result summary...")

            # ---- Build result summary ----
            stage_metrics = []
            gm = result.get("group_metrics", {})
            for gname, gmet in gm.items():
                stage_metrics.append({
                    "name": f"Group ({gname})",
                    "accuracy": gmet.get("accuracy"),
                    "balanced_acc": gmet.get("balanced_acc") or gmet.get("balanced_accuracy"),
                    "macro_f1": gmet.get("macro_f1"),
                    "macro_auc": gmet.get("macro_auc"),
                    "weighted_f1": gmet.get("weighted_f1"),
                })

            mm = result.get("meta_metrics", {})
            if mm and "accuracy" in mm:
                stage_metrics.append({
                    "name": "Meta-LGBM",
                    "accuracy": mm.get("accuracy"),
                    "balanced_acc": mm.get("balanced_acc") or mm.get("balanced_accuracy"),
                    "macro_f1": mm.get("macro_f1"),
                    "macro_auc": mm.get("macro_auc"),
                    "weighted_f1": mm.get("weighted_f1"),
                })

            tm = result.get("temporal_metrics", {})
            if tm and "accuracy" in tm:
                stage_metrics.append({
                    "name": "Temporal",
                    "accuracy": tm.get("accuracy"),
                    "balanced_acc": tm.get("balanced_acc") or tm.get("balanced_accuracy"),
                    "macro_f1": tm.get("macro_f1"),
                    "macro_auc": tm.get("macro_auc"),
                    "weighted_f1": tm.get("weighted_f1"),
                })

            temporal_accuracy = tm.get("accuracy", 0) if tm else 0

            # ---- Collect all generated images ----
            image_groups = {}  # {group_label: [path, ...]}
            if run_dir.exists():
                for png in sorted(run_dir.rglob("*.png")):
                    # Determine group label from path context
                    rel = png.relative_to(run_dir)
                    parts = rel.parts
                    # Categorize: confusion matrices at root, visualizations in subdirs
                    if len(parts) == 1:
                        # Root-level: synth_confusion_matrix.png, temporal_confusion_matrix.png
                        name = png.stem
                        if name.startswith("synth_"):
                            group = "Meta CM"
                        elif name.startswith("temporal_"):
                            group = "Temporal CM"
                        else:
                            group = "Results"
                    else:
                        # Inside visualizations/ subdir: use parent dir name
                        group = parts[0] if len(parts) <= 2 else f"{parts[0]}/{parts[1]}"
                    image_groups.setdefault(group, []).append(str(png))
                self.log(f"Found {sum(len(v) for v in image_groups.values())} images in {len(image_groups)} groups", 20)

            final_result = {
                "success": True,
                "run_dir": str(run_dir),
                "stage_metrics": stage_metrics,
                "temporal_accuracy": temporal_accuracy,
                "image_groups": image_groups,
                "output_files": result.get("output_files", {}),
            }

            self.set_progress(100, "Validation complete")
            self.result_ready.emit(final_result)
            self.log("=" * 50, 20)
            self.log("Training behavior complete!", 20)
            _weights_abs = str(Path(output_dir).resolve() / "weights")
            self.log(f"Models saved to: {_weights_abs}", 20)

        except Exception as e:
            self.log(f"Validation failed: {e}\n{traceback.format_exc()}", 40)
            self.error.emit(str(e))
        finally:
            if redirector:
                redirector.uninstall()
            self.finished.emit()

    @Slot(str, int)
    def _on_worker_log(self, msg, level):
        self.log_line.emit(msg, level)
        # Parse stage metrics from pipeline logs for live display
        if level > 20:  # Only parse INFO lines
            return
        import re
        # Group: [MultiScale] Group 'short': accuracy=0.4859  balanced_acc(per_class_acc_mean)=0.4186  ...
        m = re.search(
            r"\[MultiScale\] Group '(\w+)':\s*accuracy=([\d.]+)\s+"
            r"balanced_acc\(per_class_acc_mean\)=([\d.]+)\s+"
            r"macro_auc=([\d.]+)\s+weighted_auc=([\d.]+)\s+"
            r"macro_f1=([\d.]+)\s+weighted_f1=([\d.]+)",
            msg
        )
        if m:
            self.partial_result.emit({
                "type": "stage_metric",
                "name": f"Group ({m.group(1)})",
                "accuracy": float(m.group(2)),
                "balanced_acc": float(m.group(3)),
                "macro_auc": float(m.group(4)),
                "weighted_f1": float(m.group(7)),
                "macro_f1": float(m.group(6)),
            })
            return
        # Meta: [MultiScale][meta-LGBM] accuracy=0.5368  balanced_acc(per_class_acc_mean)=0.3688  ...
        m = re.search(
            r"\[MultiScale\]\[meta-LGBM\]\s*accuracy=([\d.]+)\s+"
            r"balanced_acc\(per_class_acc_mean\)=([\d.]+)\s+"
            r"macro_auc=([\d.]+)\s+weighted_auc=([\d.]+)\s+"
            r"macro_f1=([\d.]+)\s+weighted_f1=([\d.]+)",
            msg
        )
        if m:
            self.partial_result.emit({
                "type": "stage_metric",
                "name": "Meta-LGBM",
                "accuracy": float(m.group(1)),
                "balanced_acc": float(m.group(2)),
                "macro_auc": float(m.group(3)),
                "weighted_f1": float(m.group(7)),
                "macro_f1": float(m.group(6)),
            })
            return
        # Temporal: [Temporal][bilstm] accuracy=0.3956  balanced_acc(per_class_acc_mean)=0.4131  ...
        m = re.search(
            r"\[Temporal\]\[(\w+)\]\s*accuracy=([\d.]+)\s+"
            r"balanced_acc\(per_class_acc_mean\)=([\d.]+)\s+"
            r"macro_auc=([\d.]+)\s+weighted_auc=([\d.]+)\s+"
            r"macro_f1=([\d.]+)\s+weighted_f1=([\d.]+)",
            msg
        )
        if m:
            self.partial_result.emit({
                "type": "stage_metric",
                "name": f"Temporal ({m.group(1)})",
                "accuracy": float(m.group(2)),
                "balanced_acc": float(m.group(3)),
                "macro_auc": float(m.group(4)),
                "weighted_f1": float(m.group(7)),
                "macro_f1": float(m.group(6)),
            })
