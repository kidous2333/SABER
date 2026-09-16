"""
ValidationWorker — multi-mode model validation (Pose / Behavior).
"""
import sys
import json
import os
import traceback
import tempfile
import shutil
from pathlib import Path

import numpy as np
from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker
from gui.utils.logging_handler import ModuleLogRedirector


class ValidationWorker(BaseWorker):
    """Worker that validates models in one of two modes:
       - Pose: YOLO pose estimation metrics (mAP)
       - Behavior Prediction: SABER 3-stage pipeline + decoder metrics
    """

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    @Slot()
    def run(self):
        redirector = None
        self._cleanup_dirs = []
        try:
            mode = self._params.get("mode", "Behavior Prediction")
            self.log("=" * 50, 20)
            self.log(f"Validation Starting — Mode: {mode}", 20)
            self.log("=" * 50, 20)

            import os as _os
            project_root = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(project_root))
            # Resolve user-supplied paths before chdir
            for k in ("run_dir", "pose_run_dir"):
                v = self._params.get(k, "")
                if v and not Path(v).is_absolute():
                    self._params[k] = str(Path(v).resolve())
            _os.chdir(str(project_root))

            redirector = ModuleLogRedirector()
            redirector.log_signal.connect(self._on_worker_log)
            redirector.install()

            result = {"mode": mode}

            if mode == "Pose":
                self._run_pose_validation(result)
            elif mode == "Behavior Prediction":
                self._run_behavior_validation(result)
            else:
                self.error.emit(f"Unknown validation mode: {mode}")
                self.finished.emit()
                return

            result["success"] = True
            self.set_progress(100, "Validation complete")
            self.result_ready.emit(result)
            self.log("Validation complete.", 20)

        except Exception as e:
            self.log(f"Validation failed: {e}\n{traceback.format_exc()}", 40)
            self.error.emit(str(e))
        finally:
            if redirector:
                redirector.uninstall()
            for d in getattr(self, '_cleanup_dirs', []):
                try:
                    if d and Path(d).exists():
                        shutil.rmtree(d)
                except Exception:
                    pass
            self.finished.emit()

    def _build_temp_pose_dataset(self, image_dir: Path, label_dir: Path) -> Path:
        """Build a temporary YOLO dataset YAML for model.val()."""
        import tempfile
        import shutil as _shutil

        tmp_root = Path(tempfile.mkdtemp(prefix="pose_val_"))
        img_val = tmp_root / "images" / "val"
        lbl_val = tmp_root / "labels" / "val"
        img_val.mkdir(parents=True, exist_ok=True)
        lbl_val.mkdir(parents=True, exist_ok=True)

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        for src in image_dir.iterdir():
            if src.suffix.lower() in exts:
                _shutil.copy2(str(src), str(img_val / src.name))

        for src in label_dir.iterdir():
            if src.suffix.lower() == ".txt":
                _shutil.copy2(str(src), str(lbl_val / src.name))

        # Read pose config from params
        kpt_shape = self._params.get("pose_kpt_shape", "[6, 3]")
        flip_idx = self._params.get("pose_flip_idx", "[0, 1, 2, 3, 4, 5]")
        kpt_names_raw = self._params.get("pose_kpt_names",
            "snout, head_center, body_center, tailbase, ear_left, ear_right")
        kpt_names = [n.strip() for n in kpt_names_raw.split(",") if n.strip()]

        yaml_content = (
            f"path: {tmp_root.as_posix()}\n"
            "train: images/val\n"
            "val: images/val\n"
            f"kpt_shape: {kpt_shape}\n"
            f"flip_idx: {flip_idx}\n"
            "\n"
            "# Classes\n"
            "names:\n  0: mouse\n"
            "\n"
            "# Keypoint names per class\n"
            "kpt_names:\n  0:\n"
        )
        for kn in kpt_names:
            yaml_content += f"    - {kn}\n"

        yaml_path = tmp_root / "data.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")
        return yaml_path

    def _load_yolo_with_fallback(self, weights_path: str):
        """Load YOLO model (uses whichever engine is active)."""
        from ultralytics import YOLO
        return YOLO(weights_path)

    # ── Sample visualisation helpers ──────────────────────────────

    def _generate_pose_samples(self, model, image_dirs: list, n: int = 6) -> list:
        """Run predict on a few images from the given dirs and return annotated outputs."""
        import random

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        candidates = []
        for img_dir in image_dirs:
            if not img_dir or not Path(img_dir).is_dir():
                continue
            for f in Path(img_dir).iterdir():
                if f.suffix.lower() in exts:
                    candidates.append(str(f))

        if not candidates:
            self.log("  No images found in specified directories", 30)
            return []

        self.log(f"  Found {len(candidates)} images", 20)
        random.shuffle(candidates)
        sources = candidates[:n]

        samples_root = Path(self._params.get("pose_run_dir", "")) / "val_samples"
        if samples_root.exists():
            import shutil
            shutil.rmtree(str(samples_root))
        samples_root.mkdir(parents=True, exist_ok=True)
        try:
            model.predict(
                source=sources, save=True, project=str(samples_root), name="images",
                exist_ok=True, verbose=False,
            )
        except Exception as e:
            self.log(f"Sample prediction failed: {e}", 30)
            return []

        img_dir_out = samples_root / "images"
        img_files = sorted(
            list(img_dir_out.glob("*.jpg")) + list(img_dir_out.glob("*.png")))
        self.log(f"  Generated {len(img_files)} sample images", 20)
        return [str(p) for p in img_files]

    @staticmethod
    def _generate_behavior_raster(y_true, y_pred, class_names, output_path: str,
                                   title: str = "Ground Truth vs Predicted"):
        """Create a raster plot comparing ground-truth vs predicted behavior labels."""
        import matplotlib
        matplotlib.use("Agg")
        import logging as _logging
        _logging.getLogger("matplotlib").setLevel(_logging.WARNING)
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        yt = np.asarray(y_true).flatten()
        yp = np.asarray(y_pred).flatten()
        T = min(len(yt), len(yp), 800)  # cap for compact display
        yt, yp = yt[:T], yp[:T]

        n_classes = len(class_names)
        cmap = plt.get_cmap("tab20" if n_classes <= 20 else "hsv", n_classes)

        fig, (ax_gt, ax_pd) = plt.subplots(2, 1, figsize=(12, 4.8), sharex=True)
        fig.suptitle(title, fontsize=10, fontweight="bold")

        for ax, arr, t in [(ax_gt, yt, "Ground Truth"), (ax_pd, yp, "Predicted")]:
            ax.set_ylim(-0.5, n_classes - 0.5)
            ax.set_yticks(range(n_classes))
            ax.set_yticklabels(class_names, fontsize=7)
            ax.set_title(t, fontsize=10, fontweight="bold")
            ax.set_xlim(0, T)
            ax.set_xlabel("Frame" if ax is ax_pd else "", fontsize=8)

            # Draw each class as a horizontal strip
            for c in range(n_classes):
                mask = arr == c
                if not mask.any():
                    continue
                changes = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                for s, e in zip(starts, ends):
                    ax.axvspan(s, e, ymin=(c - 0.45) / n_classes,
                               ymax=(c + 0.45) / n_classes,
                               facecolor=cmap(c), alpha=0.7)

        # Legend
        patches = [mpatches.Patch(color=cmap(i), label=class_names[i])
                   for i in range(n_classes)]
        fig.legend(handles=patches, loc="lower center", ncol=min(n_classes, 8),
                   fontsize=7, frameon=False)

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.savefig(output_path, dpi=400, bbox_inches="tight")
        plt.close(fig)

    # ── Pose validation ──────────────────────────────────────────────

    def _run_pose_validation(self, result: dict):
        pose_run_dir = self._params.get("pose_run_dir", "")
        val_images = self._params.get("pose_val_images", [])
        val_labels = self._params.get("pose_val_labels", [])

        if not pose_run_dir:
            self.error.emit("Pose Run Dir not specified")
            return
        val_images = [p for p in val_images if p and str(p).strip()]
        val_labels = [p for p in val_labels if p and str(p).strip()]
        if not val_images:
            self.error.emit("Val Images not specified")
            return

        run_path = Path(pose_run_dir)
        weights = run_path / "weights" / "best.pt"
        if not weights.exists():
            alt = list(run_path.glob("*.pt"))
            weights = alt[0] if alt else weights
        if not weights.exists():
            self.error.emit(f"Pose weights not found: {weights}")
            return

        self.set_progress(10, "Loading pose model...")
        self.log(f"Pose model: {weights.name}", 20)

        import io as _io
        import logging as _logging
        _saved_levels = {}
        for _name in ["ultralytics", "engine", "torch", "PIL", "matplotlib"]:
            _lg = _logging.getLogger(_name)
            _saved_levels[_name] = _lg.level
            _lg.setLevel(_logging.ERROR)
        _old_stdout, _old_stderr = sys.stdout, sys.stderr
        sys.stdout = _io.StringIO()
        sys.stderr = _io.StringIO()

        try:
            from ultralytics import YOLO
            model = self._load_yolo_with_fallback(str(weights))

            self.set_progress(30, "Running pose inference...")
            self.log(f"Inference on {len(val_images)} image dir(s)", 20)

            # Predict on each image dir
            all_metrics = {}
            for i, img_dir in enumerate(val_images):
                img_path = Path(img_dir)
                if not img_path.exists():
                    self.log(f"  Skipping missing: {img_dir}", 30)
                    continue
                self.log(f"  [{i+1}/{len(val_images)}] {img_path.name}", 20)

                has_label = val_labels and i < len(val_labels) and Path(val_labels[i]).exists()
                if has_label:
                    # Build temp YAML dataset so YOLO val() can read it
                    tmp_ds = self._build_temp_pose_dataset(
                        img_path, Path(val_labels[i]))
                    try:
                        metrics = model.val(data=str(tmp_ds), verbose=False)
                        if hasattr(metrics, "results_dict") and metrics.results_dict:
                            for k, v in metrics.results_dict.items():
                                try:
                                    all_metrics[str(k)] = float(v)
                                except (TypeError, ValueError):
                                    pass
                    finally:
                        # Clean up temp dataset
                        import shutil as _shutil
                        _shutil.rmtree(str(tmp_ds.parent), ignore_errors=True)
                else:
                    model.predict(
                        source=str(img_path), save=False,
                        verbose=False,
                    )
        finally:
            sys.stdout = _old_stdout
            sys.stderr = _old_stderr
            for _name, _lvl in _saved_levels.items():
                _logging.getLogger(_name).setLevel(_lvl)

        # ── Format clean output ──────────────────────────────────
        self.log("─" * 55, 20)
        self.log("  Pose Inference Results", 20)
        self.log("─" * 55, 20)
        self.log(f"  Model      : {weights.name}", 20)
        self.log(f"  Images     : {len(val_images)} dir(s)", 20)

        # Show key metrics first, then all others
        key_keys = [
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "metrics/mAP50(P)", "metrics/mAP50-95(P)",
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/precision(P)", "metrics/recall(P)",
        ]
        logged_any = False
        for k in key_keys:
            if k in all_metrics:
                short = k.replace("metrics/", "")
                self.log(f"  {short:<20} {all_metrics[k]:.4f}", 20)
                logged_any = True

        if logged_any:
            self.log("─" * 55, 20)
        for k, v in sorted(all_metrics.items()):
            if k not in key_keys:
                short = k.replace("metrics/", "")
                self.log(f"  {short:<20} {v}", 20)
        self.log("─" * 55, 20)

        result["pose_metrics"] = {
            "weights": str(weights),
            "all_metrics": all_metrics,
        }

        # ── Generate sample prediction images ────────────────────
        self.set_progress(85, "Generating sample predictions...")
        num_samples = int(self._params.get("num_samples", 6))
        sample_images = self._generate_pose_samples(model, val_images, n=num_samples)
        if sample_images:
            result["pose_metrics"]["sample_images"] = sample_images
            self.log(f"  Sample predictions: {len(sample_images)} images", 20)

    # ── Behavior validation (existing logic) ─────────────────────────

    def _run_behavior_validation(self, result: dict):
        run_dir = self._params.get("run_dir", "")
        if not run_dir:
            self.error.emit("Behavior Run Dir not specified")
            self.finished.emit()
            return

        run_path = Path(run_dir)
        # Auto-correct: if user pointed to weights/ subdirectory, use parent
        if run_path.name == "weights" and (run_path.parent / "configs").exists():
            self.log(f"  Auto-corrected run dir: {run_path} → {run_path.parent}", 30)
            run_path = run_path.parent
        if not run_path.exists():
            self.error.emit(f"Run directory not found: {run_dir}")
            self.finished.emit()
            return

        # ---- Load configs ----
        from mining.discovery import _load_merged_config
        import yaml
        import json as _json

        # Base config from common + validation YAMLs
        cfg = _load_merged_config("config/seq/1.yaml", "config/validation.yaml")

        # Overlay run config: prefer built_config.json, fall back to merged_config.yaml
        built_cfg_path = run_path / "configs" / "built_config.json"
        merged_yaml_path = run_path / "configs" / "merged_config.yaml"
        run_cfg = {}
        if built_cfg_path.exists():
            with open(built_cfg_path, "r", encoding="utf-8") as f:
                run_cfg = _json.load(f)
        elif merged_yaml_path.exists():
            with open(merged_yaml_path, "r", encoding="utf-8") as f:
                run_cfg = yaml.safe_load(f) or {}

        if run_cfg:
            for k, v in run_cfg.items():
                # Never allow force_cache from run config — always compute from scratch
                if k == "force_cache":
                    continue
                if k not in cfg or cfg[k] is None:
                    cfg[k] = v
                elif isinstance(cfg[k], dict) and isinstance(v, dict):
                    cfg[k] = {**cfg[k], **v}   # run config overrides base
            cfg["label_map"] = run_cfg.get("label_map", cfg.get("label_map", {}))

        self.log(f"Run directory: {run_dir}", 20)
        self._run_behavior_pipeline(cfg, run_path, result)

    # ── Shared behavior pipeline ─────────────────────────────────────

    def _run_behavior_pipeline(self, cfg, run_path, result,
                                train_kp=None, train_lb=None,
                                val_kp=None, val_lb=None,
                                flat_attributes=None, num_workers=None,
                                purity_mode=None):
        """Run the SABER 3-stage behavior prediction pipeline and decoders.
        If data arrays are provided they are used directly; otherwise loaded from config."""
        # Load label_map: run's built_config.json (via cfg) or cfg base
        merged_label_map = cfg.get("label_map", {})
        if not merged_label_map or len(merged_label_map) < 2:
            self.error.emit(
                "label_map not found in experiment config. "
                "Ensure the run directory has configs/built_config.json with a valid label_map."
            )
            self.finished.emit()
            return
        self.log(f"  Label map from run config ({len(merged_label_map)} classes)", 20)
        self.log(f"  Label map: {merged_label_map}", 20)
        _t_start = __import__("time").time()

        if train_kp is None:
            self.set_progress(10, "Loading dataset...")
            self.log("━" * 40, 20)
            self.log("  [1/5] Loading dataset", 20)
            from mining.discovery import build_raw_frame_data
            import logging as _logging

            # Build dataset entries from GUI params — separate train and val
            train_mouse = self._params.get("train_mouse_dirs", [])
            train_tail = self._params.get("train_tail_dirs", [])
            train_beh_m1 = self._params.get("train_behavior_m1_files", [])
            train_beh_m2 = self._params.get("train_behavior_m2_files", [])
            val_mouse = self._params.get("val_mouse_dirs", [])
            val_tail = self._params.get("val_tail_dirs", [])
            val_beh_m1 = self._params.get("val_behavior_m1_files", [])
            val_beh_m2 = self._params.get("val_behavior_m2_files", [])
            max_inst = int(self._params.get("max_instances", 2))

            def _to_list(v):
                if isinstance(v, list):
                    return [str(x).strip() for x in v if str(x).strip()]
                return [str(v).strip()] if str(v).strip() else []

            # Train
            tr_m = _to_list(train_mouse)
            tr_t = _to_list(train_tail) if _to_list(train_tail) else tr_m
            tr_b1 = _to_list(train_beh_m1)
            tr_b2 = _to_list(train_beh_m2)
            # Val
            vm = _to_list(val_mouse)
            vt = _to_list(val_tail) if _to_list(val_tail) else vm
            bm1 = _to_list(val_beh_m1)
            bm2 = _to_list(val_beh_m2)

            n_train = len(tr_m)
            n_val = len(vm)

            # Log resolved paths so user can verify
            self.log(f"  Train Mouse KP dirs ({n_train}): {tr_m[:3]}{'...' if n_train > 3 else ''}", 20)
            self.log(f"  Val Mouse KP dirs ({n_val}): {vm[:3]}{'...' if n_val > 3 else ''}", 20)

            if n_train == 0 and n_val == 0:
                self.error.emit(
                    "Both Train and Val Mouse KP are empty — open ⚙ Configure Settings in the "
                    "Validation tab and set Train/Val Mouse KP / Tail KP / "
                    "Behavior M1/M2 paths."
                )
                self.finished.emit()
                return

            # Build training dataset config (fall back to val if train is empty)
            if n_train > 0:
                train_entries = []
                for i in range(n_train):
                    entry = {
                        "Mouse_key_point_file": [tr_m[i]],
                        "Tail_key_point_file": [tr_t[i] if i < len(tr_t) else tr_m[i]],
                        "max_instances_num": max_inst,
                        "behavior_file_mouse1": [tr_b1[i]] if i < len(tr_b1) else [""],
                        "behavior_file_mouse2": [tr_b2[i]] if i < len(tr_b2) else [""],
                    }
                    train_entries.append(entry)
            else:
                # Fall back to val entries for training (original behavior)
                train_entries = []
                for i in range(n_val):
                    entry = {
                        "Mouse_key_point_file": [vm[i]],
                        "Tail_key_point_file": [vt[i] if i < len(vt) else vm[i]],
                        "max_instances_num": max_inst,
                        "behavior_file_mouse1": [bm1[i]] if i < len(bm1) else [""],
                        "behavior_file_mouse2": [bm2[i]] if i < len(bm2) else [""],
                    }
                    train_entries.append(entry)

            # Build validation dataset config (fall back to train if val is empty)
            if n_val > 0:
                val_entries = []
                for i in range(n_val):
                    entry = {
                        "Mouse_key_point_file": [vm[i]],
                        "Tail_key_point_file": [vt[i] if i < len(vt) else vm[i]],
                        "max_instances_num": max_inst,
                        "behavior_file_mouse1": [bm1[i]] if i < len(bm1) else [""],
                        "behavior_file_mouse2": [bm2[i]] if i < len(bm2) else [""],
                    }
                    val_entries.append(entry)
            else:
                val_entries = train_entries

            self.log(f"  {len(train_entries)} training video(s), {len(val_entries)} validation video(s)", 20)
            ds_info = {
                "label_map": merged_label_map,
                "train_dataset_config": train_entries,
                "val_dataset_config": val_entries,
            }

            train_data, val_data, flat_attributes, video_lengths = build_raw_frame_data(
                ds_info, _logging.getLogger("validation"), cfg=cfg)
            train_kp, train_lb = train_data
            val_kp, val_lb = val_data
            _t1 = __import__("time").time()
            self.log(f"  Loaded: {train_kp.shape[0]} train frames, {val_kp.shape[0]} val frames, "
                     f"{len(flat_attributes)} features  ({_t1 - _t_start:.1f}s)", 20)
            if train_kp.shape[0] == 0 or val_kp.shape[0] == 0:
                self.error.emit(
                    f"Loaded empty dataset: train={train_kp.shape[0]} frames, "
                    f"val={val_kp.shape[0]} frames. Check that Train/Val Mouse/Tail KP dirs "
                    f"and behavior CSV files are correctly configured."
                )
                self.finished.emit()
                return
        # (data provided from caller)

        if num_workers is None:
            num_workers = int(self._params.get("num_workers", 8))
        if purity_mode is None:
            purity_mode = cfg.get("run", {}).get("purity_mode", "nan_boundary")

        # ---- Load factors (prefer factors.json in exp folder) ----
        self.set_progress(20, "Loading factors...")
        self.log("  [2/5] Loading factors", 20)
        _t2_start = __import__("time").time()

        # Priority 1: factors.json in the experiment folder
        _exp_factors = run_path / "factors.json"
        if _exp_factors.exists():
            factors_path = str(_exp_factors)
            self.log(f"  Factor file (from exp folder): {factors_path}", 20)
        else:
            # Priority 2: config path
            factors_path = cfg.get("run", {}).get("factors", "memory/valid_factors.json")

            # If valid_factors.json is missing, try to merge from weights/ dir
            if not Path(factors_path).exists():
                weights_dir = run_path / "weights"
                parts = []
                for name in ["common_factors_short.json", "common_factors_medium.json",
                             "common_factors_long.json"]:
                    fp = weights_dir / name
                    if fp.exists():
                        with open(fp, "r", encoding="utf-8") as f:
                            parts.extend(json.load(f))
                if parts:
                    factors_path = str(run_path / "valid_factors_merged.json")
                    with open(factors_path, "w", encoding="utf-8") as f:
                        json.dump(parts, f)
                    self.log(f"  Merged {len(parts)} factors from weights/", 20)
            self.log(f"  Factor file (fallback): {factors_path}", 20)

        if not Path(factors_path).exists():
            self.error.emit(f"Factor file not found: {factors_path}")
            self.finished.emit()
            return

        with open(factors_path, "r", encoding="utf-8") as f:
            factors = json.load(f)
        _t2 = __import__("time").time()
        self.log(f"  Loaded {len(factors)} factors ({_t2 - _t2_start:.1f}s)", 20)
        self.log(f"  Stage 1+2: computing factor values + LGBM inference (~30-120s)", 20)

        # ---- Stage 1+2 (factor LGBM + meta-LGBM) ----
        self.set_progress(30, "Running Stage1+2 (factor LGBM + meta-LGBM)...")
        self.log("  [3/5] Stage 1+2: factor computation + LGBM inference", 20)
        # Suppress LightGBM training logs (native C++ output bypasses Python logging)
        import logging as _logging2
        import io as _io2
        _lgb_logger = _logging2.getLogger("lightgbm")
        _lgb_prev = _lgb_logger.level
        _lgb_logger.setLevel(_logging2.ERROR)
        _old_stdout2, _old_stderr2 = sys.stdout, sys.stderr
        sys.stdout = _io2.StringIO()
        sys.stderr = _io2.StringIO()
        try:
            from src.synth_validator import SynthValidator
            validator = SynthValidator(cfg)
            cfg_val = dict(cfg)
            sv_cfg = cfg_val.setdefault("synth_validation", {})
            sv_cfg["refresh_stage_cache"] = False
            sv_cfg["refresh_cache"] = False
            run_cfg = cfg_val.setdefault("run", {})
            run_cfg["use_stage_cache"] = False

            # ── Validation / inference-only mode: NEVER train; load pre-trained
            #     models from the run's weights/ directory ──
            weights_dir = run_path / "weights"
            meta_lgbm_path = weights_dir / "meta_lgbm.txt"

            if not meta_lgbm_path.exists():
                self.error.emit(
                    f"meta_lgbm.txt not found in {weights_dir}. "
                    f"Validation requires pre-trained models in weights/ — "
                    f"please train first or select a different run directory."
                )
                self.finished.emit()
                return

            # Verify all required group model files exist
            _required_groups = ["short", "medium", "long"]
            _missing_models = []
            for _gn in _required_groups:
                _mp = weights_dir / f"group_{_gn}_lgbm.txt"
                _sp = weights_dir / f"scaler_mean_{_gn}.npy"
                if not _mp.exists():
                    _missing_models.append(str(_mp.name))
                if not _sp.exists():
                    _missing_models.append(str(_sp.name))
            if _missing_models:
                self.error.emit(
                    f"Missing model/scaler files in {weights_dir}: "
                    + ", ".join(_missing_models)
                )
                self.finished.emit()
                return

            self.log("  Mode: inference-only (loading pre-trained models from weights/)", 20)
            self.log(f"  Factors: {len(factors) if isinstance(factors, list) else '?'}"
                     f"  |  Frames: {val_kp.shape[0]}"
                     f"  |  Features: {len(flat_attributes)}", 20)
            self.log("  Running factor computation + LGBM inference (no training)...", 20)

            # ── Progress callback: map pipeline 0→1 fraction to worker 30→55 range ──
            def _pipeline_progress(frac: float, status: str):
                pct = 30 + int(frac * 25)
                self.set_progress(pct, status)

            synth_result = validator.run_multiscale_temporal(
                factors_path=factors_path,
                train_kp_raw=train_kp, train_lb_raw=train_lb,
                val_kp_raw=val_kp, val_lb_raw=val_lb,
                flat_attributes=flat_attributes,
                label_map=merged_label_map,
                output_dir=str(run_path),
                cfg=cfg_val,
                purity_mode=purity_mode,
                num_workers=num_workers, num_chunks=256,
                skip_temporal=True,
                inference_only=True,
                use_stage_cache=False,
                progress_callback=_pipeline_progress,
            )
        finally:
            sys.stdout = _old_stdout2
            sys.stderr = _old_stderr2
            _lgb_logger.setLevel(_lgb_prev)

        proba_meta_train = synth_result.get("_proba_meta_train")
        proba_meta_val = synth_result.get("_proba_meta_val")
        group_metrics = synth_result.get("group_metrics", {})
        meta_metrics = synth_result.get("meta_metrics", {})
        _t3 = __import__("time").time()
        self.log(f"  Stage 1+2 done ({_t3 - _t2:.1f}s)", 20)

        if proba_meta_val is None:
            self.error.emit("Meta-LGBM did not return probability matrix")
            self.finished.emit()
            return

        self.set_progress(60, "Running Stage3 (temporal model)...")
        self.log("  [4/5] Stage 3: temporal NN inference (~5-15s)", 20)

        # ---- Stage 3 (temporal NN) ----
        import torch
        ckpt_files = list((run_path / "weights").glob("*.pt"))
        # filter out non-model pt files
        ckpt_files = [p for p in ckpt_files if "scaler" not in p.name and "fill" not in p.name]
        proba_temporal = proba_meta_val

        if ckpt_files:
            ckpt_path = ckpt_files[0]
            from inference.inference import _load_temporal_nn_model
            temporal_cfg = cfg.get("temporal_validation", {})

            model, ckpt = _load_temporal_nn_model(
                ckpt_path,
                device=temporal_cfg.get("device", "cuda"),
            )
            model_type = ckpt.get("model_type", "bilstm")
            n_classes = ckpt.get("n_classes", len(merged_label_map))
            input_dim = ckpt.get("input_dim", n_classes)

            # Build seq_model_cfg from checkpoint cfg + defaults (like validation.py does)
            saved_cfg = ckpt.get("cfg", {})
            seq_model_cfg = dict(temporal_cfg.get("seq_model", {}))
            for k in ["hidden_dim", "num_layers", "dropout", "chunk_size", "stride_val",
                      "batch_size", "seq_output_residual", "seq_use_extra_features",
                      "use_layer_norm", "use_deep_projection", "use_attention_pooling"]:
                if k in saved_cfg and k not in seq_model_cfg:
                    seq_model_cfg[k] = saved_cfg[k]
            seq_model_cfg["device"] = temporal_cfg.get("device", "cuda")
            seq_model_cfg.setdefault("chunk_size", 512)
            seq_model_cfg.setdefault("stride_val", 256)
            seq_model_cfg.setdefault("batch_size", 32)
            seq_model_cfg.setdefault("seq_output_residual", False)

            if input_dim == n_classes:
                from src.temporal_models import predict_sequence_model
                proba_temporal = predict_sequence_model(
                    model, proba_meta_val, seq_model_cfg)
                self.log(f"Temporal model ({model_type}) loaded and run", 20)
            else:
                self.log(
                    f"Temporal model input_dim={input_dim} != n_classes={n_classes}"
                    f" — checkpoint uses old feature format, skipping", 30)
        else:
            self.log("No temporal checkpoint found in weights/, using meta-LGBM", 30)

        # ---- Metrics helper ----
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score)

        classes_sorted = sorted(set(int(v) for v in merged_label_map.values()))
        id_to_name = {int(v): k for k, v in merged_label_map.items()}
        class_names = [id_to_name[c] for c in classes_sorted]
        n_classes = len(classes_sorted)
        y_val_arr = np.asarray(val_lb).astype(int)

        def _compute_metrics(proba, labels, y_true):
            """Compute stage metrics from probability matrix."""
            y_pred_stage = proba.argmax(axis=1)
            acc = float(accuracy_score(y_true, y_pred_stage))
            bal = float(balanced_accuracy_score(y_true, y_pred_stage))
            mf1 = float(f1_score(y_true, y_pred_stage, labels=classes_sorted,
                                 average="macro", zero_division=0))
            wf1 = float(f1_score(y_true, y_pred_stage, labels=classes_sorted,
                                 average="weighted", zero_division=0))
            try:
                mauc = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                           average="macro", labels=classes_sorted))
                wauc = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                           average="weighted", labels=classes_sorted))
            except Exception:
                mauc = wauc = 0.0
            return acc, bal, mf1, wf1, mauc, wauc

        # Compute temporal NN stage metrics
        t_acc, t_bal, t_mf1, t_wf1, t_mauc, t_wauc = _compute_metrics(
            proba_temporal, class_names, y_val_arr)
        temporal_metrics = {
            "accuracy": round(t_acc, 4), "balanced_accuracy": round(t_bal, 4),
            "macro_f1": round(t_mf1, 4), "weighted_f1": round(t_wf1, 4),
            "macro_auc": round(t_mauc, 4), "weighted_auc": round(t_wauc, 4),
        }

        self.set_progress(80, "Applying saved decoders...")
        _t4 = __import__("time").time()
        self.log(f"  Stage 3 done ({_t4 - _t3:.1f}s)", 20)

        # ---- Apply saved decoder parameters (CRF, rule_correct, etc.) ----
        proba_final = proba_temporal
        y_pred = proba_final.argmax(axis=1)

        import pickle as _pickle
        decoder_dir = run_path / "weights"
        decoder_files = sorted(decoder_dir.glob("*_decoder.pkl"))
        decoder_metrics = None
        if decoder_files:
            self.log(f"  Loading {len(decoder_files)} saved decoder(s)...", 20)
            prev_labels = None
            last_proba = proba_final  # for AUC computation from last decoder
            for dfp in decoder_files:
                try:
                    with open(dfp, "rb") as _f:
                        decoder = _pickle.load(_f)
                    dec_name = decoder.name if hasattr(decoder, 'name') else dfp.stem
                    self.log(f"  [Decoder][{dec_name}] Loaded from {dfp.name}", 20)
                    y_pred = decoder.decode(
                        proba=proba_final, feat=None,
                        prev_labels=prev_labels,
                    )
                    try:
                        last_proba = decoder.predict_proba(
                            proba=proba_final, feat=None,
                            prev_labels=prev_labels,
                        )
                    except Exception:
                        pass  # keep previous proba if predict_proba fails
                    prev_labels = y_pred
                    self.log(
                        f"  [Decoder][{dec_name}] Applied — "
                        f"{len(y_pred)} frames", 20)
                except Exception as _e:
                    self.log(f"  [Decoder] Failed to load/apply {dfp.name}: {_e}", 30)
            # Compute final (post-decoder) metrics with proper AUC
            f_acc = float(accuracy_score(y_val_arr, y_pred))
            f_bal = float(balanced_accuracy_score(y_val_arr, y_pred))
            f_mf1 = float(f1_score(y_val_arr, y_pred, labels=classes_sorted,
                                   average="macro", zero_division=0))
            f_wf1 = float(f1_score(y_val_arr, y_pred, labels=classes_sorted,
                                   average="weighted", zero_division=0))
            try:
                f_mauc = float(roc_auc_score(y_val_arr, last_proba, multi_class="ovr",
                                             average="macro", labels=classes_sorted))
                f_wauc = float(roc_auc_score(y_val_arr, last_proba, multi_class="ovr",
                                             average="weighted", labels=classes_sorted))
            except Exception:
                f_mauc = f_wauc = 0.0
            decoder_metrics = {
                "accuracy": round(f_acc, 4), "balanced_accuracy": round(f_bal, 4),
                "macro_f1": round(f_mf1, 4), "weighted_f1": round(f_wf1, 4),
                "macro_auc": round(f_mauc, 4), "weighted_auc": round(f_wauc, 4),
            }
        else:
            self.log("  No saved decoders found, using raw argmax predictions", 20)

        # ── Generate behavior raster plot ──
        n_total = len(y_pred)
        n_per_mouse = n_total // 2
        mouse_labels = [
            ("Mouse 1", y_val_arr[:n_per_mouse], y_pred[:n_per_mouse]),
            ("Mouse 2", y_val_arr[n_per_mouse:], y_pred[n_per_mouse:]),
        ]
        raster_paths = []
        try:
            for mi, (label, gt, pd_) in enumerate(mouse_labels, 1):
                if len(gt) == 0:
                    continue
                rp = str(run_path / f"behavior_raster_m{mi}.png")
                self._generate_behavior_raster(
                    gt, pd_, class_names, rp,
                    title=f"Ground Truth vs Predicted — {label}",
                )
                raster_paths.append(rp)
            self.log(f"Raster plot saved: {len(raster_paths)} images", 20)
        except Exception as _e:
            self.log(f"Raster plot skipped: {_e}", 30)

        _t5 = __import__("time").time()
        self.log("━" * 40, 20)
        self.log(f"  Total: {_t5 - _t_start:.1f}s  |  {len(y_pred)} frames  |  {len(class_names)} classes", 20)
        self.log("━" * 40, 20)

        result["class_names"] = class_names
        result["raster_plots"] = raster_paths
        result["group_metrics"] = group_metrics
        result["meta_metrics"] = meta_metrics
        result["temporal_metrics"] = temporal_metrics
        result["decoder_metrics"] = decoder_metrics
        result["n_frames"] = len(y_pred)
        result["id_to_name"] = {int(k): v for k, v in id_to_name.items()}
        result["message"] = f"Inference complete. {len(y_pred)} frames, {len(class_names)} classes."

    @Slot(str, int)
    def _on_worker_log(self, msg, level):
        self.log_line.emit(msg, level)
