"""
TMP (Token Mixed Pose) Model Trainer.

Thin wrapper around the ultralytics training engine for TMP pose estimation
models. Supports standard and TMP-enhanced (C3k2_TMP block) backbone
architectures.

The TMP architecture replaces standard C3k2 blocks in the backbone with
C3k2_TMP blocks, which use TMP attention (IASA intra-group + IRCA
inter-group) for improved keypoint detection.

Usage:
    from src.pose_trainer import train_tmp_model

    result = train_tmp_model(
        model_yaml="config/models/tmp_n.yaml",
        dataset_yaml="config/datasets/coco8-pose_mouse.yaml",
        epochs=30, imgsz=640, batch=16,
    )
    print(result["best_pt"])       # path to best checkpoint
    print(result["best_map50"])    # best mAP@50
"""

import os as _os
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pose_trainer")

# ── early noise suppression ─────────────────────────────────────────────
_os.environ.setdefault("ULTRALYTICS_UPDATE_CHECK", "0")

# ── ensure ultralytics is installed ─────────────────────────────────────
try:
    import ultralytics  # noqa: F401
except ImportError:
    raise ImportError(
        "ultralytics is not installed. Run:  pip install ultralytics>=8.0.0 einops>=0.6.0"
    )


# ------------------------------------------------------------------
#  TMP module registration
# ------------------------------------------------------------------
def _ensure_tmp_registered():
    """
    Ensure the TMP blocks are recognised by the model YAML parser.

    Strategy:
      1. Inject TMP / Bottleneck_TMP / C3k_TMP / C3k2_TMP into the tasks
         module namespace.
      2. Globally replace C3k2 with C3k2_TMP (same signature, drop-in
         compatible) so both backbone and head use TMP-enhanced blocks.
    """
    try:
        import ultralytics.nn.tasks as tasks
    except ImportError:
        return

    from src.tmp_module import TMP, Bottleneck_TMP, C3k_TMP, C3k2_TMP

    # -- 1. inject classes -------------------------------------------------
    tasks.TMP = TMP
    tasks.Bottleneck_TMP = Bottleneck_TMP
    tasks.C3k_TMP = C3k_TMP
    tasks.C3k2_TMP = C3k2_TMP

    # -- 2. check whether parse_model already knows C3k2_TMP ---------------
    import inspect
    try:
        src = inspect.getsource(tasks.parse_model)
    except OSError:
        src = ""
    if "C3k2_TMP" in src:
        return  # already patched at source level

    # ── replace C3k2 globally with C3k2_TMP ──────────────────────────────
    #      C3k2_TMP(C2f) has the same signature as C3k2, so this is a
    #      drop-in replacement.  Both backbone and head will use the
    #      TMP-enhanced blocks.
    _c3k2 = getattr(tasks, "C3k2", None) or getattr(__import__("ultralytics.nn.modules.block"), "C3k2", None)
    if _c3k2 is not None:
        tasks.C3k2 = C3k2_TMP
        tasks.C3k2_TMP = C3k2_TMP  # alias (also covers explicit YAML names)

    # ── mirror the frozenset entries used by parse_model ------------------
    try:
        tasks.repeat_modules = tasks.repeat_modules | {"C3k2_TMP"}
    except Exception:
        pass
    try:
        tasks.multi_output_modules = tasks.multi_output_modules | {"C3k2_TMP"}
    except Exception:
        pass

    # -- 3. patch yaml_model_load to respect YAML's own scale key -----------
    _orig_yaml_load = tasks.yaml_model_load
    def _patched_yaml_load(path):
        d = _orig_yaml_load(path)
        if not d.get("scale"):
            import yaml as _y
            try:
                with open(path, encoding="utf-8") as fh:
                    raw = _y.safe_load(fh)
                if raw and raw.get("scale"):
                    d["scale"] = raw["scale"]
            except Exception:
                pass
        return d
    tasks.yaml_model_load = _patched_yaml_load

    tasks._tmp_patched = True
    logger.info("TMP blocks registered (C3k2 → C3k2_TMP global replacement)")


# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------
def train_tmp_model(
    model_yaml: str,
    dataset_yaml: str,
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 16,
    device: Optional[str] = None,
    pretrained: Optional[str] = None,
    single_cls: bool = True,
    project: str = "runs/train",
    name: str = "tmp_train",
    workers: int = 8,
    callbacks: Optional[dict] = None,
    **kwargs,
) -> dict:
    """
    Train a TMP (Token Mixed Pose) estimation model.

    Parameters
    ----------
    model_yaml : str
        Path to model YAML config (e.g. tmp_n.yaml for nano).
    dataset_yaml : str
        Path to dataset YAML config.
    epochs : int
        Number of training epochs.
    imgsz : int
        Input image size.
    batch : int
        Batch size.
    device : str or None
        Device to use ('cuda', 'cpu', etc.).  None = auto.
    pretrained : str or None
        Path to pretrained weights (.pt) to load before training.
    single_cls : bool
        Whether the dataset has a single class.
    project : str
        Save directory for training runs.
    name : str
        Experiment name (subdirectory under project).
    workers : int
        Number of data loader workers.
    callbacks : dict or None
        Optional training callbacks (event_name → [callable]).
    **kwargs
        Additional arguments passed to the training engine.

    Returns
    -------
    dict with keys: best_pt, last_pt, best_map50, best_map50_95,
                    epochs_completed, save_dir
    """
    import logging as _logging

    # Ensure TMP blocks are available before model YAML parsing
    _ensure_tmp_registered()

    # ── suppress engine chatter ───────────────────────────────────────
    _ul_loggers = ["ultralytics", "engine", "torch"]
    _old_levels = {}
    for _name in _ul_loggers:
        _lg = _logging.getLogger(_name)
        _old_levels[_name] = _lg.level
        _lg.setLevel(_logging.WARNING)
    try:
        import ultralytics.utils.checks as _uc
        _uc.check_pip_update_available = lambda: None
    except Exception:
        pass

    from ultralytics import YOLO

    # Validate paths
    model_path = Path(model_yaml)
    if not model_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_yaml}")

    dataset_path = Path(dataset_yaml)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    logger.info(f"Loading TMP model from YAML: {model_yaml}")

    try:
        model = YOLO(str(model_path), verbose=False)

        if pretrained:
            pretrained_path = Path(pretrained)
            if pretrained_path.exists():
                logger.info(f"Loading pretrained weights: {pretrained}")
                model.load(str(pretrained_path))
            else:
                logger.warning(f"Pretrained weights not found: {pretrained}, skipping")

        if callbacks:
            for event, funcs in callbacks.items():
                if not isinstance(funcs, (list, tuple)):
                    funcs = [funcs]
                for func in funcs:
                    model.add_callback(event, func)
            total_cb = sum(len(v) if isinstance(v, (list, tuple)) else 1 for v in callbacks.values())
            logger.info(f"Registered {total_cb} callback(s)")

        logger.info(
            f"Starting TMP training: epochs={epochs}, imgsz={imgsz}, batch={batch}, "
            f"device={device or 'auto'}, project={project}, name={name}"
        )

        results = model.train(
            data=str(dataset_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            single_cls=single_cls,
            project=project,
            name=name,
            workers=workers,
            verbose=False,
            plots=False,
            pretrained=False,
            **kwargs,
        )
    finally:
        for _name, _lvl in _old_levels.items():
            _logging.getLogger(_name).setLevel(_lvl)

    # Gather results
    save_dir = Path(project) / name
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    metrics = {}
    if hasattr(model, "metrics") and model.metrics:
        metrics = model.metrics

    result = {
        "best_pt": str(best_pt) if best_pt.exists() else None,
        "last_pt": str(last_pt) if last_pt.exists() else None,
        "best_map50": getattr(metrics, "map50", None),
        "best_map50_95": getattr(metrics, "map", None),
        "epochs_completed": epochs,
        "save_dir": str(save_dir),
        "raw_metrics": str(metrics),
    }

    logger.info(f"TMP training complete. Best checkpoint: {result['best_pt']}")
    return result


def validate_tmp_model(
    weights: str,
    dataset_yaml: str,
    device: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Validate a trained TMP model on a dataset.

    Parameters
    ----------
    weights : str
        Path to model weights (.pt).
    dataset_yaml : str
        Path to dataset YAML config.
    device : str or None
        Device to use.

    Returns
    -------
    dict with validation metrics.
    """
    from ultralytics import YOLO

    _ensure_tmp_registered()
    model = YOLO(weights)
    metrics = model.val(data=dataset_yaml, device=device, **kwargs)

    return {
        "map50": getattr(metrics, "map50", None),
        "map50_95": getattr(metrics, "map", None),
        "raw": str(metrics),
    }
