"""
TrainWorker — runs TMP (Token Mixed Pose) model training in a QThread.
"""

import sys
import logging
import os as _os
import traceback
from pathlib import Path
from contextlib import contextmanager

# ── Silence the training engine BEFORE any of its modules are imported ──
_os.environ.setdefault("ULTRALYTICS_UPDATE_CHECK", "0")
for _name in ["ultralytics", "engine", "torch"]:
    logging.getLogger(_name).setLevel(logging.WARNING)

from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker

logger = logging.getLogger("gui.train_worker")


# ------------------------------------------------------------------
#  stdout / logging suppression
# ------------------------------------------------------------------
@contextmanager
def suppress_engine_output():
    """Silence training engine stdout, stderr, and logging chatter."""
    import io as _io
    _devnull = _io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = _devnull
    sys.stderr = _devnull

    saved = {}
    for _name in ["ultralytics", "engine", "torch", "PIL", "matplotlib"]:
        _lg = logging.getLogger(_name)
        saved[_name] = _lg.level
        _lg.setLevel(logging.ERROR)

    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        for _name, _lvl in saved.items():
            logging.getLogger(_name).setLevel(_lvl)


# ------------------------------------------------------------------
#  Model.train() hook — inject per-epoch callback
# ------------------------------------------------------------------
def _install_epoch_hook(worker, total_epochs: int):
    """
    Patch Model.train() so our per-epoch callback is added to
    ``model.callbacks`` *before* the trainer is created.
    """
    import ultralytics.engine.model as _mmod
    _orig = _mmod.Model.train
    _header_printed = [False]

    def _patched(model_self, trainer=None, **kwargs):
        worker.log("[TMP] Hook installed — callbacks registered", 10)

        # -- per-epoch (after train+val) --
        def _on_fit_epoch_end(trainer_obj):
            epoch = getattr(trainer_obj, 'epoch', -1) + 1
            worker.log(f"[TMP] on_fit_epoch_end (epoch={epoch})", 10)
            _log_epoch(trainer_obj, total_epochs, _header_printed, worker)

        model_self.callbacks.setdefault("on_fit_epoch_end", []).append(_on_fit_epoch_end)

        # -- per-epoch (training phase only, before val) --
        def _on_train_epoch_end(trainer_obj):
            epoch = getattr(trainer_obj, 'epoch', -1) + 1
            pct = min(int(epoch / max(total_epochs, 1) * 90) + 5, 95)
            worker.set_progress(pct, f"Training epoch {epoch}/{total_epochs} done, validating...")
            worker.log(f"[TMP] Epoch {epoch}/{total_epochs} training done — validating", 20)

        model_self.callbacks.setdefault("on_train_epoch_end", []).append(_on_train_epoch_end)

        class _StopTraining(Exception):
            pass

        # -- every 500 batches + cancel check --
        _batch_count = [0]
        def _on_train_batch_end(trainer_obj):
            if worker._cancelled:
                worker.log("[TMP] Stop requested — aborting training", 30)
                raise _StopTraining("User stopped training")
            _batch_count[0] += 1
            if _batch_count[0] % 500 == 0:
                loss = getattr(trainer_obj, 'loss', None)
                loss_str = f"loss={loss.item():.4f}" if hasattr(loss, 'item') else ""
                worker.log(f"[TMP] ...batch {_batch_count[0]} {loss_str}", 10)

        model_self.callbacks.setdefault("on_train_batch_end", []).append(_on_train_batch_end)

        # -- training start --
        def _on_train_start(trainer_obj):
            worker.log("TMP training on device: "
                       + str(getattr(trainer_obj, 'device', 'auto')), 20)
        model_self.callbacks.setdefault("on_train_start", []).append(_on_train_start)

        try:
            return _orig(model_self, trainer=trainer, **kwargs)
        except _StopTraining:
            worker.log("[TMP] Training aborted by user", 20)
            return None

    _mmod.Model.train = _patched


def _log_epoch(trainer, total_epochs: int, header_printed: list, worker):
    """Format and emit one epoch status line via worker.log()."""
    epoch = getattr(trainer, 'epoch', -1) + 1
    # Avoid double-print: on_fit_epoch_end fires once from the epoch loop
    # and once from final_eval; only log each epoch once.
    last_logged = getattr(trainer, '_tmp_last_logged_epoch', 0)
    if epoch <= last_logged:
        return
    trainer._tmp_last_logged_epoch = epoch

    pct = min(int(epoch / max(total_epochs, 1) * 90) + 5, 95)

    # Loss items (extract early for debug dump)
    loss_items = getattr(trainer, 'loss_items', None)
    loss_names = getattr(trainer, 'loss_names',
                         ['box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'])

    if not header_printed[0] or epoch % 10 == 0:
        worker.log(
            f"{'Epoch':>6} {'box':>8} {'pose':>8} {'kobj':>8} {'cls':>8} {'dfl':>8} "
            f"{'mAP50':>8} {'mAP50-95':>10} {'LR':>10}", 20)
        worker.log("-" * 85, 20)
        header_printed[0] = True
    loss_vals = {}
    if loss_items is not None:
        try:
            vals = loss_items.tolist() if hasattr(loss_items, 'tolist') else list(loss_items)
            loss_vals = dict(zip(loss_names, vals))
        except Exception:
            pass

    # mAP — metrics is a plain dict with keys like 'metrics/mAP50(B)'
    metrics = getattr(trainer, 'metrics', None)
    map50 = None
    map50_95 = None
    if isinstance(metrics, dict):
        map50 = metrics.get('metrics/mAP50(B)')
        if map50 is None:
            map50 = metrics.get('metrics/mAP50(M)')
        map50_95 = metrics.get('metrics/mAP50-95(B)')
        if map50_95 is None:
            map50_95 = metrics.get('metrics/mAP50-95(M)')
    elif metrics is not None:
        map50 = getattr(metrics, 'map50', None)
        map50_95 = getattr(metrics, 'map', None)
    if isinstance(map50, (int, float)):
        map50 = float(map50)
    else:
        map50 = None
    if isinstance(map50_95, (int, float)):
        map50_95 = float(map50_95)
    else:
        map50_95 = None

    # LR — trainer.lr is a dict {'lr/pg0': ..., 'lr/pg1': ..., 'lr/pg2': ...}
    lr = None
    raw_lr = getattr(trainer, 'lr', None)
    if isinstance(raw_lr, dict):
        lr = raw_lr.get('lr/pg0', 0) or raw_lr.get(list(raw_lr.keys())[0] if raw_lr else '', 0)
    elif raw_lr is not None:
        try:
            lr = float(raw_lr)
        except (TypeError, ValueError):
            pass
    if lr is None and hasattr(trainer, 'optimizer'):
        try:
            lr = trainer.optimizer.param_groups[0]['lr']
        except Exception:
            pass
    try:
        lr = float(lr) if lr is not None else None
    except (TypeError, ValueError):
        lr = None

    parts = [
        f"{epoch:>5}/{total_epochs:<3}",
        f"{loss_vals.get('box_loss', 0):>8.4f}",
        f"{loss_vals.get('pose_loss', 0):>8.4f}",
        f"{loss_vals.get('kobj_loss', 0):>8.4f}",
        f"{loss_vals.get('cls_loss', 0):>8.4f}",
        f"{loss_vals.get('dfl_loss', 0):>8.4f}",
    ]
    parts.append(f"{map50:>8.4f}" if isinstance(map50, (int, float)) else f"{'—':>8}")
    parts.append(f"{map50_95:>10.4f}" if isinstance(map50_95, (int, float)) else f"{'—':>10}")
    parts.append(f"{lr:>10.2e}" if isinstance(lr, float) else f"{'—':>10}")

    worker.log("".join(parts), 20)
    worker.set_progress(pct, f"Epoch {epoch}/{total_epochs}")

    # Extract val losses from metrics dict
    val_losses = {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if k.startswith('val/'):
                val_losses[k.replace('val/', '')] = v

    # Emit partial result so the GUI tab can update charts & table
    worker.partial_result.emit({
        "epoch": epoch,
        "total_epochs": total_epochs,
        "box_loss": loss_vals.get('box_loss', 0),
        "pose_loss": loss_vals.get('pose_loss', 0),
        "kobj_loss": loss_vals.get('kobj_loss', 0),
        "cls_loss": loss_vals.get('cls_loss', 0),
        "dfl_loss": loss_vals.get('dfl_loss', 0),
        "val_box_loss": val_losses.get('box_loss'),
        "val_pose_loss": val_losses.get('pose_loss'),
        "val_kobj_loss": val_losses.get('kobj_loss'),
        "val_cls_loss": val_losses.get('cls_loss'),
        "val_dfl_loss": val_losses.get('dfl_loss'),
        "map50": map50,
        "map50_95": map50_95,
        "lr": lr,
    })


# ------------------------------------------------------------------
#  Worker
# ------------------------------------------------------------------
class TrainWorker(BaseWorker):
    """Background worker for TMP pose model training."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    @Slot()
    def run(self):
        try:
            self.log("=" * 50, 20)
            self.log("TMP (Token Mixed Pose) — Training Starting", 20)
            self.set_progress(0, "Initializing...")

            # --- Validate paths ---
            model_yaml = self._params.get("model_yaml", "")
            dataset_yaml = self._params.get("dataset_yaml", "")
            if not model_yaml or not Path(model_yaml).exists():
                self.error.emit(f"Model YAML not found: {model_yaml}")
                self.finished.emit(); return
            if not dataset_yaml or not Path(dataset_yaml).exists():
                self.error.emit(f"Dataset YAML not found: {dataset_yaml}")
                self.finished.emit(); return

            # --- Gather params ---
            epochs = int(self._params.get("epochs", 30))
            imgsz = int(self._params.get("imgsz", 640))
            batch = int(self._params.get("batch", 16))
            device = self._params.get("device", "") or None
            single_cls = bool(self._params.get("single_cls", True))
            workers = int(self._params.get("workers", 8))
            project = self._params.get("project", "runs/train")
            name = self._params.get("name", "tmp_exp")
            pretrained = self._params.get("pretrained", "") or None

            self.log(f"Model: {model_yaml}", 20)
            self.log(f"Dataset: {dataset_yaml}", 20)
            self.log(f"Epochs: {epochs}, ImgSz: {imgsz}, Batch: {batch}", 20)
            self.log(f"Device: {device or 'auto'}, Workers: {workers}", 20)
            if pretrained:
                self.log(f"Pretrained weights: {pretrained}", 20)
            self.log(f"Output: {project}/{name}", 20)
            self.log("-" * 40, 20)

            # --- Setup ---
            self.set_progress(5, "Importing modules...")
            from src.pose_trainer import train_tmp_model, _ensure_saga_registered
            _ensure_saga_registered()
            _install_epoch_hook(self, epochs)

            self.set_progress(10, "Starting training...")

            # --- Train (engine chatter fully suppressed) ---
            with suppress_engine_output():
                result = train_tmp_model(
                    model_yaml=model_yaml,
                    dataset_yaml=dataset_yaml,
                    epochs=epochs, imgsz=imgsz, batch=batch,
                    device=device, pretrained=pretrained,
                    single_cls=single_cls,
                    project=project, name=name, workers=workers,
                )

            self.log("-" * 40, 20)
            self.set_progress(100, "Training complete")
            if result["best_pt"]:
                self.log(f"Best checkpoint: {result['best_pt']}", 20)
                self.log(f"Best mAP@50: {result.get('best_map50', 'N/A')}", 20)
                self.log(f"Best mAP@50-95: {result.get('best_map50_95', 'N/A')}", 20)
            self.result_ready.emit(result)

        except Exception as e:
            self.log(f"Training failed: {e}", 40)
            self.log(traceback.format_exc(), 40)
            self.error.emit(str(e))
        finally:
            self.finished.emit()
