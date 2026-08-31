"""
DiscoveryWorker — runs discovery in QThread (same process).
Monkey-patching works because it's all in-process.
"""

import sys
import re
import json
import tempfile
import logging
import traceback
from pathlib import Path

from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker

logger = logging.getLogger("gui.discovery_worker")

_RE_ROUND = re.compile(r"Round\s+(\d+)")

LABEL_NAMES = [
    "explore_object", "climb", "self_grooming", "stand",
    "blank", "positive_sniffs", "approach",
]


def _parse_label_map(params: dict) -> dict:
    # Prefer new-style key-value pairs from editor
    pairs = params.get("label_map_pairs", {})
    if pairs and len(pairs) >= 2:
        return {k: int(v) for k, v in pairs.items() if k.strip() and v.strip()}
    # Fallback: individual {name}_id fields
    result = {}
    for name in LABEL_NAMES:
        key = f"{name}_id"
        if key in params:
            result[name] = int(params[key])
    return result


def _build_dataset_config(params: dict) -> str:
    label_map = _parse_label_map(params)
    max_inst = int(params.get("max_instances", 2))

    def _entry(mouse, tail, m1, m2):
        return {
            "Mouse_key_point_file": [p for p in (mouse or []) if p.strip()],
            "Tail_key_point_file": [p for p in (tail or []) if p.strip()],
            "behavior_file_mouse1": [p for p in (m1 or []) if p.strip()],
            "behavior_file_mouse2": [p for p in (m2 or []) if p.strip()],
            "max_instances_num": max_inst,
        }

    ds = {
        "train": [_entry(
            params.get("train_mouse_dirs", []), params.get("train_tail_dirs", []),
            params.get("train_behavior_m1_files", []), params.get("train_behavior_m2_files", []),
        )],
        "val": [_entry(
            params.get("val_mouse_dirs", []), params.get("val_tail_dirs", []),
            params.get("val_behavior_m1_files", []), params.get("val_behavior_m2_files", []),
        )],
        "label_map": label_map,
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="gui_ds_", delete=False, encoding="utf-8")
    json.dump(ds, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name


def _build_config(params: dict) -> dict:
    label_map = _parse_label_map(params)
    return {
        "output": {"logs_dir": "logs", "memory_dir": "memory"},
        "dataset_config_file": _build_dataset_config(params),
        "label_map": label_map,
        "data": {"behavior_classes": sorted(label_map, key=label_map.get)},
        "llm": {
            "provider": params.get("llm_provider", "anthropic"),
            "model": params.get("llm_model", "claude-sonnet-4-20250514"),
            "base_url": params.get("llm_base_url", "https://www.micuapi.ai"),
            "api_key": params.get("llm_api_key", ""),
            "temperature": float(params.get("llm_temperature", 0.7)),
            "user_agent": params.get("llm_user_agent", ""),
            "custom_headers": _parse_headers(params.get("llm_custom_headers", "")),
            "timeout": float(params.get("llm_timeout", 600)),
            "stream_timeout": float(params.get("llm_timeout", 600)),
        },
        "sequence": {
            "seq_length": int(params.get("seq_length", 5)),
            "stride": int(params.get("stride", 1)),
            "purity_threshold": float(params.get("purity_threshold", 1.0)),
        },
        "validation": {
            "n_splits": int(params.get("n_splits", 5)),
            "min_auc": float(params.get("min_auc", 0.65)),
            "min_f1": float(params.get("min_f1", 0.50)),
            "use_gpu": bool(params.get("use_gpu", False)),
            "lgbm_params": {
                "n_estimators": int(params.get("lgbm_estimators", 200)),
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": int(params.get("lgbm_depth", 4)),
                "random_state": 42,
                "n_jobs": 1,
            },
        },
        "factor_engine": {"max_error_ratio": 0.1, "min_valid_ratio": 0.5},
        "loop": {
            "max_rounds": int(params.get("max_rounds", 200)),
            "early_stop_rounds": int(params.get("early_stop_rounds", 5)),
            "max_valid_factors": int(params.get("max_valid_factors", 5000)),
            "hypotheses_per_round": int(params.get("hypotheses_per_round", 8)),
            "keep_recent_rounds": 5, "keep_recent_factors": 50,
        },
    }


def _parse_headers(text: str) -> dict:
    if not text or not text.strip():
        return {}
    try:
        return json.loads(text)
    except Exception:
        result = {}
        for line in text.strip().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                result[k.strip()] = v.strip()
        return result


class DiscoveryWorker(BaseWorker):
    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    @Slot()
    def run(self):
        try:
            self.log("=" * 50, 20)
            self.log("Factor Discovery Starting", 20)
            self.set_progress(0, "Building config...")

            # Validate: dataset paths must be configured
            train_mouse = [p for p in self._params.get("train_mouse_dirs", []) if p.strip()]
            if not train_mouse:
                self.log("ERROR: No dataset paths configured!", 40)
                self.log("Please open Configure Settings → Dataset, and set at least the Train Mouse KP Dirs paths.", 40)
                self.error.emit("No dataset paths configured. Please configure Dataset paths first.")
                self.finished.emit()
                return

            project_root = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(project_root))

            cfg = _build_config(self._params)
            self._max_rounds = cfg["loop"]["max_rounds"]

            self.log(f"Model: {cfg['llm']['model']}", 20)
            self.log(f"Min AUC: {cfg['validation']['min_auc']}, Max rounds: {self._max_rounds}", 20)
            self.set_progress(3, "Config built")

            # Monkey-patch for live updates (same process — works reliably)
            self._patch_all()

            self.set_progress(5, "Starting discovery...")
            self.log("-" * 40, 20)

            from mining.discovery import run
            valid_factors = run(cfg, config_file="gui_settings.json",
                               cancel_check=lambda: self._cancelled)

            # Save output
            output_path = self._params.get("factors_output_path", "memory/gui_discovered_factors.json")
            if output_path and valid_factors:
                try:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(valid_factors, f, ensure_ascii=False, indent=2)
                    self.log(f"Saved {len(valid_factors)} factors to {output_path}", 20)
                except Exception as e:
                    self.log(f"Save failed: {e}", 30)

            self.log("-" * 40, 20)
            self.set_progress(100, "Complete")

            self.result_ready.emit({
                "factors": valid_factors or [],
                "n_factors": len(valid_factors) if valid_factors else 0,
                "output_path": output_path,
            })

        except Exception as e:
            self.log(f"Discovery failed: {e}", 40)
            self.log(traceback.format_exc(), 40)
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _patch_all(self):
        """Monkey-patch validator + factor engine to emit partial results."""
        current_round = [0]
        worker = self

        # Patch Validator
        try:
            from src.validator import FactorValidator
            orig = FactorValidator.validate_single_holdout
            def patched(self_v, train_f, train_l, val_f, val_l):
                result = orig(self_v, train_f, train_l, val_f, val_l)
                try:
                    worker.partial_result.emit({
                        "type": "eval_result",
                        "round": current_round[0],
                        "name": getattr(FactorValidator, '_hn', 'unknown'),
                        "code": getattr(FactorValidator, '_hc', ''),
                        "description": getattr(FactorValidator, '_hd', ''),
                        "target": getattr(FactorValidator, '_ht', ''),
                        "seq_length": getattr(FactorValidator, '_hs', 5),
                        "valid": result.get("valid", False),
                        "best_auc": result.get("best_auc"),
                        "best_f1": result.get("best_f1"),
                        "best_class": result.get("best_class"),
                        "valid_classes": result.get("valid_classes", []),
                        "per_class": result.get("per_class", {}),
                        "reason": result.get("reason", ""),
                    })
                except RuntimeError:
                    pass  # worker already deleted
                return result
            FactorValidator.validate_single_holdout = patched
            self.log("Validator patched OK", 10)
        except Exception as e:
            self.log(f"Validator patch failed: {e}", 30)

        # Patch FactorEngine
        try:
            from src.factor_engine import FactorEngine
            from src.validator import FactorValidator
            for method, is_batch in [("compute_factor", False), ("compute_factor_batch", True)]:
                orig_method = getattr(FactorEngine, method)
                def _patched(self_e, hypothesis, kp, fa, _orig=orig_method, _batch=is_batch):
                    FactorValidator._hn = hypothesis.get("name", "?")
                    FactorValidator._hc = hypothesis.get("code", "")
                    FactorValidator._hd = hypothesis.get("description", "")
                    FactorValidator._ht = hypothesis.get("target", "")
                    FactorValidator._hs = hypothesis.get("seq_length", 5)
                    return _orig(self_e, hypothesis, kp, fa)
                setattr(FactorEngine, method, _patched)
            self.log("FactorEngine patched OK", 10)
        except Exception as e:
            self.log(f"FactorEngine patch failed: {e}", 30)

        # Patch MemoryManager
        try:
            from src.memory import MemoryManager
            orig_save = MemoryManager.save_valid_factor
            def patched_save(self_m, hypothesis, val_result, seq_length=None):
                orig_save(self_m, hypothesis, val_result, seq_length=seq_length)
                self.log(f"  ✓ Saved: {hypothesis.get('name','?')} (AUC={val_result.get('best_auc',0):.4f})", 20)
            MemoryManager.save_valid_factor = patched_save
        except Exception:
            pass

        # Track rounds via log handler
        import logging as _l
        class RT(_l.Handler):
            def emit(self, record):
                m = _RE_ROUND.search(self.format(record))
                if m:
                    r = int(m.group(1))
                    if r > current_round[0]:
                        current_round[0] = r
                        worker.partial_result.emit({"type": "round_start", "round": r})
                        worker.progress.emit(min(r * 100 // max(worker._max_rounds, 1), 95), f"Round {r}/{worker._max_rounds}")
        tracker = RT()
        tracker.setFormatter(_l.Formatter("%(message)s"))
        _l.getLogger("main").addHandler(tracker)
