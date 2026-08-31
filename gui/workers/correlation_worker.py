"""
CorrelationWorker — runs factor correlation analysis in background.

Replicates correlation.py CLI logic:
  1. Group factors by seq_length
  2. Per group: compute factor values via SynthValidator multi-resolution
  3. Per group: Pearson correlation + greedy dedup
  4. Aggregate kept factors across groups

Config is built directly from GUI params — no external config files needed.
"""

import sys
import json
import os
import traceback
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker
from gui.utils.logging_handler import ModuleLogRedirector


def _pearson_corr_matrix(X: np.ndarray, log: logging.Logger) -> np.ndarray:
    """Column-wise Pearson correlation, NaN filled with column mean."""
    T, K = X.shape
    log.info(f"  Correlation: {T} samples x {K} factors")
    col_means = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), col_means[np.newaxis, :], X)
    stds = X_filled.std(axis=0)
    X_filled[:, stds == 0] = 0.0
    corr = np.corrcoef(X_filled.T)
    return np.nan_to_num(corr, nan=0.0).astype(np.float32)


def _greedy_dedup(
    factor_names: list,
    factors_meta: dict,
    corr: np.ndarray,
    threshold: float,
    keep_by: str,
    log: logging.Logger,
) -> tuple:
    """
    Greedy redundancy removal per group.

    Returns (kept_names, redundant_pairs) where redundant_pairs are in
    GUI format: factor_a=kept, factor_b=removed, correlation, kept, reason.
    """
    def score(name):
        meta = factors_meta.get(name, {})
        if keep_by == "best_auc":
            return meta.get("best_auc", 0.0)
        if keep_by == "weighted_auc":
            vcs = meta.get("valid_classes", [])
            if not vcs:
                return meta.get("best_auc", 0.0)
            return sum(v["auc"] for v in vcs) / len(vcs)
        return meta.get("best_auc", 0.0)

    n = len(factor_names)
    order = sorted(range(n), key=lambda i: score(factor_names[i]), reverse=True)
    abs_corr = np.abs(corr)
    kept_idx = []
    redundant_pairs = []

    for i in order:
        if kept_idx:
            max_corr_with_kept = abs_corr[i, kept_idx].max()
            if max_corr_with_kept >= threshold:
                j = kept_idx[int(abs_corr[i, kept_idx].argmax())]
                kept_name = factor_names[j]
                removed_name = factor_names[i]
                redundant_pairs.append({
                    "factor_a": kept_name,
                    "factor_b": removed_name,
                    "correlation": round(float(abs_corr[i, j]), 4),
                    "kept": kept_name,
                    "reason": f"Correlation >= {threshold} (kept higher {keep_by})",
                })
                continue
        kept_idx.append(i)

    kept_names = [factor_names[i] for i in kept_idx]
    log.info(f"  Dedup: {n} -> {len(kept_names)} (removed {len(redundant_pairs)})")
    return kept_names, redundant_pairs


class CorrelationWorker(BaseWorker):
    """Worker that runs correlation analysis and redundancy removal."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    @Slot()
    def run(self):
        redirector = None
        dataset_config_file = ""
        try:
            self.log("=" * 50, 20)
            self.log("Correlation Analysis Starting", 20)
            self.log("=" * 50, 20)
            self.set_progress(0, "Loading configuration...")

            project_root = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(project_root))

            params = self._params
            factors_path = params.get("factors_path", "memory/valid_factors.json")
            threshold = float(params.get("threshold", 0.95))
            keep_by = params.get("keep_by", "best_auc")

            self.log(f"Factors: {factors_path}", 20)
            self.log(f"Threshold: {threshold}, Keep by: {keep_by}", 20)

            # ---- Build dataset config + cfg from GUI params (no config files) ----
            from gui.workers.discovery_worker import _build_dataset_config, _parse_label_map

            # Fallback: use discovery tab's data paths if unset in correlation tab
            from gui.utils.gui_settings import get_tab_settings
            ds_params = get_tab_settings("discovery")
            if ds_params:
                for k in ("train_mouse_dirs", "train_tail_dirs", "train_behavior_m1_files",
                          "train_behavior_m2_files", "max_instances"):
                    if k not in params or not params.get(k):
                        params[k] = ds_params.get(k, params.get(k))
                # Also inherit label_map_pairs if not configured here
                if not params.get("label_map_pairs") and ds_params.get("label_map_pairs"):
                    params["label_map_pairs"] = ds_params["label_map_pairs"]
                    self.log("Using label map from Discovery tab settings", 20)

            if not params.get("train_mouse_dirs") or not any(
                p.strip() for p in (params.get("train_mouse_dirs") or [])
            ):
                self.error.emit("No training data paths configured. Please set them in the Data group or Discovery tab.")
                self.finished.emit()
                return

            # Correlation analysis only uses training data, but build_raw_frame_data
            # requires val paths too — clone train paths into val as a workaround.
            for prefix in ("val_mouse_dirs", "val_tail_dirs", "val_behavior_m1_files", "val_behavior_m2_files"):
                if not params.get(prefix) or not any(p.strip() for p in (params.get(prefix) or [])):
                    train_key = prefix.replace("val_", "train_")
                    params[prefix] = params.get(train_key, [])
                    self.log(f"  Using train data for {prefix} (val not configured)", 20)

            dataset_config_file = _build_dataset_config(params)
            self.log(f"Dataset config built from GUI settings: {dataset_config_file}", 20)
            label_map = _parse_label_map(params)
            self.log(f"Label map: {label_map}", 20)

            # Build minimal cfg dict for load_dataset_config / build_raw_frame_data
            cfg = {
                "dataset_config_file": dataset_config_file,
                "label_map": label_map,
                "data": {"behavior_classes": sorted(label_map, key=label_map.get)},
                "sequence": {"seq_length": 1, "stride": 1, "frame_interval": 1,
                             "purity_threshold": 1.0, "boundary_margin": 10},
                "preprocessing": {"augmentation": {"enabled": False}},
            }

            from mining.discovery import load_dataset_config, build_raw_frame_data
            ds_info = load_dataset_config(cfg, dataset_config_file, logging.getLogger("correlation"))

            # ---- Load factors ----
            self.set_progress(10, "Loading factors...")
            if not Path(factors_path).exists():
                self.error.emit(f"Factor file not found: {factors_path}")
                self.finished.emit()
                return

            with open(factors_path, "r", encoding="utf-8") as f:
                factors = json.load(f)
            self.log(f"Loaded {len(factors)} factors", 20)

            # ---- Install redirector ----
            redirector = ModuleLogRedirector()
            redirector.log_signal.connect(self._on_worker_log)
            redirector.install()

            # ---- Group factors by seq_length ----
            self.set_progress(20, "Grouping factors by seq_length...")
            groups = defaultdict(list)
            for f in factors:
                sl = f.get("seq_length", 1)
                groups[int(sl)].append(f)

            self.log(f"Groups: {dict((k, len(v)) for k, v in sorted(groups.items()))}", 20)

            # ---- Load training data (raw frames) ----
            self.set_progress(30, "Loading training data...")
            train_data, _, flat_attributes, _video_lengths = build_raw_frame_data(ds_info, logging.getLogger("correlation"), cfg=cfg)
            train_kp, train_lb = train_data

            # Downsample if needed
            max_samples = int(params.get("max_samples", 50000))
            if max_samples > 0 and len(train_kp) > max_samples:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(train_kp), max_samples, replace=False)
                idx.sort()
                train_kp = train_kp[idx]
                train_lb = train_lb[idx]

            self.log(f"Training data: {train_kp.shape[0]} frames", 20)

            # ---- Per-group: compute factor matrix + correlation + dedup ----
            self.set_progress(40, "Computing factor values...")
            from src.synth_validator import SynthValidator

            num_workers = int(params.get("num_workers", 8))
            num_chunks = int(params.get("num_chunks", 256))
            purity_mode = params.get("purity_mode", "nan_boundary")

            sv = SynthValidator({"factor_engine": {"max_error_ratio": 0.5, "min_valid_ratio": 0.1}})
            factors_meta = {fac["name"]: fac for fac in factors}

            log = logging.getLogger("correlation")
            all_kept_names = []
            all_redundant_pairs = []
            total_count = len(factors)
            group_keys = sorted(groups.keys())
            n_groups = len(group_keys)

            for gi, seq_len in enumerate(group_keys):
                if self._cancelled:
                    break

                group_factors = groups[seq_len]
                gname = f"seq_{seq_len}"
                self.set_progress(
                    40 + int((gi / max(n_groups, 1)) * 40),
                    f"Group {gname}: computing {len(group_factors)} factors...",
                )
                self.log(f"\n--- Group {gname}: {len(group_factors)} factors ---", 20)

                try:
                    if num_workers > 1:
                        X, _, used_names, dropped_names = (
                            sv.compute_factor_matrix_multiresolution_fast(
                                factors=group_factors,
                                kp_full=train_kp,
                                labels=train_lb,
                                flat_attributes=flat_attributes,
                                purity_mode=purity_mode,
                                num_workers=num_workers,
                                num_chunks=num_chunks,
                                split_name=f"corr_{gname}",
                            )
                        )
                    else:
                        X, _, used_names, dropped_names = (
                            sv.compute_factor_matrix_multiresolution(
                                factors=group_factors,
                                kp_full=train_kp,
                                labels=train_lb,
                                flat_attributes=flat_attributes,
                                purity_mode=purity_mode,
                                split_name=f"corr_{gname}",
                            )
                        )
                except RuntimeError as e:
                    self.log(f"  Group {gname} factor computation failed: {e}", 30)
                    continue

                if len(used_names) == 0:
                    self.log(f"  Group {gname}: no valid factors, skipping.", 20)
                    continue

                # Correlation matrix
                corr = _pearson_corr_matrix(X, log)

                # Greedy dedup per group
                kept, pairs = _greedy_dedup(
                    factor_names=used_names,
                    factors_meta=factors_meta,
                    corr=corr,
                    threshold=threshold,
                    keep_by=keep_by,
                    log=log,
                )

                all_kept_names.extend(kept)
                all_redundant_pairs.extend(pairs)

            if not all_kept_names:
                self.error.emit("No valid factor values computed")
                self.finished.emit()
                return

            n_after = len(all_kept_names)
            self.log(f"\nTotal dedup: {total_count} -> {n_after} factors (removed {len(all_redundant_pairs)})", 20)

            result = {
                "success": True,
                "n_total": total_count,
                "n_after": n_after,
                "threshold": threshold,
                "redundant_pairs": all_redundant_pairs[:100],
                "message": f"Analysis complete. {total_count} -> {n_after} factors (threshold={threshold}).",
            }

            self.set_progress(100, "Correlation analysis complete")
            self.result_ready.emit(result)

        except Exception as e:
            self.log(f"Correlation analysis failed: {e}\n{traceback.format_exc()}", 40)
            self.error.emit(str(e))
        finally:
            if redirector:
                redirector.uninstall()
            # Clean up temp dataset config file
            try:
                if dataset_config_file and os.path.exists(dataset_config_file):
                    os.unlink(dataset_config_file)
            except Exception:
                pass
            self.finished.emit()

    @Slot(str, int)
    def _on_worker_log(self, msg, level):
        self.log_line.emit(msg, level)
