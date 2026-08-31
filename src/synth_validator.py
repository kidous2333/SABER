"""
synth_validator.py
Factor synthesis validation module.

Takes all valid factors from memory/valid_factors.json as a unified input
to a LightGBM multiclass classifier, using class_weight='balanced' to
handle class imbalance. Outputs on the val set:
  - Confusion matrix (raw counts + normalized)
  - macro/weighted AUC
  - macro/weighted F1
  - Per-class precision / recall / f1 / support / auc
"""

import csv
import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

import atexit
import gc
import glob
import multiprocessing as mp
import os
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.factor_engine import _SAFE_BUILTINS


# ------------------------------------------------------------------
# Temp file management (memmap sharing)
# ------------------------------------------------------------------

def _cleanup_tmp_file(path: str) -> None:
    """Remove a single memmap temp file if it exists. Idempotent."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _cleanup_stale_tmp_files(cache_dir: str) -> None:
    """Clean up _tmp_win_*.dat files whose owner process is dead."""
    pattern = os.path.join(cache_dir, "_tmp_win_*.dat")
    for f in glob.glob(pattern):
        basename = os.path.basename(f)
        try:
            stem = basename.replace("_tmp_win_", "").replace(".dat", "")
            pid = int(stem.split("_")[0])
        except (ValueError, IndexError):
            continue
        try:
            os.kill(pid, 0)
        except (OSError, Exception):
            try:
                os.remove(f)
            except Exception:
                pass


# ------------------------------------------------------------------
# Parallel worker function (module-level, required picklable for Windows spawn)
# ------------------------------------------------------------------

def _mp_compute_batch_factor(
    tmp_path: str,
    shape: tuple,
    dtype_str: str,
    flat_attributes: list,
    factor_dict: dict,
) -> tuple:
    """
    Worker process: compute one batch-mode factor from memmap windows.

    Returns (factor_name, result_array) or (factor_name, None) on failure.
    Does NOT log, validate NaN-ratio, or apply purity masks — those are
    the main processʼs responsibility.
    """
    import numpy as np
    from src.factor_engine import _SAFE_BUILTINS as _BLT

    windows = np.memmap(tmp_path, dtype=np.dtype(dtype_str), mode="r", shape=shape)
    name = factor_dict["name"]
    code = factor_dict["code"]
    name_to_idx = {n: i for i, n in enumerate(flat_attributes)}

    # Reproduce FactorEngine.compute_factor_batch compilation logic
    func_code = "def _factor_func(windows, idx, np):\n"
    func_code += "    with np.errstate(invalid='ignore', divide='ignore'):\n"
    for line in code.strip().split("\n"):
        func_code += f"        {line}\n"
    func_code += "        return _result\n"

    namespace = {"__builtins__": _BLT}
    try:
        exec(compile(func_code, "<factor_batch>", "exec"), namespace)
        result = namespace["_factor_func"](windows, name_to_idx, np)
    except Exception:
        return (name, None)

    if result is None:
        return (name, None)

    result = np.asarray(result, dtype=np.float32)
    if result.ndim == 0:
        result = np.full(len(windows), float(result), dtype=np.float32)

    result[~np.isfinite(result)] = np.nan
    return (name, result)


def _write_csv(path: Path, rows: list, header: list) -> None:
    """Write CSV alongside a plot PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
import lightgbm as lgb

from src.factor_engine import FactorEngine
from src.visualization import (
    plot_factor_valid_ratio,
    plot_factor_distribution,
    plot_standardize_effect,
    plot_feature_importance,
    plot_proba_distribution,
    plot_meta_correlation,
    plot_group_confusion_matrix,
    plot_learning_curve,
    plot_per_class_metrics_comparison,
    plot_confidence_vs_correctness,
    plot_best_model_metrics_table,
)

logger = logging.getLogger(__name__)


def _group_factors_by_seqlength(factors: list) -> dict:
    """Group factors by seq_length into short/medium/long, filtering empty groups."""
    groups = {"short": [], "medium": [], "long": []}
    for f in factors:
        sl = int(f.get("seq_length", 1))
        if sl <= 3:
            groups["short"].append(f)
        elif sl <= 15:
            groups["medium"].append(f)
        else:
            groups["long"].append(f)
    return {k: v for k, v in groups.items() if v}


def _train_single_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classes_sorted: list,
    lgbm_params: dict,
    class_weight,
    early_stopping_rounds: int,
    return_evals_result: bool = False,
    return_model: bool = False,
) -> tuple:
    """Train a single LGBM, returns (proba_train [T_tr,C], proba_val [T_vl,C]) aligned to the full class order.

    If return_evals_result=True, additionally returns evals_result dict.
    If return_model=True, additionally returns the trained LGBMClassifier model object.
    """
    import gc
    n_classes = len(classes_sorted)
    # Dynamic min_data_in_leaf: prevent split failure when samples are scarce.
    # Use max() (not min()) so the safeguard floor is always honoured.
    # Wide factor matrices (K > 500) need extra headroom to avoid
    # "left_count > 0" assertion failures on degenerate splits.
    safe_params = dict(lgbm_params)
    min_leaf = max(1, len(y_train) // (n_classes * 10))
    if X_train.shape[1] > 500:
        min_leaf = max(min_leaf, n_classes * 50)
    safe_params.setdefault("min_data_in_leaf", min_leaf)
    safe_params["min_data_in_leaf"] = max(safe_params["min_data_in_leaf"], min_leaf)
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        class_weight=class_weight if class_weight else None,
        verbose=-1,
        **safe_params,
    )
    evals_result = {}
    fit_kwargs = {}
    callbacks = []
    if early_stopping_rounds > 0:
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]
        fit_kwargs["eval_names"] = ["training", "valid_1"]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=50))
    if return_evals_result:
        callbacks.append(lgb.record_evaluation(evals_result))
    if callbacks:
        fit_kwargs["callbacks"] = callbacks
    model.fit(X_train, y_train, **fit_kwargs)

    model_classes = list(model.classes_)
    raw_tr = model.predict_proba(X_train).astype(np.float32)
    raw_vl = model.predict_proba(X_val).astype(np.float32)

    if not return_model:
        del model; gc.collect()
        _model = None
    else:
        _model = model

    def _align(raw_proba: np.ndarray) -> np.ndarray:
        aligned = np.zeros((len(raw_proba), n_classes), dtype=np.float32)
        for i, c in enumerate(model_classes):
            if c in classes_sorted:
                aligned[:, classes_sorted.index(c)] = raw_proba[:, i]
        return aligned

    p_tr = _align(raw_tr)
    p_vl = _align(raw_vl)
    del raw_tr, raw_vl
    if return_evals_result and return_model:
        return p_tr, p_vl, evals_result, _model
    if return_evals_result:
        return p_tr, p_vl, evals_result
    if return_model:
        return p_tr, p_vl, _model
    return p_tr, p_vl


class SynthValidator:
    """Factor synthesis validator: trains a multiclass LGBM using all valid factors as a feature matrix."""

    DEFAULT_LGBM_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self, cfg: dict):
        self.cfg = cfg
        synth_cfg = cfg.get("synth_validation", {}) or {}
        self.lgbm_params = {**self.DEFAULT_LGBM_PARAMS, **synth_cfg.get("lgbm_params", {})}
        self.nan_fill = synth_cfg.get("nan_fill", "mean")
        self.normalize_cm = synth_cfg.get("normalize_cm", True)
        self.class_weight = synth_cfg.get("class_weight", "balanced")
        self.early_stopping_rounds = synth_cfg.get("early_stopping_rounds", 50)
        self.top_k = int(synth_cfg.get("top_k", 2))
        self.feature_selection_top_k = int(synth_cfg.get("feature_selection_top_k", 0))
        self.use_gpu = bool(synth_cfg.get("use_gpu", False))
        if self.use_gpu:
            self.lgbm_params["device"] = "gpu"
            self.lgbm_params.pop("n_jobs", None)  # n_jobs is invalid in GPU mode
            logger.info("LightGBM GPU acceleration enabled (synth_validation)")
        self.residual_stage1 = bool(synth_cfg.get("residual_stage1", False))
        self.residual_stage2 = bool(synth_cfg.get("residual_stage2", False))
        _top_k = synth_cfg.get("top_k_per_group", 0)
        if isinstance(_top_k, dict):
            self.top_k_per_group = {str(k): int(v) for k, v in _top_k.items()}
            logger.info(f"top_k_per_group configured per-group: {self.top_k_per_group}")
        else:
            _val = int(_top_k)
            self.top_k_per_group = _val
            if _val > 0:
                logger.info(f"top_k_per_group={_val} (retain top-k per group by mean_auc)")
        self.tmp_dir = synth_cfg.get("tmp_dir", "")
        self.engine = FactorEngine(cfg)

    def _get_top_k(self, group_name: str = None) -> int:
        """Return the top_k limit for a given group (dict mode) or the global limit (int mode).

        group_name only valid in dict mode; returns 0 when None (disabled for non-group methods).
        """
        if isinstance(self.top_k_per_group, dict):
            return self.top_k_per_group.get(group_name, 0) if group_name else 0
        return self.top_k_per_group

    # ------------------------------------------------------------------
    # 1) Load valid factors
    # ------------------------------------------------------------------
    @staticmethod
    def load_valid_factors(memory_path: str) -> list:
        p = Path(memory_path)
        if not p.exists():
            raise FileNotFoundError(
                f"valid_factors.json not found: {p}. Please run discovery.py first to discover factors."
            )
        with open(p, "r", encoding="utf-8") as f:
            factors = json.load(f)
        if not factors:
            raise ValueError(f"{p} is empty, no valid factors found.")
        logger.info(f"Loaded {len(factors)} valid factors: {p}")
        return factors

    # ------------------------------------------------------------------
    # 1b) Filter factors by correlation report
    # ------------------------------------------------------------------
    @staticmethod
    def filter_factors_by_correlation(
        factors: list,
        corr_report_path: str,
        threshold: float = None,
    ) -> list:
        """
        Filter the factor list based on the correlation report generated by correlation.py.

        corr_report_path: path to factor_correlation.json
        threshold: if specified, re-filter from the report using this threshold
                   (report must contain corr_sparse);
                   if None, directly use the recommended_factors list from the report.

        Returns the filtered factor list (preserving original order).
        """
        p = Path(corr_report_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Correlation report not found: {p}. Please run correlation.py first."
            )
        with open(p, "r", encoding="utf-8") as f:
            report = json.load(f)

        if threshold is None:
            # Directly use the precomputed recommended list from the report
            kept_set = set(report.get("recommended_factors", []))
            report_threshold = report.get("threshold", "?")
            logger.info(
                f"[corr filter] Using report recommended list (threshold={report_threshold}): "
                f"{len(kept_set)} recommended factors"
            )
        else:
            # Re-do greedy redundancy removal with specified threshold
            from collections import defaultdict

            def _score(fac):
                return fac.get("best_auc", 0.0)

            # Collect corr_sparse per group and redo redundancy removal
            kept_set = set()
            for gname, g in report.get("groups", {}).items():
                used_names = g.get("used_factor_names", [])
                corr_sparse = g.get("corr_sparse", {})
                if not used_names:
                    continue
                n = len(used_names)
                name_idx = {name: i for i, name in enumerate(used_names)}
                corr_mat = np.zeros((n, n), dtype=np.float32)
                np.fill_diagonal(corr_mat, 1.0)
                for key, val in corr_sparse.items():
                    parts = key.split("|", 1)
                    if len(parts) == 2 and parts[0] in name_idx and parts[1] in name_idx:
                        i, j = name_idx[parts[0]], name_idx[parts[1]]
                        corr_mat[i, j] = abs(val)
                        corr_mat[j, i] = abs(val)

                fac_meta = {fac["name"]: fac for fac in factors}
                order = sorted(range(n), key=lambda i: _score(fac_meta.get(used_names[i], {})), reverse=True)
                kept_idx = []
                for i in order:
                    if all(corr_mat[i, j] < threshold for j in kept_idx):
                        kept_idx.append(i)
                kept_set.update(used_names[i] for i in kept_idx)

            logger.info(
                f"[corr filter] Using custom threshold={threshold}: {len(kept_set)} recommended factors"
            )

        original_count = len(factors)
        filtered = [f for f in factors if f.get("name") in kept_set]
        removed_count = original_count - len(filtered)
        logger.info(
            f"[corr filter] Factor filtering: {original_count} -> {len(filtered)}, removed {removed_count} high-correlation redundant factors"
        )
        return filtered

    # ------------------------------------------------------------------
    # 2) Compute factor matrix
    # ------------------------------------------------------------------
    def compute_factor_matrix(
        self,
        factors: list,
        keypoints: np.ndarray,
        flat_attributes: list,
        split_name: str = "",
    ) -> tuple:
        """
        Returns (X [N, K_used], used_factor_names, dropped_factor_names).
        keypoints: [N, seq_length, D]
        """
        N = keypoints.shape[0]
        columns = []
        used_names = []
        dropped = []

        for fac_idx, fac in enumerate(factors, 1):
            name = fac["name"]
            mode = fac.get("mode", "row")
            logger.info(f"[{split_name}] Computing factor {fac_idx}/{len(factors)}: {name}")
            try:
                if mode == "batch":
                    vals = self.engine.compute_factor_batch(fac, keypoints, flat_attributes)
                else:
                    vals = self.engine.compute_factor(fac, keypoints, flat_attributes)
            except Exception as e:
                logger.warning(f"[{split_name}] Factor '{name}' computation exception: {e}")
                vals = None

            if vals is None:
                logger.warning(f"[{split_name}] Factor '{name}' computation failed, dropping.")
                dropped.append(name)
                continue

            if len(vals) != N:
                logger.warning(
                    f"[{split_name}] Factor '{name}' length {len(vals)} != expected {N}, dropping."
                )
                dropped.append(name)
                continue

            if not np.isfinite(vals).any():
                logger.warning(f"[{split_name}] Factor '{name}' all NaN, dropping.")
                dropped.append(name)
                continue

            columns.append(vals.astype(np.float32))
            used_names.append(name)

        if not columns:
            raise RuntimeError(f"[{split_name}] All factors failed to compute.")

        X = np.stack(columns, axis=1)  # [N, K_used]
        logger.info(
            f"[{split_name}] Factor matrix shape={X.shape}, using {len(used_names)}, dropped {len(dropped)}"
        )
        return X, used_names, dropped

    # ------------------------------------------------------------------
    # 2b) Fast batch factor matrix computation (fully vectorized batch mode)
    # ------------------------------------------------------------------
    def compute_factor_matrix_fast(
        self,
        factors: list,
        keypoints: np.ndarray,
        flat_attributes: list,
        num_chunks: int = 256,
        num_workers: int = 8,
        split_name: str = "",
        cache_dir: str = "dataset_cache",
    ) -> tuple:
        """
        Batch version: all factors in batch mode, main process computes vectorially per factor.
        keypoints: [N, seq_length, D]
        Returns the same triple as compute_factor_matrix (X [N, K_used], used_names, dropped).
        """
        N = keypoints.shape[0]

        columns = []
        used_names = []
        dropped = []

        # ── Parallel / Serial branch ──────────────────────────────
        if num_workers > 1 and len(factors) > 1:
            # === Multi-process parallel ===
            os.makedirs(cache_dir, exist_ok=True)
            _cleanup_stale_tmp_files(cache_dir)

            pid = os.getpid()
            tmp_path = os.path.join(cache_dir, f"_tmp_win_{pid}_fixed.dat")
            atexit.register(_cleanup_tmp_file, tmp_path)

            win_shape = keypoints.shape
            win_dtype = keypoints.dtype
            _t_mmap = _time.time()
            try:
                win_mmap = np.memmap(
                    tmp_path, dtype=win_dtype, mode="w+", shape=win_shape,
                )
                win_mmap[:] = keypoints[:]
                win_mmap.flush()
                del win_mmap
            except OSError:
                _cleanup_stale_tmp_files(cache_dir)
                win_mmap = np.memmap(
                    tmp_path, dtype=win_dtype, mode="w+", shape=win_shape,
                )
                win_mmap[:] = keypoints[:]
                win_mmap.flush()
                del win_mmap
            logger.info(
                f"[{split_name}] memmap written {tmp_path} "
                f"({_time.time() - _t_mmap:.1f}s)"
            )

            ctx = mp.get_context("spawn")
            n_workers = min(num_workers, len(factors))
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                future_map = {}
                for fac in factors:
                    fut = pool.submit(
                        _mp_compute_batch_factor,
                        tmp_path, win_shape, str(win_dtype),
                        flat_attributes, fac,
                    )
                    future_map[fut] = fac["name"]

                n_facs = len(factors)
                done = 0
                milestone = max(1, n_facs // 10)
                for future in as_completed(future_map):
                    name = future_map[future]
                    try:
                        _, vals = future.result()
                    except Exception:
                        logger.warning(
                            f"[{split_name}] Factor '{name}' worker exception"
                        )
                        dropped.append(name)
                        done += 1
                        continue

                    if vals is None or len(vals) != N or not np.isfinite(vals).any():
                        logger.warning(f"[{split_name}] Factor '{name}' invalid, dropping.")
                        dropped.append(name)
                        done += 1
                        continue

                    nan_ratio = np.isnan(vals).mean()
                    if nan_ratio > (1 - self.engine.min_valid_ratio):
                        logger.warning(f"[{split_name}] Factor '{name}' insufficient valid values, dropping.")
                        dropped.append(name)
                        done += 1
                        continue

                    columns.append(vals.astype(np.float32))
                    used_names.append(name)
                    done += 1

                    if done % milestone == 0 or done == n_facs:
                        logger.info(
                            f"[{split_name}] {done}/{n_facs} factors completed"
                        )

            _cleanup_tmp_file(tmp_path)
        else:
            # === Single-process serial (original logic) ===
            for fac_idx, fac in enumerate(factors, 1):
                name = fac["name"]
                logger.info(f"[{split_name}] Factor {fac_idx}/{len(factors)}: {name}")
                try:
                    vals = self.engine.compute_factor_batch(fac, keypoints, flat_attributes)
                except Exception as e:
                    logger.warning(f"[{split_name}] Factor '{name}' exception: {e}")
                    vals = None
                if vals is None or len(vals) != N or not np.isfinite(vals).any():
                    logger.warning(f"[{split_name}] Factor '{name}' invalid, dropping.")
                    dropped.append(name)
                    continue
                nan_ratio = np.isnan(vals).mean()
                if nan_ratio > (1 - self.engine.min_valid_ratio):
                    logger.warning(f"[{split_name}] Factor '{name}' insufficient valid values, dropping.")
                    dropped.append(name)
                    continue
                columns.append(vals.astype(np.float32))
                used_names.append(name)

        if not columns:
            raise RuntimeError(f"[{split_name}] All factors failed to compute.")

        X = np.stack(columns, axis=1)
        logger.info(
            f"[{split_name}] Factor matrix shape={X.shape}, "
            f"using {len(used_names)}, dropped {len(dropped)}"
        )
        return X, used_names, dropped

    # ------------------------------------------------------------------
    # Multi-resolution helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _build_centered_windows(kp_full: np.ndarray, seq_length: int,
                                video_lengths: list = None) -> np.ndarray:
        """
        kp_full: [T, D]
        Returns a window matrix [T, seq_length, D] centered on each frame.

        When video_lengths is provided (list of per-video frame counts),
        each video's windows are built independently with edge-padding
        restricted to that video — windows never cross video boundaries.
        """
        if seq_length == 1:
            return kp_full[:, np.newaxis, :]

        if video_lengths is None or len(video_lengths) <= 1:
            T, D = kp_full.shape
            half_before = (seq_length - 1) // 2
            half_after  = seq_length - 1 - half_before
            kp_padded = np.pad(kp_full, ((half_before, half_after), (0, 0)), mode='edge')
            idx = np.arange(T)[:, None] + np.arange(seq_length)[None, :]
            return kp_padded[idx]

        # Per-video window building
        D = kp_full.shape[1]
        half_before = (seq_length - 1) // 2
        half_after  = seq_length - 1 - half_before
        windows_list = []
        offset = 0
        for vlen in video_lengths:
            vlen = int(vlen)
            if vlen == 0:
                continue
            vkp = kp_full[offset:offset + vlen]  # [vlen, D]
            vpadded = np.pad(vkp, ((half_before, half_after), (0, 0)), mode='edge')
            idx = np.arange(vlen)[:, None] + np.arange(seq_length)[None, :]
            windows_list.append(vpadded[idx])
            offset += vlen
        return np.concatenate(windows_list, axis=0)  # [T, seq_length, D]

    @staticmethod
    def _compute_purity_mask(labels: np.ndarray, seq_length: int) -> np.ndarray:
        """
        Returns boolean mask [T]: True means all labels within the centered seq_length window are consistent.
        Out-of-bounds positions are filled with -1 (deemed impure; edge frames are marked as impure).
        """
        T = len(labels)
        if seq_length <= 1:
            return np.ones(T, dtype=bool)
        half_before = (seq_length - 1) // 2
        half_after  = seq_length - 1 - half_before
        labels_padded = np.pad(
            labels.astype(np.int64), (half_before, half_after),
            mode='constant', constant_values=-1,
        )
        center = labels  # [T]
        mask = np.ones(T, dtype=bool)
        for offset in range(seq_length):
            mask &= (labels_padded[offset : offset + T] == center)
        return mask

    def compute_factor_matrix_multiresolution(
        self,
        factors: list,
        kp_full: np.ndarray,
        labels: np.ndarray,
        flat_attributes: list,
        purity_mode: str = "nan_boundary",
        split_name: str = "",
        video_lengths: list = None,
    ) -> tuple:
        """
        Multi-resolution (single-process) factor matrix computation.

        Each factor builds a window centered on the validation frame according to its seq_length,
        computing factor values independently. All factors correspond to the same frame index;
        labels are taken from the center frame.

        purity_mode:
          "nan_boundary"  (default): when a multi-frame window crosses a behavior boundary,
                                     set that frame's factor value to NaN;
                                     NaN is filled with training mean during standardization,
                                     frames are not dropped.
          "none"          : no purity processing, compute directly.
          "strict"        : only keep frames where all factors' maximum windows are pure
                            (may drop many boundary frames).

        Returns (X [T_out, K], labels_out [T_out], used_names, dropped_names)
        """
        T = len(labels)
        from collections import defaultdict
        by_sl: dict = defaultdict(list)
        for fac in factors:
            by_sl[fac.get("seq_length", 1)].append(fac)

        # Pre-build window matrices and purity masks for each seq_length
        windows_cache = {}
        purity_masks  = {}
        for sl in by_sl:
            windows_cache[sl] = self._build_centered_windows(kp_full, sl, video_lengths)
            purity_masks[sl]  = self._compute_purity_mask(labels, sl)

        # strict: take the intersection of all seq_length purity masks
        strict_mask = np.ones(T, dtype=bool)
        if purity_mode == "strict":
            for sl in by_sl:
                strict_mask &= purity_masks[sl]
            logger.info(f"[{split_name}] strict purity: {strict_mask.sum()}/{T} frames retained")

        columns, used_names, dropped = [], [], []

        for fac_idx, fac in enumerate(factors, 1):
            name = fac["name"]
            sl   = fac.get("seq_length", 1)
            mode = fac.get("mode", "row")
            windows = windows_cache[sl]
            logger.info(f"[{split_name}] Computing factor {fac_idx}/{len(factors)}: {name} (seq_length={sl})")
            try:
                if mode == "batch":
                    vals = self.engine.compute_factor_batch(fac, windows, flat_attributes)
                else:
                    vals = self.engine.compute_factor(fac, windows, flat_attributes)
            except Exception as e:
                logger.warning(f"[{split_name}] Factor '{name}' computation error: {e}")
                dropped.append(name)
                continue

            if vals is None or len(vals) != T or not np.isfinite(vals).any():
                logger.warning(f"[{split_name}] Factor '{name}' invalid, removed.")
                dropped.append(name)
                continue

            vals = vals.astype(np.float32)
            if purity_mode == "nan_boundary" and sl > 1:
                vals[~purity_masks[sl]] = np.nan
            columns.append(vals)
            used_names.append(name)

        if not columns:
            raise RuntimeError(f"[{split_name}] All factors failed to compute.")

        X = np.stack(columns, axis=1)  # [T, K]
        out_labels = labels
        if purity_mode == "strict":
            X          = X[strict_mask]
            out_labels = labels[strict_mask]

        logger.info(
            f"[{split_name}] Multi-resolution factor matrix shape={X.shape},"
            f"using {len(used_names)}, dropped {len(dropped)}"
        )
        return X, out_labels, used_names, dropped

    def compute_factor_matrix_multiresolution_fast(
        self,
        factors: list,
        kp_full: np.ndarray,
        labels: np.ndarray,
        flat_attributes: list,
        purity_mode: str = "nan_boundary",
        num_chunks: int = 256,
        num_workers: int = 8,
        split_name: str = "",
        cache_dir: str = "dataset_cache",
        video_lengths: list = None,
    ) -> tuple:
        """
        Multi-resolution factor matrix computation (fully batch vectorized).

        Group by seq_length: each group builds its window matrix, all factors computed by main process in batch.

        Returns the same quadruple as compute_factor_matrix_multiresolution.
        """
        T = len(labels)
        from collections import defaultdict
        by_sl: dict = defaultdict(list)
        for fac in factors:
            by_sl[fac.get("seq_length", 1)].append(fac)

        purity_masks = {
            sl: self._compute_purity_mask(labels, sl) for sl in by_sl
        }
        strict_mask = np.ones(T, dtype=bool)
        if purity_mode == "strict":
            for sl in by_sl:
                strict_mask &= purity_masks[sl]
            logger.info(f"[{split_name}] strict purity: {strict_mask.sum()}/{T} frames retained")

        all_columns: dict = {}
        dropped: list = []

        sorted_groups = sorted(by_sl.items())
        total_groups = len(sorted_groups)
        for gi, (sl, sl_factors) in enumerate(sorted_groups, 1):
            logger.info(
                f"[{split_name}] Subgroup {gi}/{total_groups}: seq_length={sl}, "
                f"{len(sl_factors)} factors, building windows..."
            )
            _t_win = _time.time()
            windows      = self._build_centered_windows(kp_full, sl, video_lengths)   # [T, sl, D]
            logger.info(
                f"[{split_name}] Window matrix shape={windows.shape} "
                f"({windows.nbytes / 1024**2:.0f} MB), "
                f"build time {_time.time() - _t_win:.1f}s"
            )
            if windows.shape[0] == 0:
                logger.warning(
                    f"[{split_name}] Empty window matrix (0 frames), skipping subgroup seq_length={sl}"
                )
                continue
            purity_mask  = purity_masks[sl]
            apply_purity = purity_mode == "nan_boundary" and sl > 1

            # ── Parallel / Serial branch ──────────────────────────────
            if num_workers > 1 and len(sl_factors) > 1:
                # === Multi-process parallel ===
                os.makedirs(cache_dir, exist_ok=True)
                _cleanup_stale_tmp_files(cache_dir)

                pid = os.getpid()
                tmp_path = os.path.join(cache_dir, f"_tmp_win_{pid}_{sl}.dat")
                atexit.register(_cleanup_tmp_file, tmp_path)

                # Remove any leftover file (e.g. empty file from crashed run with same PID)
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

                win_shape = windows.shape
                win_dtype = windows.dtype
                _t_mmap = _time.time()
                try:
                    win_mmap = np.memmap(
                        tmp_path, dtype=win_dtype, mode="w+", shape=win_shape,
                    )
                    win_mmap[:] = windows[:]
                    win_mmap.flush()
                    del win_mmap
                except (OSError, ValueError) as e:
                    logger.warning(
                        f"[{split_name}] memmap write failed ({e}), retrying after cleanup..."
                    )
                    _cleanup_stale_tmp_files(cache_dir)
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                    win_mmap = np.memmap(
                        tmp_path, dtype=win_dtype, mode="w+", shape=win_shape,
                    )
                    win_mmap[:] = windows[:]
                    win_mmap.flush()
                    del win_mmap
                logger.info(
                    f"[{split_name}] memmap wrote {tmp_path} "
                    f"({_time.time() - _t_mmap:.1f}s)"
                )

                del windows
                gc.collect()

                ctx = mp.get_context("spawn")
                n_workers = min(num_workers, len(sl_factors))
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                    future_map = {}
                    for fac in sl_factors:
                        fut = pool.submit(
                            _mp_compute_batch_factor,
                            tmp_path, win_shape, str(win_dtype),
                            flat_attributes, fac,
                        )
                        future_map[fut] = fac["name"]

                    n_facs = len(sl_factors)
                    done = 0
                    milestone = max(1, n_facs // 10)
                    for future in as_completed(future_map):
                        name = future_map[future]
                        try:
                            _, vals = future.result()
                        except Exception:
                            logger.warning(
                                f"[{split_name}] Factor '{name}' worker error"
                            )
                            dropped.append(name)
                            done += 1
                            continue

                        if vals is None or len(vals) != T or not np.isfinite(vals).any():
                            dropped.append(name)
                            done += 1
                            continue

                        vals = vals.astype(np.float32)
                        if apply_purity:
                            vals[~purity_mask] = np.nan
                        all_columns[name] = vals
                        done += 1

                        if done % milestone == 0 or done == n_facs:
                            logger.info(
                                f"[{split_name}] seq_length={sl}: "
                                f"{done}/{n_facs} factors completed"
                            )

                _cleanup_tmp_file(tmp_path)
            else:
                # === Single-process serial (original logic) ===
                for fac in sl_factors:
                    name = fac["name"]
                    logger.info(f"[{split_name}] Factor '{name}' (seq_length={sl})")
                    try:
                        vals = self.engine.compute_factor_batch(fac, windows, flat_attributes)
                    except Exception as e:
                        logger.warning(f"[{split_name}] Factor '{name}' error: {e}")
                        dropped.append(name)
                        continue
                    if vals is None or len(vals) != T or not np.isfinite(vals).any():
                        dropped.append(name)
                        continue
                    vals = vals.astype(np.float32)
                    if apply_purity:
                        vals[~purity_mask] = np.nan
                    all_columns[name] = vals

                del windows
                gc.collect()

            # This seq_length group processing complete
            n_ok = sum(1 for f in sl_factors if f["name"] in all_columns)
            n_drop = sum(1 for f in sl_factors if f["name"] in dropped)
            logger.info(
                f"[{split_name}] Subgroup {gi}/{total_groups} completed: "
                f"seq_length={sl}, success {n_ok}, dropped {n_drop}"
            )

        used_names = [f["name"] for f in factors if f["name"] in all_columns]
        for fac in factors:
            if fac["name"] not in all_columns and fac["name"] not in dropped:
                dropped.append(fac["name"])

        if not used_names:
            raise RuntimeError(f"[{split_name}] All factors failed to compute.")

        columns    = [all_columns[n] for n in used_names]
        out_labels = labels
        if purity_mode == "strict":
            columns    = [c[strict_mask] for c in columns]
            out_labels = labels[strict_mask]

        X = np.stack(columns, axis=1)
        logger.info(
            f"[{split_name}] Parallel multi-resolution factor matrix shape={X.shape},"
            f"using {len(used_names)}, dropped {len(dropped)}"
        )
        return X, out_labels, used_names, dropped

    def run_multiresolution(
        self,
        factors_path: str,
        train_kp_raw: np.ndarray,
        train_lb_raw: np.ndarray,
        val_kp_raw: np.ndarray,
        val_lb_raw: np.ndarray,
        flat_attributes: list,
        label_map: dict,
        output_dir: str,
        purity_mode: str = "nan_boundary",
        num_workers: int = 1,
        num_chunks: int = 256,
        cache_dir: str = "dataset_cache",
    ) -> dict:
        """
        Multi-resolution end-to-end entry.

        train_kp_raw / val_kp_raw: [T, D] raw per-frame keypoints (not windowed)
        train_lb_raw / val_lb_raw: [T] per-frame labels

        Difference from run(): input is raw frame sequence rather than pre-sliced fixed-width windows,
        each factor constructs its own centered window at inference time according to its seq_length.
        """
        factors = self.load_valid_factors(factors_path)

        # ---- Select top-k factors by mean_auc ----
        _tk = self._get_top_k()
        if _tk > 0 and len(factors) > _tk:
            for f in factors:
                pc = f.get("per_class", {})
                aucs = [v["auc"] if isinstance(v, dict) else v for v in pc.values()] if pc else [0.0]
                f["_mean_auc"] = float(np.mean(aucs))
            n_before = len(factors)
            factors = sorted(factors, key=lambda x: x["_mean_auc"], reverse=True)[:_tk]
            logger.info(
                f"[run_multiresolution] top-k selection: {n_before} -> {len(factors)} factors"
                f" (mean_auc cutoff={factors[-1]['_mean_auc']:.4f})"
            )

        if num_workers > 1:
            logger.info(f"Multi-resolution parallel mode: num_workers={num_workers}, num_chunks={num_chunks}")
            X_train, train_lb, used_tr, dropped_tr = self.compute_factor_matrix_multiresolution_fast(
                factors, train_kp_raw, train_lb_raw, flat_attributes,
                purity_mode=purity_mode, num_chunks=num_chunks,
                num_workers=num_workers, split_name="train", cache_dir=cache_dir,
            )
            X_val, val_lb, used_vl, dropped_vl = self.compute_factor_matrix_multiresolution_fast(
                factors, val_kp_raw, val_lb_raw, flat_attributes,
                purity_mode=purity_mode, num_chunks=num_chunks,
                num_workers=num_workers, split_name="val", cache_dir=cache_dir,
            )
        else:
            logger.info("Multi-resolution single-process mode (--num-workers > 1 enables parallel)")
            X_train, train_lb, used_tr, dropped_tr = self.compute_factor_matrix_multiresolution(
                factors, train_kp_raw, train_lb_raw, flat_attributes,
                purity_mode=purity_mode, split_name="train",
            )
            X_val, val_lb, used_vl, dropped_vl = self.compute_factor_matrix_multiresolution(
                factors, val_kp_raw, val_lb_raw, flat_attributes,
                purity_mode=purity_mode, split_name="val",
            )

        # Align common factor columns between train/val
        common = [n for n in used_tr if n in set(used_vl)]
        if len(common) != len(used_tr) or len(common) != len(used_vl):
            logger.warning(
                f"train/val successful factors differ, taking intersection of {len(common)};"
                f"train unique: {set(used_tr)-set(common)}, val unique: {set(used_vl)-set(common)}"
            )
            tr_idx = [used_tr.index(n) for n in common]
            vl_idx = [used_vl.index(n) for n in common]
            X_train = X_train[:, tr_idx]
            X_val   = X_val[:, vl_idx]

        dropped_all = sorted(set(dropped_tr) | set(dropped_vl))
        X_train_std, X_val_std, scaler, fill = self.standardize(X_train, X_val)

        eval_result, cm, cm_norm, class_names = self.train_and_eval(
            X_train_std, train_lb, X_val_std, val_lb, label_map
        )
        eval_result.update({
            "timestamp": datetime.now().isoformat(),
            "n_factors_used": len(common),
            "factors_used": common,
            "factors_dropped": dropped_all,
            "label_map": label_map,
            "lgbm_params": self.lgbm_params,
            "purity_mode": purity_mode,
            "multiresolution": True,
        })
        files = self.save_report(eval_result, cm, cm_norm, class_names, output_dir)
        eval_result["output_files"] = files
        return eval_result

    # ------------------------------------------------------------------
    # Multi-scale stacking + temporal second-stage validation
    # ------------------------------------------------------------------
    @staticmethod
    def _load_pretrained_group_model(weights_dir: Path, group_name: str,
                                     n_classes: int):
        """Load a pre-trained group LGBM model and scaler params from weights/.

        Returns (model, scaler_mean, scaler_scale, fill_vals) or raises
        FileNotFoundError if any required file is missing.
        """
        import lightgbm as lgb

        model_path = weights_dir / f"group_{group_name}_lgbm.txt"
        mean_path = weights_dir / f"scaler_mean_{group_name}.npy"
        scale_path = weights_dir / f"scaler_scale_{group_name}.npy"
        fill_path = weights_dir / f"fill_{group_name}.npy"

        missing = []
        for p, label in [(model_path, "model"), (mean_path, "scaler_mean"),
                         (scale_path, "scaler_scale"), (fill_path, "fill")]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                f"[Inference] Missing required files for group '{group_name}': "
                + ", ".join(missing))

        model = lgb.Booster(model_file=str(model_path))
        scaler_mean = np.load(str(mean_path))
        scaler_scale = np.load(str(scale_path))
        fill_vals = np.load(str(fill_path))
        logger.info(
            f"[Inference] Loaded pre-trained model for group '{group_name}': "
            f"{model_path.name} (features={model.num_feature()})")
        return model, scaler_mean, scaler_scale, fill_vals

    @staticmethod
    def _predict_with_group_model(model, X: np.ndarray, n_classes: int) -> np.ndarray:
        """Run forward inference with a loaded group LGBM model.
        Returns probability matrix of shape [N, n_classes].
        """
        raw = model.predict(X)  # 1D: [N * n_classes] for multiclass
        return raw.reshape(-1, n_classes).astype(np.float64)

    def run_multiscale_temporal(
        self,
        factors_path: str,
        train_kp_raw: np.ndarray,
        train_lb_raw: np.ndarray,
        val_kp_raw: np.ndarray,
        val_lb_raw: np.ndarray,
        flat_attributes: list,
        label_map: dict,
        output_dir: str,
        cfg: dict,
        purity_mode: str = "nan_boundary",
        val_purity_mode: str = None,
        num_workers: int = 1,
        num_chunks: int = 256,
        cache_dir: str = "dataset_cache",
        stage_cache_dir: str = "",
        use_stage_cache: bool = False,
        refresh_stage_cache: bool = False,
        viz_dir: str = "",
        skip_temporal: bool = False,
        inference_only: bool = False,
        progress_callback=None,
        train_video_lengths: tuple = None,
        val_video_lengths: tuple = None,
    ) -> dict:
        """
        Three-stage pipeline:
          1. Group by seq_length (short<=3 / medium 4-15 / long>15) -> each group independent LGBM -> proba [T,C]
          2. Concatenate group probas -> meta features [T, C x n_groups] -> meta-LGBM
          3. meta-LGBM proba -> temporal sliding window features -> temporal LGBM (TemporalValidator)

        skip_temporal: if True, skip stage 3 temporal training/inference,
                       returns dict containing _proba_meta_train and _proba_meta_val.

        inference_only: if True, load pre-trained models from output_dir/weights/
                        and run forward inference only. No LGBM training is performed.
                        Requires model files (group_*_lgbm.txt, meta_lgbm.txt) and
                        scaler params (scaler_mean_*.npy, scaler_scale_*.npy, fill_*.npy)
                        to exist in weights/. Raises FileNotFoundError if missing.

        progress_callback: optional callable(frac: float, status: str) called at
                           sub-stage milestones. frac is 0.0–1.0 within the pipeline.

        Note: in strict purity mode, effective frame counts may differ across groups, auto-fallback to nan_boundary.
        """
        from src.temporal_validator import TemporalValidator

        if purity_mode == "strict":
            logger.warning(
                "[MultiScale] In strict mode, frame counts may differ across groups, auto-switching to nan_boundary"
            )
            purity_mode = "nan_boundary"

        # val_purity_mode=None means same as training set
        _val_purity = val_purity_mode if val_purity_mode is not None else purity_mode
        if _val_purity == "strict":
            logger.warning("[MultiScale] val strict mode auto-switching to nan_boundary")
            _val_purity = "nan_boundary"
        if _val_purity != purity_mode:
            logger.info(f"[MultiScale] Train purity={purity_mode}, val purity={_val_purity}")

        factors = self.load_valid_factors(factors_path)
        # Build factor name → full definition map for reproducibility
        _factor_def_map = {}
        for _f in factors:
            _fn = _f.get("name", "")
            if _fn:
                _factor_def_map[_fn] = {k: v for k, v in _f.items()
                                        if k in ("name", "code", "seq_length",
                                                 "description", "target", "mode")}
        classes_sorted = sorted(set(int(v) for v in label_map.values()))
        id_to_name = {int(v): k for k, v in label_map.items()}
        class_names = [id_to_name[c] for c in classes_sorted]

        # Stage cache root directory
        # Split design: X_std (factor matrix) stored in group_{name}/, proba (LGBM output) in group_{name}_proba/
        # When refresh_stage_cache=true, only recompute proba layer, factor matrix still loaded from cache
        sc_root = Path(stage_cache_dir) if stage_cache_dir else Path("pipeline_stage_cache")
        _PROBA_SUFFIX = "_proba"  # proba subdirectory suffix (separated from X_std)
        _r1_suffix = "_r1" if self.residual_stage1 else ""
        _r2_suffix = "_r2" if self.residual_stage2 else ""
        _use_factor_cache = use_stage_cache               # X_std layer: only attempt load when stage_cache is enabled
        _use_proba_cache = use_stage_cache and not refresh_stage_cache  # proba layer: do not load old proba during refresh
        _save_factor_cache = use_stage_cache              # X_std layer: save when enabled
        _save_proba_cache = use_stage_cache or refresh_stage_cache     # proba layer: also save new proba during refresh

        # ---- Step 1: Group ----
        factor_groups = _group_factors_by_seqlength(factors)
        logger.info(
            f"[MultiScale] Factor grouping: { {g: len(v) for g, v in factor_groups.items()} }"
        )

        # ---- Step 2: Each group factor matrix -> within-group LGBM -> proba ----
        group_proba_train_list = []
        group_proba_val_list = []
        group_metrics = {}
        final_tr_lb = None
        final_vl_lb = None
        # Accumulate standardized factor matrices for OvR reuse (avoid recomputation)
        all_X_tr_std_parts: list = []
        all_X_vl_std_parts: list = []
        all_used_names: list = []

        # ── Build combined video-length lists for per-video window building ──
        _tv_m1, _tv_m2 = train_video_lengths if train_video_lengths else ([], [])
        _vv_m1, _vv_m2 = val_video_lengths   if val_video_lengths   else ([], [])
        # Feature order is mouse1-then-mouse2 → concatenate both halves
        _train_vlens = _tv_m1 + _tv_m2 if _tv_m1 and _tv_m2 else None
        _val_vlens   = _vv_m1 + _vv_m2 if _vv_m1 and _vv_m2 else None

        # ── Progress tracking: allocate fraction by factor count ──
        _total_factors = sum(len(v) for v in factor_groups.values())
        _groups_frac = 0.75 if skip_temporal else 0.60
        _meta_frac = 1.0 if skip_temporal else 0.80
        # temporal phase occupies 0.80→1.0 when not skipped
        _progress_base = 0.0

        for group_name, group_factors in factor_groups.items():
            # ── Sub-progress: start of group ──
            _group_weight = len(group_factors) / max(_total_factors, 1)
            _group_span = _group_weight * _groups_frac
            _group_start = _progress_base
            _group_mid = _group_start + _group_span * 0.70   # factor computation
            _group_lgbm = _group_start + _group_span * 0.95  # LGBM training done
            _group_end = _group_start + _group_span
            if progress_callback:
                progress_callback(_group_start,
                                  f"Group '{group_name}': computing {len(group_factors)} factors...")
            # -- Select top-k factors by mean_auc --
            _tk = self._get_top_k(group_name)
            if _tk > 0 and len(group_factors) > _tk:
                for f in group_factors:
                    pc = f.get("per_class", {})
                    aucs = [v["auc"] if isinstance(v, dict) else v for v in pc.values()] if pc else [0.0]
                    f["_mean_auc"] = float(np.mean(aucs))
                n_before = len(group_factors)
                group_factors = sorted(group_factors, key=lambda x: x["_mean_auc"], reverse=True)[:_tk]
                logger.info(
                    f"[MultiScale][top-k] Group '{group_name}': {n_before} -> {len(group_factors)} factors"
                    f" (mean_auc cutoff={group_factors[-1]['_mean_auc']:.4f})"
                )

            logger.info(
                f"[MultiScale] Processing group '{group_name}' ({len(group_factors)} factors)"
            )

            # -- Split cache: X_std layer (group_{name}/) vs proba layer (group_{name}_proba/) --
            group_sc_dir = sc_root / f"group_{group_name}"
            group_proba_dir = sc_root / f"group_{group_name}{_PROBA_SUFFIX}"
            factor_cache_hit = False
            proba_cache_hit = False
            _cache_has_std = False  # Whether cache contains standardized matrix
            _cache_scaler_mean = None  # Standardization parameters in cache
            _cache_scaler_scale = None
            _cache_fill = None

            # ---- Layer 1: Try loading factor matrix cache (X_std + common + labels + scaler) ----
            if _use_factor_cache:
                _factor_keys = ["tr_lb", "vl_lb", "X_std_train", "X_std_val", "common"]
                sc_factor = self._load_stage(group_sc_dir, _factor_keys)
                if sc_factor is not None:
                    # Optionally load scaler params (old caches may not have them)
                    for _opt_key in ["scaler_mean", "scaler_scale", "fill"]:
                        _opt_path = group_sc_dir / f"{_opt_key}.npy"
                        if _opt_path.exists():
                            sc_factor[_opt_key] = np.load(_opt_path)
                    tr_lb = sc_factor["tr_lb"]
                    vl_lb = sc_factor["vl_lb"]
                    X_tr_std_cached = sc_factor["X_std_train"]
                    X_vl_std_cached = sc_factor["X_std_val"]
                    common = sc_factor["common"]
                    _cache_has_std = True
                    _cache_scaler_mean = sc_factor.get("scaler_mean")
                    _cache_scaler_scale = sc_factor.get("scaler_scale")
                    _cache_fill = sc_factor.get("fill")
                    factor_cache_hit = True
                    logger.info(
                        f"[StageCache] X_std cache hit: {group_sc_dir}"
                        f" ({len(common)} factors, X_std shape={X_tr_std_cached.shape})"
                    )

            # ---- Layer 2: Try loading proba cache (only in non-refresh mode) ----
            if _use_proba_cache and factor_cache_hit:
                sc_proba = self._load_stage(group_proba_dir, ["proba_train", "proba_val"])
                if sc_proba is not None:
                    p_tr = sc_proba["proba_train"]       # pure proba
                    p_vl = sc_proba["proba_val"]         # pure proba
                    proba_cache_hit = True
                    logger.info(
                        f"[StageCache] proba cache hit: {group_proba_dir}"
                    )
                else:
                    # _use_proba_cache requires proba to exist; treat missing as refresh scenario
                    logger.info(
                        f"[StageCache] X_std cache hit but proba cache missing ({group_proba_dir}),"
                        f"will retrain LGBM"
                    )

            full_cache_hit = factor_cache_hit and proba_cache_hit
            factor_only_hit = factor_cache_hit and not proba_cache_hit

            if full_cache_hit:
                # === FULL CACHE HIT: X_std + proba both loaded from cache ===
                if progress_callback:
                    progress_callback(_group_mid, f"Group '{group_name}': loaded from cache")
                # residual_stage1 concat (needs rebuild when loading from cache)
                if self.residual_stage1:
                    all_X_tr_std_parts.append(X_tr_std_cached)
                    all_X_vl_std_parts.append(X_vl_std_cached)
                    all_used_names.extend(common)
                    p_tr = np.hstack([X_tr_std_cached, p_tr])
                    p_vl = np.hstack([X_vl_std_cached, p_vl])
                    logger.info(
                        f"[MultiScale][Residual-Stage1] Group '{group_name}' cache hit rebuild,"
                        f"after concat shape={p_tr.shape}"
                    )
                # Generate visualizations on cache hit too (data fully from cache, no recomputation needed)
                n_classes = len(classes_sorted)
                p_vl_proba = p_vl[:, -n_classes:] if self.residual_stage1 else p_vl
                y_pred_g = np.array([classes_sorted[i] for i in p_vl_proba.argmax(axis=1)])
                group_cm = confusion_matrix(vl_lb, y_pred_g, labels=classes_sorted)
                _prec, _rec, _f1, _sup = precision_recall_fscore_support(vl_lb, y_pred_g, labels=classes_sorted, zero_division=0)
                group_per_class = {}
                for _ci, _c in enumerate(classes_sorted):
                    _cname = id_to_name[_c]
                    _y_bin = (vl_lb == _c).astype(int)
                    try:
                        _auc_val = float(roc_auc_score(_y_bin, p_vl_proba[:, _ci]))
                    except ValueError:
                        _auc_val = None
                    group_per_class[_cname] = {
                        "precision": round(float(_prec[_ci]), 4),
                        "recall": round(float(_rec[_ci]), 4),
                        "f1": round(float(_f1[_ci]), 4),
                        "auc": round(_auc_val, 4) if _auc_val is not None else None,
                        "support": int(_sup[_ci]),
                    }
                if viz_dir:
                    try:
                        plot_proba_distribution(
                            proba=p_vl_proba, labels=vl_lb,
                            class_names=class_names,
                            viz_dir=viz_dir, group_name=group_name,
                        )
                        plot_group_confusion_matrix(
                            cm=group_cm, class_names=class_names,
                            viz_dir=viz_dir, group_name=group_name,
                        )
                    except Exception as _e:
                        logger.warning(f"[Viz] Cache hit proba/CM visualization failed: {_e}")
                    if self.residual_stage1:
                        try:
                            plot_feature_importance(
                                X_train=X_tr_std_cached, y_train=tr_lb,
                                X_val=X_vl_std_cached, y_val=vl_lb,
                                used_names=common,
                                lgbm_params=self.lgbm_params,
                                class_weight=self.class_weight,
                                early_stopping_rounds=self.early_stopping_rounds,
                                viz_dir=viz_dir, group_name=group_name,
                            )
                        except Exception as _e:
                            logger.warning(f"[Viz] Cache hit feature importance visualization failed: {_e}")
                # -- Collect intermediate data for cross-stage visualization --
                if not hasattr(self, '_stage_data'):
                    self._stage_data = {}
                sd = self._stage_data
                sd.setdefault("group_per_class", {})[group_name] = group_per_class
                sd.setdefault("group_cms", {})[group_name] = group_cm
                sd.setdefault("group_probas", {})[group_name] = p_vl_proba
                sd.setdefault("group_preds", {})[group_name] = y_pred_g

                # -- On cache hit: if weight file missing, rebuild from cache data and save --
                _weights_dir = Path(output_dir) / "weights"
                _model_missing = not (_weights_dir / f"group_{group_name}_lgbm.txt").exists()
                _scaler_missing = not (_weights_dir / f"scaler_mean_{group_name}.npy").exists()
                _cf_missing = not (_weights_dir / f"common_factors_{group_name}.json").exists()

                if _model_missing:
                    # Model missing -> must retrain LGBM (needs cached X_std)
                    if _cache_has_std:
                        _use_cached_scaler = _cache_scaler_mean is not None
                        logger.info(
                            f"[MultiScale] Group '{group_name}' cache hit but model missing,"
                            f"rebuilding model from cached data..."
                        )
                        try:
                            _weights_dir.mkdir(parents=True, exist_ok=True)
                            # Use deterministic parameters for rebuild to ensure reproducibility on GPU
                            _rebuild_params = dict(self.lgbm_params)
                            _rebuild_params["n_jobs"] = 1
                            _rebuild_params["deterministic"] = True
                            _p_tr_tmp, _p_vl_tmp, _evals_tmp, _gm = _train_single_lgbm(
                                X_tr_std_cached, tr_lb,
                                X_vl_std_cached, vl_lb,
                                classes_sorted, _rebuild_params,
                                self.class_weight, self.early_stopping_rounds,
                                return_evals_result=True,
                                return_model=True,
                            )
                            _gm.booster_.save_model(
                                str(_weights_dir / f"group_{group_name}_lgbm.txt")
                            )
                            logger.info(
                                f"[MultiScale] Group LGBM model saved (rebuild):"
                                f"group_{group_name}_lgbm.txt"
                            )
                            if _use_cached_scaler:
                                np.save(_weights_dir / f"scaler_mean_{group_name}.npy",
                                        _cache_scaler_mean.astype(np.float32))
                                np.save(_weights_dir / f"scaler_scale_{group_name}.npy",
                                        _cache_scaler_scale.astype(np.float32))
                                np.save(_weights_dir / f"fill_{group_name}.npy",
                                        _cache_fill.astype(np.float32))
                            else:
                                _tr_mean = X_tr_std_cached.mean(axis=0).astype(np.float32)
                                _tr_std = X_tr_std_cached.std(axis=0).astype(np.float32)
                                _tr_std[_tr_std == 0] = 1.0
                                _fill_rb = np.nanmean(X_tr_std_cached, axis=0).astype(np.float32)
                                np.save(_weights_dir / f"scaler_mean_{group_name}.npy", _tr_mean)
                                np.save(_weights_dir / f"scaler_scale_{group_name}.npy", _tr_std)
                                np.save(_weights_dir / f"fill_{group_name}.npy", _fill_rb)
                            _cf_tmp = _weights_dir / f"common_factors_{group_name}.json.tmp"
                            _fc = {str(n): _factor_def_map[n] for n in common if n in _factor_def_map}
                            _cf_data = {
                                "common": [str(n) for n in common],
                                "factor_code": _fc,
                                "classes_sorted": [int(c) for c in classes_sorted],
                                "id_to_name": {str(k): str(v) for k, v in id_to_name.items()},
                                "n_classes": int(n_classes),
                            }
                            with open(_cf_tmp, "w", encoding="utf-8") as _cf:
                                json.dump(_cf_data, _cf, ensure_ascii=False)
                            _cf_tmp.replace(_weights_dir / f"common_factors_{group_name}.json")
                            logger.info(
                                f"[MultiScale] Group standardization parameters saved (rebuild):"
                                f"scaler_mean/scale/fill + common_factors ({group_name})"
                            )
                            del _gm, _p_tr_tmp, _p_vl_tmp, _evals_tmp
                        except Exception as _e:
                            logger.warning(
                                f"[MultiScale] Group '{group_name}' cache hit model rebuild failed: {_e}"
                            )
                    else:
                        logger.warning(
                            f"[MultiScale] Group '{group_name}' cache hit but model missing,"
                            f"and cache has no X_std (residual_stage1=false), cannot auto-rebuild."
                            f"Please re-run train_behavior.py and ensure "
                            f"synth_validation.residual_stage1=true。"
                        )

                elif _scaler_missing or _cf_missing:
                    # Model exists, only scaler or common_factors missing -> backfill from cache directly, no retrain!
                    _missing_parts = []
                    if _scaler_missing:
                        _missing_parts.append("scaler")
                    if _cf_missing:
                        _missing_parts.append("common_factors")
                    logger.info(
                        f"[MultiScale] Group '{group_name}' cache hit,"
                        f"only {'/'.join(_missing_parts)} missing, backfilling directly from cache (no model retrain)..."
                    )
                    try:
                        _weights_dir.mkdir(parents=True, exist_ok=True)
                        if _scaler_missing:
                            if _cache_scaler_mean is not None:
                                np.save(_weights_dir / f"scaler_mean_{group_name}.npy",
                                        _cache_scaler_mean.astype(np.float32))
                                np.save(_weights_dir / f"scaler_scale_{group_name}.npy",
                                        _cache_scaler_scale.astype(np.float32))
                                np.save(_weights_dir / f"fill_{group_name}.npy",
                                        _cache_fill.astype(np.float32))
                            elif _cache_has_std:
                                _tr_mean = X_tr_std_cached.mean(axis=0).astype(np.float32)
                                _tr_std = X_tr_std_cached.std(axis=0).astype(np.float32)
                                _tr_std[_tr_std == 0] = 1.0
                                _fill_rb = np.nanmean(X_tr_std_cached, axis=0).astype(np.float32)
                                np.save(_weights_dir / f"scaler_mean_{group_name}.npy", _tr_mean)
                                np.save(_weights_dir / f"scaler_scale_{group_name}.npy", _tr_std)
                                np.save(_weights_dir / f"fill_{group_name}.npy", _fill_rb)
                        if _cf_missing:
                            _cf_tmp = _weights_dir / f"common_factors_{group_name}.json.tmp"
                            _fc = {str(n): _factor_def_map[n] for n in common if n in _factor_def_map}
                            _cf_data = {
                                "common": [str(n) for n in common],
                                "factor_code": _fc,
                                "classes_sorted": [int(c) for c in classes_sorted],
                                "id_to_name": {str(k): str(v) for k, v in id_to_name.items()},
                                "n_classes": int(n_classes),
                            }
                            with open(_cf_tmp, "w", encoding="utf-8") as _cf:
                                json.dump(_cf_data, _cf, ensure_ascii=False)
                            _cf_tmp.replace(_weights_dir / f"common_factors_{group_name}.json")
                        logger.info(
                            f"[MultiScale] Group '{group_name}' {'/'.join(_missing_parts)} backfilled "
                            f" (total {len(common)} factors)"
                        )
                    except Exception as _e:
                        logger.warning(
                            f"[MultiScale] Group '{group_name}' {'/'.join(_missing_parts)} backfill failed: {_e}"
                        )

            elif factor_only_hit:
                # === FACTOR CACHE HIT + PROBA REFRESH (or INFERENCE_ONLY) ===
                _weights_dir = Path(output_dir) / "weights"
                n_classes = len(classes_sorted)

                if inference_only:
                    # Load pre-trained model and predict directly (no retraining)
                    _gm, _sm, _ss, _fl = self._load_pretrained_group_model(
                        _weights_dir, group_name, n_classes)
                    # X_tr_std_cached / X_vl_std_cached already loaded from cache
                    p_tr = self._predict_with_group_model(_gm, X_tr_std_cached, n_classes)
                    p_vl = self._predict_with_group_model(_gm, X_vl_std_cached, n_classes)
                    del _gm, _sm, _ss, _fl
                    logger.info(
                        f"[Inference] Group '{group_name}': predicted from pre-trained model"
                        f" (X_std cache hit, {len(common)} factors)")
                    if progress_callback:
                        progress_callback(_group_lgbm,
                                          f"Group '{group_name}': inference from pre-trained model")
                else:
                    # === original: retrain LGBM from cached X_std ===
                    if progress_callback:
                        progress_callback(_group_mid, f"Group '{group_name}': cache loaded, retraining LGBM...")
                    import gc
                    logger.info(
                        f"[MultiScale] Group '{group_name}' X_std cache hit ({len(common)} factors),"
                        f"retraining with new LGBM parameters..."
                    )

                    # Train LGBM using cached X_std
                    p_tr, p_vl, evals_result, group_model = _train_single_lgbm(
                        X_tr_std_cached, tr_lb, X_vl_std_cached, vl_lb,
                        classes_sorted, self.lgbm_params,
                        self.class_weight, self.early_stopping_rounds,
                        return_evals_result=True,
                        return_model=True,
                    )

                    # -- Save group LGBM model + standardization params --
                    _weights_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        group_model.booster_.save_model(
                            str(_weights_dir / f"group_{group_name}_lgbm.txt")
                        )
                        logger.info(
                            f"[MultiScale] Group LGBM model saved (refresh):"
                            f"group_{group_name}_lgbm.txt"
                        )
                    except Exception as _e:
                        logger.warning(f"[MultiScale] Group model save failed: {_e}")
                    try:
                        if _cache_scaler_mean is not None:
                            np.save(_weights_dir / f"scaler_mean_{group_name}.npy",
                                    _cache_scaler_mean.astype(np.float32))
                            np.save(_weights_dir / f"scaler_scale_{group_name}.npy",
                                    _cache_scaler_scale.astype(np.float32))
                            np.save(_weights_dir / f"fill_{group_name}.npy",
                                    _cache_fill.astype(np.float32))
                        else:
                            _tr_mean = X_tr_std_cached.mean(axis=0).astype(np.float32)
                            _tr_std = X_tr_std_cached.std(axis=0).astype(np.float32)
                            _tr_std[_tr_std == 0] = 1.0
                            _fill_rb = np.nanmean(X_tr_std_cached, axis=0).astype(np.float32)
                            np.save(_weights_dir / f"scaler_mean_{group_name}.npy", _tr_mean)
                            np.save(_weights_dir / f"scaler_scale_{group_name}.npy", _tr_std)
                            np.save(_weights_dir / f"fill_{group_name}.npy", _fill_rb)
                        _cf_tmp = _weights_dir / f"common_factors_{group_name}.json.tmp"
                        _cf_data = {
                            "common": [str(n) for n in common],
                            "classes_sorted": [int(c) for c in classes_sorted],
                            "id_to_name": {str(k): str(v) for k, v in id_to_name.items()},
                            "n_classes": int(n_classes),
                        }
                        with open(_cf_tmp, "w", encoding="utf-8") as _cf:
                            json.dump(_cf_data, _cf, ensure_ascii=False)
                        _cf_tmp.replace(_weights_dir / f"common_factors_{group_name}.json")
                        logger.info(
                            f"[MultiScale] Group standardization parameters saved (refresh):"
                            f"scaler_mean/scale/fill + common_factors ({group_name})"
                        )
                    except Exception as _e:
                        logger.warning(f"[MultiScale] Group standardization parameters save failed: {_e}")
                    del group_model; gc.collect()

                    # -- Only write proba cache (X_std layer unchanged) --
                    if _save_proba_cache:
                        self._save_stage(
                            group_proba_dir,
                            proba_train=p_tr,
                            proba_val=p_vl,
                        )

                    if progress_callback:
                        progress_callback(_group_lgbm, f"Group '{group_name}': LGBM trained (refresh)")

                # ---- SHARED (both inference_only and retrain paths) ----
                import gc as _gc_shared

                # Accumulate X_std for OvR
                all_X_tr_std_parts.append(X_tr_std_cached)
                all_X_vl_std_parts.append(X_vl_std_cached)
                all_used_names.extend(common)

                # Residual connection (Stage 1)
                if self.residual_stage1:
                    p_tr = np.hstack([X_tr_std_cached, p_tr])
                    p_vl = np.hstack([X_vl_std_cached, p_vl])
                    logger.info(
                        f"[MultiScale][Residual-Stage1] Group '{group_name}': "
                        f"after concat shape={p_tr.shape}"
                    )

                # -- Compute group-level per-class metrics + confusion matrix --
                n_classes = len(classes_sorted)
                p_vl_proba = p_vl[:, -n_classes:] if self.residual_stage1 else p_vl
                y_pred_g = np.array([classes_sorted[i] for i in p_vl_proba.argmax(axis=1)])
                group_cm = confusion_matrix(vl_lb, y_pred_g, labels=classes_sorted)
                group_per_class = {}
                _prec, _rec, _f1, _sup = precision_recall_fscore_support(vl_lb, y_pred_g, labels=classes_sorted, zero_division=0)
                for _ci, _c in enumerate(classes_sorted):
                    _cname = id_to_name[_c]
                    _y_bin = (vl_lb == _c).astype(int)
                    try:
                        _auc_val = float(roc_auc_score(_y_bin, p_vl_proba[:, _ci]))
                    except ValueError:
                        _auc_val = None
                    group_per_class[_cname] = {
                        "precision": round(float(_prec[_ci]), 4),
                        "recall": round(float(_rec[_ci]), 4),
                        "f1": round(float(_f1[_ci]), 4),
                        "auc": round(_auc_val, 4) if _auc_val is not None else None,
                        "support": int(_sup[_ci]),
                    }

                # -- Visualization (skip learning curve for inference_only) --
                if viz_dir:
                    try:
                        plot_feature_importance(
                            X_train=X_tr_std_cached, y_train=tr_lb,
                            X_val=X_vl_std_cached, y_val=vl_lb,
                            used_names=common,
                            lgbm_params=self.lgbm_params,
                            class_weight=self.class_weight,
                            early_stopping_rounds=self.early_stopping_rounds,
                            viz_dir=viz_dir, group_name=group_name,
                        )
                        plot_proba_distribution(
                            proba=p_vl_proba, labels=vl_lb,
                            class_names=class_names,
                            viz_dir=viz_dir, group_name=group_name,
                        )
                        plot_group_confusion_matrix(
                            cm=group_cm, class_names=class_names,
                            viz_dir=viz_dir, group_name=group_name,
                        )
                        if not inference_only and 'evals_result' in dir():
                            plot_learning_curve(
                                evals_result=evals_result,
                                viz_dir=viz_dir, group_name=group_name,
                            )
                    except Exception as _e:
                        logger.warning(f"[Viz] Group LGBM visualization failed: {_e}")

                # -- Collect intermediate data for cross-stage visualization --
                if not hasattr(self, '_stage_data'):
                    self._stage_data = {}
                sd = self._stage_data
                sd.setdefault("group_per_class", {})[group_name] = group_per_class
                sd.setdefault("group_cms", {})[group_name] = group_cm
                sd.setdefault("group_probas", {})[group_name] = p_vl_proba
                sd.setdefault("group_preds", {})[group_name] = y_pred_g

            else:
                # === FULL MISS (or INFERENCE_ONLY): Compute factor matrix, then train or load model ===
                import gc
                if num_workers > 1:
                    X_tr, tr_lb, used_tr, _ = self.compute_factor_matrix_multiresolution_fast(
                        group_factors, train_kp_raw, train_lb_raw, flat_attributes,
                        purity_mode=purity_mode, num_chunks=num_chunks,
                        num_workers=num_workers, split_name=f"train_{group_name}",
                        cache_dir=cache_dir, video_lengths=_train_vlens,
                    )
                    X_vl, vl_lb, used_vl, _ = self.compute_factor_matrix_multiresolution_fast(
                        group_factors, val_kp_raw, val_lb_raw, flat_attributes,
                        purity_mode=_val_purity, num_chunks=num_chunks,
                        num_workers=num_workers, split_name=f"val_{group_name}",
                        cache_dir=cache_dir, video_lengths=_val_vlens,
                    )
                else:
                    X_tr, tr_lb, used_tr, _ = self.compute_factor_matrix_multiresolution(
                        group_factors, train_kp_raw, train_lb_raw, flat_attributes,
                        purity_mode=purity_mode, split_name=f"train_{group_name}",
                        video_lengths=_train_vlens,
                    )
                    X_vl, vl_lb, used_vl, _ = self.compute_factor_matrix_multiresolution(
                        group_factors, val_kp_raw, val_lb_raw, flat_attributes,
                        purity_mode=_val_purity, split_name=f"val_{group_name}",
                        video_lengths=_val_vlens,
                    )

                # Align common factor columns between train/val
                common = [n for n in used_tr if n in set(used_vl)]
                if not common:
                    logger.warning(f"[MultiScale] Group '{group_name}' has no valid common factors, skipping")
                    del X_tr, X_vl; gc.collect()
                    continue
                if len(common) < len(used_tr) or len(common) < len(used_vl):
                    tr_idx = [used_tr.index(n) for n in common]
                    vl_idx = [used_vl.index(n) for n in common]
                    X_tr = X_tr[:, tr_idx].copy()
                    X_vl = X_vl[:, vl_idx].copy()

                n_classes = len(classes_sorted)
                _weights_dir = Path(output_dir) / "weights"

                if inference_only:
                    # ── Load pre-trained model + scaler from weights/, predict only ──
                    _gm, _scaler_mean, _scaler_scale, _fill = \
                        self._load_pretrained_group_model(
                            _weights_dir, group_name, n_classes)

                    # ── Align factor matrix to training common_factors order ──
                    # Reorder + pad/trim to exactly match the K columns the scaler
                    # and LGBM expect (common_factors_{group}.json).
                    _cf_path = _weights_dir / f"common_factors_{group_name}.json"
                    if _cf_path.exists():
                        import json as _json_cf
                        with open(_cf_path, "r", encoding="utf-8") as _f_cf:
                            _cf_data = _json_cf.load(_f_cf)
                        _training_common = _cf_data.get("common", [])
                        _K = len(_training_common)  # expected column count
                        _name_to_col = {n: i for i, n in enumerate(used_tr)}
                        # Reorder existing columns; missing ones filled with NaN
                        _aligned_tr = np.full((X_tr.shape[0], _K), np.nan, dtype=np.float32)
                        _aligned_vl = np.full((X_vl.shape[0], _K), np.nan, dtype=np.float32)
                        _matched = 0
                        for _ci, _name in enumerate(_training_common):
                            _src_col = _name_to_col.get(_name)
                            if _src_col is not None:
                                _aligned_tr[:, _ci] = X_tr[:, _src_col]
                                _aligned_vl[:, _ci] = X_vl[:, _src_col]
                                _matched += 1
                        X_tr, X_vl = _aligned_tr, _aligned_vl
                        used_tr = used_vl = common = list(_training_common)
                        logger.info(
                            f"[Inference] Group '{group_name}': aligned factor matrix "
                            f"-> ({X_tr.shape[1]} cols), matched {_matched}/{_K} "
                            f"common factors, {_K - _matched} filled with NaN"
                        )
                        del _aligned_tr, _aligned_vl

                    # Standardize using saved scaler params
                    _sm = _scaler_mean
                    _ss = _scaler_scale
                    _fl = _fill

                    # Fill NaN with saved fill values
                    _tr_nan = np.isnan(X_tr)
                    _vl_nan = np.isnan(X_vl)
                    if _tr_nan.any():
                        X_tr = np.where(_tr_nan, _fl, X_tr)
                    if _vl_nan.any():
                        X_vl = np.where(_vl_nan, _fl, X_vl)

                    # Standardize
                    _ss[_ss == 0] = 1.0
                    X_tr_std = (X_tr - _sm) / _ss
                    X_vl_std = (X_vl - _sm) / _ss
                    np.clip(X_tr_std, -8, 8, out=X_tr_std)
                    np.clip(X_vl_std, -8, 8, out=X_vl_std)
                    del X_tr, X_vl; gc.collect()
                    del _sm, _ss, _fl, _tr_nan, _vl_nan

                    # Predict
                    p_tr = self._predict_with_group_model(_gm, X_tr_std, n_classes)
                    p_vl = self._predict_with_group_model(_gm, X_vl_std, n_classes)
                    del _gm
                    _save_scaler_mean = _scaler_mean.astype(np.float32)
                    _save_scaler_scale = _scaler_scale.astype(np.float32)
                    _save_fill = _fill.astype(np.float32)
                    logger.info(
                        f"[Inference] Group '{group_name}': predicted from pre-trained model"
                        f" ({len(common)} factors)")
                    if progress_callback:
                        progress_callback(_group_lgbm,
                                          f"Group '{group_name}': inference from pre-trained model")

                else:
                    # ── Training path: standardize + train LGBM ──
                    if progress_callback:
                        progress_callback(_group_mid, f"Group '{group_name}': factors computed, training LGBM...")

                    # -- Visualization: factor valid ratio + factor value distribution --
                    if viz_dir:
                        try:
                            plot_factor_valid_ratio(
                                used_names=common,
                                dropped_names=[f["name"] for f in group_factors
                                               if f["name"] not in common],
                                X=X_tr,
                                viz_dir=viz_dir,
                                group_name=group_name,
                            )
                            plot_factor_distribution(
                                X=X_tr, used_names=common,
                                viz_dir=viz_dir, group_name=group_name,
                            )
                        except Exception as _e:
                            logger.warning(f"[Viz] Factor visualization failed: {_e}")

                    X_tr_std, X_vl_std, _scaler, _fill = self.standardize(
                        X_tr, X_vl,
                        viz_dir=viz_dir,
                        split_name=f"train_{group_name}",
                    )
                    del X_tr, X_vl; gc.collect()

                    p_tr, p_vl, evals_result, group_model = _train_single_lgbm(
                        X_tr_std, tr_lb, X_vl_std, vl_lb,
                        classes_sorted, self.lgbm_params,
                        self.class_weight, self.early_stopping_rounds,
                        return_evals_result=True,
                        return_model=True,
                    )

                    if progress_callback:
                        progress_callback(_group_lgbm, f"Group '{group_name}': LGBM trained")

                    # -- Save group LGBM model + standardization params --
                    _weights_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        group_model.booster_.save_model(
                            str(_weights_dir / f"group_{group_name}_lgbm.txt")
                        )
                        logger.info(
                            f"[MultiScale] Group LGBM model saved:"
                            f"group_{group_name}_lgbm.txt"
                        )
                    except Exception as _e:
                        logger.warning(f"[MultiScale] Group model save failed: {_e}")
                    try:
                        np.save(_weights_dir / f"scaler_mean_{group_name}.npy",
                                _scaler.mean_.astype(np.float32))
                        np.save(_weights_dir / f"scaler_scale_{group_name}.npy",
                                _scaler.scale_.astype(np.float32))
                        np.save(_weights_dir / f"fill_{group_name}.npy",
                                _fill.astype(np.float32))
                        _cf_tmp = _weights_dir / f"common_factors_{group_name}.json.tmp"
                        _cf_data = {
                            "common": [str(n) for n in common],
                            "classes_sorted": [int(c) for c in classes_sorted],
                            "id_to_name": {str(k): str(v) for k, v in id_to_name.items()},
                            "n_classes": int(n_classes),
                        }
                        with open(_cf_tmp, "w", encoding="utf-8") as _cf:
                            json.dump(_cf_data, _cf, ensure_ascii=False)
                        _cf_tmp.replace(_weights_dir / f"common_factors_{group_name}.json")
                        logger.info(
                            f"[MultiScale] Group standardization parameters saved:"
                            f"scaler_mean/scale/fill + common_factors ({group_name})"
                        )
                    except Exception as _e:
                        logger.warning(f"[MultiScale] Group standardization parameters save failed: {_e}")
                    _save_scaler_mean = _scaler.mean_.astype(np.float32)
                    _save_scaler_scale = _scaler.scale_.astype(np.float32)
                    _save_fill = _fill.astype(np.float32)
                    del group_model, _scaler, _fill; gc.collect()

                # ---- SHARED (both inference_only and training paths for FULL MISS) ----
                # Accumulate standardized matrix for OvR reuse
                all_X_tr_std_parts.append(X_tr_std)
                all_X_vl_std_parts.append(X_vl_std)
                all_used_names.extend(common)

                # -- Compute group-level per-class metrics + confusion matrix --
                n_classes = len(classes_sorted)
                p_vl_proba = p_vl[:, -n_classes:] if self.residual_stage1 else p_vl
                y_pred_g = np.array([classes_sorted[i] for i in p_vl_proba.argmax(axis=1)])
                group_cm = confusion_matrix(vl_lb, y_pred_g, labels=classes_sorted)
                group_per_class = {}
                _prec, _rec, _f1, _sup = precision_recall_fscore_support(vl_lb, y_pred_g, labels=classes_sorted, zero_division=0)
                for _ci, _c in enumerate(classes_sorted):
                    _cname = id_to_name[_c]
                    _y_bin = (vl_lb == _c).astype(int)
                    try:
                        _auc_val = float(roc_auc_score(_y_bin, p_vl_proba[:, _ci]))
                    except ValueError:
                        _auc_val = None
                    group_per_class[_cname] = {
                        "precision": round(float(_prec[_ci]), 4),
                        "recall": round(float(_rec[_ci]), 4),
                        "f1": round(float(_f1[_ci]), 4),
                        "auc": round(_auc_val, 4) if _auc_val is not None else None,
                        "support": int(_sup[_ci]),
                    }

                # -- Visualization (skip learning curve for inference_only) --
                if viz_dir:
                    try:
                        plot_feature_importance(
                            X_train=X_tr_std, y_train=tr_lb,
                            X_val=X_vl_std, y_val=vl_lb,
                            used_names=common,
                            lgbm_params=self.lgbm_params,
                            class_weight=self.class_weight,
                            early_stopping_rounds=self.early_stopping_rounds,
                            viz_dir=viz_dir, group_name=group_name,
                        )
                        plot_proba_distribution(
                            proba=p_vl_proba, labels=vl_lb,
                            class_names=class_names,
                            viz_dir=viz_dir, group_name=group_name,
                        )
                        plot_group_confusion_matrix(
                            cm=group_cm, class_names=class_names,
                            viz_dir=viz_dir, group_name=group_name,
                        )
                        if not inference_only and 'evals_result' in dir():
                            plot_learning_curve(
                                evals_result=evals_result,
                                viz_dir=viz_dir, group_name=group_name,
                            )
                    except Exception as _e:
                        logger.warning(f"[Viz] Group LGBM visualization failed: {_e}")

                # -- Collect intermediate data for cross-stage visualization --
                if not hasattr(self, '_stage_data'):
                    self._stage_data = {}
                sd = self._stage_data
                sd.setdefault("group_per_class", {})[group_name] = group_per_class
                sd.setdefault("group_cms", {})[group_name] = group_cm
                sd.setdefault("group_probas", {})[group_name] = p_vl_proba
                sd.setdefault("group_preds", {})[group_name] = y_pred_g

                # -- Split save: X_std -> group_{name}/, proba -> group_{name}_proba/ --
                if _save_factor_cache:
                    self._save_stage(
                        group_sc_dir,
                        tr_lb=tr_lb,
                        vl_lb=vl_lb,
                        X_std_train=X_tr_std,      # float32, for stage1 residual reconstruction
                        X_std_val=X_vl_std,
                        common=common,
                        scaler_mean=_save_scaler_mean,
                        scaler_scale=_save_scaler_scale,
                        fill=_save_fill,
                    )
                if _save_proba_cache:
                    self._save_stage(
                        group_proba_dir,
                        proba_train=p_tr,          # pure, not concat
                        proba_val=p_vl,
                    )

                # Residual connection (Stage 1): concat standardized factor matrix to within-group LGBM output
                if self.residual_stage1:
                    p_tr = np.hstack([X_tr_std, p_tr])  # [T_tr, K_group + C]
                    p_vl = np.hstack([X_vl_std, p_vl])  # [T_vl, K_group + C]
                    logger.info(
                        f"[MultiScale][Residual-Stage1] Group '{group_name}': "
                        f"after concat shape={p_tr.shape}"
                    )

                del X_tr_std, X_vl_std; gc.collect()

            group_proba_train_list.append(p_tr)
            group_proba_val_list.append(p_vl)
            final_tr_lb = tr_lb
            final_vl_lb = vl_lb

            # Group metrics: p_vl_proba / y_pred_g / group_cm / group_per_class already computed above
            acc_g = float(accuracy_score(vl_lb, y_pred_g))
            balanced_acc_g = float(balanced_accuracy_score(vl_lb, y_pred_g))
            f1_g = float(f1_score(
                vl_lb, y_pred_g, labels=classes_sorted, average="macro", zero_division=0
            ))
            weighted_f1_g = float(f1_score(
                vl_lb, y_pred_g, labels=classes_sorted, average="weighted", zero_division=0
            ))
            # OvR macro AUC (only computable when at least 2 classes appear in val)
            present_g = sorted(set(int(v) for v in vl_lb))
            macro_auc_g = 0.0
            weighted_auc_g = 0.0
            if len(present_g) >= 2:
                idx_present_g = [classes_sorted.index(c) for c in present_g]
                proba_sub_g = p_vl_proba[:, idx_present_g]
                row_sum_g = proba_sub_g.sum(axis=1, keepdims=True)
                row_sum_g = np.where(row_sum_g <= 0, 1.0, row_sum_g)
                proba_sub_n_g = proba_sub_g / row_sum_g
                try:
                    macro_auc_g = float(roc_auc_score(
                        vl_lb, proba_sub_n_g, multi_class="ovr",
                        average="macro", labels=present_g,
                    ))
                except ValueError as e:
                    logger.warning(f"[MultiScale] Group '{group_name}' macro AUC computation failed: {e}")
                try:
                    weighted_auc_g = float(roc_auc_score(
                        vl_lb, proba_sub_n_g, multi_class="ovr",
                        average="weighted", labels=present_g,
                    ))
                except ValueError as e:
                    logger.warning(f"[MultiScale] Group '{group_name}' weighted AUC computation failed: {e}")
            group_metrics[group_name] = {
                "accuracy": round(acc_g, 4),
                "balanced_accuracy": round(balanced_acc_g, 4),
                "macro_auc": round(macro_auc_g, 4),
                "weighted_auc": round(weighted_auc_g, 4),
                "macro_f1": round(f1_g, 4),
                "weighted_f1": round(weighted_f1_g, 4),
                "n_factors": len(common),
            }
            logger.info(
                f"[MultiScale] Group '{group_name}': accuracy={acc_g:.4f}  "
                f"balanced_acc(per_class_acc_mean)={balanced_acc_g:.4f}  "
                f"macro_auc={macro_auc_g:.4f}  weighted_auc={weighted_auc_g:.4f}  "
                f"macro_f1={f1_g:.4f}  weighted_f1={weighted_f1_g:.4f}  "
                f"n_factors={len(common)}"
            )
            # ── Sub-progress: group complete ──
            if progress_callback:
                progress_callback(_group_end, f"Group '{group_name}': complete ({len(common)} factors)")
            _progress_base = _group_end

        if not group_proba_train_list:
            raise RuntimeError("[MultiScale] All factor groups have no valid factors, cannot continue.")

        tr_lb = final_tr_lb
        vl_lb = final_vl_lb

        # ---- Step 3: Concatenate -> meta-LGBM ----
        n_groups = len(group_proba_train_list)
        meta_X_train = np.hstack(group_proba_train_list)
        meta_X_val   = np.hstack(group_proba_val_list)
        # Group probas already concatenated, release immediately
        del group_proba_train_list, group_proba_val_list
        import gc; gc.collect()
        logger.info(
            f"[MultiScale] Meta feature matrix: train={meta_X_train.shape}, val={meta_X_val.shape}"
        )
        if progress_callback:
            progress_callback(_groups_frac, "Training meta-LGBM...")

        # -- Visualization: meta feature correlation + cross-group metric comparison --
        if viz_dir:
            if n_groups > 1:
                try:
                    plot_meta_correlation(
                        meta_X=meta_X_train,
                        group_names=list(factor_groups.keys()),
                        n_classes=len(classes_sorted),
                        viz_dir=viz_dir,
                    )
                except Exception as _e:
                    logger.warning(f"[Viz] Meta correlation visualization failed: {_e}")
            # Cross-group per-class metric comparison
            try:
                sd = getattr(self, '_stage_data', {})
                if sd.get("group_per_class") and len(sd["group_per_class"]) > 1:
                    plot_per_class_metrics_comparison(
                        group_per_class=sd["group_per_class"],
                        class_names=class_names,
                        viz_dir=viz_dir,
                    )
            except Exception as _e:
                logger.warning(f"[Viz] Cross-group metric comparison visualization failed: {_e}")

        # Save copy for ModelBenchmark (regardless of cache hit)
        meta_X_train_for_bm = meta_X_train.copy() if self.residual_stage2 else None
        meta_X_val_for_bm = meta_X_val.copy() if self.residual_stage2 else None

        # -- Try loading meta proba from stage cache --
        # stage1 residual changes meta input (X_std mixed into group proba), so meta subdirectory differentiates by stage1
        meta_sc_dir = sc_root / f"meta_lgbm{_r1_suffix}"
        meta_cache_hit = False
        if _use_proba_cache and n_groups > 1:
            sc_meta = self._load_stage(meta_sc_dir, ["proba_meta_train", "proba_meta_val"])
            if sc_meta is not None:
                proba_meta_train = sc_meta["proba_meta_train"]
                proba_meta_val   = sc_meta["proba_meta_val"]
                meta_cache_hit = True
                # Derive full metrics from cached proba (no retraining needed)
                logger.info("[StageCache] Meta-LGBM proba cache hit, deriving full metrics")
                y_pred_meta = np.array([classes_sorted[i] for i in proba_meta_val.argmax(axis=1)])
                meta_cm = confusion_matrix(vl_lb, y_pred_meta, labels=classes_sorted)
                meta_cm_norm = meta_cm.astype(np.float64) / np.maximum(
                    meta_cm.sum(axis=1, keepdims=True), 1
                )
                # Derive full per_class metrics from proba
                _prec_m, _rec_m, _f1_m, _sup_m = precision_recall_fscore_support(
                    vl_lb, y_pred_meta, labels=classes_sorted, zero_division=0)
                meta_per_class = {}
                present_vl = sorted(set(int(v) for v in vl_lb))
                for _ci, _c in enumerate(classes_sorted):
                    _cname = id_to_name[_c]
                    _y_bin = (vl_lb == _c).astype(int)
                    try:
                        _auc_val = float(roc_auc_score(_y_bin, proba_meta_val[:, _ci]))
                    except ValueError:
                        _auc_val = None
                    meta_per_class[_cname] = {
                        "precision": round(float(_prec_m[_ci]), 4),
                        "recall": round(float(_rec_m[_ci]), 4),
                        "f1": round(float(_f1_m[_ci]), 4),
                        "auc": round(_auc_val, 4) if _auc_val is not None else None,
                        "support": int(_sup_m[_ci]),
                    }
                # Rebuild aggregate metrics from per_class
                _macro_f1_m = float(np.mean([meta_per_class[cn]["f1"] for cn in class_names]))
                _macro_auc_m = float(np.mean([meta_per_class[cn]["auc"] for cn in class_names
                                             if meta_per_class[cn]["auc"] is not None]))
                meta_result = {
                    "metrics": {
                        "accuracy": round(float(accuracy_score(vl_lb, y_pred_meta)), 4),
                        "balanced_accuracy": round(float(balanced_accuracy_score(vl_lb, y_pred_meta)), 4),
                        "macro_f1": round(_macro_f1_m, 4),
                        "macro_auc": round(_macro_auc_m, 4),
                        "weighted_f1": round(float(f1_score(vl_lb, y_pred_meta, labels=classes_sorted, average="weighted", zero_division=0)), 4),
                        "per_class": meta_per_class,
                    },
                }
                self.save_report(meta_result, meta_cm, meta_cm_norm, class_names, output_dir)
                # -- Collect meta stage data --
                sd = getattr(self, '_stage_data', {})
                sd["meta_per_class"] = meta_per_class
                sd["meta_cm"] = meta_cm
                sd["meta_proba"] = proba_meta_val
                sd["meta_preds"] = y_pred_meta
                # -- Generate meta stage visualizations to 05_meta --
                if viz_dir:
                    try:
                        _meta_viz = str(Path(viz_dir) / "05_meta")
                        plot_group_confusion_matrix(
                            cm=meta_cm, class_names=class_names,
                            viz_dir=_meta_viz, group_name="meta",
                        )
                    except Exception as _e:
                        logger.warning(f"[Viz] Meta cache hit visualization failed: {_e}")
                # On cache hit, check and backfill missing meta-LGBM model / pipeline_meta.json
                _weights_dir = Path(output_dir) / "weights"
                _meta_txt = _weights_dir / "meta_lgbm.txt"
                _pmeta_json = _weights_dir / "pipeline_meta.json"
                _missing_model = not _meta_txt.exists()
                _missing_pmeta = not _pmeta_json.exists()
                if _missing_model and n_groups > 1:
                    logger.info(
                        "[MultiScale] Meta cache hit but meta_lgbm.txt missing,"
                        "Retraining meta-LGBM..."
                    )
                    try:
                        _weights_dir.mkdir(parents=True, exist_ok=True)
                        meta_result_rebuild, _, _, _ = self.train_and_eval(
                            meta_X_train, tr_lb, meta_X_val, vl_lb, label_map,
                            return_model=True,
                        )
                        _rm = meta_result_rebuild.pop("_model", None)
                        if _rm is not None:
                            _rm.booster_.save_model(str(_meta_txt))
                            logger.info(
                                "[MultiScale] meta-LGBM model saved (rebuild):"
                                "meta_lgbm.txt"
                            )
                            _missing_pmeta = True  # Also write pipeline_meta together
                        del _rm
                    except Exception as _e:
                        logger.warning(
                            f"[MultiScale] Meta model rebuild failed: {_e}"
                        )
                if _missing_pmeta:
                    _weights_dir.mkdir(parents=True, exist_ok=True)
                    with open(_pmeta_json, "w", encoding="utf-8") as _pm:
                        json.dump({
                            "n_groups": n_groups,
                            "group_names": list(factor_groups.keys()),
                            "classes_sorted": classes_sorted,
                            "id_to_name": id_to_name,
                            "n_classes": n_classes,
                            "residual_stage1": self.residual_stage1,
                            "residual_stage2": self.residual_stage2,
                        }, _pm, ensure_ascii=False)
                    logger.info(
                        "[MultiScale] pipeline_meta.json saved (backfill)"
                    )
                # On cache hit, meta_X no longer needed, release immediately
                del meta_X_train, meta_X_val; gc.collect()

        if not meta_cache_hit:
            if n_groups > 1:
                _weights_dir = Path(output_dir) / "weights"
                if inference_only:
                    # ── Load pre-trained meta-LGBM from weights/, predict only ──
                    import lightgbm as lgb
                    _meta_txt = _weights_dir / "meta_lgbm.txt"
                    if not _meta_txt.exists():
                        raise FileNotFoundError(
                            f"[Inference] meta-LGBM model not found: {_meta_txt}")
                    _meta_model = lgb.Booster(model_file=str(_meta_txt))
                    logger.info(
                        f"[Inference] Loaded pre-trained meta-LGBM: meta_lgbm.txt"
                        f" (features={_meta_model.num_feature()})")

                    # Predict
                    _raw_tr = _meta_model.predict(meta_X_train)
                    _raw_vl = _meta_model.predict(meta_X_val)
                    proba_meta_train = _raw_tr.reshape(-1, n_classes).astype(np.float64)
                    proba_meta_val   = _raw_vl.reshape(-1, n_classes).astype(np.float64)
                    del _meta_model, _raw_tr, _raw_vl

                    # Compute metrics from predictions
                    y_pred_meta = np.array([classes_sorted[i] for i in proba_meta_val.argmax(axis=1)])
                    meta_cm = confusion_matrix(vl_lb, y_pred_meta, labels=classes_sorted)
                    meta_cm_norm = meta_cm.astype(np.float64) / np.maximum(
                        meta_cm.sum(axis=1, keepdims=True), 1)
                    _prec_m, _rec_m, _f1_m, _sup_m = precision_recall_fscore_support(
                        vl_lb, y_pred_meta, labels=classes_sorted, zero_division=0)
                    meta_per_class = {}
                    for _ci, _c in enumerate(classes_sorted):
                        _cname = id_to_name[_c]
                        _y_bin = (vl_lb == _c).astype(int)
                        try:
                            _auc_val = float(roc_auc_score(_y_bin, proba_meta_val[:, _ci]))
                        except ValueError:
                            _auc_val = None
                        meta_per_class[_cname] = {
                            "precision": round(float(_prec_m[_ci]), 4),
                            "recall": round(float(_rec_m[_ci]), 4),
                            "f1": round(float(_f1_m[_ci]), 4),
                            "auc": round(_auc_val, 4) if _auc_val is not None else None,
                            "support": int(_sup_m[_ci]),
                        }
                    _macro_f1_mi = float(np.mean([meta_per_class[cn]["f1"] for cn in class_names]))
                    _macro_auc_mi = float(np.mean([meta_per_class[cn]["auc"] for cn in class_names
                                                 if meta_per_class[cn]["auc"] is not None]))
                    _acc_mi = round(float(accuracy_score(vl_lb, y_pred_meta)), 4)
                    _bal_mi = round(float(balanced_accuracy_score(vl_lb, y_pred_meta)), 4)
                    _wf1_mi = round(float(f1_score(vl_lb, y_pred_meta, labels=classes_sorted,
                                                  average="weighted", zero_division=0)), 4)
                    try:
                        _wauc_mi = float(roc_auc_score(vl_lb, proba_meta_val,
                                                       multi_class="ovr", average="weighted",
                                                       labels=classes_sorted))
                    except Exception:
                        _wauc_mi = 0.0
                    meta_result = {
                        "metrics": {
                            "accuracy": _acc_mi,
                            "balanced_accuracy": _bal_mi,
                            "macro_f1": round(_macro_f1_mi, 4),
                            "macro_auc": round(_macro_auc_mi, 4),
                            "weighted_f1": _wf1_mi,
                            "weighted_auc": round(_wauc_mi, 4),
                            "per_class": meta_per_class,
                        },
                    }
                    _mm = meta_result["metrics"]
                    logger.info(
                        f"[Inference][meta-LGBM] accuracy={_mm.get('accuracy')}  "
                        f"balanced_acc(per_class_acc_mean)={_mm.get('balanced_accuracy')}  "
                        f"macro_auc={_mm.get('macro_auc')}  "
                        f"macro_f1={_mm.get('macro_f1')}  weighted_f1={_mm.get('weighted_f1')}"
                    )
                    self.save_report(meta_result, meta_cm, meta_cm_norm, class_names, output_dir)
                    meta_result.update({
                        "timestamp": datetime.now().isoformat(),
                        "n_factors_used": sum(m["n_factors"] for m in group_metrics.values()),
                        "group_metrics": group_metrics,
                        "label_map": label_map,
                        "lgbm_params": self.lgbm_params,
                        "purity_mode": purity_mode,
                        "multiresolution": True,
                        "multiscale_temporal": True,
                    })
                    sd = getattr(self, '_stage_data', {})
                    sd["meta_per_class"] = meta_per_class
                    sd["meta_cm"] = meta_cm
                    sd["meta_proba"] = proba_meta_val
                    sd["meta_preds"] = y_pred_meta

                else:
                    # Multiple groups: train meta-LGBM
                    logger.info(
                        f"[MultiScale][meta-LGBM] Starting meta-LGBM training:"
                        f"train={meta_X_train.shape}，val={meta_X_val.shape}"
                    )
                    meta_result, meta_cm, meta_cm_norm, _ = self.train_and_eval(
                        meta_X_train, tr_lb, meta_X_val, vl_lb, label_map,
                        return_model=True,
                    )
                    # -- Save meta-LGBM model for inference.py to use --
                    _weights_dir.mkdir(parents=True, exist_ok=True)
                    _meta_model = meta_result.pop("_model", None)
                    if _meta_model is not None:
                        try:
                            _meta_model.booster_.save_model(
                                str(_weights_dir / "meta_lgbm.txt")
                            )
                            logger.info("[MultiScale] meta-LGBM model saved: meta_lgbm.txt")
                            # Also save meta config
                            with open(_weights_dir / "pipeline_meta.json", "w",
                                      encoding="utf-8") as _pm:
                                json.dump({
                                    "n_groups": n_groups,
                                    "group_names": list(factor_groups.keys()),
                                    "classes_sorted": classes_sorted,
                                    "id_to_name": id_to_name,
                                    "n_classes": n_classes,
                                    "residual_stage1": self.residual_stage1,
                                    "residual_stage2": self.residual_stage2,
                                }, _pm, ensure_ascii=False)
                        except Exception as _e:
                            logger.warning(f"[MultiScale] Meta model save failed: {_e}")
                        del _meta_model
                    # Record key meta stage metrics
                    _mm = meta_result.get("metrics", {})
                    logger.info(
                        f"[MultiScale][meta-LGBM] accuracy={_mm.get('accuracy')}  "
                        f"balanced_acc(per_class_acc_mean)={_mm.get('balanced_accuracy')}  "
                        f"macro_auc={_mm.get('macro_auc')}  weighted_auc={_mm.get('weighted_auc')}  "
                        f"macro_f1={_mm.get('macro_f1')}  weighted_f1={_mm.get('weighted_f1')}  "
                        f"top_k_accuracy={_mm.get('top_k_accuracy')}"
                    )
                    meta_result.update({
                        "timestamp": datetime.now().isoformat(),
                        "n_factors_used": sum(m["n_factors"] for m in group_metrics.values()),
                        "group_metrics": group_metrics,
                        "label_map": label_map,
                        "lgbm_params": self.lgbm_params,
                        "purity_mode": purity_mode,
                        "multiresolution": True,
                        "multiscale_temporal": True,
                    })
                    files = self.save_report(meta_result, meta_cm, meta_cm_norm, class_names, output_dir)
                    meta_result["output_files"] = files
                    proba_meta_train = meta_result.pop("_proba_train")
                    proba_meta_val   = meta_result.pop("_proba_val")

                # -- Collect meta stage data for cross-stage visualization --
                sd = getattr(self, '_stage_data', {})
                sd["meta_per_class"] = meta_result.get("metrics", {}).get("per_class", {})
                sd["meta_cm"] = meta_cm
                sd["meta_proba"] = proba_meta_val
                sd["meta_preds"] = np.array([classes_sorted[i] for i in proba_meta_val.argmax(axis=1)])

                # -- Write meta proba cache --
                if _save_proba_cache:
                    self._save_stage(
                        meta_sc_dir,
                        proba_meta_train=proba_meta_train,
                        proba_meta_val=proba_meta_val,
                        tr_lb=tr_lb,
                        vl_lb=vl_lb,
                    )
                del meta_X_train, meta_X_val; gc.collect()
            else:
                logger.info("[MultiScale] Only single group, skipping meta-LGBM, directly entering temporal stage")
                proba_meta_train = meta_X_train
                proba_meta_val   = meta_X_val
                only_group = list(group_metrics.keys())[0]
                only_gm = group_metrics[only_group]
                # Single group: meta = that group (reuse group per_class + CM)
                sd = getattr(self, '_stage_data', {})
                only_per_class = sd.get("group_per_class", {}).get(only_group, {})
                only_cm = sd.get("group_cms", {}).get(only_group)
                only_preds = sd.get("group_preds", {}).get(only_group)
                only_proba = sd.get("group_probas", {}).get(only_group)
                meta_result = {
                    "metrics": {**only_gm, "per_class": only_per_class},
                    "note": "single_group_passthrough",
                }
                if only_per_class:
                    sd["meta_per_class"] = only_per_class
                if only_cm is not None:
                    sd["meta_cm"] = only_cm
                if only_proba is not None:
                    sd["meta_proba"] = only_proba
                if only_preds is not None:
                    sd["meta_preds"] = only_preds

        # Residual connection (Stage 2): meta_X passed as extra features to temporal stage (concat after temporal feature construction)
        _residual2_train = meta_X_train_for_bm if self.residual_stage2 else None
        _residual2_val   = meta_X_val_for_bm   if self.residual_stage2 else None
        if self.residual_stage2:
            logger.info(
                f"[MultiScale][Residual-Stage2] meta_X shape={meta_X_train_for_bm.shape},"
                f"will concat after temporal feature construction"
            )

        # ---- Meta stage dedicated visualizations (generated to 05_meta) ----
        if viz_dir:
            try:
                _meta_viz = str(Path(viz_dir) / "05_meta")
                sd = getattr(self, '_stage_data', {})
                meta_cm_viz = sd.get("meta_cm")
                meta_proba_viz = sd.get("meta_proba")
                meta_preds_viz = sd.get("meta_preds")

                if meta_cm_viz is not None:
                    plot_group_confusion_matrix(
                        cm=meta_cm_viz, class_names=class_names,
                        viz_dir=_meta_viz, group_name="meta",
                    )
                if meta_proba_viz is not None and vl_lb is not None:
                    plot_proba_distribution(
                        proba=meta_proba_viz, labels=vl_lb,
                        class_names=class_names,
                        viz_dir=_meta_viz, group_name="meta",
                    )
                    plot_confidence_vs_correctness(
                        proba=meta_proba_viz, labels=vl_lb,
                        class_names=class_names,
                        viz_dir=_meta_viz, stage_name="meta",
                    )
                # Meta per-class metrics table
                meta_pc = sd.get("meta_per_class", {})
                if meta_pc:
                    plot_best_model_metrics_table(
                        metrics={"per_class": meta_pc, **meta_result.get("metrics", {})},
                        model_name="meta-LGBM",
                        viz_dir=_meta_viz,
                    )
                logger.info("[MultiScale] Meta stage visualizations generated in 05_meta")
            except Exception as _e:
                logger.warning(f"[Viz] Meta stage visualization failed: {_e}")

        # ---- skip_temporal: early return meta proba + extra features (for validation.py to use) ----
        if skip_temporal:
            logger.info("[MultiScale] skip_temporal=True, skipping temporal stage, returning meta proba")
            if progress_callback:
                progress_callback(1.0, "Pipeline complete (meta-LGBM)")
            result = {
                "group_metrics": group_metrics,
                "meta_metrics": meta_result.get("metrics", {}),
                "_proba_meta_train": proba_meta_train,
                "_proba_meta_val": proba_meta_val,
                "_extra_train": _residual2_train,
                "_extra_val": _residual2_val,
            }
            return result

        # ---- Step 4: Temporal LGBM ----
        if progress_callback:
            progress_callback(_meta_frac, "Running temporal model...")
        # stage1 affects meta proba, stage2 additionally concats meta_X after temporal features, both change temporal LGBM input, so both reflected in suffix
        temporal_sc_dir = sc_root / f"temporal_lgbm{_r1_suffix}{_r2_suffix}"
        temporal_cache_hit = False
        if _use_proba_cache:
            sc_temporal = self._load_stage(temporal_sc_dir, ["proba_temporal_train", "proba_temporal_val"])
            if sc_temporal is not None:
                # Check weight file integrity: if missing, cannot skip temporal model training
                _weights_dir = Path(output_dir) / "weights"
                _temporal_model_type = cfg.get("temporal_validation", {}).get("temporal_model", "lgbm")
                if _temporal_model_type in ("bilstm", "transformer", "mamba"):
                    _temporal_weight = _weights_dir / f"{_temporal_model_type}_best.pt"
                else:
                    _temporal_weight = _weights_dir / "temporal_lgbm.txt"
                if _temporal_weight.exists():
                    temporal_cache_hit = True
                else:
                    logger.info(
                        f"[MultiScale][StageCache] Temporal cache hit but weight file missing "
                        f"({_temporal_weight.name}), will retrain temporal model to generate weights"
                    )
                logger.info("[MultiScale][StageCache] Step 4 hit: skipping temporal LGBM training")
                # Generate confusion matrix from cached proba
                try:
                    proba_temporal_val = sc_temporal["proba_temporal_val"]
                    y_pred_temporal = np.argmax(proba_temporal_val, axis=1)
                    temporal_cm = confusion_matrix(vl_lb, y_pred_temporal, labels=classes_sorted)
                    temporal_cm_norm = temporal_cm.astype(np.float64) / np.maximum(
                        temporal_cm.sum(axis=1, keepdims=True), 1
                    )
                    t_validator_temp = TemporalValidator(cfg)
                    t_files = t_validator_temp.save_report(
                        {"metrics": {}}, temporal_cm, temporal_cm_norm,
                        class_names, output_dir,
                    )
                    logger.info("[StageCache] Generated temporal confusion matrix from cached proba")
                except Exception as _e:
                    logger.warning(f"[StageCache] Failed to generate temporal confusion matrix from cache: {_e}")
                    t_files = {}
                # Construct a dummy t_result
                t_result = {
                    "metrics": {"note": "stage_cache_hit"},
                    "output_files": t_files,
                }

        if not temporal_cache_hit:
            # ---- GPU VRAM cleanup (must release fragmented VRAM before BiLSTM training) ----
            # Previous group LGBM (short/medium/long) + meta LGBM trained on GPU,
            # PyTorch caching allocator leaves a lot of fragmented VRAM.
            # Without cleanup, BiLSTM large batch forward pass will OOM due to inability to allocate contiguous VRAM blocks.
            import gc as _gc_pre
            _gc_pre.collect()
            try:
                import torch as _torch_pre
                if _torch_pre.cuda.is_available():
                    _torch_pre.cuda.empty_cache()
                    _torch_pre.cuda.synchronize()
                    _alloc = _torch_pre.cuda.memory_allocated() / (1024**3)
                    _reserved = _torch_pre.cuda.memory_reserved() / (1024**3)
                    logger.info(
                        f"[MultiScale] GPU VRAM cleanup done: allocated={_alloc:.2f}GiB, "
                        f"reserved={_reserved:.2f}GiB (fragments released)"
                    )
            except Exception as _e_clean:
                logger.warning(f"[MultiScale] GPU VRAM cleanup failed (non-fatal): {_e_clean}")

            logger.info("[MultiScale] Starting temporal validation (TemporalValidator)")
            t_validator = TemporalValidator(cfg)
            t_result = t_validator.run(
                proba_train=proba_meta_train,
                y_train=tr_lb,
                proba_val=proba_meta_val,
                y_val=vl_lb,
                label_map=label_map,
                output_dir=output_dir,
                min_segment=cfg.get("temporal_validation", {}).get("min_segment", 30),
                extra_train=_residual2_train,
                extra_val=_residual2_val,
                viz_dir=viz_dir,
            )

        # Concatenate each group's standardized factor matrix for OvR direct reuse (avoid recomputation)
        import gc as _gc
        if all_X_tr_std_parts:
            X_all_tr_std = np.hstack(all_X_tr_std_parts)
            X_all_vl_std = np.hstack(all_X_vl_std_parts)
            del all_X_tr_std_parts, all_X_vl_std_parts; _gc.collect()
        else:
            X_all_tr_std = None
            X_all_vl_std = None

        # ---- Cross-stage visualization: collect stage data and generate comparative analysis ----
        if viz_dir:
            try:
                from src.visualization import (
                    plot_confidence_vs_correctness,
                    plot_confusion_matrix_delta,
                    plot_per_class_metrics_heatmap,
                    plot_topk_accuracy_curve,
                    plot_error_vs_duration,
                    plot_prediction_flip_flow,
                    plot_calibration_curve,
                    plot_per_class_auc_comparison,
                    plot_stage_radar,
                    plot_segment_duration_analysis,
                )
                sd = getattr(self, '_stage_data', {})

                # -- Collect temporal stage data --
                tm = t_result.get("metrics", {})
                sd["temporal_per_class"] = tm.get("per_class", {})
                sd["temporal_preds"] = np.array([classes_sorted[i] for i in proba_meta_val.argmax(axis=1)])
                sd["temporal_proba"] = proba_meta_val

                # -- Collect sequence decoding stage data --
                seq_dec = t_result.get("sequence_decoding", {})
                for dec_key, dec_m in seq_dec.items():
                    if dec_m and "per_class" in dec_m:
                        sd.setdefault("decode_per_class", {})[dec_key] = dec_m.get("per_class", {})
                        if dec_m.get("y_pred") is not None:
                            sd.setdefault("decode_preds", {})[dec_key] = np.asarray(dec_m["y_pred"])

                # -- Build per_class_by_stage --
                per_class_by_stage = {}
                for gname, gpc in sd.get("group_per_class", {}).items():
                    per_class_by_stage[gname] = gpc
                if sd.get("meta_per_class"):
                    per_class_by_stage["meta"] = sd["meta_per_class"]
                if sd.get("temporal_per_class"):
                    per_class_by_stage["temporal"] = sd["temporal_per_class"]
                for dk, dpc in sd.get("decode_per_class", {}).items():
                    per_class_by_stage[dk] = dpc

                # -- Build cm_stages --
                cm_stages = {}
                for gname, gcm in sd.get("group_cms", {}).items():
                    cm_stages[gname] = gcm
                if sd.get("meta_cm") is not None:
                    cm_stages["meta"] = sd["meta_cm"]

                # -- Build proba_by_stage (for top-k curves) --
                proba_by_stage = {}
                for gname, gp in sd.get("group_probas", {}).items():
                    proba_by_stage[gname] = gp
                if sd.get("meta_proba") is not None:
                    proba_by_stage["meta"] = sd["meta_proba"]
                if sd.get("temporal_proba") is not None:
                    proba_by_stage["temporal"] = sd["temporal_proba"]

                # -- Build y_pred_by_stage (for error vs duration, flip flow) --
                y_pred_by_stage = {}
                for gname, gpred in sd.get("group_preds", {}).items():
                    y_pred_by_stage[gname] = gpred
                if sd.get("meta_preds") is not None:
                    y_pred_by_stage["meta"] = sd["meta_preds"]
                if sd.get("temporal_preds") is not None:
                    y_pred_by_stage["temporal"] = sd["temporal_preds"]
                for dk, dpred in sd.get("decode_preds", {}).items():
                    y_pred_by_stage[dk] = dpred

                # -- Build stage_metrics for radar --
                stage_metrics = {}
                for gname, gm in group_metrics.items():
                    stage_metrics[gname] = gm
                mm = meta_result.get("metrics", {})
                if mm and "accuracy" in mm:
                    stage_metrics["meta"] = mm
                if tm and "accuracy" in tm:
                    stage_metrics["temporal"] = tm
                for dec_key, dec_m in seq_dec.items():
                    if dec_m and "accuracy" in dec_m:
                        stage_metrics[dec_key] = dec_m

                # -- Generate cross-stage visualizations --
                if len(per_class_by_stage) > 1:
                    plot_per_class_metrics_heatmap(
                        per_class_by_stage=per_class_by_stage,
                        class_names=class_names, viz_dir=viz_dir,
                    )
                    plot_per_class_auc_comparison(
                        per_class_by_stage=per_class_by_stage,
                        class_names=class_names, viz_dir=viz_dir,
                    )

                if cm_stages and len(cm_stages) > 1:
                    plot_confusion_matrix_delta(
                        cm_stages=cm_stages,
                        class_names=class_names, viz_dir=viz_dir,
                    )

                if proba_by_stage and len(proba_by_stage) > 1:
                    plot_topk_accuracy_curve(
                        proba_by_stage=proba_by_stage,
                        labels=vl_lb, class_names=class_names, viz_dir=viz_dir,
                    )

                if y_pred_by_stage and len(y_pred_by_stage) > 1:
                    plot_prediction_flip_flow(
                        y_pred_by_stage=y_pred_by_stage,
                        class_names=class_names, viz_dir=viz_dir,
                    )
                    plot_error_vs_duration(
                        y_true=vl_lb, y_pred_by_stage=y_pred_by_stage,
                        class_names=class_names, viz_dir=viz_dir,
                    )

                if sd.get("meta_proba") is not None and vl_lb is not None:
                    plot_confidence_vs_correctness(
                        proba=sd["meta_proba"], labels=vl_lb,
                        class_names=class_names, viz_dir=viz_dir, stage_name="meta",
                    )
                    plot_calibration_curve(
                        proba=sd["meta_proba"], labels=vl_lb,
                        class_names=class_names, viz_dir=viz_dir, stage_name="meta",
                    )

                if stage_metrics and len(stage_metrics) > 1:
                    plot_stage_radar(
                        stage_metrics=stage_metrics, viz_dir=viz_dir,
                    )

                if vl_lb is not None:
                    plot_segment_duration_analysis(
                        y_true=vl_lb, class_names=class_names, viz_dir=viz_dir,
                    )

                logger.info("[MultiScale] Cross-stage model performance visualizations all completed")
            except Exception as _e:
                logger.warning(f"[Viz] Cross-stage visualization failed: {_e}", exc_info=True)

        if progress_callback:
            progress_callback(1.0, "Pipeline complete")

        return {
            "group_metrics": group_metrics,
            "meta_metrics": meta_result.get("metrics", {}),
            "temporal_metrics": {**t_result.get("metrics", {}), "model": t_result.get("model", "lgbm")},
            "sequence_decoding": t_result.get("sequence_decoding", {}),
            "n_groups": n_groups,
            "purity_mode": purity_mode,
            "val_purity_mode": _val_purity,
            "multiresolution": True,
            "multiscale_temporal": True,
            "output_files": {
                **meta_result.get("output_files", {}),
                **t_result.get("output_files", {}),
            },
            "_proba_meta_train": proba_meta_train,
            "_proba_meta_val": proba_meta_val,
            "_proba_temporal_val": t_result.get("_proba_val"),  # For post-processing
            # Meta stage feature matrix for ModelBenchmark to use
            "_X_meta_train": meta_X_train_for_bm,
            "_X_meta_val":   meta_X_val_for_bm,
            "_y_meta_train": tr_lb,
            "_y_meta_val":   vl_lb,
            # Full standardized factor matrix for OvR reuse (groups concatenated, avoid recomputation)
            "_X_all_train_std": X_all_tr_std,
            "_X_all_val_std":   X_all_vl_std,
            "_all_used_names":  all_used_names,
        }

    # ------------------------------------------------------------------
    # Cache helper methods
    # ------------------------------------------------------------------
    @staticmethod
    def _fingerprint(arr: np.ndarray) -> dict:
        raw = arr.ravel().view(np.uint8)
        nbytes = len(raw)
        head = raw[:4096].tobytes() if nbytes >= 4096 else raw.tobytes()
        tail = raw[-4096:].tobytes() if nbytes >= 4096 else raw.tobytes()
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "head_md5": hashlib.md5(head).hexdigest(),
            "tail_md5": hashlib.md5(tail).hexdigest(),
        }

    def _cache_key(
        self,
        factors: list,
        train_kp: np.ndarray,
        val_kp: np.ndarray,
        flat_attributes: list,
        nan_fill: str,
        clip_lo: float = -8.0,
        clip_hi: float = 8.0,
    ) -> str:
        payload = {
            "factors": sorted(
                [(f["name"], f["code"], f.get("mode", "row"), f.get("seq_length", 1)) for f in factors]
            ),
            "train_fp": self._fingerprint(train_kp),
            "val_fp": self._fingerprint(val_kp),
            "seq_length": int(train_kp.shape[1]),
            "flat_attr": list(flat_attributes),
            "nan_fill": nan_fill,
            "clip": [clip_lo, clip_hi],
            "min_valid_ratio": self.engine.min_valid_ratio,
            "max_error_ratio": self.engine.max_error_ratio,
            "code_version": "v1",
        }
        return hashlib.md5(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

    @staticmethod
    def _save_cache(
        cache_dir: Path,
        X_train_std: np.ndarray,
        X_val_std: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        scaler,
        fill: np.ndarray,
        used_names: list,
        dropped: list,
        meta_extra: dict,
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "X_train_std.npy", X_train_std)
        np.save(cache_dir / "X_val_std.npy", X_val_std)
        np.save(cache_dir / "y_train.npy", y_train.astype(np.int64))
        np.save(cache_dir / "y_val.npy", y_val.astype(np.int64))
        np.save(cache_dir / "scaler_mean.npy", scaler.mean_.astype(np.float32))
        np.save(cache_dir / "scaler_scale.npy", scaler.scale_.astype(np.float32))
        np.save(cache_dir / "fill.npy", fill.astype(np.float32))
        meta = {
            "used_factor_names": used_names,
            "factors_dropped": dropped,
            "created_at": datetime.now().isoformat(),
            "code_version": "v1",
        }
        meta.update(meta_extra)
        with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"Factor matrix cache written: {cache_dir}")

    @staticmethod
    def _load_cache(cache_dir: Path) -> dict:
        required = [
            "X_train_std.npy", "X_val_std.npy",
            "y_train.npy", "y_val.npy",
            "scaler_mean.npy", "scaler_scale.npy",
            "fill.npy", "meta.json",
        ]
        missing = [f for f in required if not (cache_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Cache directory {cache_dir} missing files: {missing}, will recompute."
            )
        with open(cache_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        state = {
            "X_train_std": np.load(cache_dir / "X_train_std.npy"),
            "X_val_std": np.load(cache_dir / "X_val_std.npy"),
            "y_train": np.load(cache_dir / "y_train.npy"),
            "y_val": np.load(cache_dir / "y_val.npy"),
            "scaler_mean": np.load(cache_dir / "scaler_mean.npy"),
            "scaler_scale": np.load(cache_dir / "scaler_scale.npy"),
            "fill": np.load(cache_dir / "fill.npy"),
            "used_names": meta.get("used_factor_names", []),
            "dropped": meta.get("factors_dropped", []),
            "meta": meta,
        }
        logger.info(
            f"Cache hit: {cache_dir},"
            f"using {len(state['used_names'])} factors,"
            f"X_train {state['X_train_std'].shape}，X_val {state['X_val_std'].shape}"
        )
        return state

    # ------------------------------------------------------------------
    # Pipeline intermediate stage cache helper methods
    # ------------------------------------------------------------------
    # proba-like arrays stored as float16 (sufficient precision, half space)
    _PROBA_KEYS = {"proba_train", "proba_val", "proba_meta_train", "proba_meta_val",
                   "proba_temporal_train", "proba_temporal_val"}

    @staticmethod
    def _save_stage(stage_dir: Path, **arrays) -> None:
        """Save named numpy arrays to stage_dir, proba arrays compressed to float16."""
        stage_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for name, arr in arrays.items():
            if isinstance(arr, np.ndarray):
                # proba matrix uses float16 to reduce disk usage
                if name in SynthValidator._PROBA_KEYS:
                    arr = arr.astype(np.float16)
                np.save(stage_dir / f"{name}.npy", arr)
                saved.append(name)
            elif isinstance(arr, list):
                with open(stage_dir / f"{name}.json", "w", encoding="utf-8") as f:
                    json.dump(arr, f, ensure_ascii=False)
                saved.append(name)
        with open(stage_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"saved": saved, "created_at": datetime.now().isoformat()}, f)
        logger.info(f"[StageCache] Saved: {stage_dir} ({saved})")

    @staticmethod
    def _load_stage(stage_dir: Path, keys: list) -> Optional[dict]:
        """Load arrays for specified keys from stage_dir, proba arrays restored to float32."""
        if not stage_dir.exists():
            return None
        result = {}
        for key in keys:
            npy_p = stage_dir / f"{key}.npy"
            json_p = stage_dir / f"{key}.json"
            if npy_p.exists():
                arr = np.load(npy_p)
                # Restore float16 proba to float32 for subsequent computation
                if key in SynthValidator._PROBA_KEYS and arr.dtype == np.float16:
                    arr = arr.astype(np.float32)
                result[key] = arr
            elif json_p.exists():
                with open(json_p, "r", encoding="utf-8") as f:
                    result[key] = json.load(f)
            else:
                logger.debug(f"[StageCache] Missing {key}, skipping cache hit")
                return None
        logger.info(f"[StageCache] Hit: {stage_dir} ({keys})")
        return result

    # ------------------------------------------------------------------
    # 3) Standardization (NaN fill + StandardScaler, fit training set only)
    # ------------------------------------------------------------------
    def standardize(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        viz_dir: str = "",
        split_name: str = "",
    ) -> tuple:
        # Convert inf/-inf to nan in-place, avoiding creating full copies
        X_train = X_train.astype(np.float32, copy=False)
        X_val   = X_val.astype(np.float32, copy=False)
        X_train[~np.isfinite(X_train)] = np.nan
        X_val[~np.isfinite(X_val)]     = np.nan

        if self.nan_fill == "mean":
            fill = np.nanmean(X_train, axis=0)
            np.nan_to_num(fill, nan=0.0, copy=False)
        else:
            fill = np.zeros(X_train.shape[1], dtype=np.float32)

        # -- Visualization: snapshot per-column mean/std before standardization --
        if viz_dir:
            try:
                pre_mean = np.nanmean(X_train, axis=0)
                pre_std  = np.nanstd(X_train, axis=0)
            except Exception:
                pre_mean = pre_std = None

        # In-place NaN fill, without creating extra copies
        nan_mask_tr = np.isnan(X_train)
        X_train[nan_mask_tr] = np.broadcast_to(fill, X_train.shape)[nan_mask_tr]
        del nan_mask_tr

        nan_mask_vl = np.isnan(X_val)
        X_val[nan_mask_vl] = np.broadcast_to(fill, X_val.shape)[nan_mask_vl]
        del nan_mask_vl

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train).astype(np.float32)
        del X_train
        # Fix zero-variance columns: scale_=0 produces inf/nan in transform, replace scale_ with 1
        zero_scale_mask = scaler.scale_ == 0
        if zero_scale_mask.any():
            n_zero = zero_scale_mask.sum()
            logger.warning(
                f"[{split_name}] StandardScaler: {n_zero} columns have zero variance, scale_ replaced with 1.0 to avoid NaN"
            )
            scaler.scale_[zero_scale_mask] = 1.0
        X_val_std = scaler.transform(X_val).astype(np.float32)
        del X_val
        np.clip(X_train_std, -8.0, 8.0, out=X_train_std)
        np.clip(X_val_std,   -8.0, 8.0, out=X_val_std)
        # Safe fallback: clip cannot remove NaN, fill remaining NaN/inf with 0
        X_train_std[~np.isfinite(X_train_std)] = 0.0
        X_val_std[~np.isfinite(X_val_std)]     = 0.0

        # -- Visualization: mean/std comparison after standardization --
        if viz_dir and pre_mean is not None:
            try:
                post_mean = X_train_std.mean(axis=0)
                post_std  = X_train_std.std(axis=0)
                plot_standardize_effect(
                    pre_mean=pre_mean, pre_std=pre_std,
                    post_mean=post_mean, post_std=post_std,
                    fill_values=fill,
                    viz_dir=viz_dir, split_name=split_name,
                )
            except Exception as _e:
                logger.warning(f"[Viz] Standardization visualization failed: {_e}")

        return X_train_std, X_val_std, scaler, fill

    # ------------------------------------------------------------------
    # 4) Training + evaluation
    # ------------------------------------------------------------------
    def train_and_eval(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        label_map: dict,
        return_model: bool = False,
    ) -> dict:
        import gc
        y_train = np.asarray(y_train).astype(int)
        y_val = np.asarray(y_val).astype(int)

        classes_sorted = sorted(set(int(v) for v in label_map.values()))
        id_to_name = {int(v): k for k, v in label_map.items()}
        class_names = [id_to_name[c] for c in classes_sorted]

        n_classes = len(classes_sorted)
        logger.info(
            f"Training multiclass LGBM: n_classes={n_classes}, class_weight={self.class_weight}, "
            f"early_stopping_rounds={self.early_stopping_rounds}"
        )
        logger.info(f"Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        logger.info(f"Val label distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")

        safe_params = dict(self.lgbm_params)
        min_leaf = max(1, len(y_train) // (n_classes * 20))
        safe_params["min_data_in_leaf"] = min(safe_params.get("min_data_in_leaf", min_leaf), min_leaf)

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            class_weight=self.class_weight if self.class_weight else None,
            verbose=-1,
            **safe_params,
        )
        fit_kwargs = {}
        if self.early_stopping_rounds > 0:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=10),
            ]
        model.fit(X_train, y_train, **fit_kwargs)

        # ---- Feature selection: select top-k factors by importance then retrain ----
        selected_indices = None
        if self.feature_selection_top_k > 0 and X_train.shape[1] > self.feature_selection_top_k:
            k_sel = min(self.feature_selection_top_k, X_train.shape[1])
            imp = model.feature_importances_
            selected_indices = np.argsort(imp)[::-1][:k_sel]
            selected_indices = np.sort(selected_indices)
            logger.info(f"Feature selection: selecting top-{k_sel} from {X_train.shape[1]} factors, retraining")
            # Release original matrix after making slice copy
            X_train = X_train[:, selected_indices].copy()
            X_val   = X_val[:, selected_indices].copy()
            del model; gc.collect()
            model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                class_weight=self.class_weight if self.class_weight else None,
                verbose=-1,
                **safe_params,
            )
            fit_kwargs2 = {}
            if self.early_stopping_rounds > 0:
                fit_kwargs2["eval_set"] = [(X_val, y_val)]
                fit_kwargs2["callbacks"] = [
                    lgb.early_stopping(self.early_stopping_rounds, verbose=True),
                    lgb.log_evaluation(period=10),
                ]
            model.fit(X_train, y_train, **fit_kwargs2)

        y_pred        = model.predict(X_val)
        y_proba       = model.predict_proba(X_val).astype(np.float32)   # [N_val, n_classes]
        y_proba_train = model.predict_proba(X_train).astype(np.float32) # [N_train, n_classes]
        model_classes = list(model.classes_)
        if not return_model:
            del model; gc.collect()
            _model = None
        else:
            _model = model

        # Align to full class order
        full_proba       = np.zeros((len(y_val),   n_classes), dtype=np.float32)
        full_proba_train = np.zeros((len(y_train), n_classes), dtype=np.float32)
        for i, c in enumerate(model_classes):
            if c in classes_sorted:
                j = classes_sorted.index(c)
                full_proba[:, j]       = y_proba[:, i]
                full_proba_train[:, j] = y_proba_train[:, i]
        del y_proba, y_proba_train; gc.collect()

        cm      = confusion_matrix(y_val, y_pred, labels=classes_sorted)
        cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        # top-k accuracy
        top_k_acc = None
        top_k_per_class = {}
        if self.top_k >= 2:
            k = min(self.top_k, n_classes)
            top_k_cols     = np.argsort(full_proba, axis=1)[:, -k:]
            top_k_class_ids = np.array([[classes_sorted[j] for j in row] for row in top_k_cols])
            hit = np.array([int(y_val[i]) in top_k_class_ids[i] for i in range(len(y_val))])
            top_k_acc = float(hit.mean())
            for c in classes_sorted:
                mask = y_val == c
                top_k_per_class[id_to_name[c]] = (
                    round(float(hit[mask].mean()), 4) if mask.any() else None
                )
            logger.info(f"top-{k} accuracy: {top_k_acc:.4f}")

        acc          = float(accuracy_score(y_val, y_pred))
        balanced_acc = float(balanced_accuracy_score(y_val, y_pred))
        macro_f1     = float(f1_score(y_val, y_pred, labels=classes_sorted, average="macro",    zero_division=0))
        weighted_f1  = float(f1_score(y_val, y_pred, labels=classes_sorted, average="weighted", zero_division=0))
        per_class_f1 = f1_score(y_val, y_pred, labels=classes_sorted, average=None, zero_division=0)

        present   = sorted(set(int(v) for v in y_val))
        macro_auc = 0.0
        weighted_auc_val = 0.0
        per_class_auc = {cname: None for cname in class_names}
        if len(present) >= 2:
            idx_present  = [classes_sorted.index(c) for c in present]
            proba_sub    = full_proba[:, idx_present]
            row_sum      = proba_sub.sum(axis=1, keepdims=True)
            row_sum      = np.where(row_sum <= 0, 1.0, row_sum)
            proba_sub_n  = proba_sub / row_sum
            del proba_sub
            try:
                macro_auc = float(roc_auc_score(y_val, proba_sub_n, multi_class="ovr",
                                                 average="macro", labels=present))
            except ValueError as e:
                logger.warning(f"Macro AUC computation failed: {e}")
            try:
                weighted_auc_val = float(roc_auc_score(y_val, proba_sub_n, multi_class="ovr",
                                                        average="weighted", labels=present))
            except ValueError as e:
                logger.warning(f"Weighted AUC computation failed: {e}")
            del proba_sub_n
            for c in present:
                cname = id_to_name[c]
                y_bin = (y_val == c).astype(int)
                col   = classes_sorted.index(c)
                try:
                    per_class_auc[cname] = float(roc_auc_score(y_bin, full_proba[:, col]))
                except ValueError:
                    per_class_auc[cname] = None

        report = classification_report(
            y_val, y_pred, labels=classes_sorted, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        per_class_metrics = {}
        for i, c in enumerate(classes_sorted):
            cname = class_names[i]
            r = report.get(cname, {})
            per_class_metrics[cname] = {
                "precision": round(float(r.get("precision", 0.0)), 4),
                "recall":    round(float(r.get("recall",    0.0)), 4),
                "f1":        round(float(per_class_f1[i]),         4),
                "support":   int(r.get("support", 0)),
                "auc":       round(per_class_auc[cname], 4) if per_class_auc[cname] is not None else None,
            }

        result = {
            "metrics": {
                "accuracy":          round(acc,          4),
                "balanced_accuracy": round(balanced_acc, 4),
                "top_k_accuracy":    round(top_k_acc, 4) if top_k_acc is not None else None,
                "top_k":             self.top_k if top_k_acc is not None else None,
                "top_k_per_class":   top_k_per_class if top_k_acc is not None else {},
                "macro_auc":         round(macro_auc,         4),
                "weighted_auc":      round(weighted_auc_val,  4),
                "macro_f1":          round(macro_f1,          4),
                "weighted_f1":       round(weighted_f1,       4),
                "per_class":         per_class_metrics,
            },
            "confusion_matrix":            cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "class_ids_in_order":          classes_sorted,
            "class_names_in_order":        class_names,
            "n_train":                     int(len(y_train)),
            "n_val":                       int(len(y_val)),
            "n_selected_factors":          int(X_train.shape[1]),
            "selected_indices":            selected_indices.tolist() if selected_indices is not None else None,
            "_proba_train": full_proba_train,
            "_proba_val":   full_proba,
        }
        # Overall metrics summary log
        logger.info(
            f"[train_and_eval] accuracy={acc:.4f}  "
            f"balanced_acc(per_class_acc_mean)={balanced_acc:.4f}  "
            f"macro_auc={macro_auc:.4f}  weighted_auc={weighted_auc_val:.4f}  "
            f"macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}"
        )
        # Per-class metrics log
        for cname, m in per_class_metrics.items():
            logger.info(
                f"[train_and_eval][per_class] {cname}: "
                f"precision={m['precision']}  recall={m['recall']}  "
                f"f1={m['f1']}  auc={m['auc']}  support={m['support']}"
            )
        if return_model and _model is not None:
            result["_model"] = _model
        return result, cm, cm_norm, class_names

    # ------------------------------------------------------------------
    # 4b) OvR binary classification metrics (computed directly from multiclass proba, no retraining)
    # ------------------------------------------------------------------
    def ovr_metrics_from_proba(
        self,
        proba_val: np.ndarray,
        y_val: np.ndarray,
        label_map: dict,
    ) -> dict:
        """
        Compute OvR binary classification metrics for each class directly from multiclass model proba output.

        For each class C:
          - y_bin = (y_true == C).astype(int)  positive class label
          - y_pred_bin = (argmax(proba) == C)  model prediction for this class
          - score = proba[:, C]                probability for this class
        All val frames participate in computation (no sampling, no retraining).
        """
        y_val = np.asarray(y_val).astype(int)
        proba_val = np.asarray(proba_val, dtype=np.float64)

        classes_sorted = sorted(set(int(v) for v in label_map.values()))
        id_to_name = {int(v): k for k, v in label_map.items()}

        y_pred = np.argmax(proba_val, axis=1)

        per_class: dict = {}
        accs, f1s, aucs, precs, recs = [], [], [], [], []

        for cls_id in classes_sorted:
            cls_name = id_to_name[cls_id]

            y_bin = (y_val == cls_id).astype(int)
            y_pred_bin = (y_pred == cls_id).astype(int)
            score = proba_val[:, cls_id]

            n_pos = int(np.sum(y_bin == 1))
            n_neg = int(np.sum(y_bin == 0))
            if n_pos == 0 or n_neg == 0:
                continue

            acc = float(accuracy_score(y_bin, y_pred_bin))
            f1  = float(f1_score(y_bin, y_pred_bin, average="binary", zero_division=0))
            tp  = int(np.sum((y_bin == 1) & (y_pred_bin == 1)))
            prec = float(tp / max(np.sum(y_pred_bin == 1), 1))
            rec  = float(tp / n_pos)
            try:
                auc = float(roc_auc_score(y_bin, score))
            except ValueError:
                auc = float("nan")

            per_class[cls_name] = {
                "acc": round(acc, 4), "f1": round(f1, 4),
                "auc": round(auc, 4) if not np.isnan(auc) else None,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "n_pos_val": n_pos, "n_neg_val": n_neg,
            }
            accs.append(acc); f1s.append(f1)
            precs.append(prec); recs.append(rec)
            if not np.isnan(auc):
                aucs.append(auc)

        macro_avg = {
            "acc": round(float(np.mean(accs)),  4) if accs  else None,
            "f1":  round(float(np.mean(f1s)),   4) if f1s   else None,
            "auc": round(float(np.mean(aucs)),  4) if aucs  else None,
            "precision": round(float(np.mean(precs)), 4) if precs else None,
            "recall":    round(float(np.mean(recs)),  4) if recs  else None,
        }
        for cn, v in per_class.items():
            logger.info(
                f"[OvR] '{cn}': auc={v['auc']}  f1={v['f1']}  "
                f"prec={v['precision']}  rec={v['recall']}  "
                f"pos/neg={v['n_pos_val']}/{v['n_neg_val']}"
            )
        logger.info(
            f"[OvR] macro_avg: auc={macro_avg['auc']}  f1={macro_avg['f1']}  "
            f"prec={macro_avg['precision']}  rec={macro_avg['recall']}"
        )
        per_class["macro_avg"] = macro_avg
        return per_class

    # ------------------------------------------------------------------
    # OvR metric entry point (computed directly from multiclass proba, no retraining)
    # ------------------------------------------------------------------
    def run_ovr(
        self,
        factors_path: str,
        train_kp_raw: np.ndarray,
        train_lb_raw: np.ndarray,
        val_kp_raw: np.ndarray,
        val_lb_raw: np.ndarray,
        flat_attributes: list,
        label_map: dict,
        output_dir: str,
        purity_mode: str = "nan_boundary",
        val_purity_mode: str = None,
        num_workers: int = 1,
        num_chunks: int = 256,
        cache_dir: str = "dataset_cache",
        stage_cache_dir: str = "",
        use_stage_cache: bool = False,
        refresh_stage_cache: bool = False,
        use_cache: bool = False,
        refresh_cache: bool = False,
        proba_val_precomputed: "np.ndarray | None" = None,
        y_val_precomputed: "np.ndarray | None" = None,
    ) -> dict:
        """
        Compute OvR per-class metrics directly from multiclass proba (full val, no sampling, no retraining).

        Preferentially use proba_val_precomputed (existing model output). Otherwise compute factor matrix +
        train a multiclass LGBM to get proba, then derive OvR metrics.
        """
        _val_purity = val_purity_mode if val_purity_mode is not None else purity_mode

        # ---- Fast path: directly use precomputed proba ----
        if proba_val_precomputed is not None and y_val_precomputed is not None:
            logger.info("[OvR] Computing OvR metrics directly from multiclass proba (no retraining)")
            ovr_result = self.ovr_metrics_from_proba(
                proba_val_precomputed, y_val_precomputed, label_map,
            )
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            report = {
                "timestamp": datetime.now().isoformat(),
                "precomputed": True,
                "label_map": label_map,
                "ovr_results": ovr_result,
            }
            json_path = out / "ovr_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"[OvR] Report saved: {json_path}")
            return {
                "ovr_results": ovr_result,
                "n_factors_used": 0,
                "factors_dropped": [],
                "output_files": {"json": str(json_path)},
            }

        # ---- Slow path: compute factor matrix + train multiclass LGBM + get proba ----
        logger.info("[OvR] No precomputed proba, computing factor matrix from scratch + training multiclass LGBM...")
        factors = self.load_valid_factors(factors_path)
        _tk = self._get_top_k()
        if _tk > 0 and len(factors) > _tk:
            for f in factors:
                pc = f.get("per_class", {})
                aucs = [v["auc"] if isinstance(v, dict) else v for v in pc.values()] if pc else [0.0]
                f["_mean_auc"] = float(np.mean(aucs))
            n_before = len(factors)
            factors = sorted(factors, key=lambda x: x["_mean_auc"], reverse=True)[:_tk]
            logger.info(f"[OvR] top-k selection: {n_before} -> {len(factors)} factors")

        if num_workers > 1:
            X_train, train_lb, used_tr, dropped_tr = self.compute_factor_matrix_multiresolution_fast(
                factors, train_kp_raw, train_lb_raw, flat_attributes,
                purity_mode=purity_mode, num_chunks=num_chunks,
                num_workers=num_workers, split_name="ovr_train", cache_dir=cache_dir,
            )
            X_val, val_lb, used_vl, dropped_vl = self.compute_factor_matrix_multiresolution_fast(
                factors, val_kp_raw, val_lb_raw, flat_attributes,
                purity_mode=_val_purity, num_chunks=num_chunks,
                num_workers=num_workers, split_name="ovr_val", cache_dir=cache_dir,
            )
        else:
            X_train, train_lb, used_tr, dropped_tr = self.compute_factor_matrix_multiresolution(
                factors, train_kp_raw, train_lb_raw, flat_attributes,
                purity_mode=purity_mode, split_name="ovr_train",
            )
            X_val, val_lb, used_vl, dropped_vl = self.compute_factor_matrix_multiresolution(
                factors, val_kp_raw, val_lb_raw, flat_attributes,
                purity_mode=_val_purity, split_name="ovr_val",
            )

        common = [n for n in used_tr if n in set(used_vl)]
        if len(common) < len(used_tr) or len(common) < len(used_vl):
            tr_idx = [used_tr.index(n) for n in common]
            vl_idx = [used_vl.index(n) for n in common]
            X_train, X_val = X_train[:, tr_idx], X_val[:, vl_idx]
        dropped_all = sorted(set(dropped_tr) | set(dropped_vl))

        X_train_std, X_val_std, _, _ = self.standardize(X_train, X_val)
        eval_result, _, _, _ = self.train_and_eval(
            X_train_std, train_lb, X_val_std, val_lb, label_map,
        )
        proba_val = eval_result.get("_proba_val")
        if proba_val is None:
            proba_val = np.zeros((len(val_lb), len(label_map)))

        ovr_result = self.ovr_metrics_from_proba(proba_val, val_lb, label_map)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": datetime.now().isoformat(),
            "n_factors_used": len(common),
            "factors_used": common,
            "factors_dropped": dropped_all,
            "label_map": label_map,
            "lgbm_params": self.lgbm_params,
            "purity_mode": purity_mode,
            "val_purity_mode": _val_purity,
            "ovr_results": ovr_result,
        }
        json_path = out / "ovr_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"[OvR] Report saved: {json_path}")
        return {
            "ovr_results": ovr_result,
            "n_factors_used": len(common),
            "factors_dropped": dropped_all,
            "output_files": {"json": str(json_path)},
        }

    def save_report(
        self,
        result: dict,
        cm: np.ndarray,
        cm_norm: np.ndarray,
        class_names: list,
        output_dir: str,
    ) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "synth_report.json"
        serializable = {k: v for k, v in result.items() if not k.startswith("_")}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved: {json_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping confusion matrix image output.")
            return {"json": str(json_path)}

        def _plot_cm(mat, title, fname, fmt):
            fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.9),
                                             max(5, len(class_names) * 0.8)))
            im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha="right")
            ax.set_yticklabels(class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            thresh = mat.max() / 2.0 if mat.size else 0
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, format(mat[i, j], fmt),
                            ha="center", va="center",
                            color="white" if mat[i, j] > thresh else "black",
                            fontsize=8)
            plt.tight_layout()
            path = out / fname
            plt.savefig(path, dpi=150)
            plt.close(fig)
            logger.info(f"Confusion matrix saved: {path}")
            return str(path)

        raw_png = _plot_cm(cm, "Confusion Matrix (raw)", "synth_confusion_matrix_raw.png", "d")
        # CSV: raw confusion matrix
        _cm_csv_rows = [[class_names[i]] + [str(cm[i, j]) for j in range(len(class_names))]
                        for i in range(len(class_names))]
        _write_csv(out / "synth_confusion_matrix_raw.csv", _cm_csv_rows,
                   ["true_class"] + [f"pred_{c}" for c in class_names])

        norm_png = ""
        if self.normalize_cm:
            norm_png = _plot_cm(cm_norm, "Confusion Matrix (row-normalized)",
                                "synth_confusion_matrix.png", ".2f")
            # CSV: normalized confusion matrix
            _cm_norm_csv_rows = [[class_names[i]] + [f"{cm_norm[i, j]:.4f}" for j in range(len(class_names))]
                                 for i in range(len(class_names))]
            _write_csv(out / "synth_confusion_matrix.csv", _cm_norm_csv_rows,
                       ["true_class"] + [f"pred_{c}" for c in class_names])

        return {"json": str(json_path), "cm_raw_png": raw_png, "cm_norm_png": norm_png}

    # ------------------------------------------------------------------
    # End-to-end entry
    # ------------------------------------------------------------------
    def run(
        self,
        factors_path: str,
        train_kp: np.ndarray,
        train_lb: np.ndarray,
        val_kp: np.ndarray,
        val_lb: np.ndarray,
        flat_attributes: list,
        label_map: dict,
        output_dir: str,
        num_workers: int = 32,
        num_chunks: int = 256,
        use_cache: bool = False,
        refresh_cache: bool = False,
        cache_dir: str = "dataset_cache",
    ) -> dict:
        factors = self.load_valid_factors(factors_path)

        # ---- Select top-k factors by mean_auc ----
        _tk = self._get_top_k()
        if _tk > 0 and len(factors) > _tk:
            for f in factors:
                pc = f.get("per_class", {})
                aucs = [v["auc"] if isinstance(v, dict) else v for v in pc.values()] if pc else [0.0]
                f["_mean_auc"] = float(np.mean(aucs))
            n_before = len(factors)
            factors = sorted(factors, key=lambda x: x["_mean_auc"], reverse=True)[:_tk]
            logger.info(
                f"[run] top-k selection: {n_before} -> {len(factors)} factors"
                f" (mean_auc cutoff={factors[-1]['_mean_auc']:.4f})"
            )

        cache_root = Path(cache_dir)

        # ---- Cache hit check ----
        key = self._cache_key(
            factors, train_kp, val_kp, flat_attributes,
            nan_fill=self.nan_fill,
        )
        cache_d = cache_root / f"factor_matrix_{key}"

        if use_cache and not refresh_cache and cache_d.exists():
            try:
                state = self._load_cache(cache_d)
                X_train_std = state["X_train_std"]
                X_val_std = state["X_val_std"]
                common = state["used_names"]
                dropped_all = state["dropped"]
            except FileNotFoundError as e:
                logger.warning(f"{e}, falling back to recompute.")
                state = None
            else:
                # Directly jump to train_and_eval
                eval_result, cm, cm_norm, class_names = self.train_and_eval(
                    X_train_std, state["y_train"], X_val_std, state["y_val"], label_map
                )
                eval_result.update({
                    "timestamp": datetime.now().isoformat(),
                    "n_factors_used": len(common),
                    "factors_used": common,
                    "factors_dropped": dropped_all,
                    "label_map": label_map,
                    "lgbm_params": self.lgbm_params,
                    "cache_hit": str(cache_d),
                })
                files = self.save_report(eval_result, cm, cm_norm, class_names, output_dir)
                eval_result["output_files"] = files
                return eval_result

        # ---- Compute factor matrix ----
        if num_workers > 1:
            logger.info(f"Using parallel mode: num_workers={num_workers}, num_chunks={num_chunks}")
            X_train, used_tr, dropped_tr = self.compute_factor_matrix_fast(
                factors, train_kp, flat_attributes,
                num_chunks=num_chunks, num_workers=num_workers,
                split_name="train", cache_dir=cache_dir,
            )
            X_val, used_vl, dropped_vl = self.compute_factor_matrix_fast(
                factors, val_kp, flat_attributes,
                num_chunks=num_chunks, num_workers=num_workers,
                split_name="val", cache_dir=cache_dir,
            )
        else:
            logger.info("Using single-process mode (--num-workers > 1 enables parallel)")
            X_train, used_tr, dropped_tr = self.compute_factor_matrix(
                factors, train_kp, flat_attributes, split_name="train")
            X_val, used_vl, dropped_vl = self.compute_factor_matrix(
                factors, val_kp, flat_attributes, split_name="val")

        # ---- Align train/val common factor columns ----
        common = [n for n in used_tr if n in set(used_vl)]
        if len(common) != len(used_tr) or len(common) != len(used_vl):
            logger.warning(
                f"train/val successful factors differ, taking intersection of {len(common)};"
                f"train unique: {set(used_tr)-set(common)}, val unique: {set(used_vl)-set(common)}"
            )
            tr_idx = [used_tr.index(n) for n in common]
            vl_idx = [used_vl.index(n) for n in common]
            X_train = X_train[:, tr_idx]
            X_val = X_val[:, vl_idx]

        dropped_all = sorted(set(dropped_tr) | set(dropped_vl))

        X_train_std, X_val_std, scaler, fill = self.standardize(X_train, X_val)

        # ---- Write cache ----
        if use_cache or refresh_cache:
            self._save_cache(
                cache_d,
                X_train_std, X_val_std,
                train_lb, val_lb,
                scaler, fill,
                used_names=common,
                dropped=dropped_all,
                meta_extra={"cache_key": key},
            )

        eval_result, cm, cm_norm, class_names = self.train_and_eval(
            X_train_std, train_lb, X_val_std, val_lb, label_map
        )

        eval_result.update({
            "timestamp": datetime.now().isoformat(),
            "n_factors_used": len(common),
            "factors_used": common,
            "factors_dropped": dropped_all,
            "label_map": label_map,
            "lgbm_params": self.lgbm_params,
        })

        files = self.save_report(eval_result, cm, cm_norm, class_names, output_dir)
        eval_result["output_files"] = files
        return eval_result
