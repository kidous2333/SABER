#!/usr/bin/env python
"""
inference.py -- Inference script based on trained models.

Runs the complete SABER 3-stage pipeline inference on new keypoint data,
outputting behavior/event annotation timelines.

Usage:
  python inference.py --run runs/train/expN \
      --mouse-keypoints path/to/mouse_keypoint_dir \
      --tail-keypoints path/to/tail_keypoint_dir \
      --output results.csv

  # Specify target behavior class (outputs 0/1 binary labels)
  python inference.py --run runs/train/expN \
      --mouse-keypoints path/to/mouse_keypoint_dir \
      --tail-keypoints path/to/tail_keypoint_dir \
      --target-class "positive_sniffs" --output results.csv

  # Use external config file override
  python inference.py --run runs/train/expN \
      --mouse-keypoints path/to/mouse_keypoint_dir \
      --tail-keypoints path/to/tail_keypoint_dir \
      --config-validation config/validation.yaml

Workflow:
  1. Load merged_config.yaml from run directory (factor list, model config, etc.)
  2. Load keypoint data (auto-detect two mice)
  3. Load trained factor-group LGBM + meta-LGBM + temporal model
  4. Run inference for mouse1 (center_id=0) and mouse2 (center_id=1) separately
  5. Output behavior annotation timeline (CSV / XLSX)

Output format (CSV):
  frame, time_sec, label_mouse1, label_mouse2, label_name_mouse1, label_name_mouse2, ...
  0, 0.000, 0, 0, explore_object, positive_sniffs, ...
  1, 0.033, 0, 1, explore_object, positive_sniffs, ...

If --target-class is specified, label column is 0/1 (0=non-target behavior, 1=target behavior occurring).
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
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Reuse project-internal modules
# ---------------------------------------------------------------------------
from mining.discovery import setup_logging, _load_merged_config
from src.synth_validator import SynthValidator, _group_factors_by_seqlength, _train_single_lgbm
from src.temporal_validator import TemporalValidator, _build_temporal_features
from src.factor_engine import FactorEngine

logger = logging.getLogger("inference")


# =============================================================================
# Model loading helper functions
# =============================================================================

def _load_lgbm_model(model_path: Path):
    """Load LightGBM model file (.txt format)."""
    import lightgbm as lgb
    if not model_path.exists():
        raise FileNotFoundError(f"LGBM model file not found: {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    logger.info(f"Loaded LGBM model: {model_path}")
    return model


def _load_temporal_nn_model(ckpt_path: Path, device: str = "cuda"):
    """Load temporal neural network (BiLSTM/Transformer/Mamba) checkpoint."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    n_classes  = ckpt["n_classes"]
    input_dim  = ckpt["input_dim"]
    cfg_saved  = ckpt["cfg"]

    logger.info(f"[inference] checkpoint: model={model_type}, "
                f"input_dim={input_dim}, n_classes={n_classes}")

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
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout, output_dim=n_classes,
            use_layer_norm=use_ln, use_deep_projection=use_dp,
            use_attention_pooling=use_ap, nhead=nhead,
        )
    elif model_type == "transformer":
        from src.temporal_models import TransformerTemporalModel
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
    logger.info(f"[inference] Temporal NN model loaded on {_dev}: {model_type}")
    return model, ckpt


# =============================================================================
# Data loading
# =============================================================================

def load_inference_data(
    mouse_keypoint_dir: str,
    tail_keypoint_dir: str,
    max_instances_num: int = 2,
    fps: int = 30,
    short_gap_max: int = 10,
) -> tuple:
    """
    Load keypoint data for inference (single video, no behavior annotations).

    Parameters:
        mouse_keypoint_dir: Mouse keypoint txt directory (YOLO pose format)
        tail_keypoint_dir:  Tail keypoint txt directory
        max_instances_num:  Maximum number of mice instances (default 2)
        fps:                Frame rate
        short_gap_max:      Max frames for short gap interpolation

    Returns:
        keypoints:      np.ndarray [T, D] float32  per-frame keypoint features
        flat_attributes: list[str]                 feature name list
        merged_data:    list                       fused data (for center_id switching)
        total_frames:   int
    """
    from src.data_loader import (
        MouseBehaviorDataset,
    )

    logger.info(f"[inference] Loading keypoint data...")
    logger.info(f"  Mouse: {mouse_keypoint_dir}")
    logger.info(f"  Tail:  {tail_keypoint_dir}")

    # Build minimal dataset_config
    dataset_config = {
        "Mouse_key_point_file": [mouse_keypoint_dir],
        "Tail_key_point_file":  [tail_keypoint_dir],
        "max_instances_num":    max_instances_num,
    }

    # ── Invalidate stale cache ──────────────────────────────────────
    # MouseBehaviorDataset caches based solely on file PATHS (not content).
    # In E2E mode, label files are freshly generated each run at the same
    # path, so a stale cache from a previous run would be incorrectly
    # reused. Delete the cache file(s) to force a fresh read.
    import hashlib as _hashlib, json as _json, os as _os
    _cache_dir = ".\\dataset_cache"
    _config_content = {
        "Mouse_files": [mouse_keypoint_dir],
        "Tail_files": [tail_keypoint_dir],
        "max_instances": max_instances_num,
        "method_version": "v1.4",
    }
    _config_str = _json.dumps(_config_content, sort_keys=True)
    _cache_hash = _hashlib.md5(_config_str.encode('utf-8')).hexdigest()
    _cache_path = _os.path.join(_cache_dir, f"cached_data_{_cache_hash}.pt")
    _norm_path  = _os.path.join(_cache_dir, f"feature_norm_{_cache_hash}.npz")
    if _os.path.exists(_cache_path):
        _os.remove(_cache_path)
        logger.info(f"[inference] Removed stale cache: {_cache_path}")
    if _os.path.exists(_norm_path):
        _os.remove(_norm_path)
        logger.info(f"[inference] Removed stale cache: {_norm_path}")
    # ─────────────────────────────────────────────────────────────────

    # Use MouseBehaviorDataset to load (is_train=False, no labels)
    mbd = MouseBehaviorDataset(
        dataset_config=dataset_config,
        label_map=None,
        seq_length=1,
        stride=1,
        frame_interval=1,
        transform=None,
        is_train=False,
        purity_threshold=1.0,
        short_gap_max=short_gap_max,
    )

    keypoints_np = mbd.keypoints.cpu().numpy().astype(np.float32)  # [T, D]
    flat_attributes = mbd.feature_indexer.flat_attributes
    merged_data = mbd.Merged_data
    normalizer = mbd.normalizer  # FeatureNormalizer fitted on THIS data
    total_frames = keypoints_np.shape[0]

    logger.info(
        f"[inference] Data loading complete: {total_frames} frames, "
        f"D={keypoints_np.shape[1]}, instances={max_instances_num}"
    )
    return keypoints_np, flat_attributes, merged_data, normalizer


def _find_norm_path(merged_data: list = None):
    """Find feature normalization parameter file in dataset_cache."""
    from pathlib import Path
    cache_dir = Path("dataset_cache")
    if not cache_dir.is_dir():
        # Try GUI cache location
        gui_cache = Path("gui") / "dataset_cache"
        if gui_cache.is_dir():
            cache_dir = gui_cache
        else:
            return None
    norm_files = sorted(cache_dir.glob("feature_norm_*.npz"))
    return str(norm_files[-1]) if norm_files else None


def determine_mouse_order(
    merged_data: list,
    max_instance_num: int = 2,
) -> list:
    """
    Determine the mouse1/mouse2 mapping based on mouse positions in the first frame.

    Rule: Using the first frame, the mouse closer to the top-left (smaller x+y) is mouse1,
    the other is mouse2.

    Parameters:
        merged_data:      MouseTailMerger fused data
        max_instance_num: Maximum number of instances

    Returns:
        center_order:  list[int], length = max_instance_num
                       center_order[logical_id] = original_instance_index
                       e.g. [1, 0] means original index 1 is mouse1 (top-left),
                       original index 0 is mouse2 (bottom-right)
    """
    # Find the first frame where both mice are present
    first_positions = None
    for video_data in merged_data:
        if not video_data:
            continue
        for frame_instances in video_data:
            positions = []
            all_present = True
            for i in range(max_instance_num):
                if i < len(frame_instances) and frame_instances[i] is not None:
                    x, y = frame_instances[i].xy
                    # Handle possible None coordinates
                    if x is None or y is None:
                        all_present = False
                        break
                    positions.append((i, float(x), float(y)))
                else:
                    all_present = False
                    break
            if all_present and len(positions) == max_instance_num:
                first_positions = positions
                break
        if first_positions is not None:
            break

    # If no frame with both mice found, keep original order
    if first_positions is None:
        logger.warning(
            "[inference] Cannot find first frame with both mice present, keeping original instance order"
        )
        return list(range(max_instance_num))

    # Sort by x+y (Manhattan distance to top-left), smaller is more top-left
    first_positions.sort(key=lambda p: p[1] + p[2])

    # Build mapping: logical_id -> original_instance_index
    center_order = [first_positions[logical_id][0] for logical_id in range(len(first_positions))]

    logger.info(
        f"[inference] Mouse order determined (first-frame top-left): "
        f"mouse1 = original instance {center_order[0]} (xy=({first_positions[0][1]:.4f},{first_positions[0][2]:.4f})), "
        f"mouse2 = original instance {center_order[1]} (xy=({first_positions[1][1]:.4f},{first_positions[1][2]:.4f}))"
    )
    return center_order


def build_keypoints_for_center(
    keypoints_full: np.ndarray,
    merged_data: list,
    center_id: int,
    max_instance_num: int = 2,
    num_keypoints: int = 10,  # 7 mouse + 3 tail
    fps: int = 30,
    normalizer = None,
) -> tuple:
    """
    Rebuild keypoint tensor for a specified center_id.

    Reuses logic from MouseBehaviorDataset.build_centered_tensors_concatenated,
    but only generates a [T, D] tensor for a single center_id.

    Parameters:
        keypoints_full: All raw keypoints [T_all, D]
        merged_data:    MouseTailMerger fused data
        center_id:      0 = mouse1 as subject, 1 = mouse2 as subject
        max_instance_num: Maximum number of instances
        num_keypoints:  Number of keypoints
        fps:            Frame rate
        normalizer:     Optional FeatureNormalizer fitted on the SAME data.
                        If None, falls back to _find_norm_path (legacy).

    Returns:
        kp_center:  np.ndarray [T, D]  keypoints with center_id as subject
        flat_attrs: list[str]          feature name list
    """
    import torch
    from src.data_loader import (
        video_to_tensor, VectorizedContext, FeatureIndexer, FEATURE_REGISTRY,
    )

    feature_list = ['skeleton', 'motion', 'tail', 'social']

    all_tensors = []
    feature_indexer = FeatureIndexer()
    indexer_built = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for video_data in merged_data:
        if not video_data:
            continue
        points_t, box_t, mask_t = video_to_tensor(
            video_data, max_instance_num, num_keypoints, device=device
        )
        T = points_t.shape[0]
        if T == 0:
            continue
        ctx = VectorizedContext(points_t, box_t, mask_t, fps, num_keypoints, device)

        features_for_center = []
        for feat_name in feature_list:
            if feat_name not in FEATURE_REGISTRY:
                continue
            calc_func = FEATURE_REGISTRY[feat_name]
            feat_tensor, attr_names = calc_func(ctx, center_id)
            features_for_center.append(feat_tensor)
            if not indexer_built:
                feature_indexer.add_feature(feat_name, feat_tensor.shape[1], attr_names)

        if features_for_center:
            center_tensor = torch.cat(features_for_center, dim=1)
            all_tensors.append(center_tensor)

        if not indexer_built:
            indexer_built = True

    if all_tensors:
        kp_center = torch.cat(all_tensors, dim=0).cpu().numpy().astype(np.float32)
    else:
        kp_center = np.empty((0,), dtype=np.float32)

    # Apply the SAME normalizer that was fitted on this data by
    # MouseBehaviorDataset.  Do NOT use _find_norm_path() — it picks
    # the alphabetically-last feature_norm_*.npz which may belong to a
    # different dataset (training vs inference), causing feature-scale
    # mismatch and degraded predictions.
    from src.data_loader import FeatureNormalizer
    if normalizer is not None:
        kp_center = normalizer.transform(kp_center)
    else:
        # Legacy fallback — only used when no normalizer is provided
        norm_path = _find_norm_path(merged_data)
        if norm_path:
            normalizer = FeatureNormalizer().load(norm_path)
            kp_center = normalizer.transform(kp_center)

    return kp_center, feature_indexer.flat_attributes


# =============================================================================
# Factor matrix computation (multi-resolution)
# =============================================================================

def compute_factor_matrix_for_factors(
    factors: list,
    factor_names: list,
    kp_full: np.ndarray,
    flat_attributes: list,
    engine: FactorEngine,
    purity_mode: str = "nan_boundary",
    labels: np.ndarray = None,
    split_name: str = "inference",
    num_workers: int = 0,
    num_chunks: int = 256,
    cache_dir: str = "dataset_cache",
) -> tuple:
    """
    Compute multi-resolution factor matrix for the specified factor list.

    Each factor builds centered windows at its seq_length, then computes factor values.
    Uses multiprocessing when num_workers > 1 (for row-mode factors),
    uses original serial path when num_workers <= 1.

    Returns:
        X:         np.ndarray [T, K]  factor matrix
        ok_names:  list[str]          successfully computed factor names (in order matching factor_names input)
        dropped:   list[str]          factor names that failed computation
    """
    from collections import defaultdict
    import gc

    T = kp_full.shape[0]
    factor_name_set = set(factor_names)

    # Group by seq_length
    by_sl = defaultdict(list)
    for fac in factors:
        if fac["name"] in factor_name_set:
            by_sl[fac.get("seq_length", 1)].append(fac)

    # Build window matrices for each seq_length
    windows_cache = {}
    for sl in by_sl:
        windows_cache[sl] = SynthValidator._build_centered_windows(kp_full, sl)

    # Compute purity masks (if labels available)
    purity_masks = {}
    if labels is not None and purity_mode == "nan_boundary":
        for sl in by_sl:
            purity_masks[sl] = SynthValidator._compute_purity_mask(labels, sl)

    all_columns = {}
    dropped = []

    # Diagnostic log: check if each factor's referenced feature names exist in flat_attributes
    import re
    attr_set = set(flat_attributes)
    factors_with_missing_refs = []
    for fac in factors:
        if fac["name"] not in factor_name_set:
            continue
        code = fac.get("code", "")
        refs = set(re.findall(r"idx\[?['\"]([^'\"]+)['\"]\]?", code))
        refs |= set(re.findall(r"idx\.get\(['\"]([^'\"]+)['\"]", code))
        missing = refs - attr_set
        if missing:
            factors_with_missing_refs.append((fac["name"], sorted(missing)[:5]))
    if factors_with_missing_refs:
        logger.warning(
            f"[DIAG][{split_name}] {len(factors_with_missing_refs)} factors reference non-existent feature names (will return NaN):"
        )
        for fname, miss in factors_with_missing_refs[:10]:
            logger.warning(f"  [DIAG] '{fname}': missing features {miss}")
        if len(factors_with_missing_refs) > 10:
            logger.warning(f"  [DIAG] ... and {len(factors_with_missing_refs) - 10} more")

    # -- Unified path (all batch vectorized) --
    max_error_ratio = getattr(engine, "max_error_ratio", 0.1)
    sorted_groups = sorted(by_sl.items())
    total_groups = len(sorted_groups)

    for gi, (sl, sl_factors) in enumerate(sorted_groups, 1):
        windows = windows_cache[sl]
        apply_purity = purity_mode == "nan_boundary" and sl > 1
        purity_mask = purity_masks.get(sl)

        logger.info(
            f"[{split_name}] Group {gi}/{total_groups}: seq_length={sl}, "
            f"{len(sl_factors)} factors, window shape={windows.shape}"
        )

        for fac in sl_factors:
            name = fac["name"]
            try:
                vals = engine.compute_factor_batch(fac, windows, flat_attributes)
            except Exception as e:
                logger.warning(f"[{split_name}] Factor '{name}' exception: {e}")
                dropped.append(name)
                continue
            if vals is None or len(vals) != T or not np.isfinite(vals).any():
                dropped.append(name)
                continue
            vals = vals.astype(np.float32)
            if apply_purity and purity_mask is not None:
                vals[~purity_mask] = np.nan
            all_columns[name] = vals

        n_ok = sum(1 for f in sl_factors if f["name"] in all_columns)
        n_drop = sum(1 for f in sl_factors if f["name"] in dropped)
        logger.info(
            f"[{split_name}] Group {gi}/{total_groups} complete: "
            f"seq_length={sl}, ok {n_ok}, dropped {n_drop}"
        )
        del windows; gc.collect()

    # Assemble matrix in factor_names order (missing factors filled with NaN, handled by subsequent standardization)
    ok_names = [n for n in factor_names if n in all_columns]
    for n in factor_names:
        if n not in all_columns and n not in dropped:
            dropped.append(n)

    if not ok_names:
        raise RuntimeError(f"[{split_name}] All factors failed computation.")

    # Build full matrix: missing columns filled with NaN
    X_full = np.full((T, len(factor_names)), np.nan, dtype=np.float32)
    for col_idx, name in enumerate(factor_names):
        if name in all_columns:
            X_full[:, col_idx] = all_columns[name]

    logger.info(
        f"[{split_name}] Factor matrix shape={X_full.shape}, "
        f"used {len(ok_names)}, missing {len(factor_names) - len(ok_names)} (filled NaN), "
        f"dropped {len(dropped)}"
    )

    # Diagnostic log: factor matrix statistics
    nan_ratio = np.isnan(X_full).mean()
    finite_mask = np.isfinite(X_full)
    if finite_mask.any():
        logger.info(
            f"[DIAG][{split_name}] Factor matrix stats: NaN ratio={nan_ratio:.4f}, "
            f"valid min={X_full[finite_mask].min():.4f}, "
            f"max={X_full[finite_mask].max():.4f}, "
            f"mean={X_full[finite_mask].mean():.4f}, "
            f"std={X_full[finite_mask].std():.4f}"
        )
        # Check how many columns are all NaN (factor completely failed)
        all_nan_cols = np.where(np.all(np.isnan(X_full), axis=0))[0]
        if len(all_nan_cols) > 0:
            col_names = [factor_names[i] for i in all_nan_cols[:10]]
            logger.warning(
                f"[DIAG][{split_name}] {len(all_nan_cols)} columns are all NaN (factor fully failed), "
                f"first 10: {col_names}"
            )
    else:
        logger.warning(f"[DIAG][{split_name}] Factor matrix is entirely NaN!")

    return X_full, factor_names, dropped


def apply_scaler(
    X: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    fill: np.ndarray,
    clip_lo: float = -8.0,
    clip_hi: float = 8.0,
) -> np.ndarray:
    """
    Apply standardization to factor matrix using trained normalization parameters.

    Parameters:
        X:            [T, K] raw factor matrix
        scaler_mean:  [K] training set mean
        scaler_scale: [K] training set standard deviation
        fill:         [K] NaN fill values
        clip_lo/hi:   clip bounds

    Returns:
        X_std:  [T, K] standardized factor matrix
    """
    X = X.astype(np.float32, copy=True)
    X[~np.isfinite(X)] = np.nan

    # NaN fill
    nan_mask = np.isnan(X)
    nan_ratio_before = nan_mask.mean()
    X[nan_mask] = np.broadcast_to(fill.astype(np.float32), X.shape)[nan_mask]

    # Standardization
    zero_scale = (scaler_scale == 0)
    safe_scale = np.where(zero_scale, 1.0, scaler_scale).astype(np.float32)
    X_std = (X - scaler_mean.astype(np.float32)) / safe_scale

    # Clip
    X_std = np.clip(X_std, clip_lo, clip_hi)
    X_std[~np.isfinite(X_std)] = 0.0

    # Diagnostic log: scaler application effects
    import logging
    _diag_logger = logging.getLogger("inference")
    _diag_logger.info(
        f"[DIAG][scaler] NaN fill ratio before={nan_ratio_before:.4f}, "
        f"after fill min={X_std.min():.4f}, max={X_std.max():.4f}, "
        f"mean={X_std.mean():.4f}, std={X_std.std():.4f}, "
        f"clipped_at_lo={(X_std <= clip_lo + 0.01).mean():.4f}, "
        f"clipped_at_hi={(X_std >= clip_hi - 0.01).mean():.4f}, "
        f"zero_scale_cols={(zero_scale).sum()}"
    )

    return X_std.astype(np.float32)


# =============================================================================
# Main inference pipeline
# =============================================================================

class InferencePipeline:
    """
    SABER 3-stage inference pipeline.

    Loads trained factor-group LGBM + meta-LGBM + temporal model,
    runs full inference on input data.
    """

    def __init__(self, run_dir: Path, cfg: dict):
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.weights_dir = self.run_dir / "weights"
        self.factor_engine = FactorEngine(cfg)

        if not self.weights_dir.exists():
            raise FileNotFoundError(
                f"Weights directory not found: {self.weights_dir}. "
                f"Please complete training with train_behavior.py first."
            )

        # Load config
        self._load_pipeline_config()

        # Load factors
        self._load_factors()

        # Load models
        self._load_models()

    def _load_pipeline_config(self):
        """Load pipeline config from weights directory."""
        # Load pipeline_meta.json
        meta_path = self.weights_dir / "pipeline_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.pipeline_meta = json.load(f)
            logger.info(f"[inference] Pipeline config loaded: {meta_path}")
        else:
            logger.warning("[inference] pipeline_meta.json not found, will infer grouping from factors")
            self.pipeline_meta = {}

        self.group_names = self.pipeline_meta.get("group_names", ["short", "medium", "long"])
        self.classes_sorted = [int(c) for c in self.pipeline_meta.get("classes_sorted", [])]
        # JSON keys can only be strings, convert back to int ({"0": "explore_object"} -> {0: "explore_object"})
        self.id_to_name = {int(k): v for k, v in self.pipeline_meta.get("id_to_name", {}).items()}
        self.n_classes = self.pipeline_meta.get("n_classes", len(self.classes_sorted))
        self.residual_stage1 = self.pipeline_meta.get(
            "residual_stage1",
            self.cfg.get("synth_validation", {}).get("residual_stage1", False),
        )
        # Also read residual_stage2 from config (for seq_use_extra_features extra feature determination)
        self.residual_stage2 = self.pipeline_meta.get(
            "residual_stage2",
            self.cfg.get("synth_validation", {}).get("residual_stage2", False),
        )
        if self.residual_stage1:
            logger.info("[inference] Stage1 residual enabled (factor matrix -> group LGBM output)")
        if self.residual_stage2:
            logger.info("[inference] Stage2 residual enabled (meta_X -> temporal model extra)")

        if not self.id_to_name:
            label_map = self.cfg.get("label_map", {})
            if label_map:
                self.id_to_name = {int(v): k for k, v in label_map.items()}
                self.classes_sorted = sorted(self.id_to_name.keys())
                self.n_classes = len(self.classes_sorted)
            else:
                # Fallback: load label_map from the JSON pointed to by dataset_config_file
                label_map = self._load_label_map_from_dataset_config()
                if label_map:
                    self.id_to_name = {int(v): k for k, v in label_map.items()}
                    self.classes_sorted = sorted(self.id_to_name.keys())
                    self.n_classes = len(self.classes_sorted)
                else:
                    raise ValueError(
                        "Cannot determine class mapping. Please ensure pipeline_meta.json, "
                        "the config contains label_map, or dataset_config_file is accessible."
                    )

        self.class_names = [self.id_to_name[c] for c in self.classes_sorted]
        logger.info(f"[inference] n_classes={self.n_classes}, classes: {self.class_names}")

        # Load post-processing calibration parameters (PostCalibrator)
        self.calib_params = None
        calib_path = self.run_dir / "optimized_model_report.json"
        if calib_path.exists():
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    calib_report = json.load(f)
                best = calib_report.get("best_params", {})
                calib = best.get("calibrator", {})
                if calib:
                    self.calib_params = {
                        "temperature": calib.get("temperature", 1.0),
                        "mix_alpha": np.asarray(calib.get("mix_alpha", 1.0), dtype=np.float64),
                        "biases": np.asarray(calib.get("biases", None), dtype=np.float64) if calib.get("biases") is not None else None,
                    }
                    logger.info(
                        f"[inference] Post-processing calibration params loaded: {calib_path}\n"
                        f"  temperature={self.calib_params['temperature']:.4f}\n"
                        f"  mix_alpha={self.calib_params['mix_alpha']}\n"
                        f"  biases={self.calib_params['biases']}"
                    )
                else:
                    logger.info(f"[inference] Calibration params empty, skipping: {calib_path}")
            except Exception as e:
                logger.warning(f"[inference] Failed to load calibration params: {e}")
        else:
            logger.info(f"[inference] Post-processing calibration file not found: {calib_path} (skipping)")
        # Sanity check: behavior classification typically has <= 20 classes; warn if abnormally large
        if self.n_classes > 20:
            logger.warning(
                f"[inference] n_classes={self.n_classes} is abnormally large (expected <= 20 behavior classes), "
                f"may cause subsequent dimension errors. Please check the n_classes field in pipeline_meta.json "
                f"or the label_map in config."
            )

    def _load_label_map_from_dataset_config(self) -> dict:
        """Load label_map from the JSON pointed to by dataset_config_file (fallback)."""
        import json as _json

        dataset_config_file = self.cfg.get("dataset_config_file", "")
        if not dataset_config_file:
            return {}

        # Try multiple paths
        candidates = [
            Path(dataset_config_file),
            Path("config") / dataset_config_file,
            self.run_dir / ".." / ".." / dataset_config_file,
            self.run_dir / dataset_config_file,
        ]
        for cand in candidates:
            try:
                cand_resolved = cand.resolve()
            except OSError:
                continue
            if not cand_resolved.exists():
                continue
            try:
                with open(cand_resolved, "r", encoding="utf-8") as f:
                    ds_json = _json.load(f)
            except Exception:
                continue

            # label_map could be a file path or an inline dict
            label_map_raw = ds_json.get("label_map", None)
            if label_map_raw is None:
                # Also try the top level directly as label_map (old format)
                if all(isinstance(v, (int, float)) for v in ds_json.values()):
                    return ds_json
                return {}

            if isinstance(label_map_raw, dict):
                return label_map_raw
            elif isinstance(label_map_raw, str):
                # label_map is a file path
                for lm_cand in [
                    Path(label_map_raw),
                    Path("config") / label_map_raw,
                    cand_resolved.parent / label_map_raw,
                ]:
                    try:
                        lm_resolved = lm_cand.resolve()
                    except OSError:
                        continue
                    if lm_resolved.exists():
                        try:
                            with open(lm_resolved, "r", encoding="utf-8") as f:
                                return _json.load(f)
                        except Exception:
                            continue
            return {}

        logger.warning("[inference] Could not load label_map from dataset_config_file")
        return {}

    def _load_factors(self):
        """Load factor definitions (prefer factors.json in exp folder)."""
        # Priority 1: factors.json in the experiment folder
        factors_file = self.run_dir / "factors.json"
        if factors_file.exists():
            logger.info(f"[inference] Factor file (from exp folder): {factors_file}")
        else:
            # Priority 2: config path
            run_cfg = self.cfg.get("run", {})
            factors_path = run_cfg.get("factors", "memory/valid_factors.json")
            factors_file = Path(factors_path)
            if not factors_file.exists():
                # Also try relative path from run directory
                factors_file = self.run_dir.parent.parent / factors_path
            if not factors_file.exists():
                # Try common locations
                for cand in [Path("memory_before/valid_factors.json"),
                             Path("memory_before/evolved_factors.json")]:
                    if cand.exists():
                        factors_file = cand
                        break
            if not factors_file.exists():
                raise FileNotFoundError(
                    f"Factor file not found: {self.run_dir / 'factors.json'} or {factors_path}"
                )
            logger.info(f"[inference] Factor file (fallback): {factors_file}")

        self.all_factors = SynthValidator.load_valid_factors(str(factors_file))
        logger.info(f"[inference] Loaded {len(self.all_factors)} factors")

        # Group by seq_length
        self.factor_groups = _group_factors_by_seqlength(self.all_factors)
        logger.info(
            f"[inference] Factor grouping: "
            f"{ {g: len(v) for g, v in self.factor_groups.items()} }"
        )

        # Index by name
        self.factor_by_name = {f["name"]: f for f in self.all_factors}

    def _load_models(self):
        """Load all model weights."""
        import lightgbm as lgb

        # ---- Load group LGBM models + scaler params + common factor lists ----
        self.group_models = {}
        self.group_scalers = {}
        self.group_common_factors = {}

        for group_name in self.group_names:
            model_path = self.weights_dir / f"group_{group_name}_lgbm.txt"
            if not model_path.exists():
                logger.warning(
                    f"[inference] Group '{group_name}' LGBM model not found: {model_path}, skipping"
                )
                continue

            self.group_models[group_name] = _load_lgbm_model(model_path)

            factors_path = self.weights_dir / f"common_factors_{group_name}.json"
            if not factors_path.exists():
                _tmp_path = self.weights_dir / f"common_factors_{group_name}.json.tmp"
                if _tmp_path.exists():
                    _tmp_path.rename(factors_path)
                    logger.info(f"Recovered common_factors from .tmp: {group_name}")

            # Load standardization params (before common_factors, for dimension check)
            mean_path = self.weights_dir / f"scaler_mean_{group_name}.npy"
            scale_path = self.weights_dir / f"scaler_scale_{group_name}.npy"
            fill_path = self.weights_dir / f"fill_{group_name}.npy"
            scaler_K = None
            if all(p.exists() for p in [mean_path, scale_path, fill_path]):
                self.group_scalers[group_name] = {
                    "mean": np.load(mean_path).astype(np.float32),
                    "scale": np.load(scale_path).astype(np.float32),
                    "fill": np.load(fill_path).astype(np.float32),
                }
                scaler_K = len(self.group_scalers[group_name]["mean"])
                scaler_info_loaded = True
                logger.info(
                    f"[inference] Group '{group_name}' standardization params loaded "
                    f"(K={scaler_K})"
                )
            else:
                logger.warning(f"[inference] Group '{group_name}' standardization params incomplete")

            # Load common factor list
            cf_loaded = False
            if factors_path.exists():
                try:
                    with open(factors_path, "r", encoding="utf-8") as f:
                        cf_info = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[inference] Group '{group_name}' common_factors file corrupted or empty: "
                        f"{factors_path}, deleted; please re-run train_behavior.py to generate weights."
                    )
                    factors_path.unlink(missing_ok=True)
                else:
                    self.group_common_factors[group_name] = cf_info.get("common", [])
                    if not self.classes_sorted:
                        self.classes_sorted = cf_info.get("classes_sorted", [])
                        self.id_to_name = cf_info.get("id_to_name", {})
                        self.n_classes = cf_info.get("n_classes", len(self.classes_sorted))
                    cf_loaded = True
                    n_common = len(self.group_common_factors[group_name])
                    logger.info(
                        f"[inference] Group '{group_name}' common factors: {n_common}"
                    )
                    # Verify: common_factors count must match scaler dimension
                    if scaler_K is not None and n_common != scaler_K:
                        raise ValueError(
                            f"Group '{group_name}' factor count mismatch: "
                            f"common_factors={n_common}, scaler_K={scaler_K}. "
                            f"Please delete {self.weights_dir} and re-run train_behavior.py to generate weights."
                        )
            if not cf_loaded:
                if scaler_K is not None:
                    # Fallback: use all factors from this group, sorted by best_auc
                    all_group_factors = sorted(
                        self.factor_groups.get(group_name, []),
                        key=lambda f: f.get("best_auc", 0) or 0, reverse=True)
                    fallback_names = [f["name"] for f in all_group_factors[:scaler_K]]
                    if len(fallback_names) < scaler_K:
                        fallback_names += [f["name"] for f in all_group_factors[scaler_K:]]
                    self.group_common_factors[group_name] = fallback_names[:scaler_K]
                    logger.warning(
                        f"[inference] Group '{group_name}' common_factors missing, "
                        f"using top {len(fallback_names)} factors by AUC as fallback")
                else:
                    # Neither common_factors nor scaler, fall back to all factors in this group
                    logger.warning(
                        f"[inference] Group '{group_name}' missing common_factors and scaler files, "
                        f"will use all factors in this group (may be incompatible with trained model)"
                    )
                    self.group_common_factors[group_name] = [
                        f["name"] for f in self.factor_groups.get(group_name, [])
                    ]

        # ---- Load meta-LGBM model ----
        meta_model_path = self.weights_dir / "meta_lgbm.txt"
        if meta_model_path.exists() and len(self.group_models) > 1:
            self.meta_model = _load_lgbm_model(meta_model_path)
            logger.info("[inference] meta-LGBM model loaded")
        else:
            self.meta_model = None
            if len(self.group_models) <= 1:
                logger.info("[inference] Single group only, skipping meta-LGBM")

        # ---- Load temporal model ----
        temporal_cfg = self.cfg.get("temporal_validation", {})
        temporal_model_type = temporal_cfg.get("temporal_model", "bilstm")

        if temporal_model_type == "lgbm":
            temporal_lgbm_path = self.weights_dir / "temporal_lgbm_model.txt"
            if temporal_lgbm_path.exists():
                self.temporal_model = _load_lgbm_model(temporal_lgbm_path)
                self.temporal_nn_model = None
                self.temporal_model_type = "lgbm"
                logger.info("[inference] Temporal LGBM model loaded")
            else:
                logger.warning(
                    f"[inference] Temporal LGBM model not found: {temporal_lgbm_path}"
                )
                self.temporal_model = None
                self.temporal_nn_model = None
                self.temporal_model_type = "lgbm"
        else:
            ckpt_path = self.weights_dir / f"{temporal_model_type}_best.pt"
            if ckpt_path.exists():
                self.temporal_nn_model, self.temporal_ckpt = _load_temporal_nn_model(
                    ckpt_path,
                    device=temporal_cfg.get("seq_model", {}).get("device", "cuda"),
                )
                self.temporal_model = None
                self.temporal_model_type = temporal_model_type
                # Get seq_model_cfg from checkpoint
                self.seq_model_cfg = dict(self.temporal_ckpt.get("cfg", {}))
                self.seq_model_cfg.update(
                    temporal_cfg.get("seq_model", {})
                )
                self.seq_model_cfg.setdefault("chunk_size", 512)
                self.seq_model_cfg.setdefault("stride_val", 256)
                self.seq_model_cfg.setdefault("batch_size", 32)
                self.seq_use_extra = temporal_cfg.get("seq_use_extra_features", False)
                logger.info(f"[inference] Temporal NN model loaded: {temporal_model_type}")
            else:
                raise FileNotFoundError(
                    f"Temporal model checkpoint not found: {ckpt_path}"
                )

        # ---- Load temporal LGBM parameters ----
        if self.temporal_model_type == "lgbm" and self.temporal_model is not None:
            self.temporal_window_size = temporal_cfg.get("window_size", 31)
            self.multi_scale_windows = temporal_cfg.get("multi_scale_windows", [])
            if isinstance(self.multi_scale_windows, (int, float)):
                self.multi_scale_windows = [int(self.multi_scale_windows)]
            self.multi_scale_windows = [
                int(w) for w in self.multi_scale_windows if int(w) > 0
            ]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        kp_full: np.ndarray,
        flat_attributes: list,
        center_id: int = 0,
        num_workers: int = 0,
        num_chunks: int = 256,
        skip_temporal: bool = False,
    ) -> np.ndarray:
        """
        Run full inference pipeline for a single center_id.

        Parameters:
            kp_full:         [T, D] raw keypoints
            flat_attributes: feature name list
            center_id:       subject mouse index (0 or 1)

        Returns:
            proba:  np.ndarray [T, n_classes]  per-frame probability distribution
        """
        T = kp_full.shape[0]
        group_proba_list = []

        for group_name in self.group_names:
            if group_name not in self.group_models:
                logger.warning(
                    f"[inference] Group '{group_name}' has no model, skipping"
                )
                continue

            common_factors = self.group_common_factors.get(group_name, [])
            if not common_factors:
                logger.warning(
                    f"[inference] Group '{group_name}' has no common factors, skipping"
                )
                continue

            # Get full definitions for these factors
            group_factors = [
                self.factor_by_name[name] for name in common_factors
                if name in self.factor_by_name
            ]
            if not group_factors:
                logger.warning(
                    f"[inference] Group '{group_name}' factor definitions missing, skipping"
                )
                continue

            logger.info(
                f"[inference] Group '{group_name}': computing {len(group_factors)} factors..."
            )
            t0 = _time.time()

            # Compute factor matrix
            X, ok_names, dropped = compute_factor_matrix_for_factors(
                factors=group_factors,
                factor_names=common_factors,
                kp_full=kp_full,
                flat_attributes=flat_attributes,
                engine=self.factor_engine,
                purity_mode="none",  # No purity filtering during inference
                split_name=f"inference_{group_name}",
                num_workers=num_workers,
                num_chunks=num_chunks,
            )

            # Apply scaling
            scaler_info = self.group_scalers.get(group_name, {})
            if scaler_info:
                X_std = apply_scaler(
                    X,
                    scaler_info["mean"],
                    scaler_info["scale"],
                    scaler_info["fill"],
                )
            else:
                logger.warning(
                    f"[inference] Group '{group_name}' has no standardization params, using raw values"
                )
                X_std = X.astype(np.float32)

            del X

            # Run group LGBM
            model = self.group_models[group_name]
            proba_raw = model.predict(X_std).astype(np.float32)  # [T, n_classes]

            # Align to full class order
            n_model_classes = proba_raw.shape[1]
            full_proba = np.zeros((T, self.n_classes), dtype=np.float32)
            for c in range(min(n_model_classes, self.n_classes)):
                full_proba[:, c] = proba_raw[:, c]

            # Diagnostic log: group LGBM output distribution
            _proba_mean = full_proba.mean(axis=0)
            _top_class = np.argmax(_proba_mean)
            logger.info(
                f"[DIAG][{group_name}] Group LGBM proba mean: "
                f"{dict(zip(self.class_names, _proba_mean.round(4)))}, "
                f"dominant class='{self.class_names[_top_class]}'({_proba_mean[_top_class]:.4f}), "
                f"argmax distribution: {dict(zip(self.class_names, np.bincount(full_proba.argmax(axis=1), minlength=self.n_classes)))}"
            )

            # ---- Residual connection Stage 1: concatenate standardized factor matrix to group LGBM output ----
            if self.residual_stage1:
                group_output = np.hstack([X_std, full_proba])  # [T, K_group + C]
                logger.info(
                    f"[inference] Group '{group_name}' Stage1 residual: "
                    f"X_std={X_std.shape} + proba={full_proba.shape} -> {group_output.shape}"
                )
            else:
                group_output = full_proba

            del X_std

            logger.info(
                f"[inference] Group '{group_name}' inference complete "
                f"({_time.time() - t0:.1f}s, ok={len(ok_names)}, dropped={len(dropped)})"
            )
            group_proba_list.append(group_output)

        if not group_proba_list:
            raise RuntimeError("[inference] All groups failed inference.")

        # ---- Meta LGBM ----
        # Save meta_X (for seq_use_extra_features when passing to temporal model)
        meta_X = np.hstack(group_proba_list).astype(np.float32)
        logger.info(f"[inference] Meta features: {meta_X.shape}")

        if self.meta_model is not None and len(group_proba_list) > 1:
            meta_proba = self.meta_model.predict(meta_X).astype(np.float32)
            # Align classes
            n_meta = meta_proba.shape[1]
            full_meta = np.zeros((T, self.n_classes), dtype=np.float32)
            for c in range(min(n_meta, self.n_classes)):
                full_meta[:, c] = meta_proba[:, c]
            proba = full_meta
        else:
            if self.meta_model is None and len(group_proba_list) > 1:
                logger.warning(
                    "[inference] meta_lgbm.txt missing, falling back to single group output. "
                    "Please run train_behavior.py to generate complete weights."
                )
            proba_raw = group_proba_list[0]
            # If residual_stage1 is enabled, group_output contains [X_std, proba],
            # need to take only the last n_classes columns as pure probabilities
            if self.residual_stage1:
                proba_raw = proba_raw[:, -self.n_classes:]
            proba = proba_raw

        del group_proba_list

        # Save meta-LGBM output (for PostCalibrator stage mixing)
        proba_meta = proba.copy()
        logger.info(
            f"[inference] meta-LGBM proba_meta saved, shape={proba_meta.shape}"
        )

        # Diagnostic log: meta-LGBM output distribution (before temporal model input)
        _pmean = proba.mean(axis=0)
        _top = np.argmax(_pmean)
        logger.info(
            f"[DIAG][pre-temporal] proba mean: "
            f"{dict(zip(self.class_names, _pmean.round(4)))}, "
            f"dominant class='{self.class_names[_top]}'({_pmean[_top]:.4f}), "
            f"argmax distribution: {dict(zip(self.class_names, np.bincount(proba.argmax(axis=1), minlength=self.n_classes)))}"
        )

        # ---- Temporal model ----
        if skip_temporal:
            logger.info("[inference] --skip-temporal: skipping temporal model, using meta-LGBM output")
        elif self.temporal_model_type == "lgbm" and self.temporal_model is not None:
            from src.temporal_validator import _build_temporal_features_multiscale

            if self.multi_scale_windows:
                temporal_X = _build_temporal_features_multiscale(
                    proba, self.multi_scale_windows
                )
            else:
                temporal_X = _build_temporal_features(
                    proba, self.temporal_window_size
                )

            logger.info(f"[inference] Temporal features: {temporal_X.shape}")
            temporal_proba = self.temporal_model.predict(temporal_X).astype(np.float32)
            n_temporal = temporal_proba.shape[1]
            full_temporal = np.zeros((T, self.n_classes), dtype=np.float32)
            for c in range(min(n_temporal, self.n_classes)):
                full_temporal[:, c] = temporal_proba[:, c]
            proba = full_temporal

        elif self.temporal_nn_model is not None:
            from src.temporal_models import predict_sequence_model

            # If training used seq_use_extra_features, pass meta_X as extra features
            extra_for_temporal = None
            if self.seq_use_extra:
                extra_for_temporal = meta_X
                logger.info(
                    f"[inference] Temporal NN inference (with extra features): "
                    f"proba={proba.shape}, extra={extra_for_temporal.shape}"
                )
            else:
                logger.info(
                    f"[inference] Temporal NN inference: {self.temporal_model_type}, "
                    f"proba shape={proba.shape}"
                )

            # Fix proba column count: auto-correct using checkpoint's stored n_classes
            # When residual_stage1=True and meta_model is missing, proba may contain [X_std, proba]
            # resulting in K+C columns instead of n_classes; correct using training-time n_classes from checkpoint
            ckpt_n_classes = self.temporal_ckpt.get("n_classes", None)
            if ckpt_n_classes is not None and proba.shape[1] != ckpt_n_classes:
                logger.warning(
                    f"[inference] proba column count {proba.shape[1]} does not match checkpoint n_classes={ckpt_n_classes}, "
                    f"auto-correcting to last {ckpt_n_classes} columns "
                    f"(possible residual_stage1 not properly trimmed, or pipeline_meta.n_classes incorrect)"
                )
                proba = proba[:, -ckpt_n_classes:]

            # Dimension check: input dimension must match model expectation
            model_input_dim = proba.shape[1] + (extra_for_temporal.shape[1] if extra_for_temporal is not None else 0)
            expected_input_dim = self.temporal_ckpt.get("input_dim", None)
            if expected_input_dim is not None and model_input_dim != expected_input_dim:
                n_classes = proba.shape[1]
                extra_dim = extra_for_temporal.shape[1] if extra_for_temporal is not None else 0
                expected_nc = self.temporal_ckpt.get("n_classes", expected_input_dim - extra_dim if extra_for_temporal is not None else expected_input_dim)
                raise ValueError(
                    f"Temporal model input dimension mismatch:\n"
                    f"  Model expects input_dim = {expected_input_dim}"
                    f"  (n_classes={expected_nc}, extra={expected_input_dim - expected_nc})\n"
                    f"  Actual input_dim = {model_input_dim}"
                    f"  (n_classes={n_classes}, extra={extra_dim})\n"
                    f"  n_classes difference = {n_classes - expected_nc} columns\n"
                    f"Possible causes:\n"
                    f"  1. BiLSTM not updated after train_behavior.py re-run -> check runs/.../weights/bilstm_best.pt timestamp\n"
                    f"  2. Factor set inconsistent with training -> compare common_factors counts with scaler K values\n"
                    f"  3. Cache interference -> delete pipeline_stage_cache/ and re-run train_behavior.py\n"
                    f"  4. n_classes field in pipeline_meta.json incorrect -> check if it matches actual behavior class count"
                )

            proba = predict_sequence_model(
                model=self.temporal_nn_model,
                proba=proba,
                cfg=self.seq_model_cfg,
                extra=extra_for_temporal,
            )
            logger.info(f"[inference] Temporal NN inference complete, shape={proba.shape}")

        # ---- Post-processing calibration (PostCalibrator): only applied when calibration params are loaded ----
        if self.calib_params is not None:
            logger.info(
                f"[inference] Applying post-processing calibration: T={self.calib_params['temperature']:.4f}, "
                f"mix_alpha={self.calib_params['mix_alpha']}, "
                f"biases={self.calib_params['biases']}"
            )
            try:
                proba_cal = proba.astype(np.float64).copy()

                # 1. Stage mixing: alpha[c] * BiLSTM[c] + (1-alpha[c]) * meta-LGBM[c]
                alpha = self.calib_params["mix_alpha"]
                if proba_meta is not None:
                    if alpha.ndim == 0:
                        if float(alpha) < 1.0 - 1e-8:
                            proba_cal = float(alpha) * proba_cal + (1.0 - float(alpha)) * proba_meta.astype(np.float64)
                    else:
                        meta = proba_meta.astype(np.float64)
                        for c in range(min(len(alpha), proba_cal.shape[1])):
                            a = float(alpha[c])
                            if a < 1.0 - 1e-8:
                                proba_cal[:, c] = a * proba_cal[:, c] + (1.0 - a) * meta[:, c]

                # 2. Temperature scaling
                T = self.calib_params["temperature"]
                if abs(T - 1.0) > 1e-6:
                    log_p = np.log(np.clip(proba_cal, 1e-300, 1.0))
                    proba_cal = np.exp(log_p / T)
                    proba_cal /= proba_cal.sum(axis=1, keepdims=True)

                # 3. Per-class bias
                biases = self.calib_params["biases"]
                if biases is not None:
                    log_p = np.log(np.clip(proba_cal, 1e-300, 1.0))
                    log_p += biases.astype(np.float64)
                    proba_cal = np.exp(log_p)
                    proba_cal /= proba_cal.sum(axis=1, keepdims=True)

                proba = proba_cal.astype(np.float32)

                _pmean = proba.mean(axis=0)
                _top = np.argmax(_pmean)
                logger.info(
                    f"[DIAG][post-calib] Post-calibration proba mean: "
                    f"{dict(zip(self.class_names, _pmean.round(4)))}, "
                    f"dominant class='{self.class_names[_top]}'({_pmean[_top]:.4f}), "
                    f"argmax distribution: {dict(zip(self.class_names, np.bincount(proba.argmax(axis=1), minlength=self.n_classes)))}"
                )
            except Exception as _e:
                logger.warning(f"[inference] Post-processing calibration failed, using raw BiLSTM output: {_e}")

        # Normalize probabilities
        row_sum = proba.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        proba = proba / row_sum

        # Diagnostic log: final output distribution
        _pmean = proba.mean(axis=0)
        _top = np.argmax(_pmean)
        logger.info(
            f"[DIAG][final] Final proba mean: "
            f"{dict(zip(self.class_names, _pmean.round(4)))}, "
            f"dominant class='{self.class_names[_top]}'({_pmean[_top]:.4f}), "
            f"argmax distribution: {dict(zip(self.class_names, np.bincount(proba.argmax(axis=1), minlength=self.n_classes)))}"
        )

        return proba.astype(np.float32)

    def decode_to_labels(
        self,
        proba: np.ndarray,
        target_class: str = None,
    ) -> np.ndarray:
        """
        Decode probability matrix to discrete labels.

        Parameters:
            proba:        [T, n_classes] probability distribution
            target_class: target behavior name (if set, outputs 0/1 binary labels)

        Returns:
            labels:  [T] int  label sequence
        """
        if target_class is not None:
            # Binary output: target behavior=1, others=0
            if target_class in self.id_to_name.values():
                target_id = None
                for cid, cname in self.id_to_name.items():
                    if cname == target_class:
                        target_id = int(cid)
                        break
                if target_id is not None:
                    labels = (proba.argmax(axis=1) == target_id).astype(int)
                else:
                    raise ValueError(f"Target class '{target_class}' not in class mapping")
            else:
                raise ValueError(
                    f"Target class '{target_class}' not in known classes: "
                    f"{list(self.id_to_name.values())}"
                )
        else:
            # Multi-class output
            labels = np.array(
                [self.classes_sorted[i] for i in proba.argmax(axis=1)]
            )

        return labels


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SABER behavior factor inference -- outputs behavior annotation timeline"
    )
    parser.add_argument(
        "--run", type=str, required=True,
        help="Training experiment directory, e.g. runs/train/exp4",
    )
    parser.add_argument(
        "--mouse-keypoints", type=str, required=True,
        help="Mouse keypoint txt directory (YOLO pose format)",
    )
    parser.add_argument(
        "--tail-keypoints", type=str, required=True,
        help="Tail keypoint txt directory",
    )
    parser.add_argument(
        "--output", type=str, default="inference_results.csv",
        help="Output file path (default inference_results.csv)",
    )
    parser.add_argument(
        "--target-class", type=str, default=None,
        help="Target behavior class name (if set, outputs 0/1 binary; otherwise multi-class ID)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Video frame rate (default 30)",
    )
    parser.add_argument(
        "--max-instances", type=int, default=2,
        help="Maximum mouse instances (default 2)",
    )
    parser.add_argument(
        "--skip-temporal", action="store_true",
        help="Skip temporal model stage, use meta-LGBM output directly as final result",
    )
    parser.add_argument(
        "--trim-start-frames", type=int, default=0,
        help="Trim first N frames from inference (default 0)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Number of parallel processes for factor computation (default 0 = single process)",
    )
    parser.add_argument(
        "--num-chunks", type=int, default=256,
        help="Number of window chunks (default 256)",
    )
    # Compatible with old interface: accepts --config / --config-common / --config-validation
    parser.add_argument(
        "--config", type=str, default=None,
        help="External configuration file (optional, overrides decoder config in run directory)",
    )
    parser.add_argument(
        "--config-validation", type=str, default=None,
        help="External validation configuration file (optional)",
    )
    parser.add_argument(
        "--config-common", type=str, default=None,
        help="External common configuration file (optional)",
    )
    args = parser.parse_args()

    # ---- Validate inputs ----
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"Error: Training experiment directory does not exist -- {run_dir}")
        sys.exit(1)

    mouse_dir = Path(args.mouse_keypoints)
    if not mouse_dir.is_dir():
        print(f"Error: Mouse keypoint directory does not exist -- {mouse_dir}")
        sys.exit(1)

    tail_dir = Path(args.tail_keypoints)
    if not tail_dir.is_dir():
        print(f"Error: Tail keypoint directory does not exist -- {tail_dir}")
        sys.exit(1)

    # ---- Load config ----
    merged_path = run_dir / "configs" / "merged_config.yaml"
    if not merged_path.exists():
        print(f"Error: merged_config.yaml not found -- {merged_path}")
        sys.exit(1)

    with open(merged_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Merge external config
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            ext_cfg = yaml.safe_load(f)
        for section in ["temporal_validation", "synth_validation"]:
            if section in ext_cfg:
                cfg.setdefault(section, {}).update(ext_cfg[section])
    elif args.config_common and args.config_validation:
        cfg = _load_merged_config(args.config_common, args.config_validation)
    elif args.config_validation:
        with open(args.config_validation, "r", encoding="utf-8") as f:
            ext_cfg = yaml.safe_load(f)
        for section in ["temporal_validation", "synth_validation"]:
            if section in ext_cfg:
                cfg.setdefault(section, {}).update(ext_cfg[section])

    # ---- Logging ----
    setup_logging(cfg.get("output", {}).get("logs_dir", "logs"))
    logger.info("=" * 60)
    logger.info("SABER inference starting")
    logger.info(f"  Experiment directory: {run_dir}")
    logger.info(f"  Mouse keypoints: {mouse_dir}")
    logger.info(f"  Tail keypoints: {tail_dir}")
    logger.info(f"  Target class: {args.target_class or '(multi-class)'}")
    logger.info("=" * 60)

    # ---- Load data (get merged_data + flat_attributes) ----
    keypoints_full, flat_attributes, merged_data, normalizer = load_inference_data(
        str(mouse_dir), str(tail_dir),
        max_instances_num=args.max_instances,
        fps=args.fps,
    )

    # ---- Initialize inference pipeline ----
    pipeline = InferencePipeline(run_dir, cfg)

    # ---- Determine mouse order (first-frame top-left) ----
    max_instance_num = args.max_instances
    center_order = determine_mouse_order(merged_data, max_instance_num)
    # center_order[logical_id] = original_instance_index
    # logical_id=0 -> mouse1 (top-left), logical_id=1 -> mouse2

    # ---- Run inference for both mice ----
    all_probas = {}
    all_labels = {}
    all_label_names = {}
    total_frames = 0  # Will be determined after first inference round

    for logical_id in range(max_instance_num):
        actual_center_id = center_order[logical_id]
        logger.info(f"\n{'=' * 40}")
        logger.info(
            f"[inference] Mouse {logical_id + 1} inference"
            f" (logical_id={logical_id}, original center_id={actual_center_id})"
        )
        logger.info(f"{'=' * 40}")

        # Build keypoints for current center_id (recompute from merged_data)
        kp_center, flat_attrs_center = build_keypoints_for_center(
            keypoints_full, merged_data,
            center_id=actual_center_id,
            max_instance_num=max_instance_num,
            num_keypoints=10,  # 7 mouse + 3 tail
            fps=args.fps,
            normalizer=normalizer,
        )

        if kp_center.size == 0:
            logger.warning(f"[inference] Mouse {logical_id + 1} has no valid data, skipping")
            all_labels[logical_id] = np.array([], dtype=int)
            all_label_names[logical_id] = []
            continue

        total_frames = kp_center.shape[0]
        logger.info(f"[inference] Mouse {logical_id + 1} keypoints: {kp_center.shape}")

        # --trim-start-frames: trim first N frames
        if args.trim_start_frames > 0:
            trim_n = min(args.trim_start_frames, total_frames)
            kp_center = kp_center[trim_n:]
            logger.info(
                f"[inference] Mouse {logical_id + 1} trimmed first {trim_n} frames, "
                f"remaining {kp_center.shape[0]} frames"
            )
            total_frames = kp_center.shape[0]
            if total_frames == 0:
                logger.warning(f"[inference] Mouse {logical_id + 1} has no data after trimming, skipping")
                continue

        # Run inference
        t0 = _time.time()
        proba = pipeline.predict(
            kp_center, flat_attrs_center,
            center_id=actual_center_id,
            num_workers=args.num_workers,
            num_chunks=args.num_chunks,
            skip_temporal=args.skip_temporal,
        )
        logger.info(f"[inference] Mouse {logical_id + 1} inference time: {_time.time() - t0:.1f}s")

        # Decode to labels
        labels = pipeline.decode_to_labels(proba, target_class=args.target_class)
        all_probas[logical_id] = proba
        all_labels[logical_id] = labels

        # Generate class name labels
        if args.target_class is None:
            label_names = [pipeline.id_to_name.get(int(l), f"class_{l}")
                          for l in labels]
        else:
            label_names = [
                args.target_class if l == 1 else "other" for l in labels
            ]
        all_label_names[logical_id] = label_names

    # ---- Verify at least one valid result ----
    if not all_labels or total_frames == 0:
        print("Error: No valid inference results. Please check input data and model.")
        sys.exit(1)

    # ---- Save results ----
    output_path = Path(args.output)
    output_ext = output_path.suffix.lower()

    # Build time column
    time_sec = np.arange(total_frames, dtype=np.float32) / args.fps

    if output_ext == ".xlsx":
        import pandas as pd
        data = {"frame": np.arange(total_frames), "time_sec": time_sec.round(4)}
        for center_id in sorted(all_labels.keys()):
            data[f"label_mouse{center_id + 1}"] = all_labels[center_id]
            data[f"behavior_mouse{center_id + 1}"] = all_label_names[center_id]
        # Add per-class probabilities
        for center_id in sorted(all_probas.keys()):
            for ci, cname in enumerate(pipeline.class_names):
                if ci < all_probas[center_id].shape[1]:
                    data[f"proba_mouse{center_id + 1}_{cname}"] = \
                        all_probas[center_id][:, ci].round(6)
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        logger.info(f"Results saved: {output_path}")
    else:
        # CSV output
        import csv
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # Build header
            header = ["frame", "time_sec"]
            for center_id in sorted(all_labels.keys()):
                header.append(f"label_mouse{center_id + 1}")
                header.append(f"behavior_mouse{center_id + 1}")
            # Add probability columns
            for center_id in sorted(all_probas.keys()):
                for ci, cname in enumerate(pipeline.class_names):
                    if ci < all_probas[center_id].shape[1]:
                        header.append(f"proba_mouse{center_id + 1}_{cname}")
            writer.writerow(header)

            # Write per-frame
            for t in range(total_frames):
                row = [t, round(float(time_sec[t]), 4)]
                for center_id in sorted(all_labels.keys()):
                    row.append(int(all_labels[center_id][t]))
                    row.append(str(all_label_names[center_id][t]))
                for center_id in sorted(all_probas.keys()):
                    for ci, cname in enumerate(pipeline.class_names):
                        if ci < all_probas[center_id].shape[1]:
                            row.append(round(float(all_probas[center_id][t, ci]), 6))
                writer.writerow(row)
        logger.info(f"Results saved: {output_path}")

    # ---- Output summary ----
    logger.info("\n" + "=" * 60)
    logger.info("Inference complete summary")
    logger.info("=" * 60)
    for center_id in sorted(all_labels.keys()):
        labels_arr = all_labels[center_id]
        unique, counts = np.unique(labels_arr, return_counts=True)
        dist = dict(zip(unique, counts))
        logger.info(f"Mouse {center_id + 1} label distribution: {dist}")
        if args.target_class is not None:
            pct = 100.0 * counts[1] / len(labels_arr) if 1 in unique else 0.0
            logger.info(
                f"  Target behavior '{args.target_class}' ratio: {pct:.1f}% "
                f"({dist.get(1, 0)}/{len(labels_arr)} frames)"
            )

    print(f"\nInference complete! Results saved to: {output_path.resolve()}")
    print(f"Total {total_frames} frames, {args.fps} fps, "
          f"duration {total_frames / args.fps / 60:.1f} minutes")


if __name__ == "__main__":
    main()
