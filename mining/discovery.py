"""
discovery.py
Main workflow: LLM-driven behavior factor discovery loop.

Flow:
  Step 0: Build MouseBehaviorDataset, extract keypoints tensor and labels
  Step 1: LLM generates factor hypotheses (based on feature_indexer.flat_attributes)
  Step 2: FactorEngine executes hypothesis code on keypoints -> factor value array [T]
  Step 3: LightGBM validates factor predictive power against labels
  Step 4: Save valid factors + round experience -> return to Step 1
"""


import sys
from pathlib import Path
# Allow running as a script from a subdirectory (e.g. `python mining/discovery.py`)
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import os
import json
import gc
import ctypes
import yaml
import logging
import argparse
import numpy as np
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from tqdm import tqdm
import tqdm as tqdm_module

from src.llm_client import LLMClient
from src.hypothesis_generator import HypothesisGenerator
from src.factor_engine import FactorEngine
# FactorValidator imported lazily inside run() — importing it at module level
# causes lightgbm → OpenMP initialization in the parent process, which makes
# fork() unsafe for tuner.py and any other tool that imports from discovery.
from src.memory import MemoryManager


# ------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------
def setup_logging(log_dir: str = "logs"):
    import sys
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    # Force stdout to use UTF-8 (compatible with subprocess pipes)
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    return logging.getLogger("main"), str(log_file)


# ------------------------------------------------------------------
# label_merge helper
# ------------------------------------------------------------------
def _apply_label_merge(label_map: dict, merge_cfg: dict, logger) -> dict:
    """
    Merge categories according to label_merge config and renumber compactly.

    merge_cfg format:
      {
        "enabled": true,
        "groups": [
          { "target": "stand", "sources": ["climbsocial"] }
        ]
      }

    Each group maps source classes to the target's ID,
    removes source classes from label_map, and renumbers remaining classes
    in ascending original ID order (0, 1, 2, ...).
    """
    # Build source -> target remapping table (at original ID level)
    remap: dict[int, int] = {}
    to_remove: set[str] = set()

    for group in merge_cfg.get("groups", []):
        target_name = group.get("target")
        sources = group.get("sources", [])
        if target_name not in label_map:
            logger.warning(f"label_merge: target '{target_name}' not in label_map, skipping")
            continue
        target_id = label_map[target_name]
        for src in sources:
            if src not in label_map:
                logger.warning(f"label_merge: source '{src}' not in label_map, skipping")
                continue
            remap[label_map[src]] = target_id
            to_remove.add(src)
            logger.info(f"label_merge: '{src}'({label_map[src]}) -> '{target_name}'({target_id})")

    # Remove source classes, keep the rest
    merged = {k: v for k, v in label_map.items() if k not in to_remove}

    # Renumber compactly in ascending original ID order
    sorted_names = sorted(merged, key=lambda k: merged[k])
    new_label_map = {name: i for i, name in enumerate(sorted_names)}

    # Log final mapping
    logger.info(f"label_map after label_merge: {new_label_map}")
    return new_label_map


# ------------------------------------------------------------------
# Memory management
# ------------------------------------------------------------------
def _trim_memory(logger):
    """Force Python garbage collection and attempt to return memory to the OS."""
    gc.collect()
    # On Windows, ctypes call to malloc_trim is unavailable; use gc twice
    # and release numpy internal caches
    try:
        # Try libc malloc_trim (Linux/macOS)
        libc = ctypes.CDLL("libc.so.6", use_errno=False)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass  # Not supported on Windows, rely on gc.collect() only
    gc.collect()
    logger.debug("Memory reclamation complete")


# ------------------------------------------------------------------
# Load external dataset configuration (JSON)
# ------------------------------------------------------------------
def load_dataset_config(cfg: dict, config_file_path: str, logger) -> dict:
    """
    Read the JSON file pointed to by dataset_config_file, parse out:
      - train_dataset_config : list[dict]  training set config for MouseBehaviorDataset
      - val_dataset_config   : list[dict]  validation set config for MouseBehaviorDataset
      - label_map            : dict        behavior label -> integer mapping

    JSON file structure:
      {
        "train": [ { "Mouse_key_point_file": [...], "Tail_key_point_file": [...],
                     "behavior_file_mouse1": [...], "behavior_file_mouse2": [...],
                     "max_instances_num": 2 } ],
        "val":   [ { ... } ],
        "label_map": "<path_to_label_map.json>"   # or directly a dict
      }

    If config_file_path is relative, it is resolved relative to the directory of config.yaml.
    """
    config_dir = Path(config_file_path).parent
    json_path = Path(config_file_path)
    if not json_path.is_absolute():
        # Relative to the config.yaml directory
        json_path = config_dir / json_path.name
        # If cfg comes from a config file specified on the command line, use its path as base
        # Using the parent directory of config_file_path is sufficient here

    json_path = json_path.resolve()
    logger.info(f"Loading dataset configuration file: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        ds_json = json.load(f)

    # --- label_map ---
    raw_label_map = ds_json.get("label_map", {})
    if isinstance(raw_label_map, str):
        # It is a file path, resolved relative to the JSON file's directory
        lm_path = (json_path.parent / raw_label_map).resolve()
        if not lm_path.exists():
            # Also try as an absolute path
            lm_path = Path(raw_label_map)
        logger.info(f"Loading label_map file: {lm_path}")
        with open(lm_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    elif isinstance(raw_label_map, dict):
        label_map = raw_label_map
    else:
        logger.warning("No valid label_map found in JSON, falling back to label_map in config.yaml")
        label_map = cfg.get("label_map", {})

    train_ds_cfg = ds_json.get("train", [])
    val_ds_cfg   = ds_json.get("val",   [])

    # --- label_merge processing ---
    label_merge_cfg = ds_json.get("label_merge", {})
    if label_merge_cfg.get("enabled", False):
        label_map = _apply_label_merge(label_map, label_merge_cfg, logger)

    logger.info(
        f"Dataset configuration loaded: train groups {len(train_ds_cfg)} entries, "
        f"val groups {len(val_ds_cfg)} entries, "
        f"label_map has {len(label_map)} classes"
    )
    return {
        "train_dataset_config": train_ds_cfg,
        "val_dataset_config":   val_ds_cfg,
        "label_map":            label_map,
    }


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def merge_dataset_config_list(cfg_list: list) -> dict:
    """
    Merge the list[dict] from train/val in dataset_config.json into a single dict.

    MouseBehaviorDataset expects dataset_config to be a dict where
    Mouse_key_point_file / Tail_key_point_file / behavior_file_mouse1 /
    behavior_file_mouse2 are all path lists, and max_instances_num is an integer.

    In dataset_config.json, train/val are lists, each element is already
    this type of dict (path fields are lists). If the list has only one element,
    return it directly; if multiple elements, concatenate the path lists
    and take max_instances_num from the first element.
    """
    if not cfg_list:
        raise ValueError("dataset_config list is empty, please check dataset_config.json")

    if len(cfg_list) == 1:
        return cfg_list[0]

    # Merge path lists for multiple dicts
    list_keys = [
        "Mouse_key_point_file",
        "Tail_key_point_file",
        "behavior_file_mouse1",
        "behavior_file_mouse2",
    ]
    merged = {k: [] for k in list_keys}
    merged["max_instances_num"] = cfg_list[0].get("max_instances_num", 2)

    for entry in cfg_list:
        for k in list_keys:
            val = entry.get(k, [])
            if isinstance(val, list):
                merged[k].extend(val)
            elif val:  # Single string
                merged[k].append(val)

    return merged


def build_train_val_datasets(ds_info: dict, seq_cfg: dict, logger, cfg: dict = None):
    """
    Build training and validation MouseBehaviorDatasets from parsed dataset configuration.

    dataset_config.json structure:
      train / val are each list[dict], each dict's path fields are already lists
      (each path corresponds to one video segment).
    MouseBehaviorDataset accepts a single dict (path fields as lists),
    so first call merge_dataset_config_list to merge the list into one dict.

    Returns:
      train_data : (keypoints_np [T_train, D], labels_np [T_train])
      val_data   : (keypoints_np [T_val,   D], labels_np [T_val])
      flat_attributes : list[str]  feature column names (shared across datasets)
    """
    label_map = ds_info["label_map"]

    # --- force_cache fast path ---
    if cfg:
        fc = cfg.get("force_cache", {})
        fc_path = fc.get("windowed") or fc.get("path")
        if fc_path:
            return _load_force_cache(fc_path, label_map, logger)

    from src.data_loader import MouseBehaviorDataset

    train_ds_cfg = merge_dataset_config_list(ds_info["train_dataset_config"])
    val_ds_cfg   = merge_dataset_config_list(ds_info["val_dataset_config"])

    seq_len   = seq_cfg.get("seq_length", 1)
    stride    = seq_cfg.get("stride", 1)
    frame_int = seq_cfg.get("frame_interval", 1)
    purity    = seq_cfg.get("purity_threshold", 1.0)
    boundary_margin = seq_cfg.get("boundary_margin", 0)
    short_gap_max   = (cfg or {}).get("imputation", {}).get("short_gap_max", 10)

    import threading
    import src.data_loader as _dl

    def _count_txt_files(ds_cfg):
        """Count total txt files in Mouse + Tail directories."""
        total = 0
        for key in ("Mouse_key_point_file", "Tail_key_point_file"):
            for folder in ds_cfg.get(key, []):
                p = Path(folder)
                if p.is_dir():
                    total += len(list(p.glob("*.txt")))
        return total

    total_files = _count_txt_files(train_ds_cfg) + _count_txt_files(val_ds_cfg)

    @contextmanager
    def _forwarding_tqdm(master_pbar):
        """
        Replace data_loader's internal tqdm with a forwarding version:
        - Iterator mode (for-loop): update master by 1 per item consumed
        - Context manager mode (video frame level): silent, don't interfere with total count
        """
        lock = threading.Lock()

        class _ForwardingBar:
            def __init__(self, iterable=None, **kwargs):
                self._iter = iterable
                # Only forward progress when there is an iterable (file-level loop)
                self._forward = iterable is not None

            def __iter__(self):
                for item in (self._iter or []):
                    yield item
                    if self._forward:
                        with lock:
                            master_pbar.update(1)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def update(self, n=1):
                # Frame-by-frame update in context-manager mode — do not forward, avoid interfering with total count
                pass

            def set_description(self, *args, **kwargs):
                pass

            def close(self):
                pass

        orig = _dl.tqdm
        _dl.tqdm = _ForwardingBar
        try:
            yield
        finally:
            _dl.tqdm = orig

    def _inject_cache(ds_cfg, split_name):
        """If ds_cfg specifies cache_name or cache_path, link/copy it to the hash path expected by data_loader."""
        cache_path = ds_cfg.pop("cache_path", None)
        cache_name = ds_cfg.pop("cache_name", None)

        # cache_name takes priority: look for {cache_name}.pt in dataset_cache/
        if cache_name and not cache_path:
            candidate = Path("dataset_cache") / f"{cache_name}.pt"
            if candidate.exists():
                cache_path = str(candidate)
            else:
                logger.warning(f"[{split_name}] cache_name={cache_name} file does not exist: {candidate}")

        if not cache_path:
            return
        src = Path(cache_path)
        if not src.exists():
            logger.warning(f"[{split_name}] specified cache file does not exist, ignoring: {src}")
            return
        config_content = {
            "Mouse_files": ds_cfg.get("Mouse_key_point_file", []),
            "Tail_files": ds_cfg.get("Tail_key_point_file", []),
            "max_instances": ds_cfg.get("max_instances_num", 2),
            "method_version": "v1.1",
        }
        config_str = json.dumps(config_content, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()
        cache_dir = Path("dataset_cache")
        cache_dir.mkdir(exist_ok=True)
        dst = cache_dir / f"cached_data_{config_hash}.pt"
        if dst.exists():
            logger.info(f"[{split_name}] target cache already exists, skipping: {dst}")
            return
        # Prefer hard link (zero extra disk usage), fall back to copy
        try:
            os.link(src, dst)
            logger.info(f"[{split_name}] hard-linked cache {src.name} -> {dst.name}")
        except OSError:
            shutil.copy2(src, dst)
            logger.info(f"[{split_name}] copied cache {src.name} -> {dst.name}")

    def _build(ds_cfg, split_name, master_pbar):
        _inject_cache(ds_cfg, split_name)
        n_videos = len(ds_cfg.get("Mouse_key_point_file", []))
        logger.info(f"Building {split_name} MouseBehaviorDataset ({n_videos} video segments)...")
        master_pbar.set_description(f"Loading dataset [{split_name}]")
        with _forwarding_tqdm(master_pbar):
            mbd = MouseBehaviorDataset(
                dataset_config=ds_cfg,
                label_map=label_map,
                seq_length=seq_len,
                stride=stride,
                frame_interval=frame_int,
                transform=None,
                is_train=True,
                purity_threshold=purity,
                boundary_margin=boundary_margin,
                short_gap_max=short_gap_max,
            )
        logger.info(f"{split_name} dataset construction complete")
        kp_full = mbd.keypoints.cpu().numpy().astype(np.float32)  # [T, D]
        attrs = mbd.feature_indexer.flat_attributes
        valid_starts = mbd.valid_start_indices  # list of (start_frame, label) or int
        del mbd  # Release MouseBehaviorDataset internal cache (after extracting all needed attributes)
        if not valid_starts:
            logger.warning(f"{split_name} no valid windows, returning empty arrays")
            D = kp_full.shape[1] if kp_full.ndim == 2 else 0
            return np.empty((0, seq_len, D), dtype=np.float32), np.empty(0, dtype=np.int64), attrs

        # Compatible with test mode (elements are int) and training mode (elements are (start, label) tuples)
        if isinstance(valid_starts[0], tuple):
            starts = [s for s, _ in valid_starts]
            labels = np.array([lbl for _, lbl in valid_starts], dtype=np.int64)
        else:
            starts = list(valid_starts)
            labels = np.zeros(len(starts), dtype=np.int64)

        windows = np.stack(
            [kp_full[[s + i * frame_int for i in range(seq_len)]] for s in starts],
            axis=0,
        )  # [N, seq_length, D]

        # Release intermediate arrays to reduce peak memory (kp_full may be large, already copied into windows)
        del kp_full
        _trim_memory(logger)

        logger.info(f"{split_name} window matrix shape={windows.shape}, labels shape={labels.shape}")
        return windows, labels, attrs

    with tqdm(total=total_files or None, desc="Loading dataset [train]",
              unit="files", dynamic_ncols=True, leave=True) as master_pbar:
        train_kp, train_lb, flat_attributes = _build(train_ds_cfg, "train", master_pbar)
        val_kp,   val_lb,   _               = _build(val_ds_cfg,   "val",   master_pbar)

    logger.info(f"Feature dimension D={len(flat_attributes)}, first 10 features: {flat_attributes[:10]} ...")
    return (train_kp, train_lb), (val_kp, val_lb), flat_attributes


def save_raw_frame_cache(train_data, val_data, flat_attributes, label_map, output_path, logger):
    """
    Save the result of build_raw_frame_data as a .pt file for force_cache direct loading.

    Usage:
      save_raw_frame_cache(
          train_data=(train_kp, train_lb),
          val_data=(val_kp, val_lb),
          flat_attributes=flat_attributes,
          label_map=label_map,
          output_path="dataset_cache/my_cache.pt",
          logger=logger,
      )
    """
    import torch
    cache_dict = {
        "train_kp": train_data[0],
        "train_lb": train_data[1],
        "val_kp": val_data[0],
        "val_lb": val_data[1],
        "flat_attributes": flat_attributes,
        "label_map": label_map,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_dict, output_path)
    logger.info(f"[force_cache] Saved frame-level cache: {output_path}")


def _load_force_cache(cache_path, label_map, logger):
    """
    Load the .pt file specified by force_cache, skipping all MouseBehaviorDataset processing.

    The .pt file must contain: train_kp, train_lb, val_kp, val_lb, flat_attributes
    """
    import torch
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"[force_cache] File does not exist: {path}")

    logger.info(f"[force_cache] Directly loading preprocessed cache: {path}")
    cache_dict = torch.load(path, weights_only=False)

    train_kp = np.asarray(cache_dict["train_kp"], dtype=np.float32)
    train_lb = np.asarray(cache_dict["train_lb"], dtype=np.int64)
    val_kp = np.asarray(cache_dict["val_kp"], dtype=np.float32)
    val_lb = np.asarray(cache_dict["val_lb"], dtype=np.int64)
    flat_attributes = cache_dict["flat_attributes"]

    valid_labels = set(int(v) for v in label_map.values())
    train_mask = np.isin(train_lb, list(valid_labels))
    val_mask = np.isin(val_lb, list(valid_labels))
    if not train_mask.all() or not val_mask.all():
        train_kp, train_lb = train_kp[train_mask], train_lb[train_mask]
        val_kp, val_lb = val_kp[val_mask], val_lb[val_mask]
        logger.info(f"[force_cache] After filtering invalid labels: train {train_kp.shape[0]} frames, val {val_kp.shape[0]} frames")

    logger.info(
        f"[force_cache] Load complete: train {train_kp.shape[0]} frames, val {val_kp.shape[0]} frames, "
        f"D={train_kp.shape[1]}, n_features={len(flat_attributes)}"
    )
    return (train_kp, train_lb), (val_kp, val_lb), flat_attributes


def build_raw_frame_data(ds_info: dict, logger, cfg: dict = None):
    """
    Return raw per-frame data (no windowing), for multi-resolution synthetic validation.

    If force_cache is configured in cfg, directly load the specified .pt file,
    skipping all data processing.

    Returns:
      train_data: (kp_full [T_train, D], labels [T_train]) — only frames with labels in label_map
      val_data:   (kp_full [T_val,   D], labels [T_val])
      flat_attributes: list[str]
    """
    label_map = ds_info["label_map"]

    # force_cache has been removed — always compute from scratch.

    from src.data_loader import MouseBehaviorDataset

    valid_labels = set(int(v) for v in label_map.values())
    train_ds_cfg = merge_dataset_config_list(ds_info["train_dataset_config"])
    val_ds_cfg   = merge_dataset_config_list(ds_info["val_dataset_config"])

    def _build(ds_cfg, split_name):
        logger.info(f"Building {split_name} raw frame data (multi-resolution mode)...")
        preprocess_cfg = (cfg or {}).get("preprocessing", {})
        aug_cfg = preprocess_cfg.get("augmentation", {}) if preprocess_cfg else {}
        mbd = MouseBehaviorDataset(
            dataset_config=ds_cfg,
            label_map=label_map,
            seq_length=1,
            stride=1,
            frame_interval=1,
            transform=None,
            is_train=True,
            purity_threshold=1.0,
            boundary_margin=0,
            per_video_normalize=True,
        )
        kp_full = mbd.keypoints.cpu().numpy().astype(np.float32)  # [T, D]
        if mbd.labels is None:
            raise RuntimeError(
                f"[{split_name}] MouseBehaviorDataset.labels is None, "
                "please check if behavior_file_mouse1 path exists and format is correct."
            )
        labels  = mbd.labels.astype(np.int64)                # [T]
        attrs   = mbd.feature_indexer.flat_attributes
        # Per-video frame counts for downstream per-video window building.
        # Mouse1 + mouse2 use the same video lengths (same videos, same frames).
        _vlen_m1 = list(getattr(mbd, '_video_m1_lengths', []))
        _vlen_m2 = list(getattr(mbd, '_video_m2_lengths', []))
        del mbd  # Release MouseBehaviorDataset internal cache (may be large)

        # Under some configurations, keypoints and labels frame counts may differ slightly; take min to align
        T = min(len(kp_full), len(labels))
        if T < max(len(kp_full), len(labels)):
            logger.warning(
                f"[{split_name}] keypoints({len(kp_full)}) and labels({len(labels)}) "
                f"frame counts differ, truncating to {T} frames"
            )
        kp_full = kp_full[:T]
        labels  = labels[:T]

        # Log original label distribution BEFORE filtering for diagnosis
        orig_unique, orig_counts = np.unique(labels, return_counts=True)
        logger.info(
            f"{split_name} raw frames (before filtering): {len(labels)} frames, "
            f"labels present: {dict(zip(orig_unique.astype(int).tolist(), orig_counts.astype(int).tolist()))}"
        )
        logger.info(f"{split_name} valid label set (label_map values): {sorted(valid_labels)}")

        mask = np.isin(labels, list(valid_labels))
        kp_full = kp_full[mask]
        labels  = labels[mask]

        # Compute per-video frame counts AFTER label filtering (for window building)
        _video_lengths = None
        if _vlen_m1:
            _m1_total = sum(_vlen_m1)
            _vl_m1 = []
            _off = 0
            for vl in _vlen_m1:
                _vmask = mask[_off:_off + vl]
                _surv = int(_vmask.sum())
                if _surv > 0:
                    _vl_m1.append(_surv)
                _off += vl
            # Mouse2 part of mask
            _vl_m2 = []
            for vl in _vlen_m2:
                _vmask = mask[_off:_off + vl]
                _surv = int(_vmask.sum())
                if _surv > 0:
                    _vl_m2.append(_surv)
                _off += vl
            if len(_vl_m1) == len(_vl_m2) and len(_vl_m1) > 1:
                _video_lengths = (_vl_m1, _vl_m2)

        logger.info(
            f"{split_name} raw frames: {kp_full.shape[0]} frames (after filtering), "
            f"D={kp_full.shape[1]}, label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}"
        )
        return kp_full, labels, attrs, _video_lengths

    train_kp, train_lb, flat_attributes, train_video_lengths = _build(train_ds_cfg, "train")
    val_kp,   val_lb,   _,               val_video_lengths   = _build(val_ds_cfg,   "val")

    # Register class names for GUI factor detail dialogs
    try:
        from gui.widgets.factor_detail import set_class_names
        id_to_name = {str(v): k for k, v in label_map.items()}
        set_class_names(id_to_name)
    except Exception:
        pass

    return (train_kp, train_lb), (val_kp, val_lb), flat_attributes, (train_video_lengths, val_video_lengths)


# ------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------
def run(cfg: dict, config_file: str = "config.yaml"):
    logger, _ = setup_logging(cfg["output"]["logs_dir"])
    logger.info("=" * 60)
    logger.info("Factor Discovery System Starting")
    logger.info("=" * 60)

    # ---------- Step 0: Load dataset configuration + build datasets ----------
    ds_cfg_file = cfg.get("dataset_config_file", "")
    if not ds_cfg_file:
        raise ValueError("config.yaml is missing the dataset_config_file field. Please specify the path to dataset_config.json.")

    # Relative paths are based on the directory of config.yaml
    cfg_dir = Path(config_file).resolve().parent
    ds_cfg_path = Path(ds_cfg_file)
    if not ds_cfg_path.is_absolute():
        ds_cfg_path = cfg_dir / ds_cfg_path

    ds_info = load_dataset_config(cfg, str(ds_cfg_path), logger)

    # Sync behavior_classes with merged label_map to ensure LLM prompt matches actual classes
    merged_label_map = ds_info["label_map"]
    cfg.setdefault("data", {})["behavior_classes"] = sorted(merged_label_map, key=merged_label_map.get)

    seq_cfg = cfg.get("sequence", {})
    train_data, val_data, flat_attributes = build_train_val_datasets(ds_info, seq_cfg, logger, cfg)
    train_kp, train_lb = train_data
    val_kp,   val_lb   = val_data

    logger.info(f"Training set: {train_kp.shape[0]} windows (seq_length={train_kp.shape[1]}), Validation set: {val_kp.shape[0]} windows")

    # Force memory reclamation after data loading (release MouseBehaviorDataset internal cache)
    _trim_memory(logger)

    # ---------- Initialize modules ----------
    from src.validator import FactorValidator  # lazy import: keeps parent process OpenMP-free for fork safety
    llm = LLMClient(cfg)
    generator = HypothesisGenerator(llm, cfg, seq_length=int(train_kp.shape[1]))
    engine = FactorEngine(cfg)
    validator = FactorValidator(cfg)
    memory = MemoryManager(cfg)

    # ---------- Main loop configuration ----------
    loop_cfg = cfg["loop"]
    max_rounds = loop_cfg.get("max_rounds", 20)
    max_valid = loop_cfg.get("max_valid_factors", 50)
    early_stop = loop_cfg.get("early_stop_rounds", 5)
    no_new_rounds = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    round_num = 0
    for round_num in range(1, max_rounds + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Round {round_num} | Saved valid factors: {len(memory.get_valid_factors())}")
        logger.info(f"{'=' * 60}")

        # Use _factor_names (includes shared library synced factors) instead of just this worker's valid_factors
        total_known = len(memory._factor_names)
        if total_known >= max_valid:
            logger.info(f"Reached maximum valid factor count (known {total_known} >= {max_valid}, "
                        f"of which this worker contributed {len(memory.get_valid_factors())}), stopping loop.")
            break

        if no_new_rounds >= early_stop:
            logger.info(f"{early_stop} consecutive rounds with no new valid factors, stopping early.")
            break

        # ---------- Cross-process sync: load parallel worker's new factors from shared library ----------
        if loop_cfg.get("reload_each_round"):
            shared_factors = loop_cfg.get("shared_factors_path", "")
            shared_exp = loop_cfg.get("shared_experience_path", "")
            if shared_factors or shared_exp:
                memory.reload_from_shared(
                    shared_factors_path=shared_factors or None,
                    shared_experience_path=shared_exp or None,
                )

        # ---------- Step 1: LLM generates hypotheses ----------
        use_memory = loop_cfg.get("use_memory", True)
        max_summary_chars = loop_cfg.get("max_summary_chars", 500)
        keep_recent_rounds = loop_cfg.get("keep_recent_rounds", 15)
        keep_recent_factors = loop_cfg.get("keep_recent_factors", 80)
        earlier_max_chars = loop_cfg.get("earlier_max_chars", 1500)
        experience_summary = (
            memory.get_experience_summary(
                max_chars=max_summary_chars,
                keep_recent_rounds=keep_recent_rounds,
                keep_recent_factors=keep_recent_factors,
                earlier_max_chars=earlier_max_chars,
            )
            if use_memory
            else "(Memory disabled)"
        )

        # Append diversity direction (assigned by batch_mining.py, guides different workers to explore different directions)
        diversity_focus = loop_cfg.get("diversity_focus", "")
        if use_memory and diversity_focus:
            direction = memory.get_diversity_direction(diversity_focus)
            if direction:
                experience_summary = experience_summary + "\n" + direction

        # Compute weak classes (behavior categories with no valid factors)
        min_auc = cfg["validation"].get("min_auc", 0.65)
        covered = set()
        for f in memory.get_valid_factors():
            for v in f.get("valid_classes", []):
                covered.add(v["class"])
        id_to_name = {str(v): k for k, v in ds_info["label_map"].items()}
        weak_classes = [id_to_name[cid] for cid in sorted(id_to_name) if cid not in covered]
        if weak_classes:
            logger.info(f"Weak classes (no valid factors): {weak_classes}")

        hypotheses = generator.generate(flat_attributes, experience_summary, weak_classes=weak_classes)
        if not hypotheses:
            logger.warning("LLM returned no valid hypotheses, skipping this round.")
            no_new_rounds += 1
            continue

        logger.info(f"This round generated {len(hypotheses)} hypotheses: {[h['name'] for h in hypotheses]}")

        # ---------- Step 2 & 3: Compute factors + Validate ----------
        round_results = []
        round_new_valid = 0

        for hypothesis in hypotheses:
            name = hypothesis["name"]

            if name in memory.get_valid_factor_names():
                logger.info(f"  [{name}] Already exists, skipping.")
                round_results.append({"valid": False, "reason": "Already exists"})
                continue

            logger.info(f"  [{name}] Computing factor values (mode={hypothesis.get('mode','row')})...")

            # Compute factor values on training set (for LightGBM training)
            if hypothesis.get("mode") == "batch":
                train_factor = engine.compute_factor_batch(hypothesis, train_kp, flat_attributes)
                val_factor   = engine.compute_factor_batch(hypothesis, val_kp,   flat_attributes)
            else:
                train_factor = engine.compute_factor(hypothesis, train_kp, flat_attributes)
                val_factor   = engine.compute_factor(hypothesis, val_kp,   flat_attributes)

            if train_factor is None or val_factor is None:
                logger.info(f"  [{name}] Factor computation failed, skipping.")
                round_results.append({"valid": False, "reason": "Computation failed"})
                continue

            logger.info(f"  [{name}] Validating (train -> val holdout)...")
            val_result = validator.validate_single_holdout(
                train_factor, train_lb, val_factor, val_lb
            )
            round_results.append(val_result)

            status = "VALID" if val_result["valid"] else "INVALID"
            logger.info(
                f"  [{name}] {status} | best_auc={val_result['best_auc']:.4f} | "
                f"class={val_result.get('best_class','')} | {val_result.get('reason', '')}"
            )

            # ---------- Step 4: Save valid factors ----------
            if val_result["valid"]:
                memory.save_valid_factor(hypothesis, val_result, seq_length=int(train_kp.shape[1]))
                round_new_valid += 1

        # Save round experience
        memory.save_round_experience(round_num, hypotheses, round_results)

        if round_new_valid == 0:
            no_new_rounds += 1
        else:
            no_new_rounds = 0
            logger.info(f"This round added {round_new_valid} new valid factors!")

        # Force memory reclamation after each round (reduce memory fragmentation over long runs)
        _trim_memory(logger)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    valid_factors = memory.get_valid_factors()
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Factor discovery complete, ran {round_num} rounds")
    logger.info(f"Valid factors found: {len(valid_factors)}")
    if valid_factors:
        logger.info("Valid factor list:")
        for f in sorted(valid_factors, key=lambda x: x.get("best_auc", x.get("auc", 0)), reverse=True):
            logger.info(
                f"  {f['name']:35s} | valid_classes={[v['class'] for v in f.get('valid_classes', [])]} | "
                f"best_auc={f.get('best_auc', f.get('auc', 0)):.4f}"
            )
    logger.info(f"{'=' * 60}\n")

    memory.log_run({
        "timestamp": datetime.now().isoformat(),
        "rounds": round_num,
        "n_valid_factors": len(valid_factors),
    })

    return valid_factors


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------
def _load_merged_config(*paths: str) -> dict:
    """Load multiple YAML files in order; later files' top-level keys override earlier ones."""
    merged: dict = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            merged.update(yaml.safe_load(f) or {})
    return merged


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM-driven behavior factor discovery system")
    parser.add_argument("--config-common",    default="config/seq/1.yaml",    help="Common configuration file")
    parser.add_argument("--config-discovery", default="config/seq/1.yaml", help="Factor mining configuration file")
    parser.add_argument("--config", default=None, help="Single config file path (legacy usage)")
    args = parser.parse_args()

    if args.config is not None:
        cfg = _load_merged_config(args.config)
        config_file = args.config
    else:
        cfg = _load_merged_config(args.config_common, args.config_discovery)
        config_file = args.config_common

    run(cfg, config_file=config_file)


if __name__ == "__main__":
    main()
