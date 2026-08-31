"""
tuner.py -- Factor tuning pipeline (cross-platform, v2 rewrite)

Phase 1: Generate variants (structure mutation + parameter injection)
Phase 2: Parallel evaluation via FactorValidator (aligned with original mining)
Phase 3: Per-factor summary, keep best variant

Usage:
  python tuner.py
  python tuner.py --num-workers 4 --max-param-combos 10
  python tuner.py --no-structure-mutation
  python tuner.py --dry-run
"""


import sys
from pathlib import Path
# Allow running as a script from a subdirectory (e.g. `python mining/discovery.py`)
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

# -- CRITICAL: Set OpenMP threads BEFORE any C-extension import --
# IMPORTANT: Use os.environ[] (not setdefault) to FORCE override,
# because the server environment may already have OMP_NUM_THREADS set
# to a large value, which would cause LightGBM to spawn many threads
# and trigger fork-safety deadlocks.
import os as _os
_os.environ["OMP_NUM_THREADS"] = "1"
_os.environ["OPENBLAS_NUM_THREADS"] = "1"
_os.environ["MKL_NUM_THREADS"] = "1"
_os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # cap loky/sklearn parallelism
_os.environ["LIGHTGBM_TREE_LEARNER"] = "serial"  # force serial tree learner

import argparse
import hashlib
import json
import logging
import multiprocessing
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_IS_WINDOWS = sys.platform == "win32"
# Linux: use fork (fast, CoW memory sharing). Safe because the parent
# process NO LONGER imports lightgbm or initializes OpenMP — validators
# are created inside workers after fork, so there is no OpenMP state to
# corrupt during fork.
# Windows: use spawn (fork not available).
_MP_CTX = "spawn" if _IS_WINDOWS else "fork"

if _IS_WINDOWS:
    import io
    if getattr(sys.stdout, "encoding", None) != "utf-8" and getattr(sys.stdout, "buffer", None) is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# -- Logging --
log = logging.getLogger("factor_tuner")
log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
if not log.handlers:
    log.addHandler(h)


def _pct(v):
    return f"{v*100:.1f}%"


def _delta(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v*100:.1f}%"


# ===================================================================
# Feature extraction & column pruning (per-seed, one pruned window total)
# ===================================================================

def _extract_feature_names(code: str) -> Optional[set]:
    """Extract feature names from idx['name'] / idx.get('name') patterns.
    Returns set of feature name strings, or None if dynamic access prevents pruning.
    """
    features = set()
    for m in re.finditer(r"idx\.get\(\s*['\"]([^'\"]+)['\"]", code):
        features.add(m.group(1))
    for m in re.finditer(r"idx\[\s*['\"]([^'\"]+)['\"]", code):
        features.add(m.group(1))
    for m in re.finditer(r"names\s*=\s*\[([^\]]+)\]", code):
        for feat in re.finditer(r"['\"]([^'\"]+)['\"]", m.group(1)):
            features.add(feat.group(1))
    # Dynamic access → can't prune
    if re.search(r"idx\[\s*(?!['\"])", code):
        return None
    if re.search(r"idx\.get\(\s*(?!['\"])", code):
        return None
    if re.search(r"idx\[f['\"]|idx\.get\(f['\"]", code):
        return None
    return features if features else None


def _build_pruned_windows_and_idx(
    tw: np.ndarray, vw: np.ndarray,
    feature_names: set,
    flat_attributes: list
) -> tuple:
    """Build column-pruned windows with only the specified features.
    Returns (pruned_tw, pruned_vw, pruned_idx, pruned_attrs).
    pruned_idx maps feature_name → new column index (0..K-1).
    """
    orig_idx = {n: i for i, n in enumerate(flat_attributes)}
    existing = [(n, orig_idx[n]) for n in sorted(feature_names) if n in orig_idx]
    if not existing:
        return tw, vw, orig_idx, flat_attributes
    names, cols = zip(*existing)
    col_list = list(cols)
    pruned_tw = np.ascontiguousarray(tw[:, :, col_list])
    pruned_vw = np.ascontiguousarray(vw[:, :, col_list])
    pruned_idx = {name: i for i, name in enumerate(names)}
    pruned_attrs = list(names)
    return pruned_tw, pruned_vw, pruned_idx, pruned_attrs


# ===================================================================
# Auto worker count optimizer
# ===================================================================

def _compute_optimal_workers(
    requested_max: int, n_seeds: int, seq_length: int,
    train_windows: np.ndarray, val_windows: np.ndarray,
) -> int:
    """Compute optimal number of parallel workers based on CPU cores,
    available memory, and memory bandwidth (seq_length).
    requested_max=0 → auto; >0 → use as upper bound.
    """
    cpu_count = _os.cpu_count() or 8
    is_auto = (requested_max <= 0)
    cpu_limit = cpu_count if is_auto else min(requested_max, cpu_count)

    base = cpu_limit
    try:
        import psutil as _psutil
        vm = _psutil.virtual_memory()
        available_gb = vm.available / (1024**3)
    except Exception:
        available_gb = None

    if available_gb is not None and available_gb > 0:
        D = train_windows.shape[2]
        per_seed_features = min(12, D)
        ratio = per_seed_features / max(1, D)
        tw_gb = train_windows.nbytes / (1024**3)
        vw_gb = val_windows.nbytes / (1024**3)
        per_worker_gb = (tw_gb + vw_gb) * ratio + 1.0  # window copy + LGBM
        max_by_mem = int(available_gb * 0.50 / max(0.1, per_worker_gb))
        base = min(base, max_by_mem)

    if is_auto:
        if seq_length <= 5:    bw = 1.0
        elif seq_length <= 15: bw = 0.90
        elif seq_length <= 30: bw = 0.75
        elif seq_length <= 60: bw = 0.60
        else:                  bw = 0.45
        base = max(1, int(base * bw))

    return max(1, min(base, n_seeds))


class StatusPanel:
    """Background thread: prints progress panel every N seconds."""

    def __init__(self, interval=30):
        self.interval = interval
        self._thread = None
        self._stop = threading.Event()
        # External fields — updated by main thread
        self.variant_done = 0
        self.variant_total = 0
        self.variant_rate = 0.0
        self.eta_str = "..."
        self.seeds_total = 0
        self.seeds_done = 0
        self.seeds_improved = 0
        self.current_seq = 0
        self.current_label = ""
        self.recent_improvements = []  # list of (name, old_auc, new_auc)

    def start(self):
        if self._thread is not None: return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.wait(self.interval):
            self._print()

    def _print(self):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [""]
        lines.append(f"[{ts}] " + "=" * 59)
        pct = f" ({self.variant_done*100//max(1,self.variant_total)}%)" if self.variant_total else ""
        lines.append(f"  Seeds: {self.seeds_done}/{self.seeds_total} done"
                     f"  |  Improved: {self.seeds_improved}"
                     f"  |  Current seq_length={self.current_seq}")
        if self.variant_total > 0:
            lines.append(f"  Variants: {self.variant_done}/{self.variant_total}{pct}"
                         f"  ~{self.variant_rate:.2f}/s  ETA {self.eta_str}")
        if self.current_label:
            lines.append(f"  {self.current_label}")

        try:
            import psutil
            proc = psutil.Process()
            cpu = proc.cpu_percent(interval=0.1)
            total_threads = proc.num_threads()
            for child in proc.children(recursive=True):
                try: cpu += child.cpu_percent(interval=0.05); total_threads += child.num_threads()
                except psutil.NoSuchProcess: pass
            cores = psutil.cpu_count()
            mem = proc.memory_info().rss / 1024**2
            sysmem = psutil.virtual_memory()
            sysmem_used = (sysmem.total - sysmem.available) / 1024**3
            sysmem_total = sysmem.total / 1024**3
            lines.append(f"  CPU: {cpu/cores:.0f}% ({cpu/100:.1f}/{cores} cores)"
                         f"  |  RSS: {mem:.0f}MB"
                         f"  |  SysMem: {sysmem_used:.1f}/{sysmem_total:.0f}GB"
                         f"  |  Children: {len(proc.children())}  |  Threads: {total_threads}")
        except ImportError:
            pass

        if self.recent_improvements:
            lines.append(f"  -- Latest improvements ({len(self.recent_improvements)} total) --")
            for name, old, new in self.recent_improvements[-5:]:
                lines.append(f"  [+] {_pct(old)} -> {_pct(new)}  ({_delta(new-old)})  {name[:55]}")

        lines.append(f"[{ts}] " + "=" * 59)
        # Print directly to stdout for clean console display (no log timestamp prefix)
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()


# ===================================================================
# Module-level globals (fork mode on Linux — copy-on-write sharing).
# CRITICAL: NO lightgbm/FactorValidator instances here! The parent
# process must stay OpenMP-free so fork is safe. Workers import
# lightgbm and create validators from config dicts after fork.
# On Windows (spawn), these are ignored in favor of memmap + task dicts.
# ===================================================================
_train_windows: Optional[np.ndarray] = None
_val_windows: Optional[np.ndarray] = None
_train_labels: Optional[np.ndarray] = None
_val_labels: Optional[np.ndarray] = None
_flat_attributes: list = []
_validator_cfg_fast: Optional[dict] = None   # config dict, NOT instance
_validator_cfg_precise: Optional[dict] = None
_seed_generator_config: dict = {}
_seed_max_variants = 50
_seed_precise_top_k = 5
# ── memmap paths (Windows spawn only) ──
_tw_mmap_path: Optional[str] = None
_vw_mmap_path: Optional[str] = None
_tw_mmap_dtype: Optional[str] = None
_tw_mmap_shape: Optional[tuple] = None
_vw_mmap_dtype: Optional[str] = None
_vw_mmap_shape: Optional[tuple] = None


# ===================================================================
# Worker: compute factor values + FactorValidator evaluation
# ===================================================================

def _worker_eval(args: dict) -> Optional[dict]:
    """Evaluate one factor variant. Uses memmap-backed data from task args.

    Always runs in spawn subprocess — opens memmap files in read-only mode
    and creates its own FactorValidator instance from config.
    """
    try:
        code = args["factor"]["code"]
        flat_attrs = args.get("flat_attributes", [])
        name_to_idx = {n: i for i, n in enumerate(flat_attrs)}

        # Open memmap windows (read-only)
        tw = np.memmap(
            args["tw_path"], dtype=np.dtype(args["tw_dtype"]),
            mode="r", shape=tuple(args["tw_shape"]))
        vw = np.memmap(
            args["vw_path"], dtype=np.dtype(args["vw_dtype"]),
            mode="r", shape=tuple(args["vw_shape"]))
        tl = args.get("train_labels")
        vl = args.get("val_labels")

        # Build function from code
        func_code = "def _f(windows, idx, np):\n"
        func_code += "    with np.errstate(invalid='ignore', divide='ignore'):\n"
        for line in code.strip().split("\n"):
            func_code += f"        {line}\n"
        func_code += "        return _result\n"
        ns = {"__builtins__": {
            "abs": abs, "min": min, "max": max, "sum": sum,
            "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "list": list, "dict": dict, "tuple": tuple,
            "float": float, "int": int, "bool": bool, "str": str,
            "isinstance": isinstance, "hasattr": hasattr,
            "print": print, "__import__": __import__,
        }}
        exec(compile(func_code, "<factor>", "exec"), ns)
        func = ns["_f"]

        # Compute factor values
        train_fv = np.asarray(func(tw, name_to_idx, np), dtype=np.float32)
        val_fv = np.asarray(func(vw, name_to_idx, np), dtype=np.float32)
        if np.isnan(train_fv).mean() > 0.8:
            return None

        # Create validator from config (spawn-safe, no pickled LGBM state)
        vcfg = args.get("validator_cfg")
        if vcfg is not None:
            from src.validator import FactorValidator
            vld = FactorValidator(vcfg)
        else:
            return None

        # Validate via FactorValidator
        result = vld.validate_single_holdout(train_fv, tl, val_fv, vl)
        if not result.get("valid"):
            return None

        return {
            "best_auc": float(result["best_auc"]),
            "best_f1": float(result.get("best_f1", 0)),
            "best_class": result.get("best_class", ""),
            "valid_classes": result.get("valid_classes", []),
            "per_class": result.get("per_class", {}),
            "n_train": result.get("n_train", 0),
            "n_val": result.get("n_val", 0),
        }
    except Exception:
        return None


# ===================================================================
# Per-seed pipeline worker (replaces Phase 1-3 global barrier)
# ===================================================================

def _process_seed(task: dict) -> dict:
    """Process one seed end-to-end: generate variants → fast-eval → precise-eval.

    Data access strategy:
    - Fork (Linux):  inherits parent memory via CoW → use module globals
    - Spawn (Windows): opens memmap files → use task dict paths

    FactorValidator instances are ALWAYS created inside the worker from
    config dicts — the parent process never imports lightgbm, so the
    OpenMP runtime is never initialized pre-fork.

    Returns dict with: seed_name, seed_auc, best_auc, improved, best_variant,
    n_variants, n_evaluated, n_precise, n_failed.
    """
    from src.validator import FactorValidator

    seed = task["seed"]
    seed_name = seed.get("name", "?")
    seed_auc = float(seed.get("best_auc", 0) or 0)

    # ── data source: globals (fork CoW) or task dict (spawn memmap) ──
    if _IS_WINDOWS:
        # Spawn: open memmap files
        tw = np.memmap(
            task["tw_path"], dtype=np.dtype(task["tw_dtype"]),
            mode="r", shape=tuple(task["tw_shape"]))
        vw = np.memmap(
            task["vw_path"], dtype=np.dtype(task["vw_dtype"]),
            mode="r", shape=tuple(task["vw_shape"]))
        tl = task["train_labels"]
        vl = task["val_labels"]
        flat_attrs = task["flat_attributes"]
        vcfg_fast = task["validator_cfg_fast"]
        vcfg_precise = task["validator_cfg_precise"]
    else:
        # Fork: inherit via CoW from module globals
        tw = _train_windows
        vw = _val_windows
        tl = _train_labels
        vl = _val_labels
        flat_attrs = _flat_attributes
        vcfg_fast = _validator_cfg_fast
        vcfg_precise = _validator_cfg_precise

    # ── create validators from config (safe: no LGBM in parent) ──
    vfast = FactorValidator(vcfg_fast)
    vprecise = FactorValidator(vcfg_precise)

    # ── config ──
    gen_cfg = task.get("generator_config") or {}
    enable_structure = gen_cfg.get("structure", True)
    max_param_combos = gen_cfg.get("max_param_combos", 20)
    max_variants = task.get("max_variants_per_seed", 50)
    precise_top_k = task.get("precise_top_k", 5)

    name_to_idx = {n: i for i, n in enumerate(flat_attrs)}

    # ── helper: eval one factor ──
    def _eval_one(factor, validator):
        try:
            code = factor["code"]
            func_code = "def _f(windows, idx, np):\n"
            func_code += "    with np.errstate(invalid='ignore', divide='ignore'):\n"
            for line in code.strip().split("\n"):
                func_code += f"        {line}\n"
            func_code += "        return _result\n"
            ns = {"__builtins__": {
                "abs": abs, "min": min, "max": max, "sum": sum,
                "len": len, "range": range, "enumerate": enumerate,
                "zip": zip, "list": list, "dict": dict, "tuple": tuple,
                "float": float, "int": int, "bool": bool, "str": str,
                "isinstance": isinstance, "hasattr": hasattr,
                "print": print, "__import__": __import__,
            }}
            exec(compile(func_code, "<factor>", "exec"), ns)
            func = ns["_f"]

            train_fv = np.asarray(func(tw, name_to_idx, np), dtype=np.float32)
            val_fv   = np.asarray(func(vw, name_to_idx, np), dtype=np.float32)
            if np.isnan(train_fv).mean() > 0.8:
                return None

            result = validator.validate_single_holdout(train_fv, tl, val_fv, vl)
            if not result.get("valid"):
                return None

            return {
                "best_auc": float(result["best_auc"]),
                "best_f1": float(result.get("best_f1", 0)),
                "best_class": result.get("best_class", ""),
                "valid_classes": result.get("valid_classes", []),
                "per_class": result.get("per_class", {}),
                "n_train": result.get("n_train", 0),
                "n_val": result.get("n_val", 0),
            }
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Generate all variants (structure + param + combined)
    # ═══════════════════════════════════════════════════════════════
    generator = VariantGenerator(enable_structure, max_param_combos)
    variants = generator.generate(seed)[:max_variants]

    # LLM param injection (mirrors original Phase 1 logic)
    lparams = _extract_params(seed.get("code", ""))
    apc = None
    if not lparams:
        apc, lparams = _auto_param(seed.get("code", ""))
    pcombos = _gen_param_combos(lparams, max_param_combos * 2) if lparams else []

    extra_tasks = []
    for pd in pcombos:
        if all(abs(pd[p["name"]] - p["default"]) < 0.0001 for p in lparams):
            continue
        bc = apc if apc else seed.get("code", "")
        nc = _apply_params(bc, pd)
        if nc == seed.get("code", ""):
            continue
        ns = dict(seed)
        ns["code"] = nc
        ns["_mutation"] = "llm_param"
        ns["_param_label"] = ",".join(f"{p['name']}={pd[p['name']]:.4g}" for p in lparams)
        ns["name"] = f"{seed_name}_llmp"
        extra_tasks.append(ns)

    # combined: structure × param on top-5 structure variants
    if lparams and variants:
        cpc = _gen_param_combos(lparams, 5)
        for v in variants[:5]:
            if v.get("_mutation", "") == "seed":
                continue
            for pd in cpc:
                if all(abs(pd[p["name"]] - p["default"]) < 0.0001 for p in lparams):
                    continue
                nc = _apply_params(v["code"], pd)
                if nc == v["code"]:
                    continue
                cv = dict(v)
                cv["code"] = nc
                cv["_mutation"] = v.get("_mutation", "") + "+llmp"
                cv["_param_label"] = ",".join(f"{p['name']}={pd[p['name']]:.4g}" for p in lparams)
                cv["name"] = v["name"] + "_llmp"
                extra_tasks.append(cv)

    all_variants = list(variants) + extra_tasks

    # ═══════════════════════════════════════════════════════════════
    # Column pruning (per-seed): extract union of features across
    # all variants → build ONE pruned window → all variants share it.
    # _eval_one closure captures tw/vw/name_to_idx by reference,
    # so reassigning them here is all that's needed.
    # ═══════════════════════════════════════════════════════════════
    all_features = set()
    can_prune = True
    for v in all_variants:
        feats = _extract_feature_names(v["code"])
        if feats is None:
            can_prune = False
            break
        all_features.update(feats)

    if can_prune and all_features:
        try:
            ptw, pvw, pidx, pattrs = _build_pruned_windows_and_idx(
                tw, vw, all_features, flat_attrs)
            tw, vw, name_to_idx = ptw, pvw, pidx  # closure sees new values
            flat_attrs = pattrs
        except Exception:
            pass  # pruning failed → fall back to full window

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Fast-eval all variants
    # ═══════════════════════════════════════════════════════════════
    fast_results = []
    for v in all_variants:
        r = _eval_one(v, vfast)
        if r is not None:
            r["_factor"] = v
            r["_mutation"] = v.get("_mutation", "")
            r["_param_label"] = v.get("_param_label", "")
            r["_seed_auc"] = seed_auc
            r["_auc_improvement"] = round(r["best_auc"] - seed_auc, 4)
            fast_results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # Step 3-4: Pick top-K → precise-eval
    # ═══════════════════════════════════════════════════════════════
    fast_results.sort(key=lambda r: r["best_auc"], reverse=True)
    topk = fast_results[:precise_top_k]

    best_auc = seed_auc
    best_variant = None
    precise_count = 0
    for fr in topk:
        r = _eval_one(fr["_factor"], vprecise)
        precise_count += 1
        if r is not None:
            r["_factor"] = fr["_factor"]
            r["_mutation"] = fr["_mutation"]
            r["_param_label"] = fr["_param_label"]
            r["_seed_auc"] = seed_auc
            r["_auc_improvement"] = round(r["best_auc"] - seed_auc, 4)
            if r["best_auc"] > best_auc:
                best_auc = r["best_auc"]
                best_variant = fr["_factor"]
        else:
            # precise eval failed → fall back to fast result
            if fr["best_auc"] > best_auc:
                best_auc = fr["best_auc"]
                best_variant = fr["_factor"]

    improved = best_auc > seed_auc + 0.002

    return {
        "seed_name": seed_name,
        "seed_auc": seed_auc,
        "best_auc": best_auc,
        "improved": improved,
        "best_variant": best_variant,
        "n_variants": len(all_variants),
        "n_evaluated": len(fast_results),
        "n_precise": precise_count,
        "n_failed": len(all_variants) - len(fast_results),
    }


# ===================================================================
# Variant generation helpers
# ===================================================================

_SAFE_NP = {
    "np.mean", "np.std", "np.var", "np.min", "np.max", "np.median",
    "np.percentile", "np.abs", "np.sqrt", "np.square", "np.log1p",
    "np.clip", "np.sum", "np.prod", "np.polyfit", "np.arange",
    "np.nan", "np.full", "np.isnan", "np.where", "np.zeros",
    "np.ones", "np.exp", "np.log", "np.sign", "np.sin", "np.cos",
    "np.floor", "np.ceil", "np.round",
    "np.all", "np.any", "np.isfinite", "np.isinf", "np.array",
    "np.concatenate", "np.stack", "np.linspace", "np.diff",
    "np.power", "np.true_divide", "np.maximum", "np.minimum",
    "np.argsort", "np.argmax", "np.argmin", "np.sort",
}


def _code_valid(code):
    if code.count("(") != code.count(")"):
        return False
    if "return" not in code and "_result" not in code:
        return False
    if "idx.get" not in code and "idx[" not in code and "names=" not in code:
        return False
    for m in re.finditer(r"np\.(\w+)\s*\(", code):
        if f"np.{m.group(1)}" not in _SAFE_NP:
            return False
    return True


class _AggMutator:
    MAP = {"mean": ["std","min","max","median"], "std": ["mean","var"],
           "min": ["max","mean","median"], "max": ["min","mean","median"],
           "median": ["mean","percentile"]}
    def mutate(self, code, mode):
        vs = []
        for old, news in self.MAP.items():
            if f"np.{old}(" not in code: continue
            for new in news:
                if new == "percentile":
                    for p in [10,25,75,90]:
                        nc = re.sub(rf"np\.{old}\(([^)]+)\)", rf"np.percentile(\1, {p})", code)
                        vs.append((nc, f"{old}2pctl{p}", f"np.{old}->percentile({p})"))
                elif new == "var":
                    vs.append((code.replace(f"np.{old}(", "np.var("), f"{old}2var", f"np.{old}->var"))
                else:
                    vs.append((code.replace(f"np.{old}(", f"np.{new}("), f"{old}2{new}", f"np.{old}->{new}"))
        return vs


class _TempMutator:
    def mutate(self, code, mode):
        vs = []
        for m in re.finditer(r"(window\[-1,\s*(\w+)\]\s*-\s*window\[0,\s*\2\])", code):
            old, idx = m.group(1), m.group(2)
            nc = code.replace(old, f"np.polyfit(np.arange(window.shape[0]), window[:, {idx}], 1)[0]")
            vs.append((nc, "diff2trend", "diff->polyfit trend"))
            break
        for m in re.finditer(r"np\.mean\(window\[:\s*,\s*:\s*,\s*(\w+)\]", code):
            old, idx = m.group(0), m.group(1)
            nc = code.replace(old, f"np.std(window[:, :, {idx}])/(np.mean(window[:, :, {idx}])+1e-8)")
            vs.append((nc, "mean2vol", "mean->volatility"))
            break
        return vs


class _CrossMutator:
    def mutate(self, code, mode):
        vs = []
        for m in re.finditer(r"(\w+)\s*/\s*\((\w+)\s*\+\s*1e-?\d+\)", code):
            a, b, old = m.group(1), m.group(2), m.group(0)
            vs.append((code.replace(old, f"np.abs({a} - {b})"), "ratio2diff", f"{a}/{b}->|{a}-{b}|"))
            vs.append((code.replace(old, f"{a} * {b}"), "ratio2prod", f"{a}/{b}->{a}*{b}"))
            break
        for m in re.finditer(r"(\w+)\s*\*\s*(\w+)", code):
            a, b = m.group(1), m.group(2)
            if a in ("np","float") or b in ("np","float"): continue
            ls = code.rfind("\n", 0, m.start()) + 1
            if any(kw in code[ls:m.end()+20] for kw in ("return","_result")):
                vs.append((code.replace(m.group(0), f"np.abs({a} - {b})", 1), "prod2diff", f"{a}*{b}->|{a}-{b}|"))
                break
        return vs


def _replace_call(code, func_name, transform):
    prefix = func_name + "("
    idx = code.find(prefix)
    if idx < 0: return code
    start, depth, pos = idx + len(prefix), 1, idx + len(prefix)
    while pos < len(code) and depth > 0:
        ch = code[pos]
        if ch == "(": depth += 1
        elif ch == ")": depth -= 1
        if depth == 0: return code[:idx] + transform(code[start:pos]) + code[pos+1:]
        pos += 1
    return code


class _NonlinMutator:
    XFORMS = [("np.abs","abs"), ("np.square","square"), ("np.sqrt","sqrt"), ("np.log1p","log1p")]
    def mutate(self, code, mode):
        vs = []
        has = any(t[0] in code for t in self.XFORMS)
        if not has:
            for pat in [r"return\s+float\(\s*(.+?)\s*\)\s*$", r"_result\s*=\s*(.+?)\s*$"]:
                m = re.search(pat, code, re.MULTILINE)
                if m:
                    rhs = m.group(1).strip()
                    for tf, ts in [("np.abs","abs"), ("np.square","square"), ("np.sqrt","sqrt")]:
                        inner = f"np.abs({rhs})" if tf == "np.sqrt" else rhs
                        nc = code.replace(f"return float({rhs})", f"return float({tf}({inner}))") if "return" in pat else code.replace(f"_result = {rhs}", f"_result = {tf}({inner})")
                        if nc != code: vs.append((nc, f"add_{ts}", f"wrap in {tf}()"))
                    break
        else:
            for old_tf, old_s in self.XFORMS:
                if old_tf not in code: continue
                for new_tf, new_s in self.XFORMS:
                    if old_tf == new_tf: continue
                    nc = _replace_call(code, old_tf, lambda inner, nt=new_tf: f"{nt}({inner})" if nt != "np.sqrt" else f"np.sqrt(np.abs({inner}))")
                    if nc != code: vs.append((nc, f"{old_s}2{new_s}", f"{old_tf}->{new_tf}"))
                nc = _replace_call(code, old_tf, lambda s: s)
                if nc != code: vs.append((nc, f"rm_{old_s}", f"remove {old_tf}()"))
        return vs


class _ConstMutator:
    SCALES = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0]
    SKIP = {0, 1, -1, 5, 10, 25, 50, 75, 90, 100}
    def mutate(self, code, mode):
        vs = []
        if "if " in code:
            cs, ss = code[code.find("if "):], code[:code.find("if ")]
        else:
            cs, ss = code, ""
        for m in re.finditer(r"(?<![\w.>])(\d+\.?\d*)(?![\w'])(?![.\d]*\s*(?:\)|,|idx|window))", cs):
            val = float(m.group(1))
            if val in self.SKIP: continue
            if self._is_slice(cs, m.start(), m.end()): continue
            for s in self.SCALES:
                nv = val * s
                ns = f"{nv:.1f}".rstrip("0").rstrip(".") if abs(nv-round(nv,1))<0.001 else f"{nv:.4f}".rstrip("0").rstrip(".")
                vs.append((ss + cs.replace(m.group(1), ns, 1), f"const_{m.group(1)}_x{s}", f"{m.group(1)}->{ns}"))
        return vs
    @staticmethod
    def _is_slice(text, start, end):
        depth = 0
        for i in range(end-1, -1, -1):
            if text[i] == ']': depth += 1
            elif text[i] == '[':
                if depth > 0: depth -= 1
                else: return True
        return False


class _ParamInjector:
    def inject(self, code, mode, max_combos=20):
        rvar = "_result" if mode == "batch" else "result"
        m = re.search(rf"{rvar}\s*=\s*(.+)", code)
        if not m:
            m = re.search(r"return\s+float\(\s*(.+)\s*\)", code)
        if not m: return []
        rhs = m.group(1).strip()
        terms = self._split(rhs)
        return self._single(code, rhs, rvar, max_combos) if len(terms) <= 1 else self._multi(code, rhs, terms, rvar, max_combos)

    def _single(self, code, rhs, rvar, max_combos):
        vs = []
        pv = re.findall(r'\(1\.0\s*-\s*(\w+)\)', rhs)
        if len(pv) >= 2:
            rng = np.random.default_rng(42+len(pv))
            seen = set()
            for _ in range(max_combos):
                combo = tuple(float(rng.choice([0.5,1.0,2.0])) for _ in pv)
                if combo in seen or all(abs(c-1)<0.001 for c in combo): continue
                seen.add(combo)
                nr = rhs
                for vi, vn in enumerate(pv):
                    nr = nr.replace(f'(1.0 - {vn})', f'(1.0 - {combo[vi]:.4g} * {vn})', 1)
                if nr != rhs:
                    nc = self._replace(code, rhs, nr, rvar)
                    if nc != code:
                        vs.append((nc, f'w=[{",".join(f"{c:.3g}" for c in combo)}]'))
        return vs

    def _multi(self, code, rhs, terms, rvar, max_combos):
        import itertools
        nt = len([t for t in terms if t[1].strip()])
        if nt <= 0: return []
        grids = [[0.5,1.0,1.5,2.0],[0.5,1.0,1.5,2.0]] if nt==2 else ([[0.5,1.0,2.0]]*3 if nt==3 else [[0.5,1.0,2.0]]*nt)
        vs = []
        for combo in list(itertools.product(*grids))[:max_combos]:
            parts, pi = [], 0
            for sign, term in terms:
                term = term.strip()
                if not term: continue
                pv = combo[pi] if pi < len(combo) else 1.0; pi += 1
                s = f" {sign} " if sign else ""
                parts.append(f"{s}({term})" if abs(pv-1)<0.001 else f"{s}({pv:.4g})*({term})")
            nr = "".join(parts).lstrip(" +")
            nc = re.sub(rf"{rvar}\s*=\s*{re.escape(rhs)}", f"{rvar} = {nr}", code)
            if nc == code: nc = code.replace(f"{rvar} = {rhs}", f"{rvar} = {nr}")
            if nc != code:
                vs.append((nc, f'params={",".join(f"{v:.3g}" for v in combo)}'))
        return vs

    @staticmethod
    def _split(rhs):
        terms, depth, cur, sign = [], 0, "", ""
        for i, ch in enumerate(rhs):
            if ch == "(": depth += 1; cur += ch
            elif ch == ")": depth -= 1; cur += ch
            elif depth == 0 and ch in "+-":
                if ch == "-" and (i==0 or rhs[i-1] in " eE"): cur += ch; continue
                if cur.strip(): terms.append((sign, cur.strip()))
                sign, cur = "+" if ch == "+" else "-", ""
            else: cur += ch
        if cur.strip(): terms.append((sign, cur.strip()))
        return terms

    @staticmethod
    def _replace(code, old, new, rvar):
        for op, np_ in [(f"return float({old})", f"return float({new})"), (f"{rvar} = {old}", f"{rvar} = {new}")]:
            if op in code: return code.replace(op, np_, 1)
        return code if code.count(old) != 1 else code.replace(old, new, 1)


# -- LLM param helpers --
def _extract_params(code):
    params = []
    for m in re.finditer(r'^(_P\d+)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*(?:#.*)?$', code, re.MULTILINE):
        name, default = m.group(1), float(m.group(2))
        if default == 1.0:
            kind, sr = "weight", [0.05,0.1,0.2,0.35,0.5,0.7,1.0,1.5,2.0,3.0,5.0]
        elif default >= 2.0:
            kind, sr = "scale", sorted(set([max(0.05,default*0.1),default*0.3,default*0.6,default*0.8,default,default*1.25,default*1.5,default*2.0,default*3.0]))
        elif default <= 0.15:
            kind, sr = "threshold_small", sorted(set([0.01,0.03,0.05,0.08,max(0.01,default*0.5),default,min(0.95,default*2),min(0.95,default*3)]))
        else:
            kind, sr = "threshold", sorted(set([max(0.01,default*0.2),default*0.5,default*0.75,default,min(0.95,default*1.5),min(0.95,default*2)]))
        params.append({"name": name, "default": default, "kind": kind, "range": sorted(set(round(v,4) for v in sr))})
    return params


def _auto_param(code):
    lines = code.strip().split("\n")
    fstart = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"): fstart = i+1; continue
        if re.search(r"idx\.get\(|idx\[", s) and "=" in s: fstart = i+1; continue
        if re.search(r"if\s+.*<\s*0\s*:\s*return\s+np\.nan", s): fstart = i+1; continue
        if s.startswith("return np.nan"): fstart = i+1; continue
        if re.match(r"^_P\d+\s*=", s): fstart = i+1; continue
        break
    header, formula_lines = lines[:fstart], lines[fstart:]
    ftext = "\n".join(formula_lines)
    candidates = []
    for m in re.finditer(r'(?<![\w.])(\d+\.\d+)\s*\*\s*([a-zA-Z_]\w*)', ftext):
        val = float(m.group(1))
        if val in (0.0,1.0,2.0) and not re.search(r'feat|val|score|factor', m.group(2), re.I): continue
        candidates.append({"value": val, "start": m.start(1), "end": m.end(1), "kind": "weight"})
    for m in re.finditer(r'([a-zA-Z_]\w*)\s*([><]=?)\s*(\d+\.\d+)', ftext):
        val = float(m.group(3))
        if val == 0.0 and m.group(2) in ("<","<="): continue
        prefix = ftext[max(0,m.start()-30):m.start()]
        if re.search(r'\b(?:min|max|clip|percentile)\s*\([^)]*$', prefix): continue
        candidates.append({"value": val, "start": m.start(3), "end": m.end(3), "kind": "threshold"})
    seen = set(); candidates = [c for c in candidates if not (c["start"] in seen or seen.add(c["start"]))]
    if not candidates: return code, []
    candidates.sort(key=lambda c: c["start"], reverse=True)
    nf, pinfo = ftext, []
    for i, c in enumerate(candidates):
        pname = f"_P{i}"; nf = nf[:c["start"]] + pname + nf[c["end"]:]
        val, kind = c["value"], c["kind"]
        sr = [0.05,0.1,0.2,0.35,0.5,0.7,1.0,1.5,2.0,3.0,5.0] if kind=="weight" else sorted(set([max(0.01,val*0.2),val*0.5,val*0.75,val,min(0.95,val*1.5),min(0.95,val*2)]))
        pinfo.append({"name": pname, "default": val, "kind": kind, "range": [round(v,4) for v in sr]})
    pinfo.reverse()
    pdefs = [f"{p['name']} = {p['default']:.6g}" + (".0" if '.' not in f"{p['default']:.6g}" else "") + f"  # auto: {p['kind']}" for p in pinfo]
    ipos = 0
    for i in range(len(header)-1, -1, -1):
        if re.search(r"idx\.get\(|idx\[", header[i]) and "=" in header[i]: ipos = i+1; break
    return "\n".join(header[:ipos] + pdefs + header[ipos:] + [nf]), pinfo


def _gen_param_combos(params, max_combos=100):
    import itertools
    n = len(params)
    if n == 0: return [{}]
    if n <= 2:
        grids = [p["range"][:7] for p in params]; combos = list(itertools.product(*grids))
    elif n <= 4:
        grids = [sorted(set([p["default"],p["range"][0],p["range"][-1],p["range"][len(p["range"])//2]])) for p in params]
        combos = list(itertools.product(*grids))
    else:
        centers = tuple(p["default"] for p in params)
        combos = [centers]
        for i, p in enumerate(params):
            for v in p["range"][:5]: t = list(centers); t[i] = v; combos.append(tuple(t))
        rng = np.random.default_rng(42+n)
        for _ in range(min(max_combos, 200)):
            t = tuple(float(rng.choice(p["range"])) for p in params)
            if t not in combos: combos.append(t)
    result = []
    for combo in combos[:max_combos]:
        result.append({params[i]["name"]: float(v) for i, v in enumerate(combo)})
    return result


def _apply_params(code, param_dict):
    result = code
    for pname, pval in param_dict.items():
        nv = f"{pval:.6g}"
        if '.' not in nv: nv += '.0'
        result = re.sub(rf'^{re.escape(pname)}\s*=\s*[\d.]+(?:[eE][+-]?\d+)?(\s*.*)?$', rf'{pname} = {nv}\g<1>', result, flags=re.MULTILINE)
    return result


class VariantGenerator:
    def __init__(self, structure=True, max_param_combos=20):
        self.mutators = [_AggMutator(), _TempMutator(), _CrossMutator(), _NonlinMutator(), _ConstMutator()] if structure else []
        self.injector = _ParamInjector()
        self.max_param_combos = max_param_combos

    def generate(self, factor):
        code = factor["code"]
        mode = factor.get("mode", "row")
        seen = set()
        variants = []

        def _add(c, mname, mdesc, plabel=""):
            if not _code_valid(c): return
            ch = hashlib.md5(c.encode()).hexdigest()
            if ch in seen: return
            seen.add(ch)
            v = dict(factor)
            v["code"] = c; v["_seed_name"] = factor["name"]
            v["_mutation"] = mname; v["_param_label"] = plabel
            sfx = mname if mname != "seed" else "tuned"
            if plabel: sfx += f"_{plabel.replace('=','').replace(',','_')[:30]}"
            v["name"] = f"{factor['name']}_{sfx}"
            v["description"] = f"{factor.get('description','')} [{mdesc}]" + (f" [{plabel}]" if plabel else "")
            variants.append(v)

        structs = [(code, "seed", "Original")]
        for mut in self.mutators:
            for nc, mn, md in mut.mutate(code, mode):
                if hashlib.md5(nc.encode()).hexdigest() not in seen:
                    structs.append((nc, mn, md))

        for sc, mn, md in structs:
            baked = self.injector.inject(sc, mode, max_combos=self.max_param_combos)
            if baked:
                for bc, pl in baked:
                    _add(bc, mn, md, pl)
            else:
                _add(sc, mn, md)

        return variants


# ===================================================================
# Main pipeline
# ===================================================================

def tune_factors(
    seed_factors, kp_train, labels_train, kp_val, labels_val, flat_attributes,
    num_workers=4, enable_structure=True, max_param_combos=20,
    max_variants_per_seed=50, validation_cfg=None,
    output_path="memory/tuned_factors.json",
    progress_path="memory/tuner_progress.json",
    completed_set=None, initial_results=None,
    fast_trees=50, precise_trees=200, precise_top_k=5,
):
    import copy as _copy
    vcfg = {"validation": {"min_auc": 0.65, "min_f1": 0.50, "early_stopping_rounds": 20, "use_gpu": False,
            "lgbm_params": {"n_estimators": precise_trees, "max_depth": 4, "learning_rate": 0.05, "num_leaves": 31,
                             "random_state": 42, "n_jobs": 1}}}
    if validation_cfg and "validation" in validation_cfg:
        vcfg["validation"].update(validation_cfg["validation"])
    vcfg["validation"].setdefault("lgbm_params", {})
    vcfg["validation"]["lgbm_params"].setdefault("n_jobs", 1)

    # Fast-stage validator config with fewer trees for screening
    vcfg_fast = _copy.deepcopy(vcfg)
    vcfg_fast["validation"]["lgbm_params"]["n_estimators"] = fast_trees

    completed = set(completed_set) if completed_set else set()
    all_tuned = list(initial_results) if initial_results else []
    total_improved = sum(1 for f in all_tuned if f.get("_auc_improvement", 0) > 0.001)

    log.info(f"Seeds: {len(seed_factors)} ({len(completed)} done, {len(seed_factors)-len(completed)} todo)")
    log.info(f"LGBM: fast={fast_trees}t, precise={precise_trees}t, top_k={precise_top_k}")
    log.info(f"Workers: {num_workers}, structure: {enable_structure}, param_combos: {max_param_combos}")

    groups = defaultdict(list)
    for f in seed_factors:
        groups[f.get("seq_length", 1)].append(f)

    t_start = time.time()

    # Background status panel
    panel = StatusPanel(interval=30)
    panel.seeds_total = len(seed_factors)
    panel.seeds_done = len(all_tuned)
    panel.seeds_improved = total_improved
    panel.start()

    for sl in sorted(groups):
        group = groups[sl]
        n_todo = len(group) - sum(1 for s in group if s.get("name","") in completed)
        if n_todo == 0:
            log.info(f"  seq={sl}: all {len(group)} done, skip")
            continue

        panel.current_seq = sl
        panel.current_label = f"Building windows for seq={sl}..."
        log.info(f"--- seq={sl} ({n_todo} todo / {len(group)} total) ---")

        # Build windows
        t0 = time.time()
        T, D = kp_train.shape
        if sl == 1:
            tw = kp_train[:, np.newaxis, :].astype(np.float32).copy()
            vw = kp_val[:, np.newaxis, :].astype(np.float32).copy()
        else:
            tw = np.zeros((T, sl, D), dtype=np.float32); tw[:sl-1] = np.nan
            from numpy.lib.stride_tricks import sliding_window_view
            sw = sliding_window_view(kp_train, (sl, D)).reshape(-1, sl, D)
            tw[sl-1:] = sw[:, :, :]
            Tv = kp_val.shape[0]
            vw = np.zeros((Tv, sl, D), dtype=np.float32); vw[:sl-1] = np.nan
            swv = sliding_window_view(kp_val, (sl, D)).reshape(-1, sl, D)
            vw[sl-1:] = swv[:, :, :]
        log.info(f"  Windows: train{tw.shape} val{vw.shape} ({time.time()-t0:.1f}s)")

        # ── Fork mode (Linux): set globals for CoW sharing ──
        # ── Spawn mode (Windows): write memmap temp files ──
        global _train_windows, _val_windows, _train_labels, _val_labels
        global _flat_attributes, _validator_cfg_fast, _validator_cfg_precise
        global _seed_generator_config, _seed_max_variants, _seed_precise_top_k
        global _tw_mmap_path, _vw_mmap_path, _tw_mmap_dtype, _tw_mmap_shape
        global _vw_mmap_dtype, _vw_mmap_shape

        if not _IS_WINDOWS:
            # Fork: set globals — children inherit via CoW (fast, zero-copy)
            _train_windows = tw; _val_windows = vw
            _train_labels = labels_train; _val_labels = labels_val
            _flat_attributes = list(flat_attributes)
            _validator_cfg_fast = vcfg_fast
            _validator_cfg_precise = vcfg
            tw_path = vw_path = None
        else:
            # Spawn: write memmap files
            import tempfile as _tempfile
            _tw_tmp = _tempfile.NamedTemporaryFile(
                suffix=".dat", prefix="tuner_tw_", delete=False)
            _vw_tmp = _tempfile.NamedTemporaryFile(
                suffix=".dat", prefix="tuner_vw_", delete=False)
            tw_path = _tw_tmp.name
            vw_path = _vw_tmp.name
            _tw_tmp.close()
            _vw_tmp.close()
            _tw_mmap_path = tw_path; _vw_mmap_path = vw_path
            _tw_mmap_dtype = tw.dtype.str; _tw_mmap_shape = tw.shape
            _vw_mmap_dtype = vw.dtype.str; _vw_mmap_shape = vw.shape
            try:
                tw_mmap = np.memmap(tw_path, dtype=tw.dtype, mode="w+", shape=tw.shape)
                tw_mmap[:] = tw[:]; tw_mmap.flush(); del tw_mmap
                vw_mmap = np.memmap(vw_path, dtype=vw.dtype, mode="w+", shape=vw.shape)
                vw_mmap[:] = vw[:]; vw_mmap.flush(); del vw_mmap
            except Exception:
                for p in (tw_path, vw_path):
                    try: _os.remove(p)
                    except Exception: pass
                raise
            log.info(f"  memmap: train{tw.shape} ({tw.nbytes/1024**2:.0f}MB) -> {tw_path}")
            log.info(f"  memmap: val{vw.shape} ({vw.nbytes/1024**2:.0f}MB) -> {vw_path}")

        _seed_generator_config = {"structure": enable_structure, "max_param_combos": max_param_combos}
        _seed_max_variants = max_variants_per_seed
        _seed_precise_top_k = precise_top_k

        # ── Build per-seed tasks ──
        seed_tasks = []
        for seed in group:
            sname = seed.get("name", "?")
            if sname in completed:
                continue
            task = {
                "seed": seed,
                "max_variants_per_seed": max_variants_per_seed,
                "precise_top_k": precise_top_k,
                "generator_config": {"structure": enable_structure, "max_param_combos": max_param_combos},
            }
            if _IS_WINDOWS:
                # Spawn: pass everything explicitly via task dict
                task.update({
                    "tw_path": tw_path, "tw_dtype": tw.dtype.str, "tw_shape": tw.shape,
                    "vw_path": vw_path, "vw_dtype": vw.dtype.str, "vw_shape": vw.shape,
                    "train_labels": labels_train, "val_labels": labels_val,
                    "flat_attributes": list(flat_attributes),
                    "validator_cfg_fast": vcfg_fast,
                    "validator_cfg_precise": vcfg,
                })
            seed_tasks.append(task)

        n_seeds = len(seed_tasks)
        if n_seeds == 0:
            continue

        # ─── Adaptive optimal worker count ──────────────────────────
        nw = _compute_optimal_workers(num_workers, n_seeds, sl, tw, vw)
        mode = "auto" if num_workers <= 0 else f"cap={num_workers}"
        t2 = time.time()

        # Estimate per-worker memory for logging
        D = tw.shape[2]
        ratio = min(12, D) / max(1, D)
        est_per_worker_gb = (tw.nbytes + vw.nbytes) * ratio / (1024**3) + 1.0
        try:
            import psutil as _psutil
            avail_gb = _psutil.virtual_memory().available / (1024**3)
        except Exception:
            avail_gb = 0

        # Compute safe batch size (limit concurrent startup memory spike)
        per_worker_safe_gb = est_per_worker_gb * 1.5
        safe_concurrent = max(1, int((avail_gb * 0.40) / max(0.1, per_worker_safe_gb))) if avail_gb > 0 else 16
        batch_size = min(nw, safe_concurrent)
        n_batches = (n_seeds + batch_size - 1) // batch_size

        log.info(f"  seq={sl}: {n_seeds} seeds → {nw} workers ({mode}), "
                 f"~{est_per_worker_gb:.1f}GB/worker, batches={n_batches}×≤{batch_size}, "
                 f"mem_free={avail_gb:.0f}GB")
        panel.variant_total = n_seeds
        panel.variant_done = 0
        panel.variant_rate = 0
        panel.eta_str = "..."
        panel.current_label = f"seq={sl}: {nw} workers ({mode}), {n_batches} batches"

        # ═══════════════════════════════════════════════════════════
        # Batched parallel pipeline — limits startup OOM spikes
        # ═══════════════════════════════════════════════════════════
        n_done = 0
        _last_eta = 0.0
        for batch_idx in range(0, n_seeds, batch_size):
            batch_tasks = seed_tasks[batch_idx:batch_idx + batch_size]
            n_batch_workers = min(batch_size, len(batch_tasks))
            t_batch = time.time()

            with ProcessPoolExecutor(max_workers=n_batch_workers,
                                     mp_context=multiprocessing.get_context(_MP_CTX)) as pool:
                fmap = {}
                for t in batch_tasks:
                    fut = pool.submit(_process_seed, t)
                    fmap[fut] = t

                for future in as_completed(fmap):
                    n_done += 1
                    task_info = fmap[future]
                    seed = task_info["seed"]
                    sname = seed.get("name", "?")
                    seed_auc = float(seed.get("best_auc", 0) or 0)

                    try:
                        result = future.result(timeout=600)
                    except Exception as e:
                        result = None
                        log.warning(f"  [!] {sname[:55]} worker crashed: {e}")

                    if result is not None and result.get("improved"):
                        total_improved += 1
                        best = dict(seed)
                        if result.get("best_variant"):
                            best.update(result["best_variant"])
                        best["best_auc"] = result["best_auc"]
                        best["_auc_improvement"] = round(result["best_auc"] - seed_auc, 4)
                        all_tuned.append(best)
                        panel.seeds_improved = total_improved
                        panel.recent_improvements.append(
                            (sname, seed_auc, result["best_auc"]))
                        if len(panel.recent_improvements) > 50:
                            panel.recent_improvements = panel.recent_improvements[-50:]
                        log.info(f"  [+] {sname[:55]} {_pct(seed_auc)} -> {_pct(result['best_auc'])}"
                                 f" ({_delta(result['best_auc']-seed_auc)})"
                                 f"  {result.get('n_variants',0)}v/{result.get('n_evaluated',0)}e")
                    else:
                        all_tuned.append(dict(seed))
                        if result is not None:
                            log.info(f"  [-] {sname[:55]} {_pct(seed_auc)}"
                                     f"  {result.get('n_variants',0)}v/{result.get('n_evaluated',0)}e"
                                     f"  best={_pct(result['best_auc'])}")
                        else:
                            log.info(f"  [!] {sname[:55]} {_pct(seed_auc)}  CRASHED")

                    completed.add(sname)
                    panel.seeds_done = len(all_tuned)
                    panel.variant_done = n_done

                    # ── Incremental save after each seed ──
                    tmp = output_path + ".tmp"
                    _os.makedirs(_os.path.dirname(output_path or ".") or ".", exist_ok=True)
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(all_tuned, f, ensure_ascii=False, indent=2)
                    _os.replace(tmp, output_path)

                    if progress_path:
                        try:
                            _os.makedirs(_os.path.dirname(progress_path) or ".", exist_ok=True)
                            with open(progress_path, "w", encoding="utf-8") as f:
                                json.dump({"completed": sorted(completed),
                                           "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                                          f, ensure_ascii=False, indent=2)
                        except Exception:
                            pass

                    # ── Per-seed ETA update (every ~30s or first 3) ──
                    now = time.time()
                    if now - t2 > 0 and (n_done <= 3 or now - _last_eta > 30):
                        _last_eta = now
                        elapsed = now - t2
                        rate = n_done / max(1, elapsed)
                        eta = (n_seeds - n_done) / max(rate, 0.001)
                        eta_s = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
                        panel.variant_rate = rate
                        panel.eta_str = eta_s

            # ── Batch done ──
            log.info(f"  batch {batch_idx//batch_size + 1}/{n_batches} done in {time.time()-t_batch:.0f}s")

            # ── Update ETA ──
            elapsed = time.time() - t2
            rate = n_done / max(1, elapsed)
            eta = (n_seeds - n_done) / max(rate, 0.001)
            eta_s = f"{eta/60:.0f}m" if eta < 3600 else f"{eta/3600:.1f}h"
            panel.variant_rate = rate
            panel.eta_str = eta_s
            panel.current_label = (f"seq={sl}: seeds {n_done}/{n_seeds}"
                                   f"  ~{rate:.2f}/s  ETA {eta_s}  improved={total_improved}")

        # ── Clean up memmap temp files (Windows spawn only) ──
        if _IS_WINDOWS:
            for _p in (tw_path, vw_path):
                if _p:
                    try: _os.remove(_p)
                    except Exception: pass

        log.info(f"  seq={sl} done in {time.time()-t2:.0f}s  |  elapsed {time.time()-t_start:.0f}s")

    panel.stop()
    log.info(f"DONE: {len(all_tuned)} factors, {total_improved} improved")
    return all_tuned


# ===================================================================
# CLI
# ===================================================================

def main():
    ap = argparse.ArgumentParser(description="Factor tuning pipeline v2")
    ap.add_argument("--input", default="memory/valid_factors_deduped.json")
    ap.add_argument("--output", default="memory/tuned_factors.json")
    ap.add_argument("--config-common", default="config/seq/1.yaml")
    ap.add_argument("--config-validation", default="config/validation.yaml")
    ap.add_argument("--num-workers", type=int, default=128)
    ap.add_argument("--no-structure-mutation", action="store_true")
    ap.add_argument("--max-param-combos", type=int, default=20)
    ap.add_argument("--max-variants-per-seed", type=int, default=50)
    ap.add_argument("--max-factors", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--fast-trees", type=int, default=50, help="Trees for fast screening stage")
    ap.add_argument("--precise-trees", type=int, default=200, help="Trees for precise evaluation stage")
    ap.add_argument("--precise-top-k", type=int, default=5, help="Top-K variants per seed to re-evaluate precisely")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        all_factors = json.load(f)
    log.info(f"Loaded {len(all_factors)} factors")

    if args.max_factors > 0:
        all_factors = all_factors[:args.max_factors]

    if args.dry_run:
        gen = VariantGenerator(not args.no_structure_mutation, args.max_param_combos)
        total = sum(min(len(gen.generate(s)), args.max_variants_per_seed) for s in all_factors)
        log.info(f"DRY RUN: ~{total} variants")
        return

    log.info("Loading data...")
    from mining.discovery import _load_merged_config, load_dataset_config, build_raw_frame_data
    cfg = _load_merged_config(args.config_common, args.config_validation)
    cfg_dir = Path(args.config_common).resolve().parent

    ds_cfg_file = cfg.get("dataset_config_file", "")
    ds_cfg_path = Path(ds_cfg_file)
    if not ds_cfg_path.is_absolute():
        ds_cfg_path = cfg_dir / ds_cfg_path

    ds_info = load_dataset_config(cfg, str(ds_cfg_path), log)
    train_data, val_data, flat_attributes, video_lengths = build_raw_frame_data(ds_info, log, cfg=cfg)
    kp_train, labels_train = train_data
    kp_val, labels_val = val_data
    log.info(f"Train: {kp_train.shape[0]} frames, Val: {kp_val.shape[0]}, D={kp_train.shape[1]}")

    # Resume
    output_path = str(Path(args.output))
    progress_path = str(Path(args.output).with_suffix(".progress.json"))
    completed_set = set()
    existing_results = []
    if not args.no_resume:
        if _os.path.exists(progress_path):
            try:
                with open(progress_path) as f:
                    completed_set = set(json.load(f).get("completed", []))
            except Exception: pass
        if _os.path.exists(output_path):
            try:
                with open(output_path) as f:
                    existing_results = json.load(f)
                completed_set |= {r.get("name","") for r in existing_results if r.get("name")}
            except Exception: pass
        if existing_results:
            log.info(f"Resume: {len(existing_results)} existing, {len(completed_set)} done")

    tuned = tune_factors(
        seed_factors=all_factors,
        kp_train=kp_train, labels_train=labels_train,
        kp_val=kp_val, labels_val=labels_val,
        flat_attributes=flat_attributes,
        num_workers=args.num_workers,
        enable_structure=not args.no_structure_mutation,
        max_param_combos=args.max_param_combos,
        max_variants_per_seed=args.max_variants_per_seed,
        validation_cfg=cfg,
        output_path=output_path,
        progress_path=progress_path,
        completed_set=completed_set,
        initial_results=existing_results,
        fast_trees=args.fast_trees,
        precise_trees=args.precise_trees,
        precise_top_k=args.precise_top_k,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tuned, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(tuned)} factors -> {output_path}")

    improved = sum(1 for f in tuned if f.get("_auc_improvement", 0) > 0.001)
    log.info(f"Improved: {improved}/{len(tuned)}")


if __name__ == "__main__":
    main()
