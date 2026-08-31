"""
factor_tuner.py -- Joint search over factor mathematical form + parameters

Two-phase pipeline:
  Phase 1: Formula structure mutation (5 Mutators)
  Phase 2: Parameter injection + grid/random search

Usage:
  python src/factor_tuner.py \\
    --input memory/valid_factors_deduped.json \\
    --output memory/tuned_factors.json \\
    --config-common config/seq/1.yaml \\
    --config-validation config/validation.yaml \\
    --num-workers 8

  # Parameter search only (no formula structure changes)
  python src/factor_tuner.py --input ... --no-structure-mutation
  # Estimate number of variants
  python src/factor_tuner.py --input ... --dry-run
"""

import argparse
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if not (_PROJECT_ROOT / "config").is_dir():
    _PROJECT_ROOT = _PROJECT_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

if sys.platform == "win32":
    import io
    if getattr(sys.stdout, "encoding", None) != "utf-8" and getattr(sys.stdout, "buffer", None) is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ─────────────────────────── logging ────────────────────────────

def _setup_logging() -> logging.Logger:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    h.flush = sys.stdout.flush
    log = logging.getLogger("factor_tuner")
    log.setLevel(logging.INFO)
    if not log.handlers:
        log.addHandler(h)
    return log

log = _setup_logging()


# ──────────────────────── Percentage display utilities ─────────────────────────

def _pct(val: float) -> str:
    """Format a 0~1 float as a percentage string, e.g. 0.751 -> '75.1%'"""
    return f"{val * 100:.1f}%"


def _delta_pct(val: float) -> str:
    """Format a delta as a signed percentage string, e.g. +0.01 -> '+1.0%', -0.005 -> '-0.5%'"""
    sign = "+" if val >= 0 else ""
    return f"{sign}{val * 100:.1f}%"


# ──────────────────────── Progress persistence utilities ─────────────────────────

def _load_progress(progress_path: str) -> set:
    """Load the set of completed factor names. Returns empty set if file does not exist."""
    if not progress_path or not os.path.exists(progress_path):
        return set()
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = data.get("completed", [])
        log.info(f"Loaded progress file: {len(names)} factors completed -> {progress_path}")
        return set(names)
    except Exception:
        log.warning(f"Progress file corrupted, starting from scratch: {progress_path}")
        return set()


def _save_progress(progress_path: str, completed: set):
    """Write the set of completed factor names to the progress file."""
    if not progress_path:
        return
    from datetime import datetime
    data = {
        "completed": sorted(completed),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_existing_output(output_path: str) -> tuple[list[dict], set]:
    """Load results from an existing output file. Returns (results_list, completed_names_set)."""
    if not output_path or not os.path.exists(output_path):
        return [], set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return [], set()
        names = {item.get("name", "") for item in data if item.get("name")}
        log.info(f"Loaded existing output: {len(data)} results -> {output_path}")
        return data, names
    except Exception:
        log.warning(f"Failed to read output file, starting from scratch: {output_path}")
        return [], set()


def _save_incremental_output(output_path: str, results: list[dict]):
    """Write all current results to the output JSON file (atomic write: temp file then replace)."""
    if not output_path:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)


# ──────────── ProcessPoolExecutor worker functions (module-level, fork-safe) ────────────
# On Linux fork, child processes automatically inherit shared variable memory (copy-on-write)

_shared_train_windows: Optional[np.ndarray] = None
_shared_val_windows: Optional[np.ndarray] = None
_shared_labels_train: Optional[np.ndarray] = None
_shared_labels_val: Optional[np.ndarray] = None
_shared_flat_attributes: list = []
_shared_validation_cfg_snippet: dict = {}  # LGBM params snapshot


def _worker_eval_factor(args: dict) -> Optional[dict]:
    """Subprocess worker: compute factor values + train LGBM, crashes do not affect the main process."""
    import numpy as _np, lightgbm as _lgb, os as _os, copy as _cp
    from sklearn.metrics import roc_auc_score as _roc_auc

    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Suppress LightGBM native logging (subprocess stdout leaks to terminal)
    import logging as _logging
    _logging.getLogger("lightgbm").setLevel(_logging.ERROR)

    factor = args["factor"]
    seed_auc = args.get("seed_auc", 0.0)
    try:
        code = factor["code"]
        mode = factor.get("mode", "row")
        name = factor.get("name", "?")

        # ── Compute factor values ──
        name_to_idx = {n: i for i, n in enumerate(_shared_flat_attributes)}
        func_code = "def _f(windows, idx, np):\n"
        func_code += "    with np.errstate(invalid='ignore', divide='ignore'):\n"
        for line in code.strip().split("\n"):
            func_code += f"        {line}\n"
        func_code += "        return _result\n"
        ns: dict = {"__builtins__": {
            "abs": abs, "min": min, "max": max, "sum": sum,
            "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "list": list, "dict": dict, "tuple": tuple,
            "float": float, "int": int, "bool": bool, "str": str,
            "isinstance": isinstance, "hasattr": hasattr,
            "print": print, "__import__": __import__,
        }}
        exec(compile(func_code, "<factor_subproc>", "exec"), ns)
        func = ns["_f"]

        train_fv = func(_shared_train_windows, name_to_idx, _np)
        val_fv = func(_shared_val_windows, name_to_idx, _np)
        if train_fv is None or val_fv is None:
            return None
        train_fv = _np.asarray(train_fv, dtype=_np.float32)
        val_fv = _np.asarray(val_fv, dtype=_np.float32)
        train_fv[~_np.isfinite(train_fv)] = _np.nan
        val_fv[~_np.isfinite(val_fv)] = _np.nan
        if _np.isnan(train_fv).mean() > 0.8:
            return None

        # ── Early elimination: 40-tree pre-check ──
        if seed_auc > 0:
            try:
                nan_m = _np.isnan(train_fv)
                tv = train_fv[~nan_m]; lv = _shared_labels_train[~nan_m]
                if len(tv) < 10: return None
                qm = _lgb.LGBMClassifier(n_estimators=40, max_depth=3, num_leaves=15,
                    learning_rate=0.05, random_state=42, n_jobs=1, verbose=-1)
                qm.fit(tv.reshape(-1,1), lv)
                vv = ~_np.isnan(val_fv)
                if vv.sum() < 5: return None
                pb = qm.predict_proba(val_fv[vv].reshape(-1,1))
                cls = _np.unique(_shared_labels_val[vv])
                aucs = [_roc_auc((_shared_labels_val[vv]==c).astype(int), pb[:,c])
                         for c in cls if c < pb.shape[1]]
                if max(aucs) if aucs else 0 < seed_auc - 0.05:
                    return None
            except Exception: pass

        # ── Full LGBM OvR training ──
        cfg = _shared_validation_cfg_snippet
        lgbm_params = dict(cfg.get("lgbm_params", {}))
        lgbm_params.setdefault("n_jobs", 1)
        lgbm_params["verbosity"] = -1  # Suppress C++ layer stdout output
        min_auc = cfg.get("min_auc", 0.65)
        early_stop = cfg.get("early_stopping_rounds", 20)

        nan_m = _np.isnan(train_fv); tv = train_fv[~nan_m]; lv = _shared_labels_train[~nan_m]
        best_auc, best_f1, best_cls = 0.0, 0.0, ""
        per_class, valid_cls_list = {}, []

        for cls_id in _np.unique(lv):
            yb = (lv == cls_id).astype(int)
            if yb.sum() < 5: continue
            sw = max(1.0, (len(yb)-yb.sum())/max(yb.sum(),1))
            m = _lgb.LGBMClassifier(**{**lgbm_params, "scale_pos_weight": sw}, verbose=-1)
            m.fit(tv.reshape(-1,1), yb,
                  eval_set=[(val_fv.reshape(-1,1), (_shared_labels_val==cls_id).astype(int))],
                  eval_metric="auc",
                  callbacks=[_lgb.early_stopping(early_stop, verbose=False)] if early_stop>0 else None)
            vv = ~_np.isnan(val_fv)
            if vv.sum() < 5: continue
            pb = m.predict_proba(val_fv[vv].reshape(-1,1))
            if pb.shape[1] < 2: continue
            try: auc = float(_roc_auc((_shared_labels_val[vv]==cls_id).astype(int), pb[:,1]))
            except Exception: continue
            try:
                from sklearn.metrics import f1_score
                f1 = float(f1_score((_shared_labels_val[vv]==cls_id).astype(int), pb[:,1]>0.5, zero_division=0))
            except Exception: f1 = 0.0
            per_class[str(cls_id)] = {"auc": round(auc,4), "f1": round(f1,4)}
            if auc > min_auc:
                valid_cls_list.append({"class": str(cls_id), "auc": round(auc,4), "f1": round(f1,4)})
            if auc > best_auc: best_auc, best_f1, best_cls = auc, f1, str(cls_id)

        if best_auc < min_auc: return None
        return {"best_auc": float(best_auc), "best_f1": float(best_f1), "best_class": best_cls,
                "valid_classes": valid_cls_list, "per_class": per_class,
                "n_train": len(tv), "n_val": int((~_np.isnan(val_fv)).sum())}
    except Exception:
        return None


# ──────────── Background status monitor (periodic progress + resource + error printing) ────────────

class StatusMonitor:
    """Background thread: prints a progress panel and resource usage every N seconds."""

    def __init__(self, interval: float = 30.0):
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Externally updated fields
        self.total_factors = 0
        self.completed_count = 0
        self.current_factor = ""
        self.current_seq_length = 0
        self.total_improved = 0
        self.errors: list[str] = []  # Recent errors (max 20 kept)
        self._errors_lock = threading.Lock()
        self._last_snapshot_time = 0.0

    def add_error(self, msg: str):
        with self._errors_lock:
            self.errors.append(msg)
            if len(self.errors) > 20:
                self.errors = self.errors[-20:]

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        import time as _time
        while not self._stop.wait(self.interval):
            self._print_status()

    def _print_status(self):
        import time as _time, os as _os
        now = _time.time()
        elapsed = now - self._last_snapshot_time if self._last_snapshot_time else 0
        self._last_snapshot_time = now

        lines = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("  factor_tuner runtime status")
        lines.append(f"  Progress: {self.completed_count}/{self.total_factors}"
                     f" ({self.completed_count / max(1, self.total_factors) * 100:.1f}%)"
                     f"  |  Improved: {self.total_improved}"
                     f"  |  Current window seq={self.current_seq_length}")
        if self.current_factor:
            lines.append(f"  Current: {self.current_factor[:70]}")

        # Resource usage (including child processes; under ProcessPoolExecutor the main process is nearly idle)
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            mem_mb = mem.rss / 1024 / 1024
            # CPU total of main process + all child processes
            cpu = proc.cpu_percent(interval=0.1)
            for child in proc.children(recursive=True):
                try:
                    cpu += child.cpu_percent(interval=0.05)
                except psutil.NoSuchProcess:
                    pass
            cpu_cores = psutil.cpu_count()
            sys_mem = psutil.virtual_memory()
            lines.append(f"  CPU: {cpu/cpu_cores:.0f}% (equiv {cpu/100:.1f}/{cpu_cores} cores)  |  "
                         f"Memory: {mem_mb:.0f}MB  |  System available: {sys_mem.available / 1024**3:.1f}GB"
                         f"  |  Child processes: {len(proc.children())}  |  Threads: {proc.num_threads()}")
        except ImportError:
            lines.append(f"  (pip install psutil to view resource usage)")

        # Recent errors
        with self._errors_lock:
            if self.errors:
                lines.append(f"  -- Recent errors ({len(self.errors)} total) --")
                for e in self.errors[-5:]:
                    lines.append(f"  ! {e[:90]}")

        lines.append("=" * 60)
        log.info("\n".join(lines))


# ===================================================================
# Code structure parsing
# ===================================================================

# Whitelist of safe numpy functions for sandboxed environment
_SAFE_NP_FUNCS = {
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


# ──────────── ProcessPoolExecutor worker functions (module-level, pickleable) ────────────

def _worker_lgbm_eval(args: dict) -> Optional[dict]:
    """Multi-process worker: only uses LGBM to evaluate pre-computed factor values, no windows/engine involved."""
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"     # worker does not do heavy numpy
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import roc_auc_score

    # Suppress LightGBM native logging
    import logging as _logging
    _logging.getLogger("lightgbm").setLevel(_logging.ERROR)

    train_factor = args["train_factor"]
    val_factor = args["val_factor"]
    labels_train = args["labels_train"]
    labels_val = args["labels_val"]
    lgbm_params = dict(args["lgbm_params"])
    lgbm_params["verbosity"] = -1  # Suppress C++ layer stdout output
    min_auc = args.get("min_auc", 0.65)
    min_f1 = args.get("min_f1", 0.50)
    early_stop = args.get("early_stopping_rounds", 20)
    seed_auc = args.get("seed_auc", 0.0)

    try:
        # NaN pre-filtering
        nan_mask = np.isnan(train_factor)
        t_valid = train_factor[~nan_mask]
        l_valid = labels_train[~nan_mask]
        if len(t_valid) < 10:
            return None

        # Early elimination: 40-tree pre-check
        if seed_auc > 0:
            try:
                quick_lgbm = lgb.LGBMClassifier(
                    n_estimators=40, max_depth=3, num_leaves=15,
                    learning_rate=0.05, random_state=42, n_jobs=1,
                    verbose=-1,
                )
                quick_lgbm.fit(t_valid.reshape(-1, 1), l_valid)
                v_valid = ~np.isnan(val_factor)
                if v_valid.sum() < 5:
                    return None
                proba = quick_lgbm.predict_proba(val_factor[v_valid].reshape(-1, 1))
                valid_cls = np.unique(labels_val[v_valid])
                aucs = []
                for c in valid_cls:
                    if c < proba.shape[1]:
                        try:
                            aucs.append(roc_auc_score(
                                (labels_val[v_valid] == c).astype(int), proba[:, c]))
                        except Exception:
                            pass
                quick_auc = max(aucs) if aucs else 0.0
                if quick_auc < seed_auc - 0.05:
                    return None
            except Exception:
                pass

        # Full LGBM training (OvR multi-class)
        unique_cls = np.unique(l_valid)
        best_auc = 0.0
        best_f1 = 0.0
        best_class = ""
        per_class = {}
        valid_classes = []

        for cls_id in unique_cls:
            y_train_bin = (l_valid == cls_id).astype(int)
            pos_count = y_train_bin.sum()
            neg_count = len(y_train_bin) - pos_count
            if pos_count < 5:
                continue

            scale_pos_weight = max(1.0, neg_count / max(pos_count, 1))
            model = lgb.LGBMClassifier(
                **{**lgbm_params, "scale_pos_weight": scale_pos_weight},
                verbose=-1,
            )
            model.fit(
                t_valid.reshape(-1, 1), y_train_bin,
                eval_set=[(val_factor.reshape(-1, 1),
                           (labels_val == cls_id).astype(int))],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(early_stop, verbose=False)] if early_stop > 0 else None,
            )

            v_valid = ~np.isnan(val_factor)
            if v_valid.sum() < 5:
                continue
            proba = model.predict_proba(val_factor[v_valid].reshape(-1, 1))
            y_val_bin = (labels_val[v_valid] == cls_id).astype(int)

            if proba.shape[1] < 2:
                continue
            try:
                auc = float(roc_auc_score(y_val_bin, proba[:, 1]))
            except Exception:
                continue

            # F1 (single class, threshold 0.5 or optimal)
            try:
                from sklearn.metrics import f1_score
                pred = proba[:, 1] > 0.5
                f1 = float(f1_score(y_val_bin, pred, zero_division=0))
            except Exception:
                f1 = 0.0

            per_class[str(cls_id)] = {"auc": round(auc, 4), "f1": round(f1, 4)}
            if auc > min_auc:
                valid_classes.append({"class": str(cls_id), "auc": round(auc, 4), "f1": round(f1, 4)})
            if auc > best_auc:
                best_auc = auc
                best_f1 = f1
                best_class = str(cls_id)

        if best_auc < min_auc:
            return None

        return {
            "best_auc": float(best_auc),
            "best_f1": float(best_f1),
            "best_class": best_class,
            "valid_classes": valid_classes,
            "per_class": per_class,
            "n_train": len(t_valid),
            "n_val": int((~np.isnan(val_factor)).sum()),
        }
    except Exception:
        return None


def _is_slice_index(text: str, start: int, end: int) -> bool:
    """
    Check if the numeric constant at position [start, end) is an array index
    (an integer inside square brackets).
    Includes slice indices ([:, 3:, ...]) and normal indices (inds[2], arr[0]).
    These constants should not be freely scaled.
    """
    # Scan backwards for the nearest unclosed '[' -- any digit inside [...] is an index
    bracket_depth = 0
    for i in range(end - 1, -1, -1):
        ch = text[i]
        if ch == ']':
            bracket_depth += 1
        elif ch == '[':
            if bracket_depth > 0:
                bracket_depth -= 1
            else:
                # Found the nearest unclosed '[', the number is inside brackets -> index
                return True
    return False


def _replace_func_call(code: str, func_name: str, transform) -> str:
    """
    Find the first occurrence of func_name(...) in code, use bracket counting to find
    the matching closing parenthesis, and call transform(inner_text) to replace the
    entire call. Returns the new code, or the original if not found.

    Example: _replace_func_call("np.abs(np.mean(x) - 0.5)", "np.abs", lambda s: s)
    -> "np.mean(x) - 0.5"
    """
    prefix = func_name + "("
    idx = code.find(prefix)
    if idx < 0:
        return code
    start = idx + len(prefix)  # First character after '('
    depth = 1
    pos = start
    while pos < len(code) and depth > 0:
        ch = code[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                inner = code[start:pos]
                replacement = transform(inner)
                return code[:idx] + replacement + code[pos + 1:]
        pos += 1
    return code  # Unmatched brackets, return original


def _code_is_valid(code: str) -> tuple[bool, str]:
    """
    Pre-validate whether factor code is likely to execute correctly.

    Returns (is_valid, reason).
    Checks:
    1. np.clip / np.percentile have correct argument counts
    2. idx.get check exists
    3. Basic bracket matching
    4. return/_result exists
    """
    # Bracket matching
    if code.count("(") != code.count(")"):
        return False, "Unmatched parentheses"

    # Must have return or _result
    if "return" not in code and "_result" not in code:
        return False, "Missing return or _result"

    # Must have idx.get or idx['...'] or names= list
    if "idx.get" not in code and "idx[" not in code and "names=" not in code:
        return False, "Missing idx feature access"

    # Check np.clip argument count (must have 3 arguments)
    for m in re.finditer(r"np\.clip\(([^)]+)\)", code):
        args = m.group(1).split(",")
        if len(args) != 3:
            return False, f"np.clip wrong argument count: {len(args)} (need 3)"

    # Check np.percentile argument count (must have 2 arguments)
    for m in re.finditer(r"np\.percentile\(([^)]+)\)", code):
        args = m.group(1).split(",")
        if len(args) != 2:
            return False, f"np.percentile wrong argument count: {len(args)} (need 2)"

    # Check for unknown function calls
    unknown_funcs = set()
    for m in re.finditer(r"np\.(\w+)\s*\(", code):
        fname = f"np.{m.group(1)}"
        if fname not in _SAFE_NP_FUNCS:
            unknown_funcs.add(fname)
    if unknown_funcs:
        return False, f"Unknown numpy function(s): {unknown_funcs}"

    return True, "ok"


def _classify_code_structure(code: str) -> dict:
    """
    Split factor code into: setup (feature extraction) + safety (checks) + compute (formula) + return.

    Returns {"setup": [...], "safety": [...], "compute": [...], "return": [...], "mode": "row"|"batch"}
    """
    lines = code.strip().split("\n")
    segments = {"setup": [], "safety": [], "compute": [], "return": [], "mode": "row"}

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Mode detection
        if "_result" in stripped and "=" in stripped:
            segments["mode"] = "batch"

        # Classification
        if re.search(r"idx\.get\(|idx\[", stripped) and "=" in stripped:
            segments["setup"].append(stripped)
        elif re.search(r"if\s+.*<\s*0\s*:|return\s+np\.nan", stripped):
            segments["safety"].append(stripped)
        elif stripped.startswith("return") or stripped.startswith("_result"):
            segments["return"].append(stripped)
        else:
            # Intermediate computation (feature extraction + formula)
            if re.match(r"^\w+\s*=\s*window\[", stripped) or \
               re.match(r"^\w+\s*=\s*np\.", stripped) or \
               "=" in stripped and ("np." in stripped or "+" in stripped or "-" in stripped or "*" in stripped or "/" in stripped):
                segments["compute"].append(stripped)

    return segments


def _extract_terms_from_compute(compute_lines: list[str], mode: str) -> list[dict]:
    """
    Extract independent "terms" from compute lines. Each term is a scalar output of feature x aggregator.

    Returns list[{"expr": str, "feat": str|null, "agg": str|null, "coef": float}]
    """
    terms = []
    for line in compute_lines:
        # Match var = expression pattern
        m = re.match(r"(\w+)\s*=\s*(.+)", line)
        if not m:
            continue
        var_name, expr = m.group(1), m.group(2).strip()

        # Extract aggregator
        agg_match = re.search(r"np\.(\w+)", expr)
        agg = agg_match.group(1) if agg_match else None

        # Extract feature (infer from variable name: if starts with i_ then it is not a term)
        if var_name.startswith("i_"):
            continue

        # Extract coefficient
        coef_match = re.match(r"([\d.]+)\s*\*", expr)
        coef = float(coef_match.group(1)) if coef_match else 1.0

        terms.append({
            "var": var_name,
            "expr": expr,
            "agg": agg,
            "coef": coef,
            "line": line,
        })

    # Also detect direct result = expr1 + expr2 - expr3 forms
    return_line = None
    for line in compute_lines:
        if re.match(r"(_?result|score|val)\s*=", line):
            return_line = line
            break

    if return_line and not terms:
        # Try to split terms from result = ... (split by + and -)
        m = re.match(r"\w+\s*=\s*(.+)", return_line)
        if m:
            rhs = m.group(1)
            # Simple + - splitting (not precise but covers most cases)
            sub_terms = re.split(r"\s*([+-])\s*", rhs)
            if len(sub_terms) > 1:
                for i, chunk in enumerate(sub_terms):
                    chunk = chunk.strip()
                    if chunk in ("+", "-"):
                        continue
                    # Check if it has a numeric coefficient
                    coef = 1.0
                    coef_match = re.match(r"([\d.]+)\s*\*", chunk)
                    if coef_match:
                        coef = float(coef_match.group(1))
                    elif i > 0 and sub_terms[i-1].strip() == "-":
                        coef = -1.0
                    terms.append({
                        "var": f"_term{i}",
                        "expr": chunk,
                        "agg": re.search(r"np\.(\w+)", chunk).group(1) if re.search(r"np\.(\w+)", chunk) else None,
                        "coef": coef,
                        "line": chunk,
                    })

    return terms


# ===================================================================
# Phase 2a: Parameter injection (bake into code)
# ===================================================================

class ParameterInjector:
    """
    Parse factor formulas, inject parameters, and bake them as concrete values.
    Returns multiple baked variant codes (one per parameter combination).
    """

    PARAM_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    PARAM_VALUES_FINE = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

    def inject_and_bake(
        self, code: str, mode: str, max_combos: int = 20,
    ) -> list[tuple[str, int, str]]:
        """
        Generate multiple parameter-baked code variants.
        Returns list[(baked_code, n_params, variant_label)]
        """
        # Find the result expression
        result_var = "_result" if (mode == "batch") else "result"
        result_pattern = rf"{result_var}\s*=\s*(.+)"
        m_result = re.search(result_pattern, code)
        if not m_result:
            # May also be in return float(...)
            m_result = re.search(rf"return\s+float\(\s*(.+)\s*\)", code)
        if not m_result:
            return []

        rhs = m_result.group(1).strip()

        # Split by +/-
        terms = self._split_rhs(rhs)
        if len(terms) <= 1:
            # Single term: wrap whole expression with parameters
            return self._bake_single_term(code, rhs, result_var, max_combos)

        return self._bake_multi_term(code, rhs, terms, result_var, max_combos)

    def _bake_single_term(self, code: str, rhs: str, result_var: str, max_combos: int):
        """
        For product-chain expressions (1.0-var1)*(1.0-var2)*..., inject per-feature independent weights.
        Use three levels [0.5, 1.0, 2.0] per feature variable, sample up to max_combos combinations.
        """
        results = []
        # Find all (1.0 - var_name) patterns
        var_pattern = re.findall(r'\(1\.0\s*-\s*(\w+)\)', rhs)
        if len(var_pattern) >= 2:
            # Per-feature independent weights: each feature [0.5, 1.0, 2.0], random sample combinations
            n_feats = len(var_pattern)
            rng = np.random.default_rng(42 + n_feats)
            choices = [0.5, 1.0, 2.0]
            seen_combos = set()
            for _ in range(max_combos):
                combo = tuple(float(rng.choice(choices)) for _ in range(n_feats))
                if combo in seen_combos or all(abs(c - 1.0) < 0.001 for c in combo):
                    continue
                seen_combos.add(combo)
                new_rhs = rhs
                for vi, var_name in enumerate(var_pattern):
                    w = combo[vi]
                    old_term = f'(1.0 - {var_name})'
                    new_term = f'(1.0 - {w:.4g} * {var_name})'
                    new_rhs = new_rhs.replace(old_term, new_term, 1)
                if new_rhs != rhs:
                    new_code = self._replace_rhs(code, rhs, new_rhs, result_var)
                    if new_code != code:
                        w_str = ','.join(f'{w:.3g}' for w in combo)
                        results.append((new_code, n_feats, f'w=[{w_str}]'))
        else:
            # Non-product-chain expressions: do not force parameter injection (monotonic transforms like np.power are useless for AUC)
            pass
        return results

    def _replace_rhs(self, code: str, old_rhs: str, new_rhs: str, result_var: str) -> str:
        """Safely replace the right-hand side of a formula, supports return float() and result = modes."""
        # Try multiple exact replacement patterns
        for old_pat, new_pat in [
            (f"return float({old_rhs})", f"return float({new_rhs})"),
            (f"{result_var} = {old_rhs}", f"{result_var} = {new_rhs}"),
        ]:
            if old_pat in code:
                return code.replace(old_pat, new_pat, 1)
        # fallback: if rhs appears uniquely in code
        if code.count(old_rhs) == 1:
            return code.replace(old_rhs, new_rhs, 1)
        return code

    def _bake_multi_term(self, code: str, rhs: str, terms: list, result_var: str,
                         max_combos: int):
        import itertools
        n = len([t for t in terms if t[1].strip()])  # non-empty terms
        if n <= 0:
            return []

        # Limit parameter combinations
        if n == 2:
            param_grids = [[0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 1.5, 2.0]]
        elif n == 3:
            param_grids = [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
        else:
            param_grids = [[0.5, 1.0, 2.0]] * n

        results = []
        combos = list(itertools.product(*param_grids))
        for combo in combos[:max_combos]:
            new_parts = []
            pi = 0
            for sign, term in terms:
                term = term.strip()
                if not term:
                    continue
                pv = combo[pi] if pi < len(combo) else 1.0
                pi += 1
                sign_str = f" {sign} " if sign else ""
                if abs(pv - 1.0) < 0.001:
                    new_parts.append(f"{sign_str}({term})")
                else:
                    new_parts.append(f"{sign_str}({pv:.4g}) * ({term})")
            new_rhs = "".join(new_parts).lstrip(" +")
            new_code = re.sub(
                rf"{result_var}\s*=\s*{re.escape(rhs)}",
                f"{result_var} = {new_rhs}",
                code,
            )
            if new_code == code:
                # fallback: just replace the rhs directly
                new_code = code.replace(
                    f"{result_var} = {rhs}",
                    f"{result_var} = {new_rhs}",
                )
            label = f"params={','.join(f'{v:.3g}' for v in combo)}"
            results.append((new_code, n, label))
        return results

    @staticmethod
    def _split_rhs(rhs: str) -> list[tuple[str, str]]:
        """Split the right-hand side of a formula by +/- into (sign, term) pairs. Preserve brackets."""
        terms = []
        depth = 0
        current = ""
        current_sign = ""
        for i, ch in enumerate(rhs):
            if ch == "(":
                depth += 1
                current += ch
            elif ch == ")":
                depth -= 1
                current += ch
            elif depth == 0 and ch in "+-":
                # Check if it is a minus sign (not in scientific notation)
                if ch == "-" and (i == 0 or rhs[i-1] in " eE"):
                    current += ch
                    continue
                if current.strip():
                    terms.append((current_sign, current.strip()))
                current_sign = "+" if ch == "+" else "-"
                current = ""
            else:
                current += ch
        if current.strip():
            terms.append((current_sign, current.strip()))
        return terms


# ===================================================================
# Phase 2b: Parameter search
# ===================================================================

# ──────────── LLM parameterized formula support ────────────

def _extract_llm_params(code: str) -> list[dict]:
    """
    Extract _P{n} parameter definitions from LLM-generated code.

    Recognizes patterns: _P0 = 1.0 or _P1 = 0.3  # comment
    Returns list[{"name": "_P0", "default": 1.0, "kind": "weight", "range": [0.05, 5.0]}]

    kind inference rules:
      - Default in (0, 0.15] -> "threshold_small"
      - Default in (0.15, 0.85) and not 1.0 -> "threshold"
      - Default > 2.0 -> "scale"
      - Default == 1.0 -> "weight" (neutral default)
    """
    params = []
    # Match _P{n} = numeric_value pattern (allows trailing comments)
    pat = re.compile(r'^(_P\d+)\s*=\s*([\d.]+(?:[eE][+-]?\d+)?)\s*(?:#.*)?$', re.MULTILINE)

    for m in pat.finditer(code):
        name = m.group(1)
        default = float(m.group(2))

        # Infer parameter type and search range
        if default == 1.0:
            kind = "weight"
            search_range = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
        elif default >= 2.0:
            kind = "scale"
            search_range = [max(0.05, default*0.1), default*0.3, default*0.6,
                           default*0.8, default, default*1.25, default*1.5, default*2.0, default*3.0]
        elif default <= 0.15:
            kind = "threshold_small"
            search_range = [0.01, 0.03, 0.05, 0.08, max(0.01, default*0.5),
                           default, min(0.95, default*2.0), min(0.95, default*3.0)]
        else:
            kind = "threshold"
            search_range = [max(0.01, default*0.2), default*0.5, default*0.75,
                           default, min(0.95, default*1.5), min(0.95, default*2.0)]

        # Deduplicate and sort
        search_range = sorted(set(round(v, 4) for v in search_range))

        params.append({
            "name": name,
            "default": default,
            "kind": kind,
            "range": search_range,
        })

    return params


def _generate_llm_param_combos(params: list[dict], max_combos: int = 100) -> list[dict[str, float]]:
    """
    Generate search combinations for LLM parameters. Adaptive strategy:
      - 1-2 params: fine grid (max 7 levels per param)
      - 3-4 params: coarse grid (4-5 levels per param) + random sampling
      - 5+ params: center + single-param variation + random sampling
    """
    import itertools
    n = len(params)
    if n == 0:
        return [{}]

    if n <= 2:
        # Fine grid: take up to 7 values per param
        grids = [p["range"][:7] for p in params]
        combos = list(itertools.product(*grids))
    elif n <= 4:
        # Coarse grid: take 4-5 key values per param (including default)
        grids = []
        for p in params:
            rng = p["range"]
            d = p["default"]
            key_vals = [d]
            # Add min, max, median
            if rng[0] not in key_vals: key_vals.append(rng[0])
            if rng[-1] not in key_vals: key_vals.append(rng[-1])
            mid = rng[len(rng)//2]
            if mid not in key_vals: key_vals.append(mid)
            grids.append(sorted(set(round(v, 4) for v in key_vals)))
        combos = list(itertools.product(*grids))
    else:
        # Center + single-param variation + random sampling
        centers = tuple(p["default"] for p in params)
        combos = [centers]
        for i, p in enumerate(params):
            for v in p["range"][:5]:
                t = list(centers)
                t[i] = v
                combos.append(tuple(t))
        # Random supplement
        rng = np.random.default_rng(42 + n)
        for _ in range(min(max_combos, 200)):
            t = tuple(float(rng.choice(p["range"])) for p in params)
            if t not in combos:
                combos.append(t)

    result = []
    for combo in combos[:max_combos]:
        d = {params[i]["name"]: float(v) for i, v in enumerate(combo)}
        result.append(d)
    return result


def _apply_llm_params(code: str, param_dict: dict[str, float]) -> str:
    """
    Inject parameter values into code: replace _P0 = 1.0 -> _P0 = 2.5

    Only replaces the right-hand side of _P{n} = value assignment, keeping comments unchanged.
    """
    result = code
    for pname, pval in param_dict.items():
        new_val_str = f"{pval:.6g}"
        if '.' not in new_val_str:
            new_val_str += '.0'
        pat = re.compile(
            rf'^({re.escape(pname)}\s*=\s*)[\d.]+(?:[eE][+-]?\d+)?(\s*.*)?$',
            re.MULTILINE,
        )
        result = pat.sub(rf'\g<1>{new_val_str}\g<2>', result)
    return result


def _auto_parameterize(code: str) -> tuple[str, list[dict]]:
    """
    Automatically convert hardcoded constants in legacy factors to _P{n} parameter form.

    Conservative strategy: only convert explicit weight coefficients (num * var) and
    thresholds (var > num), skip all structural constants (min/max boundaries,
    abs center values, 1.0-x complement values, etc.).
    Better to miss a few tunable constants than break formula semantics.

    Returns (parameterized_code, params_info), or (code, []) if no convertible constants found.
    """
    # 1. Find the formula region (skip idx.get assignments + safety check lines)
    lines = code.strip().split("\n")
    formula_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            formula_start = i + 1
            continue
        if re.search(r"idx\.get\(|idx\[", stripped) and "=" in stripped:
            formula_start = i + 1
            continue
        if re.search(r"if\s+.*<\s*0\s*:\s*return\s+np\.nan", stripped):
            formula_start = i + 1
            continue
        if stripped.startswith("return np.nan"):
            formula_start = i + 1
            continue
        if re.match(r"^_P\d+\s*=", stripped):
            formula_start = i + 1
            continue
        break

    header_lines = lines[:formula_start]
    formula_lines = lines[formula_start:]
    formula_text = "\n".join(formula_lines)

    # 2. Find safely parameterizable constants (only weight coefficients and thresholds)
    const_candidates = []

    # Pattern A: weight coefficient -- <num> * <identifier>
    for m in re.finditer(
        r'(?<![\w.])(\d+\.\d+)\s*\*\s*([a-zA-Z_]\w*)',
        formula_text,
    ):
        val_str = m.group(1)
        val = float(val_str)
        var = m.group(2)
        # Skip structural values
        if val in (0.0, 1.0, 2.0):
            # 1.0 and 2.0 are likely structural, but keep if var name contains feat/val/score
            if not re.search(r'feat|val|score|factor', var, re.IGNORECASE):
                continue
        const_candidates.append({
            "value": val,
            "start": m.start(1),
            "end": m.end(1),
            "kind": "weight",
        })

    # Pattern B: threshold -- <var> > <num> or <var> < <num> (not sentinel check)
    for m in re.finditer(
        r'([a-zA-Z_]\w*)\s*([><]=?)\s*(\d+\.\d+)',
        formula_text,
    ):
        var = m.group(1)
        op = m.group(2)
        val = float(m.group(3))
        # Skip if i_xxx < 0 pattern (sentinel check)
        if val == 0.0 and op in ("<", "<="):
            continue
        # Skip structural values
        if val == 0.5 and 'abs' in formula_text[max(0, m.start()-20):m.start()]:
            continue  # abs(v - 0.5) * 2.0 pattern
        # Skip values inside min/max/clip
        prefix = formula_text[max(0, m.start()-30):m.start()]
        if re.search(r'\b(?:min|max|clip|percentile)\s*\([^)]*$', prefix):
            continue
        const_candidates.append({
            "value": val,
            "start": m.start(3),
            "end": m.end(3),
            "kind": "threshold",
        })

    # Deduplicate (by position)
    seen_positions = set()
    unique_candidates = []
    for c in const_candidates:
        if c["start"] not in seen_positions:
            seen_positions.add(c["start"])
            unique_candidates.append(c)
    const_candidates = unique_candidates

    if not const_candidates:
        return code, []

    # 3. Replace from back to front by position
    const_candidates.sort(key=lambda c: c["start"], reverse=True)

    new_formula = formula_text
    params_info = []
    for i, c in enumerate(const_candidates):
        pname = f"_P{i}"
        new_formula = (
            new_formula[:c["start"]] + pname + new_formula[c["end"]:]
        )
        val = c["value"]
        kind = c["kind"]

        # Generate search range for each parameter
        if kind == "weight":
            search_range = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
        else:
            search_range = sorted(set([
                max(0.01, val*0.2), val*0.5, val*0.75,
                val, min(0.95, val*1.5), min(0.95, val*2.0)
            ]))

        params_info.append({
            "name": pname,
            "default": val,
            "kind": kind,
            "range": [round(v, 4) for v in search_range],
        })

    # 4. Build new code
    params_info.reverse()
    param_def_lines = [
        f"{p['name']} = {p['default']:.6g}" +
        (".0" if '.' not in f"{p['default']:.6g}" else "") +
        f"  # auto: {p['kind']} (was {p['default']})"
        for p in params_info
    ]

    # Insert point: after the last idx.get assignment line
    insert_pos = len(header_lines)
    for i in range(len(header_lines) - 1, -1, -1):
        if re.search(r"idx\.get\(|idx\[", header_lines[i]) and "=" in header_lines[i]:
            insert_pos = i + 1
            break

    new_lines = (
        header_lines[:insert_pos]
        + param_def_lines
        + header_lines[insert_pos:]
        + [new_formula]
    )

    return "\n".join(new_lines), params_info


# ===================================================================
# Phase 1: Formula structure mutation (Mutators)
# ===================================================================

class AggregatorMutator:
    """Aggregator replacement: mean <-> std <-> median <-> min <-> max <-> percentile"""

    AGG_MAP = {
        "mean": ["std", "min", "max", "median"],
        "std": ["mean", "var"],
        "min": ["max", "mean", "median"],
        "max": ["min", "mean", "median"],
        "median": ["mean", "percentile"],
    }
    PERCENTILE_VALUES = [10, 25, 75, 90]

    def mutate(self, code: str, mode: str) -> list[tuple[str, str, str]]:
        variants = []
        for old_agg, new_aggs in self.AGG_MAP.items():
            pattern = rf"np\.{old_agg}\("
            if pattern not in code:
                continue
            for new_agg in new_aggs:
                if new_agg == "percentile":
                    for p in self.PERCENTILE_VALUES:
                        new_code = re.sub(
                            rf"np\.{old_agg}\(([^)]+)\)",
                            rf"np.percentile(\1, {p})",
                            code,
                        )
                        variants.append((
                            new_code,
                            f"{old_agg}2pctl{p}",
                            f"np.{old_agg} -> np.percentile(..., {p})",
                        ))
                elif new_agg == "var":
                    new_code = re.sub(
                        rf"np\.{old_agg}\(", "np.var(", code
                    )
                    variants.append((
                        new_code,
                        f"{old_agg}2var",
                        f"np.{old_agg} -> np.var",
                    ))
                else:
                    new_code = re.sub(
                        rf"np\.{old_agg}\(", f"np.{new_agg}(", code
                    )
                    variants.append((
                        new_code,
                        f"{old_agg}2{new_agg}",
                        f"np.{old_agg} -> np.{new_agg}",
                    ))
        return variants


class TemporalMutator:
    """Temporal transforms: first-last diff -> trend slope -> second-order diff -> volatility"""

    def mutate(self, code: str, mode: str) -> list[tuple[str, str, str]]:
        variants = []

        # Detect seq_length hints
        if "window[-1," not in code and "window[0," not in code:
            return variants

        # First-last diff -> trend slope
        if re.search(r"window\[-1,\s*\w+\]\s*-\s*window\[0,\s*\w+\]", code):
            for m in re.finditer(r"(window\[-1,\s*(\w+)\]\s*-\s*window\[0,\s*\2\])", code):
                old_expr = m.group(1)
                idx_var = m.group(2)
                # Replace with polyfit slope
                t_expr = f"np.polyfit(np.arange(window.shape[0]), window[:, {idx_var}], 1)[0]"
                new_code = code.replace(old_expr, t_expr)
                variants.append((new_code, "diff2trend",
                                f"first-last diff -> polyfit trend slope"))
                break

        # Mean -> volatility (std/mean)
        if "np.mean(" in code:
            for m in re.finditer(r"np\.mean\(window\[:\s*,\s*:\s*,\s*(\w+)\]", code):
                idx_var = m.group(1)
                old = m.group(0)
                new = f"np.std(window[:, :, {idx_var}]) / (np.mean(window[:, :, {idx_var}]) + 1e-8)"
                new_code = code.replace(old, new)
                variants.append((new_code, "mean2vol",
                                f"np.mean -> volatility (std/mean)"))
                break

        return variants


class CrossFeatureMutator:
    """Cross-feature operation replacement: ratio <-> difference <-> product"""

    def mutate(self, code: str, mode: str) -> list[tuple[str, str, str]]:
        variants = []

        # ratio -> difference
        ratio_patterns = list(re.finditer(r"(\w+)\s*/\s*\((\w+)\s*\+\s*1e-?\d+\)", code))
        for m in ratio_patterns:
            a, b = m.group(1), m.group(2)
            old = m.group(0)
            # -> abs diff
            new = f"np.abs({a} - {b})"
            new_code = code.replace(old, new)
            variants.append((new_code, "ratio2diff", f"{a}/{b} -> |{a}-{b}|"))
            # -> product
            new2 = f"{a} * {b}"
            new_code2 = code.replace(old, new2)
            variants.append((new_code2, "ratio2prod", f"{a}/{b} -> {a}*{b}"))
            break

        # product -> difference
        prod_patterns = list(re.finditer(r"(\w+)\s*\*\s*(\w+)", code))
        for m in prod_patterns:
            a, b = m.group(1), m.group(2)
            # Exclude np module name matched as variable (e.g. extension * np.clip -> "float - module")
            if a == "np" or b == "np":
                continue
            # Exclude pure numbers matched from float literals (e.g. 0.45 * far -> matches "45 * far")
            if a.isdigit() and m.start() > 0 and code[m.start() - 1] == ".":
                continue
            if b.isdigit() and m.end() < len(code) and code[m.end()] == ".":
                continue
            # Ensure matched product is in return / _result line or nearby formula line
            line_start = code.rfind("\n", 0, m.start()) + 1
            line_prefix = code[line_start:m.start()]
            if "return" in line_prefix or "_result" in line_prefix or \
               re.search(r"(return|_result)\s*=", code[line_start:m.end() + 20]):
                old = m.group(0)
                new = f"np.abs({a} - {b})"
                new_code = code.replace(old, new, 1)
                variants.append((new_code, "prod2diff", f"{a}*{b} -> |{a}-{b}|"))
                break

        return variants


class NonlinearMutator:
    """Nonlinear transform injection/replacement (safe version: only single-parameter functions, not clip-like multi-parameter functions)."""

    # Safe transform list (all single-parameter, won't break syntax)
    SAFE_TRANSFORMS = [
        ("np.abs", "abs"),
        ("np.square", "square"),
        ("np.sqrt", "sqrt"),
        ("np.log1p", "log1p"),
    ]

    def mutate(self, code: str, mode: str) -> list[tuple[str, str, str]]:
        variants = []
        has_nonlinear = any(t[0] in code for t in self.SAFE_TRANSFORMS)

        if not has_nonlinear:
            # Inject nonlinear: find return float(expr) line, wrap with np.abs()
            for pattern in [r"return\s+float\(\s*(.+?)\s*\)\s*$", r"_result\s*=\s*(.+?)\s*$"]:
                m = re.search(pattern, code, re.MULTILINE)
                if m:
                    rhs = m.group(1).strip()
                    inner_stripped = rhs.rstrip()
                    for tf_name, tf_short in self.SAFE_TRANSFORMS[:3]:  # abs, square, sqrt
                        # sqrt needs abs first to prevent negative numbers
                        inner = inner_stripped
                        if tf_name == "np.sqrt":
                            inner = f"np.abs({inner_stripped})"
                            tf_expr = f"np.sqrt({inner})"
                        else:
                            tf_expr = f"{tf_name}({inner})"
                        # Wrap entire expression
                        if pattern.startswith("return"):
                            new_code = code.replace(
                                f"return float({rhs})",
                                f"return float({tf_expr})",
                            )
                        else:
                            new_code = code.replace(
                                f"_result = {rhs}",
                                f"_result = {tf_expr}",
                            )
                        if new_code != code:
                            variants.append((
                                new_code,
                                f"add_{tf_short}",
                                f"Wrap result in {tf_name}()",
                            ))
                    break
        else:
            # Replace existing nonlinear
            for old_tf, old_short in self.SAFE_TRANSFORMS:
                if old_tf not in code:
                    continue
                for new_tf, new_short in self.SAFE_TRANSFORMS:
                    if old_tf == new_tf:
                        continue
                    # sqrt replacement: preserve abs protection of original function
                    if new_tf == "np.sqrt":
                        # Find full old_tf(...) call, replace with np.sqrt(np.abs(...))
                        new_code = _replace_func_call(code, old_tf, lambda inner: f"np.sqrt(np.abs({inner}))")
                    else:
                        new_code = code.replace(old_tf, new_tf)
                    variants.append((
                        new_code,
                        f"{old_short}2{new_short}",
                        f"{old_tf} -> {new_tf}",
                    ))
                # Also try removal (bracket counting correctly handles nesting)
                rm_code = _replace_func_call(code, old_tf, lambda inner: inner)
                if rm_code != code:
                    variants.append((
                        rm_code,
                        f"rm_{old_short}",
                        f"Remove {old_tf}()",
                    ))

        return variants


class ConstantMutator:
    """Tune hardcoded constants: x0.5, x0.75, x0.9, x1.1, x1.25, x1.5, x2.0"""

    SCALES = [0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0]

    def mutate(self, code: str, mode: str) -> list[tuple[str, str, str]]:
        variants = []
        # Find numeric constants in the formula section (exclude idx.get return value -1 and array indices)
        compute_start = code.find("if ")  # safety check usually starts with "if"
        if compute_start < 0:
            # fallback: find after all idx.get lines
            for m in re.finditer(r"idx\.get\(.+\)", code):
                compute_start = max(compute_start, m.end())

        compute_section = code[compute_start:] if compute_start > 0 else code
        setup_section = code[:compute_start] if compute_start > 0 else ""

        const_pattern = r"(?<![\w.>])(\d+\.?\d*)(?![\w'])(?![.\d]*\s*(?:\)|,|idx|window))"
        for m in re.finditer(const_pattern, compute_section):
            val = float(m.group(1))
            if val in (0, 1, -1, 5, 10, 25, 50, 75, 90, 100):
                continue  # skip common defaults
            # Skip slice indices: context has ':' and is inside brackets (e.g. windows[:, 3:, ...] or windows[:, :2, ...])
            if _is_slice_index(compute_section, m.start(), m.end()):
                continue
            for scale in self.SCALES:
                new_val = val * scale
                if abs(new_val - round(new_val, 1)) < 0.001:
                    new_str = f"{new_val:.1f}".rstrip("0").rstrip(".")
                else:
                    new_str = f"{new_val:.4f}".rstrip("0").rstrip(".")
                old_str = m.group(1)
                new_compute = compute_section.replace(old_str, new_str, 1)
                new_code = setup_section + new_compute
                variants.append((
                    new_code,
                    f"const_{old_str}_x{scale}",
                    f"Constant {old_str} -> {new_str}",
                ))

        return variants


# ===================================================================
# Phase 1 orchestration: VariantGenerator
# ===================================================================

class VariantGenerator:
    """Combine all Mutators + ParameterInjector, generate baked variants for each seed factor."""

    def __init__(self, enable_structure_mutation: bool = True, max_param_combos: int = 20):
        self.mutators = (
            [
                AggregatorMutator(),
                TemporalMutator(),
                CrossFeatureMutator(),
                NonlinearMutator(),
                ConstantMutator(),
            ]
            if enable_structure_mutation
            else []
        )
        self.injector = ParameterInjector()
        self.max_param_combos = max_param_combos

    def generate(self, factor: dict) -> list[dict]:
        """Generate all baked variants for one seed factor. Pre-validate to filter illegal code."""
        code = factor["code"]
        mode = factor.get("mode", "row")
        variants = []
        seen_codes: set[str] = set()
        n_dropped = 0

        def _add(code_str: str, mut_name: str, mut_desc: str, param_label: str = ""):
            nonlocal n_dropped

            # Pre-validate
            is_ok, reason = _code_is_valid(code_str)
            if not is_ok:
                n_dropped += 1
                return

            code_hash = hashlib.md5(code_str.encode()).hexdigest()
            if code_hash in seen_codes:
                return
            seen_codes.add(code_hash)

            variant = dict(factor)
            variant["code"] = code_str
            variant["_seed_name"] = factor["name"]
            variant["_mutation"] = mut_name
            variant["_param_label"] = param_label

            name_suffix = mut_name if mut_name != "seed" else "tuned"
            if param_label:
                name_suffix += f"_{param_label.replace('=', '').replace(',', '_')[:30]}"
            variant["name"] = f"{factor['name']}_{name_suffix}"
            variant["description"] = (
                f"{factor.get('description', '')} [{mut_desc}]"
                + (f" [{param_label}]" if param_label else "")
            )
            variants.append(variant)

        # 1. Collect structure variants
        structure_variants = [(code, "seed", "Original seed")]

        for mutator in self.mutators:
            for new_code, mut_name, mut_desc in mutator.mutate(code, mode):
                code_hash = hashlib.md5(new_code.encode()).hexdigest()
                if code_hash not in seen_codes:
                    structure_variants.append((new_code, mut_name, mut_desc))

        # 2. For each structure variant, inject parameters and bake
        for struct_code, mut_name, mut_desc in structure_variants:
            baked_list = self.injector.inject_and_bake(
                struct_code, mode, max_combos=self.max_param_combos,
            )
            if baked_list:
                for baked_code, n_params, param_label in baked_list:
                    _add(baked_code, mut_name, mut_desc, param_label)
            else:
                # Cannot inject parameters, keep original structure variant
                _add(struct_code, mut_name, mut_desc)

        return variants


# ===================================================================
# Validation
# ===================================================================

class TunerValidator:
    """
    Factor validator: aligned with the main loop, uses LGBM train->val holdout evaluation.

    In the main loop, FactorValidator.validate_single_holdout trains LightGBM on the
    train set and evaluates AUC on the val set. This validator strictly replicates the
    same process to ensure AUC from factor_tuner is comparable to AUC in valid_factors.json.

    """

    def __init__(self, kp_train: np.ndarray, labels_train: np.ndarray,
                 kp_val: np.ndarray, labels_val: np.ndarray,
                 flat_attributes: list, num_workers: int = 1,
                 validation_cfg: dict = None, lgbm_jobs: int = 0):
        self.kp_train = kp_train
        self.labels_train = labels_train
        self.kp_val = kp_val
        self.labels_val = labels_val
        self.flat_attributes = flat_attributes
        self.num_workers = num_workers

        # FactorEngine created once, reuses compilation cache (hundreds of evals per seed)
        from src.factor_engine import FactorEngine
        engine_cfg = {"factor_engine": {"max_error_ratio": 0.5, "min_valid_ratio": 0.1}}
        self.engine = FactorEngine(engine_cfg)

        # Reuse the same FactorValidator as the main loop (LGBM holdout validation)
        from src.validator import FactorValidator
        # Build validation config consistent with discovery.yaml; user-provided cfg overrides
        default_validation = {
            "validation": {
                "min_auc": 0.65,
                "min_f1": 0.50,
                "early_stopping_rounds": 20,
                "use_gpu": False,
                "lgbm_params": {
                    "n_estimators": 200, "max_depth": 4,
                    "learning_rate": 0.05, "num_leaves": 31,
                    "random_state": 42, "n_jobs": -1,
                },
            }
        }
        if validation_cfg and "validation" in validation_cfg:
            # User config has a validation section (e.g. discovery.yaml), override defaults
            default_validation["validation"].update(validation_cfg["validation"])

        # Control LGBM n_jobs in parallel mode (can be explicitly specified via --lgbm-jobs)
        # Note: Under ThreadPoolExecutor, each thread's LGBM n_jobs must be 1,
        # otherwise 24 threads x 8 n_jobs = 192 threads will trigger OpenBLAS multi-thread deadlock
        if num_workers > 1:
            cpu_count = os.cpu_count() or 4
            if lgbm_jobs > 0:
                lgbm_n_jobs = lgbm_jobs
            else:
                lgbm_n_jobs = 1  # Thread pool already provides outer parallelism, LGBM single-threaded to avoid deadlock
            default_validation["validation"]["lgbm_params"]["n_jobs"] = lgbm_n_jobs
            log.info(f"Parallel mode: LGBM n_jobs={lgbm_n_jobs} ({cpu_count} cores / {num_workers} workers)")

        self.validator = FactorValidator(default_validation)

        # Window cache: lazy-build by seq_length, avoiding rebuilding million-scale windows for every factor
        self._train_window_cache: dict[int, np.ndarray] = {}
        self._val_window_cache: dict[int, np.ndarray] = {}

    def _get_windows(self, kp: np.ndarray, seq_length: int,
                     cache: dict[int, np.ndarray], tag: str = "?") -> np.ndarray:
        """Get or build causal window matrix [T, seq_length, D] from cache."""
        if seq_length in cache:
            return cache[seq_length]

        T, D = kp.shape
        log.info(
            f"Building window cache seq_length={seq_length} "
            f"({T} frames x {D} dims, {tag})..."
        )
        t0 = time.time()
        if seq_length == 1:
            windows = kp[:, np.newaxis, :].astype(np.float32).copy()
        else:
            windows = np.zeros((T, seq_length, D), dtype=np.float32)
            windows[:seq_length - 1] = np.nan
            from numpy.lib.stride_tricks import sliding_window_view
            valid_start = seq_length - 1
            if valid_start < T:
                swv = sliding_window_view(kp, (seq_length, D))
                swv = swv.reshape(-1, seq_length, D)
                windows[valid_start:] = swv[:, :, :]
        cache[seq_length] = windows
        log.info(f"Window cache complete seq_length={seq_length}, took {time.time() - t0:.1f}s, "
                 f"shape={windows.shape}")
        return windows

    def compute_factor_values(self, factor: dict) -> Optional[dict]:
        """Compute only factor values (no LGBM training), returns {train_factor, val_factor} or None.
        Used for ProcessPoolExecutor path: main thread pre-computes, worker only does LGBM."""
        import os as _os
        _os.environ.setdefault("OMP_NUM_THREADS", "1")
        _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        _os.environ.setdefault("MKL_NUM_THREADS", "1")

        code = factor["code"]
        seq_length = factor.get("seq_length", 1)
        mode = factor.get("mode", "row")
        factor_name = factor.get("name", "?")

        train_windows = self._get_windows(
            self.kp_train, seq_length, self._train_window_cache, tag="train")
        val_windows = self._get_windows(
            self.kp_val, seq_length, self._val_window_cache, tag="val")

        hypothesis = {
            "name": factor_name,
            "code": code,
            "mode": mode,
            "seq_length": seq_length,
        }

        try:
            train_factor = self.engine.compute_factor_batch(
                hypothesis, train_windows, self.flat_attributes)
            val_factor = self.engine.compute_factor_batch(
                hypothesis, val_windows, self.flat_attributes)
        except Exception as e:
            log.debug(f"[FactorEngine] {factor_name} computation failed: {e}")
            return None

        if train_factor is None or val_factor is None:
            return None

        nan_ratio = np.isnan(train_factor).mean()
        if nan_ratio > 0.8:
            return None

        return {"train_factor": train_factor, "val_factor": val_factor}

    def compute_factor_only(self, factor: dict) -> Optional[dict]:
        """
        Compute factor values only (train + val), no LGBM training.

        Used by Phase 2 two-stage pipeline: factor computation (thread pool) + LGBM training (process pool).
        Returns dict containing train_factor/val_factor, or None on failure.
        """
        import os as _os
        _os.environ.setdefault("OMP_NUM_THREADS", "1")
        _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        _os.environ.setdefault("MKL_NUM_THREADS", "1")

        import numpy as np
        code = factor["code"]
        seq_length = factor.get("seq_length", 1)
        mode = factor.get("mode", "row")
        factor_name = factor.get("name", "?")

        train_windows = self._get_windows(
            self.kp_train, seq_length, self._train_window_cache, tag="train")
        val_windows = self._get_windows(
            self.kp_val, seq_length, self._val_window_cache, tag="val")

        hypothesis = {
            "name": factor_name,
            "code": code,
            "mode": mode,
            "seq_length": seq_length,
        }

        try:
            train_factor = self.engine.compute_factor_batch(
                hypothesis, train_windows, self.flat_attributes)
            val_factor = self.engine.compute_factor_batch(
                hypothesis, val_windows, self.flat_attributes)
        except Exception:
            return None

        if train_factor is None or val_factor is None:
            return None

        nan_ratio = np.isnan(train_factor).mean()
        if nan_ratio > 0.8:
            return None

        return {
            "factor_dict": factor,
            "train_factor": np.asarray(train_factor, dtype=np.float32),
            "val_factor": np.asarray(val_factor, dtype=np.float32),
        }

    def compute_and_eval(self, factor: dict, seed_auc: float = 0.0) -> Optional[dict]:
        """
        Compute factor values (train + val), then validate with LGBM holdout.

        Fully aligned with the main loop: train LightGBM on train set, evaluate AUC on val set.
        Returns updated factor dict (including best_auc, etc.), or None on failure.

        seed_auc: AUC of the seed factor, used for early elimination (if 40 trees are significantly
                  below the seed, skip the full 200 trees)
        """
        # Thread safety: prevent OpenBLAS/MKL deadlock under Python multi-threading
        # (ThreadPoolExecutor path does not go through _worker_* functions, needs explicit restriction here)
        import os as _os
        _os.environ.setdefault("OMP_NUM_THREADS", "1")
        _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        _os.environ.setdefault("MKL_NUM_THREADS", "1")

        code = factor["code"]
        seq_length = factor.get("seq_length", 1)
        mode = factor.get("mode", "row")
        factor_name = factor.get("name", "?")

        # Get or build windows from cache (built once per seq_length)
        train_windows = self._get_windows(
            self.kp_train, seq_length, self._train_window_cache, tag="train")
        val_windows = self._get_windows(
            self.kp_val, seq_length, self._val_window_cache, tag="val")

        hypothesis = {
            "name": factor_name,
            "code": code,
            "mode": mode,
            "seq_length": seq_length,
        }

        # Reuse TunerValidator-level engine (compilation cache shared across variants)
        try:
            train_factor = self.engine.compute_factor_batch(
                hypothesis, train_windows, self.flat_attributes)
            val_factor = self.engine.compute_factor_batch(
                hypothesis, val_windows, self.flat_attributes)
        except Exception as e:
            log.debug(f"[FactorEngine] {factor_name} computation failed: {e}")
            return None

        if train_factor is None or val_factor is None:
            return None

        # NaN pre-filter: skip if too few valid values (avoid wasting LGBM time)
        nan_ratio = np.isnan(train_factor).mean()
        if nan_ratio > 0.8:
            return None

        # --- Early elimination: 40-tree fast pre-check ---
        if seed_auc > 0:
            try:
                import lightgbm as lgb
                from sklearn.metrics import roc_auc_score
                quick_lgbm = lgb.LGBMClassifier(
                    n_estimators=40, max_depth=3, num_leaves=15,
                    learning_rate=0.05, random_state=42, n_jobs=1,
                    verbose=-1,
                )
                nan_mask = np.isnan(train_factor)
                t_valid = train_factor[~nan_mask]
                l_valid = self.labels_train[~nan_mask]
                if len(t_valid) < 10:
                    return None
                quick_lgbm.fit(t_valid.reshape(-1, 1), l_valid)
                v_valid = ~np.isnan(val_factor)
                if v_valid.sum() < 5:
                    return None
                proba = quick_lgbm.predict_proba(val_factor[v_valid].reshape(-1, 1))
                valid_cls = np.unique(self.labels_val[v_valid])
                aucs = []
                for c in valid_cls:
                    if c < proba.shape[1]:
                        try:
                            aucs.append(roc_auc_score(
                                (self.labels_val[v_valid] == c).astype(int),
                                proba[:, c]))
                        except Exception:
                            pass
                quick_auc = max(aucs) if aucs else 0.0
                if quick_auc < seed_auc - 0.05:
                    return None  # Significantly below seed, skip full training
            except Exception:
                pass  # Pre-check failure does not block, continue full evaluation

        # Use FactorValidator holdout validation (identical to main loop)
        val_result = self.validator.validate_single_holdout(
            train_factor, self.labels_train,
            val_factor, self.labels_val,
        )

        if not val_result:
            return None

        best_auc = val_result.get("best_auc", 0)
        if best_auc == 0:
            return None

        factor["best_auc"] = float(best_auc)
        factor["best_f1"] = val_result.get("best_f1", 0)
        factor["best_class"] = val_result.get("best_class", "")
        factor["valid_classes"] = val_result.get("valid_classes", [])
        factor["per_class"] = val_result.get("per_class", {})
        factor["n_train"] = val_result.get("n_train", 0)
        factor["n_val"] = val_result.get("n_val", 0)
        return factor


# ===================================================================
# Phase 1 helper: update seed batch tracking variables with variant evaluation results
# ===================================================================

def _update_seed_batch(batch: dict, result: dict, factor_dict: dict,
                        task_type: str, extra, seed_auc: float):
    """Accumulate a single variant evaluation result into the seed batch tracking variables."""
    var_auc = float(result.get("best_auc", 0))
    result["_seed_auc"] = batch.get("stored_auc", 0)
    result["_auc_improvement"] = round(var_auc - seed_auc, 4)
    result["_mutation"] = factor_dict.get("_mutation", task_type)

    llm_params = batch.get("llm_params", [])

    if task_type == "llm_param":
        batch["n_param_evaled"] = batch.get("n_param_evaled", 0) + 1
        if llm_params:
            result["_llm_params"] = {
                p["name"]: extra[p["name"]] for p in llm_params
            }
        if var_auc > batch.get("best_llm_auc", 0):
            batch["best_llm_auc"] = var_auc
            batch["best_llm_label"] = result.get("_param_label", "")
    elif task_type == "combined":
        batch["n_combined_evaled"] = batch.get("n_combined_evaled", 0) + 1
        if llm_params:
            result["_llm_params"] = {
                p["name"]: extra[p["name"]] for p in llm_params
            }
        if var_auc > batch.get("best_combined_auc", 0):
            batch["best_combined_auc"] = var_auc
            batch["best_combined_label"] = result.get("_mutation", "")
    else:
        batch["n_evaled"] = batch.get("n_evaled", 0) + 1
        if var_auc > batch.get("best_struct_auc", 0):
            batch["best_struct_auc"] = var_auc
            batch["best_struct_mutation"] = factor_dict.get("_mutation", "?")

    if var_auc > batch.get("best_auc", 0):
        batch["best_auc"] = var_auc
        batch["best_variant"] = result
        if task_type in ("llm_param", "combined"):
            batch["llm_param_improved"] = True


# ===================================================================
# Main pipeline
# ===================================================================

def tune_factors(
    seed_factors: list[dict],
    kp_train: np.ndarray,
    labels_train: np.ndarray,
    kp_val: np.ndarray,
    labels_val: np.ndarray,
    flat_attributes: list,
    num_workers: int = 1,
    enable_structure_mutation: bool = True,
    max_param_combos: int = 20,
    max_variants_per_seed: int = 50,
    dry_run: bool = False,
    validation_cfg: dict = None,
    lgbm_jobs: int = 0,
    output_path: str = "",
    progress_path: str = "",
    completed_set: set = None,
    initial_results: list[dict] = None,
) -> list[dict]:
    """
    Tuning pipeline: structure mutation -> parameter injection (bake) -> per-variant validation -> keep best per seed

    Validation aligned with the main loop: LGBM trains on train set, evaluates AUC on val set.

    Supports resume: skip completed factors via completed_set, incrementally write to output_path
    and update progress_path after each factor completes.
    initial_results retains existing results on resume; new results are appended to this list.
    """
    generator = VariantGenerator(enable_structure_mutation, max_param_combos)
    validator = TunerValidator(
        kp_train, labels_train, kp_val, labels_val,
        flat_attributes, num_workers, validation_cfg=validation_cfg,
        lgbm_jobs=lgbm_jobs,
    )

    completed = set(completed_set) if completed_set else set()

    all_tuned = list(initial_results) if initial_results else []
    total_variants = 0
    total_improved = 0
    total_skipped = 0
    total_resumed = 0
    t_start = time.time()

    # --- Background status monitor ---
    monitor = StatusMonitor(interval=30)
    monitor.total_factors = len(seed_factors)
    monitor.completed_count = len(all_tuned)  # Actual number of existing results (not completed_set)
    # Count already improved from existing results (_auc_improvement > 0.001)
    total_improved = sum(1 for f in all_tuned if f.get("_auc_improvement", 0) > 0.001)
    monitor.total_improved = total_improved
    monitor.start()

    # Count skipped factors
    n_skip = sum(1 for f in seed_factors if f.get("name", "") in completed)
    n_todo = len(seed_factors) - n_skip
    log.info(f"Seed factors: {len(seed_factors)} ({n_skip} completed, {n_todo} pending)")
    log.info(f"Structure mutation: {'enabled' if enable_structure_mutation else 'disabled'}")
    log.info(f"Max parameter combinations: {max_param_combos}")

    # Group by seq_length, process group by group: build windows -> process factors -> release cache, control peak memory
    from collections import defaultdict
    groups: dict[int, list[dict]] = defaultdict(list)
    for f in seed_factors:
        sl = f.get("seq_length", 1)
        groups[sl].append(f)

    sorted_seq_lens = sorted(groups)
    log.info(
        f"Grouped by seq_length: {[(sl, len(groups[sl])) for sl in sorted_seq_lens]}")

    for sl in sorted_seq_lens:
        monitor.current_seq_length = sl
        group = groups[sl]
        log.info(
            f"=== seq_length={sl} group ({len(group)} factors) === "
            f"Build windows + process -> release cache")

        # Build windows for this group (only this seq_length)
        validator._get_windows(validator.kp_train, sl,
                               validator._train_window_cache, tag="train")
        validator._get_windows(validator.kp_val, sl,
                               validator._val_window_cache, tag="val")

        # ============================================================
        # Phase 1: Parallel compute all seed AUCs + generate variant tasks
        # ============================================================
        # n_done_before_group: completed before this group started
        n_done_before_group = len(all_tuned)

        # Phase 1a: Collect pending seeds + generate variants (does not depend on AUC, CPU-light)
        pending_seeds: list[dict] = []  # Each element is a seed metadata dict
        collected_in_group = 0

        for seed in group:
            t_factor_start = time.time()
            seed_name = seed.get("name", f"?")
            seed_code = seed.get("code", "")
            seq_len = seed.get("seq_length", 1)

            if seed_name in completed:
                total_resumed += 1
                log.debug(f"  {seed_name[:60]} skipped (already completed)")
                continue

            variants = generator.generate(seed)
            variants = variants[:max_variants_per_seed]
            n_variants_expected = len(variants)
            total_variants += n_variants_expected

            if dry_run:
                continue

            collected_in_group += 1
            monitor.current_factor = f"queuing: {seed_name[:50]}..."
            # Do not advance completed_count: factor not yet done, only queued
            log.info(
                f"[{n_done_before_group + collected_in_group}/{len(seed_factors)}] "
                f"{seed_name[:60]} queued "
                f"(seq={seq_len}, mode={seed.get('mode', 'row')})..."
            )

            # --- LLM parameterization (does not depend on AUC, done early) ---
            llm_params = _extract_llm_params(seed_code)
            auto_param_code = None
            if not llm_params:
                auto_param_code, llm_params = _auto_parameterize(seed_code)
                if llm_params:
                    log.debug(
                        f"  Auto-parameterized: {len(llm_params)} constants -> "
                        f"{', '.join(p['name'] + '=' + str(p['default']) for p in llm_params)}"
                    )

            if llm_params:
                param_combos = _generate_llm_param_combos(llm_params, max_combos=max_param_combos * 2)
                log.debug(
                    f"  Detected {len(llm_params)} LLM parameters "
                    f"({', '.join(p['name'] + '=' + str(p['default']) for p in llm_params)}), "
                    f"{len(param_combos)} combinations to search"
                )

            # --- Generate variant tasks (use stored_auc as fallback) ---
            tasks: list = []

            if llm_params:
                for param_dict in param_combos:
                    is_all_default = all(
                        abs(param_dict[p["name"]] - p["default"]) < 0.0001
                        for p in llm_params
                    )
                    if is_all_default:
                        continue
                    base_code = auto_param_code if auto_param_code else seed_code
                    param_code = _apply_llm_params(base_code, param_dict)
                    if param_code == seed_code:
                        continue
                    param_seed = dict(seed)
                    param_seed["code"] = param_code
                    param_seed["_mutation"] = "llm_param_search"
                    param_seed["_param_label"] = ",".join(
                        f"{p['name']}={param_dict[p['name']]:.4g}" for p in llm_params
                    )
                    param_seed["name"] = f"{seed_name}_llmp"
                    tasks.append((param_seed, "llm_param", param_dict))

            for variant in variants:
                tasks.append((variant, "variant", None))

            n_combined = 0
            if llm_params and len(variants) > 0:
                MAX_COMBINED_STRUCT = 5
                MAX_COMBINED_PARAMS = 5
                comb_param_combos = _generate_llm_param_combos(llm_params, max_combos=MAX_COMBINED_PARAMS)
                for variant in variants[:MAX_COMBINED_STRUCT]:
                    if variant.get("_mutation", "") == "seed":
                        continue
                    variant_code = variant["code"]
                    for param_dict in comb_param_combos:
                        is_all_default = all(
                            abs(param_dict[p["name"]] - p["default"]) < 0.0001
                            for p in llm_params
                        )
                        if is_all_default:
                            continue
                        base = auto_param_code if auto_param_code else variant_code
                        if variant.get("_mutation", "") != "seed":
                            base = variant_code
                        combined_code = _apply_llm_params(base, param_dict)
                        if combined_code == variant_code:
                            continue
                        combined = dict(variant)
                        combined["code"] = combined_code
                        combined["_mutation"] = variant.get("_mutation", "variant") + "+llmp"
                        combined["_param_label"] = ",".join(
                            f"{p['name']}={param_dict[p['name']]:.4g}" for p in llm_params
                        )
                        combined["name"] = variant["name"] + "_llmp"
                        tasks.append((combined, "combined", param_dict))
                        n_combined += 1
                if n_combined > 0:
                    log.debug(f"  Combined tasks: {MAX_COMBINED_STRUCT} structures x {MAX_COMBINED_PARAMS} params = {n_combined} combined variants")

            stored_auc = float(seed.get("best_auc", 0))
            pending_seeds.append({
                "seed": seed,
                "seed_name": seed_name,
                "seq_len": seq_len,
                "t_factor_start": t_factor_start,
                "stored_auc": stored_auc,
                "llm_params": llm_params,
                "n_variants_expected": n_variants_expected,
                "tasks": tasks,
            })

        # Phase 1b: Parallel compute all seed AUCs (process pool, avoid LGBM thread conflicts)
        seed_batches: list[dict] = []
        all_tasks: list[tuple] = []
        n_pending = len(pending_seeds)
        if n_pending == 0:
            log.info(f"  No pending seeds, skipping variant evaluation")
        else:
            # Set shared variables for child processes after fork
            global _shared_train_windows, _shared_val_windows
            global _shared_labels_train, _shared_labels_val
            global _shared_flat_attributes, _shared_validation_cfg_snippet
            _shared_train_windows = validator._train_window_cache.get(sl)
            _shared_val_windows = validator._val_window_cache.get(sl)
            _shared_labels_train = labels_train
            _shared_labels_val = labels_val
            _shared_flat_attributes = list(flat_attributes)
            _shared_validation_cfg_snippet = {
                "lgbm_params": dict(validator.validator.lgbm_params),
                "min_auc": validator.validator.min_auc,
                "min_f1": validator.validator.min_f1,
                "early_stopping_rounds": validator.validator.early_stopping_rounds,
            }

            n_seed_workers = min(num_workers, n_pending) if num_workers > 1 else 1
            log.info(
                f"  Computing {n_pending} seed AUCs in parallel -> {n_seed_workers} processes"
            )
            # Single factor timeout: 600 seconds (batch mode may be slow on 1M+ frames)
            _SEED_TIMEOUT = 600
            seed_futures: dict = {}
            if n_seed_workers > 1:
                # Process pool: each child process has independent memory space, LGBM does not interfere
                ctx = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=n_seed_workers, mp_context=ctx) as seed_pool:
                    for i, ps in enumerate(pending_seeds):
                        fut = seed_pool.submit(
                            _worker_eval_factor,
                            {"factor": dict(ps["seed"]), "seed_auc": 0.0})
                        seed_futures[fut] = i
                    n_seed_done = 0
                    for future in as_completed(seed_futures):
                        idx = seed_futures[future]
                        ps = pending_seeds[idx]
                        n_seed_done += 1
                        try:
                            seed_result = future.result(timeout=_SEED_TIMEOUT)
                        except Exception:
                            seed_result = None
                        ps["seed_result"] = seed_result
                        ps["seed_auc"] = float(
                            seed_result.get("best_auc", 0)) if seed_result else 0
                        # Update progress: every ~5% or first factor
                        if n_seed_done == 1 or n_seed_done % max(1, n_pending // 20) == 0:
                            monitor.completed_count = n_done_before_group + n_seed_done
                            monitor.current_factor = (
                                f"(seed AUC {n_seed_done}/{n_pending})")
            else:
                for ps in pending_seeds:
                    ps["seed_result"] = validator.compute_and_eval(dict(ps["seed"]))
                    ps["seed_auc"] = float(
                        ps["seed_result"].get("best_auc", 0)) if ps["seed_result"] else 0

            # Phase 1c: Build seed batches (filter failed seeds) + collect variant tasks
            # completed_count already updated progressively in Phase 1b, no jump here
            monitor.current_factor = f"(building {n_pending} seed batches)"
            log.info(
                f"  All seed AUCs done, building {n_pending} seed batches "
                f"+ {sum(len(ps['tasks']) for ps in pending_seeds)} variant tasks"
            )
            for ps in pending_seeds:
                seed_auc = ps["seed_auc"]
                seed_name = ps["seed_name"]
                seed = ps["seed"]
                seed_result = ps["seed_result"]

                if seed_auc == 0:
                    # Do not mark as completed: computation failure may be transient (OOM/process crash, etc.),
                    # retry on resume rather than permanently skipping
                    total_skipped += 1
                    n_done_before_group -= 1  # Factor not completed, rollback count
                    log.info(
                        f"  {seed_name[:60]} x computation failed/invalid, will retry on resume "
                        f"({time.time() - ps['t_factor_start']:.0f}s)"
                    )
                    # Still append original seed to output (keep AUC), but do not mark completed
                    all_tuned.append(dict(seed))
                    _save_incremental_output(output_path, all_tuned)
                    continue

                seed_best_class = seed_result.get("best_class", "?") if seed_result else "?"
                seed_best_f1 = seed_result.get("best_f1", 0) if seed_result else 0

                batch = {
                    "seed": seed,
                    "seed_name": seed_name,
                    "seq_len": ps["seq_len"],
                    "t_factor_start": ps["t_factor_start"],
                    "seed_auc": seed_auc,
                    "stored_auc": ps["stored_auc"],
                    "seed_best_class": seed_best_class,
                    "seed_best_f1": seed_best_f1,
                    "llm_params": ps["llm_params"],
                    "n_variants_expected": ps["n_variants_expected"],
                    "best_variant": None,
                    "best_auc": seed_auc,
                    "best_llm_auc": 0.0,
                    "best_llm_label": "",
                    "best_struct_auc": 0.0,
                    "best_struct_mutation": "",
                    "best_combined_auc": 0.0,
                    "best_combined_label": "",
                    "llm_param_improved": False,
                    "n_param_evaled": 0,
                    "n_evaled": 0,
                    "n_combined_evaled": 0,
                }
                batch_idx = len(seed_batches)
                seed_batches.append(batch)
                for factor_dict, task_type, extra in ps["tasks"]:
                    all_tasks.append((factor_dict, task_type, extra, batch_idx))

            # Phase 1c end: failed seeds already appended to all_tuned, correct progress
            monitor.completed_count = len(all_tuned)

        # ============================================================
        # Phase 2: Cross-factor parallel evaluation (process pool, isolate LGBM to prevent segfaults)
        # ============================================================
        if num_workers > 1 and len(all_tasks) > 1:
            n_workers = min(num_workers, len(all_tasks))
            log.info(
                f"  Cross-factor batch: {len(all_tasks)} variant tasks -> "
                f"{n_workers} processes (covering {len(seed_batches)} seed factors)"
            )
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                future_map = {}
                for factor_dict, task_type, extra, batch_idx in all_tasks:
                    batch = seed_batches[batch_idx]
                    fut = executor.submit(
                        _worker_eval_factor,
                        {"factor": factor_dict, "seed_auc": batch["seed_auc"]})
                    future_map[fut] = (factor_dict, task_type, extra, batch_idx)

                n_completed_variants = 0
                n_total_variants = len(all_tasks)
                for future in as_completed(future_map):
                    factor_dict, task_type, extra, batch_idx = future_map[future]
                    batch = seed_batches[batch_idx]
                    n_completed_variants += 1
                    # Variant progress (do not change completed_count, factors counted only when done)
                    if n_completed_variants % max(1, n_total_variants // 20) == 0:
                        monitor.current_factor = (
                            f"(variant evaluation {n_completed_variants}/{n_total_variants})")
                    try:
                        result = future.result()
                    except Exception:
                        result = None
                        monitor.add_error(
                            f"[{batch['seed_name'][:40]}] "
                            f"{factor_dict.get('_mutation','?')} CRASHED")

                    if result is not None:
                        _update_seed_batch(
                            batch, result, factor_dict, task_type, extra,
                            batch["seed_auc"])
        elif len(all_tasks) > 1:
            # === Single-threaded serial ===
            for factor_dict, task_type, extra, batch_idx in all_tasks:
                batch = seed_batches[batch_idx]
                result = validator.compute_and_eval(
                    factor_dict, batch["seed_auc"])
                if result is not None:
                    _update_seed_batch(
                        batch, result, factor_dict, task_type, extra,
                        batch["seed_auc"])

        # ============================================================
        # Phase 3: Per-factor result summary + logging + incremental save
        # ============================================================
        n_done_phase3 = len(all_tuned)  # Number completed before this group
        for i, batch in enumerate(seed_batches):
            seed = batch["seed"]
            seed_name = batch["seed_name"]
            seed_auc = batch["seed_auc"]
            stored_auc = batch["stored_auc"]
            seed_best_class = batch["seed_best_class"]
            seed_best_f1 = batch["seed_best_f1"]
            best_variant = batch["best_variant"]
            best_auc = batch["best_auc"]
            n_param_evaled = batch["n_param_evaled"]
            n_evaled = batch["n_evaled"]
            n_combined_evaled = batch["n_combined_evaled"]
            best_llm_auc = batch["best_llm_auc"]
            best_struct_auc = batch["best_struct_auc"]
            best_struct_mutation = batch["best_struct_mutation"]
            llm_param_improved = batch["llm_param_improved"]
            seq_len = batch["seq_len"]
            t_factor_start = batch["t_factor_start"]

            t_factor = time.time() - t_factor_start
            improved = best_variant is not None and best_auc > seed_auc + 0.002
            source_tag = ""
            if improved and llm_param_improved and best_llm_auc >= best_struct_auc:
                source_tag = "LLM param search"
            elif improved:
                source_tag = f"structure mutation({best_struct_mutation})"

            n_done_total = n_done_phase3 + i + 1
            elapsed = time.time() - t_start
            avg_per = elapsed / max(n_done_total, 1)
            remaining = avg_per * (len(seed_factors) - n_done_total)

            # --- Detailed result block ---
            lines = []
            lines.append(
                f"[{n_done_total:>4}/{len(seed_factors)}] {seed_name[:60]}  "
                f"seq={seq_len}  mode={seed.get('mode', 'row')}  "
                f"{t_factor:.0f}s"
            )
            lines.append(
                f"       seed:     AUC={_pct(seed_auc):>7s}  "
                f"class={seed_best_class:<12s}  F1={seed_best_f1:.3f}"
            )

            # All variants summary
            total_evaled = n_param_evaled + n_evaled + n_combined_evaled
            if total_evaled > 0:
                all_delta = _delta_pct(best_auc - seed_auc)
                all_icon = "up" if best_auc > seed_auc + 0.002 else "->"
                lines.append(
                    f"       variants:  {total_evaled:>4} total  "
                    f"{all_icon} best AUC={_pct(best_auc):>7s}  "
                    f"({all_delta})  [{source_tag}]"
                )
            else:
                lines.append(f"       variants:  empty  all failed")

            # Composite result
            if improved:
                total_improved += 1
                monitor.total_improved = total_improved
                all_tuned.append(best_variant)
                lines.append(
                    f"       v IMPROVED: {_pct(seed_auc)} -> {_pct(best_auc)}  "
                    f"({_delta_pct(best_auc - seed_auc)})  source={source_tag}"
                )
            else:
                all_tuned.append(dict(seed))
                lines.append(
                    f"       x No improvement.  Best AUC={_pct(best_auc)}"
                )

            # ETA (every 10 or first)
            if n_done_total % 10 == 0 or n_done_total == 1:
                eta_str = f"  Est. remaining {remaining:.0f}s ({remaining/60:.0f}min)" if remaining < 3600 else f"  Est. remaining {remaining/3600:.1f}h"
                lines[-1] += f"  | Improved {total_improved}{eta_str}"

            log.info("\n".join(lines))

            # --- Mark completed + incremental save ---
            completed.add(seed_name)
            _save_progress(progress_path, completed)
            _save_incremental_output(output_path, all_tuned)

        # Release window cache for current seq_length, free memory for next group
        validator._train_window_cache.clear()
        validator._val_window_cache.clear()
        log.info(f"  seq_length={sl} group complete, cache released")

    log.info(
        f"Done: {len(all_tuned)} factor results "
        f"({total_improved} AUC improvements, {total_resumed} resumed/skipped, "
        f"{total_skipped} failed pending retry)"
    )
    monitor.stop()
    return all_tuned


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Joint search over factor mathematical form + parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", default=r"memory/valid_factors_deduped.json",
                        help="Input factor JSON")
    parser.add_argument("--output", default=r"memory/tuned_factors.json",
                        help="Output tuned factors JSON")
    parser.add_argument("--config-common", default=r"config/seq/1.yaml",
                        help="Common configuration file")
    parser.add_argument("--config-validation", default=r"config/validation.yaml",
                        help="Synthetic validation configuration file")
    parser.add_argument("--num-workers", type=int, default=24,
                        help="Parallel workers (default 96)")
    parser.add_argument("--lgbm-jobs", type=int, default=8,
                        help="Parallel threads per LGBM (0=auto: cpu_count//num_workers)")
    parser.add_argument("--no-structure-mutation", action="store_true",
                        help="Parameter search only, skip formula structure mutation")
    parser.add_argument("--max-param-combos", type=int, default=20,
                        help="Max parameter combinations per structure variant (default 20)")
    parser.add_argument("--max-factors", type=int, default=0,
                        help="Max factors to process (0=all)")
    parser.add_argument("--max-variants-per-seed", type=int, default=50,
                        help="Max variants per seed (default 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only count variants, skip actual computation")
    parser.add_argument("--top-n", type=int, default=0,
                        help="Only tune top N seeds by AUC (0=all)")
    parser.add_argument("--progress-file", default=r"memory/tuner_progress.json",
                        help="Progress file path for resume (default memory/tuner_progress.json)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore progress file and existing output, start from scratch")
    args = parser.parse_args()

    # Load factors
    with open(args.input, "r", encoding="utf-8") as f:
        all_factors = json.load(f)
    log.info(f"Loaded {len(all_factors)} factors: {args.input}")

    if args.top_n > 0:
        all_factors.sort(key=lambda f: f.get("best_auc", 0), reverse=True)
        all_factors = all_factors[:args.top_n]
        log.info(f"Taking Top-{args.top_n} AUC factors")

    if args.max_factors > 0:
        all_factors = all_factors[:args.max_factors]
        log.info(f"Limiting to {args.max_factors} factors")

    if args.dry_run:
        generator = VariantGenerator(not args.no_structure_mutation)
        total = 0
        for seed in all_factors:
            variants = generator.generate(seed)
            total += len(variants[:args.max_variants_per_seed])
        log.info(f"[DRY RUN] Estimated total variants: {total} ({len(all_factors)} seeds x ~{total//max(1,len(all_factors))} variants)")
        return

    # --- Resume: load progress & existing output ---
    output_path = str(Path(args.output))
    progress_path = str(Path(args.progress_file))

    if args.no_resume:
        completed_set = set()
        existing_results = []
        log.info("--no-resume: ignoring progress file, starting from scratch")
    else:
        completed_set = _load_progress(progress_path)
        existing_results, output_names = _load_existing_output(output_path)
        # Merge completed sets from both sources
        completed_set |= output_names
        if existing_results:
            log.info(f"Resume mode: {len(existing_results)} existing results, "
                     f"{len(completed_set)} seed factors completed")

    # Load data (train/val split, consistent with main loop factor mining)
    log.info("Loading train/validation data...")
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

    log.info(f"Training set: {kp_train.shape[0]} frames, Validation set: {kp_val.shape[0]} frames, "
             f"D={kp_train.shape[1]}, num_classes={len(set(labels_train))}")

    # Tune (pass progress and existing results for resume + incremental save)
    tuned = tune_factors(
        seed_factors=all_factors,
        kp_train=kp_train,
        labels_train=labels_train,
        kp_val=kp_val,
        labels_val=labels_val,
        flat_attributes=flat_attributes,
        num_workers=args.num_workers,
        enable_structure_mutation=not args.no_structure_mutation,
        max_param_combos=args.max_param_combos,
        max_variants_per_seed=args.max_variants_per_seed,
        dry_run=False,
        validation_cfg=cfg,
        lgbm_jobs=args.lgbm_jobs,
        output_path=output_path,
        progress_path=progress_path,
        completed_set=completed_set,
        initial_results=existing_results,
    )

    # Final save (incremental writes cover most cases, this is a safety net)
    tuned_output = Path(args.output)
    tuned_output.parent.mkdir(parents=True, exist_ok=True)
    with open(tuned_output, "w", encoding="utf-8") as f:
        json.dump(tuned, f, ensure_ascii=False, indent=2)

    log.info(f"Saved {len(tuned)} factors: {tuned_output}")

    # Statistics (percentage display)
    improved = sum(1 for f in tuned if f.get("_auc_improvement", 0) > 0.001)
    mean_imp = sum(f.get("_auc_improvement", 0) for f in tuned) / max(len(tuned), 1)
    log.info(f"Factors with AUC improvement: {improved}/{len(tuned)}, average improvement={_delta_pct(mean_imp)}")


if __name__ == "__main__":
    main()
