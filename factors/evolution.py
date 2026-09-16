#!/usr/bin/env python3
"""
evolution.py
Genetic Programming factor evolution engine -- based on DEAP library.

Crosses and evolves existing factors from memory/valid_factors.json to generate new candidate factors.
Output format is fully compatible with valid_factors.json and can be directly consumed by train_behavior.py.

Core ideas:
  - Extract (feature, aggregator) pairs from existing factors as DEAP terminals
  - Arithmetic operators (add/sub/mul/div) + unary operators (neg/abs/sqrt/square/log1p) as DEAP primitives
  - Use DEAP eaMuPlusLambda (mu+lambda) evolutionary algorithm, single-point crossover, uniform mutation
  - Proxy fitness = novelty + complexity + feature density + rarity + uniqueness
  - After evolution, convert DEAP expression tree to FactorEngine-compatible Python code

Usage:
  python evolution.py
  python evolution.py --factors memory/valid_factors.json --output memory/evolved_factors.json
  python evolution.py --pop-size 256 --generations 80 --survivors 200
  python evolution.py --mu 128 --lambda 256

Dependencies: deap, numpy (pip install deap)
"""


import sys
from pathlib import Path
# Allow running as a script from a subdirectory (e.g. `python mining/discovery.py`)
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import argparse
import atexit
import json
import functools
import logging
import os
import pickle
import random
import re
import shutil
import sys
import tempfile
import textwrap
import warnings

import yaml
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# DEAP imports
try:
    from deap import base, creator, gp, tools, algorithms
except ImportError:
    print("Error: deap library required. Run: pip install deap", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("factor_evolution")

# ===================================================================
# Multiprocess parallel evaluation infrastructure (Windows spawn safe, memmap shared data)
# ===================================================================
_EVO_WORKER_STATE: Dict[str, Any] = {}
_EVO_MMAP_DIR: str = ""
_EVO_MMAP_FILES: List[str] = []


def _evo_cleanup_mmap():
    """Clean up memmap temporary directory."""
    global _EVO_MMAP_DIR, _EVO_MMAP_FILES
    if _EVO_MMAP_DIR and os.path.isdir(_EVO_MMAP_DIR):
        try:
            shutil.rmtree(_EVO_MMAP_DIR, ignore_errors=True)
        except Exception:
            pass
    _EVO_MMAP_DIR = ""
    _EVO_MMAP_FILES.clear()


def _evo_worker_init(mmap_dir: str, pair_arrays_meta: dict, seq_labels_meta: dict,
                     classes_sorted: list, tindex_to_pair: dict,
                     use_gpu: bool, gpu_backend: str,
                     n_estimators: int, cv_folds: int, reduced_cv: bool):
    """Worker process initialization: load precomputed arrays from stacked memmap into _EVO_WORKER_STATE."""
    state: dict = {}
    state["pair_arrays"] = {}
    for L_str, meta in pair_arrays_meta.items():
        L = int(L_str)
        fpath = os.path.join(mmap_dir, meta["fname"])
        N = meta["shape"][0]
        row_map = meta["row_map"]
        if not os.path.exists(fpath):
            continue
        # Open memmap: shape [K, N]
        K = len(row_map)
        stacked = np.memmap(fpath, dtype=np.float32, mode="r", shape=(K, N))
        state["pair_arrays"][L] = {}
        for pair_key, row_idx in row_map.items():
            state["pair_arrays"][L][tuple(pair_key.split("|||"))] = stacked[row_idx]

    state["seq_labels"] = {}
    for L_str, info in seq_labels_meta.items():
        fpath = os.path.join(mmap_dir, info["fname"])
        if os.path.exists(fpath):
            state["seq_labels"][int(L_str)] = np.memmap(
                fpath, dtype=np.dtype(info["dtype"]), mode="r",
                shape=tuple(info["shape"]),
            )

    state["classes_sorted"] = list(classes_sorted)
    state["tindex_to_pair"] = {int(k): tuple(v) for k, v in tindex_to_pair.items()}
    state["use_gpu"] = use_gpu
    state["gpu_backend"] = gpu_backend
    state["n_estimators"] = n_estimators
    state["cv_folds"] = cv_folds
    state["reduced_cv"] = reduced_cv

    _EVO_WORKER_STATE.clear()
    _EVO_WORKER_STATE.update(state)


def _evo_worker_evaluate(ind_pickle: bytes) -> tuple:
    """Worker process evaluates a DEAP individual, returns (fitness_value,)."""
    ind = pickle.loads(ind_pickle)
    state = _EVO_WORKER_STATE
    if not state:
        return (0.0,)

    pair_arrays = state["pair_arrays"]
    seq_labels = state["seq_labels"]
    classes_sorted = state["classes_sorted"]
    tindex_to_pair = state["tindex_to_pair"]
    use_gpu = state["use_gpu"]
    gpu_backend = state["gpu_backend"]
    n_estimators = state["n_estimators"]
    cv_folds = state["cv_folds"]
    reduced_cv = state["reduced_cv"]

    return _eval_individual_impl(ind, pair_arrays, seq_labels, classes_sorted,
                                  tindex_to_pair, use_gpu, gpu_backend,
                                  n_estimators, cv_folds, reduced_cv)


def _eval_individual_impl(ind, pair_arrays, seq_labels, classes_sorted,
                           tindex_to_pair, use_gpu, gpu_backend,
                           n_estimators, cv_folds, reduced_cv) -> tuple:
    """Actual individual evaluation implementation (shared by main and worker processes)."""
    _clip = np.clip
    _abs = np.abs
    _sqrt = np.sqrt
    _log1p = np.log1p
    _isfinite = np.isfinite
    _where = np.where
    _sign = np.sign
    _errstate = np.errstate

    tree_list = list(ind)
    L = 5
    for candidate_L in [5, 15, 30, 60, 1]:
        if candidate_L in pair_arrays:
            L = candidate_L
            break

    pair_map = pair_arrays.get(L, {})
    labels = seq_labels.get(L)
    if pair_map is None or labels is None or len(pair_map) == 0:
        return (0.0,)

    def _eval_subtree(pos: int):
        node = tree_list[pos]
        pos += 1
        if node.arity == 0:
            tname = node.name
            if not tname.startswith("t_"):
                return None, pos
            try:
                t_index = int(tname[2:])
            except (ValueError, IndexError):
                return None, pos
            pair = tindex_to_pair.get(t_index)
            if pair is None:
                return None, pos
            arr = pair_map.get(pair)
            if arr is None:
                return None, pos
            return arr, pos

        name = node.name
        arity = node.arity
        if arity == 1:
            child, pos = _eval_subtree(pos)
            if child is None:
                return None, pos
            if name == "neg":
                return -child, pos
            if name == "abs":
                return _abs(child), pos
            if name == "sqrt":
                return _sqrt(_abs(child) + 1e-10), pos
            if name == "square":
                return _clip(child, -1e5, 1e5) ** 2, pos
            if name == "log1p":
                return _log1p(_abs(child) + 1e-10), pos
            return child, pos

        if arity == 2:
            left, pos = _eval_subtree(pos)
            right, pos = _eval_subtree(pos)
            if left is None or right is None:
                return None, pos
            left = _clip(left, -1e10, 1e10)
            right = _clip(right, -1e10, 1e10)
            if name == "add":
                return left + right, pos
            if name == "sub":
                return left - right, pos
            if name == "mul":
                return left * right, pos
            if name == "div":
                denom = _where(_abs(right) < 1e-10, _sign(right + 1e-15) * 1e-10, right)
                return left / denom, pos
            return left, pos
        return None, pos

    with _errstate(over="ignore", invalid="ignore"):
        factor_values, _ = _eval_subtree(0)
    if factor_values is None or len(factor_values) == 0:
        return (0.0,)

    mask = _isfinite(factor_values)
    if mask.mean() < 0.5:
        return (0.0,)
    X = np.ascontiguousarray(factor_values[mask].reshape(-1, 1))
    y = np.ascontiguousarray(labels[mask])

    if len(np.unique(y)) < 2:
        return (0.0,)

    n_classes = len(classes_sorted)
    actual_cv = 2 if (reduced_cv and cv_folds >= 3) else min(cv_folds, 3)

    try:
        if use_gpu and gpu_backend in ("xgboost", "auto"):
            macro_auc = _train_xgboost_gpu(X, y, n_classes, n_estimators, actual_cv)
        elif use_gpu and gpu_backend == "lightgbm":
            macro_auc = _train_lgbm_gpu(X, y, n_classes, n_estimators, actual_cv)
        else:
            macro_auc = _train_lgbm_cpu(X, y, n_classes, n_estimators, actual_cv)
    except Exception:
        macro_auc = 0.0

    return (macro_auc,)


def _train_lgbm_cpu(X, y, n_classes, n_estimators, cv_folds):
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=n_estimators,
            max_depth=3,
            num_leaves=15,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            min_data_in_leaf=max(5, len(y) // (n_classes * 20)),
            n_jobs=1,
        )
        scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring="roc_auc_ovr", n_jobs=1,
        )
    return float(scores.mean())


def _train_xgboost_gpu(X, y, n_classes, n_estimators, cv_folds):
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.utils.class_weight import compute_class_weight

    classes_ = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes_, y=y)
    weight_map = dict(zip(classes_, cw))
    sw = np.array([weight_map[yi] for yi in y])

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            model = XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                n_estimators=n_estimators,
                max_depth=3,
                min_child_weight=max(5, len(y) // (n_classes * 20)),
                random_state=42,
                verbosity=0,
                tree_method="hist",
                device="cuda",
                eval_metric="mlogloss",
            )
            scores = cross_val_score(
                model, X, y, cv=cv_folds, scoring="roc_auc_ovr", n_jobs=1,
                params={"sample_weight": sw},
            )
        return float(scores.mean())
    except Exception:
        # Fall back to LGBM CPU when GPU fails
        return _train_lgbm_cpu(X, y, n_classes, n_estimators, cv_folds)


def _train_lgbm_gpu(X, y, n_classes, n_estimators, cv_folds):
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=n_estimators,
            max_depth=3,
            num_leaves=15,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            min_data_in_leaf=max(5, len(y) // (n_classes * 20)),
            device="cuda",
            n_jobs=1,
        )
        scores = cross_val_score(
            model, X, y, cv=cv_folds, scoring="roc_auc_ovr", n_jobs=1,
        )
    return float(scores.mean())


# ===================================================================
# Aggregator definitions
# ===================================================================
AGGREGATORS = [
    "mean", "std", "var", "max", "min",
    "trend",        # polyfit slope
    "range",        # max - min
    "first",        # first frame value
    "last",         # last frame value
    "diff",         # last - first
    "mad",          # median absolute deviation
    "rms",          # root mean square
]

# Aggregator keyword -> agg name mapping (detected from existing code)
# Note: abs/sqrt/square/log1p are unary operators, used as primitives in DEAP, not terminal aggregators
AGG_KEYWORDS = {
    "np.mean": "mean", "np.std": "std", "np.var": "var",
    "np.max": "max", "np.min": "min",
    "np.polyfit": "trend", "polyfit": "trend",
    "np.median": "mad",
    "np.sum": "mean",
}

# ===================================================================
# Code generator (converts DEAP expression tree to FactorEngine-compatible Python code)
# ===================================================================
class CodeGenerator:
    """Compiles DEAP expression trees into FactorEngine-compatible Python code strings."""

    @staticmethod
    def _agg_to_expr(agg: str, col_var: str) -> str:
        mapping = {
            "mean":  f"np.mean({col_var})",
            "std":   f"np.std({col_var})",
            "var":   f"np.var({col_var})",
            "max":   f"np.max({col_var})",
            "min":   f"np.min({col_var})",
            "trend": f"(np.polyfit(np.arange(len({col_var})), {col_var}, 1)[0] if len({col_var}) >= 2 else 0.0)",
            "range": f"np.max({col_var}) - np.min({col_var})",
            "first": f"{col_var}[0]",
            "last":  f"{col_var}[-1]",
            "diff":  f"{col_var}[-1] - {col_var}[0]",
            "mad":   f"np.mean(np.abs({col_var} - np.median({col_var})))",
            "rms":   f"np.sqrt(np.mean({col_var} ** 2))",
        }
        return mapping.get(agg, f"np.mean({col_var})")

    @staticmethod
    def _op_to_expr(op: str, left: str, right: str) -> str:
        op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        py_op = op_map.get(op, op)
        if py_op == "/":
            expr = f"({left}) / ({right} + 1e-8)"
        else:
            expr = f"({left}) {py_op} ({right})"
        return f"np.clip({expr}, -1e10, 1e10)"

    @staticmethod
    def _unary_to_expr(op: str, operand: str) -> str:
        mapping = {
            "neg":    f"-({operand})",
            "abs":    f"np.abs({operand})",
            "sqrt":   f"np.sqrt(np.abs({operand}) + 1e-10)",
            "square": f"np.clip({operand}, -1e5, 1e5) ** 2",
            "log1p":  f"np.log1p(np.abs({operand}) + 1e-10)",
        }
        return mapping.get(op, operand)

    def generate(self, deap_individual, name_to_pair: Dict[str, Tuple[str, str]]) -> str:
        """
        Generate factor code from a DEAP individual (PrimitiveTree in prefix list representation).
        All nodes in the DEAP tree are Primitive objects; arity==0 means terminal.
        name_to_pair: terminal name "t_{index}" -> (feature_name, aggregator_name)
        """
        tree_list = list(deap_individual)
        if len(tree_list) == 0:
            return "return np.nan"

        # Collect all used (feature, agg) pairs
        used_pairs: Set[Tuple[str, str]] = set()
        for node in tree_list:
            if node.arity == 0 and node.name in name_to_pair:
                used_pairs.add(name_to_pair[node.name])

        features_used = sorted(set(feat for feat, _ in used_pairs))
        lines: List[str] = []

        # Step 1: idx lookup + availability check
        var_map: Dict[str, str] = {}  # feature -> idx var name
        checks: List[str] = []
        for i, feat in enumerate(features_used):
            vname = f"i_{i}"
            var_map[feat] = vname
            lines.append(f"{vname} = idx.get('{feat}', -1)")
            checks.append(f"{vname} < 0")

        if checks:
            lines.append(f"if {' or '.join(checks)}:")
            lines.append("    return np.nan")

        # Step 2: Recursively compile DEAP prefix tree
        agg_cache: Dict[Tuple[str, str], str] = {}
        varcounter = [0]

        def consume_subtree(pos: int) -> Tuple[str, int]:
            """Consume a subtree from prefix list at position pos; returns (variable_name, next_position)."""
            node = tree_list[pos]
            pos += 1

            # Terminal: arity==0
            if node.arity == 0:
                pair = name_to_pair.get(node.name)
                if pair is None:
                    return "np.nan", pos
                feat, agg = pair
                cache_key = (feat, agg)
                if cache_key in agg_cache:
                    return agg_cache[cache_key], pos
                col_ref = f"window[:, {var_map[feat]}]"
                agg_expr = self._agg_to_expr(agg, col_ref)
                vname = f"v{varcounter[0]}"
                varcounter[0] += 1
                lines.append(f"{vname} = {agg_expr}")
                agg_cache[cache_key] = vname
                return vname, pos

            # Internal node
            name = node.name
            arity = node.arity

            if arity == 1:
                child, pos = consume_subtree(pos)
                vname = f"v{varcounter[0]}"
                varcounter[0] += 1
                expr = self._unary_to_expr(name, child)
                lines.append(f"{vname} = {expr}")
                return vname, pos
            elif arity == 2:
                left, pos = consume_subtree(pos)
                right, pos = consume_subtree(pos)
                vname = f"v{varcounter[0]}"
                varcounter[0] += 1
                expr = self._op_to_expr(name, left, right)
                lines.append(f"{vname} = {expr}")
                return vname, pos
            else:
                return "np.nan", pos

        root_var, _ = consume_subtree(0)
        lines.append(f"return float(np.clip({root_var}, -1e10, 1e10))")
        return "\n".join(lines)


def repair_factor_code(code: str) -> str:
    """Repair numerically unstable GP-generated factor code by adding np.clip guards.

    Wraps intermediate binary operation results with np.clip(..., -1e10, 1e10),
    clips square operand to [-1e5, 1e5], and clips the final return value.
    Already-repaired code (containing np.clip) is left unchanged.
    """
    import re

    lines = code.split('\n')
    repaired = []

    for line in lines:
        stripped = line.strip()

        # Blank lines, idx lookups, if-guards, early returns -- pass through
        if (not stripped
                or stripped.startswith('if ')
                or stripped == 'return np.nan'
                or re.match(r'^i_\d+\s*=\s*idx\.get\(', stripped)):
            repaired.append(line)
            continue

        # Aggregation and function calls (output bounded by data) -- pass through
        if re.match(r'^v\d+\s*=\s*np\.', stripped):
            repaired.append(line)
            continue

        # Final return line
        if stripped.startswith('return float('):
            inner = stripped[len('return float('):-1]
            repaired.append(
                f"return float(np.clip({inner}, -1e10, 1e10))")
            continue

        # Variable assignment: vN = RHS
        m = re.match(r'^(\s*v\d+\s*=\s*)(.*)$', line)
        if m:
            prefix, rhs = m.group(1), m.group(2)

            # Already repaired -- skip
            if 'np.clip' in rhs:
                repaired.append(line)
                continue

            # Square: (expr) ** 2 -> clip operand before squaring
            sq_match = re.match(r'^\((.+)\)\s*\*\*\s*2\s*$', rhs)
            if sq_match:
                operand = sq_match.group(1)
                repaired.append(
                    f"{prefix}np.clip({operand}, -1e5, 1e5) ** 2")
                continue

            # Binary op: ) op (  or negation: -(expr)
            if (re.search(r'\)\s*[+*/-]\s*\(', rhs)
                    or re.match(r'^-\(', rhs)):
                repaired.append(
                    f"{prefix}np.clip({rhs}, -1e10, 1e10)")
                continue

        # Fallback: unrecognized pattern, leave unchanged
        repaired.append(line)

    return '\n'.join(repaired)


# ===================================================================
# Gene pool extraction
# ===================================================================
class GenePool:
    """Extract gene pool information from existing factors."""

    # -- Feature name normalization: LLM naming -> dataset naming --
    @staticmethod
    def normalize_feature_name(feat: str) -> "str | None":
        """Convert LLM factor code feature names to dataset flat_attributes naming convention.

        Rules:
          Self_vel_kp0_x  -> Self_kp0_vx
          Self_acc_kp0_x  -> Self_kp0_ax
          Self_box_vel_x  -> Self_box_vx
          Self_box_acc_x  -> Self_box_ax
          Other1_vel_kp0_y -> Other1_kp0_vy
          (and so on)

        Returns the original if no match. Returns None if the feature should be discarded (e.g., non-existent kp index).
        """
        # Template patterns should be discarded directly
        if "%" in feat or feat.endswith("_"):
            return None
        # Filter non-existent keypoint indices (only kp0-kp6, 7 keypoints total)
        m_kp = re.search(r'(?:Self|Other\d+)_kp(\d+)', feat)
        if m_kp and int(m_kp.group(1)) >= 7:
            return None
        # vel_kp{n}_{xy} -> kp{n}_v{xy}
        m = re.match(r'(Self|Other\d+)_vel_kp(\d+)_([xy])', feat)
        if m:
            return f"{m.group(1)}_kp{m.group(2)}_v{m.group(3)}"
        # acc_kp{n}_{xy} -> kp{n}_a{xy}
        m = re.match(r'(Self|Other\d+)_acc_kp(\d+)_([xy])', feat)
        if m:
            return f"{m.group(1)}_kp{m.group(2)}_a{m.group(3)}"
        # box_vel_{xy} -> box_v{xy}
        m = re.match(r'(Self|Other\d+)_box_vel_([xy])', feat)
        if m:
            return f"{m.group(1)}_box_v{m.group(2)}"
        # box_acc_{xy} -> box_a{xy}
        m = re.match(r'(Self|Other\d+)_box_acc_([xy])', feat)
        if m:
            return f"{m.group(1)}_box_a{m.group(2)}"
        return feat

    def __init__(self, factors: List[dict]):
        # Extract (feature, aggregator) pairs and their frequencies
        pair_counter: Counter = Counter()
        feature_counter: Counter = Counter()
        agg_counter: Counter = Counter()
        target_counter: Counter = Counter()
        seqlen_counter: Counter = Counter()

        for f in factors:
            code = f.get("code", "")
            # Extract idx keys
            feats_in_code: Set[str] = set()
            for m in re.finditer("idx\\.get\\('([^']+)'", code):
                feat = m.group(1)
                norm = self.normalize_feature_name(feat)
                if norm is None:
                    continue
                feats_in_code.add(norm)
                feature_counter[norm] += 1

            # Detect aggregators
            aggs_in_code: Set[str] = set()
            for kw, agg in AGG_KEYWORDS.items():
                if kw in code:
                    aggs_in_code.add(agg)
                    agg_counter[agg] += 1

            # Count each (feat, agg) combination
            for feat in feats_in_code:
                for agg in aggs_in_code:
                    pair_counter[(feat, agg)] += 1

            # Target
            tgt = f.get("target", "")
            if tgt:
                target_counter[tgt] += 1

            # seq_length
            sl = f.get("seq_length", 5)
            if isinstance(sl, (int, float)) and sl > 0:
                seqlen_counter[int(sl)] += 1

        # Supplement common features
        common_features = {
            "speed", "dist_to_other", "approach_dot", "compactness",
            "sniff_min_dist", "wall_dist_score", "width", "height",
            "Self_box_x", "Self_box_y",
        }
        for feat in common_features:
            if feat not in feature_counter:
                feature_counter[feat] = 1
                for agg in ["mean", "std", "trend", "range"]:
                    pair_counter[(feat, agg)] += 1

        # Supplement velocity/acceleration features that exist in the dataset but are never referenced by factors
        # Uses dataset convention (Self_kp{n}_vx) rather than factor code convention (Self_vel_kp{n}_x)
        for prefix, n_kp in [("Self", 7), ("Other1", 7)]:
            for k in range(n_kp):
                for ax, xy in [("vx", "x"), ("vy", "y"), ("ax", "x"), ("ay", "y")]:
                    # Velocity: Self_kp0_vx, Acceleration: Self_kp0_ax
                    fname = f"{prefix}_kp{k}_{ax}"
                    if fname not in feature_counter:
                        feature_counter[fname] = 1
                        for agg in ["mean", "std", "trend"]:
                            pair_counter[(fname, agg)] += 1
        for prefix in ["Self", "Other1"]:
            for suf in ["box_vx", "box_vy", "box_ax", "box_ay"]:
                fname = f"{prefix}_{suf}"
                if fname not in feature_counter:
                    feature_counter[fname] = 1
                    for agg in ["mean", "std", "trend"]:
                        pair_counter[(fname, agg)] += 1

        self.feature_agg_pairs: List[Tuple[str, str]] = sorted(pair_counter.keys())
        self.pair_weights: List[float] = [pair_counter[p] for p in self.feature_agg_pairs]
        total_p = sum(self.pair_weights)
        self.pair_weights = [w / total_p for w in self.pair_weights]

        self.features: List[str] = sorted(feature_counter.keys())
        self.feature_freq: Dict[str, int] = dict(feature_counter)
        total_f = sum(feature_counter.values()) or 1
        self.feature_rarity: Dict[str, float] = {
            f: 1.0 - (c / total_f) for f, c in feature_counter.items()
        }

        self.targets: List[str] = list(target_counter.keys()) or ["climb"]
        self.target_weights: List[float] = [
            target_counter[t] / max(sum(target_counter.values()), 1)
            for t in self.targets
        ]

        self.seq_lengths: List[int] = sorted(seqlen_counter.keys()) or [1, 5, 15, 30, 60]
        self.seqlen_weights: List[float] = [
            seqlen_counter[s] / max(sum(seqlen_counter.values()), 1)
            for s in self.seq_lengths
        ]

        # Existing feature combination signatures (for novelty computation)
        self.existing_signatures: Set[str] = set()
        for f in factors:
            code = f.get("code", "")
            feats = set()
            for m in re.finditer("idx\\.get\\('([^']+)'", code):
                feat = m.group(1)
                norm = GenePool.normalize_feature_name(feat)
                if norm:
                    feats.add(norm)
            if feats:
                self.existing_signatures.add(",".join(sorted(feats)))

        logger.info(
            f"Gene pool: {len(self.feature_agg_pairs)} (feature,aggregator) pairs, "
            f"{len(self.features)} features, "
            f"{len(self.targets)} targets, "
            f"{len(self.existing_signatures)} existing feature combination signatures"
        )


# ===================================================================
# Build DEAP primitive set
# ===================================================================
def build_pset(gene_pool: GenePool) -> Tuple[gp.PrimitiveSet, Dict[str, Tuple[str, str]]]:
    """
    Build DEAP primitive set, returns (pset, name_to_pair).

    Each (feature, aggregator) pair corresponds to a terminal Primitive(arity=0).
    Terminal names are encoded as t_{index}; name_to_pair maps terminal name back to (feature, aggregator).

    All nodes in the DEAP tree are Primitive objects:
    - arity=0: terminal (leaf node), name like t_0, t_1, ...
    - arity=1: unary operations (neg, abs, sqrt, square, log1p)
    - arity=2: binary operations (add, sub, mul, div)
    """
    pset = gp.PrimitiveSet("MAIN", 0)

    name_to_pair: Dict[str, Tuple[str, str]] = {}

    for i, (feat, agg) in enumerate(gene_pool.feature_agg_pairs):
        tname = f"t_{i}"
        name_to_pair[tname] = (feat, agg)
        # Terminal stores int i; code generation looks up feat/agg via name
        pset.addTerminal(i, name=tname)

    # Binary operation primitives
    pset.addPrimitive(lambda x, y: None, 2, name="add")
    pset.addPrimitive(lambda x, y: None, 2, name="sub")
    pset.addPrimitive(lambda x, y: None, 2, name="mul")
    pset.addPrimitive(lambda x, y: None, 2, name="div")

    # Unary operation primitives
    pset.addPrimitive(lambda x: None, 1, name="neg")
    pset.addPrimitive(lambda x: None, 1, name="abs")
    pset.addPrimitive(lambda x: None, 1, name="sqrt")
    pset.addPrimitive(lambda x: None, 1, name="square")
    pset.addPrimitive(lambda x: None, 1, name="log1p")

    return pset, name_to_pair


# ===================================================================
# Seed initial population from existing factors
# ===================================================================
def seed_individuals_from_factors(
    factors: List[dict],
    gene_pool: GenePool,
    name_to_pair: Dict[str, Tuple[str, str]],
    pset: gp.PrimitiveSet,
    rng: random.Random,
    n_seeds: int,
) -> list:
    """
    Extract feature combinations from existing factors and build DEAP individuals as initial population seeds.

    For each factor:
    1. Parse code, extract (feature, aggregator) pairs
    2. Find corresponding terminal names
    3. Combine terminals using simple operation trees to create DEAP individuals

    Returns at most n_seeds DEAP PrimitiveTree individuals.
    """
    if n_seeds <= 0:
        return []

    # Build terminal name -> Primitive mapping (terminal nodes have arity=0)
    term_by_name: Dict[str, Any] = {}
    for tname in name_to_pair:
        term_by_name[tname] = gp.Terminal(tname, [], object)

    # Sort by AUC, prefer high-AUC factors
    sorted_factors = sorted(
        factors,
        key=lambda f: f.get("best_auc") or 0,
        reverse=True,
    )

    seeds = []
    seen_sigs: Set[str] = set()

    for f in sorted_factors:
        if len(seeds) >= n_seeds:
            break

        code = f.get("code", "")
        # Extract (feature, aggregator) pairs: detect idx keys + aggregators
        feat_agg: Set[Tuple[str, str]] = set()
        feats_in_code = set()
        for m in re.finditer("idx\\.get\\('([^']+)'", code):
            feat = m.group(1)
            norm = GenePool.normalize_feature_name(feat)
            if norm:
                feats_in_code.add(norm)

        # Detect aggregators used on each feature (nearest match)
        aggs_used = set()
        for kw, agg in AGG_KEYWORDS.items():
            if kw in code:
                aggs_used.add(agg)

        for feat in feats_in_code:
            for agg in aggs_used:
                # Only keep (feat, agg) pairs that exist in the gene pool
                if (feat, agg) in set(gene_pool.feature_agg_pairs):
                    feat_agg.add((feat, agg))
                # If this aggregator is not in the gene pool, fall back to mean
                elif (feat, "mean") in set(gene_pool.feature_agg_pairs) and agg not in {"mean"}:
                    feat_agg.add((feat, "mean"))

        if len(feat_agg) < 2:
            continue

        # Deduplicate by signature
        sig = ",".join(sorted(feat for feat, _ in feat_agg))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        # Find the terminal name for each (feat, agg) pair
        terminals: list = []
        for pair in feat_agg:
            for tname, p in name_to_pair.items():
                if p == pair and tname in term_by_name:
                    terminals.append(term_by_name[tname])
                    break

        if len(terminals) < 2:
            continue

        # Use div as root, add/mul as branches, simple combination
        rng.shuffle(terminals)
        # Build balanced tree: combine first 2 terminals, then add remaining one by one
        tree_nodes = [terminals[0], terminals[1]]
        # Pick a binary operation
        op = rng.choice(["add", "mul", "sub", "div"])
        tree_nodes = [gp.Primitive(op, [object, object], object), terminals[0], terminals[1]]

        # Add remaining terminals one by one
        for t in terminals[2:]:
            new_op = rng.choice(["add", "mul", "sub", "div"])
            # New tree: [new_op, old_tree..., t]
            tree_nodes = [gp.Primitive(new_op, [object, object], object)] + tree_nodes + [t]

        # Optionally wrap with a unary operation
        if rng.random() < 0.2:
            uop = rng.choice(["neg", "abs", "sqrt", "square", "log1p"])
            tree_nodes = [gp.Primitive(uop, [object], object)] + tree_nodes

        # Ensure it is a valid prefix expression
        try:
            ind = creator.Individual(tree_nodes)
            seeds.append(ind)
        except Exception:
            continue

    return seeds


# ===================================================================
# Factor subtree library (for mutation injection)
# ===================================================================
def build_factor_subtree_library(
    factors: List[dict],
    gene_pool: GenePool,
    name_to_pair: Dict[str, Tuple[str, str]],
    rng: random.Random,
) -> list:
    """Build a list of DEAP subtrees for each unique feature combination from all factors, for mutation injection."""
    term_by_name = {}
    for tname in name_to_pair:
        term_by_name[tname] = gp.Terminal(tname, [], object)

    gp_pairs_set = set(gene_pool.feature_agg_pairs)
    library = []
    seen_sigs: Set[str] = set()

    for f in sorted(factors, key=lambda f: f.get("best_auc") or 0, reverse=True):
        code = f.get("code", "")
        feats_in_code = set()
        for m in re.finditer("idx\\.get\\('([^']+)'", code):
            feat = m.group(1)
            norm = GenePool.normalize_feature_name(feat)
            if norm:
                feats_in_code.add(norm)

        aggs_used = set()
        for kw, agg in AGG_KEYWORDS.items():
            if kw in code:
                aggs_used.add(agg)

        feat_agg = set()
        for feat in feats_in_code:
            for agg in aggs_used:
                if (feat, agg) in gp_pairs_set:
                    feat_agg.add((feat, agg))
                elif (feat, "mean") in gp_pairs_set:
                    feat_agg.add((feat, "mean"))

        if len(feat_agg) < 2:
            continue

        sig = ",".join(sorted(feat for feat, _ in feat_agg))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        terminals = []
        for pair in feat_agg:
            for tname, p in name_to_pair.items():
                if p == pair and tname in term_by_name:
                    terminals.append(term_by_name[tname])
                    break
        if len(terminals) < 2:
            continue

        rng.shuffle(terminals)
        op = rng.choice(["add", "mul", "sub", "div"])
        tree = [gp.Primitive(op, [object, object], object), terminals[0], terminals[1]]
        for t in terminals[2:]:
            new_op = rng.choice(["add", "mul", "sub", "div"])
            tree = [gp.Primitive(new_op, [object, object], object)] + tree + [t]

        library.append(tree)

    return library


def mutate_with_factor_injection(
    individual, pset, expr_gen, factor_library, injection_rate, rng,
):
    """Custom mutation: inject subtree from factor library at injection_rate probability."""
    if factor_library and rng.random() < injection_rate:
        donor = list(rng.choice(factor_library))
        if len(donor) == 0 or len(individual) == 0:
            return gp.mutUniform(individual, expr_gen, pset)

        i = rng.randrange(len(individual))

        def subtree_span(tree, pos):
            node = tree[pos]
            if node.arity == 0:
                return pos + 1
            end = pos + 1
            for _ in range(node.arity):
                end = subtree_span(tree, end)
            return end

        span_end = subtree_span(individual, i)
        new_ind = creator.Individual(
            individual[:i] + donor + individual[span_end:]
        )
        if len(new_ind) <= 40:
            individual.clear()
            individual.extend(new_ind)
        return (individual,)
    else:
        return gp.mutUniform(individual, expr_gen, pset)


# ===================================================================
class RealFitnessEvaluator:
    """
    Evaluate factor predictive power using real data + LightGBM.
    Precomputes window aggregation values for each (feature, aggregator) pair;
    at evaluation time, only needs to combine precomputed columns.
    """

    def __init__(
        self,
        config_path: str,
        gene_pool: GenePool,
        max_samples: int = 10000,
        n_estimators: int = 30,
        cv_folds: int = 3,
        config_validation: str = "",
        use_gpu: bool = False,
        gpu_backend: str = "auto",
        reduced_cv: bool = False,
        num_workers: int = 0,
    ):
        logger.info("=== Loading real data for validation mode ===")
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.cv_folds = cv_folds
        self.gene_pool = gene_pool
        self.use_gpu = use_gpu
        self.gpu_backend = gpu_backend
        self.reduced_cv = reduced_cv
        self.num_workers = num_workers
        self._mmap_dir: str = ""
        self._mmap_meta: dict = {}

        # ---- Load config and data (merge two config files, consistent with train_behavior.py behavior) ----
        from mining.discovery import _load_merged_config, load_dataset_config, merge_dataset_config_list
        from src.data_loader import MouseBehaviorDataset

        config_paths = [str(config_path)]
        if config_validation:
            vp = Path(config_validation)
            if not vp.is_absolute():
                vp = Path(config_path).parent / vp
            if vp.exists():
                config_paths.append(str(vp))
        cfg = _load_merged_config(*config_paths)
        config_dir = Path(config_path).parent
        ds_cfg_file = cfg.get("dataset_config_file", "")
        if not ds_cfg_file:
            raise ValueError("Config missing dataset_config_file field")
        ds_cfg_path = Path(ds_cfg_file)
        if not ds_cfg_path.is_absolute():
            ds_cfg_path = config_dir / ds_cfg_path

        ds_info = load_dataset_config(cfg, str(ds_cfg_path), logger)
        merged_label_map = ds_info["label_map"]
        valid_labels = set(int(v) for v in merged_label_map.values())

        # ---- force_cache fast path (used when server has no raw data) ----
        fc = cfg.get("force_cache", {})
        fc_path = fc.get("raw_frames") or fc.get("path")
        if fc_path:
            import torch
            cache_path = Path(fc_path)
            # Relative path is relative to CWD (project root), consistent with _load_force_cache behavior
            if not cache_path.exists():
                raise FileNotFoundError(f"[force_cache] File does not exist: {cache_path}")

            logger.info(f"[force_cache] Directly loading preprocessed cache: {cache_path}")
            cache_dict = torch.load(str(cache_path), weights_only=False)
            train_kp = np.asarray(cache_dict["train_kp"].cpu() if hasattr(cache_dict["train_kp"], 'cpu') else cache_dict["train_kp"], dtype=np.float32)
            train_lb = np.asarray(cache_dict["train_lb"].cpu() if hasattr(cache_dict["train_lb"], 'cpu') else cache_dict["train_lb"], dtype=np.int64)
            self.flat_attributes = cache_dict["flat_attributes"]

            train_mask = np.isin(train_lb, list(valid_labels))
            if not train_mask.all():
                train_kp, train_lb = train_kp[train_mask], train_lb[train_mask]
                logger.info(f"[force_cache] After filtering invalid labels: train {train_kp.shape[0]} frames")
        else:
            # Use MouseBehaviorDataset directly (avoid build_raw_frame_data to prevent preprocessing parameter incompatibility)
            train_ds_cfg = merge_dataset_config_list(ds_info["train_dataset_config"])
            logger.info("Building train raw frame data (multi-resolution mode)...")
            mbd = MouseBehaviorDataset(
                dataset_config=train_ds_cfg,
                label_map=merged_label_map,
                seq_length=1,
                stride=1,
                frame_interval=1,
                transform=None,
                is_train=True,
                purity_threshold=1.0,
                boundary_margin=0,
                per_video_normalize=True,
            )
            train_kp = mbd.keypoints.cpu().numpy().astype(np.float32) if hasattr(mbd.keypoints, 'cpu') else mbd.keypoints.numpy().astype(np.float32)
            train_lb = mbd.labels.cpu().numpy().astype(np.int64) if hasattr(mbd.labels, 'cpu') else mbd.labels.astype(np.int64)
            self.flat_attributes = mbd.feature_indexer.flat_attributes

            # Align and filter invalid labels
            T = min(len(train_kp), len(train_lb))
            train_kp, train_lb = train_kp[:T], train_lb[:T]
            mask = np.isin(train_lb, list(valid_labels))
            train_kp, train_lb = train_kp[mask], train_lb[mask]

        logger.info(
            f"Raw training data: {train_kp.shape[0]} frames, D={train_kp.shape[1]}, "
            f"n_classes={len(set(train_lb))}"
        )

        # Keep full data for resampling
        self._full_kp = train_kp.astype(np.float32)
        self._full_lb = train_lb.astype(np.int64)
        self.max_samples = max_samples

        # Build name -> column index mapping
        self.name_to_col: Dict[str, int] = {
            name: i for i, name in enumerate(self.flat_attributes)
        }

        # Supplement factor code naming -> dataset column name mapping (different naming conventions)
        # Factor uses Self_vel_kp{n}_x, dataset uses Self_kp{n}_vx
        _name_aliases = []
        for prefix, n_keypoints in [("Self", 7), ("Other1", 7)]:
            for k in range(n_keypoints):
                for code_ax, data_ax in [("vel", "v"), ("acc", "a")]:
                    for code_xy, data_xy in [("_x", "x"), ("_y", "y")]:
                        code_name = f"{prefix}_{code_ax}_kp{k}{code_xy}"
                        data_name = f"{prefix}_kp{k}_{data_ax}{data_xy}"
                        _name_aliases.append((code_name, data_name))
        # box velocity/acceleration
        for prefix in ["Self", "Other1"]:
            for code_suf, data_suf in [("box_vel_x", "box_vx"), ("box_vel_y", "box_vy"),
                                        ("box_acc_x", "box_ax"), ("box_acc_y", "box_ay")]:
                _name_aliases.append((f"{prefix}_{code_suf}", f"{prefix}_{data_suf}"))
        # Angle naming fix: Self_angle_{k} -> Self_angle_{k} (same, but ensure they are in the map)
        # kp7 is not in the dataset, not added

        aliased = 0
        for code_name, data_name in _name_aliases:
            if code_name not in self.name_to_col and data_name in self.name_to_col:
                self.name_to_col[code_name] = self.name_to_col[data_name]
                aliased += 1
        if aliased > 0:
            logger.info(f"Factor-name -> dataset-column name mapping: {aliased} aliases added")

        # First sampling + precomputation
        self._resample_and_build(seed_offset=0, base_seed=0)  # Use 0 at init, cycle loops pass args.seed

    def _resample_and_build(self, seed_offset: int = 0, base_seed: int = 42):
        """Downsample + rebuild precomputation matrix. Each cycle uses a different seed for different subsamples."""
        train_kp = self._full_kp
        train_lb = self._full_lb

        if self.max_samples > 0 and train_kp.shape[0] > self.max_samples:
            rng = np.random.default_rng(base_seed + seed_offset * 1000)
            T = train_kp.shape[0]
            chunk_len = 500
            n_chunks = max(1, self.max_samples // chunk_len)
            max_start = T - chunk_len
            if n_chunks > 1 and max_start > 0:
                region_size = max(max_start // n_chunks, chunk_len + 1)
                starts = []
                for i in range(n_chunks):
                    lo = i * region_size
                    hi = min(lo + region_size - chunk_len - 1, max_start)
                    if hi > lo:
                        starts.append(int(rng.integers(lo, hi, endpoint=True)))
                idx = np.unique(np.concatenate([
                    np.arange(s, min(s + chunk_len, T)) for s in starts
                ]))
            else:
                idx = np.arange(0, min(self.max_samples, T))
            train_kp = train_kp[idx]
            train_lb = train_lb[idx]
            logger.info(
                f"Continuous segment downsampling (seed_offset={seed_offset}): {T} -> {len(idx)} frames "
                f"({n_chunks} continuous segments, {chunk_len} frames each)"
            )

        self.train_kp = train_kp.astype(np.float32)
        self.train_lb = train_lb.astype(np.int64)

        # ---- Precompute (feature, agg) arrays for each seq_length ----
        self.pair_arrays: Dict[int, Dict[Tuple[str, str], np.ndarray]] = {}
        self.seq_labels: Dict[int, np.ndarray] = {}

        for L in [1, 5, 15, 30, 60]:
            if L > len(self.train_kp):
                continue
            N = len(self.train_kp) - L + 1
            if N < 100:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(
                self.train_kp, (L, self.train_kp.shape[1])
            )[:, 0, :, :]
            windows = np.ascontiguousarray(windows, dtype=np.float32)
            labels = self.train_lb[L // 2 : L // 2 + N]
            self.seq_labels[L] = labels
            pair_map: Dict[Tuple[str, str], np.ndarray] = {}
            for feat, agg in self.gene_pool.feature_agg_pairs:
                col = self.name_to_col.get(feat)
                if col is None:
                    continue
                arr = self._compute_agg(windows[:, :, col], agg)
                if arr is not None:
                    pair_map[(feat, agg)] = arr
            self.pair_arrays[L] = pair_map
            logger.info(
                f"  seq_length={L}: {N} windows, "
                f"{len(pair_map)} (feature,aggregator) pairs precomputed"
            )

        self.classes_sorted = sorted(set(int(v) for v in self.train_lb))
        logger.info(
            f"Real fitness evaluator ready. seq_lengths={sorted(self.pair_arrays.keys())}, "
            f"classes={self.classes_sorted}"
        )

        # If multiprocess evaluation enabled, write memmap in advance
        if self.num_workers > 0:
            self._save_memmap()

    @staticmethod
    def _compute_agg(col_data: np.ndarray, agg: str) -> Optional[np.ndarray]:
        """Compute aggregation along temporal axis for window column data [N, L] -> [N]"""
        try:
            if agg == "mean":
                return np.mean(col_data, axis=1).astype(np.float32)
            if agg == "std":
                return np.std(col_data, axis=1).astype(np.float32)
            if agg == "var":
                return np.var(col_data, axis=1).astype(np.float32)
            if agg == "max":
                return np.max(col_data, axis=1).astype(np.float32)
            if agg == "min":
                return np.min(col_data, axis=1).astype(np.float32)
            if agg == "range":
                return (np.max(col_data, axis=1) - np.min(col_data, axis=1)).astype(np.float32)
            if agg == "first":
                return col_data[:, 0].astype(np.float32)
            if agg == "last":
                return col_data[:, -1].astype(np.float32)
            if agg == "diff":
                return (col_data[:, -1] - col_data[:, 0]).astype(np.float32)
            if agg == "trend":
                L = col_data.shape[1]
                if L < 2:
                    return np.zeros(col_data.shape[0], dtype=np.float32)
                t = np.arange(L, dtype=np.float32)
                t_mean = t.mean()
                t_demean = t - t_mean
                t_denom = np.dot(t_demean, t_demean) or 1.0
                slopes = np.dot(col_data.astype(np.float64), t_demean) / t_denom
                return slopes.astype(np.float32)
            if agg == "mad":
                med = np.median(col_data, axis=1)
                return np.mean(np.abs(col_data - med[:, None]), axis=1).astype(np.float32)
            if agg == "rms":
                return np.sqrt(np.mean(np.clip(col_data, -1e5, 1e5) ** 2, axis=1)).astype(np.float32)
        except Exception:
            pass
        return None

    def evaluate(self, individual) -> Tuple[float]:
        """Evaluate DEAP individual using precomputed data: build factor values + LGBM/XGBoost -> macro AUC."""
        actual_cv = 2 if (self.reduced_cv and self.cv_folds >= 3) else min(self.cv_folds, 3)
        return _eval_individual_impl(
            individual,
            self.pair_arrays,
            self.seq_labels,
            self.classes_sorted,
            self._tindex_to_pair,
            self.use_gpu,
            self.gpu_backend,
            self.n_estimators,
            actual_cv,  # already reduced for the impl
            False,      # reduced_cv already applied above
        )

    def _save_memmap(self):
        """Write self.pair_arrays and self.seq_labels to memmap (one stacked array per seq_length).
        Returns (mmap_dir, pair_arrays_meta, seq_labels_meta)."""
        global _EVO_MMAP_DIR, _EVO_MMAP_FILES
        _evo_cleanup_mmap()
        mmap_dir = tempfile.mkdtemp(prefix="evo_mmap_")
        _EVO_MMAP_DIR = mmap_dir
        atexit.register(_evo_cleanup_mmap)

        pair_arrays_meta: dict = {}
        seq_labels_meta: dict = {}

        # Each seq_length: build [K, N] stacked array + metadata
        for L, pairs_dict in self.pair_arrays.items():
            pair_keys = sorted(pairs_dict.keys(), key=lambda x: (x[0], x[1]))
            arrays = [pairs_dict[k] for k in pair_keys]
            if not arrays:
                continue
            stacked = np.stack(arrays, axis=0).astype(np.float32)  # [K, N]
            fname = f"pairs_{L}.dat"
            fpath = os.path.join(mmap_dir, fname)
            fp = np.memmap(fpath, dtype=np.float32, mode="w+", shape=stacked.shape)
            fp[:] = stacked[:]
            fp.flush()
            del fp, stacked
            _EVO_MMAP_FILES.append(fpath)

            # Metadata (pair_key -> row_index)
            row_map = {}
            for i, (feat, agg) in enumerate(pair_keys):
                row_map[f"{feat}|||{agg}"] = i
            pair_arrays_meta[str(L)] = {
                "fname": fname,
                "shape": list(arrays[0].shape),  # [N]
                "row_map": row_map,
            }

        # Save seq_labels
        for L, arr in self.seq_labels.items():
            fname = f"labels_{L}.dat"
            fpath = os.path.join(mmap_dir, fname)
            fp = np.memmap(fpath, dtype=arr.dtype, mode="w+", shape=arr.shape)
            fp[:] = arr[:]
            fp.flush()
            del fp
            _EVO_MMAP_FILES.append(fpath)
            seq_labels_meta[str(L)] = {"fname": fname, "shape": list(arr.shape), "dtype": str(arr.dtype)}

        self._mmap_dir = mmap_dir
        self._mmap_meta = {
            "pair_arrays_meta": pair_arrays_meta,
            "seq_labels_meta": seq_labels_meta,
        }
        logger.info(f"Memmap saved: {mmap_dir} ({len(_EVO_MMAP_FILES)} files)")
        return mmap_dir, pair_arrays_meta, seq_labels_meta

    def set_tindex_map(self, name_to_pair: Dict[str, Tuple[str, str]]):
        """Build t_index -> (feat, agg) mapping from name_to_pair."""
        self._tindex_to_pair: Dict[int, Tuple[str, str]] = {}
        for tname, pair in name_to_pair.items():
            # tname: "t_0", "t_1", ...
            try:
                idx = int(tname[2:])
                self._tindex_to_pair[idx] = pair
            except (ValueError, IndexError):
                pass


# ===================================================================
# DEAP toolbox construction
# ===================================================================
def build_toolbox(
    pset: gp.PrimitiveSet,
    name_to_pair: Dict[str, Tuple[str, str]],
    gene_pool: GenePool,
    rng: random.Random,
    real_evaluator: "RealFitnessEvaluator | None" = None,
    factor_library: Optional[list] = None,
    factor_injection_rate: float = 0.15,
    pool: "multiprocessing.pool.Pool | None" = None,
) -> base.Toolbox:
    """Build DEAP toolbox, register all GP operations.

    pool: optional multiprocessing pool. When passed, toolbox.map uses parallel evaluation.
    """

    # Clear old creator types (avoid duplicate definition errors)
    for attr in ["FitnessMax", "Individual"]:
        if hasattr(creator, attr):
            delattr(creator, attr)

    if real_evaluator is not None:
        real_evaluator.set_tindex_map(name_to_pair)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Tree generation: 50% full, 50% grow
    toolbox.register("expr_full", gp.genFull, pset=pset, min_=1, max_=4)
    toolbox.register("expr_grow", gp.genGrow, pset=pset, min_=1, max_=4)

    def gen_half_and_half():
        method = rng.choice(["full", "grow"])
        if method == "full":
            return toolbox.expr_full()
        else:
            return toolbox.expr_grow()

    toolbox.register("expr", gen_half_and_half)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness function: prefer real evaluation
    if real_evaluator is not None:
        toolbox.register("evaluate", real_evaluator.evaluate)
    else:
        toolbox.register("evaluate", make_eval_func(name_to_pair, gene_pool, rng))

    # Parallel map: if pool is provided, register as toolbox.map (overrides default Python map)
    if pool is not None:

        def _parallel_map(func, iterable):
            """Serialize individuals via pickle, send to worker processes for evaluation."""
            items = [pickle.dumps(ind) for ind in iterable]
            results = pool.map(_evo_worker_evaluate, items)
            return results

        toolbox.register("map", _parallel_map)
        logger.info(f"Enabled multiprocess parallel evaluation (workers={pool._processes})")

    # Selection
    toolbox.register("select", tools.selTournament, tournsize=7)

    # Crossover (single-point)
    toolbox.register("mate", gp.cxOnePoint)

    # Mutation: prefer factor injection mutation
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=0, max_=2)
    if factor_library:
        toolbox.register(
            "mutate",
            functools.partial(
                mutate_with_factor_injection,
                factor_library=factor_library,
                injection_rate=factor_injection_rate,
                rng=rng,
            ),
            pset=pset,
            expr_gen=toolbox.expr_mut,
        )
    else:
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Tree size/depth limits
    toolbox.decorate("mate", gp.staticLimit(key=lambda ind: len(ind), max_value=40))
    toolbox.decorate("mutate", gp.staticLimit(key=lambda ind: len(ind), max_value=40))

    return toolbox


# ===================================================================
# Fitness evaluation (proxy fitness, based on tree structure analysis)
# ===================================================================
def make_eval_func(
    name_to_pair: Dict[str, Tuple[str, str]],
    gene_pool: GenePool,
    rng: random.Random,
):
    """Create fitness evaluation function."""
    existing_sigs = gene_pool.existing_signatures
    rarity_map = gene_pool.feature_rarity

    def evaluate(individual) -> Tuple[float]:
        """Extract feature set from DEAP individual, compute proxy fitness."""
        # Extract (feat, agg) pairs from terminals
        # In DEAP, all nodes are Primitive objects; arity==0 means terminal
        feats_set: Set[str] = set()
        for node in individual:
            if node.arity == 0 and node.name in name_to_pair:
                feat, _agg = name_to_pair[node.name]
                feats_set.add(feat)

        feat_list = sorted(feats_set)
        n_unique_feats = len(feat_list)
        tree_size = len(individual)

        # 1) Novelty: Jaccard distance to existing signatures
        if feats_set and existing_sigs:
            # Sampling for efficiency
            sample_sigs = list(existing_sigs)
            if len(sample_sigs) > 200:
                sample_sigs = random.Random(42).sample(sample_sigs, 200)
            novelty_scores = []
            for sig in sample_sigs:
                other_feats = set(sig.split(","))
                inter = len(feats_set & other_feats)
                union = len(feats_set | other_feats)
                jaccard = inter / union if union > 0 else 0
                novelty_scores.append(1.0 - jaccard)
            novelty_scores.sort(reverse=True)
            novelty = sum(novelty_scores[:20]) / min(20, len(novelty_scores))
        else:
            novelty = 1.0

        # 2) Complexity score
        if tree_size < 2:
            complexity = 0.3
        elif tree_size <= 5:
            complexity = 0.6
        elif tree_size <= 20:
            complexity = 1.0
        elif tree_size <= 35:
            complexity = 0.7
        else:
            complexity = 0.3

        # 3) Feature density (unique features / number of terminals)
        n_terminals = sum(1 for node in individual if node.arity == 0)
        density = n_unique_feats / max(n_terminals, 1)
        if density >= 0.6:
            density_bonus = 1.0
        elif density >= 0.3:
            density_bonus = 0.6
        else:
            density_bonus = 0.2

        # 4) Feature count bonus
        if n_unique_feats >= 4:
            feat_count_bonus = 1.0
        elif n_unique_feats >= 3:
            feat_count_bonus = 0.85
        elif n_unique_feats >= 2:
            feat_count_bonus = 0.6
        else:
            feat_count_bonus = 0.2

        # 5) Rarity
        rarity = sum(rarity_map.get(f, 0.5) for f in feats_set) / max(n_unique_feats, 1)

        # 6) Uniqueness
        sig = ",".join(feat_list)
        uniqueness = 1.0 if sig not in existing_sigs else 0.2

        fitness = (
            novelty * 0.20
            + complexity * 0.10
            + density_bonus * 0.10
            + feat_count_bonus * 0.15
            + rarity * 0.20
            + uniqueness * 0.25
        )
        fitness += rng.uniform(0, 0.0005)  # Tiny perturbation to break ties
        return (fitness,)

    return evaluate


# ===================================================================
# Post-evolution export
# ===================================================================
def deap_tree_summary(individual, name_to_pair: Dict[str, Tuple[str, str]]) -> Dict:
    """Extract summary info from DEAP individual."""
    feats = set()
    aggs = set()
    for node in individual:
        if node.arity == 0 and node.name in name_to_pair:
            feat, agg = name_to_pair[node.name]
            feats.add(feat)
            aggs.add(agg)
    return {
        "features": sorted(feats),
        "aggregators": sorted(aggs),
        "tree_size": len(individual),
    }


def export_evolved_factors(
    individuals: list,
    name_to_pair: Dict[str, Tuple[str, str]],
    gene_pool: GenePool,
    code_generator: CodeGenerator,
    existing_factors: List[dict],
    output_path: str,
    rng: random.Random,
):
    """Export evolved DEAP individuals in valid_factors.json-compatible format."""
    entries: List[dict] = []
    seen_names: Set[str] = set()
    for f in existing_factors:
        seen_names.add(f.get("name", ""))

    # Sort by fitness
    sorted_inds = sorted(individuals, key=lambda ind: ind.fitness.values[0], reverse=True)

    # ---- Deduplicate by feature combination signature (keep only highest-fitness per group) ----
    seen_feat_sigs: Set[str] = set()
    deduped: list = []
    for ind in sorted_inds:
        summary = deap_tree_summary(ind, name_to_pair)
        sig = ",".join(summary["features"])
        if sig not in seen_feat_sigs:
            seen_feat_sigs.add(sig)
            deduped.append(ind)
    logger.info(
        f"Deduplication: {len(sorted_inds)} -> {len(deduped)} unique feature combinations"
    )
    sorted_inds = deduped

    for rank, ind in enumerate(sorted_inds):
        summary = deap_tree_summary(ind, name_to_pair)
        feats = summary["features"]
        if len(feats) < 1:
            continue

        # Generate code
        try:
            code = code_generator.generate(ind, name_to_pair)
        except Exception as e:
            logger.warning(f"Code generation failed (rank {rank+1}): {e}")
            continue

        # Syntax validation (simulates FactorEngine compilation)
        try:
            func_code = "def _factor_func(window, idx, np):\n"
            func_code += "    feat = window[-1]\n"
            for line in code.strip().split("\n"):
                func_code += f"    {line}\n"
            compile(func_code, f"<evo_deap:{rank}>", "exec")
        except SyntaxError as e:
            logger.warning(f"Syntax error (rank {rank+1}): {e}")
            continue

        # Unique name
        feat_abbrev = "_".join(feats[:3]).replace("Self_", "").replace("Other1_", "O1_")
        if len(feats) > 3:
            feat_abbrev += "_etc"
        base_name = f"evo_{rng.choice(gene_pool.targets)}_{feat_abbrev}_{rank}"
        # Randomly select target again (better diversity than all factors having the same target)
        name = base_name[:80]
        counter = 1
        while name in seen_names:
            name = f"{base_name[:70]}_v{counter}"
            counter += 1
        seen_names.add(name)

        # Select target: prefer inference from existing factor feature similarity
        target = rng.choice(gene_pool.targets)

        # Select seq_length (prefer medium-range)
        seq_length = rng.choices(
            gene_pool.seq_lengths,
            weights=gene_pool.seqlen_weights,
            k=1,
        )[0]

        description = (
            f"[DEAP-GP evolved] Feature combination: {', '.join(feats[:8])}. "
            f" Aggregators: {', '.join(summary['aggregators'][:5])}. "
            f" Tree size: {summary['tree_size']}. "
        )

        entry = {
            "name": name,
            "description": description,
            "target": target,
            "code": code,
            "mode": "row",
            "seq_length": seq_length,
            "best_auc": None,
            "best_f1": None,
            "best_class": None,
            "valid_classes": [],
            "per_class": {},
            "n_samples": None,
            "saved_at": datetime.now().isoformat(),
            "_evolution_meta": {
                "fitness": round(ind.fitness.values[0], 6),
                "tree_size": summary["tree_size"],
                "features": feats,
                "aggregators": summary["aggregators"],
            },
        }
        entries.append(entry)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    # Structured lines for GUI real-time table update
    for entry in entries:
        _ev_info = {
            "name": entry["name"],
            "target": entry["target"],
            "seq_length": entry["seq_length"],
            "best_auc": entry.get("best_auc"),
            "best_f1": entry.get("best_f1"),
            "valid_classes": entry.get("valid_classes", []),
            "fitness": entry.get("_evolution_meta", {}).get("fitness"),
            "feats": entry.get("_evolution_meta", {}).get("features", [])[:10],
            "tree_size": entry.get("_evolution_meta", {}).get("tree_size"),
        }
        print(f"__FACTOR__{json.dumps(_ev_info, ensure_ascii=False)}", flush=True)

    logger.info(f"Exported {len(entries)} evolved factors -> {output}")
    return entries


# ===================================================================
# CLI
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="DEAP-GP Factor Evolution Engine -- crosses and evolves existing factors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python evolution.py
              python evolution.py --factors memory/valid_factors.json --output memory/evolved_factors.json
              python evolution.py --mu 128 --lambda 256 --seed 42
              python evolution.py --validate --mu 50 --lambda 50 --generations 10
        """),
    )
    p.add_argument("--factors", default="memory/valid_factors.json",
                   help="Input factor JSON path (default: memory/valid_factors.json)")
    p.add_argument("--output", default="memory/evolved_factors.json",
                   help="Output factor JSON path (default: memory/evolved_factors.json)")
    p.add_argument("--mu", type=int, default=100,
                   help="mu (parent population size, eaMuPlusLambda) (default: 100)")
    p.add_argument("--lambda", dest="lambda_", type=int, default=100,
                   help="lambda (offspring per generation, eaMuPlusLambda) (default: 100)")
    p.add_argument("--generations", type=int, default=50,
                   help="Number of generations (default: 50)")
    p.add_argument("--survivors", type=int, default=1,
                   help="Number of convergence cycles to run (0 = unlimited, keeps cycling until "
                        "manually stopped; default: 1)")
    p.add_argument("--crossover-rate", type=float, default=0.65,
                   help="Crossover probability (default: 0.7)")
    p.add_argument("--mutation-rate", type=float, default=0.35,
                   help="Mutation probability (default: 0.3)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed (auto-generates if not specified)")
    p.add_argument("--pop-size", type=int, default=None,
                   help="Population size (deprecated, use --mu)")
    p.add_argument("--seed-factors", type=int, default=30,
                   help="Number of factor seeds to inject into initial population (0=no injection, recommended 30-50%% of mu)")
    p.add_argument("--factor-injection-rate", type=float, default=0.3,
                   help="Probability of injecting subtrees from factor library during mutation (0=disabled, 0.15=15%%)")

    # ---- Convergence cycle mode ----
    p.add_argument("--std-threshold", type=float, default=0.0003,
                   help="Convergence std threshold (0=disable, recommended 0.001. Below this, auto-finish and restart)")
    p.add_argument("--std-patience", type=int, default=3,
                   help="Number of consecutive generations below threshold to declare convergence (default: 3)")

    # ---- Real validation mode ----
    p.add_argument("--validate", action="store_true", default=True,
                   help="Enable real LightGBM fitness evaluation (requires loading dataset)")
    p.add_argument("--validate-config", default="config/seq/1.yaml",
                   help="Validation mode main config path (default: config/seq/1.yaml)")
    p.add_argument("--validate-config-validation", default="config/validation.yaml",
                   help="Validation mode secondary config, merged with main (default: config/validation.yaml)")
    p.add_argument("--validate-max-samples", type=int, default=100000,
                   help="Validation mode max sample frames (default: 10000)")
    p.add_argument("--validate-estimators", type=int, default=50,
                   help="LGBM n_estimators (default: 30)")
    p.add_argument("--validate-cv", type=int, default=3,
                   help="LGBM cross-validation folds (default: 3)")

    # ---- GPU acceleration ----
    p.add_argument("--gpu", action="store_true", default=False,
                   help="Enable GPU acceleration (prefer XGBoost GPU, fall back LGBM GPU)")
    p.add_argument("--gpu-backend", choices=["auto", "xgboost", "lightgbm"], default="auto",
                   help="GPU backend selection (default: auto -- prefer XGBoost)")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Number of parallel evaluation worker processes (0=sequential, -1=auto-detect CPU cores)")

    # ---- Algorithm optimization ----
    p.add_argument("--reduced-cv", action="store_true", default=False,
                   help="Reduce CV folds (2-fold instead of 3-fold, speeds up fitness evaluation)")

    # Compatible with old parameters
    args = p.parse_args()
    if args.pop_size is not None and args.mu == 200:
        args.mu = args.pop_size
    return args


# ===================================================================
# Evolved factor holdout validation (get per_class AUC on insertion)
# ===================================================================
def _validate_factor_holdout(
    factor_entry: dict,
    evaluator: "RealFitnessEvaluator",
    engine: "FactorEngine",
    validator: "FactorValidator",
    rng: random.Random,
    holdout_ratio: float = 0.2,
) -> dict:
    """Run train/val holdout single-factor validation on evolved factor to get per_class AUC.

    Performs temporal splitting on evaluator's full data (maintains segment continuity):
      - train: first (1-holdout_ratio) contiguous segments
      - val:   last holdout_ratio contiguous segments

    Returns validation result dict; returns empty per_class on failure.
    """
    from src.factor_engine import FactorEngine
    from src.validator import FactorValidator

    try:
        sl = int(factor_entry.get("seq_length", 5))
        kp_full = evaluator._full_kp
        lb_full = evaluator._full_lb
        flat_attributes = list(evaluator.flat_attributes)
        T = min(len(kp_full), len(lb_full))
        kp_full = kp_full[:T].astype(np.float32)
        lb_full = lb_full[:T].astype(np.int64)

        if T < 200 or sl > T // 4:
            logger.warning(f"[Holdout] Insufficient data (T={T}, seq_length={sl}), skipping validation")
            return {"per_class": {}, "best_auc": None}

        # Temporal split train/val (maintain segment continuity)
        split_point = int(T * (1 - holdout_ratio))
        # Align near behavior boundaries (avoid splitting in the middle of a behavior)
        if split_point < T - 10:
            window = lb_full[max(0, split_point - 20):min(T, split_point + 20)]
            boundaries = np.where(np.diff(window) != 0)[0]
            if len(boundaries) > 0:
                offset = boundaries[len(boundaries) // 2] - 20
                split_point = max(T // 4, min(T * 3 // 4, split_point + int(offset)))

        train_kp = kp_full[:split_point]
        train_lb = lb_full[:split_point]
        val_kp = kp_full[split_point:]
        val_lb = lb_full[split_point:]

        logger.info(
            f"[Holdout] Data split: train={len(train_kp)} frames, val={len(val_kp)} frames, "
            f"factor={factor_entry.get('name','?')[:30]}, seq_length={sl}"
        )

        # Build centered windows
        def _build_windows(kp_raw, sl_inner):
            if sl_inner == 1:
                return kp_raw[:, np.newaxis, :]
            half_before = (sl_inner - 1) // 2
            half_after = sl_inner - 1 - half_before
            kp_padded = np.pad(kp_raw, ((half_before, half_after), (0, 0)), mode="edge")
            idx = np.arange(len(kp_raw))[:, None] + np.arange(sl_inner)[None, :]
            return np.ascontiguousarray(kp_padded[idx], dtype=np.float32)

        train_windows = _build_windows(train_kp, sl)
        val_windows = _build_windows(val_kp, sl)
        train_labels = train_lb.astype(np.int64)
        val_labels = val_lb.astype(np.int64)

        # Compute factor values
        mode = factor_entry.get("mode", "row")
        if mode == "batch":
            train_vals = engine.compute_factor_batch(factor_entry, train_windows, flat_attributes)
            val_vals = engine.compute_factor_batch(factor_entry, val_windows, flat_attributes)
        else:
            train_vals = engine.compute_factor(factor_entry, train_windows, flat_attributes)
            val_vals = engine.compute_factor(factor_entry, val_windows, flat_attributes)

        if train_vals is None or val_vals is None:
            logger.warning(f"[Holdout] Factor computation failed")
            return {"per_class": {}, "best_auc": None}

        # Align labels
        min_len = min(len(train_vals), len(train_labels), len(val_vals), len(val_labels))
        train_vals = train_vals[:min_len]
        train_labels = train_labels[:min_len]
        val_vals = val_vals[:min_len]
        val_labels = val_labels[:min_len]

        # Run holdout validation
        result = validator.validate_single_holdout(
            train_factor=train_vals,
            train_labels=train_labels,
            val_factor=val_vals,
            val_labels=val_labels,
        )
        logger.info(
            f"[Holdout] {factor_entry.get('name','?')[:40]}: "
            f"best_auc={result.get('best_auc')}, "
            f"valid={result.get('valid')}, "
            f"valid_classes={[v['class'] for v in result.get('valid_classes', [])]}"
        )
        return result

    except Exception as e:
        logger.warning(f"[Holdout] Validation exception: {e}")
        return {"per_class": {}, "best_auc": None}


def main():
    args = parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        logger.info(f"--seed not specified, using random seed: {args.seed}")
    random.seed(args.seed)  # DEAP internals (gp.genFull/genGow/crossover/mutation/selection) depend on global random
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ---- Load existing factors ----
    factors_path = Path(args.factors)
    if not factors_path.exists():
        logger.error(f"Factor file does not exist: {factors_path}")
        sys.exit(1)

    with open(factors_path, "r", encoding="utf-8") as f:
        existing_factors = json.load(f)
    logger.info(f"Loaded {len(existing_factors)} existing factors")

    # ---- Build gene pool ----
    gene_pool = GenePool(existing_factors)

    # ---- Build DEAP primitive set ----
    pset, name_to_pair = build_pset(gene_pool)
    logger.info(
        f"DEAP primitive set: {len(name_to_pair)} terminals (feature-aggregator pairs), "
        f"4 binary operations, 5 unary operations"
    )

    # ---- Real validation mode: load dataset ----
    # ---- Parse parallel worker count ----
    num_workers = args.num_workers
    if num_workers == -1:
        num_workers = os.cpu_count() or 8

    # ---- GPU backend selection ----
    use_gpu = args.gpu
    gpu_backend = args.gpu_backend
    if use_gpu and gpu_backend == "auto":
        try:
            from xgboost import XGBClassifier  # noqa: F401
            gpu_backend = "xgboost"
            logger.info("GPU backend: XGBoost (device=cuda)")
        except ImportError:
            logger.info("GPU backend: LightGBM (device=cuda) -- XGBoost not available")
            gpu_backend = "lightgbm"

    real_evaluator = None
    if args.validate:
        mode_desc = ["LightGBM"]
        if use_gpu:
            mode_desc.append(f"GPU({gpu_backend})")
        if num_workers > 0:
            mode_desc.append(f"{num_workers}workers")
        logger.info(f"=== Real validation mode ({', '.join(mode_desc)}): using macro AUC as fitness ===")
        try:
            real_evaluator = RealFitnessEvaluator(
                config_path=args.validate_config,
                gene_pool=gene_pool,
                max_samples=args.validate_max_samples,
                n_estimators=args.validate_estimators,
                cv_folds=args.validate_cv,
                config_validation=args.validate_config_validation,
                use_gpu=use_gpu,
                gpu_backend=gpu_backend,
                reduced_cv=args.reduced_cv,
                num_workers=num_workers,
            )
            per_eval = 0.3 if use_gpu else 0.7
            if num_workers > 0:
                per_eval /= num_workers
            logger.info(
                f"Real evaluator ready. Each generation: lambda={args.lambda_} evaluations, "
                f"estimated per generation ~{args.lambda_ * per_eval:.0f}-{args.lambda_ * per_eval * 2:.0f}s"
            )
        except Exception as e:
            logger.error(f"Real data loading failed: {e}")
            logger.error(
                "Cannot fall back to proxy fitness when --validate is set. "
                "Check that dataset_config_file in your config YAML points to valid data, "
                "or use the GUI to configure dataset paths."
            )
            raise

    # ---- Build factor subtree library (for mutation injection) ----
    factor_library = None
    if args.factor_injection_rate > 0 or args.seed_factors > 0:
        factor_library = build_factor_subtree_library(
            existing_factors, gene_pool, name_to_pair, rng,
        )
        logger.info(f"Factor subtree library: {len(factor_library)} unique feature combinations")

    # ---- Set terminal index map (needed for both pool creation and build_toolbox) ----
    if real_evaluator is not None:
        real_evaluator.set_tindex_map(name_to_pair)

    # ---- Create multiprocessing pool (if needed) ----
    pool = None
    if num_workers > 0 and real_evaluator is not None:
        import multiprocessing as mp
        mmap_dir = getattr(real_evaluator, "_mmap_dir", "")
        mmap_meta = getattr(real_evaluator, "_mmap_meta", {})
        if mmap_dir and mmap_meta:
            ctx = mp.get_context("spawn")
            pair_meta = mmap_meta.get("pair_arrays_meta", {})
            labels_meta = mmap_meta.get("seq_labels_meta", {})
            pool = ctx.Pool(
                processes=num_workers,
                initializer=_evo_worker_init,
                initargs=(
                    mmap_dir, pair_meta, labels_meta,
                    real_evaluator.classes_sorted,
                    real_evaluator._tindex_to_pair,
                    use_gpu, gpu_backend,
                    args.validate_estimators,
                    args.validate_cv,
                    args.reduced_cv,
                ),
            )
            logger.info(f"Multiprocess pool created: {num_workers} worker processes")
        else:
            logger.warning("Memmap data not available, falling back to single-process evaluation.")
            num_workers = 0

    # ---- Build toolbox ----
    toolbox = build_toolbox(
        pset, name_to_pair, gene_pool, rng, real_evaluator,
        factor_library=factor_library,
        factor_injection_rate=args.factor_injection_rate,
        pool=pool,
    )

    # ---- Statistics ----
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # ---- Format printing ----
    def fmt_val(v: float) -> str:
        if v == 0:
            return "    0.0000"
        av = abs(v)
        if av < 0.0001:
            return f"{v:10.2e}"
        elif av < 0.01:
            return f"{v:10.6f}"
        elif av < 10:
            return f"{v:10.4f}"
        elif av < 100:
            return f"{v:10.3f}"
        else:
            return f"{v:10.2f}"

    header = f"{'gen':>4s} {'nevals':>7s} {'avg':>10s} {'std':>10s} {'min':>10s} {'max':>10s}"

    def init_population():
        """Initialize new population: seeding + random fill."""
        n_seeds = min(args.seed_factors, args.mu)
        seeds_list = []
        if n_seeds > 0:
            seeds_list = seed_individuals_from_factors(
                existing_factors, gene_pool, name_to_pair, pset, rng, n_seeds
            )
        n_random = args.mu - len(seeds_list)
        p = toolbox.population(n=n_random) if n_random > 0 else []
        p.extend(seeds_list)
        rng.shuffle(p)
        return p[:args.mu]

    code_gen = CodeGenerator()

    # ---- Holdout validator (for per_class AUC on insertion) ----
    holdout_engine = None
    holdout_validator = None
    if real_evaluator is not None:
        from src.factor_engine import FactorEngine
        from src.validator import FactorValidator
        from mining.discovery import _load_merged_config
        import yaml
        try:
            # Load config to build FactorValidator (reuse RealFitnessEvaluator's config paths)
            config_paths = [str(args.validate_config)]
            if args.validate_config_validation:
                vp = Path(args.validate_config_validation)
                if not vp.is_absolute():
                    vp = Path(args.validate_config).parent / vp
                if vp.exists():
                    config_paths.append(str(vp))
            cfg = _load_merged_config(*config_paths)
            # Ensure validation section exists (FactorValidator depends on it)
            if "validation" not in cfg:
                cfg["validation"] = {
                    "min_auc": 0.65, "min_f1": 0.15,
                    "lgbm_params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1, "num_leaves": 15, "random_state": 42, "n_jobs": -1},
                    "early_stopping_rounds": 20,
                }
            holdout_engine = FactorEngine(cfg)
            holdout_validator = FactorValidator(cfg)
            logger.info("Holdout validator ready: per_class AUC will be computed on insertion")
        except Exception as e:
            logger.warning(f"Holdout validator initialization failed: {e}, skipping per_class validation on insertion")

    # ---- Convergence cycle mode ----
    use_convergence = args.std_threshold > 0
    max_cycles = args.survivors if use_convergence else 1
    cycle = 0
    persistent_library: List[dict] = []
    # Load existing persistent library
    lib_path = Path(args.output)
    if use_convergence and lib_path.exists():
        try:
            with open(lib_path, "r", encoding="utf-8") as f:
                persistent_library = json.load(f)
            logger.info(f"Loaded existing factor library: {len(persistent_library)} factors -> {lib_path}")
        except Exception:
            persistent_library = []
    lib_feat_sigs: Set[str] = set()
    for f in persistent_library:
        feats = f.get("_evolution_meta", {}).get("features", [])
        if feats:
            lib_feat_sigs.add(",".join(sorted(feats)))

    eval_mode = "LGBM macro AUC" if real_evaluator else "Proxy fitness"

    while True:
        if max_cycles > 0 and cycle >= max_cycles:
            break
        cycle += 1

        # Initialize
        if real_evaluator and use_convergence:
            real_evaluator._resample_and_build(seed_offset=cycle, base_seed=args.seed)
            # Convergence mode: rebuild memmap and pool per cycle after data resampling to avoid disk accumulation and stale data
            if pool is not None:
                pool.close()
                pool.join()
            if num_workers > 0:
                real_evaluator._save_memmap()
                mmap_dir = getattr(real_evaluator, "_mmap_dir", "")
                mmap_meta = getattr(real_evaluator, "_mmap_meta", {})
                if mmap_dir and mmap_meta:
                    import multiprocessing as mp
                    ctx = mp.get_context("spawn")
                    pair_meta = mmap_meta.get("pair_arrays_meta", {})
                    labels_meta = mmap_meta.get("seq_labels_meta", {})
                    pool = ctx.Pool(
                        processes=num_workers,
                        initializer=_evo_worker_init,
                        initargs=(
                            mmap_dir, pair_meta, labels_meta,
                            real_evaluator.classes_sorted,
                            real_evaluator._tindex_to_pair,
                            use_gpu, gpu_backend,
                            args.validate_estimators,
                            args.validate_cv,
                            args.reduced_cv,
                        ),
                    )

                    def _parallel_map(func, iterable):
                        items = [pickle.dumps(ind) for ind in iterable]
                        return pool.map(_evo_worker_evaluate, items)

                    toolbox.register("map", _parallel_map)
                    logger.info(
                        f"[Cycle {cycle}] Multiprocess pool rebuilt: {num_workers} worker processes"
                    )
            else:
                pool = None
        pop = init_population()
        hall_of_fame = tools.HallOfFame(50)  # Hall of fame for each cycle
        if use_convergence:
            logger.info(
                f"=== Cycle {cycle}/{max_cycles if max_cycles > 0 else 'inf'} [{eval_mode}]: "
                f"mu={args.mu}, lambda={args.lambda_}, std_threshold={args.std_threshold} ==="
            )
        else:
            logger.info(
                f"=== Starting DEAP eaMuPlusLambda evolution [{eval_mode}]: mu={args.mu}, lambda={args.lambda_}, "
                f"gen={args.generations}, cxpb={args.crossover_rate}, mutpb={args.mutation_rate} ==="
            )

        print(header, flush=True)

        # Evaluate initial population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        hall_of_fame.update(pop)

        record = stats.compile(pop) if stats else {}
        print(f"{0:4d} {len(invalid_ind):7d} {fmt_val(record['avg'])} {fmt_val(record['std'])} {fmt_val(record['min'])} {fmt_val(record['max'])}", flush=True)

        # Convergence detection state
        below_threshold_count = 0
        converged = False
        max_gen = args.generations if not use_convergence else 999999

        # Evolution by generation
        for gen in range(1, max_gen + 1):
            offspring = toolbox.select(pop, len(pop))
            offspring = algorithms.varOr(offspring, toolbox, args.lambda_, args.crossover_rate, args.mutation_rate)

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            hall_of_fame.update(offspring)

            pop[:] = toolbox.select(pop + offspring, args.mu)

            record = stats.compile(pop) if stats else {}
            print(f"{gen:4d} {len(invalid_ind):7d} {fmt_val(record['avg'])} {fmt_val(record['std'])} {fmt_val(record['min'])} {fmt_val(record['max'])}", flush=True)

            # Convergence detection
            if use_convergence and record.get("std", 1.0) < args.std_threshold:
                below_threshold_count += 1
                if below_threshold_count >= args.std_patience:
                    converged = True
                    break
            else:
                below_threshold_count = 0

        # Post-convergence or max generation handling
        if use_convergence:
            if converged:
                logger.info(
                    f"Cycle {cycle}: std < {args.std_threshold} for {args.std_patience} consecutive generations, converged"
                )
            else:
                logger.info(
                    f"Cycle {cycle}: Reached max generations {args.generations} (not converged, std={record.get('std', 0):.6f})"
                )

            # Extract best individual from this cycle
            if len(hall_of_fame) == 0:
                continue
            best = hall_of_fame[0]
            best_summary = deap_tree_summary(best, name_to_pair)
            best_feats = best_summary["features"]
            feats_sig = ",".join(sorted(best_feats))

            if feats_sig in lib_feat_sigs:
                logger.info(
                    f"Cycle {cycle}: Best factor feature combination already exists, skipping insertion. "
                    f"feats={best_feats[:5]}"
                )
            else:
                lib_feat_sigs.add(feats_sig)
                # Generate code
                try:
                    best_code = code_gen.generate(best, name_to_pair)
                except Exception:
                    best_code = ""

                # Compile check
                try:
                    func_code = "def _factor_func(window, idx, np):\n    feat = window[-1]\n"
                    for line in best_code.strip().split("\n"):
                        func_code += f"    {line}\n"
                    compile(func_code, f"<evo_cycle{cycle}>", "exec")
                except SyntaxError:
                    logger.warning(f"Cycle {cycle}: Best factor code syntax error, skipping")
                    continue

                # Build temporary factor dict for holdout validation
                target = rng.choice(gene_pool.targets)
                seq_length = rng.choices(gene_pool.seq_lengths, weights=gene_pool.seqlen_weights, k=1)[0]
                temp_factor = {
                    "name": f"evo_cycle{cycle}_{best_feats[0] if best_feats else 'unknown'}",
                    "code": best_code,
                    "mode": "row",
                    "seq_length": seq_length,
                    "target": target,
                }

                # Holdout validation: get per_class AUC
                per_class = {}
                best_auc = None
                best_f1 = None
                best_class_val = None
                valid_classes = []
                if holdout_engine is not None and holdout_validator is not None and real_evaluator is not None:
                    logger.info(f"Cycle {cycle}: Running holdout single-factor validation...")
                    v_result = _validate_factor_holdout(
                        factor_entry=temp_factor,
                        evaluator=real_evaluator,
                        engine=holdout_engine,
                        validator=holdout_validator,
                        rng=rng,
                    )
                    per_class = v_result.get("per_class", {})
                    best_auc = v_result.get("best_auc")
                    best_f1 = v_result.get("best_f1")
                    best_class_val = v_result.get("best_class")
                    valid_classes = v_result.get("valid_classes", [])
                    if best_auc is not None:
                        logger.info(
                            f"Cycle {cycle}: holdout best_auc={best_auc:.4f} "
                            f"(class={best_class_val}), per_class={list(per_class.keys())}"
                        )

                entry = {
                    "name": temp_factor["name"],
                    "description": (
                        f"[Convergence Cycle {cycle}] Feature combination: {', '.join(best_feats[:8])}. "
                        f" Aggregators: {', '.join(best_summary['aggregators'][:5])}. "
                        f" Tree size: {best_summary['tree_size']}. "
                    ),
                    "target": target,
                    "code": best_code,
                    "mode": "row",
                    "seq_length": seq_length,
                    "best_auc": best_auc,
                    "best_f1": best_f1,
                    "best_class": best_class_val,
                    "valid_classes": valid_classes,
                    "per_class": per_class,
                    "n_samples": None,
                    "saved_at": datetime.now().isoformat(),
                    "_evolution_meta": {
                        "fitness": round(best.fitness.values[0], 6),
                        "tree_size": best_summary["tree_size"],
                        "features": best_feats,
                        "aggregators": best_summary["aggregators"],
                        "cycle": cycle,
                    },
                }
                persistent_library.append(entry)
                logger.info(
                    f"Cycle {cycle}: Inserted new factor {entry['name']}, "
                    f"fitness={best.fitness.values[0]:.6f}, "
                    f"feats={best_feats[:5]}, "
                    f"library total: {len(persistent_library)}"
                )

                # Write to file in real time
                with open(lib_path, "w", encoding="utf-8") as f:
                    json.dump(persistent_library, f, ensure_ascii=False, indent=2)

                # Structured line for GUI real-time table update
                _ev_info = {
                    "name": entry["name"],
                    "target": entry["target"],
                    "seq_length": entry["seq_length"],
                    "best_auc": entry.get("best_auc"),
                    "best_f1": entry.get("best_f1"),
                    "best_class": entry.get("best_class"),
                    "valid_classes": [{"class": vc.get("class"), "auc": vc.get("auc")}
                                      for vc in entry.get("valid_classes", [])],
                    "fitness": entry["_evolution_meta"]["fitness"],
                    "feats": entry["_evolution_meta"]["features"][:10],
                    "tree_size": entry["_evolution_meta"]["tree_size"],
                }
                print(f"__FACTOR__{json.dumps(_ev_info, ensure_ascii=False)}", flush=True)

            # End of cycle, immediately clean up memmap cache to free disk space
            if pool is not None:
                pool.close()
                pool.join()
                pool = None
            _evo_cleanup_mmap()
        else:
            # Non-convergence mode: original behavior
            break

    # ---- Summary ----
    if use_convergence:
        logger.info("=" * 60)
        logger.info(f"Convergence cycles complete. {cycle} cycles total, library total: {len(persistent_library)}")
        for i, entry in enumerate(persistent_library[-10:]):
            meta = entry["_evolution_meta"]
            logger.info(
                f"  {i+1}. {entry['name']}  "
                f"fitness={meta['fitness']:.6f}  "
                f"cycle={meta.get('cycle','?')}  "
                f"feats={meta['features'][:4]}"
            )
        logger.info(f"Factor library file: {lib_path}")
        logger.info(f"Run directly: python train_behavior.py --factors {lib_path}")
    else:
        logger.info("=" * 60)
        logger.info("Evolution complete. Hall of Fame summary:")
        for i, ind in enumerate(hall_of_fame[:10]):
            s = deap_tree_summary(ind, name_to_pair)
            logger.info(
                f"  {i+1}. fitness={ind.fitness.values[0]:.6f}  "
                f"feats={s['features'][:4]}  size={s['tree_size']}"
            )
        code_gen_local = CodeGenerator()
        export_evolved_factors(
            hall_of_fame, name_to_pair, gene_pool, code_gen_local,
            existing_factors, args.output, rng,
        )
        logger.info(
            f"Complete. {len(hall_of_fame)} factors -> {args.output}. "
            f" Run directly: python train_behavior.py --factors {args.output}"
        )

    # ---- Cleanup multiprocess pool and memmap ----
    if pool is not None:
        pool.close()
        pool.join()
        logger.info("Multiprocess pool closed")
    _evo_cleanup_mmap()

if __name__ == "__main__":
    main()
