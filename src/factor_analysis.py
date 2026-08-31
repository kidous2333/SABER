"""
analysis.py (imported from src/factor_analysis)
Factor-level analysis logic — UMAP dimensionality reduction, feature usage statistics,
factor metadata extraction.

Independent of train_behavior.py; can be invoked directly via the analysis.py CLI,
or embedded on demand in the train_behavior.py pipeline.
"""

import json
import logging
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 1) Factor metadata extraction
# ------------------------------------------------------------------
def extract_factor_metadata(factors: list) -> dict:
    """Extract aggregate statistics from a list of factors.

    Returns
    -------
    dict with keys:
      n_factors, n_targets, target_counts, seq_length_distribution,
      auc_distribution, features_per_factor, tree_sizes,
      aggregator_counts, feature_usage, feature_agg_pairs,
      per_class_auc_matrix (factor_names, class_names, auc_array)
    """
    from collections import defaultdict

    n_factors = len(factors)
    target_counter: Counter = Counter()
    seqlen_counter: Counter = Counter()
    aucs: list = []
    feature_counts: list = []
    tree_sizes: list = []
    agg_counter: Counter = Counter()
    feature_counter: Counter = Counter()
    pair_counter: Counter = Counter()
    all_class_ids: Set[str] = set()

    for f in factors:
        target_counter[f.get("target", "unknown")] += 1
        sl = f.get("seq_length", 1)
        if isinstance(sl, (int, float)) and sl > 0:
            seqlen_counter[int(sl)] += 1

        best_auc = f.get("best_auc")
        if best_auc is not None:
            aucs.append(float(best_auc))

        # per-class AUC
        pc = f.get("per_class", {})
        for cid in pc:
            all_class_ids.add(str(cid))

        # Feature counting
        code = f.get("code", "")
        feats_in_code: Set[str] = set()
        for m in re.finditer(r"idx\.get\('([^']+)'", code):
            feat = m.group(1)
            if "%" not in feat and not feat.endswith("_"):
                feats_in_code.add(feat)
                feature_counter[feat] += 1
        feature_counts.append(len(feats_in_code))

        # Aggregators
        AGG_KEYWORDS = {
            "np.mean": "mean", "np.std": "std", "np.var": "var",
            "np.max": "max", "np.min": "min",
            "np.polyfit": "trend", "polyfit": "trend",
            "np.median": "mad", "np.sum": "mean",
        }
        aggs_in_code: Set[str] = set()
        for kw, agg in AGG_KEYWORDS.items():
            if kw in code:
                aggs_in_code.add(agg)
                agg_counter[agg] += 1
        for feat in feats_in_code:
            for agg in aggs_in_code:
                pair_counter[(feat, agg)] += 1

        # Tree size (evolution factors only)
        evo_meta = f.get("_evolution_meta", {})
        ts = evo_meta.get("tree_size")
        if ts is not None:
            tree_sizes.append(int(ts))

    target_names = sorted(target_counter.keys())
    class_ids_sorted = sorted(all_class_ids, key=lambda x: int(x) if x.isdigit() else x)

    # Build per-class AUC matrix
    factor_names = [f.get("name", f"factor_{i}") for i, f in enumerate(factors)]
    n_classes = len(class_ids_sorted)
    auc_matrix = np.full((n_factors, n_classes), np.nan, dtype=np.float32)
    for i, f in enumerate(factors):
        pc = f.get("per_class", {})
        for j, cid in enumerate(class_ids_sorted):
            entry = pc.get(cid)
            if isinstance(entry, dict):
                auc_matrix[i, j] = float(entry.get("auc", np.nan))
            elif isinstance(entry, (int, float)):
                auc_matrix[i, j] = float(entry)

    return {
        "n_factors": n_factors,
        "n_targets": len(target_names),
        "target_names": target_names,
        "target_counts": {t: target_counter[t] for t in target_names},
        "seq_length_distribution": {str(k): v for k, v in sorted(seqlen_counter.items())},
        "auc_distribution": sorted(aucs),
        "features_per_factor": feature_counts,
        "tree_sizes": sorted(tree_sizes),
        "aggregator_counts": {k: agg_counter[k] for k in sorted(agg_counter.keys())},
        "feature_usage": {k: feature_counter[k] for k in sorted(feature_counter.keys(), key=lambda x: -feature_counter[x])},
        "feature_agg_pairs": {f"{feat}|{agg}": cnt for (feat, agg), cnt in sorted(pair_counter.items(), key=lambda x: -x[1])},
        "factor_names": factor_names,
        "class_ids_sorted": class_ids_sorted,
        "auc_matrix": auc_matrix,
    }


# ------------------------------------------------------------------
# 2) Feature utilization matrix
# ------------------------------------------------------------------
def build_feature_utilization_matrix(factors: list, top_n: int = 40) -> dict:
    """Build a feature x aggregator usage frequency matrix.

    Returns dict: features (list), aggregators (list), matrix [F, A], top_n applied to features.
    """
    pair_counter: Counter = Counter()
    feature_counter: Counter = Counter()
    agg_counter: Counter = Counter()

    AGG_KEYWORDS = {
        "np.mean": "mean", "np.std": "std", "np.var": "var",
        "np.max": "max", "np.min": "min",
        "np.polyfit": "trend", "polyfit": "trend",
        "np.median": "mad",
    }

    for f in factors:
        code = f.get("code", "")
        feats_in_code: Set[str] = set()
        for m in re.finditer(r"idx\.get\('([^']+)'", code):
            feat = m.group(1)
            if "%" not in feat and not feat.endswith("_"):
                feats_in_code.add(feat)
                feature_counter[feat] += 1
        aggs_in_code: Set[str] = set()
        for kw, agg in AGG_KEYWORDS.items():
            if kw in code:
                aggs_in_code.add(agg)
                agg_counter[agg] += 1
        for feat in feats_in_code:
            for agg in aggs_in_code:
                pair_counter[(feat, agg)] += 1

    # Top features
    top_features = [feat for feat, _ in feature_counter.most_common(top_n)]
    agg_order = ["mean", "std", "var", "min", "max", "trend", "range", "diff", "first", "last", "mad", "rms"]
    aggregators = [a for a in agg_order if a in agg_counter]

    matrix = np.zeros((len(top_features), len(aggregators)), dtype=np.int32)
    for i, feat in enumerate(top_features):
        for j, agg in enumerate(aggregators):
            matrix[i, j] = pair_counter.get((feat, agg), 0)

    return {
        "features": top_features,
        "aggregators": aggregators,
        "matrix": matrix,
        "all_feature_counts": dict(feature_counter.most_common()),
        "all_agg_counts": {a: agg_counter[a] for a in aggregators},
    }


# ------------------------------------------------------------------
# 3) UMAP dimensionality reduction
# ------------------------------------------------------------------
def compute_umap(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 5000,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> "tuple[np.ndarray, np.ndarray] | None":
    """Apply UMAP dimensionality reduction to 2D on the factor matrix.

    X: [N, K] standardized factor matrix
    y: [N] labels
    n_samples: subsample frames (for speed, None=no subsampling)

    Returns (embedding [n, 2], labels [n]) or None if umap not available.
    """
    try:
        import umap  # noqa: F401
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP analysis. Install: pip install umap-learn")
        return None

    N = X.shape[0]
    if n_samples and N > n_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, n_samples, replace=False)
        X = X[idx]
        y = y[idx]
        logger.info(f"[UMAP] Subsampled: {N} -> {n_samples} frames")

    # Remove all-NaN columns
    valid_cols = np.isfinite(X).any(axis=0)
    if not valid_cols.all():
        n_dropped = (~valid_cols).sum()
        X = X[:, valid_cols].copy()
        logger.info(f"[UMAP] Removed {n_dropped} all-NaN columns, {X.shape[1]} columns remaining")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, X.shape[0] - 1),
            min_dist=min_dist,
            random_state=random_state,
            verbose=False,
        )
        embedding = reducer.fit_transform(X)

    return embedding.astype(np.float32), y.astype(np.int64)


# ------------------------------------------------------------------
# 4) Factor-behavior AUC ranking
# ------------------------------------------------------------------
def top_factors_per_behavior(
    auc_matrix: np.ndarray,
    factor_names: list,
    class_ids_sorted: list,
    top_k: int = 20,
) -> dict:
    """Return top-k factor names and AUC for each behavior class.

    Returns dict: class_id -> [{"name": ..., "auc": ...}, ...]
    """
    result = {}
    for j, cid in enumerate(class_ids_sorted):
        col = auc_matrix[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            result[cid] = []
            continue
        indices = np.argsort(col[valid])[::-1][:top_k]
        sorted_names = [factor_names[i] for i in np.where(valid)[0][indices]]
        sorted_aucs = col[valid][indices]
        result[cid] = [
            {"name": name, "auc": round(float(auc), 4)}
            for name, auc in zip(sorted_names, sorted_aucs)
        ]
    return result


# ------------------------------------------------------------------
# 5) Factor grouping (short/medium/long)
# ------------------------------------------------------------------
def group_factors_by_seqlength(factors: list) -> dict:
    """Group factors by seq_length, returns {group_name: [factor, ...]}."""
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
