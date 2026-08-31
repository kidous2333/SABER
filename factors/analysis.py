#!/usr/bin/env python3
"""
analysis.py
Factor-level analysis CLI entry point.

All data is derived from the --factors JSON file.
UMAP mode additionally requires pipeline stage cache (--use-cache).
Each visualization chart has a corresponding CSV table output alongside it.

Usage:
  # Metadata-only analysis (all data from factor JSON)
  python analysis.py --factors memory/evolved_factors.json

  # Full analysis (UMAP requires train_behavior.py stage cache)
  python analysis.py --factors memory/evolved_factors.json --use-cache
"""


import sys
from pathlib import Path
# Allow running as a script from a subdirectory (e.g. `python mining/discovery.py`)
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("factor_analysis")


def _load_label_map() -> Tuple[dict, dict, dict]:
    """Load label_map.json and apply label_merge from dataset_config.json.

    Returns (label_map, merge_id_map, merge_name_map):
      - label_map: {name: id}, with merged source classes removed
      - merge_id_map: {source_id: target_id}, label ID remapping table
      - merge_name_map: {source_name: target_name}, factor target name remapping table
    """
    label_map: dict = {}
    for candidate in ["config/label_map.json"]:
        p = Path(candidate)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and any(isinstance(v, int) for v in data.values()):
                    label_map = dict(data)
            except Exception:
                pass

    merge_id_map: dict = {}
    merge_name_map: dict = {}
    if label_map:
        for candidate in ["config/dataset_config.json"]:
            p = Path(candidate)
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        ds_config = json.load(f)
                    label_merge_cfg = ds_config.get("label_merge", {})
                    if label_merge_cfg.get("enabled", False):
                        for group in label_merge_cfg.get("groups", []):
                            target_name = group["target"]
                            target_id = label_map.get(target_name)
                            if target_id is not None:
                                for src_name in group["sources"]:
                                    src_id = label_map.pop(src_name, None)
                                    if src_id is not None:
                                        merge_id_map[src_id] = target_id
                                    merge_name_map[src_name] = target_name
                                    logger.info(f"  label_merge: {src_name}(id={src_id}) -> {target_name}(id={target_id})")
                except Exception:
                    pass

    return label_map, merge_id_map, merge_name_map


def _write_csv(path: Path, rows: list, header: list) -> None:
    """Write CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    logger.info(f"  Table saved: {path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Factor-level analysis: UMAP dimensionality reduction, feature utilization, AUC heatmap, metadata statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--factors", default="memory/valid_factors_deduped.json",
                   help="Factor JSON path (default: memory/evolved_factors.json)")
    p.add_argument("--viz-dir", default="visualizations",
                   help="Visualization output root directory (default: visualizations)")
    p.add_argument("--use-cache", action="store_true",
                   help="Load factor matrix from pipeline stage cache for UMAP computation")
    p.add_argument("--cache-path", default="pipeline_stage_cache",
                   help="Pipeline stage cache directory (default: pipeline_stage_cache)")
    p.add_argument("--umap-samples", type=int, default=5000,
                   help="UMAP downsampling frames (default: 5000)")
    p.add_argument("--top-k-per-class", type=int, default=5,
                   help="Top factors per class in factor-behavior heatmap (default: 20)")
    p.add_argument("--top-n-features", type=int, default=40,
                   help="Top features in feature utilization matrix (default: 40)")
    return p.parse_args()


def main():
    args = parse_args()

    factors_path = Path(args.factors)
    if not factors_path.exists():
        logger.error(f"Factor file does not exist: {factors_path}")
        sys.exit(1)

    with open(factors_path, "r", encoding="utf-8") as f:
        factors = json.load(f)
    logger.info(f"Loaded {len(factors)} factors: {factors_path}")

    # -- Load readable class name mapping (including label_merge) --
    label_map, merge_id_map, merge_name_map = _load_label_map()
    if label_map:
        id_to_name = {str(v): k for k, v in label_map.items()}
        # Merged IDs use target name
        for src_id, tgt_id in merge_id_map.items():
            id_to_name[str(src_id)] = id_to_name.get(str(tgt_id), f"class_{src_id}")
        logger.info(f"Loaded label_map: {len(label_map)} classes (merge: {len(merge_id_map)} entries)")
    else:
        id_to_name = {}
        merge_id_map = {}
        merge_name_map = {}
        logger.info("Label map not found, classes will use numeric IDs")

    # Apply target name remapping (climbsocial -> stand, etc.)
    if merge_name_map:
        for f in factors:
            old_target = f.get("target", "")
            if old_target in merge_name_map:
                f["target"] = merge_name_map[old_target]
        logger.info(f"Applied target remapping: {merge_name_map}")

    # ================================================================
    # Metadata analysis (always runs, all data from --factors JSON)
    # ================================================================
    from src.factor_analysis import (
        extract_factor_metadata,
        build_feature_utilization_matrix,
        top_factors_per_behavior,
        group_factors_by_seqlength,
    )
    from src.visualization import (
        plot_factor_overview,
        plot_factor_behavior_heatmap,
        plot_feature_utilization,
    )

    viz_dir = args.viz_dir
    out_dir_02 = Path(viz_dir) / "02_factors"

    # Step 1: Factor metadata statistics -> factor_overview.png + .csv
    logger.info("=" * 60)
    logger.info("Step 1/3: Factor metadata statistics...")
    meta = extract_factor_metadata(factors)
    if meta["auc_distribution"]:
        logger.info(
            f"  Total factors: {meta['n_factors']}, target behaviors: {meta['n_targets']}, "
            f"seq_lengths: {list(meta['seq_length_distribution'].keys())}, "
            f"AUC range: [{min(meta['auc_distribution']):.3f}, {max(meta['auc_distribution']):.3f}]"
        )
    else:
        logger.info(
            f"  Total factors: {meta['n_factors']}, target behaviors: {meta['n_targets']}, AUC: N/A"
        )

    plot_factor_overview(meta, viz_dir=viz_dir)

    # -- factor_overview.csv: raw data for each factor (drives 2x2 panel chart) --
    overview_rows = []
    for i, f in enumerate(factors):
        code = f.get("code", "")
        n_feats = len(set(
            m.group(1) for m in re.finditer(r"idx\.get\('([^']+)'", code)
            if "%" not in m.group(1) and not m.group(1).endswith("_")
        ))
        evo_meta = f.get("_evolution_meta", {})
        overview_rows.append([
            f.get("name", f"factor_{i}"),
            str(f.get("target", "")),
            str(f.get("seq_length", "")),
            f"{f['best_auc']:.4f}" if f.get("best_auc") is not None else "",
            str(n_feats),
            str(evo_meta.get("tree_size", "")),
        ])
    _write_csv(out_dir_02 / "factor_overview.csv", overview_rows,
               ["factor_name", "target", "seq_length", "best_auc", "n_features", "tree_size"])

    # ---- Factor group statistics ----
    groups = group_factors_by_seqlength(factors)
    group_rows = [["group", "n_factors", "n_with_auc", "mean_auc", "max_auc"]]
    for gname, gfactors in groups.items():
        g_aucs = [f.get("best_auc") for f in gfactors if f.get("best_auc") is not None]
        if g_aucs:
            logger.info(
                f"  Group '{gname}': {len(gfactors)} factors, "
                f"with AUC: {len(g_aucs)}, mean AUC={np.mean(g_aucs):.4f}, max AUC={max(g_aucs):.4f}"
            )
            group_rows.append([gname, str(len(gfactors)), str(len(g_aucs)),
                               f"{np.mean(g_aucs):.4f}", f"{max(g_aucs):.4f}"])
        else:
            logger.info(f"  Group '{gname}': {len(gfactors)} factors (no AUC data)")
            group_rows.append([gname, str(len(gfactors)), "0", "N/A", "N/A"])
    _write_csv(out_dir_02 / "group_summary.csv", group_rows, group_rows[0])

    # Step 2: Factor-behavior AUC association analysis
    logger.info("Step 2/3: Factor-behavior AUC association analysis...")
    class_ids_sorted = meta["class_ids_sorted"]
    auc_matrix = meta["auc_matrix"]

    # Apply label_merge to AUC matrix: merge source class columns into target class column
    if merge_id_map and auc_matrix.size > 0:
        for src_id, tgt_id in merge_id_map.items():
            src_str = str(src_id)
            tgt_str = str(tgt_id)
            if src_str in class_ids_sorted and tgt_str in class_ids_sorted:
                si = class_ids_sorted.index(src_str)
                ti = class_ids_sorted.index(tgt_str)
                # Take row-wise max for merging (handles NaN)
                merged_col = np.fmax(auc_matrix[:, ti], auc_matrix[:, si])
                auc_matrix[:, ti] = merged_col
                logger.info(f"  Merged AUC column: class {src_str} -> class {tgt_str}")
        # Remove merged source class columns
        keep_mask = np.ones(len(class_ids_sorted), dtype=bool)
        for src_id in merge_id_map:
            src_str = str(src_id)
            if src_str in class_ids_sorted:
                keep_mask[class_ids_sorted.index(src_str)] = False
        class_ids_sorted = [cid for i, cid in enumerate(class_ids_sorted) if keep_mask[i]]
        auc_matrix = auc_matrix[:, keep_mask]
        meta["class_ids_sorted"] = class_ids_sorted
        meta["auc_matrix"] = auc_matrix

    class_names_readable = [id_to_name.get(cid, f"class_{cid}") for cid in class_ids_sorted]

    if meta["auc_matrix"].size > 0 and np.isfinite(meta["auc_matrix"]).any():
        plot_factor_behavior_heatmap(
            auc_matrix=meta["auc_matrix"],
            factor_names=meta["factor_names"],
            class_ids_sorted=class_ids_sorted,
            class_names_readable=class_names_readable,
            top_k_per_class=args.top_k_per_class,
            viz_dir=viz_dir,
        )

        # -- factor_behavior_auc_heatmap.csv: full AUC matrix --
        auc_csv_rows = []
        for i, fname in enumerate(meta["factor_names"]):
            row = [fname]
            for j in range(len(class_ids_sorted)):
                v = meta["auc_matrix"][i, j]
                row.append(f"{v:.4f}" if np.isfinite(v) else "")
            auc_csv_rows.append(row)
        _write_csv(out_dir_02 / "factor_behavior_auc_heatmap.csv",
                   auc_csv_rows, ["factor_name"] + class_names_readable)

        # -- per_behavior_top_factors.csv --
        top_per_behavior = top_factors_per_behavior(
            meta["auc_matrix"], meta["factor_names"], class_ids_sorted, top_k=args.top_k_per_class,
        )
        top_rows = []
        for cid in class_ids_sorted:
            name = id_to_name.get(cid, f"class_{cid}")
            entries = top_per_behavior.get(cid, [])
            for rank, entry in enumerate(entries, 1):
                top_rows.append([name, str(rank), entry["name"], f"{entry['auc']:.4f}"])
        _write_csv(out_dir_02 / "per_behavior_top_factors.csv",
                   top_rows, ["behavior_class", "rank", "factor_name", "auc"])
    else:
        logger.warning("  No per_class AUC data, skipping factor-behavior heatmap.")

    # Step 3: Feature utilization analysis
    logger.info("Step 3/3: Feature utilization analysis...")
    util = build_feature_utilization_matrix(factors, top_n=args.top_n_features)
    logger.info(f"  Displaying top-{len(util['features'])} features x {len(util['aggregators'])} aggregators")

    plot_feature_utilization(
        features=util["features"],
        aggregators=util["aggregators"],
        matrix=util["matrix"],
        top_n_bar=min(30, len(util["all_feature_counts"])),
        viz_dir=viz_dir,
    )

    # -- feature_utilization_heatmap.csv --
    util_rows = []
    for i, feat in enumerate(util["features"]):
        util_rows.append([feat] + [str(util["matrix"][i, j]) for j in range(len(util["aggregators"]))])
    _write_csv(out_dir_02 / "feature_utilization_heatmap.csv",
               util_rows, ["feature"] + util["aggregators"])

    # -- feature_usage_ranking.csv: full feature ranking --
    rank_rows = []
    for rank, (feat, cnt) in enumerate(util["all_feature_counts"].items(), 1):
        rank_rows.append([str(rank), feat, str(cnt)])
    _write_csv(out_dir_02 / "feature_usage_ranking.csv",
               rank_rows, ["rank", "feature", "usage_count"])

    # Top-10 logging
    top_feats = list(util["all_feature_counts"].items())[:10]
    logger.info("  Top-10 most used features:")
    for feat, cnt in top_feats:
        logger.info(f"    {feat}: {cnt} factors")

    # ================================================================
    # Mode 2: UMAP dimensionality reduction (requires cache data)
    # ================================================================
    if args.use_cache:
        logger.info("=" * 60)
        logger.info("Step 4: UMAP factor space dimensionality reduction...")

        from src.factor_analysis import compute_umap
        from src.visualization import plot_umap_factor_space

        cache_root = Path(args.cache_path)
        all_X_parts = []
        all_y = None
        for group_dir in sorted(cache_root.glob("group_*")):
            X_path = group_dir / "X_std_train.npy"
            y_path = group_dir / "tr_lb.npy"
            if X_path.exists() and y_path.exists():
                X_part = np.load(X_path).astype(np.float32)
                y_part = np.load(y_path).astype(np.int64)
                all_X_parts.append(X_part)
                if all_y is None:
                    all_y = y_part
                logger.info(f"  Loaded {group_dir.name}: X={X_part.shape}")

        if not all_X_parts:
            logger.error("Pipeline stage cache not found. Please run train_behavior.py first.")
            logger.info("Falling back: skip UMAP analysis.")
        else:
            X_all = np.hstack(all_X_parts)
            y_all = all_y
            # Apply label_merge remapping
            if merge_id_map:
                y_all = np.array([merge_id_map.get(int(v), int(v)) for v in y_all])
                logger.info(f"  Applied label_merge remapping to labels")
            logger.info(f"  Concatenated factor matrix: {X_all.shape}")

            embedding_result = compute_umap(
                X_all, y_all, n_samples=args.umap_samples, random_state=42,
            )

            group_embeddings: dict = {}
            group_name_map = {0: "short", 1: "medium", 2: "long"}
            for gi, X_part in enumerate(all_X_parts):
                gname = group_name_map.get(gi, f"group_{gi}")
                logger.info(f"  Computing {gname} group UMAP (X={X_part.shape})...")
                g_result = compute_umap(
                    X_part, y_all, n_samples=args.umap_samples, random_state=42,
                )
                if g_result is not None:
                    group_embeddings[gname] = g_result

            out_dir_umap = Path(viz_dir) / "02_factors_umap"

            if embedding_result is not None:
                embedding, y_sampled = embedding_result

                # UMAP class_names: position i must correspond to label value i
                umap_max_label = int(np.max(y_sampled))
                umap_class_names = [id_to_name.get(str(i), f"class_{i}") for i in range(umap_max_label + 1)]

                plot_umap_factor_space(
                    embedding=embedding,
                    labels=y_sampled,
                    class_names=umap_class_names,
                    group_embeddings=group_embeddings,
                    viz_dir=viz_dir,
                )

                # -- umap_by_behavior.csv --
                umap_rows = []
                for i in range(len(embedding)):
                    lid = str(y_sampled[i])
                    lname = id_to_name.get(lid, f"class_{lid}")
                    umap_rows.append([f"{embedding[i, 0]:.6f}", f"{embedding[i, 1]:.6f}", lid, lname])
                _write_csv(out_dir_umap / "umap_by_behavior.csv",
                           umap_rows, ["umap_1", "umap_2", "label_id", "label_name"])

                # -- umap_by_group.csv: merged UMAP coordinates for all groups --
                all_group_rows = []
                for gname, (g_emb, g_lbl) in group_embeddings.items():
                    g_lbl = np.asarray(g_lbl).astype(int)
                    for i in range(len(g_emb)):
                        lid = str(g_lbl[i])
                        lname = id_to_name.get(lid, f"class_{lid}")
                        all_group_rows.append([gname, f"{g_emb[i, 0]:.6f}", f"{g_emb[i, 1]:.6f}", lid, lname])
                _write_csv(out_dir_umap / "umap_by_group.csv",
                           all_group_rows, ["group", "umap_1", "umap_2", "label_id", "label_name"])

                logger.info("  UMAP visualization complete.")
            else:
                logger.warning("  UMAP computation failed, skipping.")

    # ================================================================
    # Complete
    # ================================================================
    logger.info("=" * 60)
    logger.info(f"Factor analysis complete. All output saved to: {Path(viz_dir).resolve()}")
    logger.info("Output charts + corresponding tables:")
    for pattern in [
        "02_factors/factor_overview.png",      "02_factors/factor_overview.csv",
        "02_factors/group_summary.csv",
        "02_factors/factor_behavior_auc_heatmap.png", "02_factors/factor_behavior_auc_heatmap.csv",
        "02_factors/per_behavior_top_factors.csv",
        "02_factors/feature_utilization_heatmap.png", "02_factors/feature_utilization_heatmap.csv",
        "02_factors/feature_usage_ranking.png", "02_factors/feature_usage_ranking.csv",
        "02_factors_umap/umap_by_behavior.png", "02_factors_umap/umap_by_behavior.csv",
        "02_factors_umap/umap_by_group.png",    "02_factors_umap/umap_by_group.csv",
    ]:
        p = Path(viz_dir) / pattern
        tag = "[OK]" if p.exists() else "[--]"
        logger.info(f"  {tag} {pattern}")


if __name__ == "__main__":
    main()
