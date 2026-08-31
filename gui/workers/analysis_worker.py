"""
AnalysisWorker — runs factor analysis in background.

Calls factor_analysis logic: metadata statistics, feature utilization,
factor-behavior AUC heatmap, and optionally UMAP.
"""

import sys
import json
import traceback
from pathlib import Path

from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker
from gui.utils.logging_handler import ModuleLogRedirector


class AnalysisWorker(BaseWorker):
    """Worker that runs factor analysis and visualization generation."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params

    @Slot()
    def run(self):
        redirector = None
        try:
            self.log("=" * 50, 20)
            self.log("Factor Analysis Starting", 20)
            self.log("=" * 50, 20)
            self.set_progress(0, "Loading factors...")

            project_root = Path(__file__).resolve().parent.parent.parent
            sys.path.insert(0, str(project_root))

            params = self._params
            factors_path = params.get("factors_path", "memory/evolved_factors.json")
            viz_dir = params.get("viz_dir", "visualizations/02_factors")
            use_cache = params.get("use_cache", False)
            top_n_features = int(params.get("top_n_features", 40))
            top_k = int(params.get("top_k_per_class", 5))

            # ---- Load factors ----
            if not Path(factors_path).exists():
                self.error.emit(f"Factor file not found: {factors_path}")
                self.finished.emit()
                return

            with open(factors_path, "r", encoding="utf-8") as f:
                factors = json.load(f)
            self.log(f"Loaded {len(factors)} factors", 20)

            # ── Apply label_merge from GUI config ──
            merge_enabled = params.get("label_merge_enabled", True)
            merge_config = params.get("label_merge_config", "")
            merge_name_map = {}
            if merge_enabled and merge_config.strip():
                for rule in merge_config.split(","):
                    rule = rule.strip()
                    if ":" in rule:
                        src, tgt = rule.split(":", 1)
                        merge_name_map[src.strip()] = tgt.strip()
            if merge_name_map:
                for f in factors:
                    old_target = f.get("target", "")
                    if old_target in merge_name_map:
                        f["target"] = merge_name_map[old_target]
                self.log(f"  label_merge: {len(merge_name_map)} rules applied ({merge_config})", 20)

            # ---- Install redirector for progress ----
            redirector = ModuleLogRedirector()
            redirector.log_signal.connect(self._on_worker_log)
            redirector.install()

            # ---- Run metadata analysis ----
            self.set_progress(10, "Extracting factor metadata...")
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

            # Step 1: Metadata
            self.set_progress(20, "Computing metadata statistics...")
            meta = extract_factor_metadata(factors)
            n_factors = meta.get("n_factors", len(factors))
            self.log(f"Factors: {n_factors}, Targets: {meta.get('n_targets', 0)}", 20)

            try:
                plot_factor_overview(meta, viz_dir=viz_dir)
            except Exception as e:
                self.log(f"Warning: overview plot failed: {e}", 30)

            # Step 2: Factor-behavior AUC heatmap
            self.set_progress(50, "Building factor-behavior AUC heatmap...")
            class_ids = meta.get("class_ids_sorted", [])
            auc_matrix = meta.get("auc_matrix", None)

            if auc_matrix is not None and auc_matrix.size > 0:
                try:
                    plot_factor_behavior_heatmap(
                        auc_matrix=auc_matrix,
                        factor_names=meta.get("factor_names", []),
                        class_ids_sorted=class_ids,
                        class_names_readable=class_ids,
                        top_k_per_class=top_k,
                        viz_dir=viz_dir,
                    )
                except Exception as e:
                    self.log(f"Warning: AUC heatmap failed: {e}", 30)

            # Step 3: Feature utilization
            self.set_progress(70, "Building feature utilization matrix...")
            try:
                util = build_feature_utilization_matrix(factors, top_n=top_n_features)
                plot_feature_utilization(
                    features=util["features"],
                    aggregators=util["aggregators"],
                    matrix=util["matrix"],
                    top_n_bar=min(30, len(util.get("all_feature_counts", {}))),
                    viz_dir=viz_dir,
                )
            except Exception as e:
                self.log(f"Warning: feature utilization plot failed: {e}", 30)

            # Step 4: UMAP (if cache available)
            charts = [
                {"name": "Overview", "description": f"Factor metadata overview ({n_factors} factors)"},
                {"name": "Feature Usage", "description": f"Top {top_n_features} features × aggregators"},
                {"name": "Factor-Behavior AUC", "description": "Factor × Behavior AUC heatmap"},
            ]

            if use_cache:
                self.set_progress(85, "Running UMAP dimensionality reduction...")
                cache_root = Path(params.get("cache_path", "pipeline_stage_cache"))
                if cache_root.exists():
                    try:
                        from src.factor_analysis import compute_umap
                        from src.visualization import plot_umap_factor_space

                        # Need to build factor matrix from cache
                        import numpy as np
                        all_X = []
                        all_y = None
                        for gdir in sorted(cache_root.glob("group_*")):
                            xp = gdir / "X_std_train.npy"
                            yp = gdir / "tr_lb.npy"
                            if xp.exists() and yp.exists():
                                all_X.append(np.load(xp))
                                if all_y is None:
                                    all_y = np.load(yp)
                        if all_X:
                            X_all = np.hstack(all_X)
                            umap_result = compute_umap(
                                X_all, all_y,
                                n_samples=int(params.get("umap_samples", 5000)),
                                random_state=42,
                            )
                            if umap_result is not None:
                                embedding, y_sampled = umap_result
                                # Simple plot
                                charts.append({
                                    "name": "UMAP (All)",
                                    "description": f"UMAP factor space ({len(embedding)} points)"
                                })
                                try:
                                    plot_umap_factor_space(
                                        embedding=embedding,
                                        labels=y_sampled,
                                        class_names=[str(i) for i in range(int(y_sampled.max())+1)],
                                        group_embeddings={},
                                        viz_dir=viz_dir,
                                    )
                                except Exception:
                                    pass
                    except Exception as e:
                        self.log(f"UMAP failed: {e}", 30)
                else:
                    self.log("Cache not found, skipping UMAP", 30)

            result = {
                "success": True,
                "n_factors": n_factors,
                "charts": charts,
                "viz_dir": viz_dir,
            }

            self.set_progress(100, "Analysis complete")
            self.result_ready.emit(result)
            self.log("Factor analysis complete.", 20)

        except Exception as e:
            self.log(f"Analysis failed: {e}\n{traceback.format_exc()}", 40)
            self.error.emit(str(e))
        finally:
            if redirector:
                redirector.uninstall()
            self.finished.emit()

    @Slot(str, int)
    def _on_worker_log(self, msg, level):
        self.log_line.emit(msg, level)
