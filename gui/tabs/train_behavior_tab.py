"""TrainBehaviorTab — Multi-resolution 3-stage synthetic validation."""
import json
import logging
from pathlib import Path
from typing import Optional
from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QPushButton, QDialog
from PySide6.QtCore import Qt, Slot, QTimer
from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.widgets.matplotlib_widget import MplWidget
from gui.workers.train_behavior_worker import TrainBehaviorWorker

logger = logging.getLogger("gui.train_behavior")

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "train_behavior_gui.json"


class ImageViewer(QDialog):
    """Clickable, zoomable, draggable image viewer."""
    def __init__(self, img_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Path(img_path).name)
        self.resize(1100, 800)
        self.setStyleSheet("QDialog{background:#222;}")

        from PySide6.QtWidgets import QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
        from PySide6.QtGui import QPixmap
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view = QGraphicsView()
        self._view.setStyleSheet("QGraphicsView{background:#222;border:none;}")
        self._view.setRenderHints(self._view.renderHints()
                                  | self._view.renderHints().Antialiasing
                                  | self._view.renderHints().SmoothPixmapTransform)
        self._view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self._scene = QGraphicsScene()
        pixmap = QPixmap(img_path)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._view.setScene(self._scene)
        self._view.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

        layout.addWidget(self._view)

        # Close button
        from PySide6.QtWidgets import QPushButton, QHBoxLayout
        bar = QHBoxLayout()
        bar.addStretch()
        close_btn = QPushButton("✕ Close")
        close_btn.setFixedSize(80, 28)
        close_btn.setStyleSheet(
            "QPushButton{background:#DC2626;color:#FFF;border:none;border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#B91C1C;}")
        close_btn.clicked.connect(self.close)
        bar.addWidget(close_btn)
        bar.setContentsMargins(8, 4, 8, 8)
        layout.addLayout(bar)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._view.scale(factor, factor)


class TrainBehaviorTab(BaseTab):
    def __init__(self, parent=None):
        self._metrics_table = None
        self._poll_run_dir = None
        self._seen_images = set()  # Track already-displayed images
        self._auto_update = True   # Auto-follow latest images; set False on user interaction
        super().__init__(title="3-Stage Behavior Prediction", tab_key="train_behavior", parent=parent)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(3000)  # Poll every 3 seconds
        self._poll_timer.timeout.connect(self._poll_visualizations)

    def setup_params(self):
        # 1. Performance — first
        self.add_config_group(ParameterGroup("Performance", [
            {"key": "num_workers", "label": "Workers", "type": "int", "default": 8, "min": 1, "max": 128},
            {"key": "num_chunks", "label": "Chunks", "type": "int", "default": 256, "min": 1, "max": 1024},
            {"key": "use_gpu", "label": "GPU (LGBM)", "type": "checkbox", "default": False},
            {"key": "use_cache", "label": "Factor Cache", "type": "checkbox", "default": True},
            {"key": "refresh_cache", "label": "Refresh Cache", "type": "checkbox", "default": False},
            {"key": "use_stage_cache", "label": "Stage Cache", "type": "checkbox", "default": True},
            {"key": "refresh_stage_cache", "label": "Refresh Stage", "type": "checkbox", "default": False},
            {"key": "purity_mode", "label": "Purity Mode", "type": "combo", "default": "nan_boundary",
             "options": ["nan_boundary", "none", "strict"]},
            {"key": "val_purity_mode", "label": "Val Purity Mode", "type": "combo", "default": "null",
             "options": ["null", "nan_boundary", "none", "strict"]},
            {"key": "save_cache", "label": "Save Cache (.pt)", "type": "file", "default": ""},
            {"key": "force_cache_raw", "label": "Force Cache (raw)", "type": "file", "default": ""},
        ]))
        # 2. Mode & Factors — merged
        self.add_config_group(ParameterGroup("Mode & Factors", [
            {"key": "mode", "label": "Mode", "type": "combo", "default": "multiresolution", "options": ["multiresolution", "standard"]},
            {"key": "do_temporal", "label": "Temporal", "type": "checkbox", "default": True},
            {"key": "do_ovr", "label": "OvR", "type": "checkbox", "default": True},
            {"key": "temporal_window", "label": "Temp Window", "type": "int", "default": 31, "min": 5, "max": 120},
            {"key": "factors_path", "label": "Factors JSON", "type": "file", "default": "memory/valid_factors.json"},
            {"key": "max_samples", "label": "Max Samples", "type": "int", "default": 0, "min": 0, "max": 999999},
            {"key": "filter_corr", "label": "Filter Corr JSON", "type": "file", "default": ""},
            {"key": "corr_threshold", "label": "Corr Threshold", "type": "float", "default": 0.85, "min": 0.5, "max": 1.0, "decimals": 2},
        ]))
        # 3. LGBM
        self.add_config_group(ParameterGroup("LGBM", [
            {"key": "lgbm_estimators", "label": "n_estimators", "type": "int", "default": 500, "min": 10, "max": 5000},
            {"key": "lgbm_depth", "label": "max_depth", "type": "int", "default": 6, "min": 2, "max": 16},
            {"key": "lgbm_lr", "label": "learning_rate", "type": "float", "default": 0.05, "min": 0.001, "max": 1.0, "step": 0.01},
            {"key": "num_leaves", "label": "num_leaves", "type": "int", "default": 63, "min": 8, "max": 512},
            {"key": "early_stopping", "label": "Early Stop Rounds", "type": "int", "default": 50, "min": 0, "max": 500},
            {"key": "class_weight", "label": "Class Weight", "type": "combo", "default": "balanced", "options": ["balanced", "null"]},
        ]))
        # 4. Synth Validation
        self.add_config_group(ParameterGroup("Synth Validation", [
            {"key": "nan_fill", "label": "NaN Fill", "type": "combo", "default": "mean", "options": ["mean", "zero"]},
            {"key": "normalize_cm", "label": "Normalize CM", "type": "checkbox", "default": True},
            {"key": "top_k", "label": "Top-K Accuracy", "type": "int", "default": 3, "min": 0, "max": 20},
            {"key": "feature_selection_top_k", "label": "Feature Sel Top-K", "type": "int", "default": 0, "min": 0, "max": 500},
            {"key": "top_k_short", "label": "Top-K (short)", "type": "int", "default": 0, "min": 0, "max": 500},
            {"key": "top_k_medium", "label": "Top-K (medium)", "type": "int", "default": 0, "min": 0, "max": 500},
            {"key": "top_k_long", "label": "Top-K (long)", "type": "int", "default": 0, "min": 0, "max": 500},
            {"key": "use_focal_loss", "label": "Focal Loss (LGBM)", "type": "checkbox", "default": True},
            {"key": "focal_alpha", "label": "Focal Alpha", "type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "decimals": 2},
            {"key": "focal_gamma", "label": "Focal Gamma", "type": "float", "default": 2.0, "min": 0.0, "max": 5.0, "decimals": 2},
            {"key": "residual_stage1", "label": "Residual Stage1", "type": "checkbox", "default": True},
            {"key": "residual_stage2", "label": "Residual Stage2", "type": "checkbox", "default": True},
        ]))
        # 5. Temporal Model
        self.add_config_group(ParameterGroup("Temporal Model", [
            {"key": "temporal_model", "label": "Model", "type": "combo", "default": "bilstm",
             "options": ["lgbm", "bilstm", "transformer", "mamba"]},
            {"key": "monitor_metric", "label": "Monitor Metric", "type": "combo", "default": "val_balanced_acc",
             "options": ["val_acc", "val_balanced_acc", "macro_f1", "weighted_f1", "macro_auc", "weighted_auc"]},
            {"key": "hidden_dim", "label": "Hidden Dim", "type": "int", "default": 512, "min": 32, "max": 2048},
            {"key": "num_layers", "label": "Num Layers", "type": "int", "default": 3, "min": 1, "max": 8},
            {"key": "dropout", "label": "Dropout", "type": "float", "default": 0.3, "min": 0.0, "max": 0.9, "decimals": 2},
            {"key": "epochs", "label": "Epochs", "type": "int", "default": 200, "min": 10, "max": 1000},
            {"key": "lr", "label": "Learning Rate", "type": "float", "default": 0.005, "min": 0.0001, "max": 0.1, "step": 0.001},
            {"key": "weight_decay", "label": "Weight Decay", "type": "float", "default": 0.0005, "min": 0.0, "max": 0.01, "step": 0.0001},
            {"key": "batch_size", "label": "Batch Size", "type": "int", "default": 32, "min": 4, "max": 256},
            {"key": "label_smoothing", "label": "Label Smoothing", "type": "float", "default": 0.1, "min": 0.0, "max": 0.5, "decimals": 2},
            {"key": "early_stopping_patience", "label": "ES Patience", "type": "int", "default": 20, "min": 1, "max": 100},
        ]))
        # 6. Temporal Features
        self.add_config_group(ParameterGroup("Temporal Features", [
            {"key": "t_use_focal_loss", "label": "Focal Loss", "type": "checkbox", "default": True},
            {"key": "t_focal_alpha", "label": "Focal Alpha", "type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "decimals": 2},
            {"key": "t_focal_gamma", "label": "Focal Gamma", "type": "float", "default": 2.0, "min": 0.0, "max": 5.0, "decimals": 2},
            {"key": "use_class_weights", "label": "Class Weights", "type": "checkbox", "default": True},
            {"key": "seq_output_residual", "label": "Output Residual", "type": "checkbox", "default": True},
            {"key": "seq_use_extra_features", "label": "Extra Features", "type": "checkbox", "default": True},
            {"key": "enhanced_features", "label": "Enhanced Features", "type": "checkbox", "default": True},
            {"key": "autocorr_features", "label": "Autocorr Features", "type": "checkbox", "default": True},
            {"key": "cross_scale_features", "label": "Cross-Scale Features", "type": "checkbox", "default": True},
            {"key": "distribution_shape", "label": "Distribution Shape", "type": "checkbox", "default": True},
            {"key": "multi_scale_windows", "label": "Multi-Scale Windows", "type": "text", "default": "7,15,31,63"},
            {"key": "use_cosine_warmup", "label": "Cosine Warmup", "type": "checkbox", "default": True},
            {"key": "warmup_epochs", "label": "Warmup Epochs", "type": "int", "default": 10, "min": 0, "max": 100},
            {"key": "use_balanced_sampling", "label": "Balanced Sampling", "type": "checkbox", "default": True},
            {"key": "use_layer_norm", "label": "Layer Norm", "type": "checkbox", "default": True},
            {"key": "use_amp", "label": "AMP (FP16)", "type": "checkbox", "default": True},
            {"key": "device", "label": "Device", "type": "combo", "default": "cuda", "options": ["cuda", "cpu"]},
        ]))
        # 7. Decoder & Visualization — merged
        self.add_config_group(ParameterGroup("Decoder & Viz", [
            {"key": "rule_correct_enabled", "label": "Rule Corrector", "type": "checkbox", "default": True},
            {"key": "benchmark", "label": "Benchmark", "type": "checkbox", "default": False},
            {"key": "visualize_raster", "label": "Raster Plot", "type": "checkbox", "default": True},
            {"key": "raster_frames", "label": "Raster Frames", "type": "int", "default": 5000, "min": 100, "max": 50000},
        ]))
        # 8. Sequence & Preprocessing — merged
        self.add_config_group(ParameterGroup("Sequence & Preprocess", [
            {"key": "seq_length", "label": "Seq Length", "type": "int", "default": 1, "min": 1, "max": 120},
            {"key": "stride", "label": "Stride", "type": "int", "default": 1, "min": 1, "max": 60},
            {"key": "frame_interval", "label": "Frame Interval", "type": "int", "default": 1, "min": 1, "max": 30},
            {"key": "purity_threshold", "label": "Purity Thresh", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "decimals": 2},
            {"key": "boundary_margin", "label": "Boundary Margin", "type": "int", "default": 10, "min": 0, "max": 120},
            {"key": "scale_normalize", "label": "Scale Normalize", "type": "checkbox", "default": False},
            {"key": "rotation_align", "label": "Rotation Align", "type": "checkbox", "default": False},
            {"key": "bidirectional_impute", "label": "Bidirectional Impute", "type": "checkbox", "default": True},
            {"key": "soft_boundary", "label": "Soft Boundary", "type": "checkbox", "default": True},
        ]))
        # 9. Data
        self.add_config_group(ParameterGroup("Data", [
            {"key": "train_mouse_dirs", "label": "Train Mouse KP Dirs", "type": "path_list", "default": [""]},
            {"key": "train_tail_dirs", "label": "Train Tail KP Dirs", "type": "path_list", "default": [""]},
            {"key": "train_behavior_m1_files", "label": "Train Behavior M1 Files", "type": "path_list_file", "default": [""]},
            {"key": "train_behavior_m2_files", "label": "Train Behavior M2 Files", "type": "path_list_file", "default": [""]},
            {"key": "val_mouse_dirs", "label": "Val Mouse KP Dirs", "type": "path_list", "default": [""]},
            {"key": "val_tail_dirs", "label": "Val Tail KP Dirs", "type": "path_list", "default": [""]},
            {"key": "val_behavior_m1_files", "label": "Val Behavior M1 Files", "type": "path_list_file", "default": [""]},
            {"key": "val_behavior_m2_files", "label": "Val Behavior M2 Files", "type": "path_list_file", "default": [""]},
            {"key": "max_instances", "label": "Max Instances", "type": "int", "default": 2, "min": 1, "max": 10},
        ]))
        # 10. Labels — last
        self.add_config_group(ParameterGroup("Labels", [
            {"key": "label_merge_enabled", "label": "Enable Label Merge", "type": "checkbox",
             "default": True,
             "hint": "Merge source classes into target (e.g. climbsocial→stand)"},
            {"key": "label_merge_config", "label": "Merge Rules", "type": "text",
             "default": "climbsocial:stand",
             "hint": "Comma-separated pairs: src1:tgt1, src2:tgt2"},
        ]))
        from gui.widgets.key_value_editor import KeyValueEditor
        self._label_map_editor = KeyValueEditor(title="Label Map")
        self._label_map_editor.set_pairs({
            "explore_object": "0", "climb": "1", "self_grooming": "2",
            "stand": "3", "blank": "4", "positive_sniffs": "5", "approach": "6",
        })
        self._label_map_editor.setMaximumHeight(200)
        self._param_layout.addWidget(self._label_map_editor)

    def setup_results(self):
        # Compact metrics table
        self._metrics_table = QTableWidget(0, 6)
        self._metrics_table.setHorizontalHeaderLabels(
            ["Stage", "Accuracy", "macro-F1", "macro-AUC", "weighted-F1", "bal.Acc"]
        )
        self._metrics_table.horizontalHeader().setStretchLastSection(False)
        self._metrics_table.setFixedHeight(148)  # Header + exactly 5 rows
        self._metrics_table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)
        self._metrics_table.setStyleSheet("""
            QTableWidget { font-size: 11px; gridline-color: #E0E0E0; }
            QHeaderView::section { font-size: 10px; font-weight: 600; padding: 2px 4px; }
        """)

        # ── Top: image viewer (fixed height, no scrollbars) ──
        from PySide6.QtWidgets import QFrame, QVBoxLayout, QScrollArea
        self._image_frame = QFrame()
        self._image_frame.setMinimumHeight(400)
        self._image_frame.setStyleSheet("QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:4px;}")
        img_layout = QVBoxLayout(self._image_frame)
        img_layout.setContentsMargins(0, 0, 0, 0)
        self._image_label = QLabel("Run a validation to see visualizations")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet("color:#999;font-size:13px;padding:20px;border:none;")
        img_layout.addWidget(self._image_label)
        self._results_layout_main.addWidget(self._image_frame, 1)

        # ── Bottom row: metrics table + side panel ──
        from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
        bottom_row = QWidget()
        bottom_row_layout = QHBoxLayout(bottom_row)
        bottom_row_layout.setContentsMargins(0, 6, 0, 0)
        bottom_row_layout.setSpacing(6)
        bottom_row_layout.addWidget(self._metrics_table, 0)  # No stretch, fixed height

        # Side panel (fixed width, scrollable group buttons)
        self._side_panel = QWidget()
        self._side_panel.setMinimumWidth(160)
        self._side_panel.setStyleSheet("background:#FAFAFA; border:1px solid #E5E5E5; border-radius:4px;")
        side_layout = QVBoxLayout(self._side_panel)
        side_layout.setContentsMargins(6, 6, 6, 6)
        side_layout.setSpacing(3)

        self._group_label = QLabel("Image Groups")
        self._group_label.setStyleSheet("color:#888;font-size:9px;font-weight:700;border:none;padding:0 2px;")
        side_layout.addWidget(self._group_label)

        # Scrollable group button list
        self._group_scroll = QScrollArea()
        self._group_scroll.setWidgetResizable(True)
        self._group_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._group_scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        self._group_btns_widget = QWidget()
        self._group_btns_layout = QVBoxLayout(self._group_btns_widget)
        self._group_btns_layout.setContentsMargins(0, 0, 0, 0)
        self._group_btns_layout.setSpacing(2)
        self._group_btns_layout.addStretch()
        self._group_scroll.setWidget(self._group_btns_widget)
        side_layout.addWidget(self._group_scroll, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("border:none;background:#E0E0E0;max-height:1px;")
        side_layout.addWidget(sep)

        self._page_label = QLabel("—")
        self._page_label.setAlignment(Qt.AlignCenter)
        self._page_label.setStyleSheet("color:#555;font-size:10px;border:none;padding:2px;")
        side_layout.addWidget(self._page_label)

        flip_row = QHBoxLayout()
        flip_row.setSpacing(4)
        flip_btn_style = (
            "QPushButton{font-size:11px;padding:2px 8px;"
            "background:#E8E8E8;color:#333;border:1px solid #CCC;border-radius:3px;}"
            "QPushButton:hover{background:#D0D0D0;}"
            "QPushButton:disabled{color:#BBB;}"
        )
        self._prev_btn = QPushButton("◀ Prev")
        self._prev_btn.setStyleSheet(flip_btn_style)
        self._prev_btn.clicked.connect(self._flip_prev)
        flip_row.addWidget(self._prev_btn)

        self._next_btn = QPushButton("Next ▶")
        self._next_btn.setStyleSheet(flip_btn_style)
        self._next_btn.clicked.connect(self._flip_next)
        flip_row.addWidget(self._next_btn)
        side_layout.addLayout(flip_row)

        side_layout.addStretch()
        bottom_row_layout.addWidget(self._side_panel, 1)
        self._results_layout_main.addWidget(bottom_row)

        # State
        self._image_cache = {}       # {group: [paths]}
        self._active_group = None    # currently selected group
        self._active_page = 0        # current page within group (2 images/page)

        # Ensure log panel has usable height (at least 100px)
        self._log_frame.setMinimumHeight(100)

    # ---- Config persistence ----
    def _load_settings(self):
        if not CONFIG_PATH.exists():
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
        except Exception:
            return
        for group in self._config_groups:
            for item in group._schema:
                key = item["key"]
                if key in saved:
                    group.set_value(key, saved[key])
        if hasattr(self, '_label_map_editor') and "label_map_pairs" in saved:
            self._label_map_editor.set_pairs(saved["label_map_pairs"])

    def _save_settings(self):
        values = self.gather_params()
        if hasattr(self, '_label_map_editor'):
            values["label_map_pairs"] = self._label_map_editor.get_pairs()
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_active_group') and self._active_group \
                and self._active_group in self._image_cache \
                and self._image_cache[self._active_group]:
            # Throttle: only re-render at end of resize
            if not hasattr(self, '_resize_timer'):
                from PySide6.QtCore import QTimer
                self._resize_timer = QTimer(self)
                self._resize_timer.setSingleShot(True)
                self._resize_timer.setInterval(150)
                self._resize_timer.timeout.connect(self._show_current_image)
            self._resize_timer.start()

    def on_start(self):
        self._save_settings()
        self._seen_images.clear()
        self._poll_run_dir = None
        self._auto_update = True  # Re-enable auto-follow for new run
        self.run_worker(TrainBehaviorWorker, params=self.gather_params())

    def on_partial_result(self, data):
        """Handle intermediate results from worker."""
        if not isinstance(data, dict):
            return
        if data.get("type") == "run_dir":
            self._poll_run_dir = data["run_dir"]
            self._poll_timer.start()
        elif data.get("type") == "stage_metric":
            # Add or update a row in the metrics table
            name = data.get("name", "?")
            col_map = {"accuracy": 1, "macro_f1": 2, "macro_auc": 3, "weighted_f1": 4, "balanced_acc": 5}
            # Find existing row or add new
            row = -1
            for r in range(self._metrics_table.rowCount()):
                if self._metrics_table.item(r, 0).text() == name:
                    row = r
                    break
            if row < 0:
                row = self._metrics_table.rowCount()
                self._metrics_table.insertRow(row)
                self._metrics_table.setItem(row, 0, QTableWidgetItem(name))
            for key, col in col_map.items():
                val = data.get(key)
                if val is not None:
                    item = QTableWidgetItem(f"{val:.4f}")
                    item.setTextAlignment(Qt.AlignCenter)
                    self._metrics_table.setItem(row, col, item)
            self._metrics_table.resizeRowsToContents()

    def _scan_images(self, base_dir: Path, top_per_group: int = 4) -> dict:
        """Scan run directory for PNG images, grouped by subdirectory.

        Returns {group_label: [path, ...]} with each group limited to top_per_group
        images (sorted by modification time, newest first).
        """
        groups = {}
        for png in sorted(base_dir.rglob("*.png")):
            try:
                mtime = png.stat().st_mtime
            except OSError:
                mtime = 0.0
            rel = png.relative_to(base_dir)
            parts = rel.parts
            if len(parts) == 1:
                name = png.stem
                if "synth_" in name:
                    group = "Meta CM"
                elif "temporal_" in name:
                    group = "Temporal CM"
                else:
                    group = "Results"
            else:
                # Keep last 1-2 dir levels as group name (drop "visualizations/" prefix for brevity)
                group = "/".join(parts[:-1])
                # Simplify: "visualizations/05_meta" → "05_meta"
                if group.startswith("visualizations/"):
                    group = group.split("/", 1)[1]
            groups.setdefault(group, []).append((mtime, str(png)))

        # Keep only top N per group (newest first), drop mtime tuple
        limited = {}
        for group, items in groups.items():
            items.sort(key=lambda x: x[0], reverse=True)
            limited[group] = [p for _, p in items[:top_per_group]]
        return limited

    def _show_current_image(self):
        """Display up to 2 images side-by-side for the current page."""
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QHBoxLayout, QWidget
        paths = self._image_cache.get(self._active_group, []) if self._active_group else []
        # Show 2 per page only for confusion matrix groups (raw + normalized pairs)
        per_page = 2 if (len(paths) == 2 and self._active_group
                         and ("CM" in self._active_group or "cm" in self._active_group)) else 1
        total_pages = max(1, (len(paths) + per_page - 1) // per_page) if paths else 0
        page = min(self._active_page, total_pages - 1) if total_pages > 0 else 0
        start = page * per_page
        end = min(start + per_page, len(paths))
        page_paths = paths[start:end] if paths else []

        # Build container with 1-2 images side by side
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 10, 8, 10)
        layout.setSpacing(8)

        if page_paths:
            avail_w = self._image_frame.contentsRect().width()
            avail_h = self._image_frame.contentsRect().height()
            n = len(page_paths)
            margin = 36  # extra padding to avoid clipping axis labels
            for img_path in page_paths:
                p = Path(img_path)
                if not p.is_absolute() and self._poll_run_dir:
                    p = Path(self._poll_run_dir) / p
                    img_path = str(p)
                lbl = QLabel()
                lbl.setCursor(Qt.PointingHandCursor)
                lbl.setToolTip("Click to enlarge")
                lbl.mousePressEvent = lambda ev, ip=img_path: ImageViewer(ip, self).exec()
                pixmap = QPixmap(img_path)
                if not pixmap.isNull():
                    w = int(max(avail_w / n - margin, 1) * 2)
                    h = int(max(avail_h - margin, 1) * 2)
                    scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    scaled.setDevicePixelRatio(2.0)
                    lbl.setPixmap(scaled)
                    lbl.setAlignment(Qt.AlignCenter)
                    lbl.setStyleSheet("border:none;background:#FFF;padding:4px;")
                else:
                    lbl.setText(f"Cannot load:\n{img_path}")
                    lbl.setStyleSheet("color:#999;font-size:10px;border:none;")
                    lbl.setWordWrap(True)
                layout.addWidget(lbl)
        else:
            placeholder = QLabel("No images for this group")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("color:#999;font-size:13px;padding:20px;border:none;")
            layout.addWidget(placeholder)

        # Replace image frame content atomically
        ly = self._image_frame.layout()
        old = ly.takeAt(0)
        if old and old.widget():
            old.widget().hide()
            old.widget().deleteLater()
        ly.addWidget(container)

        # Update page counter and buttons
        self._page_label.setText(
            f"{page + 1}/{total_pages}" if total_pages > 0 else "—")
        self._prev_btn.setEnabled(page > 0)
        self._next_btn.setEnabled(page < total_pages - 1)

    def _flip_prev(self):
        if self._active_page > 0:
            self._active_page -= 1
            self._auto_update = False  # User manually flipped page
            self._show_current_image()

    def _flip_next(self):
        paths = self._image_cache.get(self._active_group, []) if self._active_group else []
        per_page = 2 if (len(paths) == 2 and self._active_group
                         and ("CM" in self._active_group or "cm" in self._active_group)) else 1
        total_pages = max(1, (len(paths) + per_page - 1) // per_page) if paths else 0
        if self._active_page < total_pages - 1:
            self._active_page += 1
            self._auto_update = False  # User manually flipped page
            self._show_current_image()

    def _rebuild_side_panel(self):
        """Rebuild group buttons in the side panel."""
        # Clear old buttons (keep trailing stretch)
        while self._group_btns_layout.count() > 1:
            item = self._group_btns_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Sort: CM groups first, then by directory name
        def _sort_key(g):
            if g == "Meta CM":
                return (0, "")
            if g == "Temporal CM":
                return (1, "")
            if "meta" in g.lower():
                return (2, g)
            if "temporal" in g.lower():
                return (3, g)
            return (4, g)
        sorted_groups = sorted(self._image_cache.keys(), key=_sort_key)
        btn_style = (
            "QPushButton{font-size:10px;padding:3px 6px;text-align:left;"
            "background:#F0F0F0;color:#333;border:1px solid #D0D0D0;border-radius:3px;}"
            "QPushButton:hover{background:#E0E8F0;}"
            "QPushButton:checked{background:#0078D4;color:#FFF;border-color:#0078D4;}"
        )
        first = True
        for group in sorted_groups:
            label = group.replace("_", " ").title() if "_" in group else group
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(lambda chk=False, g=group: self._show_group(g))
            self._group_btns_layout.addWidget(btn)
            if first and not self._active_group:
                btn.setChecked(True)
                self._active_group = group
                first = False

    def _show_group(self, group: str):
        self._active_group = group
        self._active_page = 0
        self._auto_update = False  # User manually selected a group — stop auto-follow
        self._show_current_image()
        # Update button checked states
        for i in range(self._group_btns_layout.count()):
            w = self._group_btns_layout.itemAt(i).widget()
            if isinstance(w, QPushButton):
                lbl = w.text().replace(" ", "_").lower()
                grp = group.replace("_", " ").title().lower()
                w.setChecked(lbl == grp.replace(" ", "_").lower())

    def _update_cache_and_nav(self, new_cache: dict, auto_switch_to: Optional[str] = None):
        """Update image cache and rebuild nav if groups changed.

        If auto_switch_to is given and exists in new_cache, switch the active
        group to it (used for auto-following latest images during training).
        """
        old_groups = set(self._image_cache.keys())
        self._image_cache = new_cache
        if set(new_cache.keys()) != old_groups:
            self._rebuild_side_panel()

        # Auto-switch to the target group (latest generated images)
        if auto_switch_to is not None and auto_switch_to in new_cache:
            if self._active_group != auto_switch_to:
                self._active_group = auto_switch_to
                self._active_page = 0
                # Update button checked states to match
                for i in range(self._group_btns_layout.count()):
                    w = self._group_btns_layout.itemAt(i).widget()
                    if isinstance(w, QPushButton):
                        lbl = w.text().replace(" ", "_").lower()
                        grp = auto_switch_to.replace(" ", "_").title().lower()
                        w.setChecked(lbl == grp.replace(" ", "_").lower())

        # Refresh current image (handles newly added images within active group)
        paths = new_cache.get(self._active_group, []) if self._active_group else []
        per_page = 2 if (len(paths) == 2 and self._active_group
                         and ("CM" in self._active_group or "cm" in self._active_group)) else 1
        total_pages = max(1, (len(paths) + per_page - 1) // per_page) if paths else 1
        if self._active_page >= total_pages:
            self._active_page = max(0, total_pages - 1)
        self._show_current_image()

    @staticmethod
    def _find_latest_group(image_cache: dict) -> Optional[str]:
        """Return the group whose most recently modified image is newest."""
        best_group = None
        best_mtime = 0.0
        for group, paths in image_cache.items():
            for p in paths:
                try:
                    mtime = Path(p).stat().st_mtime
                except OSError:
                    continue
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_group = group
        return best_group

    def _poll_visualizations(self):
        if not self._poll_run_dir:
            return
        run_dir = Path(self._poll_run_dir)
        if not run_dir.exists():
            return
        current = self._scan_images(run_dir)
        all_current = set()
        for paths in current.values():
            all_current.update(paths)
        if all_current == self._seen_images:
            return
        self._seen_images = all_current
        # Auto-follow latest group if user hasn't manually switched
        auto_target = self._find_latest_group(current) if self._auto_update else None
        self._update_cache_and_nav(current, auto_switch_to=auto_target)

    @Slot(object)
    def on_result(self, result):
        if not result: return
        self._poll_timer.stop()
        stages = result.get("stage_metrics", [])

        self._metrics_table.setRowCount(len(stages))
        col_map = {"accuracy": 1, "macro_f1": 2, "macro_auc": 3, "weighted_f1": 4, "balanced_acc": 5}
        for i, s in enumerate(stages):
            name = s.get("name", "?")
            self._metrics_table.setItem(i, 0, QTableWidgetItem(name))
            for key, col in col_map.items():
                val = s.get(key)
                item = QTableWidgetItem(f"{val:.4f}" if val is not None else "N/A")
                item.setTextAlignment(Qt.AlignCenter)
                self._metrics_table.setItem(i, col, item)
        self._metrics_table.resizeRowsToContents()

        self._update_cache_and_nav(result.get("image_groups", {}))
