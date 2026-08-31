"""ValidationTab — Multi-mode model validation (Pose / Behavior)."""
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QLabel, QTableWidget, QTableWidgetItem, QComboBox, QFrame, QVBoxLayout,
    QHBoxLayout, QScrollArea, QHeaderView, QWidget, QSizePolicy, QStackedWidget,
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap

from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.workers.validation_worker import ValidationWorker

logger = logging.getLogger("gui.validation")

MODE_OPTIONS = ["Pose", "Behavior Prediction"]


class ValidationTab(BaseTab):
    def __init__(self, parent=None):
        self._mode_combo = None
        self._result_stack = None
        super().__init__(title="Validation", tab_key="validation", parent=parent)

    # ── Config panel ──────────────────────────────────────────────────

    def setup_params(self):
        # Mode selector
        mode_frame = QFrame()
        mode_frame.setStyleSheet(
            "QFrame{background:#FFF;border:1px solid #0078D4;border-radius:8px;margin:3px 2px;}")
        mode_layout = QVBoxLayout(mode_frame)
        mode_layout.setContentsMargins(10, 8, 10, 8)
        mode_layout.setSpacing(4)
        title_lbl = QLabel("Validation Mode")
        title_lbl.setStyleSheet(
            "color:#0078D4;font-weight:700;font-size:11px;padding-bottom:4px;border:none;")
        mode_layout.addWidget(title_lbl)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(MODE_OPTIONS)
        self._mode_combo.setCurrentText("Behavior Prediction")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._mode_combo.setStyleSheet(
            "QComboBox{border:1px solid #C0C0C0;border-radius:4px;padding:6px 8px;"
            "background:#FFF;color:#333;font-size:12px;}"
            "QComboBox:focus{border:2px solid #0078D4;}"
            "QComboBox::drop-down{border:none;padding-right:6px;}")
        mode_layout.addWidget(self._mode_combo)
        self._param_layout.insertWidget(0, mode_frame)

        # Model
        self._model_group = ParameterGroup("Model", [
            {"key": "pose_run_dir", "label": "Pose Run Dir", "type": "dir",
             "default": "runs/train/tmp_exp",
             "hint": "YOLO training output — Pose mode"},
            {"key": "run_dir", "label": "Behavior Run Dir", "type": "dir",
             "default": "runs/train/exp1",
             "hint": "SABER training output — Behavior mode"},
        ])
        self.add_config_group(self._model_group)

        # Pose Data
        self._pose_data_group = ParameterGroup("Pose Data", [
            {"key": "pose_val_images", "label": "Val Images", "type": "path_list", "default": [""],
             "hint": "Image folder(s) for pose inference"},
            {"key": "pose_val_labels", "label": "Val Labels", "type": "path_list", "default": [""],
             "hint": "Label folder(s) for metrics (optional)"},
            {"key": "num_samples", "label": "Sample Count", "type": "int",
             "default": 6, "min": 1, "max": 20},
        ])
        self.add_config_group(self._pose_data_group)

        # Pose Config (YAML parameters for model.val())
        self._pose_config_group = ParameterGroup("Pose Config", [
            {"key": "pose_kpt_shape", "label": "kpt_shape", "type": "text",
             "default": "[6, 3]",
             "hint": "[num_keypoints, dims] — e.g. [6, 3]"},
            {"key": "pose_flip_idx", "label": "flip_idx", "type": "text",
             "default": "[0, 1, 2, 3, 4, 5]",
             "hint": "Keypoint indices for left-right flip augmentation"},
            {"key": "pose_kpt_names", "label": "kpt_names", "type": "text",
             "default": "snout, head_center, body_center, tailbase, ear_left, ear_right",
             "hint": "Comma-separated keypoint names per class"},
        ])
        self.add_config_group(self._pose_config_group)

        # Behavior Data
        self._behavior_data_group = ParameterGroup("Behavior Data", [
            {"key": "val_mouse_dirs", "label": "Val Mouse KP", "type": "path_list", "default": [""]},
            {"key": "val_tail_dirs", "label": "Val Tail KP", "type": "path_list", "default": [""]},
            {"key": "val_behavior_m1_files", "label": "Val Behavior M1", "type": "path_list_file", "default": [""]},
            {"key": "val_behavior_m2_files", "label": "Val Behavior M2", "type": "path_list_file", "default": [""]},
            {"key": "max_instances", "label": "Max Instances", "type": "int", "default": 2, "min": 1, "max": 10},
        ])
        self.add_config_group(self._behavior_data_group)

        # Performance
        self._perf_group = ParameterGroup("Performance", [
            {"key": "num_workers", "label": "Workers", "type": "int", "default": 8, "min": 1, "max": 128},
        ])
        self.add_config_group(self._perf_group)

        self._apply_mode_visibility()

    def _on_mode_changed(self, _idx=None):
        self._apply_mode_visibility()

    def _apply_mode_visibility(self):
        if self._mode_combo is None:
            return
        mode = self._mode_combo.currentText()
        cards = self._config_cards
        cards.get(self._pose_data_group, QFrame()).setVisible(mode == "Pose")
        cards.get(self._pose_config_group, QFrame()).setVisible(mode == "Pose")
        cards.get(self._behavior_data_group, QFrame()).setVisible(mode == "Behavior Prediction")

    def gather_params(self) -> dict:
        g = super().gather_params()
        if self._mode_combo is not None:
            g["mode"] = self._mode_combo.currentText()
        return g

    def _load_settings(self):
        from gui.utils.gui_settings import get_tab_settings
        saved = get_tab_settings(self._tab_key)
        if not saved:
            return
        if "mode" in saved and self._mode_combo is not None:
            self._mode_combo.blockSignals(True)
            idx = self._mode_combo.findText(saved["mode"])
            if idx >= 0:
                self._mode_combo.setCurrentIndex(idx)
            self._mode_combo.blockSignals(False)
        for group in self._config_groups:
            for item in group._schema:
                key = item["key"]
                if key in saved:
                    group.set_value(key, saved[key])
        self._apply_mode_visibility()

    # ── Results area — QStackedWidget with fixed per-mode pages ───────

    def setup_results(self):
        self._result_stack = QStackedWidget()
        self._result_stack.setStyleSheet("background:transparent;")
        self._result_stack.setContentsMargins(0, 0, 0, 0)

        # Page 0: Empty (before any validation runs)
        self._result_stack.addWidget(QWidget())

        # Page 1: Pose results
        self._pose_page = self._build_pose_results_page()
        self._result_stack.addWidget(self._pose_page)

        # Page 2: Behavior results
        self._behavior_page = self._build_behavior_results_page()
        self._result_stack.addWidget(self._behavior_page)

        # Default: blank
        self._result_stack.setCurrentIndex(0)

        self._results_layout_main.addWidget(self._result_stack)


    def _section_label(self, text: str) -> QLabel:
        return self._make_label(text, "color:#0078D4;font-size:14px;font-weight:700;padding:6px 0 2px 0;")

    @staticmethod
    def _make_label(text: str, style: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    def _build_pose_results_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._pose_table = QTableWidget(1, 0)
        self._pose_table.setFixedHeight(54)  # header + 1 data row
        self._pose_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._pose_table.verticalHeader().setVisible(False)
        self._pose_table.verticalHeader().setDefaultSectionSize(26)
        self._pose_table.horizontalHeader().setFixedHeight(28)
        hdr = self._pose_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        lay.addWidget(self._pose_table)

        # ── Sample image viewer with navigation ───────────────
        self._pose_samples_frame = QFrame()
        self._pose_samples_frame.setVisible(False)
        self._pose_samples_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pose_samples_frame.setStyleSheet(
            "QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:6px;}")
        samples_layout = QVBoxLayout(self._pose_samples_frame)
        samples_layout.setContentsMargins(4, 4, 4, 4)
        samples_layout.setSpacing(2)

        # Info bar: label + nav buttons
        info_bar = QHBoxLayout()
        self._pose_samples_label = self._make_label(
            "Sample Predictions", "color:#555;font-size:12px;font-weight:600;")
        info_bar.addWidget(self._pose_samples_label)
        info_bar.addStretch()
        self._samples_idx_label = QLabel("")
        self._samples_idx_label.setStyleSheet("color:#888;font-size:11px;")
        info_bar.addWidget(self._samples_idx_label)
        from PySide6.QtWidgets import QPushButton
        prev_btn = QPushButton("◀")
        prev_btn.setFixedSize(28, 24)
        prev_btn.setStyleSheet(
            "QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:4px;"
            "font-size:12px;}QPushButton:hover{background:#E0E0E0;}")
        prev_btn.clicked.connect(self._prev_sample)
        info_bar.addWidget(prev_btn)
        next_btn = QPushButton("▶")
        next_btn.setFixedSize(28, 24)
        next_btn.setStyleSheet(
            "QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:4px;"
            "font-size:12px;}QPushButton:hover{background:#E0E0E0;}")
        next_btn.clicked.connect(self._next_sample)
        info_bar.addWidget(next_btn)
        samples_layout.addLayout(info_bar)

        # Graphics view — zoomable, scrollable, fits image
        from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
        self._pose_graphics = QGraphicsView()
        self._pose_graphics.setRenderHints(
            self._pose_graphics.renderHints()
            | self._pose_graphics.renderHints().Antialiasing
            | self._pose_graphics.renderHints().SmoothPixmapTransform)
        self._pose_graphics.setDragMode(QGraphicsView.ScrollHandDrag)
        self._pose_graphics.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._pose_graphics.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._pose_graphics.setStyleSheet("QGraphicsView{background:#F5F5F5;border:none;}")
        self._pose_graphics.setMinimumHeight(150)
        self._pose_graphics.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Install zoom via wheel
        def zoom_wheel(event):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._pose_graphics.scale(factor, factor)
        self._pose_graphics.wheelEvent = zoom_wheel

        # Install double-click → full viewer
        def dbl_click(event):
            if event.button() == Qt.LeftButton:
                self._open_sample_full()
        self._pose_graphics.mouseDoubleClickEvent = dbl_click

        self._pose_scene = QGraphicsScene()
        self._pose_graphics.setScene(self._pose_scene)
        samples_layout.addWidget(self._pose_graphics)

        lay.addWidget(self._pose_samples_frame)

        # Store refs for navigation
        self._sample_images = []
        self._sample_idx = 0

        return page  # no stretch — content determines height

    def _build_behavior_results_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Per-stage metrics table (compact)
        COLS = ["Stage", "Accuracy", "Balanced Acc", "macro-F1",
                "weighted-F1", "macro-AUC", "weighted-AUC"]
        self._beh_metrics_table = QTableWidget(0, len(COLS))
        self._beh_metrics_table.setHorizontalHeaderLabels(COLS)
        self._beh_metrics_table.verticalHeader().setVisible(False)
        hdr = self._beh_metrics_table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        self._beh_metrics_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._beh_metrics_table.verticalHeader().setDefaultSectionSize(26)
        self._beh_metrics_table.horizontalHeader().setFixedHeight(30)
        self._beh_metrics_table.setMaximumHeight(30 + 26 * 7 + 2)  # header + up to 7 rows
        lay.addWidget(self._beh_metrics_table)

        # Raster label + nav
        raster_bar = QHBoxLayout()
        raster_bar.setContentsMargins(0, 0, 0, 0)
        raster_bar.setSpacing(2)
        self._beh_raster_label = QLabel("Behavior Raster")
        self._beh_raster_label.setStyleSheet("color:#555;font-size:11px;font-weight:600;")
        self._beh_raster_label.setVisible(False)
        raster_bar.addWidget(self._beh_raster_label)
        raster_bar.addStretch()
        self._beh_raster_idx = QLabel("")
        self._beh_raster_idx.setStyleSheet("color:#888;font-size:10px;")
        self._beh_raster_idx.setVisible(False)
        raster_bar.addWidget(self._beh_raster_idx)
        from PySide6.QtWidgets import QPushButton
        rprev = QPushButton("◀")
        rprev.setFixedSize(22, 20)
        rprev.setStyleSheet("QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:3px;font-size:10px;}QPushButton:hover{background:#E0E0E0;}")
        rprev.clicked.connect(self._prev_raster)
        rprev.setVisible(False)
        raster_bar.addWidget(rprev)
        self._rprev_btn = rprev
        rnext = QPushButton("▶")
        rnext.setFixedSize(22, 20)
        rnext.setStyleSheet("QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:3px;font-size:10px;}QPushButton:hover{background:#E0E0E0;}")
        rnext.clicked.connect(self._next_raster)
        rnext.setVisible(False)
        raster_bar.addWidget(rnext)
        self._rnext_btn = rnext
        lay.addLayout(raster_bar)

        self._beh_raster_scroll = QScrollArea()
        self._beh_raster_scroll.setWidgetResizable(True)
        self._beh_raster_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._beh_raster_scroll.setStyleSheet("QScrollArea{border:1px solid #E0E0E0;border-radius:6px;background:#FFF;}")
        self._beh_raster_scroll.setMinimumHeight(200)
        self._beh_raster_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._beh_raster_scroll.setVisible(False)
        self._beh_raster = QLabel()
        self._beh_raster.setAlignment(Qt.AlignCenter)
        self._beh_raster.setStyleSheet("border:none;background:#FFF;padding:4px;")
        self._beh_raster.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._beh_raster_scroll.setWidget(self._beh_raster)
        lay.addWidget(self._beh_raster_scroll)

        self._raster_images = []
        self._raster_idx = 0

        return page  # no stretch

    def on_start(self):
        self._clear_results()
        self.run_worker(ValidationWorker, params=self.gather_params())

    def _clear_results(self):
        """Clear all result visualizations before starting a new run."""
        # Reset stack to blank page
        if self._result_stack is not None:
            self._result_stack.setCurrentIndex(0)

        # Pose results
        self._pose_table.setRowCount(0)
        self._pose_table.setColumnCount(0)
        self._pose_samples_frame.setVisible(False)
        self._pose_scene.clear()
        self._sample_images = []
        self._sample_idx = 0

        # Behavior results
        self._beh_metrics_table.setRowCount(0)
        self._beh_raster_label.setVisible(False)
        self._beh_raster_scroll.setVisible(False)
        self._beh_raster_idx.setVisible(False)
        self._rprev_btn.setVisible(False)
        self._rnext_btn.setVisible(False)
        self._beh_raster.clear()
        self._raster_images = []
        self._raster_idx = 0

    # ── Result dispatch ───────────────────────────────────────────────

    @Slot(object)
    def on_result(self, result):
        if not result:
            return
        mode = result.get("mode", "Behavior Prediction")

        if mode == "Pose":
            self._result_stack.setCurrentIndex(1)
            self._show_pose_result(result)
        else:
            self._result_stack.setCurrentIndex(2)
            self._show_behavior_result(result)

    # ── Pose result ───────────────────────────────────────────────────

    def _show_pose_result(self, result):
        pm = result.get("pose_metrics", {})
        all_metrics = pm.get("all_metrics", {})

        columns, values = self._extract_metric_columns(all_metrics)
        self._pose_table.setColumnCount(len(columns))
        self._pose_table.setHorizontalHeaderLabels(columns)
        self._pose_table.setRowCount(1)
        for j, v in enumerate(values):
            item = QTableWidgetItem(f"{v:.4f}" if isinstance(v, float) else str(v))
            item.setTextAlignment(0x0004 | 0x0080)
            self._pose_table.setItem(0, j, item)

        self._sample_images = [p for p in pm.get("sample_images", []) if Path(p).exists()]
        self._sample_idx = 0
        if self._sample_images:
            self._pose_samples_frame.setVisible(True)
            self._show_current_sample()
        else:
            self._pose_samples_frame.setVisible(False)

    def _show_current_sample(self):
        if not self._sample_images:
            return
        self._pose_scene.clear()
        pix = QPixmap(self._sample_images[self._sample_idx])
        if pix.isNull():
            return
        self._pose_scene.addPixmap(pix)
        # Force scene rect to match the new pixmap exactly
        self._pose_scene.setSceneRect(self._pose_scene.itemsBoundingRect())
        self._pose_graphics.resetTransform()
        self._pose_graphics.fitInView(self._pose_scene.sceneRect(), Qt.KeepAspectRatio)
        n = len(self._sample_images)
        self._samples_idx_label.setText(f"{self._sample_idx + 1} / {n}")

    def _open_sample_full(self):
        """Open the current sample image in a zoomable dialog."""
        if not self._sample_images:
            return
        from gui.tabs.train_behavior_tab import ImageViewer
        ImageViewer(self._sample_images[self._sample_idx], self).exec()

    def _show_current_raster(self):
        if not self._raster_images:
            return
        pix = QPixmap(self._raster_images[self._raster_idx])
        if pix.isNull():
            return
        w = max(self._beh_raster_scroll.viewport().width() - 8, 400)
        self._beh_raster.setPixmap(pix.scaledToWidth(w, Qt.SmoothTransformation))
        self._beh_raster_idx.setText(f"{self._raster_idx + 1} / {len(self._raster_images)}")

    def _prev_raster(self):
        if self._raster_images:
            self._raster_idx = (self._raster_idx - 1) % len(self._raster_images)
            self._show_current_raster()

    def _next_raster(self):
        if self._raster_images:
            self._raster_idx = (self._raster_idx + 1) % len(self._raster_images)
            self._show_current_raster()

    def _prev_sample(self):
        if self._sample_images:
            self._sample_idx = (self._sample_idx - 1) % len(self._sample_images)
            self._show_current_sample()

    def _next_sample(self):
        if self._sample_images:
            self._sample_idx = (self._sample_idx + 1) % len(self._sample_images)
            self._show_current_sample()

    # ── Behavior result ───────────────────────────────────────────────

    COLS_STANDARD = ["accuracy", "balanced_accuracy", "macro_f1",
                      "weighted_f1", "macro_auc", "weighted_auc"]

    def _show_behavior_result(self, result):
        group_metrics = result.get("group_metrics", {})
        meta_metrics = result.get("meta_metrics", {})
        temporal_metrics = result.get("temporal_metrics", {})
        decoder_metrics = result.get("decoder_metrics", {})

        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "—"

        # Build rows: short, medium, long, meta, temporal, final
        rows = []
        for stage in ["short", "medium", "long"]:
            gm = group_metrics.get(stage, {})
            if gm:
                rows.append([f"LGBM-{stage}"] + [_fmt(gm.get(k)) for k in self.COLS_STANDARD])

        for label, m in [("Meta-LGBM", meta_metrics),
                         ("Temporal NN", temporal_metrics),
                         ("+ Decoder", decoder_metrics)]:
            if m:
                rows.append([label] + [_fmt(m.get(k)) for k in self.COLS_STANDARD])

        self._beh_metrics_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(0x0004 | 0x0080)
                self._beh_metrics_table.setItem(i, j, item)

        # Collapse log to make room for results
        self.collapse_log()

        # Show raster
        self._raster_images = [p for p in result.get("raster_plots", []) if Path(p).exists()]
        self._raster_idx = 0
        if self._raster_images:
            self._beh_raster_label.setVisible(True)
            self._beh_raster_scroll.setVisible(True)
            self._beh_raster_idx.setVisible(True)
            self._rprev_btn.setVisible(True)
            self._rnext_btn.setVisible(True)
            self._show_current_raster()
        else:
            self._beh_raster_label.setVisible(False)
            self._beh_raster_scroll.setVisible(False)
            self._beh_raster_idx.setVisible(False)
            self._rprev_btn.setVisible(False)
            self._rnext_btn.setVisible(False)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_metric_columns(all_metrics: dict):
        columns, values = [], []
        priority = [
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            "metrics/mAP50(P)", "metrics/mAP50-95(P)",
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/precision(P)", "metrics/recall(P)",
        ]
        seen = set()
        for k in priority:
            if k in all_metrics:
                columns.append(k.replace("metrics/", ""))
                values.append(all_metrics[k])
                seen.add(k)
        for k, v in sorted(all_metrics.items()):
            if k not in seen:
                columns.append(k.replace("metrics/", ""))
                values.append(v)
        return columns, values
