"""InferenceTab — Model inference (Pose / Behavior Prediction)."""
import logging; from pathlib import Path
from PySide6.QtWidgets import (
    QLabel, QTableWidget, QTableWidgetItem, QPushButton, QComboBox,
    QFrame, QVBoxLayout, QHBoxLayout, QHeaderView, QSlider,
    QWidget, QSizePolicy, QStackedWidget, QSpinBox,
    QGraphicsEllipseItem,
)
from PySide6.QtCore import Qt, Slot, QTimer, QEvent
from PySide6.QtGui import QPixmap
from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.workers.inference_worker import InferenceWorker

logger = logging.getLogger("gui.inference")


MODE_OPTIONS = ["Pose", "Behavior Prediction", "End-to-End"]


class _RoiShapeItem(QGraphicsEllipseItem):
    """Interactive ROI shape with hover, select, and vertex display.
    ROIs are fixed after creation — no drag/resize."""

    def __init__(self, roi_idx, roi_type, points, color, tab):
        super().__init__()
        self._roi_idx = roi_idx
        self._roi_type = roi_type
        self._points = [list(p) for p in points]
        self._color = color
        self._tab = tab
        self._hovered = False
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self._update_shape()

    def _update_shape(self):
        from PySide6.QtGui import QPainterPath, QPolygonF
        from PySide6.QtCore import QPointF, QRectF
        path = QPainterPath()
        if self._roi_type == "rect":
            x0, y0 = self._points[0]; x1, y1 = self._points[1]
            path.addRect(QRectF(x0, y0, x1 - x0, y1 - y0))
        else:
            poly = QPolygonF([QPointF(p[0], p[1]) for p in self._points])
            path.addPolygon(poly)
        self._shape_path = path

    def boundingRect(self):
        return self._shape_path.boundingRect().adjusted(-4, -4, 4, 4)

    def shape(self):
        return self._shape_path

    def paint(self, painter, option, widget=None):
        from PySide6.QtGui import QPen, QColor as _QCol, QBrush, QFont as _QF
        from PySide6.QtCore import QPointF, QRectF
        c = self._color; qc = _QCol(c[0], c[1], c[2])
        if self._hovered:
            fill = _QCol(c[0], c[1], c[2], 80); pen = QPen(_QCol(255, 255, 200), 3)
        elif self._tab._roi_selected_idx == self._roi_idx:
            fill = _QCol(c[0], c[1], c[2], 60); pen = QPen(_QCol(255, 255, 255), 3)
        else:
            fill = _QCol(c[0], c[1], c[2], 35); pen = QPen(qc, 2)
        painter.setPen(pen); painter.setBrush(QBrush(fill))
        painter.drawPath(self._shape_path)
        # Vertex dots
        painter.setPen(QPen(_QCol(255, 255, 255, 200), 1))
        painter.setBrush(QBrush(_QCol(255, 255, 255, 200)))
        for p in self._points:
            painter.drawEllipse(QPointF(p[0], p[1]), 5, 5)
        # Stats label — smart placement to avoid overlap
        stats = self._tab._roi_stats.get(self._roi_idx, {})
        if stats:
            font = _QF("Segoe UI", 12, _QF.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()
            roi_label = self._tab._rois[self._roi_idx].get("label", f"ROI {self._roi_idx + 1}")
            lines = [roi_label]
            for mid, s in sorted(stats.items(), key=lambda x: int(x[0])):
                lines.append(f"M{int(mid) + 1}: in={s['entries']} out={s['exits']} Σ={s['frames_inside']}")
            lh = fm.height() + 4
            max_w = max(fm.horizontalAdvance(l) for l in lines) + 16
            box_h = lh * len(lines) + 10
            bb = self._shape_path.boundingRect()
            bx, by = self._tab._get_roi_stats_pos(
                self._roi_idx, int(max_w), int(box_h), bb)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(_QCol(0, 0, 0, 190)))
            painter.drawRoundedRect(QRectF(bx, by, max_w, box_h), 6, 6)
            painter.setPen(_QCol(255, 255, 255))
            for li, line in enumerate(lines):
                painter.drawText(int(bx + 8), int(by + fm.ascent() + 4 + li * lh), line)

    def hoverEnterEvent(self, event):
        self._hovered = True; self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._hovered = False; self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            if self._tab._roi_selected_idx != self._roi_idx:
                self._tab._roi_selected_idx = self._roi_idx
                # Defer rebuild so self isn't destroyed mid-event
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, self._tab._rebuild_roi_handles)
            return
        super().mousePressEvent(event)


class InferenceTab(BaseTab):
    def __init__(self, parent=None):
        self._mode_combo = None
        # ── Visualization mode state ──
        self._viz_mode = "frame_only"        # frame_only | frame_trail | global_traj | heatmap
        self._trajectory_cache = {}           # {source_idx: trajectory_data_dict}
        self._trail_length = 30               # frames retained for trail (5–120)
        self._global_traj_cache = {}          # {source_idx: QPixmap} cached global trajectory base
        self._heatmap_cache = {}              # {source_idx: QImage} cached heatmap overlay
        self._e2e_source_idx = 0              # which source is displayed in E2E
        self._e2e_labels_by_source = {}       # {source_idx: {mid: [labels]}}
        self._e2e_rasters_by_source = {}      # {source_idx: [raster_paths]}
        self._viz_btn_bar = None
        self._viz_btns = {}
        self._trail_slider = None
        self._heatmap_inline = None
        self._e2e_viz_btn_bar = None
        self._e2e_viz_btns = {}
        self._e2e_trail_slider = None
        self._e2e_heatmap_inline = None
        # ── ROI analysis state ──
        self._rois = []                      # list of ROI dicts
        self._roi_draw_tool = None           # None | "rect" | "polygon"
        self._roi_drawing = False            # currently mid-draw?
        self._roi_current_pts = []           # vertices being placed (scene coords)
        self._roi_preview_item = None        # QGraphicsItem preview during draw
        self._roi_items = []                 # QGraphicsItems for completed ROIs
        self._roi_selected_idx = -1          # selected ROI index
        self._roi_stats = {}                 # {roi_idx: {mouse_id: {entries, exits, frames_inside}}}
        self._roi_inline = None             # ROI sub-toolbar widget
        self._roi_e2e_inline = None         # E2E ROI sub-toolbar widget
        self._is_live_mode = False           # True when live camera is active
        self._live_camera_group = None       # ParameterGroup for live camera settings
        self._e2e_live_frame_idx = 0         # Current frame index in live mode
        self._e2e_total_frames = 0           # Total timeline frames for raster indicator
        self._live_raw_pix = None            # Latest un-viz'd live frame (for mode switch)
        self._roi_colors = [                 # color palette for ROIs
            (255, 80, 80, 80), (80, 180, 255, 80), (80, 255, 120, 80),
            (255, 220, 60, 80), (200, 80, 255, 80), (255, 140, 40, 80),
        ]
        super().__init__(title="Inference", tab_key="inference", parent=parent)

    def setup_params(self):
        # Mode selector
        mode_frame = QFrame()
        mode_frame.setStyleSheet(
            "QFrame{background:#FFF;border:1px solid #0078D4;border-radius:8px;margin:3px 2px;}")
        mode_layout = QVBoxLayout(mode_frame)
        mode_layout.setContentsMargins(10, 8, 10, 8)
        mode_layout.setSpacing(4)
        title_lbl = QLabel("Inference Mode")
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
            {"key": "pose_run_dir", "label": "Mouse Model", "type": "dir",
             "default": "runs/train/tmp_exp",
             "hint": "YOLO mouse pose model (body keypoints)"},
            {"key": "tail_run_dir", "label": "Tail Model", "type": "dir",
             "default": "",
             "hint": "YOLO tail keypoint model (optional, leave empty to disable)"},
            {"key": "run_dir", "label": "Behavior Model", "type": "dir",
             "default": "runs/train/exp1",
             "hint": "SABER 3-stage pipeline training output"},
        ])
        self.add_config_group(self._model_group)

        # Pose Data
        self._pose_data_group = ParameterGroup("Pose Data", [
            {"key": "pose_inference_sources", "label": "Images / Videos", "type": "path_list",
             "default": [""],
             "hint": "Image folders or video files for pose inference"},
            {"key": "max_mice", "label": "Max Mice", "type": "int",
             "default": 2, "min": 1, "max": 8,
             "hint": "Maximum number of mice to track (leftmost = Mouse 1)"},
            {"key": "pose_kpt_shape", "label": "kpt_shape", "type": "text",
             "default": "[6, 3]"},
            {"key": "pose_flip_idx", "label": "flip_idx", "type": "text",
             "default": "[0, 1, 2, 3, 4, 5]"},
            {"key": "pose_kpt_names", "label": "kpt_names", "type": "text",
             "default": "snout, head_center, body_center, tailbase, ear_left, ear_right"},
        ])
        self.add_config_group(self._pose_data_group)

        # Live Camera
        self._live_camera_group = ParameterGroup("Live Camera", [
            {"key": "live_camera", "label": "Enable Live Camera", "type": "checkbox",
             "default": False,
             "hint": "Use webcam/camera as input source instead of files"},
            {"key": "camera_id", "label": "Camera ID", "type": "int",
             "default": 0, "min": 0, "max": 10,
             "hint": "Camera device ID (0 = default webcam)"},
            {"key": "camera_width", "label": "Resolution Width", "type": "int",
             "default": 640, "min": 320, "max": 3840,
             "hint": "Camera capture width"},
            {"key": "camera_height", "label": "Resolution Height", "type": "int",
             "default": 480, "min": 240, "max": 2160,
             "hint": "Camera capture height"},
            {"key": "camera_fps", "label": "Target FPS", "type": "int",
             "default": 30, "min": 5, "max": 120,
             "hint": "Target camera capture frame rate"},
            # ── Auto-save ──
            {"key": "auto_save_interval_min", "label": "Auto-Save Interval (min)", "type": "int",
             "default": 10, "min": 0, "max": 120,
             "hint": "Auto-save interval in minutes (0 = disabled)"},
            {"key": "auto_save_behavior_csv", "label": "Save Behavior CSV", "type": "checkbox",
             "default": True,
             "hint": "Auto-save frame-by-frame behavior labels as CSV"},
            {"key": "auto_save_raster_png", "label": "Save Raster PNG", "type": "checkbox",
             "default": True,
             "hint": "Auto-save cumulative behavior raster plot as PNG"},
            {"key": "auto_save_dir", "label": "Save Directory", "type": "dir",
             "default": "live_output",
             "hint": "Output directory for auto-saved files"},
        ])
        self.add_config_group(self._live_camera_group)

        # Behavior Data
        self._behavior_data_group = ParameterGroup("Behavior Data", [
            {"key": "mouse_keypoints", "label": "Mouse KP", "type": "dir", "default": ""},
            {"key": "tail_keypoints", "label": "Tail KP", "type": "dir", "default": ""},
            {"key": "skip_temporal", "label": "Skip Temporal", "type": "checkbox",
             "default": False,
             "hint": "Skip temporal model stage, use meta-LGBM output directly"},
        ])
        self.add_config_group(self._behavior_data_group)

        self._apply_mode_visibility()

    def _on_mode_changed(self, _idx=None):
        self._apply_mode_visibility()

    def _apply_mode_visibility(self):
        if self._mode_combo is None:
            return
        mode = self._mode_combo.currentText()
        cards = self._config_cards
        is_pose = mode in ("Pose", "End-to-End")
        is_behavior = mode in ("Behavior Prediction", "End-to-End")
        cards.get(self._pose_data_group, QFrame()).setVisible(is_pose)
        cards.get(self._behavior_data_group, QFrame()).setVisible(is_behavior)
        # Live Camera group only for Pose and End-to-End modes
        if self._live_camera_group is not None:
            cards.get(self._live_camera_group, QFrame()).setVisible(is_pose)

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
        super()._load_settings()
        self._apply_mode_visibility()

    def setup_results(self):
        self._result_stack = QStackedWidget()
        self._result_stack.setStyleSheet("background:transparent;")
        self._result_stack.setContentsMargins(0, 0, 0, 0)
        self._result_stack.addWidget(QWidget())  # Page 0: Empty
        self._pose_page = self._build_pose_results_page()
        self._result_stack.addWidget(self._pose_page)  # Page 1
        self._behavior_page = self._build_behavior_results_page()
        self._result_stack.addWidget(self._behavior_page)  # Page 2
        self._e2e_page = self._build_e2e_results_page()
        self._result_stack.addWidget(self._e2e_page)  # Page 3
        self._result_stack.setCurrentIndex(0)
        # Re-fit E2E pose when page becomes visible
        self._result_stack.currentChanged.connect(self._on_page_changed)
        self._results_layout_main.addWidget(self._result_stack)

    def _on_page_changed(self, idx):
        if idx == 3 and self._e2e_sample_images:
            QTimer.singleShot(150, lambda: self._e2e_pose_view.fitInView(
                self._e2e_pose_scene.sceneRect(), Qt.KeepAspectRatio))
            # Also re-fit the raster view after layout settles
            if self._e2e_raster_images:
                QTimer.singleShot(150, lambda: self._e2e_raster_view.fitInView(
                    self._e2e_raster_scene.sceneRect(), Qt.IgnoreAspectRatio))

    def _build_e2e_results_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # ── Nav bar (matches Pose mode) ──
        nav = QHBoxLayout()
        nav.setContentsMargins(2, 0, 2, 0)
        nav.setSpacing(4)
        self._e2e_pose_label = QLabel("Pose Frame")
        self._e2e_pose_label.setStyleSheet("color:#555;font-size:11px;font-weight:600;")
        nav.addWidget(self._e2e_pose_label)
        nav.addStretch()
        self._e2e_frame_idx_label = QLabel("")
        self._e2e_frame_idx_label.setStyleSheet("color:#888;font-size:10px;")
        nav.addWidget(self._e2e_frame_idx_label)
        self._e2e_play_btn = QPushButton("▶")
        self._e2e_play_btn.setFixedSize(30, 22)
        self._e2e_play_btn.setToolTip("Auto-play frames (Space)")
        self._e2e_play_btn.setStyleSheet(
            "QPushButton{background:#0078D4;color:#FFF;border:none;border-radius:3px;font-size:10px;}"
            "QPushButton:hover{background:#006CBE;}")
        self._e2e_play_btn.clicked.connect(self._e2e_toggle_play)
        nav.addWidget(self._e2e_play_btn)
        self._e2e_speed_combo = QComboBox()
        self._e2e_speed_combo.addItems(["0.5x", "1x", "2x", "4x"])
        self._e2e_speed_combo.setCurrentText("1x")
        self._e2e_speed_combo.setFixedWidth(50)
        self._e2e_speed_combo.setStyleSheet(
            "QComboBox{border:1px solid #C0C0C0;border-radius:3px;padding:1px 3px;"
            "background:#FFF;color:#333;font-size:9px;}QComboBox::drop-down{border:none;}")
        self._e2e_speed_combo.currentTextChanged.connect(self._e2e_speed_changed)
        nav.addWidget(self._e2e_speed_combo)
        self._e2e_slider = QSlider(Qt.Horizontal)
        self._e2e_slider.setRange(0, 1000)
        self._e2e_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#E0E0E0;height:6px;border-radius:3px;}"
            "QSlider::handle:horizontal{background:#0078D4;width:14px;height:14px;"
            "margin:-4px 0;border-radius:7px;}"
            "QSlider::sub-page:horizontal{background:#0078D4;border-radius:3px;}")
        self._e2e_slider.sliderPressed.connect(lambda: self._e2e_pause_during_drag())
        self._e2e_slider.sliderReleased.connect(self._e2e_slider_seek)
        nav.addWidget(self._e2e_slider, 1)
        prev_btn = QPushButton("◀")
        prev_btn.setFixedSize(26, 22)
        prev_btn.setStyleSheet("QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:3px;font-size:10px;}QPushButton:hover{background:#E0E0E0;}")
        prev_btn.clicked.connect(self._e2e_pose_prev)
        nav.addWidget(prev_btn)
        self._e2e_prev_btn = prev_btn
        next_btn = QPushButton("▶")
        next_btn.setFixedSize(26, 22)
        next_btn.setStyleSheet("QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:3px;font-size:10px;}QPushButton:hover{background:#E0E0E0;}")
        next_btn.clicked.connect(self._e2e_pose_next)
        nav.addWidget(next_btn)
        self._e2e_next_btn = next_btn
        lay.addLayout(nav)

        # ── E2E Viz mode button bar ──
        self._e2e_viz_btn_bar = QWidget()
        self._e2e_viz_btn_bar.setVisible(False)
        e2e_viz_bar = QHBoxLayout(self._e2e_viz_btn_bar)
        e2e_viz_bar.setContentsMargins(2, 2, 2, 2)
        e2e_viz_bar.setSpacing(3)
        e2e_viz_lbl = QLabel("View:")
        e2e_viz_lbl.setStyleSheet("color:#888;font-size:10px;border:none;")
        e2e_viz_bar.addWidget(e2e_viz_lbl)
        _e2e_viz_style = (
            "QPushButton{font-size:10px;padding:2px 7px;background:#F0F0F0;color:#333;"
            "border:1px solid #D0D0D0;border-radius:3px;}"
            "QPushButton:hover{background:#E0E8F0;}"
            "QPushButton:checked{background:#0078D4;color:#FFF;border-color:#0078D4;}"
        )
        for mode_key, mode_label in [("frame_only","Frame"),("frame_trail","Trail"),
                                      ("global_traj","Global"),("heatmap","Heatmap"),
                                      ("roi","ROI")]:
            btn = QPushButton(mode_label)
            btn.setCheckable(True)
            btn.setChecked(mode_key == "frame_only")
            btn.setStyleSheet(_e2e_viz_style)
            btn.clicked.connect(lambda checked=False, mk=mode_key: self._on_e2e_viz_mode_changed(mk))
            e2e_viz_bar.addWidget(btn)
            self._e2e_viz_btns[mode_key] = btn
        e2e_viz_bar.addSpacing(8)
        e2e_trail_lbl = QLabel("Tail:")
        e2e_trail_lbl.setStyleSheet("color:#888;font-size:10px;border:none;")
        e2e_viz_bar.addWidget(e2e_trail_lbl)
        self._e2e_trail_slider = QSlider(Qt.Horizontal)
        self._e2e_trail_slider.setRange(5, 120)
        self._e2e_trail_slider.setValue(30)
        self._e2e_trail_slider.setFixedWidth(80)
        self._e2e_trail_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#E0E0E0;height:4px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#0078D4;width:10px;height:10px;"
            "margin:-3px 0;border-radius:5px;}"
            "QSlider::sub-page:horizontal{background:#0078D4;border-radius:2px;}")
        self._e2e_trail_slider.valueChanged.connect(self._on_trail_length_changed)
        e2e_viz_bar.addWidget(self._e2e_trail_slider)
        e2e_viz_bar.addSpacing(6)
        e2e_export_btn = QPushButton("Export")
        e2e_export_btn.setFixedWidth(48)
        e2e_export_btn.setStyleSheet(
            "QPushButton{font-size:10px;padding:2px 6px;background:#E8E8E8;color:#333;"
            "border:1px solid #CCC;border-radius:3px;}"
            "QPushButton:hover{background:#D0D0D0;}")
        e2e_export_btn.clicked.connect(self._on_export_trajectory)
        e2e_viz_bar.addWidget(e2e_export_btn)
        self._e2e_viz_export_btn = e2e_export_btn
        # Heatmap uses pixel-level rendering (no grid controls needed)
        self._e2e_heatmap_inline = None
        e2e_viz_bar.addStretch()
        lay.addWidget(self._e2e_viz_btn_bar)

        # (E2E heatmap config row removed — controls now inline in viz bar)

        # ── Inline E2E ROI drawing controls ──
        self._roi_e2e_inline = QWidget()
        self._roi_e2e_inline.setVisible(False)
        roi_e2e_in = QHBoxLayout(self._roi_e2e_inline)
        roi_e2e_in.setContentsMargins(0, 0, 0, 0)
        roi_e2e_in.setSpacing(3)
        _roi_e2e_style = (
            "QPushButton{font-size:10px;padding:2px 5px;background:#F0F0F0;color:#333;"
            "border:1px solid #D0D0D0;border-radius:3px;}"
            "QPushButton:hover{background:#E0E8F0;}"
            "QPushButton:checked{background:#E07050;color:#FFF;border-color:#C05030;}"
        )
        self._e2e_roi_rect_btn = QPushButton("▭")
        self._e2e_roi_rect_btn.setToolTip("Draw rectangle ROI")
        self._e2e_roi_rect_btn.setCheckable(True)
        self._e2e_roi_rect_btn.setFixedWidth(24)
        self._e2e_roi_rect_btn.setStyleSheet(_roi_e2e_style)
        self._e2e_roi_rect_btn.clicked.connect(
            lambda checked=False: self._on_roi_tool("rect" if checked else None))
        roi_e2e_in.addWidget(self._e2e_roi_rect_btn)
        self._e2e_roi_poly_btn = QPushButton("⬠")
        self._e2e_roi_poly_btn.setToolTip("Draw polygon ROI (click vertices, dbl-click finish)")
        self._e2e_roi_poly_btn.setCheckable(True)
        self._e2e_roi_poly_btn.setFixedWidth(24)
        self._e2e_roi_poly_btn.setStyleSheet(_roi_e2e_style)
        self._e2e_roi_poly_btn.clicked.connect(
            lambda checked=False: self._on_roi_tool("polygon" if checked else None))
        roi_e2e_in.addWidget(self._e2e_roi_poly_btn)
        e2e_del_btn = QPushButton("✕")
        e2e_del_btn.setToolTip("Delete last ROI")
        e2e_del_btn.setFixedWidth(24)
        e2e_del_btn.setStyleSheet(
            "QPushButton{font-size:9px;padding:1px 3px;background:#F0F0F0;color:#C44;"
            "border:1px solid #D0D0D0;border-radius:3px;}"
            "QPushButton:hover{background:#FDD;}")
        e2e_del_btn.clicked.connect(self._on_roi_delete)
        roi_e2e_in.addWidget(e2e_del_btn)
        e2e_clear_btn = QPushButton("Clear")
        e2e_clear_btn.setToolTip("Clear all ROIs")
        e2e_clear_btn.setFixedWidth(36)
        e2e_clear_btn.setStyleSheet(
            "QPushButton{font-size:9px;padding:1px 4px;background:#F0F0F0;color:#333;"
            "border:1px solid #D0D0D0;border-radius:3px;}"
            "QPushButton:hover{background:#E0E0E0;}")
        e2e_clear_btn.clicked.connect(self._on_roi_clear_all)
        roi_e2e_in.addWidget(e2e_clear_btn)
        self._e2e_roi_status_lbl = QLabel("")
        self._e2e_roi_status_lbl.setStyleSheet("color:#888;font-size:9px;border:none;")
        roi_e2e_in.addWidget(self._e2e_roi_status_lbl)
        e2e_viz_bar.addWidget(self._roi_e2e_inline)

        # (E2E ROI toolbar row removed — controls now inline in viz bar)

        # ── Splitter: pose (top) / raster (bottom) ──
        from PySide6.QtWidgets import QSplitter as _QS2
        splitter = _QS2(Qt.Vertical)
        splitter.setStyleSheet("QSplitter::handle{background:#D0D0D0;height:2px;}")

        # Pose view
        from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
        self._e2e_pose_view = QGraphicsView()
        self._e2e_pose_view.setRenderHints(
            self._e2e_pose_view.renderHints()
            | self._e2e_pose_view.renderHints().Antialiasing
            | self._e2e_pose_view.renderHints().SmoothPixmapTransform)
        self._e2e_pose_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._e2e_pose_view.setMinimumHeight(180)
        self._e2e_pose_view.setStyleSheet("QGraphicsView{background:#F5F5F5;border:1px solid #E0E0E0;}")
        def e2e_zoom_wheel(event):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._e2e_pose_view.scale(factor, factor)
        self._e2e_pose_view.wheelEvent = e2e_zoom_wheel
        self._e2e_pose_scene = QGraphicsScene()
        self._e2e_pose_view.setScene(self._e2e_pose_scene)
        splitter.addWidget(self._e2e_pose_view)

        # Raster view
        from PySide6.QtGui import QPen, QColor as _QCo3
        self._e2e_raster_view = QGraphicsView()
        self._e2e_raster_view.setRenderHints(
            self._e2e_raster_view.renderHints()
            | self._e2e_raster_view.renderHints().Antialiasing
            | self._e2e_raster_view.renderHints().SmoothPixmapTransform)
        self._e2e_raster_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._e2e_raster_view.setMinimumHeight(100)
        self._e2e_raster_view.setStyleSheet("QGraphicsView{background:#FFF;border:1px solid #E0E0E0;}")
        self._e2e_raster_scene = QGraphicsScene()
        self._e2e_raster_view.setScene(self._e2e_raster_scene)
        self._e2e_raster_indicator = self._e2e_raster_scene.addLine(
            0, 0, 0, 100, QPen(_QCo3(255, 30, 30), 3))
        self._e2e_raster_indicator.setZValue(10)
        # Install resize event filter to re-fit raster after layout changes
        self._e2e_raster_view.installEventFilter(self)
        splitter.addWidget(self._e2e_raster_view)
        splitter.setSizes([600, 260])
        lay.addWidget(splitter)

        # State
        self._e2e_sample_images = []
        self._e2e_sample_idx = 0
        self._e2e_raster_images = []
        self._e2e_play_timer = None
        self._e2e_play_fps = 30
        self._e2e_labels = {}
        self._e2e_class_names = []
        self._e2e_slider_dragging = False
        return page

    def _build_pose_results_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._pose_samples_frame = QFrame()
        self._pose_samples_frame.setVisible(False)
        self._pose_samples_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pose_samples_frame.setStyleSheet(
            "QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:6px;}")
        slay = QVBoxLayout(self._pose_samples_frame)
        slay.setContentsMargins(4, 4, 4, 4)
        slay.setSpacing(2)

        # ── Top bar: label + index + nav + playback controls ──
        top_bar = QHBoxLayout()
        top_bar.setSpacing(4)
        self._pose_samples_label = QLabel("Sample Predictions")
        self._pose_samples_label.setStyleSheet("color:#555;font-size:12px;font-weight:600;border:none;")
        top_bar.addWidget(self._pose_samples_label)
        top_bar.addStretch()

        self._samples_idx_label = QLabel("")
        self._samples_idx_label.setStyleSheet("color:#888;font-size:11px;border:none;")
        top_bar.addWidget(self._samples_idx_label)

        prev_btn = QPushButton("◀")
        prev_btn.setFixedSize(28, 24)
        prev_btn.setStyleSheet(
            "QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#E0E0E0;}")
        prev_btn.clicked.connect(self._on_pose_prev)
        top_bar.addWidget(prev_btn)

        # Play / Pause (shown for video mode)
        self._vp_play_btn = QPushButton("▶")
        self._vp_play_btn.setFixedSize(32, 24)
        self._vp_play_btn.setToolTip("Auto-play frames (Space)")
        self._vp_play_btn.setStyleSheet(
            "QPushButton{background:#0078D4;color:#FFF;border:none;border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#006CBE;}")
        self._vp_play_btn.clicked.connect(self._on_play_pause)
        self._vp_play_btn.setVisible(False)
        top_bar.addWidget(self._vp_play_btn)

        # Speed selector (shown for video mode)
        self._vp_speed_combo = QComboBox()
        self._vp_speed_combo.addItems(["0.5x", "1x", "2x", "4x"])
        self._vp_speed_combo.setCurrentText("1x")
        self._vp_speed_combo.setFixedWidth(56)
        self._vp_speed_combo.setStyleSheet(
            "QComboBox{border:1px solid #C0C0C0;border-radius:3px;padding:2px 4px;"
            "background:#FFF;color:#333;font-size:10px;}"
            "QComboBox::drop-down{border:none;padding-right:2px;}")
        self._vp_speed_combo.currentTextChanged.connect(self._on_speed_changed)
        self._vp_speed_combo.setVisible(False)
        top_bar.addWidget(self._vp_speed_combo)

        next_btn = QPushButton("▶")
        next_btn.setFixedSize(28, 24)
        next_btn.setStyleSheet(
            "QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#E0E0E0;}")
        next_btn.clicked.connect(self._on_pose_next)
        top_bar.addWidget(next_btn)
        self._pose_prev_btn = prev_btn
        self._pose_next_btn = next_btn
        slay.addLayout(top_bar)

        # ── Viz mode button bar ──
        self._viz_btn_bar = QWidget()
        self._viz_btn_bar.setVisible(False)
        viz_bar_layout = QHBoxLayout(self._viz_btn_bar)
        viz_bar_layout.setContentsMargins(0, 2, 0, 2)
        viz_bar_layout.setSpacing(3)
        viz_label = QLabel("View:")
        viz_label.setStyleSheet("color:#888;font-size:10px;border:none;")
        viz_bar_layout.addWidget(viz_label)
        _viz_btn_style = (
            "QPushButton{font-size:10px;padding:2px 7px;background:#F0F0F0;color:#333;"
            "border:1px solid #D0D0D0;border-radius:3px;}"
            "QPushButton:hover{background:#E0E8F0;}"
            "QPushButton:checked{background:#0078D4;color:#FFF;border-color:#0078D4;}"
        )
        for mode_key, mode_label in [("frame_only","Frame"), ("frame_trail","Trail"),
                                      ("global_traj","Global"), ("heatmap","Heatmap"),
                                      ("roi","ROI")]:
            btn = QPushButton(mode_label)
            btn.setCheckable(True)
            btn.setChecked(mode_key == "frame_only")
            btn.setStyleSheet(_viz_btn_style)
            btn.clicked.connect(lambda checked=False, mk=mode_key: self._on_viz_mode_changed(mk))
            viz_bar_layout.addWidget(btn)
            self._viz_btns[mode_key] = btn
        viz_bar_layout.addSpacing(8)
        trail_lbl = QLabel("Tail:")
        trail_lbl.setStyleSheet("color:#888;font-size:10px;border:none;")
        viz_bar_layout.addWidget(trail_lbl)
        self._trail_slider = QSlider(Qt.Horizontal)
        self._trail_slider.setRange(5, 120)
        self._trail_slider.setValue(30)
        self._trail_slider.setFixedWidth(80)
        self._trail_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#E0E0E0;height:4px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#0078D4;width:10px;height:10px;"
            "margin:-3px 0;border-radius:5px;}"
            "QSlider::sub-page:horizontal{background:#0078D4;border-radius:2px;}")
        self._trail_slider.valueChanged.connect(self._on_trail_length_changed)
        viz_bar_layout.addWidget(self._trail_slider)
        viz_bar_layout.addSpacing(6)
        export_btn = QPushButton("Export")
        export_btn.setFixedWidth(48)
        export_btn.setStyleSheet(
            "QPushButton{font-size:10px;padding:2px 6px;background:#E8E8E8;color:#333;"
            "border:1px solid #CCC;border-radius:3px;}"
            "QPushButton:hover{background:#D0D0D0;}")
        export_btn.clicked.connect(self._on_export_trajectory)
        viz_bar_layout.addWidget(export_btn)
        self._viz_export_btn = export_btn
        # Heatmap uses pixel-level rendering (no grid controls needed)
        self._heatmap_inline = None
        viz_bar_layout.addStretch()
        slay.addWidget(self._viz_btn_bar)

        # (heatmap config row removed — controls are now inline in viz bar)

        # ── Inline ROI drawing controls ──
        self._roi_inline = QWidget()
        self._roi_inline.setVisible(False)
        roi_in = QHBoxLayout(self._roi_inline)
        roi_in.setContentsMargins(0, 0, 0, 0)
        roi_in.setSpacing(3)
        _roi_in_style = (
            "QPushButton{font-size:10px;padding:2px 5px;background:#F0F0F0;color:#333;"
            "border:1px solid #D0D0D0;border-radius:3px;}"
            "QPushButton:hover{background:#E0E8F0;}"
            "QPushButton:checked{background:#E07050;color:#FFF;border-color:#C05030;}"
        )
        self._roi_rect_btn = QPushButton("▭")
        self._roi_rect_btn.setToolTip("Draw rectangle ROI")
        self._roi_rect_btn.setCheckable(True)
        self._roi_rect_btn.setFixedWidth(24)
        self._roi_rect_btn.setStyleSheet(_roi_in_style)
        self._roi_rect_btn.clicked.connect(
            lambda checked=False: self._on_roi_tool("rect" if checked else None))
        roi_in.addWidget(self._roi_rect_btn)
        self._roi_poly_btn = QPushButton("⬠")
        self._roi_poly_btn.setToolTip("Draw polygon ROI (click vertices, dbl-click finish)")
        self._roi_poly_btn.setCheckable(True)
        self._roi_poly_btn.setFixedWidth(24)
        self._roi_poly_btn.setStyleSheet(_roi_in_style)
        self._roi_poly_btn.clicked.connect(
            lambda checked=False: self._on_roi_tool("polygon" if checked else None))
        roi_in.addWidget(self._roi_poly_btn)
        del_btn = QPushButton("✕")
        del_btn.setToolTip("Delete last ROI")
        del_btn.setFixedWidth(24)
        del_btn.setStyleSheet(
            "QPushButton{font-size:9px;padding:1px 3px;background:#F0F0F0;color:#C44;"
            "border:1px solid #D0D0D0;border-radius:3px;}"
            "QPushButton:hover{background:#FDD;}")
        del_btn.clicked.connect(self._on_roi_delete)
        roi_in.addWidget(del_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Clear all ROIs")
        clear_btn.setFixedWidth(36)
        clear_btn.setStyleSheet(
            "QPushButton{font-size:9px;padding:1px 4px;background:#F0F0F0;color:#333;"
            "border:1px solid #D0D0D0;border-radius:3px;}"
            "QPushButton:hover{background:#E0E0E0;}")
        clear_btn.clicked.connect(self._on_roi_clear_all)
        roi_in.addWidget(clear_btn)
        self._roi_status_lbl = QLabel("")
        self._roi_status_lbl.setStyleSheet("color:#888;font-size:9px;border:none;")
        roi_in.addWidget(self._roi_status_lbl)
        viz_bar_layout.addWidget(self._roi_inline)

        # (ROI toolbar row removed — controls now inline in viz bar)

        # ── Image / frame viewer ──
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
        def zoom_wheel(event):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._pose_graphics.scale(factor, factor)
        self._pose_graphics.wheelEvent = zoom_wheel
        self._pose_scene = QGraphicsScene()
        self._pose_graphics.setScene(self._pose_scene)
        self._pose_graphics.installEventFilter(self)
        slay.addWidget(self._pose_graphics)

        # ── Playback control bar (shown for video mode) ──
        self._vp_ctrl_bar = QWidget()
        self._vp_ctrl_bar.setVisible(False)
        ctrl = QHBoxLayout(self._vp_ctrl_bar)
        ctrl.setContentsMargins(4, 2, 4, 2)
        ctrl.setSpacing(6)

        self._vp_time_lbl = QLabel("00:00 / 00:00")
        self._vp_time_lbl.setStyleSheet("color:#555;font-size:10px;border:none;min-width:80px;")
        self._vp_time_lbl.setAlignment(Qt.AlignCenter)
        ctrl.addWidget(self._vp_time_lbl)

        self._vp_slider = QSlider(Qt.Horizontal)
        self._vp_slider.setRange(0, 1000)
        self._vp_slider.setValue(0)
        self._vp_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#E0E0E0;height:5px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#0078D4;width:12px;height:12px;"
            "margin:-3px 0;border-radius:6px;}"
            "QSlider::sub-page:horizontal{background:#0078D4;border-radius:2px;}")
        self._vp_slider.sliderPressed.connect(self._on_slider_pressed)
        self._vp_slider.sliderReleased.connect(self._on_slider_released)
        self._vp_slider.valueChanged.connect(self._on_slider_value_changed)
        ctrl.addWidget(self._vp_slider, 1)

        fprev = QPushButton("⏮")
        fprev.setFixedSize(26, 22)
        fprev.setToolTip("Step back 10 frames")
        fprev.setStyleSheet(
            "QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:3px;font-size:10px;}"
            "QPushButton:hover{background:#E0E0E0;}")
        fprev.clicked.connect(lambda: self._step_frames(-10))
        ctrl.addWidget(fprev)

        fnext = QPushButton("⏭")
        fnext.setFixedSize(26, 22)
        fnext.setToolTip("Step forward 10 frames")
        fnext.setStyleSheet(
            "QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:3px;font-size:10px;}"
            "QPushButton:hover{background:#E0E0E0;}")
        fnext.clicked.connect(lambda: self._step_frames(10))
        ctrl.addWidget(fnext)

        slay.addWidget(self._vp_ctrl_bar)

        # ── Folder frame navigation bar (visible for image-folder sources) ──
        self._folder_ctrl_bar = QWidget()
        self._folder_ctrl_bar.setVisible(False)
        _fctrl = QHBoxLayout(self._folder_ctrl_bar)
        _fctrl.setContentsMargins(4, 2, 4, 2)
        _fctrl.setSpacing(6)

        self._folder_label = QLabel("Frame 0 / 0")
        self._folder_label.setStyleSheet("color:#555;font-size:10px;border:none;min-width:80px;")
        self._folder_label.setAlignment(Qt.AlignCenter)
        _fctrl.addWidget(self._folder_label)

        self._folder_slider = QSlider(Qt.Horizontal)
        self._folder_slider.setRange(0, 1)
        self._folder_slider.setValue(0)
        self._folder_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#E0E0E0;height:5px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#0078D4;width:12px;height:12px;"
            "margin:-3px 0;border-radius:6px;}"
            "QSlider::sub-page:horizontal{background:#0078D4;border-radius:2px;}")
        self._folder_slider.valueChanged.connect(self._on_folder_slider)
        _fctrl.addWidget(self._folder_slider, 1)

        _fprev = QPushButton("◀")
        _fprev.setFixedSize(26, 22)
        _fprev.setToolTip("Previous frame")
        _fprev.setStyleSheet(
            "QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:3px;font-size:10px;}"
            "QPushButton:hover{background:#E0E0E0;}")
        _fprev.clicked.connect(self._on_folder_prev)
        _fctrl.addWidget(_fprev)

        _fnext = QPushButton("▶")
        _fnext.setFixedSize(26, 22)
        _fnext.setToolTip("Next frame")
        _fnext.setStyleSheet(
            "QPushButton{background:#F0F0F0;border:1px solid #D0D0D0;border-radius:3px;font-size:10px;}"
            "QPushButton:hover{background:#E0E0E0;}")
        _fnext.clicked.connect(self._on_folder_next)
        _fctrl.addWidget(_fnext)

        slay.addWidget(self._folder_ctrl_bar)
        lay.addWidget(self._pose_samples_frame)

        # State
        self._sources = []
        self._source_idx = 0
        self._sample_images = []
        self._sample_idx = 0
        self._is_video_mode = False
        self._play_timer = None
        self._play_fps = 30
        self._slider_dragging = False

        return page

    # ── Event filter for keyboard shortcuts + raster resize ──

    def eventFilter(self, obj, event):
        if obj is self._pose_graphics and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key_Left:
                self._on_pose_prev()
                return True
            elif event.key() == Qt.Key_Right:
                self._on_pose_next()
                return True
            elif event.key() == Qt.Key_Space:
                self._on_play_pause()
                return True
        # Re-fit raster whenever the view is resized and a raster is loaded.
        # This catches layout-triggered resizes (page change, collapse_log,
        # splitter adjustment) reliably regardless of timing.
        if (hasattr(self, '_e2e_raster_view')
                and obj is self._e2e_raster_view
                and event.type() == QEvent.Type.Resize
                and self._e2e_raster_images
                and self._e2e_raster_scene.sceneRect().width() > 0):
            self._e2e_raster_view.fitInView(
                self._e2e_raster_scene.sceneRect(), Qt.IgnoreAspectRatio)
            self._update_e2e_raster_indicator()
        return super().eventFilter(obj, event)

    # ── Incremental results ──────────────────────────────────────────

    def on_partial_result(self, data):
        """Handle incremental source completion — show frames as they arrive."""
        if not data.get("is_partial"):
            return
        is_live = data.get("is_live", False)
        mode = data.get("mode", "")

        # ── Live camera mode: decode JPEG bytes directly (zero disk I/O) ──
        if is_live:
            # ── Behavior-only update (labels + live raster from auto-save) ──
            if data.get("is_behavior_update"):
                if "e2e_labels" in data and data["e2e_labels"]:
                    self._e2e_labels = {
                        k: data["e2e_labels"][k]
                        for k in sorted(data["e2e_labels"].keys(), key=int)}
                if "e2e_class_names" in data and data["e2e_class_names"]:
                    self._e2e_class_names = data["e2e_class_names"]
                # Store total_frames for indicator positioning
                if "total_frames" in data:
                    self._e2e_total_frames = data["total_frames"]
                # Update live raster plot if provided
                raster_plot = data.get("raster_plot", "")
                if raster_plot and Path(raster_plot).exists():
                    self._e2e_raster_images = [raster_plot]
                    self._e2e_raster_scene.clear()
                    pix = QPixmap(raster_plot)
                    if not pix.isNull():
                        self._e2e_raster_scene.addPixmap(pix)
                        self._e2e_raster_scene.setSceneRect(
                            self._e2e_raster_scene.itemsBoundingRect())
                        self._e2e_raster_view.resetTransform()
                        self._e2e_raster_view.fitInView(
                            self._e2e_raster_scene.sceneRect(), Qt.IgnoreAspectRatio)
                    # Recreate indicator line
                    from PySide6.QtGui import QPen, QColor as _Qc3
                    sr_h = self._e2e_raster_scene.sceneRect().height()
                    self._e2e_raster_indicator = self._e2e_raster_scene.addLine(
                        0, 2, 0, max(sr_h * 0.75, 10),
                        QPen(_Qc3(255, 30, 30), 3))
                    self._e2e_raster_indicator.setZValue(10)
                    self._update_e2e_raster_indicator()
                # Re-render E2E frame with updated labels (also in live mode)
                if self._result_stack.currentIndex() == 3:
                    if self._e2e_sample_images:
                        self._show_e2e_pose_frame()
                return

            # ── Frame update: decode JPEG bytes from signal ──
            jpeg_bytes = data.get("frame_jpeg")
            if jpeg_bytes is None:
                return
            self._is_live_mode = True

            # Decode JPEG bytes to QPixmap directly (no disk I/O)
            pix = QPixmap()
            pix.loadFromData(jpeg_bytes)
            self._live_raw_pix = QPixmap(pix)  # keep un-viz'd copy for mode switch

            # Cache trajectory data for viz modes
            traj_data = data.get("trajectory_data")
            if traj_data:
                self._trajectory_cache[0] = traj_data
                # Invalidate heatmap cache periodically (~1s = 15 frames at 15Hz)
                frame_idx = data.get("frame_idx", 0)
                if frame_idx % 15 == 0:
                    self._heatmap_cache.pop(0, None)
                # Update ROI stats periodically
                if self._viz_mode == "roi" and self._rois:
                    if frame_idx % 15 == 0:
                        self._compute_roi_stats()
                        self._rebuild_roi_handles()

            # Update behavior labels (E2E mode)
            if "e2e_labels" in data and data["e2e_labels"]:
                self._e2e_labels = {
                    k: data["e2e_labels"][k]
                    for k in sorted(data["e2e_labels"].keys(), key=int)}
            if "e2e_class_names" in data and data["e2e_class_names"]:
                self._e2e_class_names = data["e2e_class_names"]

            # FPS display
            fps = data.get("fps", 0)
            frame_idx = data.get("frame_idx", 0)

            if mode == "Pose":
                # Apply viz rendering (trail/heatmap/ROI) if active
                pix = self._render_live_viz(pix, 0, frame_idx)
                # Show on Pose page
                self._pose_samples_frame.setVisible(True)
                self._samples_idx_label.setText(f"{fps:.0f} FPS" if fps > 0 else "LIVE")
                self._display_pixmap_on_pose_scene(pix)
                self._e2e_live_frame_idx = frame_idx
                self._update_e2e_raster_indicator()
                # Show viz controls if trajectory available (matches E2E behavior)
                has_traj = 0 in self._trajectory_cache
                if self._viz_btn_bar is not None:
                    self._viz_btn_bar.setVisible(has_traj)
                if self._result_stack.currentIndex() != 1:
                    self._result_stack.setCurrentIndex(1)

            elif mode == "End-to-End":
                # Apply viz rendering
                pix = self._render_live_viz(pix, 0, frame_idx)
                # Paint behavior labels as overlay (color by mouse ID, not index)
                from PySide6.QtGui import QPainter, QColor as _QCo, QFont as _QF, QPen as _QPen
                # Mouse 1 = Red, Mouse 2 = Blue (matching YOLO _INSTANCE_COLORS)
                MOUSE_COLORS = {"0": _QCo(200, 30, 30), "1": _QCo(30, 80, 220),
                                "2": _QCo(30, 180, 80), "3": _QCo(220, 180, 30)}
                behaviors = self._get_e2e_behavior_parts()
                if behaviors:
                    p = QPainter(pix)
                    p.setRenderHint(QPainter.Antialiasing)
                    font = _QF("Segoe UI", 16, _QF.Bold)
                    p.setFont(font)
                    fm = p.fontMetrics()
                    gap = 6; th = fm.height() + 10
                    total_h = len(behaviors) * th + (len(behaviors) - 1) * gap
                    y = pix.height() - total_h - 10
                    for i, (mid, text) in enumerate(behaviors):
                        tw = fm.horizontalAdvance(text) + 20
                        color = MOUSE_COLORS.get(mid, _QCo(200, 200, 200))
                        p.fillRect(8, y, tw, th, color)
                        p.setPen(_QPen(_QCo(255, 255, 255, 100)))
                        p.drawRect(8, y, tw, th)
                        p.setPen(_QCo(255, 255, 255))
                        p.drawText(18, y + fm.ascent() + 4, text)
                        y += th + gap
                    p.end()
                # Show on E2E page
                if hasattr(self, '_e2e_scene_bg_item') and self._e2e_scene_bg_item is not None:
                    try:
                        self._e2e_pose_scene.removeItem(self._e2e_scene_bg_item)
                    except RuntimeError:
                        pass
                self._e2e_scene_bg_item = self._e2e_pose_scene.addPixmap(pix)
                self._e2e_scene_bg_item.setZValue(-1)
                # Rebuild ROI handles if in ROI mode
                if self._viz_mode == "roi" and self._rois and not self._roi_drawing:
                    self._rebuild_roi_handles()
                elif self._viz_mode != "roi":
                    self._clear_roi_scene_items()
                self._e2e_pose_scene.setSceneRect(self._e2e_pose_scene.itemsBoundingRect())
                self._e2e_pose_view.resetTransform()
                self._e2e_pose_view.fitInView(self._e2e_pose_scene.sceneRect(), Qt.KeepAspectRatio)
                self._e2e_frame_idx_label.setText(f"{fps:.0f} FPS" if fps > 0 else "LIVE")
                self._e2e_live_frame_idx = frame_idx
                self._update_e2e_raster_indicator()
                # Show viz controls
                has_traj = 0 in self._trajectory_cache
                if self._e2e_viz_btn_bar is not None:
                    self._e2e_viz_btn_bar.setVisible(has_traj)
                if self._result_stack.currentIndex() != 3:
                    self._result_stack.setCurrentIndex(3)
                    self.collapse_log()
            return

        # ── Batch mode (existing logic) ──
        if mode != "Pose":
            return
        new_sources = data.get("sources", [])
        if not new_sources:
            return

        if not hasattr(self, '_sources'):
            self._sources = []
        self._sources.extend(new_sources)

        # Pre-load trajectory data for new video sources
        for src in new_sources:
            if src.get("type") == "video" and src.get("trajectory_json"):
                src_idx = len(self._sources) - len(new_sources) + new_sources.index(src)
                self._ensure_trajectory_loaded(src_idx)

        in_e2e = (self._mode_combo is not None
                  and self._mode_combo.currentText() == "End-to-End")
        if in_e2e:
            # In E2E mode, just accumulate sources — don't switch pages
            # Accumulate frames for E2E
            for src in new_sources:
                frames = [p for p in src.get("frames", []) if Path(p).exists()]
                self._e2e_sample_images.extend(frames)
            self._e2e_source_idx = len(self._sources) - 1
            self._e2e_sample_idx = 0
            return

        if len(self._sources) == len(new_sources):
            self._source_idx = 0
            self._sample_idx = 0
            self._sample_images = []
            for src in self._sources:
                self._sample_images.extend(src.get("frames", []))
            self._pose_samples_frame.setVisible(True)
            self._apply_source_mode()
            self._show_current_sample()
            self._result_stack.setCurrentIndex(1)
        else:
            self._apply_source_mode()
            self._show_current_sample()

    # ── Pose result ───────────────────────────────────────────────────

    def _show_pose_result(self, result):
        self._stop_playback()
        sources = result.get("sources", [])
        # Filter to only sources with frames, resolve paths
        valid_sources = []
        for src in sources:
            frames = [p for p in src.get("frames", []) if Path(p).exists()]
            if frames:
                src["frames"] = frames
                valid_sources.append(src)
        self._sources = valid_sources
        self._source_idx = 0
        self._sample_idx = 0
        self._sample_images = []  # legacy flat list

        if not self._sources:
            self._pose_samples_frame.setVisible(False)
            return

        # Build flat list for backward compat in _show_current_sample
        for src in self._sources:
            self._sample_images.extend(src["frames"])

        self._pose_samples_frame.setVisible(True)

        # Determine mode from current source
        self._apply_source_mode()
        self._show_current_sample()

    def _current_frames(self):
        """Return frames list for the currently selected source."""
        if not self._sources or self._source_idx >= len(self._sources):
            return []
        return self._sources[self._source_idx].get("frames", [])

    def _apply_source_mode(self):
        """Update UI visibility based on current source type."""
        if not self._sources:
            return
        src = self._sources[self._source_idx]
        self._is_video_mode = (src.get("type") == "video")
        frames = src.get("frames", [])
        name = src.get("name", "Unknown")
        n_sources = len(self._sources)
        prefix = "Video" if self._is_video_mode else "Folder"
        self._pose_samples_label.setText(
            f"{prefix} {self._source_idx + 1}/{n_sources}: {name}")

        if self._is_video_mode:
            self._vp_play_btn.setVisible(True)
            self._vp_speed_combo.setVisible(True)
            self._vp_ctrl_bar.setVisible(True)
            self._vp_slider.setRange(0, max(1, len(frames) - 1))
            self._vp_slider.setValue(0)
            self._folder_ctrl_bar.setVisible(False)
            # Show viz controls for video + Pose/End-to-End
            mode = self._mode_combo.currentText() if self._mode_combo else ""
            show_viz = mode in ("Pose", "End-to-End")
            if self._viz_btn_bar is not None:
                self._viz_btn_bar.setVisible(show_viz)
            if not show_viz:
                self._viz_mode = "frame_only"
            self._update_viz_button_states()
        else:
            self._vp_play_btn.setVisible(False)
            self._vp_speed_combo.setVisible(False)
            self._vp_ctrl_bar.setVisible(False)
            if self._viz_btn_bar is not None:
                self._viz_btn_bar.setVisible(False)
            if self._roi_inline is not None:
                self._roi_inline.setVisible(False)
            if self._roi_e2e_inline is not None:
                self._roi_e2e_inline.setVisible(False)
            self._on_roi_tool(None)  # deactivate drawing
            self._viz_mode = "frame_only"
            n_frames = len(frames)
            if n_frames > 1:
                self._folder_ctrl_bar.setVisible(True)
                self._folder_slider.setRange(0, n_frames - 1)
                self._folder_slider.setValue(0)
                self._update_folder_label()
            else:
                self._folder_ctrl_bar.setVisible(False)
    # ── Viz mode handlers ─────────────────────────────────────────────

    def _on_viz_mode_changed(self, mode_key: str):
        """Switch visualization mode and refresh display."""
        # Deactivate ROI drawing if switching away from ROI mode
        if self._viz_mode == "roi" and mode_key != "roi":
            self._on_roi_tool(None)
            self._clear_roi_scene_items()
        self._viz_mode = mode_key
        self._update_viz_button_states()
        if mode_key != "frame_only":
            self._ensure_trajectory_loaded(self._source_idx)
        if mode_key == "global_traj":
            self._global_traj_cache.pop(self._source_idx, None)
        if mode_key == "heatmap":
            self._heatmap_cache.pop(self._source_idx, None)
        if self._roi_inline is not None:
            self._roi_inline.setVisible(mode_key == "roi")
        if self._is_live_mode:
            self._refresh_current_view()
        else:
            self._show_current_sample()

    def _update_viz_button_states(self):
        """Sync button checked states with current _viz_mode."""
        for mk, btn in self._viz_btns.items():
            btn.setChecked(mk == self._viz_mode)

    def _on_trail_length_changed(self, value: int):
        self._trail_length = value
        if self._viz_mode == "frame_trail":
            self._refresh_current_view()

    # ── ROI analysis ──────────────────────────────────────────────────

    def _on_roi_tool(self, tool: str):
        """Activate/deactivate ROI drawing tool."""
        # Pause playback during drawing
        if tool is not None:
            self._stop_playback()
            self._e2e_stop_play()
        if tool is None:
            self._roi_draw_tool = None
            self._uninstall_roi_drawing()
            self._roi_rect_btn.setChecked(False)
            self._roi_poly_btn.setChecked(False)
            if hasattr(self, '_e2e_roi_rect_btn'):
                self._e2e_roi_rect_btn.setChecked(False)
                self._e2e_roi_poly_btn.setChecked(False)
            self._roi_set_status("")
            return
        self._roi_draw_tool = tool
        if tool == "rect":
            self._roi_poly_btn.setChecked(False)
            if hasattr(self, '_e2e_roi_poly_btn'):
                self._e2e_roi_poly_btn.setChecked(False)
        else:
            self._roi_rect_btn.setChecked(False)
            if hasattr(self, '_e2e_roi_rect_btn'):
                self._e2e_roi_rect_btn.setChecked(False)
        self._roi_set_status(
            "Draw rect: drag on video" if tool == "rect"
            else "Polygon: click vertices, double-click to finish")
        self._install_roi_drawing()

    def _install_roi_drawing(self):
        """Install mouse event handlers for ROI drawing."""
        from PySide6.QtWidgets import QGraphicsView
        view = self._get_active_graphics_view()
        if view is None:
            return
        view.setDragMode(QGraphicsView.NoDrag)
        # Save originals
        self._roi_orig_mousePress = view.mousePressEvent
        self._roi_orig_mouseMove = view.mouseMoveEvent
        self._roi_orig_mouseRelease = view.mouseReleaseEvent
        self._roi_orig_mouseDouble = view.mouseDoubleClickEvent
        view.mousePressEvent = self._roi_mouse_press
        view.mouseMoveEvent = self._roi_mouse_move
        view.mouseReleaseEvent = self._roi_mouse_release
        view.mouseDoubleClickEvent = self._roi_mouse_double_click
        self._roi_drawing = False
        self._roi_current_pts = []

    def _uninstall_roi_drawing(self):
        """Restore original mouse event handlers."""
        from PySide6.QtWidgets import QGraphicsView
        view = self._get_active_graphics_view()
        if view is None:
            return
        view.setDragMode(QGraphicsView.ScrollHandDrag)
        if hasattr(self, '_roi_orig_mousePress'):
            view.mousePressEvent = self._roi_orig_mousePress
            view.mouseMoveEvent = self._roi_orig_mouseMove
            view.mouseReleaseEvent = self._roi_orig_mouseRelease
            view.mouseDoubleClickEvent = self._roi_orig_mouseDouble
        self._roi_drawing = False
        self._roi_current_pts = []
        # Remove preview item
        self._remove_roi_preview()

    def _get_active_graphics_view(self):
        """Return the currently active QGraphicsView."""
        if self._result_stack.currentIndex() == 3:
            return self._e2e_pose_view if hasattr(self, '_e2e_pose_view') else None
        return self._pose_graphics if hasattr(self, '_pose_graphics') else None

    def _get_active_scene(self):
        """Return the currently active QGraphicsScene."""
        if self._result_stack.currentIndex() == 3:
            return self._e2e_pose_scene if hasattr(self, '_e2e_pose_scene') else None
        return self._pose_scene if hasattr(self, '_pose_scene') else None

    def _roi_set_status(self, text: str):
        """Set ROI status label on the active page."""
        if self._result_stack.currentIndex() == 3 and hasattr(self, '_e2e_roi_status_lbl'):
            self._e2e_roi_status_lbl.setText(text)
        elif hasattr(self, '_roi_status_lbl'):
            self._roi_status_lbl.setText(text)

    def _roi_mouse_press(self, event):
        from PySide6.QtGui import QPen, QColor as _QCol, QBrush
        from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsPolygonItem
        scene = self._get_active_scene()
        view = self._get_active_graphics_view()
        if scene is None or view is None:
            return
        sp = view.mapToScene(event.position().toPoint())
        if self._roi_draw_tool == "rect":
            if not self._roi_drawing:
                # First click: set anchor corner
                self._roi_drawing = True
                self._roi_current_pts = [(sp.x(), sp.y())]
                self._remove_roi_preview()
                pen = QPen(_QCol(255, 200, 100), 2)
                self._roi_preview_item = QGraphicsRectItem(sp.x(), sp.y(), 0, 0)
                self._roi_preview_item.setPen(pen)
                self._roi_preview_item.setBrush(QBrush(_QCol(255, 200, 100, 40)))
                scene.addItem(self._roi_preview_item)
        elif self._roi_draw_tool == "polygon":
            if not self._roi_drawing:
                self._roi_drawing = True
                self._roi_current_pts = [(sp.x(), sp.y())]
                self._remove_roi_preview()
                pen = QPen(_QCol(100, 200, 255), 2)
                self._roi_preview_item = QGraphicsPolygonItem()
                self._roi_preview_item.setPen(pen)
                self._roi_preview_item.setBrush(QBrush(_QCol(100, 200, 255, 30)))
                scene.addItem(self._roi_preview_item)

    def _roi_mouse_move(self, event):
        from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsPolygonItem
        view = self._get_active_graphics_view()
        if view is None or not self._roi_drawing:
            return
        sp = view.mapToScene(event.position().toPoint())
        if self._roi_draw_tool == "rect" and self._roi_current_pts and isinstance(
                self._roi_preview_item, QGraphicsRectItem):
            x1, y1 = self._roi_current_pts[0]
            r = QGraphicsRectItem(min(x1, sp.x()), min(y1, sp.y()),
                                  abs(sp.x() - x1), abs(sp.y() - y1))
            self._roi_preview_item.setRect(r.rect())
        elif self._roi_draw_tool == "polygon" and isinstance(self._roi_preview_item, QGraphicsPolygonItem):
            from PySide6.QtGui import QPolygonF
            from PySide6.QtCore import QPointF
            pts = [QPointF(p[0], p[1]) for p in self._roi_current_pts]
            pts.append(sp)
            self._roi_preview_item.setPolygon(QPolygonF(pts))

    def _roi_mouse_release(self, event):
        view = self._get_active_graphics_view()
        if view is None or not self._roi_drawing:
            return
        sp = view.mapToScene(event.position().toPoint())
        if self._roi_draw_tool == "rect" and self._roi_current_pts:
            # Click-twice mode: first click sets corner, second click finalizes
            if len(self._roi_current_pts) == 1:
                x1, y1 = self._roi_current_pts[0]
                x2, y2 = sp.x(), sp.y()
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                    self._roi_drawing = False
                    self._remove_roi_preview()
                    self._add_roi("rect", [
                        (min(x1, x2), min(y1, y2)),
                        (max(x1, x2), max(y1, y2))])
                    self._roi_current_pts = []
                    self._roi_set_status(
                        f"{len(self._rois)} ROI(s)")
                else:
                    self._roi_set_status("Rect too small, try again")
        elif self._roi_draw_tool == "polygon":
            # Check snap-to-close: if near any existing vertex (<12px), close polygon
            sx, sy = sp.x(), sp.y()
            snap_dist = 12
            closed = False
            for i, (vx, vy) in enumerate(self._roi_current_pts):
                if ((sx - vx) ** 2 + (sy - vy) ** 2) < snap_dist ** 2:
                    if len(self._roi_current_pts) >= 3:
                        self._roi_drawing = False
                        self._remove_roi_preview()
                        self._add_roi("polygon", list(self._roi_current_pts))
                        self._roi_current_pts = []
                        self._roi_set_status(
                            f"{len(self._rois)} ROI(s)")
                        closed = True
                    break
            if not closed:
                self._roi_current_pts.append((sx, sy))

    def _roi_mouse_double_click(self, event):
        if self._roi_draw_tool == "polygon" and len(self._roi_current_pts) >= 3:
            self._roi_drawing = False
            self._remove_roi_preview()
            self._add_roi("polygon", list(self._roi_current_pts))
            self._roi_current_pts = []
            self._roi_set_status(
                f"{len(self._rois)} ROI(s) — draw or click Polygon to add vertices")

    def _get_roi_stats_pos(self, roi_idx, box_w, box_h, shape_bb):
        """Get stable position for ROI stats box.
        Once placed, position never changes. New ROIs offset if overlapping."""
        from PySide6.QtCore import QRectF
        if not hasattr(self, '_roi_stats_pos_cache'):
            self._roi_stats_pos_cache = {}
        # Return cached position if already fixed
        if roi_idx in self._roi_stats_pos_cache:
            return self._roi_stats_pos_cache[roi_idx]
        # Default: center of ROI
        bx = int(shape_bb.center().x() - box_w / 2)
        by = int(shape_bb.center().y() - box_h / 2)
        candidate = QRectF(bx, by, box_w, box_h)
        # Check against all EXISTING fixed positions
        existing = [QRectF(*v, box_w, box_h)
                    for k, v in self._roi_stats_pos_cache.items()]
        if not any(candidate.intersects(e) for e in existing):
            self._roi_stats_pos_cache[roi_idx] = (bx, by)
            return bx, by
        # Need offset — try multiple directions
        offsets = [(0, 20), (0, 40), (30, 20), (30, 40),
                   (0, 60), (30, 60), (-30, 40), (-30, 20),
                   (0, 80), (30, 80), (-30, 60)]
        for ox, oy in offsets:
            test = QRectF(bx + ox, by + oy, box_w, box_h)
            if not any(test.intersects(e) for e in existing):
                pos = (int(bx + ox), int(by + oy))
                self._roi_stats_pos_cache[roi_idx] = pos
                return pos
        # Fallback: stack below
        pos = (bx, by + roi_idx * (box_h + 6))
        self._roi_stats_pos_cache[roi_idx] = pos
        return pos

    def _clear_roi_stats_positions(self):
        """Reset placed stats box cache (call when all ROIs deleted)."""
        self._roi_stats_pos_cache = {}

    def _remove_roi_stats_position(self, roi_idx):
        """Remove cached position for deleted ROI, re-key remaining."""
        if hasattr(self, '_roi_stats_pos_cache'):
            self._roi_stats_pos_cache.pop(roi_idx, None)
            new_cache = {}
            for old_idx, pos in self._roi_stats_pos_cache.items():
                new_idx = old_idx if old_idx < roi_idx else old_idx - 1
                new_cache[new_idx] = pos
            self._roi_stats_pos_cache = new_cache

    def _clear_roi_scene_items(self):
        """Remove all ROI shape items from the active scene."""
        scene = self._get_active_scene()
        if scene is None:
            return
        for item in getattr(self, '_roi_items', []):
            try:
                scene.removeItem(item)
            except RuntimeError:
                pass
        self._roi_items = []

    def _remove_roi_preview(self):
        scene = self._get_active_scene()
        if scene and self._roi_preview_item:
            try:
                scene.removeItem(self._roi_preview_item)
            except RuntimeError:
                pass  # C++ object already deleted
            self._roi_preview_item = None

    def _rebuild_roi_handles(self):
        """Rebuild ROI shape items on the active scene (fixed shapes, no handles)."""
        scene = self._get_active_scene()
        if scene is None:
            return
        # Remove old items
        for item in getattr(self, '_roi_items', []):
            try:
                scene.removeItem(item)
            except RuntimeError:
                pass
        self._roi_items = []
        # Create shape items — newer ROIs on top (higher Z)
        for ri, roi in enumerate(self._rois):
            item = _RoiShapeItem(ri, roi["type"], roi["points"], roi["color"], self)
            item.setZValue(5 + ri)  # newer = higher Z = click priority
            scene.addItem(item)
            self._roi_items.append(item)

    def _add_roi(self, roi_type: str, points: list):
        """Finalize an ROI: store data, select it, rebuild scene."""
        ci = len(self._rois) % len(self._roi_colors)
        color = self._roi_colors[ci]
        roi = {
            "type": roi_type,
            "points": points,
            "label": f"ROI {len(self._rois) + 1}",
            "color": color,
        }
        self._rois.append(roi)
        self._roi_selected_idx = len(self._rois) - 1
        self._compute_roi_stats()
        self._rebuild_roi_handles()

    def _on_roi_delete(self):
        """Delete the currently selected ROI."""
        if not self._rois:
            return
        if self._roi_selected_idx < 0 or self._roi_selected_idx >= len(self._rois):
            self._roi_set_status("Click an ROI first to select it for deletion")
            return
        idx = self._roi_selected_idx
        self._rois.pop(idx)
        self._roi_stats.pop(idx, None)
        new_stats = {}
        for old_idx, stats in self._roi_stats.items():
            new_idx = old_idx if old_idx < idx else old_idx - 1
            new_stats[new_idx] = stats
        self._roi_stats = new_stats
        self._roi_selected_idx = -1
        self._remove_roi_stats_position(idx)
        self._clear_roi_scene_items()
        self._roi_set_status(f"{len(self._rois)} ROI(s)")
        self._compute_roi_stats()
        self._refresh_current_view()

    def _on_roi_clear_all(self):
        """Remove all ROIs."""
        self._rois.clear()
        self._roi_stats.clear()
        self._roi_selected_idx = -1
        self._clear_roi_stats_positions()
        self._clear_roi_scene_items()
        self._roi_set_status("")
        self._refresh_current_view()

    def _compute_roi_stats(self):
        """Compute entry/exit counts independently for all ROIs and mice."""
        self._roi_stats = {}
        if not self._rois:
            return
        if self._result_stack.currentIndex() == 3:
            src_idx = self._e2e_source_idx
        else:
            src_idx = self._source_idx
        self._ensure_trajectory_loaded(src_idx)
        traj = self._trajectory_cache.get(src_idx)
        if not traj:
            return
        mice = traj.get("mice", {})
        for ri, roi in enumerate(self._rois):
            self._roi_stats[ri] = {}
            for mid, mdata in mice.items():
                trajectory = mdata.get("trajectory", [])
                entries = 0; exits = 0; frames_inside = 0
                prev_inside = False
                for p in trajectory:
                    _, cx, cy = p
                    inside = self._point_in_roi(cx, cy, roi["type"], roi["points"])
                    if inside:
                        frames_inside += 1
                        if not prev_inside:
                            entries += 1
                    else:
                        if prev_inside:
                            exits += 1
                    prev_inside = inside
                self._roi_stats[ri][mid] = {
                    "entries": entries, "exits": exits,
                    "frames_inside": frames_inside,
                }

    @staticmethod
    def _point_in_roi(px: float, py: float, roi_type: str,
                      points: list) -> bool:
        """Test if a point is inside the ROI."""
        if roi_type == "rect":
            return (points[0][0] <= px <= points[1][0] and
                    points[0][1] <= py <= points[1][1])
        else:
            # Polygon: ray-casting algorithm
            n = len(points)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = points[i]
                xj, yj = points[j]
                if ((yi > py) != (yj > py)) and \
                   (px < (xj - xi) * (py - yi) / max(yj - yi, 0.001) + xi):
                    inside = not inside
                j = i
            return inside

    def _on_export_roi(self):
        """Export ROI statistics as CSV."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        import csv as _csv
        if not self._rois or not self._roi_stats:
            QMessageBox.warning(self, "No Data", "No ROI data to export.")
            return
        src_name = ""
        if hasattr(self, '_sources') and self._sources:
            idx = self._source_idx
            if self._result_stack.currentIndex() == 3:
                idx = self._e2e_source_idx
            if idx < len(self._sources):
                src_name = Path(self._sources[idx].get("name", "")).stem
        if not src_name:
            src_name = "roi"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export ROI Statistics", f"{src_name}_roi_stats.csv",
            "CSV Files (*.csv);;All Files (*)")
        if not file_path:
            return
        rows = []
        for ri, roi in enumerate(self._rois):
            stats = self._roi_stats.get(ri, {})
            roi_label = roi.get("label", f"ROI {ri + 1}")
            for mid, s in sorted(stats.items(), key=lambda x: int(x[0])):
                label = f"Mouse {int(mid) + 1}"
                rows.append([ri, roi_label, mid, label,
                            s["entries"], s["exits"], s["frames_inside"]])
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["roi_id", "roi_label", "mouse_id", "mouse_label",
                        "entries", "exits", "frames_inside"])
            w.writerows(rows)
        QMessageBox.information(
            self, "Export Complete",
            f"ROI statistics exported:\n{file_path}\n\n"
            f"{len(self._rois)} ROI(s), {len(rows)} data rows.")

    def _on_export_trajectory(self):
        """Export data for the current source based on viz mode."""
        from PySide6.QtWidgets import QFileDialog
        if self._result_stack.currentIndex() == 3:
            src_idx = self._e2e_source_idx
        else:
            src_idx = self._source_idx
        # Default filename
        src_name = ""
        if self._sources and src_idx < len(self._sources):
            src_name = Path(self._sources[src_idx].get("name", "")).stem
        if not src_name:
            src_name = "data"
        # Frame mode: export keypoint data
        if self._viz_mode == "frame_only":
            self._ensure_trajectory_loaded(src_idx)
            default_name = f"{src_name}_keypoints.csv"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Keypoint Data", default_name,
                "CSV Files (*.csv);;All Files (*)")
            if not file_path:
                return
            self._export_keypoints_csv(file_path, src_idx)
            return
        # Other modes need trajectory data
        self._ensure_trajectory_loaded(src_idx)
        traj = self._trajectory_cache.get(src_idx)
        if not traj:
            return
        if self._viz_mode == "global_traj":
            # Export the rendered trajectory image
            default_name = f"{src_name}_global_trajectory.png"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Global Trajectory Image", default_name,
                "PNG Image (*.png);;All Files (*)")
            if not file_path:
                return
            self._export_global_image(file_path, src_idx)
            return
        if self._viz_mode == "roi":
            self._on_export_roi()
            return
        if self._viz_mode == "heatmap":
            default_name = f"{src_name}_heatmap.csv"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Heatmap Data", default_name,
                "CSV Files (*.csv);;All Files (*)")
            if not file_path:
                return
            self._export_heatmap_csv(file_path, traj)
        else:
            default_name = f"{src_name}_trajectory.csv"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Trajectory Data", default_name,
                "CSV Files (*.csv);;All Files (*)")
            if not file_path:
                return
            self._export_trajectory_csv(file_path, traj)

    def _export_keypoints_csv(self, file_path: str, src_idx: int):
        """Export keypoint data from tracking_data.json."""
        import csv as _csv, json as _json
        from PySide6.QtWidgets import QMessageBox
        # Load tracking data (keypoints + box)
        src = self._sources[src_idx] if src_idx < len(self._sources) else {}
        tk_path = src.get("tracking_json", "")
        if not tk_path or not Path(tk_path).exists():
            QMessageBox.warning(self, "No Data",
                                "Keypoint data not available for this source.")
            return
        with open(tk_path, "r", encoding="utf-8") as f:
            tracking = _json.load(f)
        rows = []
        for frame_key in sorted(tracking.keys(), key=int):
            for inst in tracking[frame_key]:
                kps = inst.get("keypoints", [])
                bc = inst.get("box_center")
                row = [int(frame_key), inst["slot"], inst["label"]]
                if bc:
                    row.extend([f"{bc[0]:.2f}", f"{bc[1]:.2f}"])
                else:
                    row.extend(["", ""])
                for ki, kp in enumerate(kps):
                    row.extend([f"{kp[0]:.4f}", f"{kp[1]:.4f}",
                                f"{kp[2]:.4f}"])
                rows.append(row)
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            # Build header
            n_kps = max((len(inst.get("keypoints", []))
                        for v in tracking.values() for inst in v), default=0)
            header = ["frame", "mouse_id", "mouse_label", "center_x", "center_y"]
            for ki in range(n_kps):
                header.extend([f"kp{ki}_x", f"kp{ki}_y", f"kp{ki}_conf"])
            w.writerow(header)
            w.writerows(rows)
        QMessageBox.information(
            self, "Export Complete",
            f"Keypoint data exported:\n{file_path}\n\n"
            f"{len(rows)} rows, {len(tracking)} frames.")

    def _export_global_image(self, file_path: str, src_idx: int):
        """Export the global trajectory pixmap as PNG."""
        from PySide6.QtWidgets import QMessageBox
        # Force re-render to get the full base image (no time indicator)
        self._global_traj_cache.pop(src_idx, None)
        pix = self._render_global_traj(src_idx, 10**9)
        if pix.isNull():
            QMessageBox.warning(self, "Export Failed",
                                "Could not render global trajectory image.")
            return
        pix.save(file_path, "PNG")
        QMessageBox.information(
            self, "Export Complete",
            f"Global trajectory image exported:\n{file_path}")

    def _export_trajectory_csv(self, file_path: str, traj: dict):
        """Export trajectory data as CSV."""
        import csv as _csv
        mice = traj.get("mice", {})
        all_rows = []
        for mid, mdata in sorted(mice.items(), key=lambda x: int(x[0])):
            label = mdata.get("label", f"Mouse {mid}")
            for p in mdata.get("trajectory", []):
                all_rows.append([p[0], mid, label, f"{p[1]:.2f}", f"{p[2]:.2f}"])
        all_rows.sort(key=lambda r: (r[0], r[1]))
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["frame", "mouse_id", "mouse_label", "center_x", "center_y"])
            w.writerows(all_rows)
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "Export Complete",
            f"Trajectory data exported:\n{file_path}\n\n"
            f"{len(all_rows)} rows, {len(mice)} mice.")

    def _export_heatmap_csv(self, file_path: str, traj: dict):
        """Export heatmap density as CSV (10px cell grid)."""
        import csv as _csv, numpy as _np
        fw = traj.get("frame_width", 640)
        fh = traj.get("frame_height", 480)
        rows, cols = fh // 10, fw // 10  # 10px cells for export
        counts = _np.zeros((rows, cols), dtype=int)
        mice = traj.get("mice", {})
        for mdata in mice.values():
            for p in mdata.get("trajectory", []):
                _, cx, cy = p
                r = min(int(cy / fh * rows), rows - 1)
                c = min(int(cx / fw * cols), cols - 1)
                counts[r, c] += 1
        cell_w = fw / cols
        cell_h = fh / rows
        all_rows = []
        for r in range(rows):
            for c in range(cols):
                all_rows.append([
                    r, c,
                    f"{r * cell_h:.0f}", f"{(r + 1) * cell_h:.0f}",
                    f"{c * cell_w:.0f}", f"{(c + 1) * cell_w:.0f}",
                    int(counts[r, c])
                ])
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["row", "col", "y_from", "y_to", "x_from", "x_to", "count"])
            w.writerows(all_rows)
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(
            self, "Export Complete",
            f"Heatmap data exported:\n{file_path}\n\n"
            f"Grid: {rows}×{cols}, total visits: {int(counts.sum())}.")

    def _on_e2e_viz_mode_changed(self, mode_key: str):
        """Switch E2E viz mode and refresh."""
        if self._viz_mode == "roi" and mode_key != "roi":
            self._on_roi_tool(None)
            self._clear_roi_scene_items()
        self._viz_mode = mode_key
        for mk, btn in self._e2e_viz_btns.items():
            btn.setChecked(mk == mode_key)
        for mk, btn in self._viz_btns.items():
            btn.setChecked(mk == mode_key)
        if mode_key != "frame_only":
            self._ensure_trajectory_loaded(self._e2e_source_idx)
        if mode_key == "global_traj":
            self._global_traj_cache.pop(self._e2e_source_idx, None)
        if mode_key == "heatmap":
            self._heatmap_cache.clear()
        if self._roi_e2e_inline is not None:
            self._roi_e2e_inline.setVisible(mode_key == "roi")
        self._show_e2e_pose_frame()

    def _refresh_current_view(self):
        """Refresh the active page (Pose or E2E)."""
        if self._is_live_mode and self._live_raw_pix is not None:
            # Live mode: re-apply viz to cached raw frame
            frame_idx = self._e2e_live_frame_idx
            pix = self._render_live_viz(QPixmap(self._live_raw_pix), 0, frame_idx)
            if self._result_stack.currentIndex() == 3:
                # E2E: paint behavior labels
                from PySide6.QtGui import QPainter, QColor as _QCo, QFont as _QF, QPen as _QPen
                MOUSE_COLORS = {"0": _QCo(200, 30, 30), "1": _QCo(30, 80, 220),
                                "2": _QCo(30, 180, 80), "3": _QCo(220, 180, 30)}
                behaviors = self._get_e2e_behavior_parts()
                if behaviors:
                    p = QPainter(pix); p.setRenderHint(QPainter.Antialiasing)
                    font = _QF("Segoe UI", 16, _QF.Bold); p.setFont(font)
                    fm = p.fontMetrics(); gap = 6; th = fm.height() + 10
                    total_h = len(behaviors) * th + (len(behaviors) - 1) * gap
                    y = pix.height() - total_h - 10
                    for mid, text in behaviors:
                        tw = fm.horizontalAdvance(text) + 20
                        color = MOUSE_COLORS.get(mid, _QCo(200, 200, 200))
                        p.fillRect(8, y, tw, th, color)
                        p.setPen(_QPen(_QCo(255, 255, 255, 100))); p.drawRect(8, y, tw, th)
                        p.setPen(_QCo(255, 255, 255)); p.drawText(18, y + fm.ascent() + 4, text)
                        y += th + gap
                    p.end()
                # Update E2E scene with ROI handles
                if hasattr(self, '_e2e_scene_bg_item') and self._e2e_scene_bg_item is not None:
                    try:
                        self._e2e_pose_scene.removeItem(self._e2e_scene_bg_item)
                    except RuntimeError:
                        pass
                self._e2e_scene_bg_item = self._e2e_pose_scene.addPixmap(pix)
                self._e2e_scene_bg_item.setZValue(-1)
                if self._viz_mode == "roi" and self._rois and not self._roi_drawing:
                    self._rebuild_roi_handles()
                elif self._viz_mode != "roi":
                    self._clear_roi_scene_items()
                self._e2e_pose_scene.setSceneRect(self._e2e_pose_scene.itemsBoundingRect())
                self._e2e_pose_view.resetTransform()
                self._e2e_pose_view.fitInView(self._e2e_pose_scene.sceneRect(), Qt.KeepAspectRatio)
            else:
                self._display_pixmap_on_pose_scene(pix)
        elif self._result_stack.currentIndex() == 3:
            self._show_e2e_pose_frame()
        else:
            self._show_current_sample()

    def _ensure_trajectory_loaded(self, source_idx: int):
        """Lazy-load trajectory.json for the given source index."""
        import json as _json
        if source_idx in self._trajectory_cache:
            return
        if not self._sources or source_idx >= len(self._sources):
            return
        src = self._sources[source_idx]
        traj_path = src.get("trajectory_json", "")
        if traj_path and Path(traj_path).exists():
            try:
                with open(traj_path, "r", encoding="utf-8") as f:
                    self._trajectory_cache[source_idx] = _json.load(f)
            except Exception:
                pass

    # ── Viz rendering methods ─────────────────────────────────────────

    def _render_viz_frame(self, frame_path: str, source_idx: int,
                          frame_number: int):
        """Dispatch to the active visualization renderer."""
        if self._viz_mode == "frame_only" or source_idx not in self._trajectory_cache:
            return QPixmap(frame_path)
        if self._viz_mode == "frame_trail":
            return self._render_trail_frame(frame_path, source_idx, frame_number)
        if self._viz_mode == "global_traj":
            pix = self._render_global_traj(source_idx, frame_number)
            if pix.isNull():
                return QPixmap(frame_path)  # fallback to frame
            return pix
        if self._viz_mode == "heatmap":
            return self._render_heatmap_frame(frame_path, source_idx)
        if self._viz_mode == "roi":
            return self._render_roi_frame(frame_path, source_idx, frame_number)
        return QPixmap(frame_path)

    def _render_live_viz(self, pix, source_idx: int, frame_number: int):
        """Apply active visualization overlay to an existing QPixmap (live mode).
        Unlike _render_viz_frame, this takes a pixmap directly instead of a file path."""
        if self._viz_mode == "frame_only" or source_idx not in self._trajectory_cache:
            return pix
        if self._viz_mode == "frame_trail":
            return self._render_trail_on_pix(pix, source_idx, frame_number)
        if self._viz_mode == "global_traj":
            pix2 = self._render_global_traj(source_idx, frame_number)
            if pix2.isNull():
                return pix  # fallback to camera frame
            return pix2
        if self._viz_mode == "heatmap":
            return self._render_heatmap_overlay(pix, source_idx)
        if self._viz_mode == "roi":
            return self._render_roi_overlay(pix, source_idx, frame_number)
        return pix

    def _render_trail_on_pix(self, pix, source_idx: int, frame_number: int):
        """Render trajectory trail overlay on existing pixmap (no disk load)."""
        from PySide6.QtGui import QPainter, QPen, QColor as _QCol, QFont as _QF
        from PySide6.QtCore import QPointF
        traj = self._trajectory_cache.get(source_idx)
        if not traj:
            return pix
        diag = (pix.width() ** 2 + pix.height() ** 2) ** 0.5
        scale = max(1.0, diag / 800.0)
        result = QPixmap(pix)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)
        mice = traj.get("mice", {})
        for mid, mdata in mice.items():
            pts = mdata.get("trajectory", [])
            visible = [p for p in pts if p[0] <= frame_number][-self._trail_length:]
            if len(visible) < 2:
                if visible:
                    cx, cy = visible[-1][1], visible[-1][2]
                    bgr = mdata.get("color_bgr", [0, 0, 255])
                    color = _QCol(int(bgr[2]), int(bgr[1]), int(bgr[0]), 220)
                    painter.setPen(QPen(_QCol(255, 255, 255), 2))
                    painter.setBrush(color)
                    painter.drawEllipse(QPointF(cx, cy), 7, 7)
                continue
            bgr = mdata.get("color_bgr", [0, 0, 255])
            base_color = _QCol(int(bgr[2]), int(bgr[1]), int(bgr[0]))
            n_seg = len(visible) - 1
            for si in range(n_seg):
                alpha = int(30 + 200 * (si + 1) / n_seg)
                pen_w = max(int(2 * scale), int(6 * scale * (si + 1) / n_seg))
                seg_color = _QCol(base_color.red(), base_color.green(),
                                  base_color.blue(), alpha)
                painter.setPen(QPen(seg_color, pen_w, Qt.PenStyle.SolidLine,
                                    Qt.PenCapStyle.RoundCap))
                painter.drawLine(
                    QPointF(visible[si][1], visible[si][2]),
                    QPointF(visible[si + 1][1], visible[si + 1][2]))
            cx, cy = visible[-1][1], visible[-1][2]
            painter.setPen(QPen(_QCol(255, 255, 255), int(2 * scale)))
            painter.setBrush(_QCol(base_color.red(), base_color.green(),
                                   base_color.blue(), 240))
            painter.drawEllipse(QPointF(cx, cy), int(7 * scale), int(7 * scale))
            label = mdata.get("label", f"Mouse {mid}")
            font = _QF("Segoe UI", max(9, int(11 * scale)), _QF.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()
            lw = fm.horizontalAdvance(label) + 8
            lh = fm.height() + 4
            lx, ly = int(cx) + int(12 * scale), int(cy) - lh
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_QCol(0, 0, 0, 140))
            painter.drawRoundedRect(lx - 2, ly, lw, lh, 4, 4)
            painter.setPen(_QCol(255, 255, 255))
            painter.drawText(lx + 2, ly + fm.ascent(), label)
        painter.end()
        return result

    def _render_heatmap_overlay(self, pix, source_idx: int):
        """Render pixel-level heatmap overlay (Gaussian-smoothed density)."""
        from PySide6.QtGui import QPainter, QColor as _QCol, QImage as _QImg
        import numpy as _np
        try:
            from scipy.ndimage import gaussian_filter as _gauss
        except ImportError:
            from cv2 import GaussianBlur as _gauss_impl
            def _gauss(arr, sigma):
                import cv2
                return cv2.GaussianBlur(arr.astype(_np.float32), (0, 0), sigma)
        traj = self._trajectory_cache.get(source_idx)
        if not traj:
            return pix
        fw, fh = pix.width(), pix.height()
        cache_key = (source_idx, fw, fh)
        if cache_key in self._heatmap_cache:
            overlay = self._heatmap_cache[cache_key]
        else:
            # Build pixel accumulation array
            acc = _np.zeros((fh, fw), dtype=_np.float64)
            mice = traj.get("mice", {})
            for mdata in mice.values():
                for p in mdata.get("trajectory", []):
                    _, cx, cy = p
                    px, py = int(round(cx)), int(round(cy))
                    if 0 <= px < fw and 0 <= py < fh:
                        acc[py, px] += 1
            # Gaussian smooth
            sigma = max(fw, fh) / 200.0  # adaptive sigma
            acc_smooth = _gauss(acc, sigma=sigma)
            if acc_smooth.max() > 0:
                acc_smooth /= acc_smooth.max()
            # Apply colormap
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("jet")
            colored = cmap(acc_smooth)  # (fh, fw, 4) RGBA
            # Build QImage
            overlay = _QImg(fw, fh, _QImg.Format.Format_ARGB32)
            overlay.fill(_QCol(0, 0, 0, 0).rgba())
            # Only render pixels with non-zero density
            mask = acc_smooth > 0.001
            ys, xs = _np.where(mask)
            for y, x in zip(ys, xs):
                rgba = colored[y, x]
                a = int(rgba[3] * 180)  # alpha cap at 180/255
                fc = _QCol(int(rgba[0] * 255), int(rgba[1] * 255),
                           int(rgba[2] * 255), a)
                overlay.setPixelColor(x, y, fc)
            self._heatmap_cache[cache_key] = overlay
        result = QPixmap(pix)
        painter = QPainter(result)
        painter.drawImage(0, 0, overlay)
        painter.end()
        return result

    def _render_roi_overlay(self, pix, source_idx: int, frame_number: int):
        """Render ROI overlays on existing pixmap."""
        from PySide6.QtGui import QPainter, QPen, QColor as _QCol
        from PySide6.QtCore import QPointF
        result = QPixmap(pix)
        traj = self._trajectory_cache.get(source_idx)
        if traj:
            painter = QPainter(result)
            painter.setRenderHint(QPainter.Antialiasing)
            mice = traj.get("mice", {})
            for mid, mdata in mice.items():
                pts = mdata.get("trajectory", [])
                cur = None
                for p in pts:
                    if p[0] <= frame_number:
                        cur = p
                    else:
                        break
                if cur is None:
                    continue
                cx, cy = cur[1], cur[2]
                bgr = mdata.get("color_bgr", [0, 0, 255])
                pc = _QCol(int(bgr[2]), int(bgr[1]), int(bgr[0]))
                painter.setPen(QPen(pc, 3))
                painter.setBrush(pc)
                painter.drawEllipse(QPointF(cx, cy), 5, 5)
                inside_rois = []
                for ri, roi in enumerate(self._rois):
                    if self._point_in_roi(cx, cy, roi["type"], roi["points"]):
                        inside_rois.append(roi.get("label", f"ROI {ri + 1}"))
                if inside_rois:
                    label = f"{mdata.get('label', '')} in {','.join(inside_rois)}"
                    painter.setPen(QPen(_QCol(0, 0, 0, 180), 4))
                    painter.drawText(int(cx) + 10, int(cy) - 8, label)
                    painter.setPen(_QCol(255, 255, 255))
                    painter.drawText(int(cx) + 10, int(cy) - 8, label)
            painter.end()
        return result

    def _display_pixmap_on_pose_scene(self, pix):
        """Replace the background pixmap on the Pose QGraphicsScene (live mode)."""
        if hasattr(self, '_scene_bg_item') and self._scene_bg_item is not None:
            try:
                self._pose_scene.removeItem(self._scene_bg_item)
            except RuntimeError:
                pass
        self._scene_bg_item = self._pose_scene.addPixmap(pix)
        self._scene_bg_item.setZValue(-1)
        # Rebuild ROI handles if in ROI mode (stats labels on shapes)
        if self._viz_mode == "roi" and self._rois and not self._roi_drawing:
            self._rebuild_roi_handles()
        elif self._viz_mode != "roi":
            self._clear_roi_scene_items()
        self._pose_scene.setSceneRect(self._pose_scene.itemsBoundingRect())
        self._pose_graphics.resetTransform()
        self._pose_graphics.fitInView(self._pose_scene.sceneRect(), Qt.KeepAspectRatio)

    def _render_trail_frame(self, frame_path: str, source_idx: int,
                            frame_number: int):
        """Overlay fading trajectory trail on the frame using QPainter."""
        from PySide6.QtGui import QPainter, QPen, QColor as _QCol, QFont as _QF
        from PySide6.QtCore import QPointF
        pix = QPixmap(frame_path)
        if pix.isNull():
            return pix
        traj = self._trajectory_cache.get(source_idx)
        if not traj:
            return pix
        # Scale pen sizes by frame size (reference: diagonal of 640×480 ≈ 800)
        diag = (pix.width() ** 2 + pix.height() ** 2) ** 0.5
        scale = max(1.0, diag / 800.0)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        mice = traj.get("mice", {})
        for mid, mdata in mice.items():
            pts = mdata.get("trajectory", [])
            visible = [p for p in pts if p[0] <= frame_number][-self._trail_length:]
            if len(visible) < 2:
                if visible:
                    cx, cy = visible[-1][1], visible[-1][2]
                    bgr = mdata.get("color_bgr", [0, 0, 255])
                    color = _QCol(int(bgr[2]), int(bgr[1]), int(bgr[0]), 220)
                    painter.setPen(QPen(_QCol(255, 255, 255), 2))
                    painter.setBrush(color)
                    painter.drawEllipse(QPointF(cx, cy), 7, 7)
                continue
            bgr = mdata.get("color_bgr", [0, 0, 255])
            base_color = _QCol(int(bgr[2]), int(bgr[1]), int(bgr[0]))
            n_seg = len(visible) - 1
            for si in range(n_seg):
                alpha = int(30 + 200 * (si + 1) / n_seg)
                pen_w = max(int(2 * scale), int(6 * scale * (si + 1) / n_seg))
                seg_color = _QCol(base_color.red(), base_color.green(),
                                  base_color.blue(), alpha)
                painter.setPen(QPen(seg_color, pen_w, Qt.PenStyle.SolidLine,
                                    Qt.PenCapStyle.RoundCap))
                painter.drawLine(
                    QPointF(visible[si][1], visible[si][2]),
                    QPointF(visible[si + 1][1], visible[si + 1][2]))
            # Current position dot (matching Global mode size)
            cx, cy = visible[-1][1], visible[-1][2]
            painter.setPen(QPen(_QCol(255, 255, 255), 2))
            painter.setBrush(_QCol(base_color.red(), base_color.green(),
                                   base_color.blue(), 240))
            painter.drawEllipse(QPointF(cx, cy), 7, 7)
            # Label with background
            label = mdata.get("label", f"Mouse {mid}")
            font = _QF("Segoe UI", max(9, int(11 * scale)), _QF.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()
            lw = fm.horizontalAdvance(label) + 8
            lh = fm.height() + 4
            lx, ly = int(cx) + int(12 * scale), int(cy) - lh
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(_QCol(0, 0, 0, 140))
            painter.drawRoundedRect(lx - 2, ly, lw, lh, 4, 4)
            painter.setPen(_QCol(255, 255, 255))
            painter.drawText(lx + 2, ly + fm.ascent(), label)
        painter.end()
        return pix

    def _render_global_traj(self, source_idx: int, frame_number: int):
        """Render full trajectory map with current-frame indicator."""
        from PySide6.QtGui import QPainter, QPen, QColor as _QCol2, QImage as _QImg
        from PySide6.QtCore import QPointF
        import logging as _log
        traj = self._trajectory_cache.get(source_idx)
        if not traj:
            return QPixmap()
        if source_idx in self._global_traj_cache:
            base = self._global_traj_cache[source_idx]
        else:
            try:
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import io as _io
                fw = traj.get("frame_width", 640)
                fh = traj.get("frame_height", 480)
                if fw <= 0 or fh <= 0:
                    return QPixmap()
                dpi = 150
                fig_w = 8
                fig_h = max(2, fig_w * fh / fw)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                ax.set_facecolor("white")
                ax.set_xlim(0, fw)
                ax.set_ylim(fh, 0)
                mice = traj.get("mice", {})
                for mid, mdata in mice.items():
                    pts = mdata.get("trajectory", [])
                    if len(pts) < 2:
                        continue
                    xs = [p[1] for p in pts]
                    ys = [p[2] for p in pts]
                    bgr = mdata.get("color_bgr", [0, 0, 255])
                    color = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
                    ax.plot(xs, ys, color=color, linewidth=1.2,
                            label=mdata.get("label", f"Mouse {mid}"),
                            alpha=0.9)
                ax.legend(fontsize=8, loc="upper right",
                          facecolor="white", edgecolor="#AAA",
                          labelcolor="#333")
                ax.set_title("Global Trajectory Map", color="#333",
                             fontsize=11)
                ax.tick_params(colors="#666", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color("#CCC")
                fig.tight_layout()
                buf = _io.BytesIO()
                fig.savefig(buf, format="png", dpi=dpi, facecolor="white")
                plt.close(fig)
                buf.seek(0)
                data = buf.read()
                if not data:
                    return QPixmap()
                img = _QImg.fromData(data)
                if img.isNull():
                    return QPixmap()
                base = QPixmap.fromImage(img)
                if base.isNull():
                    return QPixmap()
                self._global_traj_cache[source_idx] = base
            except Exception as e:
                _log.getLogger("gui.inference").warning(
                    f"Global trajectory render failed: {e}")
                return QPixmap()
        if base is None or base.isNull():
            return QPixmap()
        result = QPixmap(base)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)
        fw = traj.get("frame_width", 640)
        fh = traj.get("frame_height", 480)
        scale_x = base.width() / max(fw, 1)
        scale_y = base.height() / max(fh, 1)
        if scale_x <= 0 or scale_y <= 0:
            painter.end()
            return result
        mice = traj.get("mice", {})
        for mid, mdata in mice.items():
            pts = mdata.get("trajectory", [])
            cur = None
            for p in pts:
                if p[0] <= frame_number:
                    cur = p
                else:
                    break
            if cur is None:
                continue
            cx = cur[1] * scale_x
            cy = cur[2] * scale_y
            bgr = mdata.get("color_bgr", [0, 0, 255])
            color = _QCol2(int(bgr[2]), int(bgr[1]), int(bgr[0]))
            painter.setPen(QPen(_QCol2(255, 255, 255), 3))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(cx, cy), 7, 7)
        painter.end()
        return result

    def _render_heatmap_frame(self, frame_path: str, source_idx: int):
        """Render pixel-level heatmap overlay (Gaussian-smoothed density)."""
        from PySide6.QtGui import QPainter, QColor as _QCol3, QImage as _QImg
        import numpy as _np
        try:
            from scipy.ndimage import gaussian_filter as _gauss
        except ImportError:
            import cv2
            def _gauss(arr, sigma):
                return cv2.GaussianBlur(arr.astype(_np.float32), (0, 0), sigma)
        pix = QPixmap(frame_path)
        if pix.isNull():
            return pix
        traj = self._trajectory_cache.get(source_idx)
        if not traj:
            return pix
        fw, fh = pix.width(), pix.height()
        cache_key = (source_idx, fw, fh)
        if cache_key in self._heatmap_cache:
            overlay = self._heatmap_cache[cache_key]
        else:
            try:
                acc = _np.zeros((fh, fw), dtype=_np.float64)
                mice = traj.get("mice", {})
                for mdata in mice.values():
                    for p in mdata.get("trajectory", []):
                        _, cx, cy = p
                        px, py = int(round(cx)), int(round(cy))
                        if 0 <= px < fw and 0 <= py < fh:
                            acc[py, px] += 1
                sigma = max(fw, fh) / 200.0
                acc_smooth = _gauss(acc, sigma=sigma)
                if acc_smooth.max() > 0:
                    acc_smooth /= acc_smooth.max()
                import matplotlib; matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                cmap = plt.get_cmap("jet")
                colored = cmap(acc_smooth)
                overlay = _QImg(fw, fh, _QImg.Format.Format_ARGB32)
                overlay.fill(_QCol3(0, 0, 0, 0).rgba())
                mask = acc_smooth > 0.001
                ys, xs = _np.where(mask)
                for y, x in zip(ys, xs):
                    rgba = colored[y, x]
                    fc = _QCol3(int(rgba[0] * 255), int(rgba[1] * 255),
                                int(rgba[2] * 255), int(rgba[3] * 180))
                    overlay.setPixelColor(x, y, fc)
                self._heatmap_cache[cache_key] = overlay
            except Exception:
                return pix
        result = QPixmap(pix)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawImage(0, 0, overlay)
        painter.end()
        return result

    def _render_roi_frame(self, frame_path: str, source_idx: int,
                          frame_number: int):
        """Render ROI overlays with entry/exit stats on the frame."""
        from PySide6.QtGui import QPainter, QPen, QColor as _QCol, QFont as _QF
        from PySide6.QtCore import QPointF
        pix = QPixmap(frame_path)
        if pix.isNull():
            return pix
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)
        # ROI shapes + stats drawn by _RoiShapeItem QGraphicsItems
        # Draw current mouse positions
        traj = self._trajectory_cache.get(source_idx)
        if traj:
            mice = traj.get("mice", {})
            for mid, mdata in mice.items():
                pts = mdata.get("trajectory", [])
                cur = None
                for p in pts:
                    if p[0] <= frame_number:
                        cur = p
                    else:
                        break
                if cur is None:
                    continue
                cx, cy = cur[1], cur[2]
                bgr = mdata.get("color_bgr", [0, 0, 255])
                pc = _QCol(int(bgr[2]), int(bgr[1]), int(bgr[0]))
                painter.setPen(QPen(pc, 3))
                painter.setBrush(pc)
                painter.drawEllipse(QPointF(cx, cy), 5, 5)
                # Check if inside any ROI
                inside_rois = []
                for ri, roi in enumerate(self._rois):
                    if self._point_in_roi(cx, cy, roi["type"], roi["points"]):
                        inside_rois.append(roi.get("label", f"ROI {ri + 1}"))
                if inside_rois:
                    label = f"{mdata.get('label', '')} in {','.join(inside_rois)}"
                    painter.setPen(QPen(_QCol(0, 0, 0, 180), 4))
                    painter.drawText(int(cx) + 10, int(cy) - 8, label)
                    painter.setPen(_QCol(255, 255, 255))
                    painter.drawText(int(cx) + 10, int(cy) - 8, label)
        painter.end()
        return pix

    def _show_current_sample(self):
        frames = self._current_frames()
        if not frames or self._sample_idx >= len(frames):
            return
        pix = self._render_viz_frame(
            frames[self._sample_idx], self._source_idx, self._sample_idx)
        if pix.isNull():
            return
        # Replace only the background pixmap (preserve ROI items during drawing)
        if hasattr(self, '_scene_bg_item') and self._scene_bg_item is not None:
            try:
                self._pose_scene.removeItem(self._scene_bg_item)
            except RuntimeError:
                pass
        self._scene_bg_item = self._pose_scene.addPixmap(pix)
        self._scene_bg_item.setZValue(-1)
        # Rebuild or clear ROI items
        if self._viz_mode == "roi" and not self._roi_drawing:
            self._rebuild_roi_handles()
        elif self._viz_mode != "roi":
            self._clear_roi_scene_items()
        self._pose_scene.setSceneRect(self._pose_scene.itemsBoundingRect())
        self._pose_graphics.resetTransform()
        self._pose_graphics.fitInView(self._pose_scene.sceneRect(), Qt.KeepAspectRatio)
        n = len(frames)
        n_sources = len(self._sources)
        if n_sources > 1:
            self._samples_idx_label.setText(
                f"Video {self._source_idx + 1}/{n_sources}  |  Frame {self._sample_idx + 1}/{n}")
        else:
            self._samples_idx_label.setText(
                f"Frame {self._sample_idx + 1}/{n}" if n > 0 else "No frames")
        self._pose_prev_btn.setEnabled(n_sources > 1)
        self._pose_next_btn.setEnabled(n_sources > 1)
        if self._is_video_mode:
            pos_ms = int(self._sample_idx / self._play_fps * 1000) if self._play_fps > 0 else 0
            dur_ms = int(n / self._play_fps * 1000) if self._play_fps > 0 else 0
            self._update_time_label(pos_ms, dur_ms)
        elif self._folder_ctrl_bar.isVisible():
            self._folder_slider.blockSignals(True)
            self._folder_slider.setValue(self._sample_idx)
            self._folder_slider.blockSignals(False)
            self._update_folder_label()

    def _on_pose_prev(self):
        """◀ — previous source (always switches source, not frame)."""
        if not self._sources:
            return
        self._source_idx = (self._source_idx - 1) % len(self._sources)
        self._sample_idx = 0
        self._apply_source_mode()
        self._show_current_sample()
        # Update frame slider if visible
        if self._is_video_mode:
            self._vp_slider.blockSignals(True)
            self._vp_slider.setValue(0)
            self._vp_slider.blockSignals(False)
        elif self._folder_slider is not None and self._folder_slider.isVisible():
            self._folder_slider.blockSignals(True)
            self._folder_slider.setValue(0)
            self._folder_slider.blockSignals(False)
            self._update_folder_label()

    def _on_pose_next(self):
        """▶ — next source (always switches source, not frame)."""
        if not self._sources:
            return
        self._source_idx = (self._source_idx + 1) % len(self._sources)
        self._sample_idx = 0
        self._apply_source_mode()
        self._show_current_sample()
        if self._is_video_mode:
            self._vp_slider.blockSignals(True)
            self._vp_slider.setValue(0)
            self._vp_slider.blockSignals(False)
        elif self._folder_slider is not None and self._folder_slider.isVisible():
            self._folder_slider.blockSignals(True)
            self._folder_slider.setValue(0)
            self._folder_slider.blockSignals(False)
            self._update_folder_label()

    # ── Folder frame navigation ──────────────────────────────────────

    def _update_folder_label(self):
        frames = self._current_frames()
        n = len(frames)
        self._folder_label.setText(f"Frame {self._sample_idx + 1} / {n}" if n > 0 else "No frames")

    def _on_folder_slider(self, value):
        frames = self._current_frames()
        if not frames:
            return
        self._sample_idx = max(0, min(value, len(frames) - 1))
        self._show_current_sample()

    def _on_folder_prev(self):
        frames = self._current_frames()
        if not frames:
            return
        self._sample_idx = (self._sample_idx - 1) % len(frames)
        self._show_current_sample()
        self._folder_slider.blockSignals(True)
        self._folder_slider.setValue(self._sample_idx)
        self._folder_slider.blockSignals(False)
        self._update_folder_label()

    def _on_folder_next(self):
        frames = self._current_frames()
        if not frames:
            return
        self._sample_idx = (self._sample_idx + 1) % len(frames)
        self._show_current_sample()
        self._folder_slider.blockSignals(True)
        self._folder_slider.setValue(self._sample_idx)
        self._folder_slider.blockSignals(False)
        self._update_folder_label()

    # ── Frame-based video playback ────────────────────────────────────

    def _on_play_pause(self):
        if self._play_timer is not None:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        frames = self._current_frames()
        if not frames:
            return
        self._play_timer = QTimer(self)
        interval = int(1000 / self._play_fps)
        self._play_timer.timeout.connect(self._advance_frame)
        self._play_timer.start(interval)
        self._vp_play_btn.setText("⏸")
        self._vp_play_btn.setStyleSheet(
            "QPushButton{background:#DC2626;color:#FFF;border:none;border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#B91C1C;}")

    def _stop_playback(self):
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer.deleteLater()
            self._play_timer = None
        self._vp_play_btn.setText("▶")
        self._vp_play_btn.setStyleSheet(
            "QPushButton{background:#0078D4;color:#FFF;border:none;border-radius:4px;font-size:12px;}"
            "QPushButton:hover{background:#006CBE;}")

    def _advance_frame(self):
        frames = self._current_frames()
        if not frames:
            self._stop_playback()
            return
        self._sample_idx = (self._sample_idx + 1) % len(frames)
        self._show_current_sample()
        self._vp_slider.blockSignals(True)
        self._vp_slider.setValue(self._sample_idx)
        self._vp_slider.blockSignals(False)

    def _on_speed_changed(self, text):
        speed_map = {"0.5x": 15, "1x": 30, "2x": 60, "4x": 120}
        self._play_fps = speed_map.get(text, 30)
        if self._play_timer is not None:
            self._play_timer.setInterval(int(1000 / self._play_fps))

    def _step_frames(self, delta: int):
        frames = self._current_frames()
        if not frames:
            return
        self._sample_idx = (self._sample_idx + delta) % len(frames)
        self._show_current_sample()
        if self._is_video_mode:
            self._vp_slider.blockSignals(True)
            self._vp_slider.setValue(self._sample_idx)
            self._vp_slider.blockSignals(False)

    # ── Timeline slider ───────────────────────────────────────────────

    def _on_slider_pressed(self):
        self._slider_dragging = True
        if self._play_timer is not None:
            self._play_timer.stop()  # pause during drag

    def _on_slider_released(self):
        self._slider_dragging = False
        frames = self._current_frames()
        if not frames:
            return
        self._sample_idx = max(0, min(self._vp_slider.value(), len(frames) - 1))
        self._show_current_sample()
        if self._play_timer is not None:
            self._play_timer.start()

    def _on_slider_value_changed(self, value):
        """Update time label preview during drag."""
        if not self._slider_dragging:
            return
        frames = self._current_frames()
        if not frames:
            return
        n = len(frames)
        idx = max(0, min(value, n - 1))
        pos_ms = int(idx / self._play_fps * 1000) if self._play_fps > 0 else 0
        dur_ms = int(n / self._play_fps * 1000) if self._play_fps > 0 else 0
        self._update_time_label(pos_ms, dur_ms)

    # ── Time label helpers ────────────────────────────────────────────

    def _update_time_label(self, pos_ms, dur_ms):
        def fmt(ms):
            s = max(0, ms // 1000)
            return f"{s // 60:02d}:{s % 60:02d}"
        self._vp_time_lbl.setText(f"{fmt(pos_ms)} / {fmt(dur_ms)}")

    # ── Worker lifecycle ──────────────────────────────────────────────

    def on_start(self):
        self._result_stack.setCurrentIndex(0)
        self._stop_playback()
        self._e2e_stop_play()
        self._sources = []
        self._source_idx = 0
        self._sample_images = []
        self._sample_idx = 0
        self._is_video_mode = False
        self._is_live_mode = False
        self._e2e_live_frame_idx = 0
        self._e2e_total_frames = 0
        self._vp_play_btn.setVisible(False)
        self._vp_speed_combo.setVisible(False)
        self._vp_ctrl_bar.setVisible(False)
        self._raster_images = []
        self._e2e_sample_images = []
        self._e2e_sample_idx = 0
        self._e2e_raster_images = []
        self._e2e_labels = {}
        self._e2e_class_names = []
        self._e2e_source_idx = 0
        self._e2e_labels_by_source = {}
        self._e2e_rasters_by_source = {}
        self._raster_idx = 0
        # Clear raster view
        if hasattr(self, '_e2e_raster_scene'):
            self._e2e_raster_scene.clear()
        # Reset viz state
        self._viz_mode = "frame_only"
        self._trajectory_cache.clear()
        self._global_traj_cache.clear()
        self._heatmap_cache.clear()
        # Reset ROI state
        self._on_roi_tool(None)
        self._on_roi_clear_all()
        self._update_viz_button_states()
        if self._viz_btn_bar is not None:
            self._viz_btn_bar.setVisible(False)
        if self._roi_inline is not None:
            self._roi_inline.setVisible(False)
        if self._roi_e2e_inline is not None:
            self._roi_e2e_inline.setVisible(False)
        self.run_worker(InferenceWorker, params=self.gather_params())

    @Slot(object)
    def on_result(self, result):
        if not result:
            return
        mode = result.get("mode", "Behavior Prediction")
        in_e2e = (self._mode_combo is not None
                  and self._mode_combo.currentText() == "End-to-End")

        # ── Live mode completion: keep current page, just update status ──
        if self._is_live_mode:
            self._is_live_mode = False
            # Keep the last frame displayed; don't switch pages
            if mode == "Pose":
                pass  # Pose page already showing
            elif mode == "End-to-End":
                pass  # E2E page already showing
            return

        if mode == "Pose":
            if in_e2e:
                # Accumulate sources for E2E display (worker processes one at a time)
                new_sources = result.get("sources", [])
                if new_sources:
                    self._sources.extend(new_sources)
                    # Accumulate sample images from new sources
                    for src in new_sources:
                        frames = [p for p in src.get("frames", []) if Path(p).exists()]
                        self._e2e_sample_images.extend(frames)
                    # Load trajectory for each new source
                    for src in new_sources:
                        if src.get("trajectory_json"):
                            src_idx = len(self._sources) - len(new_sources) + new_sources.index(src)
                            self._ensure_trajectory_loaded(src_idx)
                    # Set to latest source
                    self._e2e_source_idx = len(self._sources) - 1
                    self._e2e_sample_idx = 0
                self._show_e2e_pose_frame()
                return
            self._result_stack.setCurrentIndex(1)
            self._show_pose_result(result)
        else:
            if in_e2e and self._e2e_sample_images:
                # Show E2E page with both pose frame + raster
                self._result_stack.setCurrentIndex(3)
                self._show_e2e_raster(result)
                return
            self._result_stack.setCurrentIndex(2)
            self._show_behavior_result(result)

    def _show_e2e_pose_frame(self):
        if not self._e2e_sample_images:
            return
        # Update source label
        if self._sources and self._e2e_source_idx < len(self._sources):
            src = self._sources[self._e2e_source_idx]
            name = src.get("name", "Unknown")
            prefix = "Video" if src.get("type") == "video" else "Folder"
            n_sources = len(self._sources)
            self._e2e_pose_label.setText(
                f"{prefix} {self._e2e_source_idx + 1}/{n_sources}: {name}")
        # Replace background only (preserve ROI items during drawing)
        if hasattr(self, '_e2e_scene_bg_item') and self._e2e_scene_bg_item is not None:
            try:
                self._e2e_pose_scene.removeItem(self._e2e_scene_bg_item)
            except RuntimeError:
                pass
        # Apply viz rendering if active
        pix = self._render_viz_frame(
            self._e2e_sample_images[self._e2e_sample_idx],
            self._e2e_source_idx, self._e2e_sample_idx)
        if pix.isNull():
            return

        # Paint behavior labels directly on the pixmap (color by mouse ID)
        from PySide6.QtGui import QPainter, QColor as _QCo, QFont as _QF, QPen as _QPen
        MOUSE_COLORS = {"0": _QCo(200, 30, 30), "1": _QCo(30, 80, 220),
                        "2": _QCo(30, 180, 80), "3": _QCo(220, 180, 30)}
        behaviors = self._get_e2e_behavior_parts()
        if behaviors:
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing)
            font = _QF("Segoe UI", 16, _QF.Bold)
            p.setFont(font)
            fm = p.fontMetrics()
            gap = 6; th = fm.height() + 10
            total_h = len(behaviors) * th + (len(behaviors) - 1) * gap
            y = pix.height() - total_h - 10
            for mid, text in behaviors:
                tw = fm.horizontalAdvance(text) + 20
                color = MOUSE_COLORS.get(mid, _QCo(200, 200, 200))
                p.fillRect(8, y, tw, th, color)
                p.setPen(_QPen(_QCo(255, 255, 255, 100)))
                p.drawRect(8, y, tw, th)
                p.setPen(_QCo(255, 255, 255))
                p.drawText(18, y + fm.ascent() + 4, text)
                y += th + gap
            p.end()

        self._e2e_scene_bg_item = self._e2e_pose_scene.addPixmap(pix)
        self._e2e_scene_bg_item.setZValue(-1)
        if self._viz_mode == "roi" and not self._roi_drawing:
            self._rebuild_roi_handles()
        elif self._viz_mode != "roi":
            self._clear_roi_scene_items()
        self._e2e_pose_scene.setSceneRect(self._e2e_pose_scene.itemsBoundingRect())
        self._e2e_pose_view.resetTransform()
        self._e2e_pose_view.fitInView(self._e2e_pose_scene.sceneRect(), Qt.KeepAspectRatio)

        n_frames = len(self._e2e_sample_images)
        n_sources = len(self._sources)
        self._e2e_frame_idx_label.setText(
            f"Frame {self._e2e_sample_idx + 1}/{n_frames}" if n_frames > 0 else "")
        # Enable source nav buttons only when multiple sources exist
        if hasattr(self, '_e2e_prev_btn') and self._e2e_prev_btn is not None:
            self._e2e_prev_btn.setEnabled(n_sources > 1)
            self._e2e_next_btn.setEnabled(n_sources > 1)
        self._update_e2e_raster_indicator()
        if n_frames > 1 and not self._e2e_slider_dragging:
            self._e2e_slider.blockSignals(True)
            self._e2e_slider.setRange(0, n_frames - 1)
            self._e2e_slider.setValue(self._e2e_sample_idx)
            self._e2e_slider.blockSignals(False)
        self._update_e2e_behavior_label()

    def _get_e2e_behavior_parts(self):
        """Get current frame behavior labels as (mouse_id, text) tuples."""
        parts = []
        for mi in sorted(self._e2e_labels.keys(), key=int):
            lbls = self._e2e_labels[mi]
            idx = self._e2e_sample_idx
            if idx < len(lbls):
                cls_id = lbls[idx]
                name = self._e2e_class_names[cls_id] if cls_id < len(self._e2e_class_names) else f"c{cls_id}"
                parts.append((mi, f" M{int(mi)+1}: {name} "))
        return parts

    def _current_e2e_frames(self):
        """Return frames list for the currently selected E2E source."""
        if not self._sources or self._e2e_source_idx >= len(self._sources):
            return []
        return self._sources[self._e2e_source_idx].get("frames", [])

    def _e2e_pose_prev(self):
        """◀ — previous source (always switches source, not frame)."""
        if not self._sources:
            return
        self._e2e_source_idx = (self._e2e_source_idx - 1) % len(self._sources)
        self._e2e_switch_source()

    def _e2e_pose_next(self):
        """▶ — next source (always switches source, not frame)."""
        if not self._sources:
            return
        self._e2e_source_idx = (self._e2e_source_idx + 1) % len(self._sources)
        self._e2e_switch_source()

    def _e2e_switch_source(self):
        """Switch E2E display to the current _e2e_source_idx."""
        # Stop playback when switching sources
        self._e2e_stop_play()
        src = self._sources[self._e2e_source_idx]
        # Rebuild frame list from current source
        self._e2e_sample_images = [p for p in src.get("frames", []) if Path(p).exists()]
        self._e2e_sample_idx = 0
        # Restore per-source labels
        self._e2e_labels = self._e2e_labels_by_source.get(
            self._e2e_source_idx, {})
        # Restore per-source raster
        rasters = self._e2e_rasters_by_source.get(self._e2e_source_idx, [])
        self._e2e_raster_images = rasters
        if rasters:
            self._e2e_raster_scene.clear()
            pix = QPixmap(rasters[0])
            if not pix.isNull():
                self._e2e_raster_scene.addPixmap(pix)
                self._e2e_raster_scene.setSceneRect(
                    self._e2e_raster_scene.itemsBoundingRect())
            # Recreate indicator line
            from PySide6.QtGui import QPen, QColor as _Qc3
            sr_h = self._e2e_raster_scene.sceneRect().height()
            self._e2e_raster_indicator = self._e2e_raster_scene.addLine(
                0, 2, 0, max(sr_h * 0.75, 10),
                QPen(_Qc3(255, 30, 30), 3))
            self._e2e_raster_indicator.setZValue(10)
            self._e2e_raster_view.resetTransform()
            self._e2e_raster_view.fitInView(
                self._e2e_raster_scene.sceneRect(), Qt.IgnoreAspectRatio)
        # Load trajectory for this source
        if src.get("trajectory_json"):
            self._ensure_trajectory_loaded(self._e2e_source_idx)
        # Reset slider
        n = len(self._e2e_sample_images)
        if n > 1:
            self._e2e_slider.blockSignals(True)
            self._e2e_slider.setRange(0, n - 1)
            self._e2e_slider.setValue(0)
            self._e2e_slider.blockSignals(False)
        # Show viz controls if trajectory available
        has_traj = self._e2e_source_idx in self._trajectory_cache
        if self._e2e_viz_btn_bar is not None:
            self._e2e_viz_btn_bar.setVisible(has_traj)
        self._show_e2e_pose_frame()

    def _update_e2e_behavior_label(self):
        """Show current frame's behavior for each mouse."""
        pass  # Behavior now shown as overlay on video frame

    def _update_e2e_raster_indicator(self):
        """Move the red indicator line on the raster to the current frame position."""
        if not hasattr(self, '_e2e_raster_indicator') or self._e2e_raster_indicator is None:
            return
        # Check if the C++ object is still alive (scene.clear() destroys it)
        try:
            _ = self._e2e_raster_indicator.line()
        except RuntimeError:
            self._e2e_raster_indicator = None
            return
        sr = self._e2e_raster_scene.sceneRect()
        if sr.width() == 0:
            return
        # Use live frame index if available, otherwise batch mode index
        if getattr(self, '_is_live_mode', False):
            current_frame = getattr(self, '_e2e_live_frame_idx', 0)
            total_frames = getattr(self, '_e2e_total_frames', 0)
            if total_frames <= 0:
                # Fallback: use max label length
                total_frames = 1
                for lbls in self._e2e_labels.values():
                    total_frames = max(total_frames, len(lbls))
        else:
            current_frame = self._e2e_sample_idx
            total_frames = 0
            for lbls in self._e2e_labels.values():
                total_frames = max(total_frames, len(lbls))
            if total_frames == 0:
                return
        x = sr.left() + (current_frame / total_frames) * sr.width()
        strip_bottom = sr.top() + sr.height() * 0.75
        self._e2e_raster_indicator.setLine(x, sr.top() + 2, x, strip_bottom)

    # ── E2E slider ───────────────────────────────────────────────────

    def _e2e_pause_during_drag(self):
        self._e2e_slider_dragging = True
        if self._e2e_play_timer:
            self._e2e_play_timer.stop()

    def _e2e_slider_seek(self):
        self._e2e_slider_dragging = False
        if self._e2e_sample_images:
            self._e2e_sample_idx = max(0, min(self._e2e_slider.value(), len(self._e2e_sample_images) - 1))
            self._show_e2e_pose_frame()
        if self._e2e_play_timer:
            self._e2e_play_timer.start()

    # ── E2E playback ─────────────────────────────────────────────────

    def _e2e_toggle_play(self):
        if self._e2e_play_timer is not None:
            self._e2e_stop_play()
        else:
            self._e2e_start_play()

    def _e2e_start_play(self):
        if not self._e2e_sample_images:
            return
        self._e2e_play_timer = QTimer(self)
        self._e2e_play_timer.timeout.connect(self._e2e_advance_frame)
        self._e2e_play_timer.start(int(1000 / self._e2e_play_fps))
        self._e2e_play_btn.setText("⏸")
        self._e2e_play_btn.setStyleSheet(
            "QPushButton{background:#DC2626;color:#FFF;border:none;border-radius:3px;font-size:10px;}"
            "QPushButton:hover{background:#B91C1C;}")

    def _e2e_stop_play(self):
        if self._e2e_play_timer is not None:
            self._e2e_play_timer.stop()
            self._e2e_play_timer.deleteLater()
            self._e2e_play_timer = None
        self._e2e_play_btn.setText("▶")
        self._e2e_play_btn.setStyleSheet(
            "QPushButton{background:#0078D4;color:#FFF;border:none;border-radius:3px;font-size:10px;}"
            "QPushButton:hover{background:#006CBE;}")

    def _e2e_advance_frame(self):
        if not self._e2e_sample_images:
            self._e2e_stop_play()
            return
        self._e2e_sample_idx = (self._e2e_sample_idx + 1) % len(self._e2e_sample_images)
        self._show_e2e_pose_frame()

    def _e2e_speed_changed(self, text):
        speed_map = {"0.5x": 15, "1x": 30, "2x": 60, "4x": 120}
        self._e2e_play_fps = speed_map.get(text, 30)
        if self._e2e_play_timer is not None:
            self._e2e_play_timer.setInterval(int(1000 / self._e2e_play_fps))

    def _show_e2e_raster(self, result):
        """Show raster below results frame, collapse log to make space."""
        # Collapse log FIRST -- this triggers a layout resize.
        # Must happen before fitInView so the viewport has its final size.
        self.collapse_log()

        # Show E2E viz controls if trajectory data is available
        has_traj = self._e2e_source_idx in self._trajectory_cache
        if self._e2e_viz_btn_bar is not None:
            self._e2e_viz_btn_bar.setVisible(has_traj)
        raster_images = [p for p in result.get("raster_plots", []) if Path(p).exists()]
        self._e2e_raster_images = raster_images
        if raster_images:
            self._e2e_raster_scene.clear()
            pix = QPixmap(raster_images[0])
            if not pix.isNull():
                self._e2e_raster_scene.addPixmap(pix)
                self._e2e_raster_scene.setSceneRect(self._e2e_raster_scene.itemsBoundingRect())
            # Recreate indicator line
            from PySide6.QtGui import QPen, QColor as _Qc2
            sr_h = self._e2e_raster_scene.sceneRect().height()
            self._e2e_raster_indicator = self._e2e_raster_scene.addLine(
                0, 2, 0, max(sr_h * 0.75, 10),
                QPen(_Qc2(255, 30, 30), 3))
            self._e2e_raster_indicator.setZValue(10)
            # Fit the raster to fill the view width, keeping all frames visible.
            # Layout chain: setCurrentIndex(3) -> page-change layout ->
            # collapse_log() -> resize. All are asynchronous; delay fitInView
            # until all layout updates have settled.
            def _delayed_fit():
                if self._e2e_raster_scene.sceneRect().width() > 0:
                    self._e2e_raster_view.resetTransform()
                    self._e2e_raster_view.fitInView(
                        self._e2e_raster_scene.sceneRect(), Qt.IgnoreAspectRatio)
                    self._update_e2e_raster_indicator()
            QTimer.singleShot(150, _delayed_fit)

        # Store label data per-source (E2E processes sources sequentially)
        e2e_labels = result.get("e2e_labels", {})
        if e2e_labels:
            self._e2e_labels_by_source[self._e2e_source_idx] = {
                k: e2e_labels[k] for k in sorted(e2e_labels.keys(), key=int)}
        # Also store raster per-source
        if raster_images:
            self._e2e_rasters_by_source[self._e2e_source_idx] = raster_images
        # Set active labels from current source
        self._e2e_labels = self._e2e_labels_by_source.get(
            self._e2e_source_idx, {})
        self._e2e_class_names = result.get("e2e_class_names", [])
        # Re-render current frame with behavior labels now available
        self._show_e2e_pose_frame()
        self._update_e2e_behavior_label()

    # ── Behavior results page ─────────────────────────────────────────

    def _build_behavior_results_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

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

        from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
        self._beh_raster_view = QGraphicsView()
        self._beh_raster_view.setRenderHints(
            self._beh_raster_view.renderHints()
            | self._beh_raster_view.renderHints().Antialiasing
            | self._beh_raster_view.renderHints().SmoothPixmapTransform)
        self._beh_raster_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self._beh_raster_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self._beh_raster_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._beh_raster_view.setStyleSheet("QGraphicsView{background:#FFF;border:1px solid #E0E0E0;border-radius:6px;}")
        self._beh_raster_view.setMinimumHeight(200)
        self._beh_raster_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._beh_raster_view.setVisible(False)
        def raster_zoom_wheel(event):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self._beh_raster_view.scale(factor, factor)
        self._beh_raster_view.wheelEvent = raster_zoom_wheel
        self._beh_raster_scene = QGraphicsScene()
        self._beh_raster_view.setScene(self._beh_raster_scene)
        lay.addWidget(self._beh_raster_view)

        self._raster_images = []
        self._raster_idx = 0
        return page

    # ── Behavior result ───────────────────────────────────────────────

    def _show_behavior_result(self, result):
        self._raster_images = [p for p in result.get("raster_plots", []) if Path(p).exists()]
        self._raster_idx = 0
        if self._raster_images:
            self._beh_raster_label.setVisible(True)
            self._beh_raster_view.setVisible(True)
            self._beh_raster_idx.setVisible(True)
            self._rprev_btn.setVisible(True)
            self._rnext_btn.setVisible(True)
            self._show_current_raster()
        else:
            self._beh_raster_label.setVisible(False)
            self._beh_raster_view.setVisible(False)
            self._beh_raster_idx.setVisible(False)
            self._rprev_btn.setVisible(False)
            self._rnext_btn.setVisible(False)

    def _show_current_raster(self):
        if not self._raster_images:
            return
        self._beh_raster_scene.clear()
        pix = QPixmap(self._raster_images[self._raster_idx])
        if pix.isNull():
            return
        self._beh_raster_scene.addPixmap(pix)
        self._beh_raster_scene.setSceneRect(self._beh_raster_scene.itemsBoundingRect())
        self._beh_raster_view.resetTransform()
        self._beh_raster_view.fitInView(self._beh_raster_scene.sceneRect(), Qt.KeepAspectRatio)
        self._beh_raster_idx.setText(f"{self._raster_idx + 1} / {len(self._raster_images)}")

    def _prev_raster(self):
        if self._raster_images:
            self._raster_idx = (self._raster_idx - 1) % len(self._raster_images)
            self._show_current_raster()

    def _next_raster(self):
        if self._raster_images:
            self._raster_idx = (self._raster_idx + 1) % len(self._raster_images)
            self._show_current_raster()
