"""TrainTab — TMP (Token Mixed Pose) model training interface."""
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QTableWidget,
    QTableWidgetItem, QHeaderView, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot

from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.widgets.matplotlib_widget import MplWidget
from gui.workers.train_worker import TrainWorker

logger = logging.getLogger("gui.train")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TrainTab(BaseTab):
    """Tab for training TMP (Token Mixed Pose) estimation models."""

    def __init__(self, parent=None):
        self._loss_chart = None
        self._map_chart = None
        self._metric_table = None
        self._status_label = None
        self._result_frame = None
        self._result_labels: dict = {}
        self._epoch_data = []  # list of per-epoch dicts
        super().__init__(title="TMP Pose Training", tab_key="train", parent=parent)

    # ------------------------------------------------------------------
    #  Parameter groups
    # ------------------------------------------------------------------
    def setup_params(self):
        default_model = str(PROJECT_ROOT / "config" / "models" / "tmp_n.yaml")
        default_dataset = str(PROJECT_ROOT / "config" / "datasets" / "coco8-pose_mouse.yaml")

        self.add_config_group(ParameterGroup("Model", [
            {"key": "model_yaml", "label": "Model YAML", "type": "file",
             "default": default_model},
            {"key": "pretrained", "label": "Pretrained .pt", "type": "file",
             "default": ""},
        ]))
        self.add_config_group(ParameterGroup("Dataset", [
            {"key": "dataset_yaml", "label": "Dataset YAML", "type": "file",
             "default": default_dataset},
        ]))
        self.add_config_group(ParameterGroup("Training", [
            {"key": "epochs", "label": "Epochs", "type": "int",
             "default": 30, "min": 1, "max": 10000},
            {"key": "imgsz", "label": "Image Size", "type": "int",
             "default": 640, "min": 64, "max": 2048, "step": 32},
            {"key": "batch", "label": "Batch Size", "type": "int",
             "default": 16, "min": 1, "max": 512},
            {"key": "device", "label": "Device", "type": "choice",
             "choices": ["", "cuda", "cpu", "cuda:0", "cuda:1"],
             "default": ""},
            {"key": "single_cls", "label": "Single Class", "type": "checkbox",
             "default": True},
            {"key": "workers", "label": "Workers", "type": "int",
             "default": 8, "min": 0, "max": 64},
        ]))
        self.add_config_group(ParameterGroup("Output", [
            {"key": "project", "label": "Project Dir", "type": "text",
             "default": "runs/train"},
            {"key": "name", "label": "Exp Name", "type": "text",
             "default": "tmp_exp"},
        ]))

    # ------------------------------------------------------------------
    #  Results area
    # ------------------------------------------------------------------
    def setup_results(self):
        # Status line
        self._status_label = QLabel("Configure paths and press Start to begin training.")
        self._status_label.setStyleSheet("color:#555;font-size:13px;padding:4px 0;")
        self._status_label.setWordWrap(True)
        self._results_layout_main.addWidget(self._status_label)

        # -- charts row: left=Loss, right=mAP --
        charts_row = QHBoxLayout()
        charts_row.setSpacing(6)
        self._loss_chart = MplWidget(figsize=(3.5, 2.2), toolbar=False)
        self._loss_ax = self._loss_chart.subplot(111)
        self._loss_ax.set_title("Train Loss", fontsize=9)
        self._map_chart = MplWidget(figsize=(3.5, 2.2), toolbar=False)
        self._map_ax = self._map_chart.subplot(111)
        self._map_ax.set_title("Val mAP", fontsize=9)
        charts_row.addWidget(self._loss_chart, 1)
        charts_row.addWidget(self._map_chart, 1)
        self._results_layout_main.addLayout(charts_row)

        # -- per-epoch metric table --
        self._metric_table = QTableWidget()
        self._metric_table.setColumnCount(9)
        self._metric_table.setHorizontalHeaderLabels(
            ["Epoch", "box", "pose", "kobj", "cls", "dfl", "mAP50", "mAP50-95", "LR"])
        self._metric_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._metric_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._metric_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._metric_table.setAlternatingRowColors(True)
        self._metric_table.setStyleSheet(
            "QTableWidget{background:#FFF;border:1px solid #E0E0E0;gridline-color:#F0F0F0;"
            "font-size:10px;color:#333;}"
            "QHeaderView::section{background:#F5F5F5;color:#555;padding:4px 6px;"
            "border:none;border-bottom:2px solid #D0D0D0;font-weight:600;font-size:10px;}"
        )
        self._results_layout_main.addWidget(self._metric_table)

        # -- final results card (hidden until complete) --
        self._result_frame = QFrame()
        self._result_frame.setStyleSheet(
            "QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:6px;}")
        self._result_frame.setVisible(False)
        result_layout = QVBoxLayout(self._result_frame)
        result_layout.setContentsMargins(16, 10, 16, 10)
        result_layout.setSpacing(4)
        title = QLabel("Best Results")
        title.setStyleSheet("color:#0078D4;font-weight:700;font-size:13px;border:none;")
        result_layout.addWidget(title)
        fields = [
            ("Best Checkpoint", "best_pt"),
            ("Best mAP@50", "best_map50"),
            ("Best mAP@50-95", "best_map50_95"),
            ("Save Directory", "save_dir"),
        ]
        for label, key in fields:
            row = QHBoxLayout(); row.setSpacing(8)
            nl = QLabel(label)
            nl.setStyleSheet("color:#888;font-size:10px;border:none;min-width:100px;")
            row.addWidget(nl)
            vl = QLabel("—")
            vl.setStyleSheet("color:#222;font-size:11px;font-weight:500;border:none;")
            vl.setWordWrap(True)
            vl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.addWidget(vl, 1)
            result_layout.addLayout(row)
            self._result_labels[key] = vl
        self._results_layout_main.addWidget(self._result_frame)

    # ------------------------------------------------------------------
    #  Start / Stop
    # ------------------------------------------------------------------
    def on_start(self):
        params = self.gather_params()
        model_yaml = params.get("model_yaml", "")
        dataset_yaml = params.get("dataset_yaml", "")

        if not model_yaml or not Path(model_yaml).exists():
            self._status_label.setText(f"❌ Model YAML not found: {model_yaml}")
            self._status_label.setStyleSheet("color:#DC2626;font-size:13px;")
            return
        if not dataset_yaml or not Path(dataset_yaml).exists():
            self._status_label.setText(f"❌ Dataset YAML not found: {dataset_yaml}")
            self._status_label.setStyleSheet("color:#DC2626;font-size:13px;")
            return

        self._status_label.setText("Starting TMP training...")
        self._status_label.setStyleSheet("color:#0078D4;font-size:13px;")
        self._result_frame.setVisible(False)
        self._metric_table.setVisible(True)
        self._epoch_data = []
        self._metric_table.setRowCount(0)
        self._loss_ax.clear(); self._loss_ax.set_title("Train Loss", fontsize=9)
        self._map_ax.clear(); self._map_ax.set_title("Val mAP", fontsize=9)
        self._loss_chart.draw(); self._map_chart.draw()

        self.run_worker(TrainWorker, params=params)

    # ------------------------------------------------------------------
    #  Live per-epoch updates
    # ------------------------------------------------------------------
    def on_partial_result(self, data):
        """Handle per-epoch data from the worker callback."""
        if not isinstance(data, dict) or "epoch" not in data:
            return
        self._epoch_data.append(data)
        self._status_label.setText(
            f"Training... Epoch {data['epoch']}/{data.get('total_epochs', '?')}")
        self._status_label.setStyleSheet("color:#0078D4;font-size:13px;")

        # Update table
        row = self._metric_table.rowCount()
        self._metric_table.insertRow(row)
        cols = [
            f"{data['epoch']}/{data.get('total_epochs','?')}",
            f"{data.get('box_loss', 0):.4f}",
            f"{data.get('pose_loss', 0):.4f}",
            f"{data.get('kobj_loss', 0):.4f}",
            f"{data.get('cls_loss', 0):.4f}",
            f"{data.get('dfl_loss', 0):.4f}",
            f"{data.get('map50', 0):.4f}" if data.get('map50') is not None else "—",
            f"{data.get('map50_95', 0):.4f}" if data.get('map50_95') is not None else "—",
            f"{data.get('lr', 0):.2e}" if data.get('lr') is not None else "—",
        ]
        for j, text in enumerate(cols):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            self._metric_table.setItem(row, j, item)
        self._metric_table.scrollToBottom()

        # Update charts
        if len(self._epoch_data) >= 2:
            self._update_charts()

    def _update_charts(self):
        """Redraw loss and mAP charts."""
        import numpy as np
        epochs = [d['epoch'] for d in self._epoch_data]
        loss_keys = ['box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss']
        colors = ['#0078D4', '#16A34A', '#DC2626', '#9333EA', '#F59E0B']

        # -- loss chart (train=solid, val=dashed) --
        self._loss_ax.clear()
        self._loss_ax.set_title("Loss (train — / val - -)", fontsize=9)
        for key, color in zip(loss_keys, colors):
            label = key.replace('_loss', '')
            vals = [d.get(key, np.nan) for d in self._epoch_data]
            self._loss_ax.plot(epochs, vals, color=color, linewidth=1, label=f'{label}_tr')
            vkey = f'val_{key}'
            vvals = [d.get(vkey) for d in self._epoch_data]
            if any(v is not None for v in vvals):
                self._loss_ax.plot(epochs, vvals, color=color, linewidth=0.8,
                                   linestyle='--', alpha=0.7, label=f'{label}_val')
        self._loss_ax.legend(fontsize=5, loc='upper right', ncol=2)
        self._loss_ax.set_xlabel("Epoch", fontsize=7)
        self._loss_ax.set_ylabel("Loss", fontsize=7)
        self._loss_ax.tick_params(labelsize=6)
        self._loss_chart.draw()

        # -- mAP chart (val only) --
        self._map_ax.clear()
        self._map_ax.set_title("Val mAP", fontsize=9)
        for key, color, label in [('map50', '#0078D4', 'mAP50'), ('map50_95', '#16A34A', 'mAP50-95')]:
            vals = [d.get(key) for d in self._epoch_data]
            valid = [(e, v) for e, v in zip(epochs, vals) if v is not None]
            if valid:
                ex, vy = zip(*valid)
                self._map_ax.plot(ex, vy, color=color, linewidth=1, marker='.', markersize=3, label=label)
        self._map_ax.legend(fontsize=6, loc='upper left')
        self._map_ax.set_xlabel("Epoch", fontsize=7)
        self._map_ax.set_ylabel("mAP", fontsize=7)
        self._map_ax.tick_params(labelsize=6)
        self._map_chart.draw()

    # ------------------------------------------------------------------
    #  Final result
    # ------------------------------------------------------------------
    def on_result(self, result):
        self._update_charts()
        # Replace metric table with final results card (same area, no resize)
        self._metric_table.setVisible(False)
        self._result_frame.setVisible(True)

        if result is None:
            self._status_label.setText("Training cancelled.")
            self._status_label.setStyleSheet("color:#888;font-size:13px;")
            return

        self._status_label.setText("TMP training complete!")
        self._status_label.setStyleSheet("color:#16A34A;font-size:13px;")

        best_pt = result.get("best_pt", "")
        self._result_labels["best_pt"].setText(best_pt or "(not found)")
        map50 = result.get("best_map50")
        self._result_labels["best_map50"].setText(
            f"{map50:.4f}" if isinstance(map50, (int, float)) else "N/A")
        map5095 = result.get("best_map50_95")
        self._result_labels["best_map50_95"].setText(
            f"{map5095:.4f}" if isinstance(map5095, (int, float)) else "N/A")
        self._result_labels["save_dir"].setText(result.get("save_dir", "N/A"))
