"""TunerTab — Factor tuning: variant generation + parallel eval + best-variant selection."""
import json
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QLabel, QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton,
    QFileDialog, QFrame, QProgressBar,
)
from PySide6.QtCore import Qt, Slot

from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.widgets.factor_table import FactorTable
from gui.workers.tuner_worker import TunerWorker

logger = logging.getLogger("gui.tuner")

# Column spec for tuned factors
def _fmt_improvement(f, _i):
    v = f.get("_auc_improvement")
    if v is None or abs(v) < 0.0001:
        return "—"
    return f"{v * 100:.2f}%"

def _fmt_mutation(f, _i):
    v = f.get("_mutation")
    return v if v else "—"

def _fmt_params(f, _i):
    v = f.get("_param_label")
    return v if v else "—"

TUNER_COLUMNS = [
    ("#",      30,  lambda f, i: i + 1),
    ("Name",   160, "name"),
    ("Seq",    42,  "seq_length"),
    ("AUC",    60,  "best_auc"),
    ("Δ AUC",  60,  _fmt_improvement),
    ("Method", 80,  _fmt_mutation),
    ("Params", 120, _fmt_params),
]


def _format_auc_improvement(v):
    """Format _auc_improvement as +X.XX% for display."""
    if v is None or abs(v) < 0.0001:
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.2f}%"


class TunerTab(BaseTab):
    """Factor Tuner: runs tuner.py in a subprocess, shows improved factors."""

    def __init__(self, parent=None):
        self._factor_table = None
        self._info_label = None
        self._search_box = None
        self._tuned_factors = []
        self._export_btn = None
        self._progress_bar = None
        self._stat_seeds_val = None
        self._stat_improved_val = None
        super().__init__(title="Factor Tuner", tab_key="tuner", parent=parent)

    def setup_params(self):
        self.add_config_group(ParameterGroup("Tuning", [
            {"key": "input_path", "label": "Seed Factors", "type": "file",
             "default": "memory/valid_factors_deduped.json"},
            {"key": "output_path", "label": "Output", "type": "file",
             "default": "memory/tuned_factors.json"},
            {"key": "num_workers", "label": "Workers", "type": "int", "default": 8, "min": 1, "max": 128},
            {"key": "enable_structure", "label": "Structure Mutation", "type": "checkbox", "default": True},
            {"key": "max_param_combos", "label": "Max Param Combos", "type": "int", "default": 20, "min": 1, "max": 200},
            {"key": "max_variants_per_seed", "label": "Max Variants/Seed", "type": "int", "default": 50, "min": 1, "max": 500},
            {"key": "max_factors", "label": "Max Seed Factors (0=all)", "type": "int", "default": 0, "min": 0, "max": 99999},
            {"key": "resume", "label": "Resume", "type": "checkbox", "default": True},
        ]))
        self.add_config_group(ParameterGroup("Config", [
            {"key": "common_config", "label": "Common Config", "type": "file",
             "default": "config/seq/1.yaml"},
            {"key": "validation_config", "label": "Validation Config", "type": "file",
             "default": "config/validation.yaml"},
        ]))

    def setup_results(self):
        # Top bar
        top_row = QHBoxLayout()
        self._info_label = QLabel("Click Start to run tuning.")
        self._info_label.setStyleSheet("color:#888;font-size:12px;padding:0 4px;")
        top_row.addWidget(self._info_label)

        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search...")
        self._search_box.setMaximumHeight(28)
        self._search_box.textChanged.connect(self._on_search)
        top_row.addWidget(self._search_box, 1)

        self._export_btn = QPushButton("Export")
        self._export_btn.setMaximumHeight(28)
        self._export_btn.setStyleSheet(
            "QPushButton{background:#0078D4;color:#FFF;border:none;border-radius:4px;"
            "padding:4px 12px;font-size:11px;}"
            "QPushButton:hover{background:#006CBE;}"
        )
        self._export_btn.clicked.connect(self._on_export)
        self._export_btn.setVisible(False)
        top_row.addWidget(self._export_btn)
        self._results_layout_main.addLayout(top_row)

        # Progress bar + stat cards
        dash_row = QHBoxLayout()
        dash_row.setSpacing(6)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("Ready")
        self._progress_bar.setStyleSheet(
            "QProgressBar{border:1px solid #D0D0D0;border-radius:4px;background:#EEE;"
            "text-align:center;font-size:13px;height:36px;color:#333;}"
            "QProgressBar::chunk{background:#0078D4;border-radius:3px;}"
        )
        dash_row.addWidget(self._progress_bar, 3)

        box1, self._stat_seeds_val = self._make_stat_card("Seeds", "—", "#0078D4")
        box2, self._stat_improved_val = self._make_stat_card("Improved", "—", "#16A34A")
        dash_row.addWidget(box1)
        dash_row.addWidget(box2)
        self._results_layout_main.addLayout(dash_row)

        # Factor table
        self._factor_table = FactorTable(columns=TUNER_COLUMNS)
        self._factor_table.factor_double_clicked.connect(self._show_factor_detail)
        self._results_layout_main.addWidget(self._factor_table)

    def _make_stat_card(self, label, value, color):
        box = QFrame()
        box.setStyleSheet("QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:5px;}")
        box.setFixedWidth(100)
        bl = QVBoxLayout(box)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(1)
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#888;font-size:10px;font-weight:500;")
        bl.addWidget(lbl, 0, Qt.AlignCenter)
        val_lbl = QLabel(value)
        val_lbl.setStyleSheet(f"color:{color};font-size:17px;font-weight:bold;")
        bl.addWidget(val_lbl, 0, Qt.AlignCenter)
        return box, val_lbl

    def on_start(self):
        self._tuned_factors = []
        self._factor_table.clear()
        self._info_label.setText("Running...")
        self._export_btn.setVisible(False)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Initializing...")
        self._stat_seeds_val.setText("—")
        self._stat_improved_val.setText("—")
        self.run_worker(TunerWorker, params=self.gather_params())

    def on_partial_result(self, data):
        """Live progress + new factor from subprocess."""
        if not data:
            return

        seeds_done = data.get("seeds_done", 0)
        seeds_total = data.get("seeds_total", 0)
        n_improved = data.get("n_improved", 0)
        new_factor = data.get("new_factor")

        # Append new factor to table immediately
        if new_factor:
            logger.debug(f"Received new factor: {new_factor.get('name')}")
            self._tuned_factors.append(new_factor)
            self._tuned_factors.sort(
                key=lambda f: f.get("best_auc", 0) or 0, reverse=True
            )
            self._factor_table.load_factors(self._tuned_factors)
            logger.debug(f"Factor table now has {len(self._tuned_factors)} factors")

        if seeds_total > 0:
            pct = seeds_done * 100 // seeds_total
            self._progress_bar.setValue(pct)
            self._progress_bar.setFormat(f"Seeds: {seeds_done}/{seeds_total}")
            self._stat_seeds_val.setText(f"{seeds_done}/{seeds_total}")
            self._stat_improved_val.setText(str(n_improved))
            self._info_label.setText(
                f"Progress: {seeds_done}/{seeds_total} seeds  |  "
                f"Improved: {n_improved}"
            )

    @Slot(object)
    def on_result(self, result):
        """Show only improved factors from the output file."""
        if not result:
            return

        self._tuned_factors = result.get("tuned_factors", [])
        self._export_btn.setVisible(True)

        display = sorted(
            self._tuned_factors,
            key=lambda f: f.get("best_auc", 0) or 0,
            reverse=True,
        )
        self._factor_table.load_factors(display)
        self._info_label.setText(
            f"Improved: {len(self._tuned_factors)}  |  "
            f"Saved to: {result.get('output_path', '')}"
        )
        self._progress_bar.setValue(100)
        self._progress_bar.setFormat("Complete")
        self._stat_seeds_val.setText("Done")

    @Slot(str)
    def _on_search(self, text):
        if self._factor_table:
            self._factor_table.set_filter(text)

    def _on_export(self):
        if not self._tuned_factors:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Tuned Factors", "tuned_factors_export.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._tuned_factors, f, ensure_ascii=False, indent=2)
            logger.info(f"Exported to {path}")

    def _show_factor_detail(self, factor):
        from gui.widgets.factor_detail import show_factor_detail
        show_factor_detail(factor, parent=self)
