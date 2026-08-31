"""CorrelationTab — Factor correlation analysis."""
import logging
from PySide6.QtWidgets import (
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QHBoxLayout, QLineEdit,
)
from PySide6.QtCore import Qt, Slot
from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.workers.correlation_worker import CorrelationWorker

logger = logging.getLogger("gui.correlation")


class CorrelationTab(BaseTab):
    def __init__(self, parent=None):
        self._report_table = None
        self._all_pairs = []
        self._info_label = None
        self._search_box = None
        self._name_to_factor = {}
        super().__init__(title="Correlation Analysis", tab_key="correlation", parent=parent)

    def setup_params(self):
        self.add_config_group(ParameterGroup("Analysis", [
            {"key": "threshold", "label": "Threshold", "type": "float", "default": 0.95, "min": 0.5, "max": 0.99, "step": 0.01},
            {"key": "keep_by", "label": "Keep By", "type": "combo", "default": "best_auc", "options": ["best_auc", "weighted_auc"]},
            {"key": "max_samples", "label": "Max Samples", "type": "int", "default": 50000, "min": 100, "max": 9999999},
            {"key": "num_workers", "label": "Workers", "type": "int", "default": 8, "min": 1, "max": 128},
            {"key": "purity_mode", "label": "Purity Mode", "type": "combo", "default": "nan_boundary",
             "options": ["nan_boundary", "none", "strict"]},
        ]))
        self.add_config_group(ParameterGroup("Factors", [
            {"key": "factors_path", "label": "Factors JSON", "type": "file", "default": "memory/valid_factors.json"},
        ]))
        self.add_config_group(ParameterGroup("Data", [
            {"key": "train_mouse_dirs", "label": "Train Mouse KP Dirs", "type": "path_list", "default": [""]},
            {"key": "train_tail_dirs", "label": "Train Tail KP Dirs", "type": "path_list", "default": [""]},
            {"key": "train_behavior_m1_files", "label": "Train Behavior M1 Files", "type": "path_list_file", "default": [""]},
            {"key": "train_behavior_m2_files", "label": "Train Behavior M2 Files", "type": "path_list_file", "default": [""]},
            {"key": "max_instances", "label": "Max Instances", "type": "int", "default": 2, "min": 1, "max": 10},
        ]))
        self.add_config_group(ParameterGroup("Labels", []))  # placeholder, label map uses editor below
        from gui.widgets.key_value_editor import KeyValueEditor
        self._label_map_editor = KeyValueEditor(title="Label Map")
        self._label_map_editor.set_pairs({
            "explore_object": "0", "climb": "1", "self_grooming": "2",
            "stand": "3", "blank": "4", "positive_sniffs": "5", "approach": "6",
        })
        self._label_map_editor.setMaximumHeight(200)
        self._param_layout.addWidget(self._label_map_editor)

    def setup_results(self):
        top_row = QHBoxLayout()
        self._info_label = QLabel("Click Start to run analysis.")
        self._info_label.setStyleSheet("color:#888;font-size:12px;padding:0 4px;")
        top_row.addWidget(self._info_label)
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search...")
        self._search_box.setMaximumHeight(28)
        self._search_box.textChanged.connect(self._on_search)
        top_row.addWidget(self._search_box, 1)
        self._results_layout_main.addLayout(top_row)
        self._report_table = QTableWidget(0, 5)
        self._report_table.setHorizontalHeaderLabels(["Factor A","Factor B","Correlation","Kept","Reason"])
        header = self._report_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)     # Factor A - fill space
        header.setSectionResizeMode(1, QHeaderView.Stretch)     # Factor B - fill space
        header.setSectionResizeMode(2, QHeaderView.Fixed)       # Correlation - narrow
        header.setSectionResizeMode(3, QHeaderView.Fixed)       # Kept - narrow
        header.setSectionResizeMode(4, QHeaderView.Stretch)     # Reason - fill space
        header.resizeSection(2, 80)    # Correlation
        header.resizeSection(3, 70)    # Kept
        self._report_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self._report_table.verticalHeader().setDefaultSectionSize(24)
        self._report_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self._results_layout_main.addWidget(self._report_table)

    def on_start(self):
        if hasattr(self, '_log_frame'):
            self._log_frame.show()
        # Clear previous results
        self._report_table.setRowCount(0)
        self._info_label.setText("Running correlation analysis...")
        self._search_box.clear()
        self.run_worker(CorrelationWorker, params=self.gather_params())

    @Slot(object)
    def on_result(self, result):
        if not result: return
        # Load factor definitions for detail lookup
        import json
        factors_path = self.gather_params().get("factors_path", "")
        if factors_path and not self._name_to_factor:
            try:
                with open(factors_path, "r", encoding="utf-8") as f:
                    factors = json.load(f)
                self._name_to_factor = {fac["name"]: fac for fac in factors}
            except Exception:
                self._name_to_factor = {}
        self._all_pairs = sorted(
            result.get("redundant_pairs", []),
            key=lambda p: abs(p.get("correlation", 0)),
            reverse=True,
        )
        self._populate_table(self._all_pairs)
        self._info_label.setText(f"{result.get('n_total', 0)} → {result.get('n_after', 0)} factors  ({len(self._all_pairs)} redundant pairs)")
        if hasattr(self, '_log_frame'):
            self._log_frame.hide()

    def _populate_table(self, pairs: list):
        self._report_table.setRowCount(len(pairs))
        for i, pair in enumerate(pairs):
            for j, key in enumerate(["factor_a","factor_b","correlation","kept","reason"]):
                val = pair.get(key, "")
                if key == "correlation":
                    text = f"{val:.4f}"
                elif key == "kept":
                    text = "Factor A" if val == pair.get("factor_a") else ("Factor B" if val == pair.get("factor_b") else str(val))
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self._report_table.setItem(i, j, item)

    @Slot(str)
    def _on_search(self, text):
        if not self._all_pairs:
            return
        if not text.strip():
            self._populate_table(self._all_pairs)
            return
        low = text.lower()
        filtered = [
            p for p in self._all_pairs
            if low in p.get("factor_a", "").lower()
            or low in p.get("factor_b", "").lower()
            or low in p.get("reason", "").lower()
        ]
        self._populate_table(filtered)

    @Slot(int, int)
    def _on_cell_double_clicked(self, row: int, col: int):
        """Show factor detail when double-clicking Factor A (col 0) or Factor B (col 1)."""
        if col not in (0, 1):
            return
        item = self._report_table.item(row, col)
        if not item:
            return
        factor_name = item.text()
        factor = self._name_to_factor.get(factor_name)
        if factor:
            from gui.widgets.factor_detail import show_factor_detail
            show_factor_detail(factor, parent=self)

    def gather_params(self) -> dict:
        g = super().gather_params()
        if hasattr(self, '_label_map_editor'):
            g["label_map_pairs"] = self._label_map_editor.get_pairs()
        return g

    def _load_settings(self):
        from gui.utils.gui_settings import get_tab_settings
        saved = get_tab_settings(self._tab_key)
        if not saved:
            return
        for group in self._config_groups:
            for item in group._schema:
                key = item["key"]
                if key in saved:
                    group.set_value(key, saved[key])
        if hasattr(self, '_label_map_editor') and "label_map_pairs" in saved:
            self._label_map_editor.set_pairs(saved["label_map_pairs"])
