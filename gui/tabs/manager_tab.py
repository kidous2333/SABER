"""ManagerTab — Factor management."""
import json, logging
from pathlib import Path
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QMessageBox, QGroupBox,
    QScrollArea, QWidget, QCheckBox,
)
from PySide6.QtCore import Qt, Slot
from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.widgets.factor_table import FactorTable

logger = logging.getLogger("gui.manager")


class ManagerTab(BaseTab):
    def __init__(self, parent=None):
        self._factor_table = None; self._all_factors = []
        super().__init__(title="Factor Manager", tab_key="manager", parent=parent)

    def setup_params(self):
        self.add_config_group(ParameterGroup("Data", [
            {"key": "factors_path", "label": "Factors File", "type": "file", "default": "memory/valid_factors.json"},
        ]))
        self.add_config_group(ParameterGroup("Filter", [
            {"key": "min_auc", "label": "Min AUC", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
            {"key": "min_f1", "label": "Min F1", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
            {"key": "min_valid_classes", "label": "Min Valid Classes", "type": "int", "default": 0, "min": 0, "max": 99},
            {"key": "seq_length_min", "label": "Seq Len Min", "type": "int", "default": 0, "min": 0, "max": 60},
            {"key": "seq_length_max", "label": "Seq Len Max", "type": "int", "default": 0, "min": 0, "max": 60},
        ]))
        # Target checkboxes (inline, scrollable)
        self._target_checkboxes = {}
        self._target_filter = set()
        self._target_container = QWidget()
        self._target_layout = QVBoxLayout(self._target_container)
        self._target_layout.setContentsMargins(4, 0, 4, 0)
        self._target_layout.setSpacing(1)
        self._target_header = QLabel("Valid Classes: (not loaded)")
        self._target_header.setStyleSheet("color:#888;font-size:10px;font-weight:600;padding:2px 0;")
        self._target_layout.addWidget(self._target_header)
        self._target_scroll = QScrollArea()
        self._target_scroll.setWidgetResizable(True)
        self._target_scroll.setMaximumHeight(120)
        self._target_scroll.setStyleSheet("QScrollArea{border:1px solid #E0E0E0;background:#FFF;}")
        self._target_cw = QWidget()
        self._target_cl = QVBoxLayout(self._target_cw)
        self._target_cl.setContentsMargins(4, 2, 4, 2)
        self._target_cl.setSpacing(1)
        self._target_scroll.setWidget(self._target_cw)
        self._target_layout.addWidget(self._target_scroll)
        self._select_all_btn = QPushButton("All")
        self._select_none_btn = QPushButton("None")
        self._select_all_btn.setMaximumHeight(20); self._select_none_btn.setMaximumHeight(20)
        self._select_all_btn.setStyleSheet("font-size:9px;padding:0 6px;")
        self._select_none_btn.setStyleSheet("font-size:9px;padding:0 6px;")
        self._select_all_btn.clicked.connect(lambda: self._toggle_all_targets(True))
        self._select_none_btn.clicked.connect(lambda: self._toggle_all_targets(False))
        btn_row2 = QHBoxLayout()
        btn_row2.addWidget(self._select_all_btn)
        btn_row2.addWidget(self._select_none_btn)
        btn_row2.addStretch()
        self._target_layout.addLayout(btn_row2)
        self.add_widget_to_config_panel(self._target_container)

        # Action buttons below config
        btn_row = QHBoxLayout()
        self._export_btn = QPushButton("Export JSON")
        self._export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self._export_btn)
        self._dedup_btn = QPushButton("Dedup")
        self._dedup_btn.clicked.connect(self._on_dedup)
        btn_row.addWidget(self._dedup_btn)
        self.add_widget_to_config_panel(QLabel(""))  # spacer
        btn_widget = QGroupBox()
        btn_widget.setLayout(btn_row)
        self.add_widget_to_config_panel(btn_widget)

    def setup_results(self):
        if hasattr(self, '_log_frame'):
            self._log_frame.hide()
        top_row = QHBoxLayout()
        self._info_label = QLabel("Click Start to load factors.")
        self._info_label.setStyleSheet("color:#888;font-size:12px;padding:0 4px;")
        top_row.addWidget(self._info_label)
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search...")
        self._search_box.setMaximumHeight(28)
        self._search_box.textChanged.connect(self._on_search)
        top_row.addWidget(self._search_box, 1)
        self._results_layout_main.addLayout(top_row)
        self._factor_table = FactorTable()
        self._factor_table.factor_double_clicked.connect(self._show_factor_detail)
        self._results_layout_main.addWidget(self._factor_table)

    # ---- Start / Stop ----
    def on_start(self):
        """Start: load factor file and display."""
        # Clear previous results
        self._factor_table.clear()
        self._info_label.setText("Loading factors...")
        self._search_box.clear()
        self._all_factors = []

        path = self.gather_params().get("factors_path", "")
        if not path:
            QMessageBox.warning(self, "Error", "No factor file specified.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._all_factors = json.load(f)
            self._log_viewer.append_colored(f"Loaded {len(self._all_factors)} factors from {path}", 20)
            self._populate_targets()
            self._apply_filters()
            self._on_running_changed(True)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _on_stop_clicked(self):
        """Stop: clear display."""
        self._all_factors = []
        self._factor_table.clear()
        self._info_label.setText("Click Start to load factors.")
        self._target_header.setText("Valid Classes: (not loaded)")
        self._on_running_changed(False)

    # ---- Internals ----
    def _refresh_config_display(self):
        super()._refresh_config_display()
        if self._all_factors:
            self._apply_filters()

    def _populate_targets(self):
        """Build valid-class checkboxes from loaded factors."""
        for cb in self._target_checkboxes.values():
            self._target_cl.removeWidget(cb)
            cb.deleteLater()
        self._target_checkboxes.clear()

        # Collect all unique class IDs from valid_classes across all factors
        all_ids = set()
        for f in self._all_factors:
            for vc in (f.get("valid_classes") or []):
                if isinstance(vc, dict):
                    all_ids.add(str(vc.get("class", "")))
                elif isinstance(vc, str):
                    all_ids.add(vc)
        all_ids = sorted(all_ids, key=lambda x: int(x) if x.isdigit() else x)

        for cid in all_ids:
            cb = QCheckBox(str(cid))
            cb.setChecked(True)
            cb.setStyleSheet("font-size:10px;spacing:4px;")
            cb.toggled.connect(self._on_target_changed)
            self._target_cl.addWidget(cb)
            self._target_checkboxes[cid] = cb
        self._target_filter = set(all_ids)  # all checked by default
        self._target_header.setText(f"Valid Classes: {len(all_ids)}/{len(all_ids)}")

    def _apply_filters(self):
        factors = list(self._all_factors)
        p = self.gather_params()

        min_auc = float(p.get("min_auc", 0) or 0)
        min_f1 = float(p.get("min_f1", 0) or 0)
        min_valid = int(p.get("min_valid_classes", 0) or 0)
        seq_min = int(p.get("seq_length_min", 0) or 0)
        seq_max = int(p.get("seq_length_max", 0) or 0)

        if min_auc > 0:
            factors = [f for f in factors if (f.get("best_auc") or 0) >= min_auc]
        if min_f1 > 0:
            factors = [f for f in factors if (f.get("best_f1") or 0) >= min_f1]
        # Always filter by checked classes (empty = show nothing)
        factors = [f for f in factors if any(
            str(vc.get("class", "")) in self._target_filter
            for vc in (f.get("valid_classes") or [])
        )]
        if min_valid > 0:
            factors = [f for f in factors if len(f.get("valid_classes") or []) >= min_valid]
        if seq_min > 0:
            factors = [f for f in factors if (f.get("seq_length") or 0) >= seq_min]
        if seq_max > 0:
            factors = [f for f in factors if (f.get("seq_length") or 0) <= seq_max]

        factors.sort(key=lambda f: f.get("best_auc") or 0, reverse=True)

        self._factor_table.load_factors(factors)
        self._info_label.setText(f"Showing {len(factors)} / {len(self._all_factors)} factors")

    @Slot(str)
    def _on_search(self, text):
        if self._factor_table: self._factor_table.set_filter(text)

    def _on_export(self):
        if not self._all_factors: return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "exported_factors.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._all_factors, f, ensure_ascii=False, indent=2)
            self._log_viewer.append_colored(f"Exported to {path}", 20)

    def _toggle_all_targets(self, checked):
        for cb in self._target_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._on_target_changed()

    def _on_target_changed(self):
        self._target_filter = {t for t, cb in self._target_checkboxes.items() if cb.isChecked()}
        all_targets = list(self._target_checkboxes.keys())
        n = len(self._target_filter)
        self._target_header.setText(f"Valid Classes: {n}/{len(all_targets)}")
        if self._all_factors:
            self._apply_filters()

    def _on_dedup(self):
        if not self._all_factors: return
        best = {}
        for f in self._all_factors:
            name = f.get("name", "")
            if name not in best or (f.get("best_auc") or 0) > (best[name].get("best_auc") or 0):
                best[name] = f
        removed = len(self._all_factors) - len(best)
        if removed == 0: QMessageBox.information(self, "Dedup", "No duplicates."); return
        if QMessageBox.question(self, "Dedup", f"Remove {removed} duplicates?", QMessageBox.Yes|QMessageBox.No) == QMessageBox.Yes:
            self._all_factors = list(best.values())
            self._apply_filters()
            self._log_viewer.append_colored(f"Dedup: {removed} removed, {len(self._all_factors)} remain", 20)

    def _show_factor_detail(self, factor):
        from gui.widgets.factor_detail import show_factor_detail
        show_factor_detail(factor, parent=self)
