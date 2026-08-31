"""BatchMiningTab — Multi-seq batch mining with progress bars and factor cards."""
import logging, json
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QScrollArea, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot, QTimer

from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.workers.batch_mining_worker import BatchMiningWorker

logger = logging.getLogger("gui.batch_mining")


class ResourceGauge(QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        l = QVBoxLayout(self); l.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label); lbl.setStyleSheet("color:#666;font-size:11px;font-weight:500;")
        l.addWidget(lbl)
        self._bar = QProgressBar(); self._bar.setRange(0, 100); self._bar.setTextVisible(True)
        self._bar.setFormat("%p%"); self._bar.setMaximumHeight(24)
        self._bar.setStyleSheet(
            "QProgressBar{border:1px solid #D0D0D0;border-radius:4px;background:#EEE;"
            "text-align:center;font-size:10px;color:#333;}"
            "QProgressBar::chunk{background:#0078D4;border-radius:3px;}"
        )
        l.addWidget(self._bar)

    def set_value(self, pct, text=""):
        self._bar.setValue(int(pct))
        if text:
            self._bar.setFormat(f"{text} (%p%)")


class BatchMiningTab(BaseTab):
    def __init__(self, parent=None):
        self._seq_bars = {}
        self._cpu_gauge = None
        self._mem_gauge = None
        self._mining_active = False
        self._monitor_timer = None
        self._known_factors = set()
        super().__init__(title="Batch Mining", tab_key="batch_mining", parent=parent)

    def _load_settings(self):
        """Load Batch Mining settings; fall back to Discovery settings on first run."""
        from gui.utils.gui_settings import get_tab_settings
        own = get_tab_settings("batch_mining")
        if own:
            for group in self._config_groups:
                for item in group._schema:
                    key = item["key"]
                    if key in own:
                        group.set_value(key, own[key])
            if hasattr(self, '_label_map_editor') and "label_map_pairs" in own:
                self._label_map_editor.set_pairs(own["label_map_pairs"])
        else:
            discovery = get_tab_settings("discovery")
            if discovery:
                for group in self._config_groups:
                    for item in group._schema:
                        key = item["key"]
                        if key in discovery:
                            group.set_value(key, discovery[key])
                if hasattr(self, '_label_map_editor') and "label_map_pairs" in discovery:
                    self._label_map_editor.set_pairs(discovery["label_map_pairs"])
            self._save_settings()

    # ---- Full config: Batch-specific first, then same as DiscoveryTab ----
    def setup_params(self):
        self.add_config_group(ParameterGroup("Batch", [
            {"key": "seq_groups", "label": "Seq Groups", "type": "text", "default": "1,5,15,30,60"},
            {"key": "target_n", "label": "Target N", "type": "int", "default": 600, "min": 10, "max": 99999},
            {"key": "mem_limit", "label": "Mem Limit %", "type": "int", "default": 60, "min": 10, "max": 95},
            {"key": "cpu_limit", "label": "CPU Limit %", "type": "int", "default": 80, "min": 10, "max": 100},
            {"key": "auto_merge", "label": "Auto-merge", "type": "checkbox", "default": True},
        ]))
        self.add_config_group(ParameterGroup("Sequence", [
            {"key": "seq_length", "label": "Seq Length", "type": "int", "default": 5, "min": 1, "max": 60},
            {"key": "stride", "label": "Stride", "type": "int", "default": 1, "min": 1, "max": 30},
            {"key": "purity_threshold", "label": "Purity", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
        ]))
        self.add_config_group(ParameterGroup("Validation", [
            {"key": "min_auc", "label": "Min AUC", "type": "float", "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01},
            {"key": "min_f1", "label": "Min F1", "type": "float", "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01},
            {"key": "use_gpu", "label": "Use GPU", "type": "checkbox", "default": False},
            {"key": "lgbm_estimators", "label": "LGBM Trees", "type": "int", "default": 100, "min": 10, "max": 1000},
            {"key": "lgbm_depth", "label": "LGBM Depth", "type": "int", "default": 4, "min": 2, "max": 12},
        ]))
        self.add_config_group(ParameterGroup("Loop", [
            {"key": "max_rounds", "label": "Max Rounds", "type": "int", "default": 200, "min": 1, "max": 9999},
            {"key": "early_stop_rounds", "label": "Early Stop", "type": "int", "default": 5, "min": 1, "max": 100},
            {"key": "max_valid_factors", "label": "Max Factors", "type": "int", "default": 5000, "min": 1, "max": 99999},
            {"key": "hypotheses_per_round", "label": "Hypotheses/Round", "type": "int", "default": 8, "min": 1, "max": 50},
        ]))
        self.add_config_group(ParameterGroup("Dataset", [
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
        self.add_config_group(ParameterGroup("Labels", []))  # placeholder
        from gui.widgets.key_value_editor import KeyValueEditor
        self._label_map_editor = KeyValueEditor(title="Label Map")
        self._label_map_editor.set_pairs({
            "explore_object": "0", "climb": "1", "self_grooming": "2",
            "stand": "3", "blank": "4", "positive_sniffs": "5", "approach": "6",
        })
        self._label_map_editor.setMaximumHeight(200)
        self._param_layout.addWidget(self._label_map_editor)
        self.add_config_group(ParameterGroup("LLM", [
            {"key": "llm_provider", "label": "Provider", "type": "combo", "default": "anthropic", "options": ["anthropic", "openai", "glm"]},
            {"key": "llm_model", "label": "Model", "type": "text", "default": "claude-sonnet-4-20250514"},
            {"key": "llm_base_url", "label": "Base URL", "type": "text", "default": "https://www.micuapi.ai"},
            {"key": "llm_api_key", "label": "API Key", "type": "password", "default": ""},
            {"key": "llm_temperature", "label": "Temperature", "type": "float", "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1},
            {"key": "llm_user_agent", "label": "User-Agent", "type": "text",
             "default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) Gecko/20100101 Firefox/149.0"},
            {"key": "llm_custom_headers", "label": "Custom Headers", "type": "text", "default": ""},
            {"key": "llm_timeout", "label": "Timeout (s)", "type": "int", "default": 600, "min": 60, "max": 3600},
        ]))
        self.add_config_group(ParameterGroup("Output", [
            {"key": "factors_output_path", "label": "Save Factors To", "type": "file",
             "default": "memory/gui_discovered_factors.json"},
        ]))

    # ---- Results area ----
    def setup_results(self):
        if hasattr(self, '_log_frame'):
            self._log_frame.hide()

        # CPU / Memory gauges
        hw = QHBoxLayout()
        self._cpu_gauge = ResourceGauge("CPU"); hw.addWidget(self._cpu_gauge)
        self._mem_gauge = ResourceGauge("Memory"); hw.addWidget(self._mem_gauge)
        self._results_layout_main.addLayout(hw)

        # Status label
        self._status_label = QLabel("Configure and click Start.")
        self._status_label.setStyleSheet("color:#888;font-size:13px;padding:4px 8px;font-weight:500;")
        self._results_layout_main.addWidget(self._status_label)

        # Seq progress bars
        self._seq_container = QWidget()
        self._seq_layout = QVBoxLayout(self._seq_container)
        self._seq_layout.setContentsMargins(0, 0, 0, 0)
        self._seq_layout.setSpacing(4)
        self._results_layout_main.addWidget(self._seq_container)

        # Factor cards scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self._card_container = QWidget()
        self._card_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setSpacing(6)
        self._card_layout.setContentsMargins(4, 4, 4, 4)
        self._card_layout.addStretch()
        self._scroll.setWidget(self._card_container)
        self._results_layout_main.addWidget(self._scroll)

    def _add_card(self, card):
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vp = self._scroll.viewport()
        if vp:
            vw = vp.width() - 16
            if vw > 100:
                card.setFixedWidth(vw)
        idx = max(0, self._card_layout.count() - 1)
        self._card_layout.insertWidget(idx, card)

    def _open_card_detail(self, data):
        from gui.widgets.factor_detail import show_factor_detail
        show_factor_detail(data, parent=self)

    # ---- Helpers for creating cards (same style as DiscoveryTab) ----
    def _lbl(self, text, color="#333", size=12, bold=False, wrap=False):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{color};font-size:{size}px;font-weight:{'bold' if bold else 'normal'};")
        if wrap:
            lbl.setWordWrap(True)
        return lbl

    def _build_factor_card(self, data):
        name = data.get("name", "?")
        seq = data.get("seq", data.get("seq_length", "?"))
        target = data.get("target", "")
        auc = data.get("best_auc") or 0
        f1 = data.get("best_f1") or 0
        per_class = data.get("per_class", {})
        valid_classes = data.get("valid_classes", [])
        desc = data.get("description", "")
        code = data.get("code", "")

        card = QFrame()
        is_valid = data.get("valid", False) or auc >= 0.70
        card.setStyleSheet(
            f"QFrame{{background:{'#F6FFF6' if is_valid else '#FFF'};"
            f"border:2px solid {'#16A34A' if is_valid else '#E5E5E5'};border-radius:10px;}}"
        )
        # Double-click to open factor detail
        card.mouseDoubleClickEvent = lambda e, d=data: self._open_card_detail(d)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(14, 8, 14, 6)
        cl.setSpacing(3)

        # Row 1: name + AUC
        nr = QHBoxLayout()
        nr.addWidget(self._lbl(f"seq={seq}  {name[:40]}", "#111", 14, True))
        nr.addStretch()
        nr.addWidget(self._lbl(f"AUC {auc:.4f}", "#16A34A" if is_valid else "#888", 16, True))
        cl.addLayout(nr)

        # Row 2: description or target
        if desc:
            cl.addWidget(self._lbl(desc[:120], "#555", 11, wrap=True))
        elif target:
            cl.addWidget(self._lbl(f"Target: {target}", "#555", 11))

        # Row 3: per-class AUC (compact)
        if per_class:
            mr = QHBoxLayout(); mr.setSpacing(3)
            valid_ids = {v["class"] for v in (valid_classes or []) if isinstance(v, dict)}
            for cid in sorted(per_class.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
                cls_data = per_class[cid]
                cls_auc = cls_data.get("auc", 0) if isinstance(cls_data, dict) else float(cls_data)
                is_valid = str(cid) in valid_ids
                bg = "#F0FFF0" if is_valid else "#FFF5F5"
                border = "#86EFAC" if is_valid else "#F0A0A0"
                col = "#16A34A" if is_valid else "#C04040"
                tag = QLabel(f"{cid}:{cls_auc:.2f}")
                tag.setStyleSheet(
                    f"color:{col};font-size:9px;font-weight:{'bold' if is_valid else 'normal'};"
                    f"background:{bg};border:1px solid {border};border-radius:3px;padding:2px 4px;"
                )
                mr.addWidget(tag)
            mr.addStretch()
            cl.addLayout(mr)

        return card

    # ---- Start / Stop ----
    def on_start(self):
        self._mining_active = True
        self._known_factors = set()

        # Clear old cards
        while self._card_layout.count() > 0:
            item = self._card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._card_layout.addStretch()

        # Clear old progress bars
        while self._seq_layout.count():
            item = self._seq_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget():
                        sub.widget().deleteLater()
                item.layout().deleteLater()

        self._seq_bars = {}
        self._status_label.setText("Starting batch mining...")
        self._on_running_changed(True)
        self._start_monitor()
        self.run_worker(BatchMiningWorker, params=self.gather_params())

    def _start_monitor(self):
        """Start a timer to update CPU / Memory gauges."""
        if self._monitor_timer:
            self._monitor_timer.stop()
        self._monitor_timer = QTimer(self)
        self._monitor_timer.timeout.connect(self._update_monitor)
        self._monitor_timer.start(3000)

    def _update_monitor(self):
        try:
            import psutil
            self._cpu_gauge.set_value(psutil.cpu_percent(interval=0.1))
            self._mem_gauge.set_value(psutil.virtual_memory().percent)
        except ImportError:
            pass

    def _stop_mining(self):
        self._mining_active = False
        if self._monitor_timer:
            self._monitor_timer.stop()
            self._monitor_timer = None
        if self._worker:
            self._worker.cancel()
        self._status_label.setText("Stopped.")
        self._on_running_changed(False)

    def _on_stop_clicked(self):
        self._stop_mining()

    # ---- Partial results (progress + factors) ----
    @Slot(object)
    def on_partial_result(self, data):
        if not isinstance(data, dict):
            return
        typ = data.get("type", "")

        if typ in ("seq_init", "seq_progress"):
            seq = data.get("seq")
            count = data.get("count", 0)
            target = data.get("target", 600)

            # Create progress bar if needed
            if seq not in self._seq_bars:
                row = QHBoxLayout()
                lbl = QLabel(f"seq={seq} [0w]")
                lbl.setStyleSheet("color:#555;font-size:11px;font-weight:500;min-width:80px;")
                row.addWidget(lbl)
                bar = QProgressBar()
                bar.setRange(0, 100); bar.setTextVisible(True)
                bar.setFormat("0 / ? (0%)")
                bar.setStyleSheet(
                    "QProgressBar{border:1px solid #D0D0D0;border-radius:4px;background:#EEE;"
                    "text-align:center;font-size:10px;height:20px;color:#333;}"
                    "QProgressBar::chunk{background:#0078D4;border-radius:3px;}"
                )
                row.addWidget(bar)
                # Insert before the stretch at the end of seq_layout
                self._seq_layout.insertLayout(self._seq_layout.count(), row)
                self._seq_bars[seq] = (bar, lbl, 0)

            bar, lbl, running = self._seq_bars[seq]
            pct = data.get("pct", 0)
            new_run = data.get("new_this_run", 0)
            if "running" in data:
                running = data["running"]
                self._seq_bars[seq] = (bar, lbl, running)
            bar.setValue(min(pct, 100))
            bar.setFormat(f"{count} / {target} ({pct}%)  +{new_run}  [{running}w]")
            lbl.setText(f"seq={seq} [{running}w]")

        elif typ == "factor_found":
            # Skip duplicates
            name = data.get("name", "")
            if name in self._known_factors:
                return
            self._known_factors.add(name)
            card = self._build_factor_card(data)
            self._add_card(card)

    def on_result(self, result):
        if result and isinstance(result, dict):
            n = result.get("n_factors", 0)
            self._status_label.setText(f"Complete! {n} factors discovered.")
        self._on_running_changed(False)
