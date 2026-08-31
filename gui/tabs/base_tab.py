"""
BaseTab — Common tab layout and worker lifecycle management.

Layout (fixed proportions):
  +------------------------------------------------------+
  | [Title]                              [Start] [Stop]  |
  +-----------------------------------+------------------+
  |                                   |  Config Panel    |
  |     Results Area                  |  (fixed 280px)   |
  |                                   |  [Configure] btn |
  |                                   |  full param list |
  +-----------------------------------+------------------+
  |              Log Viewer (fixed ~260px)               |
  +------------------------------------------------------+
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QScrollArea, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QFont
from typing import Optional

from gui.workers.base_worker import BaseWorker
from gui.widgets.log_viewer import LogViewer
from gui.widgets.progress_panel import ProgressPanel
from gui.utils.logging_handler import ModuleLogRedirector


CONFIG_PANEL_WIDTH = 280
LOG_PANEL_HEIGHT = 260


class BaseTab(QWidget):
    """Abstract base for all module tabs."""

    def __init__(self, title: str = "Tab", tab_key: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._tab_key = tab_key or title
        self._worker = None
        self._thread = None
        self._log_redirector = None
        self._config_groups: list = []  # list of ParameterGroup
        self._config_labels: dict = {}  # group -> {key: QLabel}
        self._config_cards: dict = {}   # group -> card QFrame (for visibility toggling)
        self._visibility_conditions: dict = {}  # group -> (source_key, condition_fn)
        self._ssh_manager = None

        self._build_layout()
        self.setup_params()
        self._load_settings()    # Load saved settings after widgets are built
        self._refresh_config_display()
        self._connect_visibility_signals()
        self._evaluate_visibility()
        self.setup_results()

    def _build_layout(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # --- Top bar: title + optional buttons ---
        top_bar = QHBoxLayout()

        self._title_label = QLabel(self._title)
        self._title_label.setStyleSheet(
            "color: #111; font-size: 20px; font-weight: 800; padding-left: 4px;"
        )
        top_bar.addWidget(self._title_label)
        top_bar.addStretch()

        hide_btns = getattr(self, '_hide_buttons', False)

        # Button width: align with Configure Settings below
        # (278px total = CONFIG_PANEL_WIDTH 280px - 2px border; split evenly: (278-6)/2 = 136px each)
        BTN_W = 136

        self._start_btn = QPushButton("▶  Start")
        self._start_btn.setFixedWidth(BTN_W)
        self._start_btn.setStyleSheet("""
            QPushButton {
                background-color: #16A34A; color: #FFF;
                border: none; border-radius: 6px;
                padding: 8px 20px; font-weight: 600; font-size: 13px;
            }
            QPushButton:hover { background-color: #15803D; }
            QPushButton:pressed { background-color: #166534; }
            QPushButton:disabled { background-color: #D1D5DB; color: #9CA3AF; }
        """)
        self._start_btn.clicked.connect(self._on_start_clicked)

        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setFixedWidth(BTN_W)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #DC2626; color: #FFF;
                border: none; border-radius: 6px;
                padding: 8px 20px; font-weight: 600; font-size: 13px;
            }
            QPushButton:hover { background-color: #B91C1C; }
            QPushButton:pressed { background-color: #991B1B; }
            QPushButton:disabled { background-color: #D1D5DB; color: #9CA3AF; }
        """)
        self._stop_btn.clicked.connect(self._on_stop_clicked)

        top_bar.addWidget(self._start_btn)
        top_bar.addWidget(self._stop_btn)
        # 1px offset to align Start button right edge with Configure button below
        top_bar.addSpacing(1)

        if hide_btns:
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)

        main_layout.addLayout(top_bar)

        # --- Main horizontal splitter: results+log | config panel ---
        self._h_splitter = QSplitter(Qt.Horizontal)
        self._h_splitter.setHandleWidth(2)
        self._h_splitter.setStyleSheet("QSplitter::handle { background: #D0D0D0; }")

        # === LEFT OF SPLITTER: results + log (vertical split) ===
        results_column = QFrame()
        results_column.setStyleSheet("QFrame{background:transparent;border:none;}")
        results_column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        results_layout = QVBoxLayout(results_column)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(0)

        # Vertical splitter inside results column
        v_split = QSplitter(Qt.Vertical)
        v_split.setHandleWidth(2)
        v_split.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v_split.setStyleSheet("QSplitter::handle{background:#D0D0D0;}")

        # Results area (top)
        self._results_frame = QFrame()
        self._results_frame.setStyleSheet("QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:6px;}")
        self._results_layout_main = QVBoxLayout(self._results_frame)
        self._results_layout_main.setContentsMargins(8, 0, 8, 8)
        self._results_layout_main.setSpacing(6)
        v_split.addWidget(self._results_frame)

        # Log panel (bottom) — can be hidden per-tab via self._log_frame.hide()
        self._log_frame = QFrame()
        self._log_frame.setStyleSheet("QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:6px;}")
        log_layout_inner = QVBoxLayout(self._log_frame)
        log_layout_inner.setContentsMargins(0, 0, 0, 0)
        log_layout_inner.setSpacing(0)
        self._progress_panel = ProgressPanel()
        log_layout_inner.addWidget(self._progress_panel)
        self._log_viewer = LogViewer()
        log_layout_inner.addWidget(self._log_viewer)
        v_split.addWidget(self._log_frame)

        # Thin toggle bar to re-expand log when collapsed
        self._log_toggle = QPushButton("▬▬  Log  ▬▬")
        self._log_toggle.setFixedHeight(22)
        self._log_toggle.setCursor(Qt.PointingHandCursor)
        self._log_toggle.setStyleSheet(
            "QPushButton{background:#F0F0F0;color:#888;border:1px solid #D0D0D0;"
            "border-radius:0;font-size:10px;}"
            "QPushButton:hover{background:#E0E0E0;color:#555;}")
        self._log_toggle.clicked.connect(
            lambda: self.expand_log() if not self._log_frame.isVisible()
            else self.collapse_log())
        self._log_toggle.setVisible(False)
        v_split.addWidget(self._log_toggle)

        v_split.setSizes([600, 260, 0])
        v_split.setStretchFactor(0, 3)
        v_split.setStretchFactor(1, 1)

        results_layout.addWidget(v_split)
        self._h_splitter.addWidget(results_column)

        # === RIGHT OF SPLITTER: config + factor list ===
        config_column = QFrame()
        config_column.setMinimumWidth(CONFIG_PANEL_WIDTH)
        config_column.setStyleSheet("QFrame{background:#FAFAFA;border:1px solid #E5E5E5;border-radius:8px;}")
        config_layout = QVBoxLayout(config_column)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(0)

        # Configure button
        self._configure_btn = QPushButton("⚙  Configure Settings")
        self._configure_btn.setStyleSheet("""
            QPushButton { background:#0078D4;color:#FFF;border:none;border-radius:6px;
            padding:10px 16px;margin:4px 6px;font-weight:600;font-size:12px; }
            QPushButton:hover { background:#006CBE; }
        """)
        self._configure_btn.clicked.connect(self._open_settings_dialog)
        config_layout.addWidget(self._configure_btn)

        # Scrollable params
        self._param_scroll = QScrollArea()
        self._param_scroll.setWidgetResizable(True)
        self._param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._param_scroll.setStyleSheet("QScrollArea{border:none;background:#FAFAFA;}")
        self._param_widget = QWidget()
        self._param_widget.setStyleSheet("background:#FAFAFA;")
        self._param_layout = QVBoxLayout(self._param_widget)
        self._param_layout.setContentsMargins(12, 8, 12, 8)
        self._param_layout.setSpacing(4)
        self._param_layout.addStretch()
        self._param_scroll.setWidget(self._param_widget)
        config_layout.addWidget(self._param_scroll)

        # Fixed factor list at bottom of config column
        self._fixed_bottom = QVBoxLayout()
        self._fixed_bottom.setContentsMargins(4, 4, 4, 4)
        config_layout.addLayout(self._fixed_bottom)

        self._h_splitter.addWidget(config_column)
        self._h_splitter.setSizes([1000, CONFIG_PANEL_WIDTH])

        main_layout.addWidget(self._h_splitter)

    # ----- Config management -----

    def add_config_group(self, group_widget):
        """
        Register a ParameterGroup. Displayed as a styled card section in the left panel.
        """
        self._config_groups.append(group_widget)

        # Section card
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background: #FFF;
                border: 1px solid #E8E8E8;
                border-radius: 8px;
                margin: 3px 2px;
            }
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(4)

        # Title
        title_lbl = QLabel(group_widget.title())
        title_lbl.setStyleSheet(
            "color: #0078D4; font-weight: 700; font-size: 11px; "
            "padding-bottom: 4px; border: none;"
        )
        card_layout.addWidget(title_lbl)

        value_labels = {}
        for item in group_widget._schema:
            key = item["key"]
            display_name = item.get("label", key)

            row = QHBoxLayout()
            row.setContentsMargins(0, 1, 0, 1)
            row.setSpacing(6)

            name_lbl = QLabel(display_name)
            name_lbl.setStyleSheet(
                "color: #888; font-size: 10px; min-width: 50px; border: none;"
            )
            name_lbl.setWordWrap(True)
            row.addWidget(name_lbl)

            val_lbl = QLabel(self._format_default(item))
            val_lbl.setStyleSheet(
                "color: #222; font-size: 11px; font-weight: 500; border: none;"
            )
            val_lbl.setWordWrap(True)
            val_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            row.addWidget(val_lbl, 1)

            card_layout.addLayout(row)
            value_labels[key] = val_lbl

        self._param_layout.insertWidget(self._param_layout.count() - 1, card)
        self._config_labels[group_widget] = value_labels
        self._config_cards[group_widget] = card

    @staticmethod
    def _elide(text: str, label: QLabel = None, max_len: int = 28) -> str:
        """Truncate long text, keeping the tail (filename) for paths."""
        s = str(text)
        if len(s) <= max_len:
            return s
        # For paths: keep the last segment
        if "/" in s or "\\" in s:
            parts = s.replace("\\", "/").split("/")
            last = parts[-1]
            avail = max_len - 3  # for "..."
            if len(last) >= avail:
                last = "..." + last[-(avail - 3):] if avail > 6 else last[:avail]
            return "…/" + last
        return s[:max_len - 1] + "…"

    def _format_default(self, item: dict) -> str:
        ptype = item.get("type", "text")
        default = item.get("default")
        if default is None:
            return "(not set)"
        if ptype == "checkbox":
            return "✓ Yes" if default else "✗ No"
        if ptype in ("file", "dir", "path_list", "path_list_file"):
            return self._elide(default, max_len=24)
        if isinstance(default, list):
            s = ", ".join(str(x) for x in default[:3])
            if len(default) > 3:
                s += f" … +{len(default)-3}"
            return self._elide(s, max_len=24)
        return self._elide(str(default), max_len=24)

    def _refresh_config_display(self):
        """Update all displayed values from widgets."""
        for group, labels in self._config_labels.items():
            for key, lbl in labels.items():
                val = group.get_value(key)
                if val is None:
                    lbl.setText("(not set)")
                elif isinstance(val, bool):
                    lbl.setText("✓ Yes" if val else "✗ No")
                elif isinstance(val, list):
                    s = ", ".join(str(x) for x in val[:3])
                    if len(val) > 3:
                        s += f" … +{len(val)-3}"
                    lbl.setText(self._elide(s, max_len=28))
                else:
                    lbl.setText(self._elide(str(val), max_len=28))

    def set_group_visibility_condition(self, group_widget, source_key: str, condition_fn):
        """Register a visibility rule: show group only when condition_fn(source_value) is True.
        The group's card is hidden initially and re-evaluated when the source parameter changes."""
        self._visibility_conditions[group_widget] = (source_key, condition_fn)
        # Initially hide the card
        card = self._config_cards.get(group_widget)
        if card:
            card.setVisible(False)

    def _connect_visibility_signals(self):
        """Connect combo/checkbox widgets directly so visibility re-evaluates on change."""
        from PySide6.QtWidgets import QComboBox, QCheckBox
        for group in self._config_groups:
            for key, widget in group._widgets.items():
                if isinstance(widget, QComboBox):
                    widget.currentIndexChanged.connect(self._evaluate_visibility)
                elif isinstance(widget, QCheckBox):
                    widget.toggled.connect(self._evaluate_visibility)

    def _evaluate_visibility(self, *args):
        """Re-evaluate all visibility conditions and show/hide cards accordingly."""
        for group, (source_key, condition_fn) in list(self._visibility_conditions.items()):
            card = self._config_cards.get(group)
            if card is None:
                continue
            # Find the source value by scanning all groups for the key
            source_val = None
            for src_group in self._config_groups:
                source_val = src_group.get_value(source_key)
                if source_val is not None:
                    break
            try:
                visible = condition_fn(source_val) if source_val is not None else False
            except Exception:
                visible = False
            card.setVisible(visible)

    def _load_settings(self):
        """Load saved GUI settings and apply to widgets."""
        from gui.utils.gui_settings import get_tab_settings
        saved = get_tab_settings(self._tab_key)
        if not saved:
            return
        for group in self._config_groups:
            for item in group._schema:
                key = item["key"]
                if key in saved:
                    group.set_value(key, saved[key])

    def _save_settings(self):
        """Save current GUI settings to disk."""
        from gui.utils.gui_settings import save_tab_settings
        values = self.gather_params()
        save_tab_settings(self._tab_key, values)

    def _open_settings_dialog(self):
        """Open settings dialog with tabbed categories."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTabWidget

        dlg = QDialog(self)
        dlg.setWindowTitle(f"{self._title} — Settings")
        dlg.setMinimumWidth(550)
        dlg.setMinimumHeight(400)
        dlg.resize(580, 600)
        dlg.setSizeGripEnabled(True)

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #D0D0D0; border-radius: 4px; background: #FFF; }
            QTabBar::tab { background: #F0F0F0; color: #555; border: 1px solid #D0D0D0;
                padding: 8px 16px; margin-right: 2px; border-radius: 4px 4px 0 0; }
            QTabBar::tab:selected { background: #FFF; color: #0078D4; border-bottom: 2px solid #0078D4; font-weight: 600; }
        """)

        for group in self._config_groups:
            card = self._config_cards.get(group)
            if card is None or card.isVisible():
                tabs.addTab(group, group.title())

        layout.addWidget(tabs)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() == QDialog.Accepted:
            self._save_settings()
            self._refresh_config_display()

    def add_widget_to_config_panel(self, widget: QWidget):
        """Add a widget to the fixed bottom area of the config panel (always visible)."""
        self._fixed_bottom.addWidget(widget)

    # ----- Subclass hooks -----

    def setup_params(self):
        """Override: call add_config_group() for each ParameterGroup."""
        pass

    def setup_results(self):
        """Override: add result widgets to self._results_layout_main."""
        pass

    def clear_results(self):
        """Remove all widgets from results area."""
        while self._results_layout_main.count() > 0:
            item = self._results_layout_main.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def gather_params(self) -> dict:
        """Collect all parameter values from registered groups."""
        g = {}
        for group in self._config_groups:
            g.update(group.get_values())
        return g

    def set_ssh_manager(self, manager):
        self._ssh_manager = manager if manager else None

    # ----- Worker lifecycle -----

    def run_worker(self, worker_class, **worker_kwargs):
        """Start computation in a QThread (same process)."""
        if self._thread and self._thread.isRunning():
            return

        self.expand_log()  # Ensure log panel is visible during run

        self._log_redirector = ModuleLogRedirector()
        self._log_redirector.log_signal.connect(self._log_viewer.append_colored)
        self._log_redirector.install()

        self._thread = QThread(self)
        self._worker = worker_class(parent=None, **worker_kwargs)
        self._worker.log_line.connect(self._log_viewer.append_colored)
        self._worker.progress.connect(self._on_progress)
        self._worker.partial_result.connect(self._on_partial_result)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._thread.start()

        self._on_running_changed(True)
        self._progress_panel.reset()
        self._log_viewer.clear_log()

    @Slot()
    def _on_start_clicked(self):
        self._save_settings()  # Persist current config before starting
        self.on_start()

    def on_start(self):
        pass

    @Slot()
    def _on_stop_clicked(self):
        """Cancel worker, force-terminate thread after 2s if still running."""
        if self._worker:
            self._worker.cancel()
        self._stop_btn.setText("Stopping...")
        self._stop_btn.setEnabled(False)
        # Force kill after 2s
        if self._thread and self._thread.isRunning():
            from PySide6.QtCore import QTimer
            QTimer.singleShot(2000, self._force_stop)

    def _force_stop(self):
        if self._thread and self._thread.isRunning():
            self._log_viewer.append_colored("Force-stopping thread...", 30)
            self._thread.terminate()
            self._thread.wait(500)
            self._log_viewer.append_colored("Thread terminated.", 30)
            self._on_finished()

    @Slot(object, object)
    def _on_progress(self, pct, status=""):
        self._progress_panel.update_progress(int(pct), str(status))

    def _on_partial_result(self, data):
        self.on_partial_result(data)

    def on_partial_result(self, data):
        pass

    def _on_result(self, result):
        self.on_result(result)

    def on_result(self, result):
        pass

    def _on_error(self, msg):
        self._log_viewer.append_colored(f"[ERROR] {msg}", 40)

    @Slot()
    def _on_finished(self):
        if self._log_redirector:
            self._log_redirector.uninstall()
            self._log_redirector = None
        self._thread = None
        self._worker = None
        self._on_running_changed(False)

    def _on_running_changed(self, running: bool):
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        if not running:
            self._stop_btn.setText("■  Stop")

    def collapse_log(self):
        """Hide the log panel completely (just a 6px toggle bar)."""
        if not hasattr(self, '_log_frame') or not hasattr(self, '_log_toggle'):
            return
        self._log_frame.setVisible(False)
        self._log_toggle.setVisible(True)

    def expand_log(self):
        """Restore the log panel to default size."""
        if not hasattr(self, '_log_frame') or not hasattr(self, '_log_toggle'):
            return
        self._log_frame.setVisible(True)
        self._log_toggle.setVisible(False)
