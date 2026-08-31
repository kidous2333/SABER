"""
MainWindow — Application shell with categorized sidebar navigation.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QStackedWidget,
    QLabel, QMenuBar, QMenu, QMessageBox, QApplication, QFrame,
    QPushButton, QScrollArea, QSizePolicy,
)
from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QAction, QFont, QKeySequence

from gui.widgets.status_bar import StatusBarWidget
from gui.tabs.discovery_tab import DiscoveryTab
from gui.tabs.train_behavior_tab import TrainBehaviorTab
from gui.tabs.evolution_tab import EvolutionTab
from gui.tabs.analysis_tab import AnalysisTab
from gui.tabs.manager_tab import ManagerTab
from gui.tabs.correlation_tab import CorrelationTab
from gui.tabs.tuner_tab import TunerTab
from gui.tabs.batch_mining_tab import BatchMiningTab
from gui.tabs.inference_tab import InferenceTab
from gui.tabs.validation_tab import ValidationTab
from gui.tabs.config_tab import ConfigTab
from gui.tabs.ssh_tab import SSHTab
from gui.tabs.train_tab import TrainTab


LIGHT_STYLE = """
QMainWindow { background-color: #F5F5F5; }
QMenuBar { background-color: #FFF; color: #333; border-bottom: 1px solid #E0E0E0; padding: 2px; font-size: 12px; }
QMenuBar::item:selected { background-color: #E8F0FE; }
QMenu { background-color: #FFF; color: #333; border: 1px solid #D0D0D0; border-radius: 6px; padding: 4px; }
QMenu::item { padding: 8px 28px; font-size: 12px; }
QMenu::item:selected { background-color: #0078D4; color: #FFF; border-radius: 4px; }
QToolTip { background-color: #FFF; color: #333; border: 1px solid #C0C0C0; padding: 6px 10px; border-radius: 4px; font-size: 11px; }
QGroupBox { border: 1px solid #E0E0E0; border-radius: 8px; margin-top: 14px; padding-top: 16px;
    font-weight: 600; color: #333; background-color: #FFF; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #0078D4; }
QLineEdit, QComboBox { border: 1px solid #C0C0C0; border-radius: 4px; padding: 6px 8px;
    background-color: #FFF; color: #333; font-size: 12px; }
QLineEdit:focus, QComboBox:focus { border: 2px solid #0078D4; padding: 5px 7px; }
QComboBox::drop-down { border: none; padding-right: 4px; }
QCheckBox { spacing: 6px; color: #333; }
QProgressBar { border: 1px solid #D0D0D0; border-radius: 4px; background: #EEE; text-align: center; font-size: 11px; color: #333; }
QProgressBar::chunk { background-color: #0078D4; border-radius: 3px; }
QTableWidget, QTableView { background: #FFF; border: 1px solid #E0E0E0; gridline-color: #F0F0F0;
    selection-background-color: #0078D4; selection-color: #FFF; color: #333; }
QHeaderView::section { background: #F5F5F5; color: #555; padding: 6px 10px; border: none;
    border-bottom: 2px solid #D0D0D0; font-weight: 600; font-size: 11px; }
QSplitter::handle { background: #D0D0D0; width: 2px; }
QScrollBar:horizontal, QScrollBar:vertical { background: #F5F5F5; border: none; width: 10px; height: 10px; }
QScrollBar::handle { background: #C0C0C0; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:hover { background: #A0A0A0; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
"""

SIDEBAR_CATEGORIES = [
    ("TRAINING", [
        ("PoseTrain",       "🏋", "Pose Training"),
        ("SynthValidate",   "✅", "3-Stage Behavior Prediction"),
    ]),
    ("FACTOR MINING", [
        ("Discovery",       "🔍", "Discovery"),
        ("BatchMining",     "⚡", "Batch Mining"),
    ]),
    ("FACTOR ENGINEERING", [
        ("Evolution",       "🧬", "Factor Evolution"),
        ("Tuner",           "🎯", "Factor Tuner"),
        ("Manager",         "📋", "Factor Manager"),
        ("Correlation",     "🔗", "Correlation"),
        ("Analysis",        "📊", "Factor Analysis"),
    ]),
    ("PREDICT", [
        ("Inference",       "🚀", "Inference"),
        ("Validation",      "💾", "Validation"),
    ]),
    ("TOOLS", [
        ("Config",          "⚙", "Config Editor"),
        ("SSH",             "🔌", "SSH Remote"),
    ]),
]

SIDEBAR_EXPANDED = 240
SIDEBAR_COLLAPSED = 52


class SidebarButton(QPushButton):
    """Sidebar nav button — supports compact (icon-only) and expanded modes."""
    def __init__(self, icon, text, tab_key="", parent=None):
        super().__init__(parent)
        self._tab_key = tab_key
        self._icon = icon
        self._text = text
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(text)
        self.set_compact(True)

    def set_compact(self, compact):
        if compact:
            self.setText(self._icon)
            self.setStyleSheet("""
                QPushButton {
                    text-align: center;
                    padding: 10px 4px; border: none; border-radius: 8px;
                    margin: 2px 6px; font-size: 16px; font-weight: bold;
                    color: #555; background: transparent;
                }
                QPushButton:hover { background: #E8F0FE; color: #0078D4; }
                QPushButton:checked { background: #0078D4; color: #FFF; }
            """)
        else:
            self.setText(f"  {self._icon}  {self._text}")
            self.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding: 11px 14px; border: none; border-radius: 8px;
                    margin: 2px 8px; font-size: 13px; color: #555; background: transparent;
                }
                QPushButton:hover { background: #E8F0FE; color: #0078D4; }
                QPushButton:checked { background: #0078D4; color: #FFF; font-weight: 600; }
            """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._tabs: dict = {}
        self._content_stack = None
        self._sidebar_buttons: list = []

        self.setWindowTitle("SABER")
        self.setMinimumSize(1200, 750)
        self.resize(1200, 750)

        self.setStyleSheet(LIGHT_STYLE)
        self._build_menu_bar()
        self._build_status_bar()
        self._build_central()

    def _build_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open Config...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_config)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(QAction("E&xit", self, shortcut=QKeySequence.StandardKey.Quit, triggered=self.close))
        tools_menu = menubar.addMenu("&Tools")
        tools_menu.addAction(QAction("Clear All Logs", self, triggered=self._on_clear_all_logs))
        tools_menu.addAction(QAction("Open Output Dir", self, triggered=self._on_open_output_dir))
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(QAction("&About", self, triggered=self._on_about))

    def _build_status_bar(self):
        self._status_bar = StatusBarWidget()
        self.setStatusBar(self._status_bar)

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- Left sidebar ----
        self._sidebar = QFrame()
        self._sidebar.setFixedWidth(SIDEBAR_COLLAPSED)
        self._sidebar.setStyleSheet("QFrame { background: #FFF; border-right: 1px solid #E5E5E5; }")
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Toggle button
        self._sidebar_toggle = QPushButton("☰")
        self._sidebar_toggle.setStyleSheet(
            "QPushButton { background: transparent; border: none; font-size: 18px; "
            "padding: 10px; color: #888; }"
            "QPushButton:hover { color: #0078D4; }"
        )
        self._sidebar_toggle.setCursor(Qt.PointingHandCursor)
        self._sidebar_toggle.clicked.connect(self._toggle_sidebar)
        sidebar_layout.addWidget(self._sidebar_toggle)

        # Scrollable button area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #FFF; }")

        btn_container = QWidget()
        btn_container.setStyleSheet("background: #FFF;")
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(0, 4, 0, 4)
        btn_layout.setSpacing(0)

        self._sidebar_buttons = []
        self._btn_by_key = {}
        self._sidebar_cats = []  # category labels
        self._sidebar_expanded = False

        for cat_idx, (cat_name, items) in enumerate(SIDEBAR_CATEGORIES):
            # Separator before each category (except first)
            if cat_idx > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.HLine)
                sep.setStyleSheet("QFrame{border:none;border-top:2px solid #E0E0E0;margin:6px 10px;}")
                sep.setFixedHeight(2)
                btn_layout.addWidget(sep)

            # Category label — only visible when expanded
            cat_lbl = QLabel(cat_name)
            cat_lbl.setStyleSheet(
                "color:#999;font-size:10px;font-weight:700;letter-spacing:1px;"
                "padding:8px 16px 2px 16px;background:#FFF;"
            )
            cat_lbl.setVisible(False)
            btn_layout.addWidget(cat_lbl)
            self._sidebar_cats.append(cat_lbl)

            for tab_key, icon, full_text in items:
                btn = SidebarButton(icon, full_text, tab_key)
                btn.clicked.connect(self._make_switcher(tab_key))
                btn_layout.addWidget(btn)
                self._sidebar_buttons.append(btn)
                self._btn_by_key[tab_key] = btn

        btn_layout.addStretch()
        scroll.setWidget(btn_container)
        sidebar_layout.addWidget(scroll)

        layout.addWidget(self._sidebar)

        # ---- Content area ----
        self._content_stack = QStackedWidget()
        self._content_stack.setStyleSheet("background: #F5F5F5;")

        tab_classes = {
            "Discovery":      DiscoveryTab,
            "SynthValidate":  TrainBehaviorTab,
            "Evolution":      EvolutionTab,
            "Analysis":       AnalysisTab,
            "Manager":        ManagerTab,
            "Correlation":    CorrelationTab,
            "Tuner":          TunerTab,
            "BatchMining":    BatchMiningTab,
            "Inference":      InferenceTab,
            "Validation":     ValidationTab,
            "Config":         ConfigTab,
            "SSH":            SSHTab,
            "PoseTrain":      TrainTab,
        }
        for key, cls in tab_classes.items():
            tab = cls()
            self._tabs[key] = tab
            self._content_stack.addWidget(tab)

        layout.addWidget(self._content_stack)

        # Wire SSH
        if "SSH" in self._tabs:
            self._tabs["SSH"].set_main_window(self)

        # Default: select first module
        first_key = SIDEBAR_CATEGORIES[0][1][0][0]
        if first_key in self._btn_by_key:
            self._btn_by_key[first_key].setChecked(True)
        if first_key in self._tabs:
            self._content_stack.setCurrentWidget(self._tabs[first_key])

    def _toggle_sidebar(self):
        self._sidebar_expanded = not self._sidebar_expanded
        if self._sidebar_expanded:
            self._sidebar.setFixedWidth(SIDEBAR_EXPANDED)
            self._sidebar_toggle.setText("✕")
        else:
            self._sidebar.setFixedWidth(SIDEBAR_COLLAPSED)
            self._sidebar_toggle.setText("☰")
        for btn in self._sidebar_buttons:
            btn.set_compact(not self._sidebar_expanded)
        for cat in self._sidebar_cats:
            if isinstance(cat, QLabel):
                cat.setVisible(self._sidebar_expanded)
        # Collapse on clicking a module
        if not self._sidebar_expanded:
            pass  # stays collapsed

    def _make_switcher(self, tab_key):
        def switch():
            for btn in self._sidebar_buttons:
                btn.setChecked(False)
            if tab_key in self._btn_by_key:
                self._btn_by_key[tab_key].setChecked(True)
            if tab_key in self._tabs:
                self._content_stack.setCurrentWidget(self._tabs[tab_key])
            # Auto-collapse sidebar after selection
            if self._sidebar_expanded:
                self._toggle_sidebar()
        return switch

    def _on_open_config(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Open Config", "config", "YAML (*.yaml *.yml)")
        if path and "Config" in self._btn_by_key:
            self._make_switcher("Config")()

    def _on_clear_all_logs(self):
        for tab in self._tabs.values():
            if hasattr(tab, '_log_viewer'):
                tab._log_viewer.clear_log()

    def _on_open_output_dir(self):
        import os
        os.startfile(str(Path("memory").resolve()))

    def _on_about(self):
        QMessageBox.about(self, "About",
            "<h3>SABER</h3>"
            "<p>System for Automated Behavioral factor Evaluation and Refinement<br>"
            "for mouse behavior analysis research.</p>")
