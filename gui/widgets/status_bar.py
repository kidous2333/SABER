"""
StatusBarWidget — Extended status bar with progress and SSH indicator.
"""

from PySide6.QtWidgets import QStatusBar, QProgressBar, QLabel, QWidget, QHBoxLayout
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor


class StatusBarWidget(QStatusBar):
    """
    Main window status bar with:
      - Task description label
      - Progress bar
      - SSH connection indicator (dot + hostname)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QStatusBar { background: #FAFAFA; color: #555; border-top: 1px solid #E0E0E0; }")

        # Task label
        self._task_label = QLabel("  Ready  ")
        self._task_label.setStyleSheet("color: #666;")
        self.addWidget(self._task_label)

        # Progress bar (compact)
        self._prog_bar = QProgressBar()
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setValue(0)
        self._prog_bar.setFixedWidth(150)
        self._prog_bar.setFixedHeight(16)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #D0D0D0;
                border-radius: 3px;
                background: #EEE;
            }
            QProgressBar::chunk {
                background: #0078D4;
                border-radius: 2px;
            }
        """)
        self._prog_bar.setVisible(False)
        self.addPermanentWidget(self._prog_bar)

        # SSH indicator
        self._ssh_indicator = QLabel(" SSH: Disconnected ")
        self._ssh_indicator.setStyleSheet(
            "color: #999; font-size: 11px; padding: 0 8px;"
        )
        self.addPermanentWidget(self._ssh_indicator)

    @Slot(str)
    def set_task(self, text: str):
        self._task_label.setText(f"  {text}  ")

    @Slot(int)
    def set_progress(self, pct: int):
        if pct > 0 and pct < 100:
            self._prog_bar.setVisible(True)
            self._prog_bar.setValue(pct)
        else:
            self._prog_bar.setVisible(False)
            self._prog_bar.setValue(0)

    def set_ssh_status(self, connected, host=""):
        if connected:
            self._ssh_indicator.setText(f" 🔗 SSH: {host} ")
            self._ssh_indicator.setStyleSheet(
                "color: #16A34A; font-weight: 600; font-size: 11px; padding: 0 8px;"
            )
        else:
            self._ssh_indicator.setText(" SSH: Disconnected ")
            self._ssh_indicator.setStyleSheet(
                "color: #999; font-size: 11px; padding: 0 8px;"
            )
