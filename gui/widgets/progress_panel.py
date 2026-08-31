"""
ProgressPanel — Composite progress bar + status label + ETA.

Reusable across all tabs for showing computation progress.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QProgressBar, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, Signal, Slot
import time


class ProgressPanel(QWidget):
    """
    Horizontal progress bar with percentage text, status label, and ETA.

    Usage:
        panel = ProgressPanel()
        worker.progress.connect(panel.update_progress)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._start_time = 0.0

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: #888; font-size: 10px; padding: 0;")
        layout.addWidget(self._status_label)

        # Progress bar row
        bar_row = QHBoxLayout()

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFormat("%p%")
        self._bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #D0D0D0;
                border-radius: 3px;
                background-color: #EEE;
                text-align: center;
                height: 14px;
                color: #333;
                font-size: 9px;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 2px;
            }
        """)
        bar_row.addWidget(self._bar)

        # ETA label
        self._eta_label = QLabel("")
        self._eta_label.setStyleSheet("color: #888; font-size: 10px;")
        bar_row.addWidget(self._eta_label)

        layout.addLayout(bar_row)

    @Slot(int, str)
    def update_progress(self, pct: int, status: str = ""):
        """Update progress bar and status text."""
        if self._start_time == 0.0 and pct > 0:
            self._start_time = time.time()

        self._bar.setValue(min(pct, 100))
        if status:
            self._status_label.setText(status)
        elif pct >= 100:
            self._status_label.setText("Complete")
        else:
            self._status_label.setText(f"{pct}%")

        # ETA estimation
        elapsed = time.time() - self._start_time
        if pct > 0 and pct < 100 and elapsed > 1:
            eta_sec = (elapsed / pct) * (100 - pct)
            if eta_sec > 3600:
                self._eta_label.setText(f"ETA: {eta_sec/3600:.1f}h")
            elif eta_sec > 60:
                self._eta_label.setText(f"ETA: {eta_sec/60:.0f}min")
            else:
                self._eta_label.setText(f"ETA: {eta_sec:.0f}s")
        elif pct >= 100:
            self._eta_label.setText(f"Done ({elapsed:.0f}s)")
        else:
            self._eta_label.setText("")

    def reset(self):
        """Reset to initial state."""
        self._bar.setValue(0)
        self._status_label.setText("Ready")
        self._eta_label.setText("")
        self._start_time = 0.0
