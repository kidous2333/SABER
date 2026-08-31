"""
LogViewer — Color-coded real-time log display widget.

Uses QPlainTextEdit with QTextCharFormat for per-level coloring.
Auto-truncates at 10,000 lines to prevent UI slowdown.
"""

from PySide6.QtWidgets import (
    QPlainTextEdit, QWidget, QVBoxLayout, QHBoxLayout,
    QCheckBox, QComboBox, QPushButton, QLabel,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor, QFont


MAX_LINES = 10000

# Log level → color mapping
LEVEL_COLORS = {
    10: QColor("#888888"),            # DEBUG: gray
    20: QColor("#333333"),            # INFO: dark
    30: QColor("#D97706"),            # WARNING: amber
    40: QColor("#DC2626"),            # ERROR: red
    50: QColor("#991B1B"),            # CRITICAL: dark red
}

LEVEL_NAMES = {
    10: "DEBUG",
    20: "INFO",
    30: "WARNING",
    40: "ERROR",
    50: "CRITICAL",
}


class LogViewer(QWidget):
    """
    Real-time log display with auto-scroll, level filtering, and color coding.

    Usage:
        viewer = LogViewer()
        # From worker signal:
        worker.log_line.connect(viewer.append_colored)
        # Or directly:
        viewer.append("Hello", logging.INFO)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._line_count = 0

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()

        self._auto_scroll_cb = QCheckBox("Auto-scroll")
        self._auto_scroll_cb.setChecked(True)
        toolbar.addWidget(self._auto_scroll_cb)

        toolbar.addWidget(QLabel("Filter:"))
        self._level_filter = QComboBox()
        self._level_filter.addItem("ALL", -1)
        for lvl in sorted(LEVEL_NAMES):
            self._level_filter.addItem(LEVEL_NAMES[lvl], lvl)
        self._level_filter.setCurrentIndex(0)
        toolbar.addWidget(self._level_filter)

        self._line_count_label = QLabel("0 lines")
        self._line_count_label.setStyleSheet("color: #666;")
        toolbar.addWidget(self._line_count_label)

        toolbar.addStretch()

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clear_log)
        toolbar.addWidget(self._clear_btn)

        layout.addLayout(toolbar)

        # Text area
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(MAX_LINES)
        font = QFont("Consolas", 9)
        self._text.setFont(font)
        self._text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #FFF;
                color: #333;
                border: 1px solid #D0D0D0;
                border-radius: 4px;
                font-size: 11px;
            }
        """)
        layout.addWidget(self._text)

    @Slot(str, int)
    def append_colored(self, message, level=20):
        """Append a color-coded log message. Thread-safe via queued connection."""
        # Check level filter
        filter_level = self._level_filter.currentData()
        if filter_level != -1 and level < filter_level:
            return

        color = LEVEL_COLORS.get(level, QColor("#E0E0E0"))
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        if level >= 50:  # CRITICAL
            fmt.setFontWeight(QFont.Bold)
            fmt.setBackground(QColor("#3A0000"))

        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(message + "\n", fmt)

        if self._auto_scroll_cb.isChecked():
            self._text.setTextCursor(cursor)
            self._text.ensureCursorVisible()

        self._line_count += 1
        self._line_count_label.setText(f"{self._line_count} lines")

    @Slot()
    def clear_log(self):
        """Clear all log entries."""
        self._text.clear()
        self._line_count = 0
        self._line_count_label.setText("0 lines")

    def set_auto_scroll(self, enabled: bool):
        self._auto_scroll_cb.setChecked(enabled)
