"""KeyValueEditor — Editable key-value pair table with +/- buttons."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QHeaderView, QSizePolicy,
)
from PySide6.QtCore import Qt


class KeyValueEditor(QWidget):
    """Inline key-value pair editor with add/remove row support.

    Usage:
        editor = KeyValueEditor(title="Label Map")
        editor.set_pairs({"explore_object": 0, "climb": 1})
        pairs = editor.get_pairs()  # -> {"explore_object": "0", ...}
    """

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        # Table: 2 columns + remove button column
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Key", "Value", ""])
        self._table.verticalHeader().setVisible(False)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.Fixed)
        hdr.resizeSection(2, 28)
        self._table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._table.verticalHeader().setDefaultSectionSize(26)
        self._table.horizontalHeader().setFixedHeight(24)
        self._table.setMaximumHeight(26 * 10 + 4)
        lay.addWidget(self._table)

        # Add row button
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 2, 0, 0)
        self._add_btn = QPushButton("+ Add Row")
        self._add_btn.setFixedHeight(22)
        self._add_btn.setCursor(Qt.PointingHandCursor)
        self._add_btn.setStyleSheet(
            "QPushButton{background:#F5F5F5;color:#555;border:1px solid #D0D0D0;"
            "border-radius:4px;font-size:10px;}"
            "QPushButton:hover{background:#E8F0FE;color:#0078D4;}")
        self._add_btn.clicked.connect(self._add_row)
        btn_row.addWidget(self._add_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

    def _add_row(self):
        row = self._table.rowCount()
        self._table.insertRow(row)
        # Key cell
        key_item = QTableWidgetItem("")
        self._table.setItem(row, 0, key_item)
        # Value cell
        val_item = QTableWidgetItem("")
        self._table.setItem(row, 1, val_item)
        # Remove button
        rm_btn = QPushButton("✕")
        rm_btn.setFixedSize(22, 22)
        rm_btn.setCursor(Qt.PointingHandCursor)
        rm_btn.setStyleSheet(
            "QPushButton{background:#FFF;color:#AAA;border:none;font-size:12px;}"
            "QPushButton:hover{color:#DC2626;}")
        rm_btn.clicked.connect(lambda checked=False, r=row: self._remove_row(r))
        self._table.setCellWidget(row, 2, rm_btn)

    def _remove_row(self, row: int):
        if 0 <= row < self._table.rowCount():
            self._table.removeRow(row)
            # Re-bind remove buttons (row indices shifted)
            for r in range(self._table.rowCount()):
                btn = self._table.cellWidget(r, 2)
                if btn:
                    try:
                        btn.clicked.disconnect()
                    except Exception:
                        pass
                    btn.clicked.connect(lambda checked=False, _r=r: self._remove_row(_r))

    def get_pairs(self) -> dict:
        """Return current key-value pairs as dict {key: value} (both strings)."""
        result = {}
        for r in range(self._table.rowCount()):
            key_item = self._table.item(r, 0)
            val_item = self._table.item(r, 1)
            key = key_item.text().strip() if key_item else ""
            val = val_item.text().strip() if val_item else ""
            if key:
                result[key] = val
        return result

    def set_pairs(self, pairs: dict):
        """Populate from a dict. Keys and values are converted to strings."""
        self._table.setRowCount(0)
        for key, val in pairs.items():
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(str(key)))
            self._table.setItem(row, 1, QTableWidgetItem(str(val)))
            rm_btn = QPushButton("✕")
            rm_btn.setFixedSize(22, 22)
            rm_btn.setCursor(Qt.PointingHandCursor)
            rm_btn.setStyleSheet(
                "QPushButton{background:#FFF;color:#AAA;border:none;font-size:12px;}"
                "QPushButton:hover{color:#DC2626;}")
            rm_btn.clicked.connect(lambda checked=False, r=row: self._remove_row(r))
            self._table.setCellWidget(row, 2, rm_btn)
