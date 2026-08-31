"""
FactorTable — Sortable, filterable table for factor data.

Uses QTableView + QStandardItemModel + QSortFilterProxyModel.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableView, QHeaderView,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, QSortFilterProxyModel, QAbstractTableModel, QModelIndex, Signal
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor
from typing import List, Dict, Optional, Callable

# Default column spec: (header, width, extractor_fn_or_key)
# If extractor is a string key → item[key]; if callable → item = callable(factor, i)
_DEFAULT_COLUMNS = [
    ("#",   30,  lambda f, i: i + 1),
    ("Name", 160, "name"),
    ("Seq",  42,  "seq_length"),
    ("AUC",  60,  "best_auc"),
    ("wAUC", 60,  "wgt_auc"),
    ("OK",   38,  "n_valid"),
    ("Target", 100, "target"),
    ("Valid Classes", 90, "vc_str"),
]


def _make_item(value, col_idx: int, numeric_cols: set) -> QStandardItem:
    """Create a centered QStandardItem. Numeric cols get float UserRole for sorting."""
    if value is None or value == "":
        text = "?"
    elif isinstance(value, float):
        text = f"{value:.4f}"
    else:
        text = str(value)
    item = QStandardItem(text)
    item.setTextAlignment(Qt.AlignCenter)
    if col_idx in numeric_cols:
        try:
            item.setData(float(value), Qt.UserRole)
        except (ValueError, TypeError):
            pass
    return item


class NumericSortProxy(QSortFilterProxyModel):
    """Proxy that sorts numeric columns properly via float comparison."""

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        l_data = left.data(Qt.UserRole)
        r_data = right.data(Qt.UserRole)
        if l_data is not None and r_data is not None:
            try:
                return float(l_data) < float(r_data)
            except (ValueError, TypeError):
                pass
        return str(left.data() or "").lower() < str(right.data() or "").lower()


class FactorTableModel(QStandardItemModel):
    """Custom table model for factor data. Supports custom column specs."""

    def __init__(self, columns=None, parent=None):
        self._col_spec = columns or _DEFAULT_COLUMNS
        headers = [c[0] for c in self._col_spec]
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)

    @property
    def column_widths(self):
        return [c[1] for c in self._col_spec]

    def _build_row(self, f: Dict, i: int) -> List[QStandardItem]:
        """Build a single row of QStandardItems from a factor dict."""
        best_auc = f.get("best_auc") if f.get("best_auc") is not None else (f.get("auc") if f.get("auc") is not None else None)

        vcs = f.get("valid_classes", [])
        if vcs:
            wgt_auc = sum(vc.get("auc", 0) for vc in vcs) / len(vcs)
        else:
            wgt_auc = best_auc
        n_valid = len(vcs)
        target = f.get("target", "")
        vc_str = ", ".join(
            f"{vc.get('class','?')}({vc.get('auc',0):.2f})"
            for vc in vcs[:5]
        )
        if len(vcs) > 5:
            vc_str += f" +{len(vcs)-5}"

        seq = f.get("seq_length")
        seq_str = str(seq) if seq is not None else "?"

        derived = {
            "best_auc": best_auc, "wgt_auc": wgt_auc,
            "n_valid": n_valid, "target": target, "vc_str": vc_str,
            "seq_length": seq_str, "name": f.get("name", f"factor_{i}"),
            "_auc_improvement": f.get("_auc_improvement"),
            "_mutation": f.get("_mutation", ""),
            "_param_label": f.get("_param_label", ""),
        }

        items = []
        for ci, (header, width, extractor) in enumerate(self._col_spec):
            if callable(extractor):
                value = extractor(f, i)
            elif isinstance(extractor, str):
                value = derived.get(extractor, f.get(extractor, ""))
            else:
                value = ""
            item = _make_item(value, ci, {0, 2, 3, 4, 5})
            items.append(item)

        # Color-code AUC column (always col 3 in default spec)
        if best_auc >= 0.80:
            items[3].setForeground(QColor("#40C040"))
        elif best_auc >= 0.70:
            items[3].setForeground(QColor("#FFB020"))
        else:
            items[3].setForeground(QColor("#FF6060"))
        return items

    def load_factors(self, factors: List[Dict]):
        """Populate table from list of factor dicts using column spec."""
        self.removeRows(0, self.rowCount())
        for i, f in enumerate(factors):
            self.appendRow(self._build_row(f, i))

    def append_factor(self, factor: Dict, index: int):
        """Append a single factor row (for real-time updates during evolution)."""
        self.appendRow(self._build_row(factor, index))

    def get_factor_data(self, row: int) -> Optional[Dict]:
        if row < 0 or row >= self.rowCount():
            return None
        return {
            "name": self.item(row, 1).text(),
            "seq_length": self.item(row, 2).text(),
            "best_auc": float(self.item(row, 3).text()),
        }


class FactorTable(QWidget):
    """
    Sortable/filterable factor data table with double-click detail.

    Usage:
        table = FactorTable()                    # default columns
        table = FactorTable(columns=custom_spec)  # custom columns
        table.factor_double_clicked.connect(on_detail)
        table.load_factors(factor_list)
    """

    factor_double_clicked = Signal(dict)

    def __init__(self, parent=None, columns=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._model = FactorTableModel(columns=columns)
        self._proxy = NumericSortProxy()
        self._proxy.setSourceModel(self._model)
        self._proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self._proxy.setFilterKeyColumn(-1)

        self._table = QTableView()
        self._table.setModel(self._proxy)
        self._table.setSortingEnabled(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setStretchLastSection(False)
        vh = self._table.verticalHeader()
        vh.setVisible(False)
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setDefaultSectionSize(26)
        self._table.doubleClicked.connect(self._on_double_click)
        self._table.setStyleSheet("""
            QTableView {
                background-color: #FFF;
                color: #333;
                gridline-color: #F0F0F0;
                border: 1px solid #E0E0E0;
                font-size: 11px;
            }
            QTableView::item:selected {
                background-color: #0078D4;
                color: #FFF;
            }
            QHeaderView::section {
                background-color: #F5F5F5;
                color: #555;
                padding: 6px 10px;
                border: none;
                border-bottom: 2px solid #D0D0D0;
                font-weight: 600;
            }
        """)
        layout.addWidget(self._table)
        for i, w in enumerate(self._model.column_widths):
            self._table.setColumnWidth(i, w)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._factors = []

    def load_factors(self, factors: List[Dict]):
        self._factors = list(factors)
        self._model.load_factors(factors)

    def append_factor(self, factor: Dict):
        """Append a single factor row in real-time (e.g. during evolution)."""
        self._factors.append(factor)
        self._model.append_factor(factor, len(self._factors) - 1)

    def _on_double_click(self, index):
        source_idx = self._proxy.mapToSource(index)
        row = source_idx.row()
        if 0 <= row < len(self._factors):
            self.factor_double_clicked.emit(self._factors[row])

    def set_filter(self, text: str):
        self._proxy.setFilterFixedString(text)

    def clear(self):
        self._factors = []
        self._model.removeRows(0, self._model.rowCount())
