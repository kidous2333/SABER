"""ConfigTab — YAML configuration editor with structured tree view."""
from pathlib import Path
from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel,
    QMessageBox, QFileDialog, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QLineEdit,
)
from PySide6.QtCore import Qt, Slot, Signal
from gui.utils.config_loader import list_config_files
from gui.tabs.base_tab import BaseTab


# ---------------------------------------------------------------------------
# Tree ↔ dict helpers
# ---------------------------------------------------------------------------

def _value_type_str(v):
    """Short type label for a leaf value."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, list):
        return f"[{len(v)} items]"
    if isinstance(v, dict):
        return f"{{{len(v)} keys}}"
    return "str"


def _is_leaf(v):
    return not isinstance(v, (dict, list))


def _leaf_display(v):
    """Pretty-print a scalar value for display in the Value column."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    return str(v)


def _parse_leaf(text: str):
    """Convert edited text back to a Python scalar."""
    t = text.strip()
    if t in ("null", "None", ""):
        return None
    if t.lower() == "true":
        return True
    if t.lower() == "false":
        return False
    # int
    try:
        return int(t)
    except ValueError:
        pass
    # float
    try:
        return float(t)
    except ValueError:
        pass
    return t  # keep as string


def _build_item(parent, key, value, editable_values=True):
    """Recursively build one QTreeWidgetItem from key+value under *parent*."""
    if isinstance(value, dict):
        item = QTreeWidgetItem(parent, [str(key), "", _value_type_str(value)])
        for k, v in value.items():
            _build_item(item, k, v, editable_values)
        return item
    elif isinstance(value, list):
        item = QTreeWidgetItem(parent, [str(key), "", _value_type_str(value)])
        for i, v in enumerate(value):
            _build_item(item, f"[{i}]", v, editable_values)
        return item
    else:
        # leaf
        flags = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if editable_values:
            flags |= Qt.ItemIsEditable
        item = QTreeWidgetItem(parent, [str(key), _leaf_display(value), _value_type_str(value)])
        item.setFlags(flags)
        return item


def _populate_tree(tree: QTreeWidget, data: dict):
    """Clear the tree and fill it from a top-level dict."""
    tree.clear()
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        _build_item(tree, key, value, editable_values=True)


def _collect_item(item: QTreeWidgetItem):
    """Recursively reconstruct a Python value from a tree item."""
    child_count = item.childCount()
    if child_count == 0:
        # leaf node
        return _parse_leaf(item.text(1))
    # Check whether children look like list indices (key == "[0]", "[1]", ...)
    keys = [item.child(i).text(0) for i in range(child_count)]
    is_list = all(k.startswith("[") and k.endswith("]") for k in keys)
    if is_list:
        result = []
        for i in range(child_count):
            result.append(_collect_item(item.child(i)))
        return result
    else:
        result = {}
        for i in range(child_count):
            child = item.child(i)
            result[child.text(0)] = _collect_item(child)
        return result


def _collect_tree(tree: QTreeWidget) -> dict:
    """Walk the tree and rebuild the top-level dict."""
    result = {}
    for i in range(tree.topLevelItemCount()):
        top = tree.topLevelItem(i)
        result[top.text(0)] = _collect_item(top)
    return result


# ---------------------------------------------------------------------------
# ConfigTab
# ---------------------------------------------------------------------------

class ConfigTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(title="YAML Configuration Editor", tab_key="config", parent=parent)
        self._current_path = ""
        self._dirty = False

    # ---- parameter bar (file selector + actions) ----

    def setup_params(self):
        # Row 1: file selector
        row = QHBoxLayout()
        row.addWidget(QLabel("File:"))
        self._file_combo = QComboBox()
        self._file_combo.setEditable(False)
        self._file_combo.setMinimumWidth(200)
        self._file_combo.currentTextChanged.connect(self._on_file_selected)
        row.addWidget(self._file_combo)
        self._param_layout.insertLayout(self._param_layout.count() - 1, row)

        # Row 2: actions
        row2 = QHBoxLayout()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._on_browse)
        row2.addWidget(browse_btn)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._on_reload)
        row2.addWidget(reload_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.setStyleSheet(
            "QPushButton{background:#0078D4;color:#FFF;border:none;border-radius:6px;"
            "padding:8px 18px;font-weight:600;}QPushButton:hover{background:#006CBE;}"
        )
        self._save_btn.clicked.connect(self._on_save)
        row2.addWidget(self._save_btn)
        validate_btn = QPushButton("Validate")
        validate_btn.clicked.connect(self._on_validate)
        row2.addWidget(validate_btn)
        self._param_layout.insertLayout(self._param_layout.count() - 1, row2)

        self._refresh_file_list()

    # ---- main area (tree) ----

    def setup_results(self):
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Key", "Value", "Type"])
        self._tree.setAlternatingRowColors(True)
        self._tree.setAnimated(True)
        self._tree.setStyleSheet(
            "QTreeWidget{background:#FFF;color:#333;border:1px solid #D0D0D0;"
            "border-radius:4px;font-size:12px;}"
            "QTreeWidget::item{padding:2px 4px;}"
            "QTreeWidget::item:selected{background:#0078D4;color:#FFF;}"
        )
        hdr = self._tree.header()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)   # Key
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)   # Value
        hdr.setSectionResizeMode(2, QHeaderView.Fixed)     # Type
        hdr.resizeSection(2, 90)

        # Inline editing: commit on Enter / focus lost
        self._tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._tree.itemChanged.connect(self._on_item_changed)

        self._results_layout_main.addWidget(self._tree)

    # ---- file operations ----

    def _refresh_file_list(self):
        self._file_combo.blockSignals(True)
        self._file_combo.clear()
        self._file_combo.addItem("— select a file —", "")
        for f in list_config_files("config"):
            self._file_combo.addItem(f, f)
        self._file_combo.blockSignals(False)

    @Slot(str)
    def _on_file_selected(self, text):
        path = self._file_combo.currentData()
        if not path:
            return
        self._load_file(path)

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open YAML", "config", "YAML (*.yaml *.yml)"
        )
        if path:
            self._load_file(path)
            # Add to combo if not present
            found = False
            for i in range(self._file_combo.count()):
                if self._file_combo.itemData(i) == path:
                    self._file_combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self._file_combo.blockSignals(True)
                self._file_combo.addItem(path, path)
                self._file_combo.setCurrentIndex(self._file_combo.count() - 1)
                self._file_combo.blockSignals(False)

    def _on_reload(self):
        if self._current_path:
            self._load_file(self._current_path)

    def _load_file(self, path: str):
        import yaml
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                data = {}
            _populate_tree(self._tree, data)
            self._tree.expandToDepth(1)  # show first 2 levels
            self._current_path = path
            self._dirty = False
            self._update_title()
            self._log_viewer.append_colored(f"Loaded: {path}", 20)
        except Exception as e:
            self._log_viewer.append_colored(f"Error loading {path}: {e}", 40)

    def _on_save(self):
        import yaml
        if not self._current_path:
            QMessageBox.warning(self, "Save", "No file selected.")
            return
        try:
            data = _collect_tree(self._tree)
            with open(self._current_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            self._dirty = False
            self._update_title()
            self._log_viewer.append_colored(f"Saved: {self._current_path}", 20)
        except Exception as e:
            self._log_viewer.append_colored(f"Save failed: {e}", 40)

    def _on_validate(self):
        import yaml
        try:
            data = _collect_tree(self._tree)
            yaml.safe_dump(data)
            QMessageBox.information(self, "Validate", "Configuration structure is valid.")
        except Exception as e:
            QMessageBox.warning(self, "Validate", f"Error:\n{e}")

    # ---- tree editing ----

    @Slot(QTreeWidgetItem, int)
    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Only allow editing leaf values in column 1."""
        if item.childCount() > 0:
            return  # non-leaf nodes are not editable
        if column != 1:
            return
        # Qt handles inline editing when EditTriggers + flags allow it;
        # make sure the item is editable (flags set in _build_item).
        self._tree.editItem(item, 1)

    @Slot(QTreeWidgetItem, int)
    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """Update Type column when a leaf value is edited."""
        if item.childCount() > 0 or column != 1:
            return
        new_val = _parse_leaf(item.text(1))
        item.setText(2, _value_type_str(new_val))
        self._dirty = True
        self._update_title()

    # ---- title bar ----

    def _update_title(self):
        name = Path(self._current_path).name if self._current_path else "Untitled"
        marker = " *" if self._dirty else ""
        self._title_label.setText(f"{name}{marker}")
