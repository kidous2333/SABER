"""
ParameterGroup — Form widgets auto-built from schema dict.

All numeric inputs use QLineEdit with validators (NO spinboxes — no scroll wheels).
Designed to be embedded in a popup QDialog.
"""

from PySide6.QtWidgets import (
    QWidget, QFormLayout, QGroupBox, QLineEdit,
    QComboBox, QCheckBox, QPushButton,
    QHBoxLayout, QVBoxLayout, QFileDialog, QLabel,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator, QDoubleValidator
from pathlib import Path


class PathListEditor(QWidget):
    """A compact path list display with an Edit button that opens a full-size dialog."""

    value_changed = Signal(object)

    def __init__(self, mode="dir", parent=None):
        super().__init__(parent)
        self._mode = mode
        self._paths = [""]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._summary = QLineEdit()
        self._summary.setReadOnly(True)
        self._summary.setStyleSheet("QLineEdit{background:#F8F8F8;color:#666;border:1px solid #C0C0C0;border-radius:4px;padding:5px 8px;font-size:11px;}")
        self._update_summary()
        layout.addWidget(self._summary, 1)

        edit_btn = QPushButton("Edit Paths")
        edit_btn.setStyleSheet("""
            QPushButton { background:#0078D4; color:#FFF; border:none; border-radius:4px;
                padding:6px 12px; font-size:11px; font-weight:600; }
            QPushButton:hover { background:#006CBE; }
        """)
        edit_btn.clicked.connect(self._open_editor)
        layout.addWidget(edit_btn)

    def _update_summary(self):
        n = len(self._paths)
        if n == 0:
            self._summary.setText("(no paths)")
        elif n <= 2:
            self._summary.setText("; ".join(self._paths))
        else:
            self._summary.setText(f"{n} paths: {self._paths[0]}; ... +{n-1} more")

    def _open_editor(self):
        """Open a large dedicated dialog for editing paths."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QScrollArea, QFileDialog
        from pathlib import Path

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Paths")
        dlg.resize(800, 550)
        dlg.setMinimumWidth(650)
        dlg.setMinimumHeight(400)

        dlg_layout = QVBoxLayout(dlg)

        # Toolbar
        toolbar = QHBoxLayout()

        add_btn = QPushButton("＋ Add Single")
        add_btn.setStyleSheet("QPushButton{background:#F0F0F0;color:#0078D4;border:1px solid #D0D0D0;border-radius:4px;padding:6px 14px;font-size:12px;font-weight:600;}QPushButton:hover{background:#E0E8F0;}")
        toolbar.addWidget(add_btn)

        batch_label = "📁 Batch Add Files" if self._mode != "dir" else "📁 Batch Add Folders"
        batch_btn = QPushButton(batch_label)
        batch_btn.setStyleSheet("QPushButton{background:#16A34A;color:#FFF;border:none;border-radius:4px;padding:6px 14px;font-size:12px;font-weight:600;}QPushButton:hover{background:#15803D;}")
        toolbar.addWidget(batch_btn)

        toolbar.addStretch()

        dlg_layout.addLayout(toolbar)

        # Scrollable list area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:1px solid #E0E0E0;border-radius:4px;background:#FFF;}")

        list_widget = QWidget()
        self._editor_list_layout = QVBoxLayout(list_widget)
        self._editor_list_layout.setSpacing(4)
        self._editor_list_layout.setContentsMargins(8, 8, 8, 8)
        self._editor_list_layout.addStretch()

        self._editor_entries = []  # (row_widget, edit, idx_label)

        def _renumber():
            for i, (_, _, lbl) in enumerate(self._editor_entries):
                lbl.setText(f"#{i+1}")

        def _add_row(path=""):
            row = QWidget()
            row.setMinimumHeight(38)
            rh = QHBoxLayout(row)
            rh.setContentsMargins(0, 2, 0, 2)
            rh.setSpacing(4)

            idx_lbl = QLabel(f"#{len(self._editor_entries)+1}")
            idx_lbl.setFixedWidth(30)
            idx_lbl.setStyleSheet("color:#AAA;font-size:11px;")
            rh.addWidget(idx_lbl)

            edit = QLineEdit(path)
            edit.setMinimumHeight(30)
            edit.setPlaceholderText("Path...")
            edit.setStyleSheet("QLineEdit{border:1px solid #D0D0D0;border-radius:4px;padding:5px 8px;font-size:12px;background:#FFF;}")
            rh.addWidget(edit, 1)

            browse_btn = QPushButton("...")
            browse_btn.setFixedWidth(32)
            browse_btn.setToolTip("Browse...")
            browse_btn.setStyleSheet("QPushButton{border:1px solid #C0C0C0;border-radius:3px;padding:3px 6px;background:#F8F8F8;}QPushButton:hover{background:#E0E0E0;}")
            browse_btn.clicked.connect(lambda *a, e=edit: self._browse_path(e))
            rh.addWidget(browse_btn)

            rm_btn = QPushButton("✕")
            rm_btn.setFixedSize(26, 26)
            rm_btn.setStyleSheet("QPushButton{background:transparent;color:#DC2626;border:none;font-size:14px;font-weight:bold;}QPushButton:hover{background:#FEE2E2;border-radius:13px;}")
            rm_btn.clicked.connect(lambda *a: _remove_row(row, edit))
            rh.addWidget(rm_btn)

            self._editor_list_layout.insertWidget(self._editor_list_layout.count() - 1, row)
            self._editor_entries.append((row, edit, idx_lbl))

        def _remove_row(row_widget, edit):
            self._editor_entries = [(r, e, l) for r, e, l in self._editor_entries if r is not row_widget]
            self._editor_list_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            _renumber()

        def _batch_add():
            """Step 1: pick root directory. Step 2: multi-select from tree."""
            from PySide6.QtWidgets import QTreeView, QFileSystemModel
            from PySide6.QtCore import QDir

            is_file_mode = self._mode != "dir"

            # Step 1: pick root directory
            root = QFileDialog.getExistingDirectory(dlg, "Select Root Directory", str(Path.cwd()))
            if not root:
                return

            # Step 2: show tree browser for multi-selection
            item_label = "Files" if is_file_mode else "Folders"
            bdlg = QDialog(dlg)
            bdlg.setWindowTitle(f"Select {item_label} — {root}")
            bdlg.resize(850, 550)
            bdlg.setMinimumWidth(650)

            bl = QVBoxLayout(bdlg)
            hint = QLabel(f"Root: {root}\nSelect {item_label.lower()}, then click OK.")
            hint.setStyleSheet("color:#666;font-size:12px;padding:4px 0;")
            bl.addWidget(hint)

            model = QFileSystemModel()
            model.setRootPath(root)
            if is_file_mode:
                model.setNameFilters(["*.txt", "*.csv", "*.json"])
                model.setNameFilterDisables(False)
                model.setFilter(QDir.Files | QDir.NoDotAndDotDot)
            else:
                model.setFilter(QDir.Dirs | QDir.NoDotAndDotDot)

            tree = QTreeView()
            tree.setModel(model)
            tree.setRootIndex(model.index(root))
            tree.setSelectionMode(QTreeView.ExtendedSelection)
            tree.setColumnHidden(1, True)
            tree.setColumnHidden(2, True)
            tree.setColumnHidden(3, True)
            tree.setHeaderHidden(True)
            tree.setRootIsDecorated(True)
            tree.setStyleSheet("QTreeView{border:1px solid #D0D0D0;border-radius:4px;font-size:12px;}")
            tree.expandAll()
            bl.addWidget(tree)

            btn_row = QHBoxLayout()
            btn_row.addWidget(QPushButton("Select All", clicked=lambda: tree.selectAll()))
            btn_row.addWidget(QPushButton("Deselect All", clicked=lambda: tree.clearSelection()))
            btn_row.addStretch()
            bl.addLayout(btn_row)

            bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            bb.accepted.connect(bdlg.accept)
            bb.rejected.connect(bdlg.reject)
            bl.addWidget(bb)

            if bdlg.exec() == QDialog.Accepted:
                for idx in tree.selectedIndexes():
                    if idx.column() == 0:
                        fpath = model.filePath(idx)
                        p = Path(fpath)
                        if is_file_mode and p.is_file():
                            _add_row(fpath)
                        elif not is_file_mode and p.is_dir():
                            _add_row(fpath)

        add_btn.clicked.connect(lambda *a: _add_row(""))
        batch_btn.clicked.connect(_batch_add)

        # Populate with existing paths (skip blanks)
        for p in self._paths:
            if p.strip():
                _add_row(p)

        scroll.setWidget(list_widget)
        dlg_layout.addWidget(scroll)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        dlg_layout.addWidget(btns)

        if dlg.exec() == QDialog.Accepted:
            self._paths = [e.text().strip() for _, e, _ in self._editor_entries if e.text().strip()]
            if not self._paths:
                self._paths = []
            self._update_summary()
            self.value_changed.emit(self._paths)

    def _browse_path(self, edit):
        from pathlib import Path
        cwd = str(Path.cwd())
        if self._mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select Directory", cwd)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select File", cwd,
                "Data Files (*.csv *.txt *.json);;All Files (*)"
            )
        if path:
            edit.setText(path)

    def set_paths(self, paths: list):
        self._paths = [p for p in (paths or []) if p.strip()] if paths else []
        self._update_summary()

    def get_paths(self) -> list:
        return self._paths


class FileSelector(QWidget):
    """Line edit + Browse button for file/directory selection."""

    path_changed = Signal(object)

    def __init__(self, mode="file", placeholder="", parent=None):
        super().__init__(parent)
        self._mode = mode
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText(placeholder)
        self._edit.textChanged.connect(lambda t: self.path_changed.emit(t))
        layout.addWidget(self._edit)

        self._btn = QPushButton("...")
        self._btn.setFixedWidth(30)
        self._btn.setToolTip("Browse...")
        self._btn.clicked.connect(self._browse)
        self._btn.setStyleSheet("""
            QPushButton { border: 1px solid #C0C0C0; border-radius: 3px; background: #F0F0F0; }
            QPushButton:hover { background: #E0E0E0; }
        """)
        layout.addWidget(self._btn)

    def _browse(self):
        cwd = str(Path.cwd())
        if self._mode == "dir":
            path = QFileDialog.getExistingDirectory(self, "Select Directory", cwd)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", cwd)
        if path:
            self._edit.setText(path)

    def text(self) -> str:
        return self._edit.text()

    def setText(self, text: str):
        self._edit.setText(text)

    def setPlaceholderText(self, text: str):
        self._edit.setPlaceholderText(text)


class ParameterGroup(QGroupBox):
    """Form group built from a schema. Embed in a QDialog."""

    param_changed = Signal(object, object)

    def __init__(self, title: str = "Parameters", schema: list = None, parent=None):
        super().__init__(title, parent)
        self._schema = schema or []
        self._widgets: dict = {}
        self._form = QFormLayout(self)
        self._form.setLabelAlignment(Qt.AlignRight)
        self._form.setSpacing(8)
        self.setStyleSheet("""
            QGroupBox {
                border: 1px solid #E0E0E0; border-radius: 6px;
                margin-top: 12px; padding-top: 16px;
                font-weight: 600; color: #333;
                background-color: #FFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 12px;
                padding: 0 6px; color: #0078D4;
            }
            QLineEdit {
                border: 1px solid #C0C0C0; border-radius: 4px;
                padding: 6px 8px; background: #FFF; color: #333;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 2px solid #0078D4; padding: 5px 7px;
            }
            QComboBox {
                border: 1px solid #C0C0C0; border-radius: 4px;
                padding: 6px 8px; background: #FFF; color: #333;
                font-size: 12px;
            }
            QComboBox:focus { border: 2px solid #0078D4; }
            QComboBox::drop-down { border: none; padding-right: 6px; }
            QCheckBox {
                spacing: 8px; color: #333; font-size: 12px;
            }
        """)
        self._build()

    def _build(self):
        for item in self._schema:
            key = item["key"]
            label = item.get("label", key)
            hint = item.get("hint", "")

            w = self._create_widget(item)
            if hint:
                w.setToolTip(hint)
            self._widgets[key] = w
            self._form.addRow(label + ":", w)

    def _create_widget(self, item: dict) -> QWidget:
        ptype = item.get("type", "text")
        default = item.get("default")
        options = item.get("options", [])

        if ptype == "file":
            w = FileSelector(mode="file")
            if default:
                w.setPlaceholderText(str(default))
                w.setText(str(default))
            return w

        elif ptype == "dir":
            w = FileSelector(mode="dir")
            if default:
                w.setPlaceholderText(str(default))
                w.setText(str(default))
            return w

        elif ptype == "path_list":
            w = PathListEditor(mode="dir")
            if default:
                if isinstance(default, list):
                    w.set_paths(default)
                elif isinstance(default, str) and default:
                    w.set_paths([default])
            return w

        elif ptype == "path_list_file":
            w = PathListEditor(mode="file")
            if default:
                if isinstance(default, list):
                    w.set_paths(default)
                elif isinstance(default, str) and default:
                    w.set_paths([default])
            return w

        elif ptype == "password":
            w = QLineEdit()
            w.setEchoMode(QLineEdit.EchoMode.Password)
            if default is not None:
                w.setText(str(default))
            return w

        elif ptype == "int":
            w = QLineEdit()
            lo = item.get("min", -999999)
            hi = item.get("max", 999999)
            w.setValidator(QIntValidator(lo, hi))
            if default is not None:
                w.setText(str(int(default)))
            w.setPlaceholderText(f"{lo} – {hi}")
            return w

        elif ptype == "float":
            w = QLineEdit()
            lo = float(item.get("min", 0.0))
            hi = float(item.get("max", 10.0))
            decimals = item.get("decimals", 4)
            validator = QDoubleValidator(lo, hi, decimals)
            validator.setNotation(QDoubleValidator.StandardNotation)
            w.setValidator(validator)
            if default is not None:
                w.setText(f"{float(default):.{decimals}f}".rstrip("0").rstrip("."))
            w.setPlaceholderText(f"{lo} – {hi}")
            return w

        elif ptype == "text":
            w = QLineEdit()
            if default is not None:
                w.setText(str(default))
            return w

        elif ptype == "combo":
            w = QComboBox()
            for opt in options:
                w.addItem(str(opt), opt)
            if default is not None:
                idx = w.findData(default)
                if idx >= 0:
                    w.setCurrentIndex(idx)
            return w

        elif ptype == "checkbox":
            w = QCheckBox()
            if default is not None:
                w.setChecked(bool(default))
            return w

        else:
            w = QLineEdit()
            return w

    def get_value(self, key: str):
        w = self._widgets.get(key)
        if w is None:
            return None

        if hasattr(w, '_file_selector'):
            w = w._file_selector

        if isinstance(w, PathListEditor):
            return w.get_paths()
        elif isinstance(w, FileSelector):
            return w.text()
        elif isinstance(w, QLineEdit):
            t = w.text()
            # Parse int/float if the validator is numeric
            if isinstance(w.validator(), QIntValidator):
                try:
                    return int(t)
                except ValueError:
                    return 0
            elif isinstance(w.validator(), QDoubleValidator):
                try:
                    return float(t)
                except ValueError:
                    return 0.0
            return t
        elif isinstance(w, QComboBox):
            return w.currentData()
        elif isinstance(w, QCheckBox):
            return w.isChecked()
        return None

    def get_values(self) -> dict:
        return {key: self.get_value(key) for key in self._widgets}

    def set_value(self, key: str, value):
        w = self._widgets.get(key)
        if w is None:
            return
        if isinstance(w, PathListEditor):
            w.set_paths(value if isinstance(value, list) else [str(value)])
        elif isinstance(w, QLineEdit):
            w.setText(str(value))
        elif isinstance(w, QComboBox):
            idx = w.findData(value)
            if idx >= 0:
                w.setCurrentIndex(idx)
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, FileSelector):
            w.setText(str(value))
