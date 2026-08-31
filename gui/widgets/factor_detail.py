"""FactorDetailDialog — Rich, structured factor detail view for non-technical users."""
import re
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QDialogButtonBox, QScrollArea, QWidget, QSizePolicy,
)
from PySide6.QtCore import Qt

# Global class-name registry: {str(id): "class_name"}
_CLASS_NAMES = {}
_LAZY_TRIED = False


def _load_label_map_from_config():
    """Try to find and load label_map from config files. Returns {str(id): name} or {}."""
    import json
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    for common_cfg in ["config/seq/1.yaml"]:
        cfg_path = project_root / common_cfg
        if not cfg_path.exists():
            continue
        try:
            import yaml
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            continue
        ds_file = cfg.get("dataset_config_file", "")
        if not ds_file:
            continue
        ds_path = Path(ds_file)
        if not ds_path.is_absolute():
            ds_path = cfg_path.parent / ds_path
        if not ds_path.exists():
            continue
        try:
            with open(ds_path, "r", encoding="utf-8") as f:
                ds_json = json.load(f)
        except Exception:
            continue
        raw_lm = ds_json.get("label_map", {})
        if isinstance(raw_lm, str):
            lm_path = Path(raw_lm)
            if not lm_path.is_absolute():
                lm_path = cfg_path.parent.parent / lm_path  # relative to project root
            if lm_path.exists():
                try:
                    with open(lm_path, "r", encoding="utf-8") as f:
                        raw_lm = json.load(f)
                except Exception:
                    continue
        if isinstance(raw_lm, dict) and raw_lm:
            return {str(v): k for k, v in raw_lm.items()}
    return {}


def set_class_names(mapping: dict):
    """Set the global class-id → class-name mapping. mapping: {str(id): name}"""
    _CLASS_NAMES.clear()
    _CLASS_NAMES.update({str(k): v for k, v in mapping.items()})


def _class_label(cid):
    """Return class name, lazily loading from config if needed."""
    global _LAZY_TRIED
    if not _CLASS_NAMES and not _LAZY_TRIED:
        _LAZY_TRIED = True
        loaded = _load_label_map_from_config()
        if loaded:
            _CLASS_NAMES.update(loaded)
    cid_str = str(cid)
    return _CLASS_NAMES.get(cid_str, str(cid))


def _lbl(text, color="#333", size=12, bold=False):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{color};font-size:{size}px;font-weight:{'bold' if bold else 'normal'};")
    lbl.setWordWrap(True)
    return lbl


def _metric_box(label, value, color, valid=True):
    """Small colored metric cell with fixed minimum size for alignment."""
    bg = "#F0FFF0" if valid else "#FFF5F5"
    border = "#86EFAC" if valid else "#F0A0A0"
    box = QFrame()
    box.setStyleSheet(f"QFrame{{background:{bg};border:1px solid {border};border-radius:4px;}}")
    box.setMinimumWidth(100)
    box.setMaximumWidth(160)
    bl = QVBoxLayout(box)
    bl.setContentsMargins(6, 4, 6, 4)
    bl.setSpacing(1)
    lbl = QLabel(label)
    lbl.setStyleSheet("color:#222;font-size:10px;font-weight:600;")
    lbl.setWordWrap(True)
    lbl.setAlignment(Qt.AlignCenter)
    bl.addWidget(lbl)
    val_lbl = QLabel(value)
    val_lbl.setStyleSheet(f"color:{color};font-size:12px;font-weight:bold;")
    val_lbl.setAlignment(Qt.AlignCenter)
    bl.addWidget(val_lbl)
    return box


def _extract_recipe(code):
    """Extract features, parameters, operators from factor code."""
    features = re.findall(r"idx\.get\('([^']+)'", code or "")
    features = list(dict.fromkeys(features))
    feat_str = ", ".join(features[:8])
    if len(features) > 8:
        feat_str += f" +{len(features) - 8} more"
    if not feat_str:
        feat_str = "(none)"

    params = re.findall(r"\bP\d+\b", code or "")
    params = sorted(set(params), key=lambda x: int(x[1:]))
    param_str = ", ".join(params) if params else "(none)"

    OP_MAP = {
        "np.mean": "mean", "np.std": "std", "np.var": "var",
        "np.max": "max", "np.min": "min", "np.median": "mad",
        "np.polyfit": "trend", "polyfit": "trend",
        "np.sqrt": "sqrt", "np.abs": "abs", "np.log1p": "log1p",
        "np.where": "where", "np.clip": "clip", "np.sum": "sum",
    }
    ops = set()
    for kw, op in OP_MAP.items():
        if kw in (code or ""):
            ops.add(op)
    op_str = ", ".join(sorted(ops)) if ops else "(none)"

    return feat_str, param_str, op_str


def show_factor_detail(factor: dict, parent=None):
    """Open a rich factor detail dialog."""
    dlg = QDialog(parent)
    name = factor.get("name", "Unknown Factor")
    dlg.setWindowTitle(f"Factor: {name}")
    dlg.setFixedWidth(800)
    dlg.setMinimumHeight(480)
    dlg.resize(800, 580)

    lo = QVBoxLayout(dlg)
    lo.setContentsMargins(0, 0, 0, 0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setStyleSheet("QScrollArea{border:none;background:#FFF;}")

    content = QWidget()
    content.setStyleSheet("background:#FFF;")
    cl = QVBoxLayout(content)
    cl.setContentsMargins(20, 16, 20, 16)
    cl.setSpacing(10)

    # ---- Header: name + AUC badge ----
    auc = factor.get("best_auc") or 0
    f1 = factor.get("best_f1") or 0
    is_good = auc >= 0.70

    hdr = QHBoxLayout()
    name_lbl = QLabel(f"📐 {name}")
    name_lbl.setStyleSheet("color:#111;font-size:16px;font-weight:bold;")
    name_lbl.setWordWrap(True)
    name_lbl.setMinimumWidth(300)
    hdr.addWidget(name_lbl, 1)
    auc_badge = QFrame()
    auc_badge.setStyleSheet(
        f"QFrame{{background:{'#16A34A' if is_good else '#888'};border:none;border-radius:8px;}}"
    )
    abl = QVBoxLayout(auc_badge)
    abl.setContentsMargins(16, 8, 16, 8)
    abl.addWidget(_lbl(f"AUC {auc:.4f}", "#FFF", 16, True))
    abl.addWidget(_lbl(f"F1 {f1:.4f}", "rgba(255,255,255,0.8)", 11))
    hdr.addWidget(auc_badge)
    cl.addLayout(hdr)

    # ---- Meta row ----
    target = factor.get("target", "?")
    seq_len = factor.get("seq_length", "?")
    rnd = factor.get("round", "?")
    meta = QHBoxLayout()
    meta.setSpacing(16)
    meta.addWidget(_lbl(f"Target: {target}", "#555", 12))
    meta.addWidget(_lbl(f"Seq Length: {seq_len}", "#555", 12))
    if rnd != "?":
        meta.addWidget(_lbl(f"Round: {rnd}", "#555", 12))
    meta.addStretch()
    cl.addLayout(meta)

    # ---- Separator ----
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("QFrame{border:none;border-top:1px solid #E0E0E0;}")
    cl.addWidget(sep)

    # ---- Description ----
    desc = factor.get("description", "")
    if desc:
        cl.addWidget(_lbl("Description", "#888", 10, True))
        cl.addWidget(_lbl(desc, "#333", 12))

    # ---- Performance: per-class AUC/F1 ----
    per_class = factor.get("per_class", {})
    valid_classes = factor.get("valid_classes", [])
    valid_ids = set()
    if isinstance(valid_classes, list):
        for vc in valid_classes:
            if isinstance(vc, dict):
                valid_ids.add(str(vc.get("class", "")))
            elif isinstance(vc, str):
                valid_ids.add(vc)

    if per_class:
        cl.addWidget(_lbl("Performance by Class", "#888", 10, True))
        grid = QHBoxLayout()
        grid.setSpacing(4)
        for cid in sorted(per_class.keys(), key=lambda x: str(x)):
            cls_data = per_class[cid]
            if isinstance(cls_data, dict):
                cls_auc = cls_data.get("auc", 0) or 0
                cls_f1 = cls_data.get("f1", 0) or 0
            else:
                cls_auc = float(cls_data) if cls_data else 0
                cls_f1 = None
            is_valid = str(cid) in valid_ids
            color = "#16A34A" if is_valid else "#C04040"
            val_str = f"AUC {cls_auc:.2f}"
            if cls_f1 is not None:
                val_str += f"\nF1 {cls_f1:.2f}"
            box = _metric_box(_class_label(cid), val_str, color, valid=is_valid)
            grid.addWidget(box, 0, Qt.AlignTop)
        cl.addLayout(grid)

    # ---- Recipe: features, params, operators ----
    code = factor.get("code", "")
    if code:
        feat_str, param_str, op_str = _extract_recipe(code)
        cl.addWidget(_lbl("Formula Recipe", "#888", 10, True))
        recipe_widget = QFrame()
        recipe_widget.setStyleSheet(
            "QFrame{background:#FFFEF5;border:1px solid #E8E0C0;border-radius:6px;}"
        )
        rl = QVBoxLayout(recipe_widget)
        rl.setContentsMargins(12, 8, 12, 8)
        rl.setSpacing(4)
        rl.addWidget(_lbl(f"Features:  {feat_str}", "#333", 11))
        rl.addWidget(_lbl(f"Parameters:  {param_str}", "#333", 11))
        rl.addWidget(_lbl(f"Operators:  {op_str}", "#333", 11))
        cl.addWidget(recipe_widget)

    # ---- Raw code (collapsible in spirit: always shown but clearly labeled) ----
    if code:
        cl.addWidget(_lbl("Source Code", "#AAA", 9, True))
        code_lbl = QLabel(f"<pre style='color:#555;font-size:10px;'>{code}</pre>")
        code_lbl.setWordWrap(True)
        code_lbl.setTextFormat(Qt.RichText)
        code_lbl.setStyleSheet(
            "background:#F8F8F8;border:1px solid #E5E5E5;border-radius:4px;padding:8px;"
        )
        cl.addWidget(code_lbl)

    cl.addStretch()
    scroll.setWidget(content)
    lo.addWidget(scroll)

    btns = QDialogButtonBox(QDialogButtonBox.Ok)
    btns.accepted.connect(dlg.accept)
    lo.addWidget(btns)

    dlg.exec()
