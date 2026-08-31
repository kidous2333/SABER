"""DiscoveryTab — Factor discovery with real-time visualization."""
import logging
from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QScrollArea,
    QSizePolicy, QListWidget, QListWidgetItem,
)
from PySide6.QtCore import Qt, Slot, QEvent

from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.workers.discovery_worker import DiscoveryWorker

logger = logging.getLogger("gui.discovery")


class FactorListWidget(QListWidget):
    """Compact factor list with double-click for details."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._factors = []
        self.setMaximumHeight(160)
        self.setStyleSheet("""
            QListWidget{background:#FAFAFA;border:1px solid #E5E5E5;border-radius:4px;
            font-size:10px;color:#333;}
            QListWidget::item{padding:2px 4px;}
            QListWidget::item:selected{background:#0078D4;color:#FFF;}
        """)
        self.itemDoubleClicked.connect(self._show_detail)
        self._detail_parent = parent

    def set_factors(self, factors):
        self._factors = factors
        self.clear()
        for f in sorted(factors, key=lambda f: f.get("best_auc") or 0, reverse=True):
            auc = f.get("best_auc") or 0
            name = f.get("name", "?")
            star = "⭐" if auc >= 0.80 else ("★" if auc >= 0.70 else "·")
            item = QListWidgetItem(f"{star} {name[:40]}  AUC {auc:.4f}")
            item.setData(1, f)  # store full factor dict
            self.addItem(item)

    def _show_detail(self, item):
        f = item.data(1)
        if not f:
            return
        from gui.widgets.factor_detail import show_factor_detail
        show_factor_detail(f, parent=self)


class DiscoveryTab(BaseTab):
    def __init__(self, parent=None):
        self._saved_factors = []
        self._round_count = 0
        self._valid_count = 0
        self._invalid_count = 0
        self._class_map = {}  # class_id (str) → class_name, built from Labels config
        super().__init__(title="Factor Discovery", tab_key="discovery", parent=parent)

    def _build_class_map(self):
        """Build {class_id_str: class_name} from the label map editor."""
        pairs = self._label_map_editor.get_pairs() if hasattr(self, '_label_map_editor') else {}
        self._class_map = {v.strip(): k for k, v in pairs.items() if k.strip() and v.strip()}

    def setup_params(self):
        self.add_config_group(ParameterGroup("Sequence", [
            {"key": "seq_length", "label": "Seq Length", "type": "int", "default": 1, "min": 1, "max": 60},
            {"key": "stride", "label": "Stride", "type": "int", "default": 1, "min": 1, "max": 30},
            {"key": "purity_threshold", "label": "Purity", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
        ]))
        self.add_config_group(ParameterGroup("Validation", [
            {"key": "min_auc", "label": "Min AUC", "type": "float", "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01},
            {"key": "min_f1", "label": "Min F1", "type": "float", "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01},
            {"key": "use_gpu", "label": "Use GPU", "type": "checkbox", "default": False},
            {"key": "lgbm_estimators", "label": "LGBM Trees", "type": "int", "default": 200, "min": 10, "max": 1000},
            {"key": "lgbm_depth", "label": "LGBM Depth", "type": "int", "default": 4, "min": 2, "max": 12},
        ]))
        self.add_config_group(ParameterGroup("Loop", [
            {"key": "max_rounds", "label": "Max Rounds", "type": "int", "default": 200, "min": 1, "max": 9999},
            {"key": "early_stop_rounds", "label": "Early Stop", "type": "int", "default": 5, "min": 1, "max": 100},
            {"key": "max_valid_factors", "label": "Max Factors", "type": "int", "default": 5000, "min": 1, "max": 99999},
            {"key": "hypotheses_per_round", "label": "Hypotheses/Round", "type": "int", "default": 8, "min": 1, "max": 50},
        ]))
        self.add_config_group(ParameterGroup("Dataset", [
            {"key": "train_mouse_dirs", "label": "Train Mouse KP Dirs", "type": "path_list", "default": [""]},
            {"key": "train_tail_dirs", "label": "Train Tail KP Dirs", "type": "path_list", "default": [""]},
            {"key": "train_behavior_m1_files", "label": "Train Behavior M1 Files", "type": "path_list_file", "default": [""]},
            {"key": "train_behavior_m2_files", "label": "Train Behavior M2 Files", "type": "path_list_file", "default": [""]},
            {"key": "val_mouse_dirs", "label": "Val Mouse KP Dirs", "type": "path_list", "default": [""]},
            {"key": "val_tail_dirs", "label": "Val Tail KP Dirs", "type": "path_list", "default": [""]},
            {"key": "val_behavior_m1_files", "label": "Val Behavior M1 Files", "type": "path_list_file", "default": [""]},
            {"key": "val_behavior_m2_files", "label": "Val Behavior M2 Files", "type": "path_list_file", "default": [""]},
            {"key": "max_instances", "label": "Max Instances", "type": "int", "default": 2, "min": 1, "max": 10},
        ]))
        self.add_config_group(ParameterGroup("Labels", []))  # placeholder, label map uses editor below
        from gui.widgets.key_value_editor import KeyValueEditor
        self._label_map_editor = KeyValueEditor(title="Label Map")
        self._label_map_editor.set_pairs({
            "explore_object": "0", "climb": "1", "self_grooming": "2",
            "stand": "3", "blank": "4", "positive_sniffs": "5", "approach": "6",
        })
        self._label_map_editor.setMaximumHeight(200)
        self._param_layout.addWidget(self._label_map_editor)
        self.add_config_group(ParameterGroup("LLM", [
            {"key": "llm_provider", "label": "Provider", "type": "combo", "default": "anthropic", "options": ["anthropic", "openai", "glm"]},
            {"key": "llm_model", "label": "Model", "type": "text", "default": "claude-sonnet-4-20250514"},
            {"key": "llm_base_url", "label": "Base URL", "type": "text", "default": "https://www.micuapi.ai"},
            {"key": "llm_api_key", "label": "API Key", "type": "password", "default": ""},
            {"key": "llm_temperature", "label": "Temperature", "type": "float", "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1},
            {"key": "llm_user_agent", "label": "User-Agent", "type": "text",
             "default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:149.0) Gecko/20100101 Firefox/149.0"},
            {"key": "llm_custom_headers", "label": "Custom Headers", "type": "text", "default": ""},
            {"key": "llm_timeout", "label": "Timeout (s)", "type": "int", "default": 600, "min": 60, "max": 3600},
        ]))
        self.add_config_group(ParameterGroup("Output", [
            {"key": "factors_output_path", "label": "Save Factors To", "type": "file",
             "default": "memory/gui_discovered_factors.json"},
        ]))

        # Separator with label
        sep = QLabel("Discovered Factors")
        sep.setAlignment(Qt.AlignCenter)
        sep.setFixedHeight(50)
        sep.setStyleSheet("color:#888;font-size:13px;font-weight:700;background:#FAFAFA;padding:12px;"
                          "border-top:1px solid #E0E0E0;border-bottom:1px solid #E0E0E0;")
        self.add_widget_to_config_panel(sep)
        self._config_factor_list = FactorListWidget(self)
        self._config_factor_list.setMaximumHeight(200)
        self.add_widget_to_config_panel(self._config_factor_list)

    def setup_results(self):
        self._stats_header = QLabel("Ready")
        self._stats_header.setStyleSheet(
            "color:#333;font-size:14px;font-weight:700;padding:8px 12px;"
            "background:#F8F8F8;border:1px solid #E0E0E0;border-radius:8px;"
        )
        self._stats_header.setAlignment(Qt.AlignCenter)
        self._results_layout_main.addWidget(self._stats_header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self._card_container = QWidget()
        self._card_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setSpacing(6)
        self._card_layout.setContentsMargins(4, 4, 4, 4)
        self._scroll.setWidget(self._card_container)
        self._results_layout_main.addWidget(self._scroll)

        # Install event filter on viewport to catch resize events
        self._scroll.viewport().installEventFilter(self)

    def _update_header(self):
        self._stats_header.setText(
            f"📊 Round {self._round_count}  |  ✅ {self._valid_count} valid  |  ❌ {self._invalid_count} invalid"
        )

    def _add_factor_card(self, data):
        name = data.get("name", "unknown")
        code = data.get("code", "")
        desc = data.get("description", "")
        target = data.get("target", "")
        valid = data.get("valid", False)
        best_auc = data.get("best_auc") or 0
        best_f1 = data.get("best_f1") or 0
        best_class = data.get("best_class", "")
        reason = data.get("reason", "")
        rnd = data.get("round", 0)
        seq_len = data.get("seq_length", 5)
        valid_classes = data.get("valid_classes", [])
        per_class = data.get("per_class", {})

        if valid:
            self._add_valid_card(name, code, desc, target, best_auc, best_f1,
                                best_class, seq_len, valid_classes, per_class, rnd, data)
        else:
            self._add_invalid_card(name, code, target, best_auc, reason, rnd, seq_len, data)

    def _add_valid_card(self, name, code, desc, target, auc, f1, best_class,
                        seq_len, valid_classes, per_class, rnd, full_data=None):
        card = QFrame()
        card.setStyleSheet("QFrame{background:#F6FFF6;border:2px solid #16A34A;border-radius:10px;}")
        # Store full factor data for double-click detail
        if full_data:
            card.setProperty("factor_data", full_data)
            card.mouseDoubleClickEvent = lambda e, d=full_data: self._open_card_detail(d)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 10, 16, 8)
        cl.setSpacing(4)

        # Line 1: Factor name + meta info inline | AUC (single row, no height increase)
        name_row = QHBoxLayout()
        name_row.addWidget(self._lbl(f"📐 {name}", "#111", 16, True))
        info = f"  ·  Round {rnd}  ·  Pred: {best_class or target}  ·  {seq_len}fr  ·  Target: {target or '-'}"
        name_row.addWidget(self._lbl(info, "#555", 12))
        name_row.addStretch()
        name_row.addWidget(self._lbl(f"AUC {auc:.4f}", "#16A34A", 20, True))
        cl.addLayout(name_row)

        # Line 2: Description (prominent, wrapping)
        readable = self._make_readable(name, code, desc, target)
        cl.addWidget(self._lbl(f"💡 {readable}", "#333", 13, wrap=True))

        # Formula area — feature / parameter / operator summary
        if code:
            recipe = self._format_recipe(code)
            if recipe:
                recipe_text = (
                    f"Feats ({recipe['n_features']}): {recipe['features']}\n"
                    f"Params ({recipe['n_params']}): {recipe['params']}\n"
                    f"Ops: {recipe['operators']}"
                )
                f_lbl = QLabel(recipe_text)
                f_lbl.setWordWrap(True)
                f_lbl.setMinimumHeight(50)
                f_lbl.setStyleSheet(
                    "color:#333;font-size:11px;font-family:Consolas,monospace;"
                    "background:#FFFEF5;border:1px solid #E8E0C0;border-radius:6px;"
                    "padding:4px 8px;"
                )
                cl.addWidget(f_lbl)

        # Metrics row — per-class AUC + F1 for all classes, fill card width
        if not self._class_map:
            self._build_class_map()
        m = QHBoxLayout()
        m.setSpacing(3)
        if self._class_map:
            for cid in sorted(self._class_map.keys(), key=lambda x: int(x) if x.isdigit() else x):
                cls_name = self._class_map[cid]
                cls_data = (per_class or {}).get(cid) or (per_class or {}).get(str(cid))
                if isinstance(cls_data, dict):
                    cls_auc = cls_data.get("auc", 0) or 0
                    cls_f1 = cls_data.get("f1", 0) or 0
                elif isinstance(cls_data, (int, float)):
                    cls_auc = float(cls_data)
                    cls_f1 = None
                else:
                    cls_auc = 0
                    cls_f1 = None
                # valid_classes is list of dicts: [{"class": "0", "auc":..., "f1":...}, ...]
                valid_class_ids = {v["class"] for v in (valid_classes or []) if isinstance(v, dict)}
                is_valid = cid in valid_class_ids or str(cid) in valid_class_ids
                m.addWidget(self._metric_cell(cls_name, f"{cls_auc:.2f}",
                    f"{cls_f1:.2f}" if cls_f1 is not None else None,
                    valid=is_valid), 1)
        cl.addLayout(m)

        self._add_card_to_layout(card)

    def _add_card_to_layout(self, card):
        """Insert a card at the top of the results layout — newest results always visible."""
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vw = self._get_card_target_width()
        if vw > 100:
            card.setFixedWidth(vw)
        # Insert at position 0 so the latest card is always at the top
        self._card_layout.insertWidget(0, card)
        # Ensure a stretch at the bottom keeps cards anchored to the top
        if self._card_layout.count() == 1:
            self._card_layout.addStretch()
        # Scroll to top so the newest card is visible
        if hasattr(self, '_scroll') and self._scroll is not None:
            self._scroll.verticalScrollBar().setValue(0)

    def _get_card_target_width(self):
        """Return the target width for cards based on the scroll viewport."""
        if hasattr(self, '_scroll') and self._scroll is not None:
            vp = self._scroll.viewport()
            if vp is not None:
                return vp.width() - 16  # scrollbar margin + padding
        return 0

    def _update_all_card_widths(self):
        """Update all card widths to match the current viewport width."""
        vw = self._get_card_target_width()
        if vw < 100:
            return
        for i in range(self._card_layout.count()):
            item = self._card_layout.itemAt(i)
            if item and item.widget():
                item.widget().setFixedWidth(vw)

    def eventFilter(self, obj, event):
        """Handle resize events on the scroll viewport to update card widths."""
        if event.type() == QEvent.Resize and hasattr(self, '_scroll') and obj is self._scroll.viewport():
            self._update_all_card_widths()
        return super().eventFilter(obj, event)

    def _make_readable(self, name, code, desc, target):
        """Create a human-readable hypothesis description."""
        if desc:
            return desc
        import re
        features = re.findall(r"idx\.get\('([^']+)'", code or "")
        if features and target:
            feat_str = ", ".join(features[:4])
            if len(features) > 4:
                feat_str += f" +{len(features)-4} more"
            return f"Uses [{feat_str}] to identify [{target}] behavior"
        if features:
            return f"Based on [{', '.join(features[:4])}] features for prediction"
        return f"Identifies: {target or 'unknown'}"

    def _format_recipe(self, code):
        """Extract feature / parameter / operator summary from factor code."""
        import re
        # Features
        features = re.findall(r"idx\.get\('([^']+)'", code or "")
        features = list(dict.fromkeys(features))  # dedup preserving order
        n_features = len(features)
        feat_str = ", ".join(features[:6])
        if n_features > 6:
            feat_str += f" +{n_features - 6}"
        if not feat_str:
            feat_str = "(none)"

        # Parameters P0, P1, P2, ...
        params = re.findall(r"\bP\d+\b", code or "")
        params = sorted(set(params), key=lambda x: int(x[1:]))
        n_params = len(params)
        param_str = ", ".join(params) if params else "(none)"

        # Operators / aggregators
        OP_MAP = {
            "np.mean": "mean", "np.std": "std", "np.var": "var",
            "np.max": "max", "np.min": "min", "np.median": "mad",
            "np.polyfit": "trend", "polyfit": "trend",
            "np.sqrt": "sqrt", "np.abs": "abs", "np.log1p": "log1p",
            "np.where": "where", "np.clip": "clip", "np.sum": "sum",
            "np.full": "fill",
        }
        ops = set()
        for kw, op in OP_MAP.items():
            if kw in (code or ""):
                ops.add(op)
        op_str = ", ".join(sorted(ops)) if ops else "(none)"

        return {
            "features": feat_str,
            "n_features": n_features,
            "params": param_str,
            "n_params": n_params,
            "operators": op_str,
        }

    def _format_formula_lines(self, code):
        """Extract formula lines — returns list of strings."""
        if not code:
            return []
        import re
        features = re.findall(r"idx\.get\('([^']+)'", code)
        lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
        result = []
        for line in lines:
            if 'idx.get' in line or line.startswith('if ') or 'return np.nan' in line:
                continue
            for i, feat in enumerate(features):
                short = feat.replace('other_','o.').replace('self_','s.')
                short = short.replace('body_','b.').replace('orientation','o.')
                short = short.replace('compactness','cp').replace('acceleration','ac')
                line = line.replace(f"'{feat}'", short)
            line = line.replace('np.mean(', 'avg(').replace('np.std(', 'std(')
            line = line.replace('np.max(', 'max(').replace('np.min(', 'min(')
            line = line.replace('np.sqrt(', '√(').replace('np.where(', 'where(')
            line = line.replace('np.clip(', 'clip(').replace('np.polyfit(', 'fit(')
            line = line.replace('windows[:, :, ', '').replace('float(', '')
            line = line.replace('return ', '→ ').replace('_P', 'P')
            line = line.replace('_result', 'out').replace('np.nan', '∅')
            line = line.replace('np.', '')
            if line.strip() and not line.startswith('#'):
                result.append(line.strip())
        return result[:8]

    def _format_formula(self, code):
        """Extract formula as clean text from factor code."""
        if not code:
            return ""
        import re
        features = re.findall(r"idx\.get\('([^']+)'", code)
        lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
        result = []
        for line in lines:
            if 'idx.get' in line or line.startswith('if ') or 'return np.nan' in line:
                continue
            line = line.replace('np.mean(', 'mean(').replace('np.std(', 'std(')
            line = line.replace('np.max(', 'max(').replace('np.min(', 'min(')
            line = line.replace('np.abs(', '|').replace('np.sqrt(', '√(')
            line = line.replace('np.where(', 'where(').replace('np.clip(', 'clip(')
            line = line.replace('np.polyfit(', 'polyfit(').replace('np.full(', 'fill(')
            for i, feat in enumerate(features):
                short = feat.replace('other_','o.').replace('self_','s.')
                short = short.replace('body_','b.').replace('orientation','orient')
                short = short.replace('compactness','comp').replace('acceleration','accel')
                line = line.replace(f"'{feat}'", short)
            line = line.replace('windows[:, :, ', '').replace('float(', '')
            line = line.replace('return ', '→ ').replace('_P', 'P')
            line = line.replace('_result', 'res').replace('np.nan', 'NaN')
            line = line.replace('np.', '')
            if line.strip() and not line.startswith('#'):
                result.append(line.strip())
        return '\n'.join(result[:6])

    def _extract_math(self, code):
        """Extract a readable mathematical expression from factor code."""
        if not code:
            return ""
        import re
        # Map idx.get('X', -1) → X
        idx_map = {}
        for i, m in enumerate(re.findall(r"idx\.get\('([^']+)'", code)):
            idx_map[f"i_{i}"] = m

        # Clean up code: remove idx lookups, simplify
        lines = code.strip().split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Skip idx.get lines and if checks
            if "idx.get" in line or line.startswith("if ") or "return np.nan" in line:
                continue
            # Replace variable names with feature names
            for var, feat in idx_map.items():
                line = line.replace(var, f"「{feat}」")
            # Simplify np.* calls
            line = line.replace("np.mean", "mean").replace("np.std", "std")
            line = line.replace("np.max", "max").replace("np.min", "min")
            line = line.replace("np.abs", "abs").replace("np.sqrt", "sqrt")
            line = line.replace("np.log1p", "ln(1+").replace("np.clip", "clip")
            line = line.replace("np.sum", "sum").replace("np.median", "median")
            line = line.replace("float(", "").replace(")", "")
            if "return" in line:
                line = line.replace("return ", "→ ")
            clean_lines.append(line)

        if not clean_lines:
            return ""

        expr = " · ".join(clean_lines)
        return f"Formula: {expr}"

    def _add_invalid_card(self, name, code, target, auc, reason, rnd, seq_len, full_data=None):
        card = QFrame()
        card.setStyleSheet("QFrame{background:#FFF;border:1px solid #E5C5C5;border-radius:8px;}")
        if full_data:
            card.setProperty("factor_data", full_data)
            card.mouseDoubleClickEvent = lambda e, d=full_data: self._open_card_detail(d)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 5, 12, 4)
        cl.setSpacing(2)

        hdr = QHBoxLayout()
        hdr.addWidget(self._lbl(f"Round {rnd}  ❌ {name}", "#333", 13, True))
        hdr.addStretch()
        if auc:
            hdr.addWidget(self._lbl(f"AUC {auc:.4f}", "#DC2626", 14, True))
        hdr.addWidget(self._lbl(f"Reason: {reason}" if reason else "", "#999", 11))
        cl.addLayout(hdr)

        # Brief readable explanation
        import re
        features = re.findall(r"idx\.get\('([^']+)'", code or "")
        if features:
            cl.addWidget(self._lbl(f"Feats: {', '.join(features[:5])}", "#888", 11))

        self._add_card_to_layout(card)

    @Slot(object)
    def on_partial_result(self, data):
        if not isinstance(data, dict):
            return
        typ = data.get("type", "")
        if typ == "round_start":
            self._round_count = data.get("round", self._round_count)
            self._update_header()
        elif typ == "eval_result":
            if data.get("valid"):
                self._valid_count += 1
                self._saved_factors.append(data)
                self._config_factor_list.set_factors(self._saved_factors)
            else:
                self._invalid_count += 1
            self._update_header()
            self._add_factor_card(data)

    def on_result(self, result):
        if not result or not isinstance(result, dict):
            return
        factors = result.get("factors", [])
        if factors:
            self._add_completion_card(factors)
        self._stats_header.setText(f"✅ Complete! {len(factors)} factors → {result.get('output_path','')}")

    def _add_completion_card(self, factors):
        card = QFrame()
        card.setStyleSheet("QFrame{background:#F0F7FF;border:2px solid #0078D4;border-radius:10px;}")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(3)
        cl.addWidget(self._lbl(f"🎉 Discovery Complete — {len(factors)} valid factors", "#0078D4", 16, True))
        for i, f in enumerate(sorted(factors, key=lambda x: x.get("best_auc") or 0, reverse=True)[:20]):
            auc = f.get("best_auc") or 0
            name = f.get("name", "?")
            target = f.get("target", "")
            star = "⭐" if auc >= 0.80 else ("★" if auc >= 0.70 else "·")
            r = QHBoxLayout()
            r.addWidget(self._lbl(f"#{i+1}", "#999", 12))
            r.addWidget(self._lbl(name, "#333", 13, True))
            r.addStretch()
            r.addWidget(self._lbl(f"({target})" if target else "", "#888", 11))
            r.addWidget(self._lbl(f"{star} {auc:.4f}", "#16A34A" if auc>=0.7 else "#888", 12))
            cl.addLayout(r)
        if len(factors) > 20:
            cl.addWidget(self._lbl(f"... and {len(factors)-20} more", "#999", 11))
        self._add_card_to_layout(card)

    def gather_params(self) -> dict:
        g = super().gather_params()
        if hasattr(self, '_label_map_editor'):
            g["label_map_pairs"] = self._label_map_editor.get_pairs()
        return g

    def _load_settings(self):
        from gui.utils.gui_settings import get_tab_settings
        saved = get_tab_settings(self._tab_key)
        if not saved:
            return
        for group in self._config_groups:
            for item in group._schema:
                key = item["key"]
                if key in saved:
                    group.set_value(key, saved[key])
        if hasattr(self, '_label_map_editor') and "label_map_pairs" in saved:
            self._label_map_editor.set_pairs(saved["label_map_pairs"])

    def on_start(self):
        self._saved_factors = []
        self._round_count = self._valid_count = self._invalid_count = 0
        self._update_header()

        while self._card_layout.count() > 0:
            item = self._card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._config_factor_list.set_factors([])
        self.run_worker(DiscoveryWorker, params=self.gather_params())

    def _open_card_detail(self, data):
        from gui.widgets.factor_detail import show_factor_detail
        show_factor_detail(data, parent=self)

    def _lbl(self, text, color="#333", size=12, bold=False, wrap=False):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{color};font-size:{size}px;font-weight:{'bold' if bold else 'normal'};")
        if wrap:
            lbl.setWordWrap(True)
        return lbl

    def _math_block(self, text):
        w = QLabel(text)
        w.setWordWrap(True)
        w.setStyleSheet(
            "color:#222;font-size:12px;"
            "background:#FFFEF5;border:1px solid #E8E0C0;border-radius:6px;"
            "padding:8px 10px;"
        )
        return w

    def _build_test_card(self):
        """Test card for design iteration — shown on every startup."""
        card = QFrame()
        card.setStyleSheet("QFrame{background:#F6FFF6;border:2px solid #16A34A;border-radius:10px;}")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 8, 12, 6)
        cl.setSpacing(3)

        # Name + meta inline | AUC (single row)
        nr = QHBoxLayout()
        nr.addWidget(self._lbl("📐 factor_speed_climb_ratio_v3", "#111", 16, True))
        nr.addWidget(self._lbl("  ·  Round 3  ·  Pred: climb  ·  5fr  ·  Target: climb", "#555", 12))
        nr.addStretch()
        nr.addWidget(self._lbl("AUC 0.8124", "#16A34A", 20, True))
        cl.addLayout(nr)

        # Hypothesis (wrapping, compact)
        hyp = QLabel("💡 Uses [speed] and [dist_to_other] temporal changes, building a ratio factor via mean and std to identify [climb] behavior. When climbing, speed drops and distance std rises, giving high factor values.")
        hyp.setWordWrap(True)
        hyp.setStyleSheet("color:#333;font-size:13px;padding:2px 0;")
        cl.addWidget(hyp)

        # Formula area — feature / parameter / operator summary
        recipe_text = (
            "Feats (2): speed, dist_to_other\n"
            "Params (4): P0, P1, P2, P3\n"
            "Ops: clip, mean, std, where"
        )
        f_lbl = QLabel(recipe_text)
        f_lbl.setWordWrap(True)
        f_lbl.setMinimumHeight(50)
        f_lbl.setStyleSheet(
            "color:#333;font-size:11px;font-family:Consolas,monospace;"
            "background:#FFFEF5;border:1px solid #E8E0C0;border-radius:6px;"
            "padding:4px 8px;"
        )
        cl.addWidget(f_lbl)

        # Metrics — per-class AUC + F1 for all 7 classes
        mr = QHBoxLayout(); mr.setSpacing(3)
        test_classes = [
            ("explore_object", "0.65", "0.48", True),
            ("climb", "0.81", "0.62", True),
            ("self_grooming", "0.58", "0.35", False),
            ("stand", "0.72", "0.55", True),
            ("blank", "0.45", "0.28", False),
            ("positive_sniffs", "0.70", "0.52", True),
            ("approach", "0.51", "0.31", False),
        ]
        for cls_name, auc_val, f1_val, is_valid in test_classes:
            mr.addWidget(self._metric_cell(cls_name, auc_val, f1_val, valid=is_valid), 1)
        cl.addLayout(mr)

        return card

    def _code_block(self, text):
        w = QLabel(text)
        w.setWordWrap(True)
        w.setStyleSheet(
            "color:#333;font-size:10px;font-family:Consolas;"
            "background:#FFF;border:1px solid #E5E5E5;border-radius:6px;"
            "padding:8px 10px;"
        )
        w.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return w

    def _metric_cell(self, label, auc_val, f1_val=None, valid=True):
        """Per-class metric cell: class name on top, AUC + F1 below.

        valid=True   → green bg
        valid=False  → red bg
        """
        if valid:
            bg = "#F0FFF0"
            border = "#86EFAC"
            lbl_color = "#16A34A"
            val_color = "#16A34A"
        else:
            bg = "#FFF5F5"
            border = "#F0A0A0"
            lbl_color = "#C04040"
            val_color = "#C04040"

        box = QFrame()
        box.setMinimumWidth(90)
        box.setStyleSheet(
            f"QFrame{{background:{bg};border:1px solid {border};border-radius:5px;}}"
        )
        bl = QVBoxLayout(box)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(1)
        bl.addWidget(self._lbl(label, lbl_color, 11, True), 0, Qt.AlignCenter)
        bl.addWidget(self._lbl(f"AUC {auc_val}", val_color, 11, True), 0, Qt.AlignCenter)
        if f1_val is not None:
            bl.addWidget(self._lbl(f"F1 {f1_val}", val_color, 11, True), 0, Qt.AlignCenter)
        return box
