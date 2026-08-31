"""AnalysisTab — Factor-level analysis with rich card-based visualization."""
import json
import logging
from pathlib import Path
from collections import Counter

from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QScrollArea,
    QSizePolicy, QPushButton, QDialog, QDialogButtonBox,
)
from PySide6.QtCore import Qt, Slot

from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.workers.analysis_worker import AnalysisWorker

logger = logging.getLogger("gui.analysis")


class AnalysisTab(BaseTab):
    def __init__(self, parent=None):
        self._factors_data = []
        self._meta = None
        super().__init__(title="Factor Analysis", tab_key="analysis", parent=parent)

    # ---- Config ----
    def setup_params(self):
        self.add_config_group(ParameterGroup("Data", [
            {"key": "factors_path", "label": "Factors File", "type": "file",
             "default": "memory/evolved_factors.json"},
        ]))
        self.add_config_group(ParameterGroup("Label Merge", [
            {"key": "label_merge_enabled", "label": "Enable Merge", "type": "checkbox",
             "default": True,
             "hint": "Merge source behavior classes into target (e.g. climbsocial→stand)"},
            {"key": "label_merge_config", "label": "Merge Rules", "type": "text",
             "default": "climbsocial:stand",
             "hint": "Comma-separated pairs: src1:tgt1, src2:tgt2"},
        ]))
        self.add_config_group(ParameterGroup("Display", [
            {"key": "top_n_features", "label": "Top Features", "type": "int",
             "default": 40, "min": 5, "max": 200},
            {"key": "top_k_per_class", "label": "Top Factors/Class", "type": "int",
             "default": 10, "min": 1, "max": 50},
        ]))

    # ---- Results area ----
    def setup_results(self):
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        self._card_container = QWidget()
        self._card_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setSpacing(8)
        self._card_layout.setContentsMargins(4, 4, 4, 4)
        self._card_layout.addStretch()
        self._scroll.setWidget(self._card_container)
        self._results_layout_main.addWidget(self._scroll)

    def _clear_cards(self):
        """Remove all cards from results area."""
        while self._card_layout.count() > 0:
            item = self._card_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_card(self, card):
        """Insert a card before the bottom stretch."""
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vw = self._get_viewport_width()
        if vw > 100:
            card.setFixedWidth(vw)
        # Insert before the stretch (last item)
        idx = max(0, self._card_layout.count() - 1)
        self._card_layout.insertWidget(idx, card)

    def _get_viewport_width(self):
        if hasattr(self, '_scroll') and self._scroll is not None:
            vp = self._scroll.viewport()
            if vp is not None:
                return vp.width() - 16
        return 0

    # ---- Helpers ----
    def _lbl(self, text, color="#333", size=12, bold=False, wrap=False):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color:{color};font-size:{size}px;"
            f"font-weight:{'bold' if bold else 'normal'};"
        )
        if wrap:
            lbl.setWordWrap(True)
        return lbl

    def _stat_card(self, label, value, color="#0078D4"):
        """Summary stat card: big value + small label."""
        box = QFrame()
        box.setStyleSheet(
            f"QFrame{{background:#FFF;border:1px solid #E0E0E0;border-radius:8px;}}"
        )
        bl = QVBoxLayout(box)
        bl.setContentsMargins(14, 10, 14, 10)
        bl.setSpacing(2)
        bl.addWidget(self._lbl(str(value), color, 22, True), 0, Qt.AlignCenter)
        bl.addWidget(self._lbl(label, "#888", 10), 0, Qt.AlignCenter)
        return box

    def _factor_mini_card(self, name, auc, target="", factor_data=None):
        """Mini factor card for per-behavior lists. Clickable to show detail."""
        card = QFrame()
        star = "⭐" if auc >= 0.80 else ("★" if auc >= 0.70 else "·")
        is_good = auc >= 0.70
        card.setStyleSheet(
            f"QFrame{{background:{'#F6FFF6' if is_good else '#FFF'};"
            f"border:1px solid {'#86EFAC' if is_good else '#E5E5E5'};border-radius:6px;}}"
            f"QFrame:hover{{border-color:#0078D4;}}"
        )
        card.setCursor(Qt.PointingHandCursor)
        if factor_data:
            card.mousePressEvent = lambda ev, fd=factor_data: self._show_factor_detail(fd)

        cl = QVBoxLayout(card)
        cl.setContentsMargins(10, 6, 10, 6)
        cl.setSpacing(1)

        nr = QHBoxLayout()
        nr.addWidget(self._lbl(name[:35], "#111", 11, True))
        nr.addStretch()
        nr.addWidget(self._lbl(f"{star} {auc:.4f}",
            "#16A34A" if is_good else "#888", 12, True))
        cl.addLayout(nr)

        if target:
            cl.addWidget(self._lbl(f"target: {target}", "#999", 9))
        return card

    def _behavior_section(self, class_name, factors, rank_start=1):
        """Expandable section for one behavior class."""
        section = QFrame()
        section.setStyleSheet(
            "QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:10px;}"
        )
        sl = QVBoxLayout(section)
        sl.setContentsMargins(16, 10, 16, 10)
        sl.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        hdr.addWidget(self._lbl(f"🎯 {class_name}", "#0078D4", 15, True))
        hdr.addStretch()
        top_auc = factors[0]["auc"] if factors else 0
        hdr.addWidget(self._lbl(f"Top AUC: {top_auc:.4f}", "#16A34A", 13, True))
        sl.addLayout(hdr)

        # Factor grid: 2 columns
        grid = QHBoxLayout()
        grid.setSpacing(6)
        mid = (len(factors) + 1) // 2
        for col_factors in [factors[:mid], factors[mid:]]:
            col_lo = QVBoxLayout()
            col_lo.setSpacing(4)
            for f in col_factors:
                col_lo.addWidget(self._factor_mini_card(
                    f.get("name", "?"), f.get("auc", 0), f.get("target", ""),
                    factor_data=f.get("_full", f),
                ))
            col_lo.addStretch()
            grid.addLayout(col_lo, 1)
        sl.addLayout(grid)

        return section

    # ---- Actions ----
    def on_start(self):
        self._clear_cards()
        self._factors_data = []
        self._meta = None

        # Load factors directly for immediate display
        params = self.gather_params()
        factors_path = params.get("factors_path", "memory/evolved_factors.json")
        if Path(factors_path).exists():
            with open(factors_path, "r", encoding="utf-8") as f:
                self._factors_data = json.load(f)

        if self._factors_data:
            self._render_analysis()
        else:
            self.run_worker(AnalysisWorker, params=params)

    @Slot(object)
    def on_result(self, result):
        if not result:
            return
        # If worker produced results but factors weren't loaded yet, load them
        if not self._factors_data:
            params = self.gather_params()
            factors_path = params.get("factors_path", "memory/evolved_factors.json")
            if Path(factors_path).exists():
                with open(factors_path, "r", encoding="utf-8") as f:
                    self._factors_data = json.load(f)
        if self._factors_data:
            self._render_analysis()

    # ---- Render ----
    def _render_analysis(self):
        self._clear_cards()
        factors = self._factors_data
        params = self.gather_params()
        top_k = int(params.get("top_k_per_class", 10))
        top_n_features = int(params.get("top_n_features", 40))

        # === 1) Summary Row ===
        n = len(factors)
        aucs = [f.get("best_auc") or 0 for f in factors]
        avg_auc = sum(aucs) / n if n else 0
        max_auc = max(aucs) if aucs else 0
        targets = Counter(f.get("target", "?") for f in factors)
        n_targets = len(targets)

        # Features used
        import re
        feat_counter = Counter()
        for f in factors:
            for m in re.finditer(r"idx\.get\('([^']+)'", f.get("code", "")):
                feat_counter[m.group(1)] += 1
        top_feat = max(feat_counter, key=feat_counter.get) if feat_counter else "?"

        summary_row = QHBoxLayout()
        summary_row.setSpacing(8)
        summary_row.addWidget(self._stat_card("Factors", str(n), "#0078D4"))
        summary_row.addWidget(self._stat_card("Avg AUC", f"{avg_auc:.4f}", "#16A34A"))
        summary_row.addWidget(self._stat_card("Best AUC", f"{max_auc:.4f}", "#E67E22"))
        summary_row.addWidget(self._stat_card("Behaviors", str(n_targets), "#8E44AD"))
        summary_row.addWidget(self._stat_card("Top Feature", str(top_feat)[:20], "#555"))
        summary_row.addStretch()
        self._card_layout.insertLayout(0, summary_row)

        # === 2) Per-Behavior Top Factors ===
        # Build per-class AUC matrix
        all_classes = set()
        for f in factors:
            for cid in (f.get("per_class") or {}):
                all_classes.add(str(cid))
        class_ids = sorted(all_classes, key=lambda x: int(x) if x.isdigit() else x)
        class_names = {str(i): name for i, name in enumerate([
            "explore_object", "climb", "self_grooming", "stand",
            "blank", "positive_sniffs", "approach"
        ])}
        # Also check label_map from factors for class names
        if factors:
            first = factors[0]
            # Try to infer from targets
            for f in factors:
                t = f.get("target", "")
                if t:
                    for cid in list(class_ids):
                        if t.lower() in class_names.get(cid, "").lower():
                            class_names[cid] = t
                            break

        # Build matrix
        import numpy as np
        n_factors = len(factors)
        n_classes = len(class_ids)
        auc_matrix = np.full((n_factors, n_classes), np.nan)
        factor_names_list = [f.get("name", f"f_{i}") for i, f in enumerate(factors)]
        for i, f in enumerate(factors):
            pc = f.get("per_class", {})
            for j, cid in enumerate(class_ids):
                entry = pc.get(cid)
                if isinstance(entry, dict):
                    auc_matrix[i, j] = float(entry.get("auc", np.nan))
                elif isinstance(entry, (int, float)):
                    auc_matrix[i, j] = float(entry)

        # Top factors per class
        for j, cid in enumerate(class_ids):
            col = auc_matrix[:, j]
            valid_mask = ~np.isnan(col)
            if not valid_mask.any():
                continue
            indices = np.argsort(col[valid_mask])[::-1][:top_k]
            top_list = []
            for idx in np.where(valid_mask)[0][indices]:
                f = factors[idx]
                top_list.append({
                    "name": f.get("name", "?"),
                    "auc": float(col[idx]),
                    "target": f.get("target", ""),
                    "_full": f,  # Store full factor data for detail view
                })
            if top_list:
                label = class_names.get(str(cid), f"Class {cid}")
                self._add_card(self._behavior_section(label, top_list))

        # === 3) Feature Utilization ===
        feat_section = QFrame()
        feat_section.setStyleSheet(
            "QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:10px;}"
        )
        fsl = QVBoxLayout(feat_section)
        fsl.setContentsMargins(16, 10, 16, 10)
        fsl.setSpacing(6)
        fsl.addWidget(self._lbl("📊 Feature Utilization", "#0078D4", 15, True))

        top_features = feat_counter.most_common(top_n_features)
        if top_features:
            max_count = top_features[0][1]
            for feat_name, count in top_features[:30]:
                row = QHBoxLayout()
                row.setSpacing(8)
                row.addWidget(self._lbl(feat_name[:30], "#555", 10), 0)
                # Mini bar
                bar = QFrame()
                bar_w = int((count / max_count) * 200)
                bar.setFixedSize(max(bar_w, 4), 14)
                bar.setStyleSheet(
                    f"QFrame{{background:#0078D4;border:none;border-radius:3px;}}"
                )
                row.addWidget(bar)
                row.addWidget(self._lbl(str(count), "#999", 9))
                row.addStretch()
                fsl.addLayout(row)
        self._add_card(feat_section)

        # === 4) Add bottom stretch to keep cards at top ===
        self._card_layout.addStretch()

    # ---- Factor detail dialog ----
    def _show_factor_detail(self, factor):
        from gui.widgets.factor_detail import show_factor_detail
        show_factor_detail(factor, self)
