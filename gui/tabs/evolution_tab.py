"""EvolutionTab — DEAP-GP factor evolution engine interface."""
import logging
from PySide6.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QProgressBar,
    QScrollArea, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot
from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.widgets.matplotlib_widget import MplWidget
from gui.widgets.factor_table import FactorTable
from gui.workers.evolution_worker import EvolutionWorker

logger = logging.getLogger("gui.evolution")


class EvolutionTab(BaseTab):
    def __init__(self, parent=None):
        self._fitness_plot = None
        self._evolved_table = None
        self._gen_cards = {}  # gen → card widget, for updating
        super().__init__(title="Factor Evolution", tab_key="evolution", parent=parent)

    def setup_params(self):
        self.add_config_group(ParameterGroup("I/O", [
            {"key": "factors_path", "label": "Input", "type": "file", "default": "memory/valid_factors.json"},
            {"key": "output_path", "label": "Output", "type": "file", "default": "memory/evolved_factors.json"},
        ]))
        self.add_config_group(ParameterGroup("Population", [
            {"key": "mu", "label": "Mu", "type": "int", "default": 100, "min": 10, "max": 10000},
            {"key": "lambda_", "label": "Lambda", "type": "int", "default": 100, "min": 10, "max": 10000},
            {"key": "generations", "label": "Generations", "type": "int", "default": 50, "min": 1, "max": 9999},
            {"key": "survivors", "label": "Cycles", "type": "int", "default": 1, "min": 1, "max": 9999},
        ]))
        self.add_config_group(ParameterGroup("Genetic", [
            {"key": "crossover_rate", "label": "Cross Rate", "type": "float", "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05},
            {"key": "mutation_rate", "label": "Mut Rate", "type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05},
            {"key": "seed_factors", "label": "Seeds", "type": "int", "default": 30, "min": 0, "max": 10000},
            {"key": "injection_rate", "label": "Inject Rate", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
        ]))
        self.add_config_group(ParameterGroup("Fitness", [
            {"key": "use_gpu", "label": "GPU", "type": "checkbox", "default": False},
            {"key": "num_workers", "label": "Workers", "type": "int", "default": 8, "min": 0, "max": 128},
            {"key": "validate_config", "label": "Val Config", "type": "file",
             "default": "config/seq/1.yaml"},
            {"key": "validate_config_val", "label": "Val Config 2", "type": "file",
             "default": "config/validation.yaml"},
            {"key": "max_samples", "label": "Max Samples", "type": "int", "default": 10000, "min": 100, "max": 100000},
            {"key": "estimators", "label": "Estimators", "type": "int", "default": 50, "min": 10, "max": 500},
        ]))

    def setup_results(self):
        # ---- Single dashboard row: progress bar + stat cards ----
        dash_row = QHBoxLayout()
        dash_row.setSpacing(6)

        self._gen_bar = QProgressBar()
        self._gen_bar.setRange(0, 100)
        self._gen_bar.setTextVisible(True)
        self._gen_bar.setFormat("Ready")
        self._gen_bar.setStyleSheet(
            "QProgressBar{border:1px solid #D0D0D0;border-radius:4px;background:#EEE;"
            "text-align:center;font-size:13px;height:36px;color:#333;}"
            "QProgressBar::chunk{background:#0078D4;border-radius:3px;}"
        )
        dash_row.addWidget(self._gen_bar, 3)

        self._stat_best_box, self._stat_best_val = self._make_stat_card("Best", "—", "#16A34A")
        self._stat_avg_box, self._stat_avg_val = self._make_stat_card("Avg", "—", "#0078D4")
        self._stat_std_box, self._stat_std_val = self._make_stat_card("Std", "—", "#888")
        gen_label = QLabel("Generations")
        gen_label.setStyleSheet("color:#888;font-size:9px;")
        self._stat_gen_val = QLabel(f"? / ?")
        self._stat_gen_val.setStyleSheet("color:#555;font-size:16px;font-weight:bold;")

        dash_row.addWidget(self._stat_best_box)
        dash_row.addWidget(self._stat_avg_box)
        dash_row.addWidget(self._stat_std_box)
        self._results_layout_main.addLayout(dash_row)

        # ---- Charts row: fitness + diversity ----
        charts_row = QHBoxLayout()
        charts_row.setSpacing(6)
        self._fitness_plot = MplWidget(figsize=(4, 2.5), toolbar=False)
        self._diversity_plot = MplWidget(figsize=(4, 2.5), toolbar=False)
        charts_row.addWidget(self._fitness_plot, 1)
        charts_row.addWidget(self._diversity_plot, 1)
        self._results_layout_main.addLayout(charts_row)

        # ---- Evolved factor table ----
        self._evolved_table = FactorTable()
        self._evolved_table.factor_double_clicked.connect(self._show_factor_detail)
        self._results_layout_main.addWidget(self._evolved_table)

    def _make_stat_card(self, label, value, color):
        box = QFrame()
        box.setStyleSheet("QFrame{background:#FFF;border:1px solid #E0E0E0;border-radius:5px;}")
        box.setFixedWidth(95)
        bl = QVBoxLayout(box)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(1)
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#888;font-size:10px;font-weight:500;")
        bl.addWidget(lbl, 0, Qt.AlignCenter)
        val_lbl = QLabel(value)
        val_lbl.setStyleSheet(f"color:{color};font-size:17px;font-weight:bold;")
        bl.addWidget(val_lbl, 0, Qt.AlignCenter)
        return box, val_lbl

    def on_start(self):
        self._fitness_history = {"gen": [], "avg": [], "best": [], "std": []}
        self._gen_bar.setValue(0)
        self._gen_bar.setFormat("Starting...")
        self._stat_best_val.setText("—")
        self._stat_avg_val.setText("—")
        self._stat_std_val.setText("—")
        # Clear factor table for fresh run
        if self._evolved_table:
            self._evolved_table.clear()
        # Clear and create fresh axes
        self._clear_figures()
        self._ax_fit = self._fitness_plot.subplot(111)
        self._ax_div = self._diversity_plot.subplot(111)
        self._ax_fit.set_xlabel("Generation"); self._ax_fit.set_ylabel("Fitness")
        self._ax_div.set_xlabel("Generation"); self._ax_div.set_ylabel("Value")
        self._fitness_plot.draw()
        self._diversity_plot.draw()
        self.run_worker(EvolutionWorker, params=self.gather_params())

    def _clear_figures(self):
        """Completely clear both figures and force canvas refresh."""
        for plot in [self._fitness_plot, self._diversity_plot]:
            plot.figure.clf()
            plot.canvas.draw_idle()

    @Slot(object)
    def on_partial_result(self, data):
        if not isinstance(data, dict):
            return

        if data.get("type") == "cycle_reset":
            cycle = data.get("cycle", 0)
            self._fitness_history = {"gen": [], "avg": [], "best": [], "std": []}
            # Re-create fresh axes
            self._clear_figures()
            self._ax_fit = self._fitness_plot.subplot(111)
            self._ax_div = self._diversity_plot.subplot(111)
            self._ax_fit.set_xlabel("Generation"); self._ax_fit.set_ylabel("Fitness")
            self._ax_fit.set_title(f"Fitness — Cycle {cycle}", fontsize=10)
            self._ax_div.set_xlabel("Generation"); self._ax_div.set_ylabel("Value")
            self._ax_div.set_title(f"Diversity — Cycle {cycle}", fontsize=10)
            self._fitness_plot.draw()
            self._diversity_plot.draw()
            self._gen_bar.setFormat(f"Cycle {cycle} — Starting...")
            self._stat_best_val.setText("—")
            self._stat_avg_val.setText("—")
            self._stat_std_val.setText("—")
            return

        if data.get("type") == "factor_inserted":
            factor = data.get("factor", {})
            if factor and self._evolved_table:
                self._evolved_table.append_factor(factor)
            return

        if data.get("type") != "generation":
            return

        gen = data.get("generation", 0)
        avg = data.get("avg_fitness", 0)
        best = data.get("best_fitness", 0)
        std_val = data.get("std", 0)
        nevals = data.get("nevals", 0)

        self._fitness_history["gen"].append(gen)
        self._fitness_history["avg"].append(avg)
        self._fitness_history["best"].append(best)
        self._fitness_history["std"].append(std_val)

        # Update stat cards
        self._stat_best_val.setText(f"{best:.4f}")
        self._stat_avg_val.setText(f"{avg:.4f}")
        self._stat_std_val.setText(f"{std_val:.4f}")

        # Update progress bar
        params = self.gather_params()
        total_gen = int(params.get("generations", 50))
        pct = int(gen / max(total_gen, 1) * 100)
        self._gen_bar.setValue(min(pct, 100))
        self._gen_bar.setFormat(f"Gen {gen}/{total_gen}  |  Best {best:.4f}  |  Avg {avg:.4f}  |  {nevals} evals")

        gens = self._fitness_history["gen"]

        # Left chart — fitness over generations (reuse axes)
        self._ax_fit.clear()
        self._ax_fit.set_xlabel("Generation"); self._ax_fit.set_ylabel("Fitness")
        self._ax_fit.plot(gens, self._fitness_history["avg"], "b-o", alpha=0.7, markersize=3, label="Avg")
        self._ax_fit.plot(gens, self._fitness_history["best"], "r-o", markersize=3, label="Best")
        self._ax_fit.legend(fontsize=8)
        self._ax_fit.set_title(f"Fitness — Gen {gen}", fontsize=10)

        # Right chart — diversity (std) + gap
        self._ax_div.clear()
        self._ax_div.set_xlabel("Generation"); self._ax_div.set_ylabel("Value")
        self._ax_div.plot(gens, self._fitness_history["std"], "purple", marker="o", alpha=0.8, markersize=3, label="Std Dev")
        gap = [b - a for a, b in zip(self._fitness_history["avg"], self._fitness_history["best"])]
        self._ax_div.plot(gens, gap, "orange", marker="o", alpha=0.8, markersize=3, label="Gap")
        self._ax_div.legend(fontsize=8)
        self._ax_div.set_title(f"Diversity — Gen {gen}", fontsize=10)

        self._fitness_plot.draw()
        self._diversity_plot.draw()

    @Slot(object)
    def on_result(self, result):
        if result and isinstance(result, dict):
            factors = result.get("factors", [])
            # Show all evolved factors — those without holdout validation
            # (non-convergence mode) will show '?' for AUC column
            if factors:
                self._evolved_table.load_factors(factors)
            self._gen_bar.setValue(100)
            self._gen_bar.setFormat(f"Complete! {len(factors)} factors evolved")
        # Reset plots so stale curves don't persist on next run
        self._clear_figures()
        self._ax_fit = self._fitness_plot.subplot(111)
        self._ax_div = self._diversity_plot.subplot(111)
        self._ax_fit.set_xlabel("Generation"); self._ax_fit.set_ylabel("Fitness")
        self._ax_div.set_xlabel("Generation"); self._ax_div.set_ylabel("Value")
        self._fitness_plot.draw()
        self._diversity_plot.draw()
        self._stat_best_val.setText("—")
        self._stat_avg_val.setText("—")
        self._stat_std_val.setText("—")

    def _show_factor_detail(self, factor):
        from gui.widgets.factor_detail import show_factor_detail
        show_factor_detail(factor, parent=self)
