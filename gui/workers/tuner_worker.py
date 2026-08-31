"""
TunerWorker — runs tuner.py as a subprocess, parses stdout for live progress.
Stop kills the subprocess tree cleanly. No threading complexity.
"""

import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker

# Parse status panel lines
_RE_SEEDS = re.compile(r"Seeds:\s*(\d+)/(\d+)")
_RE_VARIANTS = re.compile(r"Variants:\s*(\d+)/(\d+)")
_RE_CPU = re.compile(r"CPU:\s*(\d+)%")
_RE_IMPROVEMENT = re.compile(r"\[\+\]")

# Parse improvement lines: "[+] 71.0% -> 73.5%  (+2.5%)  factor_name"
# Use string splitting (more robust than regex for this format)
def _parse_improvement(line):
    """Extract factor name from a StatusPanel improvement line. Returns name or None."""
    if not line.startswith("[+]") or "->" not in line:
        return None
    # Format: "[+] OLD% -> NEW%  (+DELTA%)  name"
    # Name is after the last "  " (double space before factor name)
    idx = line.rfind("  ")
    if idx < 0:
        return None
    name = line[idx + 2:].strip()
    return name if name else None

# Phase transitions
_RE_SEQ_GROUP = re.compile(r"--- seq=(\d+)")
_RE_DONE = re.compile(r"DONE:\s*(\d+)\s+factors?,\s*(\d+)\s+improved")


class TunerWorker(BaseWorker):
    """Runs tuner.py as subprocess. Parses stdout for progress & results."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params
        self._proc = None

    @Slot()
    def run(self):
        try:
            params = self._params
            project_root = Path(__file__).resolve().parent.parent.parent

            output_path = params.get("output_path", "memory/tuned_factors.json")
            input_path = params.get("input_path", "memory/valid_factors_deduped.json")

            # Build CLI — mirrors tuner.py argparse
            cmd = [
                sys.executable, str(project_root / "factors" / "tuner.py"),
                "--input", input_path,
                "--output", output_path,
                "--config-common", params.get("common_config", "config/seq/1.yaml"),
                "--config-validation", params.get("validation_config", "config/validation.yaml"),
                "--num-workers", str(params.get("num_workers", 8)),
                "--max-param-combos", str(params.get("max_param_combos", 20)),
                "--max-variants-per-seed", str(params.get("max_variants_per_seed", 50)),
            ]

            if not params.get("enable_structure", True):
                cmd.append("--no-structure-mutation")

            max_factors = int(params.get("max_factors", 0))
            if max_factors > 0:
                cmd.extend(["--max-factors", str(max_factors)])

            if not params.get("resume", True):
                cmd.append("--no-resume")

            self.log(f"Launching: {' '.join(cmd)}", 20)
            self.set_progress(5, "Starting tuner.py...")

            self._proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            # State tracking
            seeds_done = 0
            seeds_total = 0
            variants_done = 0
            n_improved = 0
            emitted_count = 0  # how many factors we've already sent to the tab

            while self._proc.poll() is None and not self._cancelled:
                line = ""
                if self._proc.stdout:
                    line = self._proc.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue

                line = line.strip()
                if not line:
                    continue

                # Forward to log viewer
                self.log(line, 20)

                # ── Parse progress ──
                m = _RE_SEEDS.search(line)
                if m:
                    seeds_done = int(m.group(1))
                    seeds_total = int(m.group(2))
                    pct = seeds_done * 100 // max(seeds_total, 1)
                    self.set_progress(min(pct, 90), f"Seeds: {seeds_done}/{seeds_total}")

                m = _RE_VARIANTS.search(line)
                if m:
                    variants_done = int(m.group(1))
                    variants_total = int(m.group(2))
                    self.set_progress(min(seeds_done * 100 // max(seeds_total, 1), 90),
                                      f"Variants: {variants_done}/{variants_total}")

                # ── Check for new improved factors in output file ──
                if _RE_IMPROVEMENT.search(line):
                    n_new = self._emit_new_factors(output_path, emitted_count)
                    if n_new > emitted_count:
                        emitted_count = n_new
                        n_improved = emitted_count

                # ── Broadcast progress ──
                if _RE_IMPROVEMENT.search(line) or _RE_SEEDS.search(line):
                    self.partial_result.emit({
                        "seeds_done": seeds_done,
                        "seeds_total": seeds_total,
                        "n_improved": n_improved,
                    })

            # ── Cancellation ──
            if self._cancelled and self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                self.log("Tuning cancelled.", 30)

            # ── Read results: only improved factors ──
            self.set_progress(95, "Loading results...")
            output_file = project_root / output_path
            tuned_factors = []
            if output_file.exists():
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        all_factors = json.load(f)
                    # Only keep factors with actual AUC improvement
                    tuned_factors = [
                        f for f in all_factors
                        if (f.get("_auc_improvement") or 0) > 0.001
                    ]
                    self.log(f"Loaded {len(tuned_factors)} improved factors "
                             f"(from {len(all_factors)} total)", 20)
                except Exception as e:
                    self.log(f"Error reading output: {e}", 40)

            self.set_progress(100, "Complete")
            self.result_ready.emit({
                "success": True,
                "tuned_factors": tuned_factors,
                "n_total": len(tuned_factors),
                "output_path": output_path,
                "message": f"Tuning complete. {len(tuned_factors)} improved factors.",
            })

        except Exception as e:
            self.log(f"Tuner error: {e}\n{traceback.format_exc()}", 40)
            self.error.emit(str(e))
        finally:
            self._proc = None
            self.finished.emit()

    def _emit_new_factors(self, output_path, emitted_count):
        """Read output file and emit any newly improved factors beyond emitted_count."""
        output_file = Path(__file__).resolve().parent.parent.parent / output_path
        if not output_file.exists():
            return emitted_count
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                all_factors = json.load(f)
        except Exception:
            return emitted_count

        # Find new improved factors beyond what we've already emitted
        new_improved = [
            f for f in all_factors[emitted_count:]
            if (f.get("_auc_improvement") or 0) > 0.001
        ]
        for fac in new_improved:
            self.partial_result.emit({
                "new_factor": fac,
            })
            self.log(f"  [+] {fac.get('name', '?')}  AUC={fac.get('best_auc', 0):.4f}", 20)

        return len(all_factors)

    def cancel(self):
        """Kill subprocess tree on Stop."""
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
