"""EvolutionWorker — runs evolution.py as subprocess, parses stdout for progress."""
import json
import os
import re
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker
from gui.workers.discovery_worker import _build_dataset_config, _parse_label_map


# Regex to parse evolution progress: "   12      150    0.7234    0.0312    0.6501    0.8412"
_RE_GEN_LINE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

# Per-generation summary: "=== Gen 12/50 ===" or "Generation 12/50"
_RE_GEN_HEADER = re.compile(r"(?:Gen|Generation)\s*(\d+)\s*/\s*(\d+)")

# New cycle: "=== Cycle 2/5 [LGBM macro AUC]: ... ===" or with "inf"
_RE_CYCLE = re.compile(r"=== Cycle\s+(\d+)/(\d+|inf)")


class EvolutionWorker(BaseWorker):
    """Runs evolution.py as a subprocess. Stop kills it cleanly."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params
        self._proc = None

    @Slot()
    def run(self):
        dataset_config_file = ""
        validate_config_path = ""
        try:
            params = self._params
            project_root = Path(__file__).resolve().parent.parent.parent

            # ---- Build validate config from GUI dataset settings ----
            # Use discovery tab's saved dataset settings if evolution tab doesn't have its own
            from gui.utils.gui_settings import get_tab_settings
            ds_params = get_tab_settings("discovery")
            if ds_params:
                # Merge: discovery dataset settings provide defaults for keys missing in evolution params
                for k in ("train_mouse_dirs", "train_tail_dirs", "train_behavior_m1_files",
                          "train_behavior_m2_files", "val_mouse_dirs", "val_tail_dirs",
                          "val_behavior_m1_files", "val_behavior_m2_files",
                          "max_instances", "label_map_pairs"):
                    if k not in params or not params[k]:
                        params[k] = ds_params.get(k, params.get(k))
                self.log("Using dataset paths from Discovery tab settings", 20)

            # Always build temp dataset config from GUI params (uses real data)
            dataset_config_file = _build_dataset_config(params) if params.get("train_mouse_dirs") else ""
            if dataset_config_file:
                self.log(f"Dataset config built from GUI settings: {dataset_config_file}", 20)
            else:
                self.log("WARNING: No dataset paths configured, evolution will fail without real data!", 30)

            # Build minimal validation config YAML pointing to the temp dataset config
            validate_config_path = params.get("validate_config", "config/seq/1.yaml")
            if dataset_config_file:
                _tmp_cfg = {
                    "dataset_config_file": dataset_config_file,
                    "sequence": {"seq_length": 1, "stride": 1, "frame_interval": 1,
                                "purity_threshold": 1.0, "boundary_margin": 10},
                    "preprocessing": {"augmentation": {"enabled": False}},
                }
                _tmp_fd, _tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="gui_evo_cfg_")
                os.close(_tmp_fd)
                with open(_tmp_path, "w", encoding="utf-8") as f:
                    import yaml as _yaml
                    _yaml.dump(_tmp_cfg, f)
                validate_config_path = _tmp_path
                self.log(f"Temporary validate config: {_tmp_path}", 20)

            # Build CLI
            cmd = [
                "python", str(project_root / "factors" / "evolution.py"),
                "--factors", params.get("factors_path", "memory/valid_factors.json"),
                "--output", params.get("output_path", "memory/evolved_factors.json"),
                "--mu", str(params.get("mu", 100)),
                "--lambda", str(params.get("lambda_", 100)),
                "--generations", str(params.get("generations", 50)),
                "--survivors", str(params.get("survivors", 0)),
                "--crossover-rate", str(params.get("crossover_rate", 0.65)),
                "--mutation-rate", str(params.get("mutation_rate", 0.35)),
                "--seed-factors", str(params.get("seed_factors", 30)),
                "--factor-injection-rate", str(params.get("injection_rate", 0.3)),
                "--num-workers", str(params.get("num_workers", 8)),
            ]

            # Always use LGBM validation mode
            cmd.extend([
                "--validate",
                "--validate-config", validate_config_path,
                "--validate-config-validation", params.get("validate_config_val", "config/validation.yaml"),
                "--validate-max-samples", str(int(params.get("max_samples", 10000))),
                "--validate-estimators", str(int(params.get("estimators", 50))),
            ])

            if params.get("use_gpu"):
                cmd.append("--gpu")

            std_threshold = params.get("std_threshold", 0.0)
            if std_threshold > 0:
                cmd.extend(["--std-threshold", str(std_threshold)])

            self.log(f"Launching: {' '.join(cmd)}", 20)
            self.log("Mode: LGBM validation", 20)

            self._proc = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )

            # Monitor stdout
            total_gen = int(params.get("generations", 50))
            while self._proc.poll() is None and not self._cancelled:
                if self._proc.stdout:
                    line = self._proc.stdout.readline()
                    if line:
                        self._parse_line(line.strip(), total_gen)
                else:
                    time.sleep(0.5)

            # Drain remaining stdout after process exits
            if self._proc and self._proc.stdout:
                for line in self._proc.stdout:
                    if line:
                        self._parse_line(line.strip(), total_gen)

            # If cancelled, terminate subprocess tree
            if self._cancelled and self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                self.log("Evolution cancelled.", 30)

            # Check return code
            rc = self._proc.returncode if self._proc else -1
            if rc != 0 and not self._cancelled:
                self.log(f"Evolution subprocess exited with code {rc}", 40)
                self.error.emit(f"Evolution failed (exit code {rc}). See log for details.")
                self.finished.emit()
                return

            # Read results
            output_path = params.get("output_path", "memory/evolved_factors.json")
            output_file = project_root / output_path
            if output_file.exists():
                with open(output_file, "r", encoding="utf-8") as f:
                    factors = json.load(f)
                self.log(f"Loaded {len(factors)} evolved factors", 20)
                self.result_ready.emit({
                    "success": True,
                    "n_factors": len(factors),
                    "factors": factors,
                })
            else:
                if not self._cancelled:
                    self.log("Evolution produced no output file.", 30)
                self.result_ready.emit({
                    "success": True,
                    "n_factors": 0,
                    "factors": [],
                })

        except Exception as e:
            self.log(f"Evolution error: {e}\n{traceback.format_exc()}", 40)
            self.error.emit(str(e))
        finally:
            self._proc = None
            # Clean up temp config files
            try:
                if dataset_config_file and os.path.exists(dataset_config_file):
                    os.unlink(dataset_config_file)
                if validate_config_path and os.path.exists(validate_config_path):
                    os.unlink(validate_config_path)
            except Exception:
                pass
            self.finished.emit()

    def _parse_line(self, line, total_gen):
        """Parse one stdout line from evolution.py."""

        # Parse factor inserted: __FACTOR__{json} (suppress from log, emit directly)
        if line.startswith("__FACTOR__"):
            try:
                factor_data = json.loads(line[len("__FACTOR__"):])
                self.partial_result.emit({
                    "type": "factor_inserted",
                    "factor": factor_data,
                })
            except json.JSONDecodeError:
                pass
            return

        # All other lines go to log viewer
        self.log_line.emit(line, 20)
        m = _RE_GEN_LINE.match(line.strip())
        if m:
            gen = int(m.group(1))
            if gen == 0:
                return  # skip initial population, start chart from gen 1
            avg = float(m.group(3))
            std_val = float(m.group(4))
            best = float(m.group(6))
            pct = min(int(gen / max(total_gen, 1) * 90), 90)
            self.progress.emit(pct, f"Gen {gen}/{total_gen} | Best: {best:.4f}")
            self.partial_result.emit({
                "type": "generation",
                "generation": gen,
                "nevals": int(m.group(2)),
                "avg_fitness": avg,
                "best_fitness": best,
                "std": std_val,
            })
            return

        # Parse gen header: "=== Gen 12/50 ===" or "best: gen=5 auc=0.85"
        m = _RE_GEN_HEADER.search(line)
        if m:
            gen = int(m.group(1))
            pct = min(int(gen / max(total_gen, 1) * 90), 90)
            self.progress.emit(pct, f"Gen {gen}/{total_gen}")
            return

        # Parse new cycle: "=== Cycle 2/5 [LGBM macro AUC]: ... ===" or with "inf"
        m = _RE_CYCLE.search(line)
        if m:
            cycle = int(m.group(1))
            max_cycles = 0 if m.group(2) == "inf" else int(m.group(2))
            self.log(f"New evolution cycle: {cycle}/{m.group(2)}", 20)
            self.partial_result.emit({
                "type": "cycle_reset",
                "cycle": cycle,
                "max_cycles": max_cycles,
            })
            return

    def cancel(self):
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
