"""BatchMiningWorker — runs batch_mining.py as subprocess and reports progress."""
import json
import subprocess
import time
import re
from pathlib import Path

from PySide6.QtCore import Slot

from gui.workers.base_worker import BaseWorker


class BatchMiningWorker(BaseWorker):
    """Launches batch_mining.py and monitors factor files for progress."""

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self._params = params
        self._proc = None
        self._cancelled = False

    @Slot()
    def run(self):
        try:
            params = self._params
            project_root = Path(__file__).resolve().parent.parent.parent

            # Build command
            seq_str = params.get("seq_groups", "1,5,15")
            target_n = int(params.get("target_n", 600))
            tasks = ",".join(f"{s.strip()}:{target_n}" for s in seq_str.split(",") if s.strip().isdigit())

            cmd = [
                "python", str(project_root / "mining" / "batch_mining.py"),
                "--tasks", tasks,
            ]
            mem_limit = int(params.get("mem_limit", 0))
            cpu_limit = int(params.get("cpu_limit", 0))
            if mem_limit > 0:
                cmd.extend(["--mem-limit", str(mem_limit)])
            if cpu_limit > 0:
                cmd.extend(["--cpu-limit", str(cpu_limit)])
            if not params.get("auto_merge", True):
                cmd.append("--no-merge")

            self.log(f"Starting: {' '.join(cmd)}", 20)

            # Parse seqs for tracking
            seqs = [int(s.strip()) for s in seq_str.split(",") if s.strip().isdigit()]
            self._emit_progress_init(seqs, target_n)

            # Snapshot existing factors before launching — only report NEW factors
            known_factors = {}
            self._initial_counts = {}
            for seq in seqs:
                existing = set()
                ff = project_root / "memory" / f"seq{seq}" / "valid_factors.json"
                if ff.exists():
                    try:
                        with open(ff, encoding="utf-8") as f:
                            for fac in json.load(f):
                                existing.add(fac.get("name", ""))
                    except Exception:
                        pass
                known_factors[seq] = existing
                self._initial_counts[seq] = len(existing)
                # Emit initial count
                self.partial_result.emit({
                    "type": "seq_progress", "seq": seq,
                    "count": len(existing), "target": target_n,
                    "pct": min(int(len(existing) / target_n * 100), 100) if target_n > 0 else 0,
                })

            # Launch subprocess
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

            # Quick check: did the subprocess die immediately?
            time.sleep(1.0)
            if self._proc.poll() is not None:
                # Process already exited — read error output
                out = self._proc.stdout.read() if self._proc.stdout else ""
                self.log(f"batch_mining.py exited immediately (code={self._proc.returncode})", 40)
                if out:
                    self.log(f"Output: {out[:500]}", 40)
                self.error.emit(f"batch_mining.py exited with code {self._proc.returncode}")
                self.finished.emit()
                return

            # Monitor loop
            while self._proc.poll() is None and not self._cancelled:
                # Read any new output lines
                if self._proc.stdout:
                    line = self._proc.stdout.readline()
                    if line:
                        self._parse_line(line.strip(), seqs, target_n)

                # Check factor files for new factors
                self._check_factor_files(seqs, target_n, known_factors)

                time.sleep(1.5)

            # Final check
            self._check_factor_files(seqs, target_n, known_factors)

            if self._cancelled and self._proc and self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=5)
                self.log("Mining cancelled.", 30)
            else:
                self.log("Batch mining complete.", 20)

            # Emit final result with all discovered factors
            all_factors = []
            for seq in seqs:
                ff = project_root / "memory" / f"seq{seq}" / "valid_factors.json"
                if ff.exists():
                    with open(ff, encoding="utf-8") as f:
                        all_factors.extend(json.load(f))

            self.result_ready.emit({
                "success": True,
                "n_factors": len(all_factors),
                "factors": all_factors,
            })

        except Exception as e:
            import traceback
            self.log(f"Batch mining error: {e}\n{traceback.format_exc()}", 40)
            self.error.emit(str(e))
        finally:
            self._proc = None
            self.finished.emit()

    def _parse_line(self, line, seqs, target_n):
        """Parse a stdout line from batch_mining.py dashboard."""
        self.log_line.emit(line, 20)

        # Parse dashboard line: seq= 1 | target=1000 | total= 601/1000 | this_run=+  0 | need= 399 | running=1
        m = re.search(
            r"seq=\s*(\d+)\s*\|\s*target=\s*(\d+)\s*\|\s*total=\s*(\d+)/(\d+)\s*\|"
            r"\s*this_run=\+?\s*(\d+)\s*\|\s*need=\s*(\d+)\s*\|\s*running=(\d+)",
            line
        )
        if m:
            seq = int(m.group(1))
            current = int(m.group(3))
            target = int(m.group(4))
            new_this_run = int(m.group(5))
            running = int(m.group(7))
            pct = int(current / target * 100) if target > 0 else 0
            self.partial_result.emit({
                "type": "seq_progress",
                "seq": seq,
                "count": current,
                "target": target,
                "pct": min(pct, 100),
                "running": running,
                "new_this_run": new_this_run,
            })

    def _emit_progress_init(self, seqs, target_n):
        """Emit initial progress for all seq groups."""
        for seq in seqs:
            self.partial_result.emit({
                "type": "seq_init",
                "seq": seq,
                "target": target_n,
            })

    def _check_factor_files(self, seqs, target_n, known_factors):
        """Check factor files for new factors and report them."""
        project_root = Path(__file__).resolve().parent.parent.parent
        for seq in seqs:
            ff = project_root / "memory" / f"seq{seq}" / "valid_factors.json"
            if not ff.exists():
                continue

            try:
                with open(ff, encoding="utf-8") as f:
                    factors = json.load(f)
            except Exception:
                continue

            current = len(factors)
            known = known_factors.get(seq, set())
            new_factors = [f for f in factors if f.get("name") not in known]
            for f in new_factors:
                known.add(f.get("name", ""))
            known_factors[seq] = known

            # Emit progress (count/pct only; running/worker info comes from dashboard)
            pct = int(current / target_n * 100) if target_n > 0 else 0
            self.partial_result.emit({
                "type": "seq_progress",
                "seq": seq,
                "count": current,
                "target": target_n,
                "pct": min(pct, 100),
                "new_this_run": len(known_factors.get(seq, set())) - self._initial_counts.get(seq, 0),
            })

            # Emit new factors as cards
            for f in new_factors:
                self.partial_result.emit({
                    "type": "factor_found",
                    "seq": seq,
                    "name": f.get("name", "unknown"),
                    "code": f.get("code", ""),
                    "description": f.get("description", ""),
                    "target": f.get("target", ""),
                    "valid": f.get("valid", True),
                    "best_auc": f.get("best_auc"),
                    "best_f1": f.get("best_f1"),
                    "best_class": f.get("best_class", ""),
                    "valid_classes": f.get("valid_classes", []),
                    "per_class": f.get("per_class", {}),
                    "seq_length": f.get("seq_length", seq),
                    "reason": f.get("reason", ""),
                })

    def cancel(self):
        self._cancelled = True
