#!/usr/bin/env python
"""
Batch factor mining scheduler -- fully automatic resource-aware parallel mining.

Features:
  - Just specify seq groups and target N, no need to manually set worker count
  - Continuously monitor CPU/memory, auto-start new workers when resources permit
  - Fixed dashboard showing system status + mining progress for each worker in real time
  - Automatic cleanup on abnormal process exit, auto-restart when resources allow
  - Separate temp library -> formal library, three-layer deduplication

Usage:
  python batch_mining.py --seq 1,5,15 --target 600 --mem-limit 60
  python batch_mining.py --tasks 1:600,5:300 --mem-limit 60 --cpu-limit 80
  python batch_mining.py --seq 1,5 --target 600 --mem-limit 60 --no-merge  # disable auto-merge
"""


import sys
from pathlib import Path
# Allow running as a script from a subdirectory (e.g. `python mining/discovery.py`)
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import subprocess
import sys
import os
import gc
import time
import json
import re
import argparse
import signal
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import yaml
import psutil
import shutil

# ---------------------------------------------------------------------------
# stdout UTF-8
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Resolve relative to the repository root (this file lives in mining/), so the
# script works regardless of the working directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
MEMORY_DIR = PROJECT_ROOT / "memory"

# Log stage detection
RE_DATA_LOADING_DONE = re.compile(r"(Training set: \d+ windows|val window matrix)")
RE_MINING_START      = re.compile(r"Round\s+1")
RE_ROUND             = re.compile(r"Round\s+(\d+)")
RE_BEST_AUC          = re.compile(r"best_auc[=:]\s*([\d.]+)")
RE_SAVED_FACTOR      = re.compile(r"\[Memory\]\s+Saving valid factor")
RE_DISCOVERY_DONE    = re.compile(r"Factor discovery complete")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def log_temp(msg: str):
    """Temporary log (will be overwritten by dashboard), used for debugging."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def count_factors_shared(seq: int) -> int:
    """Read factor count from the formal library."""
    p = MEMORY_DIR / f"seq{seq}" / "valid_factors.json"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return len(json.load(f))
        except Exception:
            return 0
    return 0


def merge_all_seq_dirs(verbose: bool = True) -> int:
    """Scan all directories under memory/seq*/ and merge into memory/valid_factors.json.

    Does not depend on currently running groups; results from multiple different --seq runs can all be merged.
    Returns the total factor count after merging.
    """
    all_factors = []
    seen_names = set()

    seq_dirs = sorted(
        [d for d in MEMORY_DIR.glob("seq*") if d.is_dir()],
        key=lambda d: d.name,
    )
    for seq_dir in seq_dirs:
        seq_name = seq_dir.name  # e.g. "seq1", "seq15"
        p = seq_dir / "valid_factors.json"
        if not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                factors = json.load(f)
        except Exception:
            continue
        for fac in factors:
            name = fac.get("name", "")
            if name in seen_names:
                fac = dict(fac)
                fac["name"] = f"{name}_{seq_name}"
            seen_names.add(fac["name"])
            all_factors.append(fac)

    out_path = MEMORY_DIR / "valid_factors.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_factors, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"Merged {len(all_factors)} factors from {len(seq_dirs)} groups -> {out_path}")
    return len(all_factors)


def count_factors_worker(seq: int, worker_id: str) -> int:
    """Read factor count from a single worker's temp library."""
    p = MEMORY_DIR / f"seq{seq}" / "workers" / worker_id / "valid_factors.json"
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return len(data)
            else:
                with open(str(PROJECT_ROOT / "batch_progress.log"), "a", encoding="utf-8") as lf:
                    lf.write(f"[WARN] count_factors_worker({seq},{worker_id}): not a list ({type(data).__name__})\n")
                return 0
        except Exception as e:
            with open(str(PROJECT_ROOT / "batch_progress.log"), "a", encoding="utf-8") as lf:
                lf.write(f"[WARN] count_factors_worker({seq},{worker_id}): {type(e).__name__}: {e}\n")
            return 0
    return 0


def _cleanup_worker_dir(w: "Worker"):
    """Clean up temporary files from a completed worker to free disk and OS page cache.

    Note: keep valid_factors.json and experience.json because the merge thread may still need them.
    Only clean up log files (which can become very large) and config cache.
    """
    import gc as _gc
    wdir = w.memory_dir
    if not wdir.exists():
        return
    # Delete log files (can be very large after hours of running)
    log_file = wdir / "run.log"
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception:
            pass
    # Delete config cache (YAML file, no need to keep)
    cfg_file = wdir / "config.yaml"
    if cfg_file.exists():
        try:
            cfg_file.unlink()
        except Exception:
            pass
    _gc.collect()


def find_configs(seq: int):
    common = CONFIG_DIR / "seq" / f"{seq}.yaml"
    discovery = CONFIG_DIR / "seq" / f"{seq}.yaml"
    if not common.exists() or not discovery.exists():
        return None, None
    return str(common), str(discovery)


def get_system_stats():
    cpu = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": cpu,
        "memory_percent": mem.percent,
        "memory_available_gb": mem.available / (1024**3),
        "memory_total_gb": mem.total / (1024**3),
    }


def progress_bar(pct: float, width: int = 20) -> str:
    """ASCII progress bar."""
    filled = int(width * pct / 100)
    if filled > width:
        filled = width
    if filled < 0:
        filled = 0
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}]"


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def generate_worker_config(seq: int, worker_id: str, common_src: str, discovery_src: str,
                         target: int = 600) -> str:
    with open(common_src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    with open(discovery_src, "r", encoding="utf-8") as f:
        cfg.update(yaml.safe_load(f) or {})

    worker_dir = MEMORY_DIR / f"seq{seq}" / "workers" / worker_id
    # Clean old data to ensure mine starts from 0
    if worker_dir.exists():
        shutil.rmtree(worker_dir, ignore_errors=True)
    worker_dir.mkdir(parents=True, exist_ok=True)

    cfg.setdefault("output", {})["memory_dir"] = str(worker_dir)

    # dataset_config_file: discovery.py resolves relative to the config file's directory, must convert to absolute path
    config_dir = Path(common_src).resolve().parent  # config/
    for key in ("dataset_config_file",):
        val = cfg.get(key, "")
        if val and not Path(val).is_absolute():
            cfg[key] = str(config_dir / val)
    # behavior_rules: HypothesisGenerator uses open() relative to CWD, keep as is

    shared_factors = str(MEMORY_DIR / f"seq{seq}" / "valid_factors.json")
    shared_exp = str(MEMORY_DIR / f"seq{seq}" / "experience.json")
    cfg.setdefault("loop", {}).update({
        "reload_each_round": True,
        "shared_factors_path": shared_factors,
        "shared_experience_path": shared_exp,
        "worker_id": worker_id,
        "max_valid_factors": target,   # Use the target passed by scheduler; auto-stop when shared library reaches target
        "max_rounds": 999999,
    })

    # Per-worker diversity direction (avoid multiple workers saturating the same feature combination)
    DIVERSITY_FOCI = [
        "social", "skeleton_self", "skeleton_other",
        "motion_self", "motion_other", "tail_self",
        "tail_other", "cross_group",
    ]
    worker_idx = int(worker_id.replace("w", "")) if worker_id.startswith("w") else 0
    cfg["loop"]["diversity_focus"] = DIVERSITY_FOCI[worker_idx % len(DIVERSITY_FOCI)]

    out_path = worker_dir / "config.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    return str(out_path)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class Worker:
    def __init__(self, seq: int, worker_id: str, config_path: str):
        self.seq = seq
        self.worker_id = worker_id
        self.config_path = config_path
        self.proc: Optional[subprocess.Popen] = None
        self.log_path: Optional[Path] = None
        self.state = "PENDING"
        self.started_at: float = 0
        self.factor_count: int = 0          # Current factor count (read from temp library)
        self.prev_factor_count: int = 0     # Previous cycle factor count (for scaling decisions)
        self.display_factor_count: int = 0  # Factor count at last dashboard render (for delta)
        self.current_round: int = 0         # Current round
        self.auc_sum: float = 0.0         # Cumulative AUC (for averaging)
        self.auc_count: int = 0           # AUC sample count
        self.last_factor_time: float = 0  # Timestamp of most recent factor discovery
        self._log_tail_offset: int = 0
        self._restart_count: int = 0     # Number of restarts

    @property
    def label(self) -> str:
        return f"seq={self.seq}/{self.worker_id}"

    @property
    def memory_dir(self) -> Path:
        return MEMORY_DIR / f"seq{self.seq}" / "workers" / self.worker_id

    def launch(self, python: str, env: dict):
        self.log_path = self.memory_dir / "run.log"
        # Append rather than overwrite (for restart scenarios)
        log_file = open(str(self.log_path), "w" if self._restart_count == 0 else "a", encoding="utf-8")
        if self._restart_count > 0:
            log_file.write(f"\n\n=== RESTART #{self._restart_count} at {datetime.now().isoformat()} ===\n\n")
        self.proc = subprocess.Popen(
            [python, "-u", str(PROJECT_ROOT / "mining" / "discovery.py"), "--config", self.config_path],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        log_file.close()
        self.state = "LOADING"
        self.started_at = time.time()
        self._log_tail_offset = 0
        self.factor_count = 0
        self.prev_factor_count = 0
        self.display_factor_count = 0
        self.current_round = 0
        self.auc_sum = 0.0
        self.auc_count = 0

    def poll(self) -> Optional[int]:
        if self.proc is None:
            return -1
        return self.proc.poll()

    def terminate(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
                except Exception:
                    pass

    def read_new_log(self) -> str:
        if not self.log_path or not self.log_path.exists():
            return ""
        try:
            size = self.log_path.stat().st_size
            if size <= self._log_tail_offset:
                return ""
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._log_tail_offset)
                text = f.read()
                self._log_tail_offset = f.tell()
            return text
        except Exception:
            return ""

    def update_stats(self):
        """Extract statistics from new log text."""
        text = self.read_new_log()
        if not text:
            return

        # Current round
        for m in RE_ROUND.finditer(text):
            self.current_round = int(m.group(1))

        # AUC average (all factors that passed validation)
        for m in RE_BEST_AUC.finditer(text):
            self.auc_sum += float(m.group(1))
            self.auc_count += 1

        # Factor save timestamp
        if RE_SAVED_FACTOR.search(text):
            self.last_factor_time = time.time()

        # Phase transitions
        if self.state == "LOADING":
            if RE_DATA_LOADING_DONE.search(text) or RE_MINING_START.search(text):
                self.state = "MINING"

        # Normal completion
        if RE_DISCOVERY_DONE.search(text):
            self.state = "DONE"

    def check_failure(self) -> bool:
        if self.state in ("DONE", "FAILED"):
            return self.state == "FAILED"
        text = self.read_new_log()
        if "Traceback" in text:
            self.state = "FAILED"
            return True
        return False

    def refresh_factor_count(self):
        self.prev_factor_count = self.factor_count
        self.factor_count = count_factors_worker(self.seq, self.worker_id)

    def elapsed(self) -> float:
        if self.started_at == 0:
            return 0
        return time.time() - self.started_at

    def get_log_tail(self, n: int = 6) -> str:
        """Read the last N lines of the log, for failure diagnosis."""
        if not self.log_path or not self.log_path.exists():
            return "(no log)"
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            tail = [l.rstrip() for l in lines[-n:]]
            return "\n".join(f"      {l}" for l in tail)
        except Exception:
            return "(read error)"

    def elapsed_str(self) -> str:
        secs = self.elapsed()
        if secs < 60:
            return f"{secs:.0f}s"
        m = int(secs // 60)
        if m < 60:
            return f"{m}m"
        h = m // 60
        return f"{h}h{m % 60}m"


# ---------------------------------------------------------------------------
# SeqGroup
# ---------------------------------------------------------------------------
class SeqGroup:
    MAX_WORKERS = 8              # Max workers per group
    MAX_CONSECUTIVE_FAILURES = 3 # Consecutive failure limit
    MERGE_ERROR_COOLDOWN = 300   # Merge error log cooldown (seconds)

    def __init__(self, seq: int, target: int, common_cfg: str, discovery_cfg: str):
        self.seq = seq
        self.target = target
        self.common_cfg = common_cfg
        self.discovery_cfg = discovery_cfg
        self.workers: list[Worker] = []
        self._next_wid = 0
        self._consecutive_failures = 0
        self._last_failure_time: float = 0
        self._last_merge_error_time: float = 0
        self._zero_output_streak = 0       # Consecutive zero-output worker count

        self.shared_factors_path = MEMORY_DIR / f"seq{seq}" / "valid_factors.json"
        self.shared_experience_path = MEMORY_DIR / f"seq{seq}" / "experience.json"

    @property
    def shared_factor_count(self) -> int:
        return count_factors_shared(self.seq)

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.shared_factor_count)

    @property
    def running_count(self) -> int:
        return sum(1 for w in self.workers if w.state in ("LOADING", "MINING"))

    @property
    def loading_count(self) -> int:
        return sum(1 for w in self.workers if w.state == "LOADING")

    def all_done(self) -> bool:
        return all(w.state in ("DONE", "FAILED") for w in self.workers)

    def create_worker(self) -> Worker:
        wid = f"w{self._next_wid}"
        self._next_wid += 1
        cfg_path = generate_worker_config(self.seq, wid, self.common_cfg, self.discovery_cfg,
                                          target=self.target)
        w = Worker(self.seq, wid, cfg_path)
        self.workers.append(w)
        return w

    @staticmethod
    def _atomic_replace(src: Path, dst: Path, max_retries: int = 5):
        """Windows-tolerant atomic replace: retry + fallback to non-atomic write.

        On Windows, os.replace() can throw PermissionError when the target file is being read
        by another process (antivirus, worker subprocess reading shared files, etc.).
        Retry several times; if still failing, fall back to delete-then-rename.
        """
        import errno
        for attempt in range(max_retries):
            try:
                src.replace(dst)
                return
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))  # Incremental wait: 0.1, 0.2, 0.3...
                else:
                    # Final fallback: delete target first, then rename
                    try:
                        dst.unlink(missing_ok=True)
                        src.replace(dst)
                    except Exception:
                        # Still failing, abandon this write without losing data
                        if time.time() - getattr(SeqGroup, '_last_replace_fail_log', 0) > 30:
                            log_temp(f"[WARN] _atomic_replace failed for {dst.name} after retries")
                            SeqGroup._last_replace_fail_log = time.time()
                        try:
                            src.unlink(missing_ok=True)  # Clean up tmp
                        except Exception:
                            pass
            except OSError as e:
                if e.errno == errno.EACCES:
                    # Some Python versions report PermissionError as OSError + EACCES
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
                        continue
                raise

    def merge_to_shared(self):
        """
        Merge factors: shared formal library (history) + each worker's temp library (new)
        -> deduplicate -> write to formal library.

        Safety strategy:
          - Read baseline first, then increment; never overwrite and lose data
          - If baseline is corrupted, keep old file and alert
          - Atomic write (tmp + rename) to prevent file corruption from crashes
        """
        all_factors: list[dict] = []
        baseline_ok = True

        # 1. Baseline: existing shared formal library
        if self.shared_factors_path.exists():
            try:
                with open(self.shared_factors_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_factors.extend(data)
                    # Save a backup on first merge
                    bak = self.shared_factors_path.with_suffix(".bak")
                    if not bak.exists():
                        shutil.copy2(self.shared_factors_path, bak)
                else:
                    if time.time() - self._last_merge_error_time > self.MERGE_ERROR_COOLDOWN:
                        log_temp(f"[WARN] seq={self.seq} shared factors file is not a list, "
                                 f"keeping old file untouched")
                        self._last_merge_error_time = time.time()
                    baseline_ok = False
            except (json.JSONDecodeError, ValueError, OSError) as e:
                if time.time() - self._last_merge_error_time > self.MERGE_ERROR_COOLDOWN:
                    log_temp(f"[ERROR] seq={self.seq} shared factors file corrupted: {e}")
                    log_temp(f"  File preserved at {self.shared_factors_path}, NOT overwriting.")
                    log_temp(f"  Restore from .bak if available: {self.shared_factors_path}.bak")
                    self._last_merge_error_time = time.time()
                baseline_ok = False

        # 2. Increment: each worker's private temp library
        for w in self.workers:
            p = w.memory_dir / "valid_factors.json"
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_factors.extend(data)
                except Exception:
                    pass

        if not baseline_ok:
            return  # Baseline corrupted, abandon this merge, keep old file

        if all_factors:
            seen_names: set[str] = set()
            seen_codes: set[str] = set()
            deduped: list[dict] = []
            for fac in all_factors:
                name = fac.get("name", "")
                code_norm = "".join(fac.get("code", "").split())
                if name in seen_names or code_norm in seen_codes:
                    continue
                seen_names.add(name)
                seen_codes.add(code_norm)
                deduped.append(fac)

            self.shared_factors_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.shared_factors_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(deduped, f, ensure_ascii=False, indent=2)
            self._atomic_replace(tmp_path, self.shared_factors_path)

        # Experience merge (read baseline first)
        all_exp: list[dict] = []
        seen_exp: set[tuple] = set()

        if self.shared_experience_path.exists():
            try:
                with open(self.shared_experience_path, encoding="utf-8") as f:
                    for e in json.load(f):
                        sig = (e.get("round"), e.get("worker_id", ""), e.get("n_hypotheses", 0))
                        if sig not in seen_exp:
                            seen_exp.add(sig)
                            all_exp.append(e)
            except Exception:
                pass

        # Increment: each worker's private experience
        for w in self.workers:
            p = w.memory_dir / "experience.json"
            if p.exists():
                try:
                    with open(p, encoding="utf-8") as f:
                        for e in json.load(f):
                            sig = (e.get("round"), e.get("worker_id", ""), e.get("n_hypotheses", 0))
                            if sig not in seen_exp:
                                seen_exp.add(sig)
                                all_exp.append(e)
                except Exception:
                    pass

        if all_exp:
            all_exp.sort(key=lambda e: (e.get("round", 0), e.get("worker_id", "")))
            tmp_path = self.shared_experience_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(all_exp, f, ensure_ascii=False, indent=2)
            self._atomic_replace(tmp_path, self.shared_experience_path)


# ---------------------------------------------------------------------------
# MergeDaemon
# ---------------------------------------------------------------------------
class MergeDaemon(threading.Thread):
    def __init__(self, get_groups, interval: float = 120, global_merge_callback=None):
        super().__init__(daemon=True)
        self._get_groups = get_groups
        self.interval = interval
        self._stop = False
        self._global_merge = global_merge_callback

    def run(self):
        while not self._stop:
            time.sleep(self.interval)
            if self._stop:
                break
            for g in self._get_groups():
                if g.running_count > 0:
                    try:
                        g.merge_to_shared()
                    except Exception:
                        pass
            # Periodically merge each group's factors into root memory/valid_factors.json
            if self._global_merge:
                try:
                    self._global_merge(verbose=False)
                except Exception:
                    pass

    def stop(self):
        self._stop = True


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class Dashboard:
    """Terminal dashboard: fixed position, continuously refreshes system status and mining progress."""

    def __init__(self, get_groups, get_stats, get_start_time, get_max_workers):
        self._get_groups = get_groups
        self._get_stats = get_stats
        self._get_start = get_start_time
        self._get_max_workers = get_max_workers

    def render(self):
        """Clear screen and draw the full dashboard."""
        groups = self._get_groups()
        stats = self._get_stats()
        elapsed = time.time() - self._get_start()

        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')

        lines = []
        # -- Title bar --
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append("=" * 78)
        lines.append(f"  BATCH FACTOR MINING  |  {now}  |  Uptime: {timedelta(seconds=int(elapsed))}")
        lines.append("=" * 78)

        # -- System resources --
        cpu = stats["cpu_percent"]
        mem = stats["memory_percent"]
        cpu_bar = progress_bar(cpu)
        mem_bar = progress_bar(mem)
        lines.append(
            f"  System  CPU {cpu_bar} {cpu:5.1f}%  |  "
            f"MEM {mem_bar} {mem:5.1f}%  |  "
            f"{stats['memory_available_gb']:.1f}G / {stats['memory_total_gb']:.1f}G available"
        )

        # -- Separator --
        lines.append("-" * 78)

        # -- Per-group details --
        total_factors = 0
        total_running = 0
        for g in groups:
            gf = g.shared_factor_count
            total_factors += gf
            total_running += g.running_count

            # Factors mined by all workers in this group during this run
            mined_this_run = sum(w.factor_count for w in g.workers)

            saturated = " SATURATED" if g._zero_output_streak >= 2 else ""
            lines.append(
                f"  seq={g.seq:>2}  |  target={g.target:>4}  |  "
                f"total={gf:>4}/{g.target:<4}  |  "
                f"this_run=+{mined_this_run:>4}  |  "
                f"need={g.remaining:>4}  |  "
                f"running={g.running_count}{saturated}"
            )

            # Active workers (LOADING / MINING) + failed
            active_workers = [w for w in g.workers if w.state in ("LOADING", "MINING", "FAILED")]
            done_workers = [w for w in g.workers if w.state == "DONE"]

            for w in active_workers:
                w.update_stats()
                delta = w.factor_count - w.display_factor_count
                if delta > 0:
                    d_str = f"+{delta}"
                elif w.state in ("LOADING", "MINING") and w.factor_count > 0:
                    d_str = "  ="
                else:
                    d_str = "  ."
                w.display_factor_count = w.factor_count

                fresh = ""
                if w.last_factor_time > 0:
                    secs = time.time() - w.last_factor_time
                    if secs < 120:
                        fresh = f" *{secs:.0f}s"
                    elif secs < 600:
                        fresh = f" *{secs/60:.0f}m"

                state_tag = w.state
                if w.state == "MINING" and delta > 0:
                    state_tag = "MINING+"

                avg_auc = w.auc_sum / w.auc_count if w.auc_count > 0 else 0.0
                lines.append(
                    f"    {w.worker_id:>3} {state_tag:<8} "
                    f"mine={w.factor_count:>4} ({d_str:>3})  "
                    f"rnd={w.current_round:>4}  "
                    f"avg_auc={avg_auc:.3f}  "
                    f"{w.elapsed_str():>6}{fresh}"
                )
                if w.state == "FAILED":
                    tail = w.get_log_tail(3)
                    if tail:
                        for tline in tail.split('\n')[:3]:
                            lines.append(f"         {tline}")

            # Collapse completed workers into one line
            if done_workers:
                done_total = sum(w.factor_count for w in done_workers)
                lines.append(
                    f"    [{len(done_workers)} done, total +{done_total}]"
                )

        # -- Summary --
        lines.append("-" * 78)
        lines.append(
            f"  OVERALL: {total_factors} factors  |  {total_running}/{self._get_max_workers()} workers  |  "
            f"{len(groups)} groups"
        )

        # -- Action hints --
        lines.append("=" * 78)
        lines.append("  Ctrl+C or touch batch_stop.txt to stop  |  Auto-scaling active")
        lines.append("=" * 78)

        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# BatchMiner -- fully automatic scheduler
# ---------------------------------------------------------------------------
class BatchMiner:
    def __init__(
        self,
        tasks_spec: list[tuple[int, int]],
        mem_limit: float = 60.0,
        cpu_limit: float = 80.0,
        stagger_offset: float = 300,
        check_interval: int = 30,
        display_interval: int = 30,
        merge_interval: int = 120,
        max_total_workers: int = 0,       # 0=auto (cpu_count), CLI default=8
        merge_on_done: bool = True,
    ):
        self.mem_limit = mem_limit
        self.cpu_limit = cpu_limit
        self.stagger_offset = stagger_offset
        self.check_interval = check_interval
        self.display_interval = display_interval
        self.max_total_workers = max_total_workers or psutil.cpu_count(logical=False) or 4
        self.merge_on_done = merge_on_done
        self._last_display: float = 0

        # Build SeqGroup
        self.groups: list[SeqGroup] = []
        for seq, target in tasks_spec:
            common_cfg, discovery_cfg = find_configs(seq)
            if common_cfg is None:
                log_temp(f"[WARN] seq={seq} config not found, skip")
                continue
            existing = count_factors_shared(seq)
            if existing >= target > 0:
                log_temp(f"[SKIP] seq={seq} already has {existing}>={target}")
                continue
            self.groups.append(SeqGroup(seq, target, common_cfg, discovery_cfg))

        self.start_time = time.time()
        self.python = sys.executable
        self.env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        self._stop = False
        self._shutting_down = False    # Shutting down flag, avoid duplicate triggering
        self._stop_file = PROJECT_ROOT / "batch_stop.txt"

        # Dashboard
        self.dashboard = Dashboard(
            lambda: self.groups,
            get_system_stats,
            lambda: self.start_time,
            lambda: self.max_total_workers,
        )

        # Merge daemon thread (periodic per-seq + global merge)
        self._merge_daemon = MergeDaemon(
            lambda: self.groups, interval=merge_interval,
            global_merge_callback=self._merge_all_groups if self.merge_on_done else None,
        )

        # Progress file
        self._progress_log = PROJECT_ROOT / "batch_progress.log"

    # ------------------------------------------------------------------
    # Auto-scaling decisions
    # ------------------------------------------------------------------
    def _can_launch(self) -> bool:
        """
        Check if resources allow starting a new worker.

        Memory spikes during data loading are normal (MouseBehaviorDataset builds window matrices),
        so we handle two cases:
          - Workers in LOADING: relax memory limit to 95%, allow loading spikes
          - All in MINING: use available absolute memory (~2GB per worker estimate),
            also check percentage limit
        Never kill running workers due to high memory.
        """
        stats = get_system_stats()
        cpu = stats["cpu_percent"]
        mem = stats["memory_percent"]
        avail_gb = stats["memory_available_gb"]

        # Workers currently loading data -> relax limits
        any_loading = any(g.loading_count > 0 for g in self.groups)
        if any_loading:
            # Just avoid OOM (>98%), relax CPU to 95%
            return cpu < 95.0 and mem < 98.0

        # All mining -> use available memory (reserve 2.5GB per worker)
        estimated_per_worker_gb = 2.5
        current_running = self._total_running()
        if current_running > 0:
            needed_for_new = estimated_per_worker_gb
        else:
            needed_for_new = estimated_per_worker_gb * 1.5  # First worker needs more (cold load)

        # Must satisfy all: CPU below limit, enough available memory, percentage not exceeded
        return (
            cpu < self.cpu_limit
            and avail_gb >= needed_for_new
            and mem < max(self.mem_limit, 85.0)  # Allow relaxation to 85% (Windows page cache can be reclaimed)
        )

    def _select_next_group(self) -> Optional[SeqGroup]:
        """
        Select the seq group most in need of a new worker.

        Rules:
          1. Skip groups that have already reached their target
          2. Skip groups that have reached the worker limit (MAX_WORKERS)
          3. Skip groups with consecutive failures exceeding limit and still in cooldown
          4. Skip groups with a worker currently LOADING (avoid simultaneous loading memory spike)
          5. Stagger: ensure >= stagger_offset since last MINING worker start
          6. Select group with the most remaining factors
        """
        candidates = []
        for g in self.groups:
            if g.remaining <= 0:
                continue
            # Per-group limit (non-FAILED workers total; FAILED can be replaced)
            active_or_done = sum(1 for w in g.workers if w.state != "FAILED")
            if active_or_done >= g.MAX_WORKERS:
                continue
            # Global limit
            if self._total_running() >= self.max_total_workers:
                continue
            # Consecutive failure protection
            if g._consecutive_failures >= g.MAX_CONSECUTIVE_FAILURES:
                if time.time() - g._last_failure_time < 600:
                    continue
                else:
                    g._consecutive_failures = 0
            # Consecutive zero output: two workers found nothing -> seq saturated, no new workers
            if g._zero_output_streak >= 2:
                continue
            # Cannot have a worker currently loading
            if g.loading_count > 0:
                continue
            # Stagger check
            mining_workers = [w for w in g.workers if w.state == "MINING"]
            if mining_workers:
                last_start = max(w.started_at for w in mining_workers)
                if time.time() - last_start < self.stagger_offset:
                    continue
            elif g.running_count > 0:
                continue
            candidates.append(g)

        if not candidates:
            return None
        return max(candidates, key=lambda g: g.remaining)

    def _total_running(self) -> int:
        return sum(g.running_count for g in self.groups)

    # ------------------------------------------------------------------
    # Worker maintenance
    # ------------------------------------------------------------------
    def _maintain_workers(self):
        """Check all worker states, clean up dead processes, track consecutive failures."""
        for g in self.groups:
            for w in g.workers:
                if w.state not in ("LOADING", "MINING"):
                    continue

                ret = w.poll()

                # Normal exit
                if ret == 0:
                    w.update_stats()
                    w.refresh_factor_count()
                    if w.state != "DONE":
                        w.state = "DONE"
                    g._consecutive_failures = 0
                    # Track zero output: if the worker found zero factors, increment streak
                    if w.factor_count == 0:
                        g._zero_output_streak += 1
                    else:
                        g._zero_output_streak = 0
                    # Clean up worker temp directory to free disk space (reduce OS page cache pressure)
                    _cleanup_worker_dir(w)
                    continue

                # Abnormal exit
                if ret is not None and ret != 0:
                    w.state = "FAILED"
                    g._consecutive_failures += 1
                    g._last_failure_time = time.time()
                    continue

                # Still running: update stats, check for failures
                w.update_stats()
                if w.check_failure():
                    w.terminate()
                    w.state = "FAILED"
                    g._consecutive_failures += 1
                    g._last_failure_time = time.time()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        if not self.groups:
            log_temp("No tasks to run.")
            return

        self._merge_daemon.start()

        # -- Cold start: launch first worker, wait for data loading --
        self._cold_start()

        # -- Main scheduling loop --
        while not self._stop:
            # 1. Maintain existing workers (check alive, failed, log errors)
            self._maintain_workers()

            # 2. Check if each group has reached its target -> terminate extra workers
            for g in self.groups:
                if g.remaining <= 0:
                    for w in g.workers:
                        if w.state in ("LOADING", "MINING"):
                            w.terminate()
                            w.state = "DONE"

            # 3. Auto-start new workers (if resources allow)
            next_group = self._select_next_group()
            if next_group is not None and self._can_launch():
                w = next_group.create_worker()
                w.launch(self.python, self.env)

            # 4. Merge temp library -> formal library
            for g in self.groups:
                if g.running_count > 0:
                    g.merge_to_shared()
            gc.collect()  # Release temporary objects created by JSON loading after merge

            # 5. Refresh factor counts
            for g in self.groups:
                for w in g.workers:
                    if w.state in ("LOADING", "MINING", "DONE"):
                        w.refresh_factor_count()

            # 6. Dashboard
            self._maybe_render()

            # 7. Progress file
            self._write_progress()

            # 8. Check exit triggers (stop file)
            if self._check_stop_triggers():
                self._graceful_shutdown("stop file detected")
                return

            # 9. Termination condition
            all_targets_met = all(g.remaining <= 0 for g in self.groups)
            all_procs_dead = all(
                w.state in ("DONE", "FAILED") or w.poll() is not None
                for g in self.groups for w in g.workers
            )
            if all_targets_met and all_procs_dead:
                break

            self._interruptible_sleep(self.check_interval)

        # -- Normal shutdown --
        self._graceful_shutdown("all targets reached")

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    def _graceful_shutdown(self, reason: str = "user interrupt"):
        """Gradually terminate all workers, merge, save, with progress printing throughout."""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._stop = True
        self._merge_daemon.stop()

        print(f"\n{'='*60}", flush=True)
        print(f"  SHUTTING DOWN ({reason})", flush=True)
        print(f"{'='*60}", flush=True)

        # 1. Terminate all child processes
        all_workers = [w for g in self.groups for w in g.workers
                       if w.proc and w.proc.poll() is None]
        print(f"  Terminating {len(all_workers)} worker(s)...", flush=True)
        for w in all_workers:
            print(f"    [{w.label}] terminating (PID={w.proc.pid})...", flush=True)
            w.terminate()
        print(f"  All workers terminated.", flush=True)

        # 2. Final merge
        print(f"  Final merge...", flush=True)
        for g in self.groups:
            g.merge_to_shared()
            gf = g.shared_factor_count
            print(f"    seq={g.seq}: {gf}/{g.target} factors", flush=True)

        # 3. Summary
        total = sum(g.shared_factor_count for g in self.groups)
        print(f"  Total: {total} factors", flush=True)

        # 4. Optional full merge
        if self.merge_on_done:
            print(f"  Merging all groups -> memory/valid_factors.json ...", flush=True)
            self._merge_all_groups()

        # 5. Progress archive
        self._write_progress()
        print(f"  Progress saved to {self._progress_log}", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Shutdown complete.", flush=True)

    def _check_stop_triggers(self) -> bool:
        """Check for exit triggers (Ctrl+C is handled by signal handler, not here)."""
        if self._stop_file.exists():
            try:
                self._stop_file.unlink()
            except Exception:
                pass
            return True
        return False

    def _maybe_render(self):
        """Throttle dashboard: only refresh if display_interval has passed since last render."""
        if time.time() - self._last_display >= self.display_interval:
            self.dashboard.render()
            self._last_display = time.time()

    def _interruptible_sleep(self, seconds: float, render_dashboard: bool = False):
        """Sleep in segments, allowing quick response to _stop. Optionally periodically refresh dashboard (by display_interval)."""
        chunk = 1.0
        elapsed = 0.0
        while elapsed < seconds and not self._stop:
            time.sleep(min(chunk, seconds - elapsed))
            elapsed += chunk
            if render_dashboard:
                self._maybe_render()

    def _cold_start(self):
        """
        Cold start: launch first worker for each group one by one, wait for data loading.
        If a worker fails, auto-retry (max 3 times) with 30s interval.
        Memory limit is relaxed during loading (data loading memory spikes are normal).
        Dashboard renders throughout, avoiding the "nothing printed" experience.
        """
        MAX_RETRIES = 3
        retry_delay = 30

        for g in self.groups:
            if g.remaining <= 0:
                continue

            for attempt in range(1, MAX_RETRIES + 1):
                w = g.create_worker()
                w.launch(self.python, self.env)
                g.merge_to_shared()
                self._maybe_render()

                # Wait for data loading to complete
                _t0 = time.time()
                _timeout = 900
                while w.state == "LOADING" and time.time() - _t0 < _timeout and not self._stop:
                    self._interruptible_sleep(self.check_interval, render_dashboard=True)
                    w.update_stats()
                    w.refresh_factor_count()
                    g.merge_to_shared()
                    ret = w.poll()
                    if ret is not None:
                        if ret != 0:
                            w.state = "FAILED"
                            g._consecutive_failures += 1
                            g._last_failure_time = time.time()
                            tail = w.get_log_tail(8)
                            log_temp(
                                f"[{w.label}] FAILED on attempt {attempt}/{MAX_RETRIES} "
                                f"(exit={ret})\n    Log tail:\n{tail}"
                            )
                        else:
                            w.update_stats()
                            if w.state == "LOADING":
                                w.state = "DONE"
                        break

                if w.state == "FAILED":
                    if attempt < MAX_RETRIES:
                        log_temp(
                            f"[seq={g.seq}] retrying in {retry_delay}s "
                            f"(attempt {attempt}/{MAX_RETRIES})"
                        )
                        self._interruptible_sleep(retry_delay, render_dashboard=True)
                        if self._stop:
                            break
                        continue
                    else:
                        log_temp(f"[seq={g.seq}] all {MAX_RETRIES} attempts failed, skip for now")
                        break

                # Success
                if w.state == "LOADING":
                    w.state = "MINING"
                g._consecutive_failures = 0
                break

            # Wait for resource recovery after each group start
            g.merge_to_shared()
            if g.remaining > 0:
                self._interruptible_sleep(self.check_interval, render_dashboard=True)

    def _write_progress(self):
        lines = [f"=== {datetime.now().isoformat()} ==="]
        stats = get_system_stats()
        lines.append(
            f"CPU={stats['cpu_percent']:.1f}% MEM={stats['memory_percent']:.1f}% "
            f"({stats['memory_available_gb']:.1f}G avail)"
        )
        for g in self.groups:
            gf = g.shared_factor_count
            lines.append(
                f"seq={g.seq:>2} target={g.target} factors={gf}/{g.target} "
                f"remaining={g.remaining} workers={g.running_count}"
            )
            for w in g.workers:
                if w.state == "PENDING":
                    continue
                lines.append(
                    f"  {w.worker_id} {w.state} factors={w.factor_count} "
                    f"round={w.current_round} avg_auc={w.auc_sum/w.auc_count if w.auc_count else 0:.3f} "
                    f"elapsed={w.elapsed_str()}"
                )
        lines.append("")
        try:
            with open(self._progress_log, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def _merge_all_groups(self, verbose: bool = True):
        """Scan all memory/seq*/ directories and merge into memory/valid_factors.json."""
        merge_all_seq_dirs(verbose=verbose)

    def stop(self, reason: str = "signal"):
        """External signal triggers exit (Ctrl+C / SIGTERM)."""
        if not self._shutting_down:
            self._graceful_shutdown(reason)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_tasks_spec(spec: str) -> list[tuple[int, int]]:
    pairs = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            s, t = part.split(":", 1)
            pairs.append((int(s), int(t)))
        else:
            pairs.append((int(part), 0))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Batch factor mining -- fully automatic resource-aware parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_mining.py --seq 1,5,15 --target 600 --mem-limit 60
  python batch_mining.py --tasks 1:600,5:300 --mem-limit 60 --cpu-limit 80
  python batch_mining.py --seq 1,5 --target 600 --mem-limit 60 --no-merge
        """,
    )
    parser.add_argument("--seq", type=str, default="1,5,15,30,60",
                        help="seq list, comma-separated (e.g. 1,5,15)")
    parser.add_argument("--target", type=int, default=600,
                        help="target factors per seq")
    parser.add_argument("--tasks", type=str, default=None,
                        help="per-seq targets: '1:600,5:300,15:200'")

    parser.add_argument("--mem-limit", type=float, default=60.0,
                        help="memory usage %% limit, won't launch new worker above this (default: 60)")
    parser.add_argument("--cpu-limit", type=float, default=80.0,
                        help="CPU usage %% limit (default: 80)")
    parser.add_argument("--max-total-workers", type=int, default=8,
                        help="max total workers across all seqs (default: 8)")

    parser.add_argument("--stagger", type=float, default=120,
                        help="min seconds between same-seq worker launches (default: 120)")
    parser.add_argument("--check-interval", type=int, default=10,
                        help="main loop interval: scaling/merge/health checks (default: 10)")
    parser.add_argument("--display-interval", type=int, default=60,
                        help="dashboard refresh interval in seconds (default: 60)")
    parser.add_argument("--merge-interval", type=int, default=120,
                        help="factor merge daemon interval in seconds (default: 120)")

    parser.add_argument("--merge", action="store_true", default=True,
                        help="merge all into memory/valid_factors.json periodically and on shutdown (default: enabled)")
    parser.add_argument("--no-merge", action="store_false", dest="merge",
                        help="disable auto-merge to memory/valid_factors.json")
    parser.add_argument("--merge-only", action="store_true",
                        help="scan all memory/seq*/ and rebuild memory/valid_factors.json, then exit")
    parser.add_argument("--clean", action="store_true",default=True,
                        help="clean all worker dirs before starting (fresh counts)")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.tasks:
        tasks_spec = parse_tasks_spec(args.tasks)
    elif args.seq:
        seqs = [int(s.strip()) for s in args.seq.split(",")]
        tasks_spec = [(s, args.target) for s in seqs]
    else:
        log_temp("No --seq or --tasks, using default: seq=1,5,15,30,60 target=600")
        tasks_spec = [(s, 600) for s in [1, 5, 15, 30, 60]]

    if args.clean:
        log_temp("Cleaning all worker directories...")
        for seq, _ in tasks_spec:
            wdir = MEMORY_DIR / f"seq{seq}" / "workers"
            if wdir.exists():
                shutil.rmtree(wdir, ignore_errors=True)
                log_temp(f"  removed {wdir}")
        log_temp("Clean done.")

    if args.merge_only:
        merge_all_seq_dirs()
        return

    if args.dry_run:
        log_temp("=== DRY RUN ===")
        for seq, target in tasks_spec:
            common, disc = find_configs(seq)
            exists = "OK" if common else "MISSING"
            count = count_factors_shared(seq)
            log_temp(f"  seq={seq:>2}  target={target:>4}  existing={count:>4}  config={exists}")
        return

    miner = BatchMiner(
        tasks_spec=tasks_spec,
        mem_limit=args.mem_limit,
        cpu_limit=args.cpu_limit,
        stagger_offset=args.stagger,
        check_interval=args.check_interval,
        display_interval=args.display_interval,
        merge_interval=args.merge_interval,
        max_total_workers=args.max_total_workers,
        merge_on_done=args.merge,
    )

    # Graceful exit: Ctrl+C or SIGTERM
    # First trigger -> graceful shutdown; second trigger -> immediate force exit
    _exit_forced = False
    def _sig_handler(signum, frame):
        nonlocal _exit_forced
        if _exit_forced:
            print("\nForce quit!", flush=True)
            os._exit(1)
        _exit_forced = True
        miner.stop("Ctrl+C" if signum == signal.SIGINT else "SIGTERM")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    miner.run()


if __name__ == "__main__":
    main()
