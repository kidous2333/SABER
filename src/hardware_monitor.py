"""
hardware_monitor.py
Background hardware monitor: periodically prints CPU, memory, and GPU usage.
"""

import subprocess
import threading
import logging
import time
import os
import psutil

logger = logging.getLogger(__name__)


def _get_gpu_status():
    """Query GPU status via nvidia-smi, return dict or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=10,
        ).decode().strip()
        lines = out.split("\n")
        result = {}
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                result[parts[0]] = {
                    "name": parts[1],
                    "mem_used_mib": parts[2],
                    "mem_total_mib": parts[3],
                    "gpu_util_pct": parts[4],
                    "temp_c": parts[5],
                    "power_w": parts[6],
                }
        return result
    except Exception:
        return None


def _format_gpu_line(gpu: dict) -> str:
    mem = f"{gpu['mem_used_mib']}/{gpu['mem_total_mib']} MiB"
    return (
        f"GPU{gpu['name']} | Memory {mem} | GPU util {gpu['gpu_util_pct']}% | "
        f"Temp {gpu['temp_c']}C | Power {gpu['power_w']}W"
    )


class HardwareMonitor:
    """Background hardware monitor, prints hardware status to logger at interval_sec."""

    def __init__(self, interval_sec: float = 30):
        self.interval = interval_sec
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"[HardwareMonitor] Started, interval {self.interval}s")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("[HardwareMonitor] Stopped")

    def _run(self):
        while not self._stop.wait(self.interval):
            self._snapshot()

    def _snapshot(self):
        try:
            # CPU
            cpu_pct = psutil.cpu_percent(interval=1)
            # Memory
            mem = psutil.virtual_memory()
            mem_line = (
                f"Mem: {mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f} GiB "
                f"({mem.percent:.0f}%) | Available {mem.available / (1024**3):.1f} GiB"
            )
            # This process
            proc = psutil.Process(os.getpid())
            proc_mem = proc.memory_info().rss / (1024**3)
            proc_cpu = proc.cpu_percent()

            logger.debug(
                f"[HW] CPU: {cpu_pct:.0f}% | "
                f"Process: {proc_cpu:.1f}% / {proc_mem:.2f} GiB | "
                f"{mem_line}"
            )

            # GPU
            gpus = _get_gpu_status()
            if gpus:
                for gpu_id, gpu_info in gpus.items():
                    logger.debug(f"[HW] {_format_gpu_line(gpu_info)}")
            else:
                logger.debug("[HW] GPU: Not detected or nvidia-smi unavailable")

        except Exception:
            logger.debug("[HW] Hardware sampling failed", exc_info=True)
