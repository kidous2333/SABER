"""
RemoteWorker — Runs a Python module on remote server via SSH.

Connects to SSHManager for command execution and log streaming.
"""

import json
import tempfile
import yaml
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

logger_imported = False
try:
    import logging
    logging.getLogger("gui.remote_worker")
    logger_imported = True
except Exception:
    pass


class RemoteWorker(QObject):
    """
    Executes a local module on a remote server via SSH.

    Workflow:
      1. Sync config files to remote
      2. Generate launcher script with parameters
      3. Upload launcher script
      4. Execute remotely via SSHManager.exec_python_script()
      5. On completion, sync results back to local

    Signals:
        finished()         — complete (success or failure)
        result_ready(dict) — final result
        error(str)         — error message
    """

    finished = Signal()
    result_ready = Signal(object)
    error = Signal(object)

    def __init__(self, ssh_manager, module_name: str, params: dict,
                 config_files: list, parent=None):
        """
        Args:
            ssh_manager: SSHManager instance
            module_name: Python module to run (e.g., "train_behavior", "main")
            params: Module parameters dict
            config_files: List of local config file paths to sync
        """
        super().__init__(parent)
        self._ssh = ssh_manager
        self._module = module_name
        self._params = params
        self._config_files = config_files

    @Slot()
    def run(self):
        try:
            if not self._ssh or not self._ssh.is_connected:
                self.error.emit("SSH not connected")
                self.finished.emit()
                return

            # Step 1: Sync config files
            self._ssh.remote_log.emit(f"[Remote] Syncing config files...", 20)
            for cf in self._config_files:
                cf_path = Path(cf)
                if cf_path.exists():
                    remote_path = f"config/{cf_path.name}"
                    ok = self._ssh.sync_to_remote(str(cf_path), remote_path)
                    if not ok:
                        self.error.emit(f"Failed to sync {cf_path.name}")
                        self.finished.emit()
                        return

            # Step 2: Sync source code
            self._ssh.remote_log.emit(f"[Remote] Syncing source code...", 20)
            for src_dir in ["src", "config"]:
                local_path = Path(src_dir)
                if local_path.is_dir():
                    ok = self._ssh.sync_to_remote(str(local_path), src_dir)
                    if not ok:
                        self._ssh.remote_log.emit(f"Warning: sync failed for {src_dir}", 30)

            # Step 3: Prepare launcher command
            # Build CLI arguments from params
            args = []
            if self._module == "main":
                args.extend(["--config-common", "config/seq/1.yaml",
                            "--config-discovery", "config/seq/1.yaml"])
            elif self._module == "train_behavior":
                args.extend(["--config-common", "config/seq/1.yaml",
                            "--config-validation", "config/validation.yaml"])
            elif self._module == "factor_evolution":
                args.append("--validate")  # Default to validation mode on server

            # Step 4: Execute on remote
            self._ssh.remote_log.emit(f"[Remote] Launching {self._module}...", 20)
            module_to_script = {"main": "mining/discovery.py",
                              "train_behavior": "training/train_behavior.py",
                              "factor_evolution": "factors/evolution.py"}
            script_path = module_to_script.get(self._module, f"{self._module}.py")
            exit_code = self._ssh.exec_python_script(script_path, args=args)

            # Step 5: Sync results back
            if exit_code == 0:
                self._ssh.remote_log.emit(f"[Remote] Syncing results back...", 20)
                self._ssh.sync_from_remote("memory/", "memory_remote/")
                self._ssh.sync_from_remote("runs/", "runs_remote/")

            result = {
                "success": exit_code == 0,
                "module": self._module,
                "exit_code": exit_code,
                "message": f"Remote execution complete (exit {exit_code}).",
            }
            self.result_ready.emit(result)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()
