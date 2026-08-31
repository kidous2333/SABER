"""
SSHManager — paramiko-based SSH connection, remote execution, and SFTP sync.

Uses plain Python callbacks instead of Qt Signals to avoid PySide6 type-matching
issues on older Python versions (3.9).
"""

import os
import time
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("gui.ssh_manager")


class SSHManager:
    """
    Manages SSH connection and remote operations.

    Callbacks (set by SSHTab):
        on_connection_changed(connected: bool, identity: str)
        on_remote_log(message: str, level: int)
        on_remote_progress(pct: int, status: str)
        on_remote_finished(exit_code: int)
        on_remote_error(message: str)
    """

    def __init__(self):
        self._client = None
        self._sftp = None
        self._host = ""
        self._port = 22
        self._user = ""
        self._remote_python = "/usr/bin/python3"
        self._remote_workdir = "~/SABER"
        self._channel = None

        # Plain callbacks
        self.on_connection_changed: Optional[Callable] = None
        self.on_remote_log: Optional[Callable] = None
        self.on_remote_progress: Optional[Callable] = None
        self.on_remote_finished: Optional[Callable] = None
        self.on_remote_error: Optional[Callable] = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    @property
    def host(self) -> str:
        return self._host

    def _emit_log(self, msg: str, level: int = 20):
        if self.on_remote_log:
            self.on_remote_log(msg, level)

    def _emit_connection(self, connected: bool, identity: str = ""):
        if self.on_connection_changed:
            self.on_connection_changed(connected, identity)

    def _emit_error(self, msg: str):
        if self.on_remote_error:
            self.on_remote_error(msg)

    def connect(self, host: str, port: int, username: str,
                password: str = "", key_file: str = "") -> bool:
        try:
            import paramiko

            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs = {
                "hostname": host,
                "port": port,
                "username": username,
                "timeout": 15,
            }

            if key_file:
                key_path = Path(key_file).expanduser()
                if key_path.exists():
                    try:
                        pkey = paramiko.RSAKey.from_private_key_file(str(key_path))
                        connect_kwargs["pkey"] = pkey
                    except paramiko.ssh_exception.PasswordRequiredException:
                        connect_kwargs["key_filename"] = str(key_path)
                        if password:
                            connect_kwargs["password"] = password
                    except Exception:
                        connect_kwargs["key_filename"] = str(key_path)
                        if password:
                            connect_kwargs["password"] = password
                else:
                    connect_kwargs["key_filename"] = str(key_path)
                    if password:
                        connect_kwargs["password"] = password
            elif password:
                connect_kwargs["password"] = password

            self._client.connect(**connect_kwargs)
            self._sftp = self._client.open_sftp()

            self._host = host
            self._port = port
            self._user = username

            identity = f"{username}@{host}"
            self._emit_connection(True, identity)
            self._emit_log(f"Connected to {identity}:{port}", 20)

            self._detect_remote_python()
            return True

        except Exception as e:
            self._emit_log(f"Connection failed: {e}", 40)
            self._emit_connection(False, "")
            self._cleanup()
            return False

    def _detect_remote_python(self):
        for py_path in ["/usr/bin/python3", "/usr/bin/python", "/opt/anaconda3/bin/python"]:
            try:
                _, stdout, _ = self._client.exec_command(f"test -x {py_path} && echo OK", timeout=5)
                if "OK" in stdout.read().decode():
                    self._remote_python = py_path
                    self._emit_log(f"Remote Python: {py_path}", 20)
                    return
            except Exception:
                continue
        self._emit_log(f"Warning: could not detect Python", 30)

    def disconnect(self):
        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None
        self._cleanup()
        self._emit_connection(False, "")

    def _cleanup(self):
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def test_connection(self) -> tuple:
        if not self._client:
            return False, "Not connected"
        try:
            _, stdout, stderr = self._client.exec_command("echo 'OK' && hostname", timeout=10)
            out = stdout.read().decode().strip()
            err = stderr.read().decode().strip()
            if err:
                return False, f"Error: {err}"
            lines = out.split("\n")
            if len(lines) >= 2 and lines[0] == "OK":
                return True, f"Host: {lines[1]}"
            _, stdout, stderr = self._client.exec_command(
                f"{self._remote_python} --version 2>&1", timeout=10
            )
            py_ver = stdout.read().decode().strip() or stderr.read().decode().strip()
            return True, f"Python: {py_ver}"
        except Exception as e:
            return False, f"Test failed: {e}"

    def exec_remote(self, command: str, workdir: str = None,
                    env: dict = None, timeout: int = None) -> int:
        if not self._client:
            self._emit_error("Not connected")
            return -1
        try:
            full_cmd = command
            if workdir:
                full_cmd = f"cd {workdir} && {command}"
            if env:
                env_parts = [f"export {k}={v}" for k, v in env.items()]
                full_cmd = " && ".join(env_parts) + " && " + full_cmd

            self._emit_log(f"[Remote] $ {command}", 20)

            transport = self._client.get_transport()
            self._channel = transport.open_session()
            self._channel.exec_command(full_cmd)

            stdout = self._channel.makefile("r", -1)
            stderr = self._channel.makefile_stderr("r", -1)

            while not self._channel.closed or self._channel.recv_ready() or self._channel.recv_stderr_ready():
                line = stdout.readline()
                if line:
                    self._emit_log(line.rstrip(), 20)
                else:
                    time.sleep(0.05)

                err_line = stderr.readline()
                if err_line:
                    self._emit_log(err_line.rstrip(), 30)

                if self._channel.exit_status_ready():
                    break

            exit_code = self._channel.recv_exit_status()
            stdout.close()
            stderr.close()
            self._channel.close()
            self._channel = None

            if self.on_remote_finished:
                self.on_remote_finished(exit_code)
            if exit_code == 0:
                self._emit_log(f"[Remote] Completed (exit {exit_code})", 20)
            else:
                self._emit_log(f"[Remote] Failed (exit {exit_code})", 40)
            return exit_code

        except Exception as e:
            self._emit_error(str(e))
            self._emit_log(f"[Remote] Error: {e}", 40)
            return -1

    def exec_python_script(self, script_path: str, args: list = None,
                           workdir: str = None) -> int:
        if workdir is None:
            workdir = self._remote_workdir
        args_str = " ".join(args) if args else ""
        cmd = f"{self._remote_python} {script_path} {args_str}"
        return self.exec_remote(cmd, workdir=workdir)

    def sync_to_remote(self, local_path: str, remote_path: str) -> bool:
        if not self._sftp:
            self._emit_error("SFTP not available")
            return False
        try:
            local = Path(local_path)
            if local.is_file():
                self._sftp.put(str(local), remote_path)
                self._emit_log(f"Uploaded: {local.name} → {remote_path}", 20)
            elif local.is_dir():
                try:
                    self._sftp.mkdir(remote_path)
                except IOError:
                    pass
                for f in local.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(local)
                        remote_file = f"{remote_path}/{rel}".replace("\\", "/")
                        remote_dir = str(Path(remote_file).parent).replace("\\", "/")
                        try:
                            self._sftp.mkdir(remote_dir)
                        except IOError:
                            pass
                        self._sftp.put(str(f), remote_file)
                self._emit_log(f"Uploaded dir: {local.name} → {remote_path}", 20)
            return True
        except Exception as e:
            self._emit_log(f"Sync failed: {e}", 40)
            return False

    def sync_from_remote(self, remote_path: str, local_path: str) -> bool:
        if not self._sftp:
            self._emit_error("SFTP not available")
            return False
        try:
            local = Path(local_path)
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                attr = self._sftp.stat(remote_path)
                if attr.st_mode & 0o40000:
                    self._sftp_get_recursive(remote_path, str(local))
                else:
                    self._sftp.get(remote_path, str(local))
            except FileNotFoundError:
                self._emit_log(f"Remote path not found: {remote_path}", 40)
                return False
            self._emit_log(f"Downloaded: {remote_path} → {local_path}", 20)
            return True
        except Exception as e:
            self._emit_log(f"Sync failed: {e}", 40)
            return False

    def _sftp_get_recursive(self, remote_dir: str, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        for entry in self._sftp.listdir_attr(remote_dir):
            remote_path = f"{remote_dir}/{entry.filename}"
            local_path = os.path.join(local_dir, entry.filename)
            if entry.st_mode & 0o40000:
                self._sftp_get_recursive(remote_path, local_path)
            else:
                self._sftp.get(remote_path, local_path)

    def set_remote_python(self, path: str):
        self._remote_python = path

    def set_remote_workdir(self, path: str):
        self._remote_workdir = path
