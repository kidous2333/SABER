"""SSHTab — SSH remote connection management and file sync."""
import logging
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox
from PySide6.QtCore import Qt
from gui.tabs.base_tab import BaseTab
from gui.widgets.parameter_group import ParameterGroup
from gui.ssh.ssh_manager import SSHManager

logger = logging.getLogger("gui.ssh_tab")


class SSHTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(title="SSH Remote Connection", tab_key="ssh", parent=parent)
        self._ssh_manager = SSHManager()
        self._main_window = None
        self._ssh_manager.on_connection_changed = self._on_connection_changed
        self._ssh_manager.on_remote_log = self._log_viewer.append_colored

    def set_main_window(self, window):
        self._main_window = window

    def get_ssh_manager(self):
        return self._ssh_manager

    def setup_params(self):
        self.add_config_group(ParameterGroup("Connection", [
            {"key": "host", "label": "Host", "type": "text", "default": "example.com"},
            {"key": "port", "label": "Port", "type": "int", "default": 22, "min": 1, "max": 65535},
            {"key": "username", "label": "Username", "type": "text", "default": ""},
            {"key": "password", "label": "Password", "type": "password", "default": ""},
            {"key": "key_file", "label": "Key File", "type": "file", "default": ""},
        ]))
        self.add_config_group(ParameterGroup("Remote", [
            {"key": "remote_python", "label": "Python", "type": "text", "default": "/usr/bin/python3"},
            {"key": "remote_workdir", "label": "Work Dir", "type": "text", "default": "~/SABER"},
        ]))
        # Buttons below config
        btn_row = QHBoxLayout()
        self._connect_btn = QPushButton("Connect"); self._connect_btn.clicked.connect(self._on_connect)
        self._connect_btn.setStyleSheet("QPushButton{background:#16A34A;color:#FFF;border:none;border-radius:6px;padding:8px 20px;font-weight:600;font-size:13px;}QPushButton:hover{background:#15803D;}")
        btn_row.addWidget(self._connect_btn)
        self._disconnect_btn = QPushButton("Disconnect"); self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._disconnect_btn.setEnabled(False)
        self._disconnect_btn.setStyleSheet("QPushButton{background:#DC2626;color:#FFF;border:none;border-radius:6px;padding:8px 20px;font-weight:600;font-size:13px;}QPushButton:hover{background:#B91C1C;}QPushButton:disabled{background:#D1D5DB;color:#9CA3AF;}")
        btn_row.addWidget(self._disconnect_btn)
        self._test_btn = QPushButton("Test"); self._test_btn.clicked.connect(self._on_test); self._test_btn.setEnabled(False)
        btn_row.addWidget(self._test_btn)
        self._param_layout.insertLayout(self._param_layout.count() - 1, btn_row)

    def setup_results(self):
        self._status_label = QLabel("Status: Disconnected")
        self._status_label.setStyleSheet("color:#999;font-size:16px;font-weight:600;padding:20px;")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._results_layout_main.addWidget(self._status_label)
        self._sync_group = ParameterGroup("File Sync", [
            {"key": "local_dir", "label": "Local Dir", "type": "dir", "default": ""},
            {"key": "remote_dir", "label": "Remote Dir", "type": "text", "default": "~/SABER"},
        ])
        self._results_layout_main.addWidget(self._sync_group)
        sync_btns = QHBoxLayout()
        self._sync_up_btn = QPushButton("Upload to Remote"); self._sync_up_btn.clicked.connect(self._on_sync_up); self._sync_up_btn.setEnabled(False)
        sync_btns.addWidget(self._sync_up_btn)
        self._sync_down_btn = QPushButton("Download Results"); self._sync_down_btn.clicked.connect(self._on_sync_down); self._sync_down_btn.setEnabled(False)
        sync_btns.addWidget(self._sync_down_btn)
        self._results_layout_main.addLayout(sync_btns)
        self._results_layout_main.addStretch()

    def _on_connect(self):
        p = self._config_groups[0].get_values()
        host, port = p.get("host",""), int(p.get("port",22) or 22)
        user, pw, key = p.get("username",""), p.get("password",""), p.get("key_file","")
        if not host or not user: QMessageBox.warning(self, "Missing", "Host and Username required."); return
        r = self._config_groups[1].get_values()
        self._ssh_manager.set_remote_python(r.get("remote_python","/usr/bin/python3"))
        self._ssh_manager.set_remote_workdir(r.get("remote_workdir","~/SABER"))
        self._log_viewer.append_colored(f"Connecting to {user}@{host}:{port}...", 20)
        if not self._ssh_manager.connect(host, port, user, pw, key):
            QMessageBox.critical(self, "Failed", f"Could not connect to {host}.")

    def _on_disconnect(self): self._ssh_manager.disconnect()

    def _on_test(self):
        ok, msg = self._ssh_manager.test_connection()
        (QMessageBox.information if ok else QMessageBox.warning)(self, "Test", f"{'OK' if ok else 'Failed'}\n{msg}")

    def _on_connection_changed(self, connected, identity=""):
        if connected:
            self._status_label.setText(f"Connected to {identity}")
            self._status_label.setStyleSheet("color:#16A34A;font-size:16px;font-weight:600;padding:20px;")
            self._connect_btn.setEnabled(False); self._disconnect_btn.setEnabled(True)
            self._test_btn.setEnabled(True); self._sync_up_btn.setEnabled(True); self._sync_down_btn.setEnabled(True)
        else:
            self._status_label.setText("Disconnected")
            self._status_label.setStyleSheet("color:#999;font-size:16px;font-weight:600;padding:20px;")
            self._connect_btn.setEnabled(True); self._disconnect_btn.setEnabled(False)
            self._test_btn.setEnabled(False); self._sync_up_btn.setEnabled(False); self._sync_down_btn.setEnabled(False)
        if self._main_window and hasattr(self._main_window, '_status_bar'):
            self._main_window._status_bar.set_ssh_status(connected, identity)
        if self._main_window and hasattr(self._main_window, '_tabs'):
            mgr = self._ssh_manager if connected else None
            for tab in self._main_window._tabs.values():
                if hasattr(tab, 'set_ssh_manager'): tab.set_ssh_manager(mgr)

    def _on_sync_up(self):
        if self._sync_group.get_value("local_dir"):
            self._ssh_manager.sync_to_remote(self._sync_group.get_value("local_dir"), self._sync_group.get_value("remote_dir"))

    def _on_sync_down(self):
        if self._sync_group.get_value("remote_dir"):
            self._ssh_manager.sync_from_remote(self._sync_group.get_value("remote_dir"), self._sync_group.get_value("local_dir") or "remote_results")
