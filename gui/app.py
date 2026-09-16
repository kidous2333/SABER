"""
GUI application entry point.

Run:
    python -m gui.app
    python gui/app.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so all imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon

from gui.main_window import MainWindow


def setup_application() -> QApplication:
    """Configure and return the QApplication instance."""
    # On Windows, give the process its own AppUserModelID so the taskbar shows
    # the application icon instead of the python.exe icon. Must be called
    # BEFORE the QApplication is created.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "SABER.GUI"
            )
        except Exception:
            pass

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setOrganizationName("SABER")
    app.setApplicationName("SABER")

    # Set app icon
    icon_path = PROJECT_ROOT / "gui" / "icon.png"
    if icon_path.exists():
        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)

    # Set default font
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    return app


def main():
    app = setup_application()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
