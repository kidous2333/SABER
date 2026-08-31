"""
SimpleWorker — runs computation in a QThread (same process).
Signals work natively without multiprocessing queue issues.
"""

from PySide6.QtCore import QObject, Signal, Slot


class BaseWorker(QObject):
    """
    Base worker running in QThread (same process, no multiprocessing).

    Subclass and override run(). Check self._cancelled for graceful stop.
    """

    log_line = Signal(object, object)
    progress = Signal(object, object)
    partial_result = Signal(object)
    result_ready = Signal(object)
    error = Signal(object)
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    @Slot()
    def cancel(self):
        self._cancelled = True

    def set_progress(self, pct, status=""):
        self.progress.emit(pct, status)

    def log(self, msg, level=20):
        self.log_line.emit(msg, level)

    @Slot()
    def run(self):
        """Override in subclasses."""
        self.finished.emit()
