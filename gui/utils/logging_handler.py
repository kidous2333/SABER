"""
QtLogHandler — bridge Python logging to Qt signals.

Thread-safe: signals automatically cross thread boundaries via Qt's
queued connection mechanism.
"""

import logging
from PySide6.QtCore import Signal, QObject


class _SignalEmitter(QObject):
    """Internal QObject that owns the signal for cross-thread delivery."""
    log_signal = Signal(object, object)  # message, levelno


class QtLogHandler(logging.Handler):
    """
    Custom logging.Handler that forwards log records to a Qt signal.

    Usage:
        handler = QtLogHandler()
        handler.log_signal.connect(on_log)
        logging.getLogger().addHandler(handler)

    The signal delivers (formatted_message, levelno) tuples.
    """

    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__()
        self._emitter = _SignalEmitter()
        self.setFormatter(logging.Formatter(
            fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt or "%H:%M:%S",
        ))

    @property
    def log_signal(self) -> Signal:
        return self._emitter.log_signal

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self._emitter.log_signal.emit(msg, record.levelno)
        except Exception:
            self.handleError(record)


class ModuleLogRedirector:
    """
    Redirect a module's logger output through a QtLogHandler.

    Usage:
        redirector = ModuleLogRedirector("main")
        redirector.log_signal.connect(my_log_viewer.append)
        redirector.install()
        ...
        redirector.uninstall()
    """

    def __init__(self, logger_name: str = None, fmt: str = None):
        self._logger_name = logger_name
        self._handler = QtLogHandler(fmt=fmt)

    @property
    def log_signal(self) -> Signal:
        return self._handler.log_signal

    @property
    def handler(self) -> QtLogHandler:
        return self._handler

    def install(self):
        target = logging.getLogger(self._logger_name) if self._logger_name else logging.root
        self._prev_level = target.level
        target.setLevel(logging.DEBUG)  # let all messages through to the handler
        target.addHandler(self._handler)

    def uninstall(self):
        target = logging.getLogger(self._logger_name) if self._logger_name else logging.root
        target.removeHandler(self._handler)
        if hasattr(self, '_prev_level'):
            target.setLevel(self._prev_level)
