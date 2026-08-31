"""
MplWidget — Embeddable matplotlib canvas for PySide6.

Supports FigureCanvas + NavigationToolbar for interactive plots.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavToolbar,
)
from matplotlib.figure import Figure


class MplWidget(QWidget):
    """
    Embeddable matplotlib figure canvas with navigation toolbar.

    Usage:
        mpl = MplWidget()
        ax = mpl.figure.add_subplot(111)
        ax.plot([1,2,3], [4,5,6])
        mpl.draw()
    """

    def __init__(self, figsize=(6, 4), dpi=100, toolbar=True, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if toolbar:
            self.toolbar = NavToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None

        layout.addWidget(self.canvas)

    def draw(self):
        """Refresh the canvas."""
        self.figure.tight_layout()
        self.canvas.draw()

    def clear(self):
        """Clear all axes from the figure."""
        self.figure.clear()

    def subplot(self, *args, **kwargs):
        """Shortcut to figure.add_subplot."""
        return self.figure.add_subplot(*args, **kwargs)
