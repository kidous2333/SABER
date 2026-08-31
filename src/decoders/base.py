"""
Abstract base class for SequenceDecoder.

All sequence decoders (Viterbi, CRF, HMM, RL, etc.) implement this interface,
allowing DecoderPipeline to orchestrate any decoder interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class SequenceDecoder(ABC):
    """Abstract base class for sequence decoders.

    Lifecycle:
      1. fit(y_train, n_classes, ...)  — Learn parameters from training data
      2. decode(proba, ...)            — Generate label sequence [T]
      3. predict_proba(proba, ...)     — Generate probability matrix [T, C] (optional override)

    Subclasses save configuration via constructor parameters for pickle serialization (needed for CRF model saving).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier, e.g., 'viterbi', 'crf', 'hmm', 'rl'."""
        ...

    @abstractmethod
    def fit(
        self,
        y_train: np.ndarray,
        n_classes: int,
        proba_train: Optional[np.ndarray] = None,
        feat_train: Optional[np.ndarray] = None,
    ) -> "SequenceDecoder":
        """Learn decoder parameters from training data.

        Args:
            y_train: Training labels [T], dtype=int
            n_classes: Number of behavior classes
            proba_train: Optional probability matrix [T, C], not needed by Viterbi, used by CRF/HMM/RL
            feat_train: Optional feature matrix [T, D], used by CRF LGBM path for temporal features
        """
        ...

    @abstractmethod
    def decode(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Decode into a label sequence.

        Args:
            proba: Probability matrix [T, C]
            feat: Optional feature matrix [T, D]
            **kwargs: Extra arguments (e.g., prev_labels for chained decoding)

        Returns:
            Integer label array [T]
        """
        ...

    def predict_proba(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Return per-frame class probabilities [T, C].

        Default returns one-hot encoding of decode() results.
        Decoders with probabilistic output (CRF marginal probabilities, HMM posteriors) override this method.
        """
        labels = self.decode(proba, feat, **kwargs)
        n_classes = proba.shape[1]
        out = np.zeros((len(labels), n_classes), dtype=np.float32)
        out[np.arange(len(labels)), labels] = 1.0
        return out
