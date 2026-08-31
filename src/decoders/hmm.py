"""
HMM sequence decoder (based on hmmlearn).

Uses GaussianHMM to model the temporal dynamics of probability sequences:
  - Transition matrix initialized from training labels (reuses _fit_transitions_asymmetric)
  - Emission parameters initialized from per-class proba statistics (supervised_init=True)
  - Optional Baum-Welch unsupervised refinement (refine_em=False, default off)
  - Raises ImportError if hmmlearn is not installed, gracefully skipped by DecoderPipeline
"""

import logging
from typing import Optional
import numpy as np

from src.decoders.base import SequenceDecoder
from src.decoders.merging import SegmentMerger

logger = logging.getLogger(__name__)


class HMMDecoder(SequenceDecoder):
    """GaussianHMM sequence decoder.

    Args:
        n_components: Number of HMM hidden states (None = use n_classes)
        covariance_type: Covariance type ("diag", "full", "tied", "spherical")
        n_iter: Maximum Baum-Welch iterations
        min_segment: Minimum frames for short segment merging after decoding
        prob_aware_merge: Whether to use probability scores when merging
        supervised_init: Whether to initialize emission parameters from training labels
        refine_em: Whether to run Baum-Welch refinement after supervised initialization
        self_boost: Self-transition pseudo-count (for transition matrix initialization)
    """

    name = "hmm"

    def __init__(
        self,
        n_components: Optional[int] = None,
        covariance_type: str = "diag",
        n_iter: int = 100,
        min_segment: int = 30,
        prob_aware_merge: bool = True,
        supervised_init: bool = True,
        refine_em: bool = False,
        self_boost: float = 30.0,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.min_segment = min_segment
        self.prob_aware_merge = prob_aware_merge
        self.supervised_init = supervised_init
        self.refine_em = refine_em
        self.self_boost = self_boost
        self._hmm = None
        self._n_classes = None
        self._merger = SegmentMerger(min_segment, prob_aware_merge)

    def fit(
        self,
        y_train: np.ndarray,
        n_classes: int,
        proba_train: Optional[np.ndarray] = None,
        feat_train: Optional[np.ndarray] = None,
    ) -> "HMMDecoder":
        try:
            from hmmlearn import hmm
        except ImportError:
            raise ImportError(
                "hmmlearn is not installed, skipping HMM decoding. Install via `pip install hmmlearn`."
            )

        from src.temporal_validator import _fit_transitions_asymmetric

        self._n_classes = n_classes
        n_comp = self.n_components if self.n_components is not None else n_classes

        # Observation data for HMM: prefer proba, then feat
        obs_train = proba_train if proba_train is not None else feat_train
        if obs_train is None:
            raise ValueError("HMMDecoder.fit requires proba_train or feat_train")

        y_train_arr = np.asarray(y_train).astype(int)

        # Build HMM
        self._hmm = hmm.GaussianHMM(
            n_components=n_comp,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42,
            verbose=False,
        )

        # Initialize transition matrix (from label statistics)
        log_trans, log_pi = _fit_transitions_asymmetric(
            y_train_arr, n_classes,
            self_boost=self.self_boost,
            cross_smoothing=1.0,
            per_class=False,
        )
        # If n_comp != n_classes, need to resize the matrices
        if n_comp == n_classes:
            self._hmm.startprob_ = np.exp(log_pi)
            # Ensure numerical precision
            self._hmm.startprob_ /= self._hmm.startprob_.sum()
            self._hmm.transmat_ = np.exp(log_trans)
            self._hmm.transmat_ /= self._hmm.transmat_.sum(axis=1, keepdims=True)
        else:
            # Use uniform distribution (user specified different n_components)
            logger.info(
                f"[HMM] n_components={n_comp} != n_classes={n_classes}, "
                f"using uniform initialization"
            )
            self._hmm.startprob_ = np.ones(n_comp) / n_comp
            self._hmm.transmat_ = np.ones((n_comp, n_comp)) / n_comp

        # Supervised initialization of emission parameters
        if self.supervised_init and n_comp == n_classes:
            D = obs_train.shape[1]
            means = np.zeros((n_comp, D), dtype=np.float64)
            covars = np.zeros((n_comp, D), dtype=np.float64)
            for c in range(n_classes):
                mask = y_train_arr == c
                if mask.sum() > 1:
                    means[c] = obs_train[mask].mean(axis=0)
                    covars[c] = obs_train[mask].var(axis=0) + 1e-6
                else:
                    means[c] = obs_train.mean(axis=0)
                    covars[c] = obs_train.var(axis=0) + 1e-6
            self._hmm.means_ = means
            self._hmm.covars_ = np.maximum(covars, 1e-8)
            logger.info(
                f"[HMM] Supervised initialization complete: means shape={means.shape}, "
                f"covariance_type={self.covariance_type}"
            )

        # Optional Baum-Welch refinement
        if self.refine_em:
            logger.info(f"[HMM] Starting Baum-Welch refinement (n_iter={self.n_iter})...")
            self._hmm.fit(obs_train)
            logger.info("[HMM] Baum-Welch refinement complete")

        return self

    def decode(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        obs = proba if proba is not None else feat
        if obs is None:
            raise ValueError("HMMDecoder.decode requires proba or feat")
        if self._hmm is None:
            raise RuntimeError("HMMDecoder has not been fit yet")

        _, states = self._hmm.decode(obs, algorithm="viterbi")
        labels = states.astype(np.int32)

        # Post-process: merge short segments
        if self.min_segment > 1:
            labels = self._merger.merge(labels, proba)

        return labels

    def predict_proba(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return HMM posterior probabilities (predict_proba)."""
        obs = proba if proba is not None else feat
        if obs is None:
            return super().predict_proba(proba, feat)
        if self._hmm is None:
            raise RuntimeError("HMMDecoder has not been fit yet")

        posteriors = self._hmm.predict_proba(obs)
        # If n_comp != n_classes, need to map
        if posteriors.shape[1] != self._n_classes:
            out = np.zeros((len(obs), self._n_classes), dtype=np.float32)
            for c in range(min(posteriors.shape[1], self._n_classes)):
                out[:, c] = posteriors[:, c]
            row_sum = out.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum <= 0, 1.0, row_sum)
            out /= row_sum
            return out
        return posteriors.astype(np.float32)
