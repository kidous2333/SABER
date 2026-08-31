"""
CRF sequence decoder.

Linear-chain Conditional Random Field (sklearn-crfsuite), using temporal feature matrix
or probability-augmented features for sequence labeling.
"""

import gc
import logging
from typing import Optional
import numpy as np

from src.decoders.base import SequenceDecoder

logger = logging.getLogger(__name__)


class CRFDecoder(SequenceDecoder):
    """Linear-chain CRF sequence decoder (depends on sklearn-crfsuite).

    Takes a temporal feature matrix [T, D] or probability matrix (auto-augmented) as input,
    trains the CRF and decodes. Raises ImportError if sklearn-crfsuite is not installed.

    Args:
        algorithm: CRF optimization algorithm (default "lbfgs")
        c1: L1 regularization coefficient
        c2: L2 regularization coefficient
        max_iterations: Maximum iterations
        n_bins: Number of bins for continuous feature discretization
        max_seq_len: Maximum length of a single sequence (0 = no chunking)
        use_enriched_features: Whether to use _enrich_crf_features augmentation when only proba is provided
        max_train_samples: Maximum training data sampling frames (0 = no limit)
        tune_hyperparams: Whether to grid search c1/c2
    """

    name = "crf"

    def __init__(self, algorithm: str = "lbfgs", c1: float = 0.1, c2: float = 0.1,
                 max_iterations: int = 100, n_bins: int = 10, max_seq_len: int = 0,
                 use_enriched_features: bool = True,
                 max_train_samples: int = 0, tune_hyperparams: bool = False):
        self.algorithm = algorithm
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.n_bins = n_bins
        self.max_seq_len = max_seq_len
        self.use_enriched_features = use_enriched_features
        self.max_train_samples = max_train_samples
        self.tune_hyperparams = tune_hyperparams
        self.crf = None
        self._bin_edges = None  # [D, n_bins+1]
        self._fitted_c1 = c1
        self._fitted_c2 = c2

    @staticmethod
    def _to_crf_seq(X: np.ndarray, bin_edges: np.ndarray) -> list:
        """Convert [T, D] feature matrix to crfsuite format: list of dict."""
        T, D = X.shape
        all_bins = np.zeros((T, D), dtype=np.int32)
        for d in range(D):
            all_bins[:, d] = np.clip(
                np.searchsorted(bin_edges[d], X[:, d], side="right") - 1,
                0, len(bin_edges[d]) - 2,
            )
        seq = []
        for t in range(T):
            feat = {}
            for d in range(D):
                b = int(all_bins[t, d])
                feat[f"f{d}={b}"] = 1.0
                if t > 0:
                    b_prev = int(all_bins[t - 1, d])
                    feat[f"f{d}_prev={b_prev}"] = 1.0
            seq.append(feat)
        return seq

    def _resolve_features(
        self,
        proba: Optional[np.ndarray],
        feat: Optional[np.ndarray],
    ) -> np.ndarray:
        """Resolve feature matrix based on available input.

        LGBM path: feat is not None, use directly.
        NN path: feat is None, construct augmented features from proba.
        """
        if feat is not None:
            return feat.astype(np.float32)
        if proba is not None and self.use_enriched_features:
            from src.temporal_validator import _enrich_crf_features
            return _enrich_crf_features(proba)
        if proba is not None:
            return proba.astype(np.float32)
        raise ValueError("CRFDecoder requires at least one of proba or feat")

    def fit(
        self,
        y_train: np.ndarray,
        n_classes: int,
        proba_train: Optional[np.ndarray] = None,
        feat_train: Optional[np.ndarray] = None,
    ) -> "CRFDecoder":
        import sklearn_crfsuite

        X_train = self._resolve_features(proba_train, feat_train)
        y_train_arr = np.asarray(y_train).astype(int)

        # Training data sampling
        if self.max_train_samples > 0 and X_train.shape[0] > self.max_train_samples:
            _rng = np.random.default_rng(42)
            _idx = _rng.choice(X_train.shape[0], self.max_train_samples, replace=False)
            _idx.sort()
            X_train = X_train[_idx]
            y_train_arr = y_train_arr[_idx]
            logger.info(
                f"[CRF] Training data sampling: {len(y_train)} -> {self.max_train_samples} frames"
            )

        T, D = X_train.shape

        # Hyperparameter search (optional)
        if self.tune_hyperparams:
            from src.temporal_validator import _tune_crf_hyperparams
            _bin_edges = np.array([
                np.percentile(X_train[:, d], np.linspace(0, 100, self.n_bins + 1))
                for d in range(D)
            ])
            self._fitted_c1, self._fitted_c2, _ = _tune_crf_hyperparams(
                X_train=X_train,
                y_train=y_train_arr,
                bin_edges=_bin_edges,
                n_bins=self.n_bins,
            )
            logger.info(f"[CRF] Hyperparameter search result: c1={self._fitted_c1}, c2={self._fitted_c2}")

        logger.info(f"[CRF] Computing quantile bin edges: n_bins={self.n_bins}, D={D}")
        self._bin_edges = np.array([
            np.percentile(X_train[:, d], np.linspace(0, 100, self.n_bins + 1))
            for d in range(D)
        ])

        # Chunked training
        chunk = self.max_seq_len if self.max_seq_len > 0 else T
        n_chunks = max(1, (T + chunk - 1) // chunk)
        if n_chunks > 1:
            logger.info(f"[CRF] Splitting {T} frames into {n_chunks} chunks (max_seq_len={chunk})")
        else:
            logger.info(f"[CRF] Single sequence mode: {T} frames trained as one full sequence")

        X_seqs = []
        y_seqs = []
        for i in range(0, T, chunk):
            end = min(i + chunk, T)
            actual = end - i
            if actual < 2 and i > 0:
                continue
            X_seqs.append(self._to_crf_seq(X_train[i:end], self._bin_edges))
            y_seqs.append([str(int(v)) for v in y_train_arr[i:end]])
            if n_chunks > 1 and ((len(X_seqs)) % max(1, n_chunks // 5) == 0 or len(X_seqs) == n_chunks):
                logger.info(
                    f"[CRF] Feature construction progress: {len(X_seqs)}/{n_chunks} chunks "
                    f"(processed ~{end}/{T} frames)"
                )
            gc.collect()

        logger.info(
            f"[CRF] Feature construction complete: {len(X_seqs)} sequences, "
            f"starting LBFGS training (max_iter={self.max_iterations})..."
        )
        self.crf = sklearn_crfsuite.CRF(
            algorithm=self.algorithm,
            c1=self._fitted_c1,
            c2=self._fitted_c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True,
            all_possible_states=True,
        )
        self.crf.fit(X_seqs, y_seqs)
        logger.info(f"[CRF] Training complete")
        return self

    def decode(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        X_val = self._resolve_features(proba, feat)
        crf_seq = self._to_crf_seq(X_val, self._bin_edges)
        pred_str = self.crf.predict([crf_seq])[0]
        return np.array([int(s) for s in pred_str], dtype=np.int32)

    def predict_proba(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return CRF marginal probability matrix [T, C]."""
        X_val = self._resolve_features(proba, feat)
        crf_seq = self._to_crf_seq(X_val, self._bin_edges)
        marginals = self.crf.predict_marginals([crf_seq])[0]
        T = len(marginals)
        n_classes = proba.shape[1]
        out = np.zeros((T, n_classes), dtype=np.float32)
        for t, d in enumerate(marginals):
            for label_str, p in d.items():
                c = int(label_str)
                if c < n_classes:
                    out[t, c] = float(p)
        row_sum = out.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0, 1.0, row_sum)
        out /= row_sum
        return out
