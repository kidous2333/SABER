"""
Viterbi sequence decoder.

HMM-based Viterbi decoding: estimate transition matrix from training labels,
perform optimal path search in log domain, then merge short segments.
"""

import logging
from typing import Optional
import numpy as np

from src.decoders.base import SequenceDecoder
from src.decoders.merging import SegmentMerger

logger = logging.getLogger(__name__)


class ViterbiDecoder(SequenceDecoder):
    """Standard HMM-based Viterbi sequence decoder.

    Estimates transition matrix and initial probabilities from training labels,
    performs Viterbi decoding in log domain, then greedily merges short segments.

    Args:
        min_segment: Minimum behavior duration in frames (annotation prior)
        self_boost: Self-transition pseudo-count amplification factor (0=use uniform smoothing, >0=asymmetric smoothing)
        cross_smoothing: Cross-class transition pseudo-count (default 1.0)
        prob_aware_merge: Whether to use probability scores to decide short segment merge direction
    """

    name = "viterbi"

    def __init__(self, min_segment: int = 30, self_boost: float = 0.0,
                 cross_smoothing: float = 1.0, prob_aware_merge: bool = False,
                 per_class_transition: bool = False):
        self.min_segment = min_segment
        self.self_boost = self_boost
        self.cross_smoothing = cross_smoothing
        self.prob_aware_merge = prob_aware_merge
        self.per_class_transition = per_class_transition
        self.log_trans = None   # [C, C] log transition probabilities
        self.log_pi = None      # [C] log initial probabilities
        self.n_classes = None
        self._merger = SegmentMerger(min_segment, prob_aware_merge)

    def fit(
        self,
        y_train: np.ndarray,
        n_classes: int,
        proba_train: Optional[np.ndarray] = None,
        feat_train: Optional[np.ndarray] = None,
    ) -> "ViterbiDecoder":
        from src.temporal_validator import _fit_transitions_asymmetric

        self.n_classes = n_classes
        if self.self_boost > 0:
            self.log_trans, self.log_pi = _fit_transitions_asymmetric(
                y_train, n_classes,
                self_boost=self.self_boost,
                cross_smoothing=self.cross_smoothing,
                per_class=self.per_class_transition,
            )
        else:
            C = n_classes
            s = self.cross_smoothing
            trans = np.full((C, C), s, dtype=np.float64)
            for t in range(len(y_train) - 1):
                trans[int(y_train[t]), int(y_train[t + 1])] += 1.0
            trans /= trans.sum(axis=1, keepdims=True)
            self.log_trans = np.log(trans + 1e-300)
            pi = np.full(C, s, dtype=np.float64)
            pi[int(y_train[0])] += 1.0
            pi /= pi.sum()
            self.log_pi = np.log(pi + 1e-300)
        return self

    def decode(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Viterbi decoding.

        Args:
            proba: [T, C] per-frame class probabilities
        Returns:
            [T] integer label sequence
        """
        T, C = proba.shape
        log_emit = np.log(np.clip(proba, 1e-300, 1.0))

        # Viterbi DP
        dp = np.empty((T, C), dtype=np.float64)
        bp = np.empty((T, C), dtype=np.int32)

        dp[0] = self.log_pi + log_emit[0]
        for t in range(1, T):
            scores = dp[t - 1, :, None] + self.log_trans  # [C, C]
            bp[t] = scores.argmax(axis=0)
            dp[t] = scores.max(axis=0) + log_emit[t]

        # Backtracking
        path = np.empty(T, dtype=np.int32)
        path[T - 1] = dp[T - 1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]

        # Post-process: merge short segments
        if self.min_segment > 1:
            path = self._merger.merge(path, proba)
        return path


class DurationViterbiDecoder(SequenceDecoder):
    """Duration-aware Viterbi sequence decoder.

    Unlike the standard first-order HMM, this decoder explicitly tracks the duration of the current state,
    imposing a log-domain penalty on switches shorter than min_duration.

    DP state: (class, duration_bucket)
      - duration_bucket in {1, 2, ..., min_duration, "long"}
      - "long" means duration >= min_duration frames, switching is no longer penalized

    Args:
        min_duration: Minimum behavior duration in frames
        duration_penalty: Short segment switching penalty (log domain, 0=disable, fall back to standard Viterbi)
        self_boost: Self-transition pseudo-count amplification factor
        cross_smoothing: Cross-class transition pseudo-count
        prob_aware_merge: Whether to use probability-aware merging in post-processing
    """

    name = "viterbi_duration"

    def __init__(self, min_duration: int = 30, duration_penalty: float = 5.0,
                 self_boost: float = 30.0, cross_smoothing: float = 1.0,
                 prob_aware_merge: bool = True,
                 per_class_duration: bool = False,
                 per_class_transition: bool = False):
        self.min_duration = min_duration
        self.duration_penalty = duration_penalty
        self.self_boost = self_boost
        self.cross_smoothing = cross_smoothing
        self.prob_aware_merge = prob_aware_merge
        self.per_class_duration = per_class_duration
        self.per_class_transition = per_class_transition
        self.log_trans = None
        self.log_pi = None
        self.n_classes = None
        # Pre-computed duration penalty (indexed as duration in frames - 1)
        self._penalty_cache = None  # per-class: [C, D+1] or [D+1] if uniform
        self._merger = SegmentMerger(min_duration, prob_aware_merge)

    def fit(
        self,
        y_train: np.ndarray,
        n_classes: int,
        proba_train: Optional[np.ndarray] = None,
        feat_train: Optional[np.ndarray] = None,
    ) -> "DurationViterbiDecoder":
        from src.temporal_validator import _fit_transitions_asymmetric, _estimate_per_class_durations

        self.n_classes = n_classes
        self.log_trans, self.log_pi = _fit_transitions_asymmetric(
            y_train, n_classes,
            self_boost=self.self_boost,
            cross_smoothing=self.cross_smoothing,
            per_class=self.per_class_transition,
        )

        D = self.min_duration
        pen = self.duration_penalty

        if self.per_class_duration:
            dur_info = _estimate_per_class_durations(y_train, n_classes)
            per_class_logprob = dur_info["per_class_dur_logprob"]  # [C, max_dur+1]
            max_dur_src = per_class_logprob.shape[1] - 1
            self._penalty_cache = np.zeros((n_classes, D + 1), dtype=np.float64)
            for c in range(n_classes):
                for d in range(1, D + 1):
                    if d <= max_dur_src:
                        self._penalty_cache[c, d] = per_class_logprob[c, d]
                    else:
                        self._penalty_cache[c, d] = 0.0
            logger.info(
                f"[DurationViterbiDecoder] Using per-class duration distribution: "
                f"min_duration={D}"
            )
        else:
            self._penalty_cache = np.zeros(D + 1, dtype=np.float64)
            for d in range(1, D):
                self._penalty_cache[d] = -pen * (D - d) / D
        return self

    def decode(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Duration-aware Viterbi decoding.

        Args:
            proba: [T, C] per-frame class probabilities
        Returns:
            [T] integer label sequence
        """
        T, C = proba.shape
        D = self.min_duration
        B = D + 1              # total buckets
        log_emit = np.log(np.clip(proba, 1e-300, 1.0))

        # Fall back to standard Viterbi when penalty is 0
        if self.duration_penalty <= 0:
            v = ViterbiDecoder(
                min_segment=self.min_duration,
                self_boost=self.self_boost,
                cross_smoothing=self.cross_smoothing,
                prob_aware_merge=self.prob_aware_merge,
            )
            v.log_trans = self.log_trans
            v.log_pi = self.log_pi
            v.n_classes = self.n_classes
            return v.decode(proba)

        # dp[t, c, b]: optimal log probability at time t, class c, duration bucket b
        # bp[t, c, b]: predecessor (c_prev, b_prev) encoded as int
        dp = np.full((T, C, B), -1e300, dtype=np.float64)
        bp = np.zeros((T, C, B), dtype=np.int32)

        # Initialization (t=0): only dur=1 possible
        for c_idx in range(C):
            dp[0, c_idx, 0] = self.log_pi[c_idx] + log_emit[0, c_idx]

        # Forward DP
        for t in range(1, T):
            for c_idx in range(C):
                emit_c = log_emit[t, c_idx]
                log_self = self.log_trans[c_idx, c_idx]

                # case 1: stay in current class (duration increments)
                for b in range(B):
                    prev_score = dp[t - 1, c_idx, b]
                    if prev_score <= -1e299:
                        continue
                    if b < D - 1:
                        new_b = b + 1
                    else:
                        new_b = D
                    new_score = prev_score + log_self + emit_c
                    if new_score > dp[t, c_idx, new_b]:
                        dp[t, c_idx, new_b] = new_score
                        bp[t, c_idx, new_b] = c_idx * B + b

                # case 2: switch from another class c_prev != c (duration resets to 1)
                new_b = 0
                best_switch_score = -1e300
                best_switch_from = -1
                for c_prev in range(C):
                    if c_prev == c_idx:
                        continue
                    log_A = self.log_trans[c_prev, c_idx]
                    for b_prev in range(B):
                        prev_score = dp[t - 1, c_prev, b_prev]
                        if prev_score <= -1e299:
                            continue
                        dur_prev = b_prev + 1 if b_prev < D else D
                        if self.per_class_duration:
                            penalty = self._penalty_cache[c_prev, min(dur_prev, D)]
                        else:
                            penalty = self._penalty_cache[min(dur_prev, D)]
                        score = prev_score + log_A + penalty + emit_c
                        if score > best_switch_score:
                            best_switch_score = score
                            best_switch_from = c_prev * B + b_prev
                if best_switch_score > dp[t, c_idx, new_b]:
                    dp[t, c_idx, new_b] = best_switch_score
                    bp[t, c_idx, new_b] = best_switch_from

        # Backtracking
        best_score = -1e300
        best_c = 0
        best_b = 0
        for c_idx in range(C):
            for b in range(B):
                if dp[T - 1, c_idx, b] > best_score:
                    best_score = dp[T - 1, c_idx, b]
                    best_c = c_idx
                    best_b = b

        path = np.empty(T, dtype=np.int32)
        path[T - 1] = best_c
        flat_bp = bp[T - 1, best_c, best_b]
        for t in range(T - 2, -1, -1):
            c_idx = flat_bp // B
            path[t] = c_idx
            if t > 0:
                b = flat_bp % B
                flat_bp = bp[t, c_idx, b]

        # Post-process: secondary merge of residual short segments
        if self.prob_aware_merge and self.min_duration > 1:
            path = self._merger.merge(path, proba)
        elif self.min_duration > 1:
            path = self._merger.merge(path)

        return path
