"""
RuleCorrector: Lightweight rule-based post-processor (designed based on annotation data statistics).

Key characteristics of annotation data (source: labels/ full statistics):
  - All classes have 97~99.8% self-transition rate, behavior switches are extremely rare
  - approach / positive_sniffs have ~50% of segments <30 frames (genuine short behaviors)
  - Nearly all classes have <1% of segments shorter than 3 frames (isolated reversals under 3 frames are likely noise)

Therefore, the rules are designed as:
  1. Flicker correction: Only correct isolated reversals of 1~max_flicker frames (when neighbors are same class + probability does not support current class)
  2. Local window probability correction: For longer segments, flip if window probability overwhelmingly does not support current label
  3. Boundary refinement: At segment boundaries, check if 1~2 frames should be swapped based on probability
"""

import logging
from typing import Optional
import numpy as np

from src.decoders.base import SequenceDecoder

logger = logging.getLogger(__name__)


class RuleCorrector(SequenceDecoder):
    """Lightweight rule corrector -- based on data priors, only fixes the most obvious errors.

    Args:
        max_flicker: Maximum flicker frames (1~3)
        flicker_prob_threshold: Flicker correction probability threshold
        window_size: Local probability check window size (frames)
        proba_consistency_threshold: Window probability consistency threshold
        min_consecutive: Minimum consecutive frames that must satisfy the flip condition
        boundary_refine: Whether to enable boundary refinement
        confidence_margin: Frames where BiLSTM probability gap (top1-top2) exceeds this value are not corrected
        smooth_transitions: Whether to smooth overly aggressive label switches
    """

    name = "rule_correct"

    def __init__(
        self,
        max_flicker: int = 3,
        flicker_prob_threshold: float = 0.05,
        window_size: int = 31,
        proba_consistency_threshold: float = 0.20,
        min_consecutive: int = 5,
        boundary_refine: bool = True,
        confidence_margin: float = 0.3,
        smooth_transitions: bool = True,
    ):
        self.max_flicker = max_flicker
        self.flicker_prob_threshold = flicker_prob_threshold
        self.window_size = window_size
        self.proba_consistency_threshold = proba_consistency_threshold
        self.min_consecutive = min_consecutive
        self.boundary_refine = boundary_refine
        self.confidence_margin = confidence_margin
        self.smooth_transitions = smooth_transitions
        self._n_classes = None

    def fit(
        self,
        y_train: np.ndarray,
        n_classes: int,
        proba_train: Optional[np.ndarray] = None,
        feat_train: Optional[np.ndarray] = None,
    ) -> "RuleCorrector":
        self._n_classes = n_classes
        logger.info(
            f"[RuleCorrector] max_flicker={self.max_flicker}, "
            f"flicker_threshold={self.flicker_prob_threshold}, "
            f"window={self.window_size}, consistency_threshold={self.proba_consistency_threshold}, "
            f"min_consecutive={self.min_consecutive}, boundary_refine={self.boundary_refine}, "
            f"confidence_margin={self.confidence_margin}, smooth_transitions={self.smooth_transitions}"
        )
        return self

    def decode(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
        prev_labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        T, C = proba.shape

        # Step 0: Baseline labels
        if prev_labels is not None:
            labels = np.asarray(prev_labels, dtype=np.int32).copy()
        else:
            labels = proba.argmax(axis=1).astype(np.int32)

        # Pre-compute BiLSTM confidence: top1 - top2 probability gap
        top2 = -np.partition(-proba, 1, axis=1)[:, :2]
        confidence = top2[:, 0] - top2[:, 1]  # [T]

        # ---- Step 1: Flicker correction ----
        n_flicker_fixed = 0
        segs = self._find_segments(labels)
        for start, end, seg_cls in segs:
            seg_len = end - start
            if seg_len > self.max_flicker:
                continue

            # Confidence filter: if BiLSTM is very sure, do not correct
            if confidence[start:end].mean() > self.confidence_margin:
                continue

            left_cls = labels[start - 1] if start > 0 else None
            right_cls = labels[end] if end < T else None

            if left_cls is not None and right_cls is not None and left_cls == right_cls:
                neighbor_cls = left_cls
                seg_mean = proba[start:end].mean(axis=0)
                if seg_mean[neighbor_cls] - seg_mean[seg_cls] > self.flicker_prob_threshold:
                    labels[start:end] = neighbor_cls
                    n_flicker_fixed += seg_len
            elif left_cls is not None and right_cls is not None:
                seg_mean = proba[start:end].mean(axis=0)
                left_score = seg_mean[left_cls]
                right_score = seg_mean[right_cls]
                if left_score > right_score + self.flicker_prob_threshold:
                    labels[start:end] = left_cls
                    n_flicker_fixed += seg_len
                elif right_score > left_score + self.flicker_prob_threshold:
                    labels[start:end] = right_cls
                    n_flicker_fixed += seg_len
            elif left_cls is not None:
                seg_mean = proba[start:end].mean(axis=0)
                if seg_mean[left_cls] - seg_mean[seg_cls] > self.flicker_prob_threshold:
                    labels[start:end] = left_cls
                    n_flicker_fixed += seg_len
            elif right_cls is not None:
                seg_mean = proba[start:end].mean(axis=0)
                if seg_mean[right_cls] - seg_mean[seg_cls] > self.flicker_prob_threshold:
                    labels[start:end] = right_cls
                    n_flicker_fixed += seg_len

        if n_flicker_fixed > 0:
            logger.info(
                f"[RuleCorrector] Flicker correction: {n_flicker_fixed} frames "
                f"(max_flicker={self.max_flicker}, threshold={self.flicker_prob_threshold})"
            )

        # ---- Step 2: Local window probability correction (long segments only) ----
        half = max(1, self.window_size // 2)
        window = 2 * half + 1

        cs = np.cumsum(np.vstack([np.zeros((1, C), dtype=proba.dtype), proba]), axis=0)
        window_sum = cs[window:] - cs[:-window]
        window_mean = window_sum / window

        pad_left = half
        pad_right = T - pad_left - window_mean.shape[0]
        if pad_right < 0:
            pad_right = 0
        window_mean_padded = np.vstack([
            np.tile(window_mean[0], (pad_left, 1)),
            window_mean,
            np.tile(window_mean[-1], (pad_right, 1)),
        ])
        window_best = window_mean_padded.argmax(axis=1).astype(np.int32)
        window_best_score = window_mean_padded.max(axis=1)
        current_score = window_mean_padded[np.arange(T), labels]

        flip_mask = (
            (window_best != labels)
            & (window_best_score - current_score > self.proba_consistency_threshold)
            & (confidence < self.confidence_margin)  # only correct low-confidence frames
        )

        # Only keep flips within long segments
        segs_after_flicker = self._find_segments(labels)
        for start, end, _seg_cls in segs_after_flicker:
            if (end - start) < self.window_size:
                flip_mask[start:end] = False

        if self.min_consecutive > 1 and flip_mask.any():
            flip_mask = self._filter_short_runs(flip_mask, self.min_consecutive)

        n_flips = int(flip_mask.sum())
        if n_flips > 0:
            labels[flip_mask] = window_best[flip_mask]
            logger.info(
                f"[RuleCorrector] Window probability correction: {n_flips} frames "
                f"(window={window}, threshold={self.proba_consistency_threshold})"
            )

        # ---- Step 3: Boundary refinement ----
        if self.boundary_refine:
            n_boundary_fixed = 0
            segs = self._find_segments(labels)
            for start, end, seg_cls in segs:
                for offset in range(1, min(3, (end - start) // 2 + 1)):
                    t_left = start + offset - 1
                    if t_left < end and confidence[t_left] < self.confidence_margin:
                        local = proba[max(0, t_left-1):min(T, t_left+2)].mean(axis=0)
                        neighbor = labels[start - 1] if start > 0 else None
                        if neighbor is not None and local[neighbor] - local[seg_cls] > 0.1:
                            labels[t_left] = neighbor
                            n_boundary_fixed += 1
                    t_right = end - offset
                    if t_right >= start and confidence[t_right] < self.confidence_margin:
                        local = proba[max(0, t_right-1):min(T, t_right+2)].mean(axis=0)
                        neighbor = labels[end] if end < T else None
                        if neighbor is not None and local[neighbor] - local[labels[t_right]] > 0.1:
                            labels[t_right] = neighbor
                            n_boundary_fixed += 1

            if n_boundary_fixed > 0:
                logger.info(f"[RuleCorrector] Boundary refinement: {n_boundary_fixed} frames")

        # ---- Step 4: Boundary smoothing ----
        # Detect label switch points: if P(old class) is still > P(new class) at the switch frame, delay the switch
        if self.smooth_transitions:
            n_smoothed = 0
            for t in range(1, T):
                if labels[t] != labels[t-1]:
                    prev_cls = labels[t-1]
                    new_cls = labels[t]
                    # Old class probability still higher than new class at switch frame -> keep old class
                    if (confidence[t] < self.confidence_margin
                            and proba[t, prev_cls] > proba[t, new_cls]):
                        labels[t] = prev_cls
                        n_smoothed += 1
            if n_smoothed > 0:
                logger.info(f"[RuleCorrector] Boundary smoothing: {n_smoothed} frames")

        return labels

    def predict_proba(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Return softmax probabilities (not one-hot).

        RuleCorrector only corrects very few frames; using original probabilities for AUC calculation is more accurate.
        """
        return proba.astype(np.float32)

    @staticmethod
    def _find_segments(labels: np.ndarray) -> list:
        """Return [(start, end, class), ...], end is exclusive boundary."""
        segs = []
        T = len(labels)
        i = 0
        while i < T:
            c = labels[i]
            j = i
            while j < T and labels[j] == c:
                j += 1
            segs.append((i, j, int(c)))
            i = j
        return segs

    @staticmethod
    def _filter_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
        """Filter out consecutive True blocks shorter than min_len."""
        result = mask.copy()
        T = len(mask)
        i = 0
        while i in range(T):
            if mask[i]:
                j = i
                while j < T and mask[j]:
                    j += 1
                if j - i < min_len:
                    result[i:j] = False
                i = j
            else:
                i += 1
        return result
