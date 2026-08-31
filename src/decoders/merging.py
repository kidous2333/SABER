"""
SegmentMerger: Short segment merging post-processor.

Extracted from the Viterbi decoder as an independent, composable component
that can be used independently for short segment trimming of any label sequence.
"""

from typing import Optional
import numpy as np


class SegmentMerger:
    """Merges consecutive segments shorter than min_segment into adjacent longer segments.

    Two modes:
      prob_aware=False: Always merge to the left neighbor (original behavior)
      prob_aware=True: When left and right neighbors differ, use probability scores to decide direction
    """

    def __init__(self, min_segment: int = 30, prob_aware: bool = False):
        self.min_segment = min_segment
        self.prob_aware = prob_aware

    def merge(
        self,
        seq: np.ndarray,
        proba: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run iterative short segment merging.

        Args:
            seq: Integer label sequence [T]
            proba: Probability matrix [T, C], required when prob_aware=True

        Returns:
            Merged label sequence [T]
        """
        if self.min_segment <= 1:
            return seq.copy()

        if self.prob_aware and proba is not None:
            return self._merge_probabilistic(seq, proba)
        return self._merge_naive(seq)

    def _merge_naive(self, seq: np.ndarray) -> np.ndarray:
        """Always merge to the left neighbor."""
        seq = seq.copy()
        T = len(seq)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < T:
                c = seq[i]
                j = i
                while j < T and seq[j] == c:
                    j += 1
                seg_len = j - i
                if seg_len < self.min_segment:
                    left_len = i
                    right_len = T - j
                    if left_len == 0 and right_len == 0:
                        break
                    if left_len == 0:
                        fill = seq[j] if j < T else seq[i - 1]
                    elif right_len == 0:
                        fill = seq[i - 1]
                    else:
                        fill = seq[i - 1]
                    seq[i:j] = fill
                    changed = True
                i = j
        return seq

    def _merge_probabilistic(
        self, seq: np.ndarray, proba: np.ndarray
    ) -> np.ndarray:
        """When left and right neighbors differ, use probability scores to decide merge direction."""
        seq = seq.copy()
        T = len(seq)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < T:
                c = seq[i]
                j = i
                while j < T and seq[j] == c:
                    j += 1
                seg_len = j - i
                if seg_len < self.min_segment:
                    left_label = seq[i - 1] if i > 0 else None
                    right_label = seq[j] if j < T else None

                    if left_label is None and right_label is None:
                        break

                    if left_label is None:
                        fill = right_label
                    elif right_label is None:
                        fill = left_label
                    elif left_label == right_label:
                        fill = left_label
                    else:
                        seg_proba = proba[i:j]
                        left_score = seg_proba[:, int(left_label)].mean()
                        right_score = seg_proba[:, int(right_label)].mean()
                        fill = left_label if left_score >= right_score else right_label

                    seq[i:j] = fill
                    changed = True
                i = j
        return seq
