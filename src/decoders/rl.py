"""
RL sequence decoder (pure numpy Q-learning).

Models sequence labeling as a sequential decision problem:
  - State: (proba_argmax, prev_label, dur_bucket) -- 3-dimensional discrete
  - Action: select label 0..C-1
  - Reward: correctness + short segment switching penalty

Q-table size O(C^3), only ~2000 entries for 10 classes, memory friendly.
Pure numpy implementation, no external RL framework dependencies.
"""

import logging
from typing import Optional
import numpy as np

from src.decoders.base import SequenceDecoder
from src.decoders.merging import SegmentMerger

logger = logging.getLogger(__name__)


class RLDecoder(SequenceDecoder):
    """Q-learning sequence decoder.

    Args:
        min_segment: Minimum behavior duration in frames
        prob_aware_merge: Whether to use probability-aware merging in post-processing
        learning_rate: Q-learning learning rate
        n_episodes: Number of training episodes (full passes through training sequence)
        epsilon: Initial exploration rate
        gamma: Discount factor
        epsilon_min: Minimum exploration rate
        epsilon_decay: Per-episode exploration rate decay factor
    """

    name = "rl"

    def __init__(
        self,
        min_segment: int = 30,
        prob_aware_merge: bool = True,
        learning_rate: float = 0.01,
        n_episodes: int = 1000,
        epsilon: float = 0.1,
        gamma: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.min_segment = min_segment
        self.prob_aware_merge = prob_aware_merge
        self.learning_rate = learning_rate
        self.n_episodes = n_episodes
        self.epsilon = epsilon
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._Q = None          # [C, C, 2, C]
        self._n_classes = None
        self._merger = SegmentMerger(min_segment, prob_aware_merge)

    def fit(
        self,
        y_train: np.ndarray,
        n_classes: int,
        proba_train: Optional[np.ndarray] = None,
        feat_train: Optional[np.ndarray] = None,
    ) -> "RLDecoder":
        C = n_classes
        self._n_classes = C

        # Q-table: [argmax, prev_label, dur_bucket, action]
        self._Q = np.zeros((C, C, 2, C), dtype=np.float64)

        y_arr = np.asarray(y_train).astype(int)
        T = len(y_arr)

        # Use proba argmax as observation; if no proba, use labels
        obs = proba_train if proba_train is not None else feat_train
        argmax_seq = obs.argmax(axis=1).astype(int) if obs is not None else y_arr

        eps = self.epsilon
        logger.info(
            f"[RL] Starting Q-learning: n_classes={C}, n_episodes={self.n_episodes}, "
            f"lr={self.learning_rate}, epsilon={eps:.3f}, gamma={self.gamma}"
        )

        for episode in range(self.n_episodes):
            prev_label = int(y_arr[0])
            duration = 1
            episode_reward = 0.0

            for t in range(T):
                am = int(argmax_seq[t])
                dur_b = 0 if duration < self.min_segment else 1
                state = (am, prev_label, dur_b)

                # epsilon-greedy
                if np.random.random() < eps:
                    action = np.random.randint(C)
                else:
                    action = int(np.argmax(self._Q[state]))

                # Reward: correctness + duration penalty
                correct = 1.0 if action == int(y_arr[t]) else 0.0
                switch_penalty = -0.5 if (action != prev_label and dur_b == 0) else 0.0
                reward = correct + switch_penalty
                episode_reward += reward

                # Next state
                if t + 1 < T:
                    next_am = int(argmax_seq[t + 1])
                    next_dur = duration + 1 if action == prev_label else 1
                    next_dur_b = 0 if next_dur < self.min_segment else 1
                    next_state = (next_am, action, next_dur_b)
                    td_target = reward + self.gamma * self._Q[next_state].max()
                else:
                    td_target = reward

                # Q-update
                idx = state + (action,)
                self._Q[idx] += self.learning_rate * (
                    td_target - self._Q[idx]
                )

                prev_label = action
                duration = next_dur if t + 1 < T else duration

            # Decay epsilon
            eps = max(self.epsilon_min, eps * self.epsilon_decay)

            if (episode + 1) % max(1, self.n_episodes // 10) == 0:
                logger.info(
                    f"[RL] episode {episode + 1}/{self.n_episodes}  "
                    f"avg_reward={episode_reward / T:.4f}  epsilon={eps:.4f}"
                )

        logger.info(
            f"[RL] Q-learning complete: final_epsilon={eps:.4f}, "
            f"Q-table shape={self._Q.shape}"
        )
        return self

    def decode(
        self,
        proba: np.ndarray,
        feat: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        if self._Q is None:
            raise RuntimeError("RLDecoder has not been fit yet")

        obs = proba if proba is not None else feat
        if obs is None:
            raise ValueError("RLDecoder.decode requires proba or feat")

        T = obs.shape[0]
        C = self._n_classes
        argmax_seq = obs.argmax(axis=1).astype(int)

        path = np.empty(T, dtype=np.int32)
        prev_label = int(argmax_seq[0])
        duration = 1
        path[0] = prev_label

        for t in range(1, T):
            am = int(argmax_seq[t])
            dur_b = 0 if duration < self.min_segment else 1
            state = (am, prev_label, dur_b)
            # Greedy selection (no exploration during inference)
            action = int(np.argmax(self._Q[state]))
            path[t] = action
            duration = duration + 1 if action == prev_label else 1
            prev_label = action

        # Post-process: merge short segments
        if self.min_segment > 1:
            path = self._merger.merge(path, proba)

        return path
