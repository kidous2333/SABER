"""Sequence decoder package.

Provides a unified SequenceDecoder interface and multiple post-processing methods:
  - ViterbiDecoder / DurationViterbiDecoder: HMM Viterbi decoding
  - CRFDecoder: Linear-chain CRF decoding
  - HMMDecoder: Full HMM (hmmlearn)
  - RLDecoder: Q-learning sequence labeling
  - SegmentMerger: Short segment merging
  - DecoderPipeline: Unified orchestration of multiple decoders
"""

from src.decoders.base import SequenceDecoder
from src.decoders.merging import SegmentMerger
from src.decoders.viterbi import ViterbiDecoder, DurationViterbiDecoder
from src.decoders.crf import CRFDecoder
from src.decoders.pipeline import DecoderPipeline
from src.decoders.rl import RLDecoder
from src.decoders.rules import RuleCorrector

# HMMDecoder lazy import (hmmlearn is optional dependency)
try:
    from src.decoders.hmm import HMMDecoder
except ImportError:
    HMMDecoder = None

__all__ = [
    "SequenceDecoder",
    "SegmentMerger",
    "ViterbiDecoder",
    "DurationViterbiDecoder",
    "CRFDecoder",
    "DecoderPipeline",
    "RLDecoder",
    "HMMDecoder",
    "RuleCorrector",
]
