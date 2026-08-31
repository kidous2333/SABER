"""
DecoderPipeline: Sequence decoder orchestrator.

Uniformly manages the fit -> decode -> predict_proba -> evaluate workflow for multiple SequenceDecoders,
eliminating duplicated code between NN path and LGBM path in TemporalValidator.run().

The returned seq_decoding dict format is fully compatible with the original hardcoded format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.decoders.base import SequenceDecoder

logger = logging.getLogger(__name__)


class DecoderPipeline:
    """Orchestrates multiple sequence decoders.

    For each decoder in the pipeline, sequentially executes:
      1. fit(y_train, n_classes, proba_train, feat_train)
      2. decode(proba_val, feat_val) -> labels
      3. predict_proba(proba_val, feat_val) -> proba (for AUC)
      4. _eval_sequence_metrics(y_val, labels, ...) -> metrics

    Returns:
        {decoder_name: metrics_dict}, matching the seq_decoding format
        used by train_behavior.py.
    """

    def __init__(
        self,
        decoders: List[SequenceDecoder],
        classes_sorted: list,
        id_to_name: dict,
        output_dir: str = "",
        viz_dir: str = "",
    ):
        self.decoders = decoders
        self.classes_sorted = classes_sorted
        self.id_to_name = id_to_name
        self.output_dir = output_dir
        self.viz_dir = viz_dir

    def run(
        self,
        proba_val: np.ndarray,
        y_val: np.ndarray,
        proba_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        feat_train: Optional[np.ndarray] = None,
        feat_val: Optional[np.ndarray] = None,
    ) -> Dict[str, dict]:
        """Run all decoders.

        Args:
            proba_val: Validation set probability matrix [T, C]
            y_val: Validation set labels [T]
            proba_train: Training set probability matrix [T, C] (optional, needed by CRF/HMM/RL)
            y_train: Training set labels [T] (optional)
            feat_train: Training set temporal features [T, D] (optional, used by CRF LGBM path)
            feat_val: Validation set temporal features [T, D] (optional, used by CRF LGBM path)

        Returns:
            {decoder_name: {... metrics ...}}
            e.g. {"viterbi_duration": {"accuracy": 0.85, ...}, "crf": {...}}
        """
        from src.temporal_validator import _eval_sequence_metrics

        n_classes = proba_val.shape[1]
        y_val_arr = np.asarray(y_val).astype(int)
        y_train_arr = np.asarray(y_train).astype(int) if y_train is not None else None

        seq_decoding: dict = {}
        prev_labels = None  # Chained pass: output labels from previous decoder

        for decoder in self.decoders:
            dec_name = decoder.name
            try:
                # -- Dependency check --
                if dec_name == "crf":
                    import sklearn_crfsuite  # noqa: F401
                elif dec_name == "hmm":
                    import hmmlearn  # noqa: F401

                # -- Fit --
                logger.info(f"[Decoder][{dec_name}] Starting training...")
                decoder.fit(
                    y_train=y_train_arr,
                    n_classes=n_classes,
                    proba_train=proba_train,
                    feat_train=feat_train,
                )

                # -- Decode (chained: pass the previous decoder's output labels) --
                logger.info(f"[Decoder][{dec_name}] Starting decoding...")
                y_pred = decoder.decode(
                    proba=proba_val, feat=feat_val,
                    prev_labels=prev_labels,
                )

                # -- Predict proba (for AUC) --
                proba_for_auc = decoder.predict_proba(
                    proba=proba_val, feat=feat_val,
                    prev_labels=prev_labels,
                )

                # -- Evaluate --
                metrics = _eval_sequence_metrics(
                    y_val_arr, y_pred, self.classes_sorted,
                    self.id_to_name, proba=proba_for_auc,
                )
                seq_decoding[dec_name] = metrics

                logger.info(
                    f"[Decoder][{dec_name}] accuracy={metrics['accuracy']:.4f}  "
                    f"balanced_acc={metrics['balanced_accuracy']:.4f}  "
                    f"macro_auc={metrics['macro_auc']:.4f}  "
                    f"weighted_auc={metrics['weighted_auc']:.4f}  "
                    f"macro_f1={metrics['macro_f1']:.4f}  "
                    f"weighted_f1={metrics['weighted_f1']:.4f}"
                )

                # -- Chained pass: pass current decoder output to next decoder --
                prev_labels = y_pred

                # -- Save model --
                self._save_model(decoder, dec_name)

            except ImportError as e:
                _msg = str(e)
                if "crfsuite" in _msg.lower() or "hmmlearn" in _msg.lower():
                    logger.info(
                        f"[Decoder][{dec_name}] {_msg.split(chr(10))[0]}, skipping."
                    )
                else:
                    logger.warning(f"[Decoder][{dec_name}] Import failed: {e}")
                seq_decoding[dec_name] = {"skipped": _msg.split(chr(10))[0]}
            except Exception as e:
                logger.warning(f"[Decoder][{dec_name}] Failed: {e}")
                seq_decoding[dec_name] = {"error": str(e)}

        return seq_decoding

    def _save_model(self, decoder: SequenceDecoder, dec_name: str) -> None:
        """Save decoder model to output_dir/weights/."""
        if not self.output_dir:
            return
        try:
            import pickle
            out = Path(self.output_dir) / "weights"
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{dec_name}_decoder.pkl"
            with open(path, "wb") as f:
                pickle.dump(decoder, f)
            logger.info(f"[Decoder][{dec_name}] Model saved: {path}")
        except Exception as e:
            logger.warning(f"[Decoder][{dec_name}] Model save failed: {e}")
