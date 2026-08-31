"""
validator.py
Use LightGBM to train binary classification models and validate each factor's
discriminative power for every behavior class.

Strategy: for each class, perform independent OvR (One-vs-Rest) binary
classification; a factor is considered valid if it can separate any
single class from the rest.

Accepts numpy array input, directly interfacing with MouseBehaviorDataset output:
  factor_values : np.ndarray [T]      (float32, from FactorEngine)
  labels        : np.ndarray [T]      (int, from MBD.labels)
"""

import logging
import warnings
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import lightgbm as lgb

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class FactorValidator:
    """
    Perform independent OvR binary classification validation for each behavior class.

    Factor valid condition: at least one class has AUC >= min_auc and F1 >= min_f1.
    """

    def __init__(self, cfg: dict):
        val_cfg = cfg["validation"]
        self.min_auc = val_cfg.get("min_auc", 0.65)
        self.min_f1 = val_cfg.get("min_f1", 0.15)
        self.lgbm_params = val_cfg.get("lgbm_params", {})
        self.early_stopping_rounds = val_cfg.get("early_stopping_rounds", 20)
        self.use_gpu = bool(val_cfg.get("use_gpu", False))
        if self.use_gpu:
            self.lgbm_params = {**self.lgbm_params, "device": "gpu"}
            self.lgbm_params.pop("n_jobs", None)
            logger.info("LightGBM GPU acceleration enabled (validation)")

    # ------------------------------------------------------------------
    # Single-factor Holdout validation (OvR binary classification)
    # ------------------------------------------------------------------
    def validate_single_holdout(
        self,
        train_factor: np.ndarray,
        train_labels: np.ndarray,
        val_factor: np.ndarray,
        val_labels: np.ndarray,
    ) -> dict:
        """
        Perform OvR binary classification for each class, returning per-class AUC/F1
        and overall validity.

        Returns
        -------
        dict:
          valid        : bool   — at least one class passes the threshold
          best_auc     : float  — highest AUC among all classes
          best_f1      : float  — highest F1 among all classes
          best_class   : str    — name of the best class
          valid_classes: list   — list of classes passing the threshold (with auc/f1)
          per_class    : dict   — detailed metrics per class
          reason       : str
        """
        train_factor = np.asarray(train_factor, dtype=np.float32)
        val_factor   = np.asarray(val_factor,   dtype=np.float32)
        train_labels = np.asarray(train_labels)
        val_labels   = np.asarray(val_labels)

        # Filter nan
        train_mask = np.isfinite(train_factor)
        val_mask   = np.isfinite(val_factor)
        X_train = train_factor[train_mask].reshape(-1, 1)
        y_train_all = train_labels[train_mask]
        X_val   = val_factor[val_mask].reshape(-1, 1)
        y_val_all   = val_labels[val_mask]

        if len(X_train) < 50 or len(X_val) < 10:
            return self._fail("Insufficient samples")

        classes = np.unique(np.concatenate([y_train_all, y_val_all]))
        if len(classes) < 2:
            return self._fail("Insufficient number of classes")

        per_class = {}
        valid_classes = []

        for cls in classes:
            y_tr = (y_train_all == cls).astype(int)
            y_vl = (y_val_all   == cls).astype(int)

            # Skip if either training or validation set has only one label
            if y_tr.sum() == 0 or y_vl.sum() == 0:
                continue
            if y_tr.sum() == len(y_tr) or y_vl.sum() == len(y_vl):
                continue

            # Majority class undersampling: randomly downsample negative class to 1:1 ratio
            pos_idx = np.where(y_tr == 1)[0]
            neg_idx = np.where(y_tr == 0)[0]
            if len(neg_idx) > len(pos_idx):
                rng = np.random.default_rng(42)
                neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)
            sample_idx = np.concatenate([pos_idx, neg_idx])
            X_tr_bal = X_train[sample_idx]
            y_tr_bal = y_tr[sample_idx]

            model = lgb.LGBMClassifier(
                **{**self._default_lgbm_params(), **self.lgbm_params},
                verbose=-1,
            )
            fit_kwargs = {}
            if self.early_stopping_rounds > 0:
                fit_kwargs["eval_set"] = [(X_val, y_vl)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            model.fit(X_tr_bal, y_tr_bal, **fit_kwargs)
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

            try:
                auc = float(roc_auc_score(y_vl, y_prob))
            except ValueError:
                auc = 0.0

            f1 = float(f1_score(y_vl, y_pred, zero_division=0))

            per_class[str(cls)] = {"auc": round(auc, 4), "f1": round(f1, 4)}

            if auc >= self.min_auc:
                valid_classes.append({"class": str(cls), "auc": round(auc, 4), "f1": round(f1, 4)})

        if not per_class:
            return self._fail("Insufficient samples for all classes")

        best = max(per_class.items(), key=lambda x: x[1]["auc"])
        best_cls, best_metrics = best
        valid = len(valid_classes) > 0

        reason = (
            f"Passed {len(valid_classes)} classes: {[v['class'] for v in valid_classes]}"
            if valid
            else f"Best AUC={best_metrics['auc']:.3f}({best_cls}) below threshold {self.min_auc}"
        )

        return {
            "valid": valid,
            "best_auc": best_metrics["auc"],
            "best_f1": best_metrics["f1"],
            "best_class": best_cls,
            "valid_classes": valid_classes,
            "per_class": per_class,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Multi-factor joint validation (OvR binary classification, keeping interface consistent)
    # ------------------------------------------------------------------
    def validate_multi(
        self,
        factor_matrix: np.ndarray,
        labels: np.ndarray,
        factor_names: list = None,
    ) -> dict:
        """
        Multi-factor joint OvR validation, returning the best AUC/F1 for each class.
        """
        from sklearn.model_selection import StratifiedKFold

        factor_matrix = np.asarray(factor_matrix, dtype=np.float32)
        labels = np.asarray(labels)

        valid_mask = np.isfinite(factor_matrix).all(axis=1)
        X = factor_matrix[valid_mask]
        y = labels[valid_mask]

        if len(X) < 50:
            return {"auc": 0.0, "f1": 0.0, "reason": "Insufficient samples"}

        classes = np.unique(y)
        per_class = {}

        for cls in classes:
            y_bin = (y == cls).astype(int)
            if y_bin.sum() < 5:
                continue

            skf = StratifiedKFold(n_splits=min(5, y_bin.sum()), shuffle=True, random_state=42)
            aucs, f1s = [], []
            for tr, vl in skf.split(X, y_bin):
                model = lgb.LGBMClassifier(**self._default_lgbm_params(), verbose=-1)
                fit_kwargs = {}
                if self.early_stopping_rounds > 0:
                    fit_kwargs["eval_set"] = [(X[vl], y_bin[vl])]
                    fit_kwargs["callbacks"] = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
                model.fit(X[tr], y_bin[tr], **fit_kwargs)
                y_prob = model.predict_proba(X[vl])[:, 1]
                y_pred = model.predict(X[vl])
                try:
                    aucs.append(roc_auc_score(y_bin[vl], y_prob))
                except ValueError:
                    aucs.append(0.0)
                f1s.append(f1_score(y_bin[vl], y_pred, zero_division=0))
            per_class[str(cls)] = {
                "auc": round(float(np.mean(aucs)), 4),
                "f1": round(float(np.mean(f1s)), 4),
            }

        if not per_class:
            return {"auc": 0.0, "f1": 0.0, "reason": "Insufficient samples for all classes"}

        best = max(per_class.values(), key=lambda x: x["auc"])
        return {
            "auc": best["auc"],
            "f1": best["f1"],
            "per_class": per_class,
            "n_factors": X.shape[1],
            "n_samples": int(len(X)),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _fail(reason: str) -> dict:
        return {
            "valid": False, "best_auc": 0.0,
            "best_class": "", "valid_classes": [], "per_class": {}, "reason": reason,
        }

    @staticmethod
    def _default_lgbm_params() -> dict:
        return {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "num_leaves": 15,
            "random_state": 42,
            "n_jobs": -1,
        }
