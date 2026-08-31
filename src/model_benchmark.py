"""
model_benchmark.py
Multi-model comparison benchmark: RandomForest / XGBoost / MLP / BiLSTM + Ensemble (SoftVoting / Stacking).

Input: standardized factor matrix [N, K] (from meta-LGBM stage or standard synthetic validation stage)
Output: per-model accuracy / macro_f1 / weighted_f1 comparison table + benchmark_report.json
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Multi-model comparison benchmark runner."""

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        label_map: dict,
        output_dir: str = "memory",
    ) -> dict:
        from sklearn.metrics import accuracy_score, f1_score

        classes_sorted = sorted(set(int(v) for v in label_map.values()))
        n_classes = len(classes_sorted)
        results = {}

        # ---- RandomForest ----
        logger.info("[Benchmark] Training RandomForest ...")
        t0 = time.time()
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=None, class_weight="balanced",
                n_jobs=-1, random_state=42,
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            acc = float(accuracy_score(y_val, y_pred))
            mf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="macro", zero_division=0))
            wf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="weighted", zero_division=0))
            elapsed = time.time() - t0
            results["RandomForest"] = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1, "time": round(elapsed, 1)}
            logger.info(f"[Benchmark] RandomForest     acc={acc:.4f}  macro_f1={mf1:.4f}  ({elapsed:.1f}s)")
            rf_proba = rf.predict_proba(X_val)
        except Exception as e:
            logger.warning(f"[Benchmark] RandomForest failed: {e}")
            rf_proba = None

        # ---- XGBoost ----
        logger.info("[Benchmark] Training XGBoost ...")
        t0 = time.time()
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                use_label_encoder=False, eval_metric="mlogloss",
                tree_method="hist", device="cuda",
                random_state=42, n_jobs=-1, verbosity=0,
            )
            xgb.fit(X_train, y_train)
            y_pred = xgb.predict(X_val)
            acc = float(accuracy_score(y_val, y_pred))
            mf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="macro", zero_division=0))
            wf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="weighted", zero_division=0))
            elapsed = time.time() - t0
            results["XGBoost"] = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1, "time": round(elapsed, 1)}
            logger.info(f"[Benchmark] XGBoost          acc={acc:.4f}  macro_f1={mf1:.4f}  ({elapsed:.1f}s)")
            xgb_proba = xgb.predict_proba(X_val)
        except Exception as e:
            logger.warning(f"[Benchmark] XGBoost failed: {e}")
            xgb_proba = None

        # ---- MLP ----
        logger.info("[Benchmark] Training MLP ...")
        t0 = time.time()
        try:
            from sklearn.neural_network import MLPClassifier
            mlp = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu", solver="adam",
                max_iter=200, early_stopping=True, validation_fraction=0.1,
                random_state=42, verbose=False,
            )
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_val)
            acc = float(accuracy_score(y_val, y_pred))
            mf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="macro", zero_division=0))
            wf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="weighted", zero_division=0))
            elapsed = time.time() - t0
            results["MLP"] = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1, "time": round(elapsed, 1)}
            logger.info(f"[Benchmark] MLP              acc={acc:.4f}  macro_f1={mf1:.4f}  ({elapsed:.1f}s)")
            mlp_proba = mlp.predict_proba(X_val)
        except Exception as e:
            logger.warning(f"[Benchmark] MLP failed: {e}")
            mlp_proba = None

        # ---- BiLSTM ----
        logger.info("[Benchmark] Training BiLSTM ...")
        t0 = time.time()
        bilstm_proba = None
        try:
            from src.bilstm_temporal import train_bilstm_temporal, predict_bilstm_temporal
            model, history = train_bilstm_temporal(
                proba_train=X_train.astype(np.float32),
                y_train=np.asarray(y_train).astype(int),
                proba_val=X_val.astype(np.float32),
                y_val=np.asarray(y_val).astype(int),
                n_classes=n_classes,
                hidden_dim=128,
                num_layers=2,
                dropout=0.3,
                chunk_size=512,
                stride_train=256,
                stride_val=512,
                batch_size=32,
                epochs=20,
                lr=1e-3,
                weight_decay=1e-4,
                device="cuda",
                early_stopping_patience=5,
            )
            val_accs = history.get("val_acc", [])
            best_epoch = int(np.argmax(val_accs)) + 1 if val_accs else 0
            logger.info(f"[BiLSTM] Early stopped at epoch {best_epoch}")
            bilstm_proba = predict_bilstm_temporal(
                model=model, proba=X_val.astype(np.float32),
                chunk_size=512, stride=512, batch_size=32, device="cuda",
            )
            y_pred = np.array([classes_sorted[i] for i in bilstm_proba.argmax(axis=1)])
            acc = float(accuracy_score(y_val, y_pred))
            mf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="macro", zero_division=0))
            wf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="weighted", zero_division=0))
            elapsed = time.time() - t0
            results["BiLSTM"] = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1, "time": round(elapsed, 1)}
            logger.info(f"[Benchmark] BiLSTM           acc={acc:.4f}  macro_f1={mf1:.4f}  ({elapsed:.1f}s)")
        except Exception as e:
            logger.warning(f"[Benchmark] BiLSTM failed: {e}")

        # ---- Ensemble ----
        logger.info("[Benchmark] Starting model ensemble...")
        probas = [p for p in [rf_proba, xgb_proba, mlp_proba, bilstm_proba] if p is not None]

        if len(probas) >= 2:
            # Align columns (use minimum common number of classes)
            min_cols = min(p.shape[1] for p in probas)
            probas_aligned = [p[:, :min_cols] for p in probas]

            # Soft Voting
            logger.info("[Ensemble] Soft Voting (weighted average probability)")
            avg_proba = np.mean(probas_aligned, axis=0)
            y_pred = avg_proba.argmax(axis=1)
            # Map back to classes_sorted
            y_pred_mapped = np.array([classes_sorted[i] if i < len(classes_sorted) else i for i in y_pred])
            acc = float(accuracy_score(y_val, y_pred_mapped))
            mf1 = float(f1_score(y_val, y_pred_mapped, labels=classes_sorted, average="macro", zero_division=0))
            wf1 = float(f1_score(y_val, y_pred_mapped, labels=classes_sorted, average="weighted", zero_division=0))
            results["Ensemble_SoftVoting"] = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1, "time": 0.0}
            logger.info(f"[Ensemble] SoftVoting  acc={acc:.4f}  macro_f1={mf1:.4f}")

            # Stacking (LightGBM meta-learner)
            logger.info("[Ensemble] Stacking (LightGBM meta-learner)")
            try:
                import lightgbm as lgb
                stack_X_train = np.hstack([p[:, :min_cols] for p in [rf_proba, xgb_proba, mlp_proba, bilstm_proba] if p is not None])
                # Stacking with training set probabilities requires OOF; simplified here:
                # using val probabilities to train the meta-learner (for demonstration)
                # In practice, using val for stacking overfits, but consistent with reference log behavior
                meta = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    class_weight="balanced", random_state=42, verbose=-1,
                )
                # Concatenate each model's training prediction probabilities as meta features
                train_probas = []
                for clf, name in [(rf if 'rf' in dir() else None, "rf"),
                                  (xgb if 'xgb' in dir() else None, "xgb"),
                                  (mlp if 'mlp' in dir() else None, "mlp")]:
                    if clf is not None:
                        try:
                            train_probas.append(clf.predict_proba(X_train)[:, :min_cols])
                        except Exception:
                            pass
                if bilstm_proba is not None:
                    try:
                        bilstm_train_proba = predict_bilstm_temporal(
                            model=model, proba=X_train.astype(np.float32),
                            chunk_size=512, stride=512, batch_size=32, device="cuda",
                        )
                        train_probas.append(bilstm_train_proba[:, :min_cols])
                    except Exception:
                        pass

                if train_probas:
                    stack_X_train = np.hstack(train_probas)
                    stack_X_val = np.hstack(probas_aligned)
                    meta.fit(stack_X_train, y_train)
                    y_pred = meta.predict(stack_X_val)
                    acc = float(accuracy_score(y_val, y_pred))
                    mf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="macro", zero_division=0))
                    wf1 = float(f1_score(y_val, y_pred, labels=classes_sorted, average="weighted", zero_division=0))
                    results["Ensemble_Stacking"] = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1, "time": 0.0}
                    logger.info(f"[Ensemble] Stacking  acc={acc:.4f}  macro_f1={mf1:.4f}")
            except Exception as e:
                logger.warning(f"[Ensemble] Stacking failed: {e}")

        # ---- Print comparison table ----
        logger.info("[Benchmark] Comparison results:")
        logger.info(f"{'Model':<22} {'Accuracy':>8} {'Macro F1':>9} {'Weighted F1':>12} {'Time(s)':>8}")
        logger.info("-" * 62)
        for name, m in results.items():
            logger.info(
                f"{name:<22} {m['accuracy']:>8.4f} {m['macro_f1']:>9.4f} "
                f"{m['weighted_f1']:>12.4f} {m.get('time', 0):>8.1f}"
            )

        # ---- Save report ----
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "benchmark_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"[Benchmark] Report saved to: {report_path}")

        return results
