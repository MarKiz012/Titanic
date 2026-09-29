"""Кросс-валидация, метрики и сохранение результатов."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data import load_data, make_folds
from src.features import add_features, make_preprocessor, prepare
from src.models import build_model


def compute_metrics(y_true, proba, threshold):
    """Метрики качества на одном наборе предсказаний.

    accuracy / precision / recall / f1 считаются от классов и зависят от
    порога; roc_auc / pr_auc / log_loss — от вероятностей и от порога не зависят.

    Бизнес-смысл пары precision–recall: precision отвечает на вопрос «сколько
    из тех, кого мы назвали выжившими, действительно выжили» (цена ложной
    тревоги), recall — «какую долю выживших мы нашли» (цена пропуска).
    pr_auc (он же average precision) обобщает эту пару по всем порогам сразу
    и на несбалансированных классах информативнее roc_auc.
    """
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, proba),
    }


def train(cfg):
    """Обучает модель по фолдам и сохраняет всё в outputs/<запуск>/."""
    X, y, X_test, test_ids = load_data(cfg)
    X, X_test = prepare(X, X_test, cfg)

    preprocessor = make_preprocessor(X, cfg)
    folds = make_folds(X, y, cfg)
    threshold = cfg["threshold"]

    oof = np.zeros(len(X))
    test_proba = np.zeros(len(X_test)) if X_test is not None else None
    fold_metrics = []
    models = []

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        # clone -> каждый фолд учит препроцессинг заново, только на своих данных
        pipeline = Pipeline([("prep", clone(preprocessor)), ("model", build_model(cfg))])
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

        oof[valid_idx] = pipeline.predict_proba(X.iloc[valid_idx])[:, 1]
        if X_test is not None:
            test_proba += pipeline.predict_proba(X_test)[:, 1] / len(folds)

        scores = compute_metrics(y.iloc[valid_idx], oof[valid_idx], threshold)
        fold_metrics.append(scores)
        models.append(pipeline)
        print(f"fold {fold}/{len(folds)}: " + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    oof_metrics = compute_metrics(y, oof, threshold)
    print("\nOOF: " + ", ".join(f"{k}={v:.4f}" for k, v in oof_metrics.items()))
    for name in oof_metrics:
        values = [m[name] for m in fold_metrics]
        print(f"  {name}: {np.mean(values):.4f} +- {np.std(values):.4f} по фолдам")

    run_dir = save_run(cfg, models, oof, y, test_proba, test_ids, oof_metrics, fold_metrics)
    print(f"\nРезультаты: {run_dir}")
    return oof_metrics


def save_run(cfg, models, oof, y, test_proba, test_ids, oof_metrics, fold_metrics):
    run_name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{cfg['model']['name']}"
    run_dir = Path(cfg["output_dir"]) / run_name
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"config": cfg, "oof": oof_metrics, "folds": fold_metrics},
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame({"y_true": y, "proba": oof}).to_csv(run_dir / "oof.csv", index=False)
    for i, model in enumerate(models):
        joblib.dump(model, run_dir / "models" / f"fold_{i}.joblib")

    if test_proba is not None:
        submission = pd.DataFrame(
            {
                cfg["data"]["id_column"]: test_ids,
                cfg["data"]["target"]: (test_proba >= cfg["threshold"]).astype(int),
            }
        )
        submission.to_csv(run_dir / "submission.csv", index=False)
        pd.DataFrame({"proba": test_proba}).to_csv(run_dir / "test_proba.csv", index=False)

    return run_dir


def predict(cfg, run_dir=None, input_path=None):
    """Предсказание сохранёнными моделями (по умолчанию — последний запуск)."""
    output_dir = Path(cfg["output_dir"])
    if run_dir is None:
        runs = sorted(p for p in output_dir.glob("*") if (p / "models").is_dir())
        if not runs:
            raise FileNotFoundError(f"В {output_dir} нет ни одного обученного запуска")
        run_dir = runs[-1]
    run_dir = Path(run_dir)
    print(f"Использую запуск: {run_dir}")

    # Данные готовим так же, как при обучении этого запуска, — берём его конфиг.
    with open(run_dir / "metrics.json", encoding="utf-8") as f:
        run_cfg = json.load(f).get("config", cfg)

    models = [joblib.load(p) for p in sorted((run_dir / "models").glob("fold_*.joblib"))]

    frame = pd.read_csv(input_path or run_cfg["data"]["test_path"])
    id_column, target = run_cfg["data"]["id_column"], run_cfg["data"]["target"]
    ids = frame[id_column] if id_column in frame.columns else pd.Series(range(len(frame)))
    X = frame.drop(columns=[c for c in (id_column, target) if c in frame.columns])
    if run_cfg["features"]["engineering"]:
        X = add_features(X)

    proba = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    submission = pd.DataFrame({id_column: ids, target: (proba >= run_cfg["threshold"]).astype(int)})
    path = run_dir / "submission.csv"
    submission.to_csv(path, index=False)
    print(f"Сабмишн сохранён: {path}")
    return submission
