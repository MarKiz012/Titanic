"""Загрузка данных и разбиение на фолды."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def load_data(cfg):
    """Читает train и test. Возвращает (X, y, X_test, test_ids)."""
    data_cfg = cfg["data"]
    target = data_cfg["target"]
    id_column = data_cfg["id_column"]

    train_path = Path(data_cfg["train_path"])
    if not train_path.exists():
        raise FileNotFoundError(
            f"Не найден файл {train_path}. Скачайте данные соревнования "
            "и положите train.csv и test.csv в data/raw/"
        )

    train = pd.read_csv(train_path)
    y = train[target]
    X = train.drop(columns=[c for c in (target, id_column) if c in train.columns])

    X_test, test_ids = None, None
    test_path = Path(data_cfg["test_path"])
    if test_path.exists():
        test = pd.read_csv(test_path)
        test_ids = test[id_column] if id_column in test.columns else pd.Series(range(len(test)))
        X_test = test.drop(columns=[c for c in (target, id_column) if c in test.columns])

    print(f"train: {X.shape}, test: {None if X_test is None else X_test.shape}")
    return X, y, X_test, test_ids


def make_folds(X, y, cfg):
    """Список пар (train_idx, valid_idx) со стратификацией по таргету."""
    splitter = StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"],
        shuffle=cfg["cv"]["shuffle"],
        random_state=cfg["seed"] if cfg["cv"]["shuffle"] else None,
    )
    return list(splitter.split(X, y))
