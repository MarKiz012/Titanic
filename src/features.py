"""Признаки: доменные для Titanic + обычный препроцессинг (пропуски, кодирование)."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
}


def add_features(df):
    """Доменные признаки Titanic.

    Функция не запоминает ничего про данные (никакой статистики по выборке),
    поэтому её безопасно применять к train и test сразу — утечки не будет.
    """
    df = df.copy()

    if "Name" in df:
        title = df["Name"].fillna("").str.extract(r",\s*([^\.]*)\.", expand=False).str.strip()
        title = title.replace(TITLE_MAP)
        df["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master", "Rare"]), "Rare")
        df["NameLength"] = df["Name"].fillna("").str.len()

    if {"SibSp", "Parch"} <= set(df):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    if "Cabin" in df:
        df["HasCabin"] = df["Cabin"].notna().astype(int)
        df["Deck"] = df["Cabin"].fillna("U").astype(str).str[0]

    if "Fare" in df:
        df["FareLog"] = np.log1p(df["Fare"].clip(lower=0))
        if "FamilySize" in df:
            df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    if "Age" in df:
        df["AgeIsNull"] = df["Age"].isna().astype(int)

    # Текстовые колонки, из которых мы уже всё вытащили.
    return df.drop(columns=[c for c in ("Name", "Ticket", "Cabin") if c in df])


def make_preprocessor(X, cfg):
    """ColumnTransformer: заполнение пропусков + кодирование категорий.

    Колонки определяются по типу данных
    """
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if cfg["features"]["scale"]:
        numeric_steps.append(("scaler", StandardScaler()))

    if cfg["features"]["encoding"] == "onehot":
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", encoder),
    ]

    print(f"признаков: {len(numeric)} числовых, {len(categorical)} категориальных")
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline(categorical_steps), categorical),
        ],
        remainder="drop",
    )


def prepare(X, X_test, cfg):
    """Применяет доменные признаки к train и test (если он есть)."""
    if not cfg["features"]["engineering"]:
        return X, X_test
    return add_features(X), (None if X_test is None else add_features(X_test))
