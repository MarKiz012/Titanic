"""Создание модели по имени из конфига.
"""


def build_model(cfg):
    name = cfg["model"]["name"]
    # Параметры лежат по имени модели: model.params.<name>
    params = dict((cfg["model"].get("params") or {}).get(name) or {})
    seed = cfg["seed"]

    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, random_state=seed, **params)

    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_jobs=-1, random_state=seed, **params)

    if name == "histgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(random_state=seed, **params)

    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_jobs=-1, verbose=-1, random_state=seed, **params)

    if name == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(n_jobs=-1, tree_method="hist", random_state=seed, **params)

    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(verbose=0, allow_writing_files=False, random_seed=seed, **params)

    if name == "nn":
        # Нейросети нужны плотные числовые признаки одного масштаба:
        # запускайте с features.encoding=onehot и features.scale=true.
        try:
            from src.nn import TorchMLPClassifier
        except ImportError as exc:
            raise ImportError(
                "Для model.name=nn нужен PyTorch: pip install torch"
            ) from exc

        return TorchMLPClassifier(random_state=seed, **params)

    raise ValueError(
        f"Неизвестная модель '{name}'. Доступные: logreg, rf, histgb, lgbm, xgb, catboost, nn"
    )
