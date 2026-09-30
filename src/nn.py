"""
Сеть ждёт на входе плотные числовые признаки примерно одного масштаба:
    python main.py train --set model.name=nn \
        --set features.encoding=onehot --set features.scale=true
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from torch import nn


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """Полносвязная сеть: Linear -> ReLU -> Dropout, на выходе один логит.

    Parameters
    ----------
    hidden_sizes : размеры скрытых слоёв, например (64, 32)
    dropout : доля зануляемых нейронов между слоями
    lr, weight_decay : параметры Adam
    epochs : максимум эпох; реально обычно меньше из-за ранней остановки
    batch_size : размер мини-батча
    patience : сколько эпох терпеть отсутствие улучшения на валидации
    val_size : какая часть train-фолда уходит на раннюю остановку
    """

    def __init__(
        self,
        hidden_sizes=(64, 32),
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=200,
        batch_size=64,
        patience=20,
        val_size=0.15,
        random_state=42,
        verbose=False,
    ):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_size = val_size
        self.random_state = random_state
        self.verbose = verbose

    def _build_module(self, n_features):
        layers = []
        in_size = n_features
        for size in self.hidden_sizes:
            layers += [nn.Linear(in_size, int(size)), nn.ReLU(), nn.Dropout(self.dropout)]
            in_size = int(size)
        layers.append(nn.Linear(in_size, 1))  # один логит: P(выжил)
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y_raw = np.asarray(y).ravel()
        self.classes_ = np.unique(y_raw)
        self.n_features_in_ = X.shape[1]
        y = y_raw.astype(np.float32)

        torch.manual_seed(self.random_state)

        # Кусочек train-фолда уходит на раннюю остановку. Валидационный фолд
        # для этого брать нельзя — иначе он перестанет быть честной проверкой.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state, stratify=y_raw
        )

        self.module_ = self._build_module(self.n_features_in_)
        optimizer = torch.optim.Adam(
            self.module_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        X_tr_t, y_tr_t = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
        X_val_t, y_val_t = torch.from_numpy(X_val), torch.from_numpy(y_val)

        rng = np.random.default_rng(self.random_state)
        best_loss, best_state, bad_epochs = float("inf"), None, 0
        epoch = 0

        for epoch in range(1, self.epochs + 1):
            self.module_.train()
            order = rng.permutation(len(X_tr_t))
            for start in range(0, len(order), self.batch_size):
                idx = torch.from_numpy(order[start : start + self.batch_size])
                optimizer.zero_grad()
                logits = self.module_(X_tr_t[idx]).squeeze(1)
                loss = criterion(logits, y_tr_t[idx])
                loss.backward()
                optimizer.step()

            self.module_.eval()
            with torch.no_grad():
                val_loss = criterion(self.module_(X_val_t).squeeze(1), y_val_t).item()

            if val_loss < best_loss - 1e-5:
                best_loss, bad_epochs = val_loss, 0
                # копия весов лучшей эпохи: к ней вернёмся после цикла
                best_state = {k: v.detach().clone() for k, v in self.module_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

            if self.verbose and epoch % 10 == 0:
                print(f"  epoch {epoch}: val_loss={val_loss:.4f}")

        if best_state is not None:
            self.module_.load_state_dict(best_state)
        self.best_val_loss_ = best_loss
        self.epochs_run_ = epoch
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.module_.eval()
        with torch.no_grad():
            positive = torch.sigmoid(self.module_(torch.from_numpy(X)).squeeze(1)).numpy()
        # sklearn ждёт две колонки: P(класс 0) и P(класс 1)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
