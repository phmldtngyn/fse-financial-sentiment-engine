"""Wrapper 4 mô hình hồi quy: LR, DT, k-NN, ANN(MLP)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


class LinearRegressor:
    name = "linear_regression"

    def __init__(self, **kw):
        self.scaler = StandardScaler()
        self.model = LinearRegression(**kw)

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class DecisionTreeRegressorWrapper:
    name = "decision_tree"

    def __init__(self, max_depth=8, min_samples_leaf=5, **kw):
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            **kw,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class KNNRegressor:
    name = "knn"

    def __init__(self, n_neighbors=5, weights="distance", **kw):
        self.scaler = StandardScaler()
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors, weights=weights, **kw
        )

    def fit(self, X, y):
        self.model.fit(self.scaler.fit_transform(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ANNRegressor:
    name = "ann_mlp"

    def __init__(self, hidden_layers=(64, 32), dropout=0.2, epochs=100,
                 batch_size=32, lr=1e-3, early_stopping_patience=10, **kw):
        self.hidden_layers = list(hidden_layers)
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = early_stopping_patience
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()   # ← MỚI
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X).astype("float32")
        ys = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel().astype("float32")  # ← MỚI
        self.model = MLP(Xs.shape[1], self.hidden_layers, self.dropout).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        Xt = torch.tensor(Xs, device=self.device)
        yt = torch.tensor(ys, device=self.device)

        best_loss = float("inf")
        wait = 0
        n = len(Xt)
        for ep in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            ep_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                pred = self.model(Xt[idx])
                loss = loss_fn(pred, yt[idx])
                loss.backward()
                opt.step()
                ep_loss += loss.item() * len(idx)
            ep_loss /= n

            if ep_loss < best_loss - 1e-6:
                best_loss = ep_loss
                wait = 0
                self._best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X).astype("float32")
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(torch.tensor(Xs, device=self.device)).cpu().numpy()
        return self.y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()  # ← MỚI

def build_model(name: str, **cfg):
    """Factory. name ∈ {linear_regression, decision_tree, knn, ann_mlp}."""
    if name == "linear_regression":
        return LinearRegressor(**cfg)
    if name == "decision_tree":
        return DecisionTreeRegressorWrapper(**cfg)
    if name == "knn":
        return KNNRegressor(**cfg)
    if name == "ann_mlp":
        return ANNRegressor(**cfg)
    raise ValueError(f"Unknown model: {name}")