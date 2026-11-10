"""Trois familles de modèles, sans dépendance lourde (numpy uniquement).

Chaque modèle expose :
- ``fit(train)`` avec un DataFrame ``user_id, item_id, rating`` ;
- ``predict(users, items)`` : notes prédites, bornées dans [1, 5] ;
- ``score_all(user_id)`` : un score par film, utilisé pour le top-N.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class _Base:
    """Encodage commun des identifiants et matrice utilisateur × film."""

    min_rating, max_rating = 1.0, 5.0

    def _index(self, train: pd.DataFrame) -> None:
        self.users_ = np.sort(train["user_id"].unique())
        self.items_ = np.sort(train["item_id"].unique())
        self.uidx_ = {u: i for i, u in enumerate(self.users_)}
        self.iidx_ = {m: j for j, m in enumerate(self.items_)}
        self.u_ = train["user_id"].map(self.uidx_).to_numpy()
        self.i_ = train["item_id"].map(self.iidx_).to_numpy()
        self.r_ = train["rating"].to_numpy(dtype=float)
        self.mu_ = float(self.r_.mean())
        self.R_ = np.zeros((len(self.users_), len(self.items_)))
        self.R_[self.u_, self.i_] = self.r_
        self.mask_ = self.R_ > 0

    def _clip(self, x):
        return np.clip(x, self.min_rating, self.max_rating)

    def seen_items(self, user_id) -> np.ndarray:
        """Films déjà notés par l'utilisateur dans les données d'entraînement."""
        if user_id not in self.uidx_:
            return np.array([], dtype=self.items_.dtype)
        return self.items_[self.mask_[self.uidx_[user_id]]]

    def recommend(self, user_id, k: int = 10, exclude_seen: bool = True) -> list:
        """Les ``k`` films au score le plus élevé, hors films déjà vus."""
        scores = np.asarray(self.score_all(user_id), dtype=float).copy()
        if exclude_seen and user_id in self.uidx_:
            scores[self.mask_[self.uidx_[user_id]]] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, k - 1)[:k] if k else np.array([], dtype=int)
        top = top[np.lexsort((self.items_[top], -scores[top]))]
        return self.items_[top].tolist()


class PopularityBaseline(_Base):
    """Référence simple.

    - Note prédite : moyenne du film « lissée » vers la moyenne globale
      (``damping`` notes fictives égales à la moyenne globale).
    - Top-N : les films les plus notés (popularité), identiques pour tous.
    """

    def __init__(self, damping: float = 5.0):
        self.damping = damping

    def fit(self, train: pd.DataFrame) -> "PopularityBaseline":
        self._index(train)
        counts = self.mask_.sum(axis=0)
        sums = self.R_.sum(axis=0)
        self.item_mean_ = (sums + self.damping * self.mu_) / (counts + self.damping)
        self.popularity_ = counts.astype(float)
        return self

    def predict(self, users, items) -> np.ndarray:
        out = np.full(len(items), self.mu_)
        for n, m in enumerate(items):
            j = self.iidx_.get(m)
            if j is not None:
                out[n] = self.item_mean_[j]
        return self._clip(out)

    def score_all(self, user_id) -> np.ndarray:
        return self.popularity_


class ItemKNN(_Base):
    """Filtrage collaboratif par films (similarité cosinus ajustée).

    Les notes sont centrées par utilisateur, puis la similarité cosinus est
    calculée entre colonnes (films). Seuls les ``k`` voisins les plus
    similaires de chaque film sont conservés.
    - Note prédite : moyenne de l'utilisateur + moyenne pondérée des écarts
      sur les voisins qu'il a notés.
    - Top-N : somme des similarités avec les films qu'il a notés.
    """

    def __init__(self, k: int = 40, shrinkage: float = 10.0):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, train: pd.DataFrame) -> "ItemKNN":
        self._index(train)
        counts = self.mask_.sum(axis=1)
        self.user_mean_ = np.where(counts > 0, self.R_.sum(axis=1) / np.maximum(counts, 1), self.mu_)
        centered = np.where(self.mask_, self.R_ - self.user_mean_[:, None], 0.0)
        norms = np.linalg.norm(centered, axis=0)
        norms[norms == 0] = 1.0
        sim = (centered.T @ centered) / np.outer(norms, norms)
        co = self.mask_.T.astype(float) @ self.mask_.astype(float)
        sim *= co / (co + self.shrinkage)          # pénalise les co-notations rares
        np.fill_diagonal(sim, 0.0)
        if self.k < sim.shape[0]:
            weakest = np.argpartition(-sim, self.k, axis=1)[:, self.k:]
            np.put_along_axis(sim, weakest, 0.0, axis=1)
        self.sim_ = sim
        self.centered_ = centered
        return self

    def predict(self, users, items) -> np.ndarray:
        out = np.empty(len(items))
        for n, (u, m) in enumerate(zip(users, items)):
            ui, j = self.uidx_.get(u), self.iidx_.get(m)
            if ui is None:
                out[n] = self.mu_
                continue
            if j is None:
                out[n] = self.user_mean_[ui]
                continue
            w = self.sim_[j] * self.mask_[ui]
            denom = np.abs(w).sum()
            delta = (w @ self.centered_[ui]) / denom if denom > 0 else 0.0
            out[n] = self.user_mean_[ui] + delta
        return self._clip(out)

    def score_all(self, user_id) -> np.ndarray:
        if user_id not in self.uidx_:
            return self.mask_.sum(axis=0).astype(float)
        return self.sim_ @ self.mask_[self.uidx_[user_id]].astype(float)


class MatrixFactorization(_Base):
    """Factorisation matricielle biaisée (type « Funk SVD ») entraînée par SGD.

    r̂(u, i) = μ + b_u + b_i + p_u · q_i
    """

    def __init__(self, n_factors: int = 50, n_epochs: int = 20, lr: float = 0.01,
                 reg: float = 0.05, seed: int = 42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.seed = seed

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> "MatrixFactorization":
        self._index(train)
        rng = np.random.default_rng(self.seed)
        n_u, n_i = len(self.users_), len(self.items_)
        self.bu_, self.bi_ = np.zeros(n_u), np.zeros(n_i)
        self.P_ = rng.normal(0, 0.1, (n_u, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, (n_i, self.n_factors))
        self.history_ = []
        lr, reg, mu = self.lr, self.reg, self.mu_
        bu, bi, P, Q = self.bu_, self.bi_, self.P_, self.Q_
        for epoch in range(self.n_epochs):
            for n in rng.permutation(len(self.r_)):
                u, i = self.u_[n], self.i_[n]
                pu, qi = P[u], Q[i]
                err = self.r_[n] - (mu + bu[u] + bi[i] + pu @ qi)
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])
                P[u] = pu + lr * (err * qi - reg * pu)
                Q[i] = qi + lr * (err * pu - reg * qi)
            row = {"epoch": epoch + 1, "train_rmse": _rmse(self.r_, self._raw(self.u_, self.i_))}
            if valid is not None:
                row["test_rmse"] = _rmse(valid["rating"].to_numpy(float),
                                         self.predict(valid["user_id"], valid["item_id"]))
            self.history_.append(row)
        return self

    def _raw(self, u, i):
        return self.mu_ + self.bu_[u] + self.bi_[i] + np.einsum("ij,ij->i", self.P_[u], self.Q_[i])

    def predict(self, users, items) -> np.ndarray:
        out = np.empty(len(items))
        for n, (u, m) in enumerate(zip(users, items)):
            ui, j = self.uidx_.get(u), self.iidx_.get(m)
            value = self.mu_
            if ui is not None:
                value += self.bu_[ui]
            if j is not None:
                value += self.bi_[j]
            if ui is not None and j is not None:
                value += self.P_[ui] @ self.Q_[j]
            out[n] = value
        return self._clip(out)

    def score_all(self, user_id) -> np.ndarray:
        if user_id not in self.uidx_:
            return self.mu_ + self.bi_
        ui = self.uidx_[user_id]
        return self.mu_ + self.bu_[ui] + self.bi_ + self.Q_ @ self.P_[ui]


def _rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


MODELS = {
    "popularite": PopularityBaseline,
    "itemknn": ItemKNN,
    "svd": MatrixFactorization,
}
