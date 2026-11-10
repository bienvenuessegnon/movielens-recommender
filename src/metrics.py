"""Métriques de prédiction de notes et de recommandation top-N."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_recall_at_k(recommended: list, relevant: set, k: int = 10) -> tuple[float, float]:
    """Precision@k = hits / k ; Recall@k = hits / |pertinents|."""
    if not relevant:
        raise ValueError("relevant ne doit pas être vide")
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k, hits / len(relevant)


def evaluate_top_n(model, test: pd.DataFrame, k: int = 10, threshold: float = 4.0) -> dict:
    """Moyenne de Precision@k et Recall@k sur les utilisateurs du test.

    Un film est « pertinent » s'il est noté >= ``threshold`` dans le test.
    Les recommandations excluent les films vus à l'entraînement. Les
    utilisateurs sans film pertinent dans le test sont ignorés.
    """
    liked = test[test["rating"] >= threshold].groupby("user_id")["item_id"].apply(set)
    precisions, recalls = [], []
    for user_id, relevant in liked.items():
        p, r = precision_recall_at_k(model.recommend(user_id, k), relevant, k)
        precisions.append(p)
        recalls.append(r)
    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        "users_evaluated": len(precisions),
    }
