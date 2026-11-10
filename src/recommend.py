"""Recommandations pour un utilisateur.

Usage : python -m src.recommend --user 1 --k 10 [--model itemknn|svd|popularite]
Le modèle est entraîné sur les 100 000 notes, puis les films déjà notés par
l'utilisateur sont exclus.
"""
from __future__ import annotations

import argparse

import pandas as pd

from .data import load_movies, load_ratings
from .models import MODELS

DEFAULT_MODEL = "itemknn"


def recommend(user_id: int, k: int = 10, model: str = DEFAULT_MODEL,
              ratings: pd.DataFrame | None = None,
              movies: pd.DataFrame | None = None) -> pd.DataFrame:
    """Retourne un DataFrame ``rang, item_id, titre`` des ``k`` recommandations."""
    if model not in MODELS:
        raise ValueError(f"modèle inconnu : {model} (choix : {', '.join(MODELS)})")
    ratings = load_ratings() if ratings is None else ratings
    if user_id not in set(ratings["user_id"]):
        raise ValueError(f"utilisateur {user_id} absent des données")
    fitted = MODELS[model]().fit(ratings)
    items = fitted.recommend(user_id, k)
    out = pd.DataFrame({"rang": range(1, len(items) + 1), "item_id": items})
    if movies is None:
        try:
            movies = load_movies()
        except FileNotFoundError:
            movies = None
    if movies is not None:
        out = out.merge(movies[["item_id", "title"]], on="item_id", how="left")
        out = out.rename(columns={"title": "titre"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommande des films à un utilisateur MovieLens")
    parser.add_argument("--user", type=int, required=True, help="identifiant utilisateur (1–943)")
    parser.add_argument("--k", type=int, default=10, help="nombre de films")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS))
    args = parser.parse_args()
    table = recommend(args.user, args.k, args.model)
    print(f"Top {args.k} pour l'utilisateur {args.user} (modèle : {args.model})")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
