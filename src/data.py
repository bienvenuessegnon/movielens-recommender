"""Téléchargement et chargement de MovieLens 100K (GroupLens Research).

Le jeu de données n'est pas versionné : lancez ``python -m src.data`` pour
le télécharger dans ``data/ml-100k``. Conditions d'utilisation : voir le
fichier README fourni par GroupLens (usage de recherche, non commercial,
citation de Harper & Konstan, 2015).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ML_DIR = DATA_DIR / "ml-100k"

RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download(data_dir: Path = DATA_DIR, force: bool = False) -> Path:
    """Télécharge et décompresse ml-100k ; vérifie l'empreinte SHA-256."""
    target = Path(data_dir) / "ml-100k"
    if (target / "u.data").exists() and not force:
        return target
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise RuntimeError(f"Empreinte inattendue pour ml-100k.zip : {digest}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(data_dir)
    return target


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lancez d'abord : python -m src.data"
        )
    return path


def read_ratings(path: Path) -> pd.DataFrame:
    """Lit un fichier de notes au format u.data (séparateur tabulation)."""
    return pd.read_csv(_require(Path(path)), sep="\t", names=RATING_COLUMNS)


def load_ratings(ml_dir: Path = ML_DIR) -> pd.DataFrame:
    """Les 100 000 notes complètes."""
    return read_ratings(Path(ml_dir) / "u.data")


def load_movies(ml_dir: Path = ML_DIR) -> pd.DataFrame:
    """Métadonnées des 1 682 films (titre, date, genres binaires)."""
    columns = ["item_id", "title", "release_date", "video_release_date", "url"] + GENRES
    return pd.read_csv(
        _require(Path(ml_dir) / "u.item"), sep="|", names=columns, encoding="latin-1"
    )


def load_split(fold: int = 1, ml_dir: Path = ML_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage officiel 80/20 de GroupLens : u{fold}.base / u{fold}.test."""
    if fold not in range(1, 6):
        raise ValueError("fold doit être compris entre 1 et 5")
    ml_dir = Path(ml_dir)
    return read_ratings(ml_dir / f"u{fold}.base"), read_ratings(ml_dir / f"u{fold}.test")


def random_split(ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Découpage aléatoire reproductible (alternative au découpage officiel)."""
    test = ratings.sample(frac=test_size, random_state=seed)
    train = ratings.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Télécharge MovieLens 100K")
    parser.add_argument("--force", action="store_true", help="retélécharger")
    args = parser.parse_args()
    path = download(force=args.force)
    print(f"MovieLens 100K disponible dans {path}")


if __name__ == "__main__":
    main()
