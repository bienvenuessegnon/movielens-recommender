import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_ratings() -> pd.DataFrame:
    """Petit jeu synthétique : deux groupes de goûts, aucun téléchargement requis."""
    rng = np.random.default_rng(0)
    rows = []
    for user in range(1, 21):
        group = user % 2
        for item in range(1, 13):
            if rng.random() < 0.7:
                likes = (item <= 6) == (group == 0)
                rows.append((user, item, 5 if likes else 1, 0))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
