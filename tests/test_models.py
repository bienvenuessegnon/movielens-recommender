import numpy as np
import pytest

from src.data import random_split
from src.models import ItemKNN, MatrixFactorization, PopularityBaseline

MODELS = [
    PopularityBaseline(),
    ItemKNN(k=5, shrinkage=1),
    MatrixFactorization(n_factors=4, n_epochs=30, lr=0.05, reg=0.01, seed=0),
]


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_predictions_are_bounded(model, toy_ratings):
    model.fit(toy_ratings)
    pred = model.predict([1, 2, 999], [1, 2, 999])   # inclut un utilisateur et un film inconnus
    assert pred.shape == (3,)
    assert np.all((pred >= 1) & (pred <= 5))


@pytest.mark.parametrize("model", MODELS, ids=lambda m: type(m).__name__)
def test_recommend_excludes_seen_items(model, toy_ratings):
    model.fit(toy_ratings)
    recs = model.recommend(1, k=3)
    seen = set(toy_ratings.loc[toy_ratings.user_id == 1, "item_id"])
    assert len(recs) == len(set(recs)) <= 3
    assert not set(recs) & seen


def test_collaborative_models_learn_tastes(toy_ratings):
    train, test = random_split(toy_ratings, test_size=0.2, seed=1)
    for model in (ItemKNN(k=5, shrinkage=1),
                  MatrixFactorization(n_factors=4, n_epochs=40, lr=0.05, reg=0.01, seed=0)):
        model.fit(train)
        pred = model.predict(test.user_id, test.item_id)
        baseline = np.full(len(test), train.rating.mean())
        assert np.mean((pred - test.rating) ** 2) < np.mean((baseline - test.rating) ** 2)


def test_random_split_is_reproducible(toy_ratings):
    a_train, a_test = random_split(toy_ratings, seed=7)
    b_train, b_test = random_split(toy_ratings, seed=7)
    assert a_test.equals(b_test) and a_train.equals(b_train)
    assert len(a_train) + len(a_test) == len(toy_ratings)
