import pytest

from src.recommend import recommend


def test_recommend_function_returns_ranked_table(toy_ratings):
    table = recommend(2, k=4, model="itemknn", ratings=toy_ratings, movies=None)
    assert list(table["rang"]) == [1, 2, 3, 4][: len(table)]
    assert set(table.columns) >= {"rang", "item_id"}


def test_recommend_rejects_unknown_user(toy_ratings):
    with pytest.raises(ValueError):
        recommend(12345, ratings=toy_ratings, movies=None)
