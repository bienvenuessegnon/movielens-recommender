import math

import pytest

from src.metrics import mae, precision_recall_at_k, rmse


def test_rmse_and_mae_known_values():
    assert rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(math.sqrt(4 / 3))
    assert mae([1, 2, 3], [2, 2, 1]) == pytest.approx(1.0)


def test_precision_recall_at_k():
    p, r = precision_recall_at_k([1, 2, 3, 4], {2, 4, 9}, k=4)
    assert p == pytest.approx(0.5)
    assert r == pytest.approx(2 / 3)


def test_precision_requires_relevant_items():
    with pytest.raises(ValueError):
        precision_recall_at_k([1], set(), k=1)
