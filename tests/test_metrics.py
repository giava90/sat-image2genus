import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from metrics import (
    accuracy,
    collapse,
    confusion_matrix,
    macro_f1,
    precision_recall,
    row_normalize,
    weighted_precision,
)


def test_confusion_matrix_orientation():
    cm = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 0], 3)
    # rows true, columns predicted
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 0]]


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0], 2)


def test_matches_sklearn_on_random_labels():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 4, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 4, 500))
    cm = confusion_matrix(y, p, 4)
    prec, rec = precision_recall(cm)
    assert accuracy(cm) == pytest.approx(accuracy_score(y, p))
    assert prec == pytest.approx(precision_score(y, p, average=None))
    assert rec == pytest.approx(recall_score(y, p, average=None))
    assert weighted_precision(cm) == pytest.approx(precision_score(y, p, average="weighted"))
    assert macro_f1(cm) == pytest.approx(f1_score(y, p, average="macro"))


def test_precision_nan_when_class_never_predicted():
    cm = confusion_matrix([0, 1], [0, 0], 2)
    p, r = precision_recall(cm)
    assert np.isnan(p[1]) and r[1] == 0


def test_row_normalize_rows_sum_to_one_and_empty_rows_zero():
    cm = np.array([[3, 1, 0], [0, 0, 0], [2, 2, 4]])
    c = row_normalize(cm)
    assert c[0].sum() == pytest.approx(1)
    assert c[2].sum() == pytest.approx(1)
    assert (c[1] == 0).all()


def test_collapse_to_binary():
    assert collapse([0, 1, 2, 3], [0, 0, 1, 1]).tolist() == [0, 0, 1, 1]
