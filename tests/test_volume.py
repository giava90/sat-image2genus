import numpy as np
import pytest

from volume import (
    adjust,
    bootstrap_volume_errors,
    count_estimate,
    relative_error,
    true_areas,
    volume_errors,
)

GS = np.array([400.0, 300.0, 350.0, 300.0])


def test_true_areas_validates_shares():
    assert true_areas([0.5, 0.5], 100).tolist() == [50, 50]
    with pytest.raises(ValueError):
        true_areas([0.5, 0.6], 100)


def test_perfect_classifier_has_zero_error():
    a = true_areas([0.25] * 4, 1000)
    e = volume_errors(a, np.eye(4), np.eye(4), GS)
    for k in ("count", "adjusted"):
        assert np.allclose(e[k]["per_genus"], 0)
        assert e[k]["total"] == pytest.approx(0)


def test_count_estimate_conserves_area():
    rng = np.random.default_rng(0)
    c = rng.random((4, 4))
    c /= c.sum(1, keepdims=True)
    a = true_areas([0.1, 0.2, 0.3, 0.4], 500)
    assert count_estimate(a, c).sum() == pytest.approx(500)


def test_count_estimate_hand_example():
    # 100 ha of class 0, 10% labelled as class 1; 100 ha of class 1, all correct.
    c = np.array([[0.9, 0.1], [0.0, 1.0]])
    assert count_estimate([100, 100], c).tolist() == pytest.approx([90, 110])


def test_adjustment_recovers_truth_when_val_equals_test():
    c = np.array([[0.8, 0.1, 0.05, 0.05], [0.1, 0.7, 0.1, 0.1], [0.05, 0.05, 0.8, 0.1], [0.0, 0.1, 0.2, 0.7]])
    a = true_areas([0.4, 0.1, 0.2, 0.3], 1000)
    est = adjust(count_estimate(a, c), c)
    assert est == pytest.approx(a)


def test_adjustment_clips_negatives_and_keeps_total():
    cval = np.array([[0.9, 0.1], [0.1, 0.9]])
    counted = np.array([0.0, 100.0])  # inverse would give a negative class-0 area
    x = adjust(counted, cval)
    assert (x >= 0).all()
    assert x.sum() == pytest.approx(100)


def test_relative_error_and_total_volume_weighting():
    assert relative_error([110, 90], [100, 100]).tolist() == pytest.approx([0.1, -0.1])
    # 100 ha spruce all called pine: total volume falls from 400*100 to 300*100.
    c = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    e = volume_errors([100, 0, 0, 0], c, np.eye(4), GS)
    assert e["count"]["total"] == pytest.approx(-0.25)
    assert e["count"]["per_genus"][0] == pytest.approx(-1.0)


def test_per_genus_error_does_not_depend_on_growing_stock():
    c = np.array([[0.8, 0.2, 0, 0], [0.1, 0.9, 0, 0], [0, 0, 0.7, 0.3], [0, 0, 0.2, 0.8]])
    a = true_areas([0.25] * 4, 100)
    e1 = volume_errors(a, c, np.eye(4), GS)
    e2 = volume_errors(a, c, np.eye(4), GS * np.array([2, 0.5, 3, 1]))
    assert e1["count"]["per_genus"] == pytest.approx(e2["count"]["per_genus"])


def test_bootstrap_shapes_and_centre():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 800)
    p = np.where(rng.random(800) < 0.8, y, rng.integers(0, 4, 800))
    from metrics import confusion_matrix, row_normalize

    cval = row_normalize(confusion_matrix(y, p, 4))
    r = bootstrap_volume_errors(y, p, cval, [0.25] * 4, GS, 4, n_boot=200)
    assert r["count"]["per_genus"].shape == (200, 4)
    assert r["adjusted"]["total"].shape == (200,)
    # with val == test the adjusted point estimate is exact
    assert np.allclose(r["point"]["adjusted"]["per_genus"], 0, atol=1e-9)
    # bootstrap of the adjusted estimate is centred near zero
    assert abs(np.median(r["adjusted"]["per_genus"])) < 0.05
