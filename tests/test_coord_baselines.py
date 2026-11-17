import numpy as np

from coord_baselines import knn_proba


def test_1nn_returns_label_of_nearest():
    xy = np.array([[0, 0], [10, 0], [0, 10]])
    y = np.array([0, 1, 2])
    p = knn_proba(xy, y, [[1, 1], [9, 1], [0, 8]], k=1, n_classes=3)
    assert p.argmax(1).tolist() == [0, 1, 2]
    assert np.allclose(p.sum(1), 1)


def test_5nn_vote_shares():
    xy = np.array([[i, 0] for i in range(10)])
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    p = knn_proba(xy, y, [[1.9, 0]], k=5, n_classes=3)  # nearest: x=2,1,3,0,4 -> labels 0,0,1,0,1
    assert np.allclose(p, [[0.6, 0.4, 0.0]])
