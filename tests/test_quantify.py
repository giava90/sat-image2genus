import numpy as np
import pytest

from quantify import ac, all_estimates, cc, cond_matrix, em, pac, pcc, sample_region, soft_matrix, stand_counts


def test_cc_and_pcc():
    assert np.allclose(cc([0, 0, 1, 3], 4), [0.5, 0.25, 0, 0.25])
    assert np.allclose(pcc([[1, 0], [0.5, 0.5]]), [0.75, 0.25])


def test_ac_recovers_truth_with_exact_matrix():
    C = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.05, 0.15, 0.8]])
    pi = np.array([0.6, 0.3, 0.1])
    assert np.allclose(ac(pi @ C, C), pi)


def test_ac_clips_and_renormalises():
    C = np.array([[0.9, 0.1], [0.3, 0.7]])
    x = ac([0.05, 0.95], C)  # implies a negative share for class 0
    assert x.min() >= 0 and np.isclose(x.sum(), 1) and x[0] == 0


def test_pac_recovers_truth_with_exact_soft_matrix():
    M = np.array([[0.7, 0.2, 0.1], [0.25, 0.6, 0.15], [0.1, 0.2, 0.7]])
    pi = np.array([0.2, 0.05, 0.75])
    assert np.allclose(pac(pi @ M, M), pi)


def test_cond_and_soft_matrix():
    y = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 1])
    assert np.allclose(cond_matrix(y, yp, 3), [[0.5, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    P = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.0, 1.0]])
    assert np.allclose(soft_matrix(y, P, 2), [[0.65, 0.35], [0.1, 0.9]])


def _gauss_posteriors(x, prior, mus):
    lik = np.exp(-0.5 * (x[:, None] - np.asarray(mus)[None]) ** 2)
    post = lik * prior
    return post / post.sum(1, keepdims=True)


def test_em_recovers_prior_shift_with_calibrated_posteriors():
    rng = np.random.default_rng(0)
    mus = [-1.5, 0.0, 1.5]
    train_prior = np.array([1 / 3, 1 / 3, 1 / 3])
    true = np.array([0.7, 0.25, 0.05])
    y = rng.choice(3, 40000, p=true)
    P = _gauss_posteriors(rng.normal(np.asarray(mus)[y], 1.0), train_prior, mus)
    est, it = em(P, train_prior)
    assert np.abs(est - true).max() < 0.02
    assert np.abs(pcc(P) - true).max() > 0.05  # the naive mean is visibly biased here
    assert it < 1000


def test_em_no_shift_stays_close():
    rng = np.random.default_rng(1)
    mus = [-1.0, 1.0]
    prior = np.array([0.3, 0.7])
    y = rng.choice(2, 40000, p=prior)
    P = _gauss_posteriors(rng.normal(np.asarray(mus)[y], 1.0), prior, mus)
    assert np.abs(em(P, prior)[0] - prior).max() < 0.02


def test_stand_counts_largest_remainder():
    assert stand_counts([0.6, 0.25, 0.1, 0.05], 5000).tolist() == [3000, 1250, 500, 250]
    c = stand_counts([1 / 3, 1 / 3, 1 / 3], 100)
    assert c.sum() == 100 and set(c) == {33, 34}
    with pytest.raises(ValueError):
        stand_counts([0.5, 0.6], 10)


def test_sample_region_counts_and_classes():
    y = np.array([0] * 10 + [1] * 5 + [2] * 3)
    idx = sample_region(y, [0.5, 0.3, 0.2], 1000, np.random.default_rng(0))
    assert len(idx) == 1000
    assert np.bincount(y[idx], minlength=3).tolist() == [500, 300, 200]
    with pytest.raises(ValueError):
        sample_region(np.array([0, 0, 1]), [0.5, 0.25, 0.25], 100, np.random.default_rng(0))


def test_all_estimates_keys_and_simplex():
    rng = np.random.default_rng(2)
    P = rng.dirichlet(np.ones(4), 200)
    y = rng.integers(0, 4, 200)
    est = all_estimates(P.argmax(1), P, 4, cond_matrix(y, P.argmax(1), 4), soft_matrix(y, P, 4), np.full(4, 0.25))
    assert set(est) == {"cc", "ac", "pcc", "pac", "em"}
    for v in est.values():
        assert v.min() >= 0 and np.isclose(v.sum(), 1)
