import numpy as np

from blocks import block_bootstrap, block_ids, blocked_split


def _pts(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 40_000, n), rng.uniform(0, 40_000, n)


def test_block_ids_grid():
    b = block_ids([0, 1999, 2000, 0, -1], [0, 0, 0, 2000, 0])
    assert b[0] == b[1] and b[0] != b[2] and b[0] != b[3] and b[0] != b[4]


def test_blocked_split_blocks_not_shared_and_fractions():
    x, y = _pts()
    s = blocked_split(x, y, 0.2, 0.1, seed=1)
    b = block_ids(x, y)
    for blk in np.unique(b):
        labs = set(s[b == blk]) - {"dropped"}
        assert len(labs) <= 1  # a block is in one set only ('dropped' patches are former training patches)
        if "dropped" in set(s[b == blk]):
            assert labs <= {"train"}
    assert abs((s == "test").mean() - 0.2) < 0.03
    assert abs((s == "val").mean() - 0.1) < 0.03


def test_blocked_split_buffer():
    x, y = _pts()
    s = blocked_split(x, y, seed=2, buffer=200)
    held = np.isin(s, ["test", "val"])
    tr = s == "train"
    d = np.max(np.abs(np.stack([x[tr][:, None] - x[held][None], y[tr][:, None] - y[held][None]])), axis=0).min(1)
    assert (d >= 200).all()
    dr = s == "dropped"
    dd = np.max(np.abs(np.stack([x[dr][:, None] - x[held][None], y[dr][:, None] - y[held][None]])), axis=0).min(1)
    assert (dd < 200).all() and dr.sum() > 0


def test_blocked_split_seed_changes_draw():
    x, y = _pts()
    assert not np.array_equal(blocked_split(x, y, seed=0), blocked_split(x, y, seed=1))


def test_block_bootstrap_resamples_whole_groups():
    g = np.array([0, 0, 0, 1, 2, 2])
    for idx in block_bootstrap(g, 50, seed=0):
        vals, c = np.unique(g[idx], return_counts=True)
        full = {0: 3, 1: 1, 2: 2}
        assert all(cc % full[v] == 0 for v, cc in zip(vals, c))


def test_block_bootstrap_singletons_mean_is_unbiased():
    g = np.arange(200)
    v = np.random.default_rng(0).random(200)
    m = np.mean([v[i].mean() for i in block_bootstrap(g, 2000, seed=3)])
    assert abs(m - v.mean()) < 0.005
