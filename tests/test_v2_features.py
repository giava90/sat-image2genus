import numpy as np

from features import summary_features
from v2_features import centre_crop, centre_ring_features, mask_centre, raw_pixel_features, ring_mask


def _x(n=3):
    rng = np.random.default_rng(1)
    return rng.integers(1, 3000, (n, 12, 20, 20)).astype(np.float32)


def test_centre_crop_is_rows_cols_7_to_12():
    x = _x()
    c = centre_crop(x)
    assert c.shape == (3, 12, 6, 6)
    assert np.array_equal(c[:, :, 0, 0], x[:, :, 7, 7]) and np.array_equal(c[:, :, 5, 5], x[:, :, 12, 12])


def test_ring_mask_counts():
    m = ring_mask()
    assert m.sum() == 400 - 36 and not m[7:13, 7:13].any() and m[6, 6] and m[13, 13]


def test_mask_centre_only_changes_centre_and_is_dihedral_symmetric():
    x = _x()
    y = mask_centre(x, 0.0)
    assert (y[:, :, 7:13, 7:13] == 0).all()
    assert np.array_equal(y[:, :, ring_mask()], x[:, :, ring_mask()])
    assert np.array_equal(mask_centre(x[..., ::-1, :].copy())[..., ::-1, :], y)  # mask stays put under flips
    assert x[0, 0, 9, 9] != 0  # input untouched


def test_centre_ring_features_match_separate_computation():
    x = _x()
    f, names = centre_ring_features(x)
    fc, _ = summary_features(centre_crop(x))
    assert f.shape[1] == 2 * fc.shape[1] == len(names)
    assert np.allclose(f[:, : fc.shape[1]], fc)
    i = names.index("ring_B02_mean")
    assert np.isclose(f[0, i], x[0, 0][ring_mask()].mean(), rtol=1e-5)


def test_raw_pixel_features_shape_and_order():
    x = _x()
    f, names = raw_pixel_features(centre_crop(x))
    assert f.shape == (3, 12 * 36) and len(names) == 12 * 36
    assert f[1, names.index("B03_r2_c4")] == x[1, 1, 9, 11]
