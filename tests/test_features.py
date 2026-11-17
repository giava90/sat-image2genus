import numpy as np
import pytest

from data import genus_of, species_of
from features import BANDS, norm_diff, summary_features


def test_species_and_genus_from_file_name():
    assert species_of("Picea_abies_3_1234_WEFL_NLF.tif") == "Picea_abies"
    assert species_of("Betula_spec._0_77_BI_NLF.tif") == "Betula_spec."
    assert species_of("Cleared_2_99_WEFL_NLF.tif") == "Cleared"
    assert genus_of("Quercus_rubra_1_5_BI_NLF.tif") == "Quercus"


def test_norm_diff_values_and_zero_denominator():
    a = np.array([3, 0, 1], np.uint16)
    b = np.array([1, 0, 1], np.uint16)
    assert norm_diff(a, b).tolist() == pytest.approx([0.5, 0.0, 0.0])


def test_summary_features_shape_and_constant_patch():
    x = np.full((2, 12, 20, 20), 100, np.uint16)
    x[:, BANDS.index("B08")] = 300  # NIR
    f, names = summary_features(x)
    assert f.shape == (2, len(names))
    assert len(names) == (12 + 3) * 7
    row = dict(zip(names, f[0]))
    assert row["B02_mean"] == pytest.approx(100)
    assert row["B02_std"] == pytest.approx(0)
    assert row["NDVI_p50"] == pytest.approx(0.5)  # (300-100)/(300+100)
