from tfccs.utils import load_fextract_stat_json, cap_outlier_standardize, FextractStat
import os.path as op
import numpy as np

ROOT_DIR = op.dirname(op.dirname(__file__))

def test_load_fextract_stat_json():
    d, features = load_fextract_stat_json(op.join(ROOT_DIR, 'data', 'fextract.stat.json'))

    expected_mean = {'BaseCoverage_FWD': 2.977}
    expected_stdev = {'BaseCoverage_FWD': 1.806}
    expected_min = {'BaseCoverage_FWD': 0.0}
    expected_max = {'BaseCoverage_FWD': 105.0}

    assert d['BaseCoverage_FWD'].feature == 'BaseCoverage_FWD'
    np.testing.assert_almost_equal(d['BaseCoverage_FWD'].mean, expected_mean['BaseCoverage_FWD'], 3)
    np.testing.assert_almost_equal(d['BaseCoverage_FWD'].stdev, expected_stdev['BaseCoverage_FWD'], 3)
    np.testing.assert_almost_equal(d['BaseCoverage_FWD'].min, expected_min['BaseCoverage_FWD'], 3)
    np.testing.assert_almost_equal(d['BaseCoverage_FWD'].max, expected_max['BaseCoverage_FWD'], 3)

def test_cap_outlier_standardize():
    # FextractStat('F1', 2, 1, 0, 4) - mean=2, std=1, min=0, max=4
    out = cap_outlier_standardize([0, 1, 2, 3, 7], FextractStat('F1', 2, 1, 0, 4), 3)
    assert list(out) == [-2.0, -1.0, 0., 1.0, 3.0]
