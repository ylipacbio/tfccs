from tfccs.utils import load_fextract_stat_json
import os.path as op

ROOT_DIR = op.dirname(op.dirname(__file__))

def test_load_fextract_stat_json():
    d, features = load_fextract_stat_json(op.join(ROOT_DIR, 'data', 'fextract.stat.json'))

    expected_mean = {'F1': 1.0 , 'F2':2.0}
    expected_stdev = {'F1': 3.0 , 'F2':4.0}
    expected_min = {'F1': 5.0 , 'F2':4.0}
    expected_max = {'F1': 6.0 , 'F2':8.0}

    assert d['F1'].feature == 'F1'
    assert d['F1'].mean == expected_mean['F1']
    assert d['F1'].stdev == expected_stdev['F1']
    assert d['F1'].min == expected_min['F1']
    assert d['F1'].max == expected_max['F1']

    assert d['F2'].feature == 'F2'
    assert d['F2'].mean == expected_mean['F2']
    assert d['F2'].stdev == expected_stdev['F2']
    assert d['F2'].min == expected_min['F2']
    assert d['F2'].max == expected_max['F2']

    assert features == set(['F1', 'F2'])
