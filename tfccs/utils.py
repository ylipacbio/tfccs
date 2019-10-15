import numpy as np
import sys
import json


def load_fextract_npz(npz_filename):
    """
    Load fextract from a ziped file: *.npz
    """
    d = np.load(npz_filename)
    expected_keys = ['fextractinput', 'arrowqv', 'arrowqvbin8', 'ccs2genome_cigars']
    for expected_key in expected_keys:
        if expected_key not in [k for k in d.keys()]:
            raise ValueError("Key '{}' must exist in {}!".format(expected_key, npz_filename))
    nrow, ncol = d['fextractinput'].shape
    return d['fextractinput'], d['arrowqv'], d['arrowqvbin8'], d['ccs2genome_cigars'], nrow, ncol


def is_good_fextract_row(in_d, forward_only_ccs=True):
    """
    Return False if fextract row's
    1) ccs base is within 100bp end of CCS read
    2) ccs read map to reference genome in reverse strand, while forward_only_ccs is True
    3) ccs2genome cigar is in 'I=X'
    """
    dist2end = abs(int(in_d["CCSLength"]) - int(in_d["CCSPos"]))
    if 'CCSToGenomeCigar' in in_d and in_d['CCSToGenomeCigar'] not in 'I=X':
        return False
    if not forward_only_ccs:
        return dist2end > 100
    else:
        strand = in_d["CCSToGenomeStrand"]
        assert strand in 'FR'
        return dist2end > 100 and strand == 'F'


class FextractStat(object):
    STAT_NAMES = ['mean', 'stdev', 'min', 'max']  # 'q25', 'q50', 'q75', 'q999'

    def __init__(self, feature, stat_mean, stat_stdev, stat_min, stat_max):
        self.feature = feature
        self.mean = stat_mean
        self.stdev = stat_stdev
        self.min = stat_min
        self.max = stat_max

    @classmethod
    def from_json_d(cls, d, feature):
        return FextractStat(feature=feature, stat_mean=d['mean'][feature],
                            stat_stdev=d['stdev'][feature],
                            stat_min=d['min'][feature],
                            stat_max=d['max'][feature])


def load_fextract_stat_json(in_json):
    """
    Read fextract.stat.json, and load mean, stdev, min, max of trainable variables.
    return ({feature: FextractStat} , features)
    """
    d = json.load(open(in_json, 'r'))
    for expected_stat in FextractStat.STAT_NAMES:
        if expected_stat not in [k for k in d.keys()]:
            raise ValueError("Key {} must exist in {}!".format(expected_stat, in_json))
    features = d['mean'].keys()
    for expected_stat in FextractStat.STAT_NAMES:
        if d[expected_stat].keys() != features:
            raise ValueError("Features of stat {} in json file {} differ from stat 'mean', missing {}!".format(
                expected_stat, in_json, set(features).difference(set(d[expected_stat]))))

    out = {feature: FextractStat.from_json_d(d, feature) for feature in features}
    return out, set(features)


def cap_outlier_standardize(a, stat, N=4):
    """
    To standardize an input array to mostly within [-1, 1] with center at 0.
    For outliers that are too far away from center, cap at -N or N.
    --- a - 1d np array
    --- stat - FextractStat oject
    doctest:
    >>> cap_outlier_standardize([0, 1, 2, 3, 7], FextractStat('F1', 2, 1, 0, 4), 3]
    [-2, -1, 0, 1, 4]
    """
    a = (a - stat.mean) / stat.stdev   # standardize to center 0, mostly within 0, 1
    return np.clip(a, -N, N)
