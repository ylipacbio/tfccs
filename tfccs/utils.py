import numpy as np
import sys
import json
from tfccs.constants import BASE_FEATURE_STAT_KEY


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
    return ({feature: FextractStat} , features).
    E.x., Input Json looks like
    {"BaseFeatureStat": [
        {"name": "IsHP", "min": 0, "max": 1, "mean": 0.51, "stdev": 0.3}
        {"name": "CcsSNR", "min": 1.0, "max": 3.0, "mean": 2.0, "stdev": 1.0}
    ]}
    Output looks like: (out, features), where
    out = {"IsHP": FextractStat("IsHP", 0.5, 0.3, 0.0, 1.0),
           "CCSBaseSNR": FextractStat("CCSBaseSNR", 2.0, 1.0, 1.0, 3.0)}
    features = set(["IsHP", "CCSBaseSNR"])
    """
    d = json.load(open(in_json, 'r'))
    if BASE_FEATURE_STAT_KEY not in d:
        raise ValueError("Could not find {} as Json root!".format(BASE_FEATURE_STAT_KEY))
    expected_stats = FextractStat.STAT_NAMES + ['name']
    features = []
    out = {}
    for item in d[BASE_FEATURE_STAT_KEY]:
        # sanity check 'name', 'mean', ... exists in each item
        for expected_stat in expected_stats:
            if expected_stat not in item:
                raise ValueError("Key {} must exist in {}!".format(expected_stat, item))
        # add current feature and its stat
        feature = item['name']
        features.append(feature)
        out[feature] = FextractStat(feature, float(item['mean']), float(item['stdev']),
                                    float(item['min']), float(item['max']))
    return out, set(features)


def cap_outlier_standardize(a, stat, N=4):
    """
    To standardize an input array to mostly within [-1, 1] with center at 0.
    For outliers that are too far away from center, cap at -N or N.
    --- a - 1d np array
    --- stat - FextractStat oject
    """
    a = (np.asarray(a) - stat.mean) / stat.stdev   # standardize to center 0, mostly within 0, 1
    return np.clip(a, -N, N)
