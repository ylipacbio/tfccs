import numpy as np
import sys
import json
import csv
import os.path as op
import logging
import subprocess
from tfccs.constants import (BASE_FEATURE_STAT_KEY, MIN_DIST2END, ALLOWED_STRANDS,
                             ALLOWED_CIGARS, MIN_NUMPASSES, MAX_NUMPASSES)

FORMATTER = op.basename(__file__) + ':%(levelname)s:'+'%(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMATTER)
log = logging.getLogger(__name__)


def execute(cmd):
    print("CMD: " + cmd)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        raise RuntimeError(f"CMD failed: {cmd}, ret={ret}!")


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


def is_good_fextract_row(in_d, min_dist2end=MIN_DIST2END, allowed_strands=ALLOWED_STRANDS,
                         allowed_ccs2genome_cigars=ALLOWED_CIGARS, min_np=MIN_NUMPASSES,
                         max_np=MAX_NUMPASSES):
    """
    Return False if fextract row's
    1) ccs base is within 100bp end of CCS read
    2) ccs read map to reference genome in reverse strand,
    3) ccs2genome cigar is in 'I=X'
    """
    if 'CCSToGenomeCigar' in in_d and in_d['CCSToGenomeCigar'] not in allowed_ccs2genome_cigars:
        return False
    dist2end = abs(int(in_d["CCSLength"]) - int(in_d["CCSPos"]))
    if dist2end < min_dist2end:
        return False
    if 'CCSToGenomeStrand' in in_d and in_d["CCSToGenomeStrand"] not in allowed_strands:
        return False
    np = None
    if 'BaseCoverage' in in_d:
        np = int(in_d['BaseCoverage'])
    if 'BaseCoverage_FWD' in in_d and 'BaseCoverage_REV' in in_d:
        np = int(in_d['BaseCoverage_FWD']) + int(in_d['BaseCoverage_REV'])
    if np and (np < min_np or np > max_np):
        return False
    return True


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


def read_fextract(filename, is_good_fextract_row_f, return_index):
    """
        filename --- Input fextract.csv
        is_good_fextract_row_f --- a function which tells if a row is good
        return_index --- True: return indices of good rows in total rows.
            header is not considered a row and the first row has an index of 0.
    """
#    reader = csv.DictReader(open(filename, 'r'), delimiter=',')
#    for index, r in enumerate(reader):
#        res = 'good' if is_good_fextract_row_f(r) else 'bad'
#        log.debug("{index} -> {movie},{zmw},{cigar},{pd} is {res}".format(
#            index=index, movie=r["Movie"], zmw=r["HoleNumber"],
#            cigar=r["CCSToGenomeCigar"], pd=r["CcsToGenomePrevDeletions"], res=res))

    reader = csv.DictReader(open(filename, 'r'), delimiter=',')
    if not return_index:
        return [r for r in reader if is_good_fextract_row_f(r)]
    else:
        return [idx for idx, r in enumerate(reader) if is_good_fextract_row_f(r)]


def cap_outlier_standardize(a, stat, N=4):
    """
    To standardize an input array to mostly within [-1, 1] with center at 0.
    For outliers that are too far away from center, cap at -N or N.
    --- a - 1d np array
    --- stat - FextractStat oject
    """
    a = (np.asarray(a) - stat.mean) / stat.stdev   # standardize to center 0, mostly within 0, 1
    return np.clip(a, -N, N)


def read_rows_of_indices(filename, indices):
    """
    Read a txt/csv file, and return a list of rows whose indices are in indices.
    """
    indices = set(indices)
    ret = []
    with open(filename, 'r') as reader:
        for index, row in enumerate(reader):
            if index in indices:
                ret.append(row)
    return ret


def add_filter_args(p):
    p.add_argument("--min-dist2end", default=MIN_DIST2END, type=int,
                   help="Ignore a base if its distance to either ends is less than min_dist2end bp")
    p.add_argument("--allowed-strands", default=ALLOWED_STRANDS, choices=["F", "R", "FR"],
                   help=("Ignore a base if it maps to genome in a not-allowed strand. " +
                         "F - forward strand, R - reverse strand, FR - both strands"))
    p.add_argument("--allowed-cigars", default=ALLOWED_CIGARS,
                   help="Ignore a base if it maps to genome with a not-allowed cigar")
    p.add_argument("--min-np", default=MIN_NUMPASSES, type=int,
                   help="Ignore a base if its NumPasses (BaseCoverage) is less than min-np")
    p.add_argument("--max-np", default=MAX_NUMPASSES, type=int,
                   help="Ignore a base if its NumPasses (BaseCoverage) is greater than min-np")
    return p


def write_to_script(cmds, filename):
    if op.exists(filename):
        log.info(f"Overriding {filename}!")
    with open(filename, 'w') as writer:
        writer.write('set -vex -o pipefail\n')
        for cmd in cmds:
            writer.write(cmd + '\n')


def mkdir(path):
    execute(f"mkdir -p {path}")


def encode_kmer(kmer):
    GAP = '-'
    # encode ACGT^L as 0,1,2,3 * 4^k
    # -> left=mostSignif ... right=leastSignif ->
    benc = {"A": 0, "C": 1, "G": 2, "T": 3, GAP: 4}
    # A -> 0
    # AA -> 5
    # AAA -> 25
    l = len(kmer)
    if l > 10:
        raise ValueError("Not support encode of kmer > 10")
    enc = 0
    base = 1
    for i in range(l):
        enc += base * benc[kmer[i]]
        base *= 5
    return enc


def decode_kmer(enc, kmlen):
    GAP = '-'
    bdec = {0: "A", 1: "C", 2: "G", 3: "T", 4: GAP}
    dec = []
    for i in range(kmlen):
        dec.append(bdec[(enc % 5)])
        enc = enc // 5
    return("".join(dec))
