"""
Compute mean, stdev, min, max of trainable columns, and save to output file.
Example:
    python compute_stat_fextract.py in.fextract.csv out.fextract.stat.csv
"""

import datetime
import numpy as np
import timeit
import argparse
import csv
import json
import sys
from tfccs.constants import NO_TRAIN_FEATURES, BASE_FEATURE_STAT_KEY
from tfccs.utils import is_good_fextract_row


def compute_feature_stats(in_csv, out_json, forward_only_ccs):
    reader = csv.DictReader(open(in_csv, 'r'), delimiter=',')
    dataset = []
    t0 = datetime.datetime.now()
    features = sorted(set(reader.fieldnames).difference(NO_TRAIN_FEATURES + ["CCSBase"]))
    for idx, r in enumerate(reader):
        if not is_good_fextract_row(r, forward_only_ccs=forward_only_ccs):
            continue
        values = [r[feature] for feature in features]
        dataset.append(values)
        if idx % 500000 == 0:
            print("Processing {} rows".format(idx))
    if len(dataset) == 0:
        raise ValueError("Input fextract file {} contains empty good rows!".format(in_csv))
    t1 = datetime.datetime.now()
    print("Loaded input {} rows, time={}.".format(len(dataset), t1-t0))

    npa = np.asarray(dataset, dtype=np.float32)
    t2 = datetime.datetime.now()
    print("Created np array time={}.".format(t2-t1))
    n = len(features)
    mean_features = np.mean(npa, axis=0)
    stdev_features = np.std(npa, axis=0)
    min_features = np.min(npa, axis=0)
    max_features = np.max(npa, axis=0)

    def save_format_1():
        ret = {'mean': {features[idx]: float(mean_features[idx]) for idx in range(n)},
               'stdev': {features[idx]: float(stdev_features[idx]) for idx in range(n)},
               'min': {features[idx]: float(min_features[idx]) for idx in range(n)},
               'max': {features[idx]: float(max_features[idx]) for idx in range(n)}}
        return ret

    def save_format_2():
        # Save as {featureName: {"name":name, "mean":mean , "stdev": stdev, "min": min, "max": max}}
        ret = []
        for idx in range(0, n):
            d = {"name": features[idx], "mean": float(mean_features[idx]),
                 "stdev": float(stdev_features[idx]), "min": float(min_features[idx]),
                 "max": float(max_features[idx])}
            ret.append(d)
        return {BASE_FEATURE_STAT_KEY: ret}

    print("Dump mean, stdev, min, max of trainable variables to {}.".format(out_json))
    with open(out_json, 'w') as writer:
        json.dump(save_format_2(), writer, indent=4, sort_keys=True)


def run(args):
    compute_feature_stats(in_csv=args.in_csv, out_json=args.out_json, forward_only_ccs=not args.both_strands)
    return 0


def get_parser():
    """Set up and return argument parser."""
    desc = """Compute mean, stdev, min, max of trainable fextract features and save to output file."""
    p = argparse.ArgumentParser(desc)
    p.add_argument("in_csv", help="Input fextract csv file")
    p.add_argument("out_json", help="Output stat json file contain mean, stdev, min, max of trainable features")
    p.add_argument("--both-strands", default=False,
                   help="Default=False, only use CCS forward mapped to genome, otherwise use both stranded CCS",
                   action="store_true")
    return p


def main(args=sys.argv[1:]):
    """main"""
    run(get_parser().parse_args(args))


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))
