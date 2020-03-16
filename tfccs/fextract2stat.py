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
from tfccs.utils import is_good_fextract_row, add_filter_args


def compute_feature_stats(in_csv, out_stat_json, min_dist2end, allowed_strands,
                          allowed_ccs2genome_cigars, min_np, max_np):
    reader = csv.DictReader(open(in_csv, 'r'), delimiter=',')
    dataset = []
    t0 = datetime.datetime.now()
    features = sorted(set(reader.fieldnames).difference(NO_TRAIN_FEATURES + ["CCSBase", "PrevBases", "NextBases"]))
    for idx, r in enumerate(reader):
        is_good = is_good_fextract_row(r, min_dist2end=min_dist2end,
                                       allowed_strands=allowed_strands,
                                       allowed_ccs2genome_cigars=allowed_ccs2genome_cigars,
                                       min_np=min_np, max_np=max_np)
        if not is_good:
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

    print("Dump mean, stdev, min, max of trainable variables to {}.".format(out_stat_json))
    with open(out_stat_json, 'w') as writer:
        json.dump(save_format_2(), writer, indent=4, sort_keys=True)


def run(args):
    if not args.out_stat_json.endswith('.stat.json'):
        raise ValueError("Output stat json file must ends with .stat.json! {}".format(args.out_stat_json))
    compute_feature_stats(in_csv=args.in_csv, out_stat_json=args.out_stat_json,
                          min_dist2end=args.min_dist2end, allowed_strands=args.allowed_strands,
                          allowed_ccs2genome_cigars=args.allowed_cigars,
                          min_np=args.min_np, max_np=args.max_np)
    return 0


def get_parser():
    """Set up and return argument parser."""
    desc = """Compute mean, stdev, min, max of trainable fextract features and save to output file."""
    p = argparse.ArgumentParser(desc)
    p.add_argument("in_csv", help="Input fextract csv file")
    p.add_argument("out_stat_json", help="Output stat json file contain mean, stdev, min, max of trainable features")
    return add_filter_args(p)


def main(args=sys.argv[1:]):
    """main"""
    run(get_parser().parse_args(args))


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))
