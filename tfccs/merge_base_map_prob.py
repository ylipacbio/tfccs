import datetime
import numpy as np
import timeit
import argparse
import csv
import sys
import json
import logging
import os.path as op
from tfccs.constants import NO_TRAIN_FEATURES, ORDERED_FEATURES_KEY, BASE_MAP_PROBABILITY_KEY

FORMATTER = op.basename(__file__) + ':%(levelname)s:'+'%(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMATTER)
log = logging.getLogger(__name__)


def merge_base_map_prob_json(sampling_json, population_json, out_json):
    # e.g., sampling_d = {"Sampling": {
    #        "SequenceMatch": xx,
    #        "Insertion": xx,
    #        "Substitution": xx,
    #        "PreviousIsDeletion": xx
    #    }}
    sampling_d = json.load(open(sampling_json, 'r'))
    population_d = json.load(open(population_json, 'r'))

    if BASE_MAP_PROBABILITY_KEY not in sampling_d:
        raise ValueError(f"{BASE_MAP_PROBABILITY_KEY} does not exists as key in {sampling_json}")
    if 'Sampling' not in sampling_d[BASE_MAP_PROBABILITY_KEY]:
        raise ValueError(f"'Sampling' does not exists as key under {BASE_MAP_PROBABILITY_KE} in {sampling_json}")
    for key in ['SequenceMatch', 'Insertion', 'Substitution', 'PreviousIsDeletion']:
        if key not in sampling_d[BASE_MAP_PROBABILITY_KEY]['Sampling']:
            raise ValueError(f"{key} does not exist in {sampling_json}")

    if BASE_MAP_PROBABILITY_KEY not in population_d:
        raise ValueError(f"{BASE_MAP_PROBABILITY_KEY} does not exists as key in {population_json}")
    if 'Population' not in population_d[BASE_MAP_PROBABILITY_KEY]:
        raise ValueError(f"'Population' does not exists as key under {BASE_MAP_PROBABILITY_KE} in {population_json}")
    for key in ['SequenceMatch', 'Insertion', 'Substitution', 'PreviousIsDeletion']:
        if key not in population_d[BASE_MAP_PROBABILITY_KEY]['Population']:
            raise ValueError(f"{key} does not exist in {population_json}")
    merge_d = sampling_d[BASE_MAP_PROBABILITY_KEY]
    merge_d.update(population_d[BASE_MAP_PROBABILITY_KEY])
    print(merge_d)
    with open(out_json, 'w') as writer:
        json.dump(sampling_d, writer, indent=4, sort_keys=True)


def run(args):
    sampling_json = args.sampling_base_map_prob_json
    population_json = args.population_base_map_prob_json
    out_json = args.out_json
    merge_base_map_prob_json(sampling_json, population_json, out_json)


def get_parser():
    """Set up and return argument parser."""
    desc = """Merge sampling and population base map probabilty and generate output json"""
    p = argparse.ArgumentParser(desc)
    p.add_argument("sampling_base_map_prob_json", help="Sampling space base map probabilty json")
    p.add_argument("population_base_map_prob_json", help="Population space base map probabilty json")
    p.add_argument("out_json", help="Output json with both Sampling and Population space base map probability")
    return p


def main(args=sys.argv[1:]):
    """main"""
    run(get_parser().parse_args(args))


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))
