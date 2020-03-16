"""
python fextract2numpy.py /home/UNIXHOME/yli/for_the_people/zdz/chunk-0.fextract.v3.csv output

Convert each row of fextract.csv to features to use in NN training.
1) Exclude NO_TRAIN_FEATURES, and DUPLICATED_FEATURES
2) One-hot encode each CCSBase from ACGT to 1000, 0100, 0010, 0001

Compression rate: 10 fold, 700MB fextract.csv --> 70MB npz
Runtime: 2 minutes
"""
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
from tfccs.utils import load_fextract_stat_json, is_good_fextract_row, cap_outlier_standardize, add_filter_args

FORMATTER = op.basename(__file__) + ':%(levelname)s:'+'%(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMATTER)
log = logging.getLogger(__name__)


DUPLICATED_FEATURES = ["CCSBaseSNR"]  # duplication of SNR_A/SNR_C/SNR_G/SNR_T


def arrowqv2bin8(arrowqv):
    """Quantize QV bins:
0-9
10-19
20-29
30-39
40-49
50-59
60-69
70-79+
"""
    arrowqvfloor = int(arrowqv/10.0)
    if arrowqvfloor > 7:
        arrowqvfloor = 7
    myout = [0.0]*8
    myout[arrowqvfloor] = 1.0
    return(myout)


def one_hot_base(base):
    assert base in 'ACGTacgt'
    return {"CCSBaseA": 1 if base in 'Aa' else 0,
            "CCSBaseC": 1 if base in 'Cc' else 0,
            "CCSBaseG": 1 if base in 'Gg' else 0,
            "CCSBaseT": 1 if base in 'Tt' else 0}


def one_hot_base_or_gap(base_or_gap, prefix):
    assert base_or_gap in 'ACGTacgt-'
    return {prefix + "A": 1 if base_or_gap in 'Aa' else 0,
            prefix + "C": 1 if base_or_gap in 'Cc' else 0,
            prefix + "G": 1 if base_or_gap in 'Gg' else 0,
            prefix + "T": 1 if base_or_gap in 'Tt' else 0,
            prefix + "GAP": 1 if base_or_gap == '-' else 0}


def cigar_index_in_one_hot(cigar):
    # Must match one_hot_to_cigar
    d = {'=': 0, 'I': 1, 'X': 2, 'D': 3}
    return d[cigar]


def one_hot_encode_cigar(cigar):
    assert cigar in '=IDX'
    ret = [0, 0, 0, 0]
    ret[cigar_index_in_one_hot(cigar)] = 1
    return ret


def one_hot_to_cigar(one_hot):
    # Must match cigar_index_in_one_hot
    assert len(one_hot) == 4
    idx = np.argmax(one_hot)
    d = {0: '=', 1: 'I', 2: 'X', 3: 'D'}
    return d[idx]


def ccs2genome_cigar_counting_prev_dels(current_cigar, num_prev_deletions):
    """
    Return one-hot encode of ccs2genome cigar counting previous deletions.
    1) if num_prev_deletions is zero, return current_cigar
    2) otherwise, return 'D' regardless of how this CCS base map to genome
    """
    num_prev_deletions = int(num_prev_deletions)
    if num_prev_deletions != 0:
        return one_hot_encode_cigar('D')
    assert current_cigar in '=IX'
    ret = one_hot_encode_cigar(current_cigar)
    return ret


def convert_fextract_row(input_d):
    """
    1) Remove NO_TRAIN_FEATURES and DUPLICATED_FEATURES
    2) One-hot encode CCSBase, aka replace 'CCSBase' by 'CCSBaseA', 'CCSBaseC', 'CCSBaseG', 'CCSBaseT'
       Similarly, encode PrevBases and NextBases
    3) Merge 'CCSToGenomeCigar' and 'CcsToGenomePrevDeletions', and report as one-hot encode
    """
    base_d = one_hot_base(input_d['CCSBase'])
    # Input PrevBases has two bases, '{PrevBase0}{PrevBase1}'
    # Input NextBases has two bases, '{NextBase0}{NextBase1}'
    # CCS read local sequence context: PrevBase0, PrevBase1, CCSBase, NextBase0, NextBase1
    if 'PrevBases' in input_d:
        base_d.update(one_hot_base_or_gap(input_d['PrevBases'][0], "PrevBase0"))
        base_d.update(one_hot_base_or_gap(input_d['PrevBases'][1], "PrevBase1"))
    if 'NextBases' in input_d:
        base_d.update(one_hot_base_or_gap(input_d['NextBases'][0], "NextBase0"))
        base_d.update(one_hot_base_or_gap(input_d['NextBases'][1], "NextBase1"))

    arrow_qv = int(input_d["ArrowQv"])
    ccs2genome_cigar = ccs2genome_cigar_counting_prev_dels(input_d['CCSToGenomeCigar'],
                                                           input_d['CcsToGenomePrevDeletions'])
    input_d.update(base_d)
    for feature in NO_TRAIN_FEATURES + ['CCSBase', 'PrevBases', 'NextBases'] + DUPLICATED_FEATURES:
        input_d.pop(feature, None)
    return input_d, arrow_qv, ccs2genome_cigar


def fextract2numpy(fextract_filename, output_prefix,
                   min_dist2end, allowed_strands,
                   allowed_ccs2genome_cigars, min_np, max_np,
                   num_train_rows, stat_json):
    if num_train_rows == 0:
        log.info("Will convert all qualified rows to output npz!")

    reader = csv.DictReader(open(fextract_filename, 'r'), delimiter=',')
    raw_reader = open(fextract_filename, 'r')
    header = next(raw_reader)  # Skip header
    features = reader.fieldnames

    # If fextract.stat.json was provided as input, check features in csv and stat.json MATCH
    stat_d, stat_features = None, None
    if stat_json is not None:
        stat_d, stat_features = load_fextract_stat_json(stat_json)
        trainable_features = set(features).difference(set(NO_TRAIN_FEATURES + ['CCSBase', 'PrevBases', 'NextBases']))
        if trainable_features != stat_features:
            raise ValueError("Features in csv and stat.json differ!\n" +
                             "Unique features in csv: {}\nUnique features in stat.json: {}\n".format(
                                 trainable_features.difference(stat_features), stat_features.difference(trainable_features)))

    dataset = []
    arrow_qvs = []
    ccs2genome_cigars = []
    idx = 0
    out_features = None

    t0 = datetime.datetime.now()
    raw_train_writer = open(output_prefix + '.fextract.csv', 'w')
    raw_train_writer.write(header)
    for r, raw_r in zip(reader, raw_reader):
        is_good = is_good_fextract_row(r, min_dist2end=min_dist2end,
                                       allowed_strands=allowed_strands,
                                       allowed_ccs2genome_cigars=allowed_ccs2genome_cigars,
                                       min_np=min_np, max_np=max_np)
        if not is_good:
            continue
        out_r, arrow_qv, ccs2genome_cigar = convert_fextract_row(r)
        if out_features is None:
            out_features = list(out_r.keys())
        else:
            if out_features != list(out_r.keys()):
                raise ValueError("Could not convert row {} to consistent output features!".format(r[0:50]))
        if (num_train_rows == 0 or idx < num_train_rows):
            raw_train_writer.write(raw_r)
        else:
            break
        new_r = np.fromiter(out_r.values(), dtype=np.float32)
        dataset.append(new_r)
        arrow_qvs.append(arrow_qv)
        ccs2genome_cigars.append(ccs2genome_cigar)
        if idx % 500000 == 0:
            print("Processing {} rows".format(idx))
        idx += 1
    raw_train_writer.close()

    if len(dataset) == 0:
        raise ValueError("Output empty train data!")
    if num_train_rows > 0 and len(dataset) < num_train_rows:
        raise ValueError(f"Collected {len(dataset)} training data points, less than required {num_train_rows}!")
    assert len(arrow_qvs) == len(ccs2genome_cigars)
    if num_train_rows == 0:
        num_train_rows = len(arrow_qvs)

    t1 = datetime.datetime.now()
    print("Loaded input {} rows, time={}.".format(len(dataset), t1-t0))

    npa = np.asarray(dataset, dtype=np.float32)  # column: out_features, row: CCS bases
    arrow_qvs = np.asarray(arrow_qvs, dtype=np.float32)
    arrow_qv_bins = np.asarray([arrowqv2bin8(qv) for qv in arrow_qvs], dtype=np.float32)
    ccs2genome_cigars = np.asarray(ccs2genome_cigars, dtype=np.float32)
    t2 = datetime.datetime.now()
    print("Created np array time={}.".format(t2-t1))

    # If fextract.stat.json provided and check, apply normalization to each column.
    if stat_d is not None and stat_features is not None:
        npat = npa.T  # npat 2d array, row: out_features, column: CCS bases
        for stat_feature in stat_features.difference(DUPLICATED_FEATURES):
            if stat_feature not in out_features:
                raise ValueError("Feature {} exists in {} file but not in {}!".format(
                    stat_feature, stat_json, fextract_filename))
            idx = out_features.index(stat_feature)

            def standardize_func(a):
                return cap_outlier_standardize(a, stat_d[stat_feature])  # stat_d: feature -> FextractStat
            tmp = standardize_func(npat[idx])
            npat[idx] = tmp
        npa = npat.T  # npa 2d array, row: CCS bases, column: out_features

    # Write output header as txt
    out_header_filename = output_prefix + ".header"
    with open(out_header_filename, 'w') as writer:
        writer.write(','.join(out_features))

    # Write ordered output features as json
    out_ordered_features_json_filename = output_prefix + ".features.order.json"
    with open(out_ordered_features_json_filename, 'w') as writer:
        json.dump({ORDERED_FEATURES_KEY: out_features}, writer, sort_keys=True, indent=4)
    print("Created header file {}, time={}.".format(out_ordered_features_json_filename, t2-t1))

    # Write output npz
    def zipsave(out_filename, start_row, end_row):
        np.savez_compressed(out_filename,
                            fextractinput=npa[start_row:end_row],
                            arrowqv=arrow_qvs[start_row:end_row],
                            arrowqvbin8=arrow_qv_bins[start_row:end_row],
                            ccs2genome_cigars=ccs2genome_cigars[start_row:end_row])

    out_train_filename = output_prefix + ".npz"
    zipsave(out_train_filename, 0, num_train_rows)
    t3 = datetime.datetime.now()
    print("Dumped {} rows of training data, time={}".format(num_train_rows, t3-t2))

    def base_map_probability(cc2genome_cigars, start_row, end_row):
        # Return fraction of bases in 'I=XD' classes
        c = ccs2genome_cigars[start_row:end_row]
        n = len(c)
        a = sum(c)
        assert len(a) == 4, "Must have exactly 4 output classes each representing a cigar operation"
        # see one_hot_encode_cigar, order '=IDX': {0, 1, 2, 3}
        out_probs = {"Sampling": {
            "SequenceMatch": a[cigar_index_in_one_hot('=')] / n,
            "Insertion": a[cigar_index_in_one_hot('I')] / n,
            "Substitution": a[cigar_index_in_one_hot('X')] / n,
            "PreviousIsDeletion": a[cigar_index_in_one_hot('D')] / n
        }}
        return out_probs

    # Write probabilty of base map '=IXD' in Sampling spaces.
    out_base_map_prob_json = output_prefix + '.base_map_probability.json'
    out_probs = base_map_probability(ccs2genome_cigars, 0, num_train_rows)
    print("Dump Base Map probability {} to: {}".format(out_probs, out_base_map_prob_json))
    with open(out_base_map_prob_json, 'w') as writer:
        json.dump({BASE_MAP_PROBABILITY_KEY: out_probs}, writer, sort_keys=True, indent=4)


def run(args):
    if not args.stat_json:
        print("WARNING! No fextract.stat.json file provided, will NOT standardize features!")
    else:
        if not args.stat_json.endswith('.stat.json'):
            raise ValueError("Input --stat-json file {} must ends with '.stat.json'")

    fextract2numpy(fextract_filename=args.fextract_filename, output_prefix=args.output_prefix,
                   num_train_rows=args.num_train_rows, min_dist2end=args.min_dist2end,
                   allowed_strands=args.allowed_strands, allowed_ccs2genome_cigars=args.allowed_cigars,
                   min_np=args.min_np, max_np=args.max_np, stat_json=args.stat_json)
    return 0


def get_parser():
    """Set up and return argument parser."""
    desc = """Convert fextract csv file to zipped numpy file - ${output_prefix}.npz with N rows\n"""
    p = argparse.ArgumentParser(desc)
    p.add_argument("fextract_filename", help="fextract csv file")
    p.add_argument("output_prefix", help="Output prefix")
    p.add_argument("--stat-json", default=None,
                   help=("If set, standardize features using mean/stdev/min/max from stat.json. " +
                         "otherwise, do NOT standarize features"))
    p.add_argument("--num-train-rows", type=int, default=0, help="Number of training rows, 0 means no limitation")
    return add_filter_args(p)


def main(args=sys.argv[1:]):
    """main"""
    run(get_parser().parse_args(args))


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))
