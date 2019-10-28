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
from tfccs.constants import NO_TRAIN_FEATURES, ORDERED_FEATURES_KEY
from tfccs.utils import load_fextract_stat_json_2, is_good_fextract_row, cap_outlier_standardize


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


def one_hot_encode_cigar(cigar):
    assert cigar in '=IDX'
    return [1 if cigar == '=' else 0,
            1 if cigar == 'I' else 0,
            1 if cigar == 'X' else 0,
            1 if cigar == 'D' else 0]


def one_hot_to_cigar(one_hot):
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
    if num_prev_deletions == 0:
        return one_hot_encode_cigar('D')
    assert current_cigar in '=IX'
    return one_hot_encode_cigar(current_cigar)


def convert_fextract_row(input_d):
    """
    1) Remove NO_TRAIN_FEATURES and DUPLICATED_FEATURES
    2) One-hot encode CCSBase, aka replace 'CCSBase' by 'CCSBaseA', 'CCSBaseC', 'CCSBaseG', 'CCSBaseT'
    3) Merge 'CCSToGenomeCigar' and 'CcsToGenomePrevDeletions', and report as one-hot encode
    """
    base_d = one_hot_base(input_d['CCSBase'])
    arrow_qv = int(input_d["ArrowQv"])
    ccs2genomer_cigar = ccs2genome_cigar_counting_prev_dels(input_d['CCSToGenomeCigar'],
                                                            input_d['CcsToGenomePrevDeletions'])
    input_d.update(base_d)
    for feature in NO_TRAIN_FEATURES + ['CCSBase'] + DUPLICATED_FEATURES:
        input_d.pop(feature, None)
    return input_d, arrow_qv, ccs2genomer_cigar


def fextract2numpy(fextract_filename, output_prefix, num_train_rows, forward_only_ccs, no_dump_remaining, stat_json):
    reader = csv.DictReader(open(fextract_filename, 'r'), delimiter=',')
    raw_reader = open(fextract_filename, 'r')
    header = next(raw_reader)  # Skip header
    features = reader.fieldnames

    # If fextract.stat.json was provided as input, check features in csv and stat.json MATCH
    stat_d, stat_features = None, None
    if stat_json is not None:
        stat_d, stat_features = load_fextract_stat_json_2(stat_json)
        trainable_features = set(features).difference(set(NO_TRAIN_FEATURES + ['CCSBase']))
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
    raw_train_writer = open(output_prefix + '.train.fextract.csv', 'w')
    raw_train_writer.write(header)
    raw_test_writer = open(output_prefix + '.test.fextract.csv', 'w')
    raw_test_writer.write(header)
    for r, raw_r in zip(reader, raw_reader):
        if not is_good_fextract_row(r, forward_only_ccs=forward_only_ccs):
            continue
        out_r, arrow_qv, ccs2genome_cigar = convert_fextract_row(r)
        if out_features is None:
            out_features = list(out_r.keys())
        else:
            if out_features != list(out_r.keys()):
                raise ValueError("Could not convert row {} to consistent output features!".format(r[0:50]))
        if idx < num_train_rows:
            raw_train_writer.write(raw_r)
        elif no_dump_remaining:
            break
        else:
            raw_test_writer.write(raw_r)
        new_r = np.fromiter(out_r.values(), dtype=np.float32)
        dataset.append(new_r)
        arrow_qvs.append(arrow_qv)
        ccs2genome_cigars.append(ccs2genome_cigar)
        if idx % 500000 == 0:
            print("Processing {} rows".format(idx))
        idx += 1
    raw_train_writer.close()
    raw_test_writer.close()

    if len(dataset) == 0:
        raise ValueError("Output empty train data!")
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
    out_header_filename = output_prefix + ".train.header"
    with open(out_header_filename, 'w') as writer:
        writer.write(','.join(out_features))

    # Write ordered output features as json
    out_ordered_features_json_filename = output_prefix + ".ordered_features.json"
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

    out_train_filename = output_prefix + ".train.npz"
    out_test_filename = output_prefix + ".test.npz"
    zipsave(out_train_filename, 0, num_train_rows)
    t3 = datetime.datetime.now()
    print("Dumped {} rows of training data, time={}".format(num_train_rows, t3-t2))
    if len(dataset) > num_train_rows and not no_dump_remaining:
        zipsave(out_test_filename, num_train_rows, len(dataset))
        t4 = datetime.datetime.now()
        print("Dumped {} rows of test data, time={} ".format(len(dataset) - num_train_rows, t4-t3))


def run(args):
    if not args.stat_json:
        print("WARNING! No fextract.stat.json file provided, will NOT standardize features!")
    else:
        if not args.stat_json.endswith('.stat.json'):
            raise ValueError("Input --stat-json file {} must ends with '.stat.json'")

    fextract2numpy(fextract_filename=args.fextract_filename, output_prefix=args.output_prefix, num_train_rows=args.num_train_rows,
                   forward_only_ccs=not args.both_strands, no_dump_remaining=args.no_dump_remaining, stat_json=args.stat_json)
    return 0


def get_parser():
    """Set up and return argument parser."""
    desc = """Convert fextract csv file to zipped numpy files, including
    ${output_prefix}.train.npz - num_train_rows rows
    ${output_prefix}.test.npz - others\n"""
    p = argparse.ArgumentParser(desc)
    p.add_argument("fextract_filename", help="fextract csv file")
    p.add_argument("output_prefix", help="Output prefix")
    p.add_argument("--num-train-rows", type=int, default=1000000, help="Number of training rows")
    p.add_argument("--no-dump-remaining", default=False,
                   help="Do not dump remaining rows other than training",
                   action="store_true")
    p.add_argument("--both-strands", default=False,
                   help="Default=False, only use CCS forward mapped to genome, otherwise use both stranded CCS",
                   action="store_true")
    p.add_argument("--stat-json", default=None,
                   help="standardize features using mean/stdev/min/max from stat.json")
    return p


def main(args=sys.argv[1:]):
    """main"""
    run(get_parser().parse_args(args))


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))
