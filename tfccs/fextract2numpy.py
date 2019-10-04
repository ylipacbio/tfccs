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

NO_TRAIN_FEATURES = [r.strip() for r in """
Movie
HoleNumber
CCSPos
CCSLength
ArrowQv
CCSToGenomeStrand
CCSToGenomeCigar
PrevCcsToGenomeCigar
NextCcsToGenomeCigar
CcsToGenomePrevDeletions
""".split('\n') if len(r.strip())]

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


def convert_fextract_row(input_d):
    """
    1) Remove NO_TRAIN_FEATURES and DUPLICATED_FEATURES
    2) One-hot encode CCSBase
    """
    base_d = one_hot_base(input_d['CCSBase'])
    arrow_qv = int(input_d["ArrowQv"])
    input_d.update(base_d)
    for feature in NO_TRAIN_FEATURES + ['CCSBase'] + DUPLICATED_FEATURES:
        input_d.pop(feature, None)
    return input_d, arrow_qv


def is_good_fextract_row(in_d, forward_only_ccs=True):
    """
    Return False if fextract row's
    1) ccs base is within 100bp end of CCS read
    2) ccs read map to reference genome in reverse strand, while forward_only_ccs is True
    """
    dist2end = abs(int(in_d["CCSLength"]) - int(in_d["CCSPos"]))
    strand = in_d["CCSToGenomeStrand"]
    assert strand in 'FR'
    return dist2end > 100 and (not forward_only_ccs or strand == 'F')


def fextract2numpy(fextract_filename, output_prefix, num_train_rows, forward_only_ccs):
    reader = csv.DictReader(open(fextract_filename, 'r'), delimiter=',')
    dataset = []
    arrow_qvs = []
    idx = 0
    features = None
    t0 = datetime.datetime.now()
    for r in reader:
        if not is_good_fextract_row(r, forward_only_ccs=forward_only_ccs):
            continue
        out_r, arrow_qv = convert_fextract_row(r)
        if idx == 0:
            features = out_r.keys()
        new_r = np.fromiter(out_r.values(), dtype=np.float32)
        dataset.append(new_r)
        arrow_qvs.append(arrow_qv)
        if idx % 500000 == 0:
            print ("Processing {} rows".format(idx))
        idx += 1
    if len(dataset) == 0:
        raise ValueError("Output empty train data!")
    t1 = datetime.datetime.now()
    print("Loaded input {} rows, time={}.".format(len(dataset), t1-t0))

    npa = np.asarray(dataset, dtype=np.float32)
    arrow_qvs = np.asarray(arrow_qvs, dtype=np.float32)
    arrow_qv_bins = np.asarray([arrowqv2bin8(qv) for qv in arrow_qvs], dtype=np.float32)
    t2 = datetime.datetime.now()
    print("Created np array time={}.".format(t2-t1))

    out_header_filename = output_prefix + ".train.header"
    with open(out_header_filename, 'w') as writer:
        writer.write(','.join(features))
    print("Created header file {}, time={}.".format(out_header_filename, t2-t1))

    def zipsave(out_filename, start_row, end_row):
        np.savez_compressed(out_filename,
                            fextractinput=npa[start_row:end_row],
                            arrowqv=arrow_qvs[start_row:end_row],
                            arrowqvbin8=arrow_qv_bins[start_row:end_row])

    out_train_filename = output_prefix + ".train.npz"
    out_test_filename = output_prefix + ".test.npz"
    zipsave(out_train_filename, 0, num_train_rows)
    t3 = datetime.datetime.now()
    print ("Dumped {} rows of training data, time={}".format(num_train_rows, t3-t2))
    if len(dataset) > num_train_rows:
        zipsave(out_test_filename, num_train_rows, len(dataset))
        t4 = datetime.datetime.now()
        print ("Dumped {} rows of test data, time={} ".format(len(dataset) - num_train_rows, t4-t3))
    else:
        raise ValueError("Output empty test data!")


def run(args):
    fextract2numpy(args.fextract_filename, args.output_prefix, args.num_train_rows, not args.both_strands)
    return 0


def get_parser():
    """Set up and return argument parser."""
    desc = """Convert fextract csv file to zipped numpy files, including
    ${output_prefix}.train.npz - num_train_rows rows
    ${output_prefix}.test.npz - others"""
    p = argparse.ArgumentParser(desc)
    p.add_argument("fextract_filename", help="fextract csv file")
    p.add_argument("output_prefix", help="Output prefix")
    p.add_argument("--num-train-rows", type=int, default=1000000, help="Number of training rows")
    p.add_argument("--both-strands",
                   help="Default=False, only use CCS forward mapped to genome, otherwise use both stranded CCS",
                   action="store_false")
    return p


def main(args=sys.argv[1:]):
    """main"""
    run(get_parser().parse_args(args))


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))
