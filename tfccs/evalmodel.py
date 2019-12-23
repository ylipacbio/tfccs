import numpy as np
import tensorflow as tf
import sys
import os
import os.path as op
import csv
from tfccs.utils import load_fextract_npz
from tfccs.fextract2numpy import one_hot_to_cigar, convert_fextract_row
from argparse import ArgumentParser


DEBUG = False


def eval_model(model_dir, x_test, y_test):
    if DEBUG:
        for idx, row in enumerate(x_test):
            print(', '.join(['{:0.5f}'.format(col) for col in row]))
            if idx == 5:
                break
    model = tf.keras.models.load_model(model_dir)
    predicts = model.predict(x_test)
    print(predicts)
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Loss of model {} on test data is: {}".format(model_dir, loss))
    print("Accuracy of model {} on test data is: {}".format(model_dir, accuracy))
    return predicts


def npz_csv_must_match(y_test, in_csv):
    reader = csv.DictReader(open(in_csv, 'r'), delimiter=',')
    idx = 0
    for idx, r in enumerate(reader):
        _, _, one_hot_encoded_cigar = convert_fextract_row(r)
        cigar = one_hot_to_cigar(one_hot_encoded_cigar)
        y_cigar = one_hot_to_cigar(y_test[idx])
        if (idx < len(y_test)) and (y_cigar != cigar):
            raise ValueError("npz file row {} has ArrowQv {}, NOT match csv record {}".format(idx, y_cigar, r))
    if idx != len(y_test) - 1:
        raise ValueError("npz file has {} rows, while csv has {} rows, must match".format(len(y_test), idx+1))


def add_predicted_cigars_and_write(in_csv, predicted_cigars, out_csv):
    """
    For each row in in_csv, simply attach predicted_cigars[row] as the last column and write to out_csv
    """
    with open(in_csv, 'r') as reader, open(out_csv, 'w') as writer:
        for idx, r in enumerate(reader):
            predicted_cigar_idx = idx - 1
            if idx == 0:
                writer.write(
                    r.strip() + ',SamplingPredictedCigar,SamplingPredictedMatch,SamplingPredictedI,SamplingPredictedX,SamplingPredictedD\n')
            elif predicted_cigar_idx == len(predicted_cigars):
                return
            elif predicted_cigar_idx > len(predicted_cigars):
                raise ValueError("Number of rows in in_csv {} NOT match number of predicted cigars {}.".format(
                    idx, len(predicted_cigars)))
            else:
                o0, o1, o2, o3 = predicted_cigars[predicted_cigar_idx]
                predicted_cigar = one_hot_to_cigar(predicted_cigars[predicted_cigar_idx])
                writer.write(
                    r.strip() + ',{oc},{o0:.6f},{o1:.6f},{o2:.6f},{o3:.6f}\n'.format(oc=predicted_cigar, o0=o0, o1=o1, o2=o2, o3=o3))


def run(args):
    tfpb_file = op.join(args.in_model_dir, 'saved_model.pb')
    if not op.exists(tfpb_file):
        raise IOError("Could not find tensorflow saved model file {}!".format(tfpb_file))
    if not args.in_fextract_csv.endswith('.csv'):
        raise ValueError("Input fextact csv file {} must ends with csv!".format(args.in_fextract_csv))
    if not args.in_fextract_npz.endswith('.npz'):
        raise ValueError("Input fextact npz file {} must ends with npz!".format(args.in_fextract_npz))
    if not args.out_csv.endswith('.csv'):
        raise ValueError("Output fextact csv file {} must ends with csv!".format(args.out_csv))

    x_test, x_arrowqv, _, y_test, _, _ = load_fextract_npz(args.in_fextract_npz)
    npz_csv_must_match(y_test, args.in_fextract_csv)
    predicted_cigars = eval_model(args.in_model_dir, x_test, y_test)
    add_predicted_cigars_and_write(args.in_fextract_csv, predicted_cigars, args.out_csv)


def get_parser():
    """Set up and return argument parser."""
    desc = """Load a model and evaluate on test data."""
    p = ArgumentParser(desc)
    p.add_argument("in_model_dir", help="Input tensorflow model directory.")
    p.add_argument("in_fextract_npz", help="Input fextract.npz file for test")
    p.add_argument("in_fextract_csv", help="Input fextract.csv file which must match input fextract.npz")
    p.add_argument("out_csv", help="Output csv the same as input_fextract_csv with an additional column PredictedCigar")
    return p


def main(args=sys.argv[1:]):
    """main"""
    run(get_parser().parse_args(args))


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))
