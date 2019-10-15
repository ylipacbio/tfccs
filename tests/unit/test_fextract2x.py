import os
import os.path as op
import numpy as np
import pytest
from tfccs.fextract2stat import compute_feature_stats
from tfccs.utils import load_fextract_stat_json, load_fextract_npz
from tfccs.fextract2numpy import fextract2numpy


ROOT_DIR = op.dirname(op.dirname(__file__))

def test_f2stat_f2npz():
    """
    Test the whole process of fextract data preprocessing, including
    - compute stat of trainable columns and store stat to json
    - load stat json and sanity check
    - standardize fextract.csv and save to numpy zipped file
    - load numpy zipped file and sanity check
    """
    out_dir = op.join(ROOT_DIR, 'out', 'test_fextract2stat')
    if not op.exists(out_dir):
        os.mkdir(out_dir)

    # Step 1: create input fextract.csv
    # Create a csv file with two feature columns: F1 and F2,
    # - a non-trainable column Movie, and
    # - CCSLength, CCSPos which decide if a row is good to go, and
    # - CCSToGenomeCigar, and CcsToGenomePrevDeletions
    # F1=[1, 2, 3, 4, 5],  mean 3.0, stdev 1.4142, min 1, max 5
    # F1=[-1, -2, -3, -4, -10],  mean -4.0, stdev 3.1622, min -10, max -1
    in_csv = op.join(out_dir, 'in.csv')
    with open(in_csv, 'w') as writer:
        writer.write('CCSBase,CCSLength,F1,CCSPos,F2,Movie,ArrowQv,CCSToGenomeCigar,CcsToGenomePrevDeletions\n')
        writer.write('A,1000,1,100,-1,movie1,1,=,0\n')
        writer.write('C,1000,2,200,-2,movie2,10,I,0\n')
        writer.write('G,1000,3,300,-3,movie3,20,X,0\n')
        writer.write('T,1000,4,400,-4,movie4,30,=,0\n')
        writer.write('A,1000,5,500,-10,movie4,70,=,1\n')

    # Step 2: compute stats and dump stat to json
    out_json = op.join(out_dir, 'out.stat.json')
    compute_feature_stats(in_csv=in_csv, out_json=out_json, forward_only_ccs=False)

    # Step 3: read stat from json and compare with expected
    out_d, out_features = load_fextract_stat_json(out_json)
    assert out_features == set(['F1', 'F2'])

    assert out_d['F1'].mean == 3.0
    assert abs(out_d['F1'].stdev - 1.4142) <= 0.0001
    assert out_d['F1'].min == 1
    assert out_d['F1'].max == 5

    assert out_d['F2'].mean == -4.0
    assert abs(out_d['F2'].stdev - 3.1622) <= 0.0001
    assert out_d['F2'].min == -10
    assert out_d['F2'].max == -1

    # Step 4: standarize in_csv to and convert to numpy ziped
    out_train_npz = op.join(out_dir, 'f2n.train.npz')
    out_header = op.join(out_dir, 'f2n.train.header')
    fextract2numpy(fextract_filename=in_csv, output_prefix=op.join(out_dir, 'f2n'), num_train_rows=500,
                   forward_only_ccs=False, no_dump_remaining=True, stat_json=out_json)
    assert op.exists(out_train_npz)
    assert op.exists(out_header)
    assert 'F1,F2,CCSBaseA,CCSBaseC,CCSBaseG,CCSBaseT' == [r.strip() for r in open(out_header,'r')][0]

    # Step 5: load numpy zipped and standardized out_train_npz and comapre with expected
    f_input, arrowqv, arrowqvbin8, ccs2genome, nrow, ncol = load_fextract_npz(out_train_npz)
    assert nrow == 5
    assert ncol == 6
    assert list(arrowqv) == [1., 10., 20., 30., 70.]
    expected_arrowqvbin8 = np.array([
        [1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1.]])
    assert arrowqvbin8.all() == expected_arrowqvbin8.all()

    expected_f_input =  np.array([
        [-1.4142135,   0.94868326,  1.,          0.,          0.,          0.,        ],
        [-0.70710677,  0.6324555,   0.,          1.,          0.,          0.,        ],
        [ 0.,          0.31622776,  0.,          0.,          1.,          0.,        ],
        [ 0.70710677,  0.,          0.,          0.,          0.,          1.,        ],
        [ 1.4142135,  -1.8973665,   1.,          0.,          0.,          0.,        ]])
    assert f_input.all() == expected_f_input.all()

    expected_ccs2genome = np.array([
        [1, 0, 0, 0], # =
        [0, 1, 0, 0], # I
        [0, 0, 1, 0], # X
        [1, 0, 0, 0], # =
        [0, 0, 0, 1]  # D
        ])
    assert ccs2genome.all() == expected_ccs2genome.all()

    # Test special case where CCSToGenomeCigar is 'S', this is excluded from training dataset.
    in_csv = op.join(out_dir, 'in2.csv')
    with open(in_csv, 'w') as writer:
        writer.write('CCSBase,CCSLength,F1,CCSPos,F2,Movie,ArrowQv,CCSToGenomeCigar,CcsToGenomePrevDeletions\n')
        writer.write('A,1000,1,100,-1,movie1,1,S,0\n')

    out_train_npz = op.join(out_dir, 'f2n.2.train.npz')
    with pytest.raises(ValueError) as err:
        fextract2numpy(fextract_filename=in_csv, output_prefix=op.join(out_dir, 'f2n.2'), num_train_rows=500,
                       forward_only_ccs=False, no_dump_remaining=True, stat_json=out_json)
    assert err.value.args[0] == "Output empty train data!"
