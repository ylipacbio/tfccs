import pytest
import os.path as op
import numpy as np
from tfccs.evalmodel import eval_model, main

data_dir = op.join(op.dirname(__file__), '../data')
out_dir = op.join(op.dirname(__file__), '../out')
model_dir = op.join('/home/UNIXHOME/yli/repo/ccsqv/models/tf_multinomial_raw_features/')


def test_eval_model():
    # The model saved at repo models/tf_multinom_2019_10_28/
    x_test = np.asarray([[-1.01427, -0.66983, 0.27729, 0.75007, 0.56615, 1.32533, -0.79498, -0.71725, -0.63486, -0.60981, -0.74125, -0.56299, -0.55405, -0.71331, 0.79314, -0.40607, -0.39460, -0.62491, 1.18166, -0.43515, 1.98535, -0.89558, 0.75384, -0.05901, -0.32348, -0.38139, -0.62465, 0.17701, -0.04282, 2.05597, -0.37004, -0.62499, 1.73501, -0.04836, -0.36340, -0.35072, -0.89551, 1.71962, -0.03567, -0.36383, -0.33771, -0.89588, 0.75656, -0.04577, -0.41820, -0.47833, -0.62455, 0.17221, 0.06393, 2.86278, -0.48228, -0.62499, 1.74656, -0.03673, -0.36469, -0.42603, -0.89517, 1.74866, -0.04232, -0.36391, -0.43525, -0.89610, 0.56780, -0.04577, 0.78695, -0.54952, -0.62347, 0.76501, 0.03476, -0.50200, -0.55102, -0.62482, 1.55504, -0.04130, -0.08844, -0.49133, -0.89462, 1.35185, -0.07823, -0.08976, 0.96781, -0.89624, 0.62119, -0.05030, 0.51986, -0.61059, -0.62315, 0.61770, -0.05826, 0.51586, -0.60873, -0.62521, 1.61370, 0.07179, -0.15967, -0.54537, -0.89532, 1.60809, -0.08038, -0.37343, 0.70508, -0.89625, -0.37701, 1.01266, 0.96887, 0.80815, 1.13253, -0.48363, 2.82402, -0.48390, 0.31301, -0.47442, -0.41641, 2.45890, -0.43089, 0.00000, 0.00000, 0.00000, 1.00000]])
    y_test = np.asarray([[0,0,0,1]])
    out_predicts = eval_model(model_dir, x_test, y_test)
    expected_predicts = [[9.9804878e-01, 2.4705005e-04, 4.6353106e-04, 1.2406728e-03]]
    assert np.allclose(out_predicts, expected_predicts, atol=0.0001)


def test_e2e():
    # The INPUT contains a header and 9 rows. Output must match expected output.
    in_csv = op.join(data_dir, 'tiny.fextract.csv')
    in_npz = op.join(data_dir, 'tiny.fextract.npz')
    out_csv = op.join(out_dir, 'test_evalmodel_e2e.csv')
    main([model_dir, in_npz, in_csv, out_csv])
    assert op.exists(out_csv)
    outputs = [r.strip() for r in open(out_csv, 'r')]
    assert len(outputs) == 10
    assert outputs[0].split(',')[-5:] == ["SamplingPredictedCigar", "SamplingPredictedMatch", "SamplingPredictedI", "SamplingPredictedX", "SamplingPredictedD"]

    output_predicts = []
    for output in outputs[1:]:
        output_predicts.append( [float(x) for x in output.split(',')[-4:]] )

    expected_output_predicts = np.asarray([
        [9.98048782e-01, 2.47048156e-04, 4.63530858e-04, 1.24070293e-03],
        [6.87714815e-01, 1.77004805e-03, 6.28944207e-03, 3.04225683e-01],
        [9.82707560e-01, 3.21107327e-05, 4.87336365e-05, 1.72115620e-02],
        [9.68669593e-01, 2.35860585e-03, 8.43188551e-04, 2.81286426e-02],
        [3.43294902e-04, 6.98207676e-01, 1.78025104e-04, 3.01270932e-01],
        [3.83517444e-02, 1.27527863e-04, 1.45796685e-05, 9.61506069e-01],
        [9.65197861e-01, 1.67185010e-03, 1.29109336e-04, 3.30011547e-02],
        [7.78753683e-02, 2.81183384e-02, 8.78825128e-01, 1.51811885e-02],
        [9.95871723e-01, 1.78275164e-03, 1.11308123e-04, 2.23415415e-03]])
    assert len(output_predicts) == 9

    assert np.allclose(np.asarray(output_predicts), expected_output_predicts, atol=0.0001)
