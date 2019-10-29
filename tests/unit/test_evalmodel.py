import pytest
import os.path as op
import numpy as np
from tfccs.evalmodel import eval_model, main

data_dir = op.join(op.dirname(__file__), '../data')
out_dir = op.join(op.dirname(__file__), '../out')
model_dir = op.join('/pbi/dept/secondary/siv/yli/jira/tak-59/multi-ccs2genome/m64002_190608_021007.chunk9.1M.model')

def test_eval_model():
    # This is also the model under ccsqv repo models/tf_multinom_2019_10_28/, which we can restore if path below got deleted
    x_test = np.asarray([[-1.01427, -0.66983, 0.27729, 0.75007, 0.56615, 1.32533, -0.79498, -0.71725, -0.63486, -0.60981, -0.74125, -0.56299, -0.55405, -0.71331, 0.79314, -0.40607, -0.39460, -0.62491, 1.18166, -0.43515, 1.98535, -0.89558, 0.75384, -0.05901, -0.32348, -0.38139, -0.62465, 0.17701, -0.04282, 2.05597, -0.37004, -0.62499, 1.73501, -0.04836, -0.36340, -0.35072, -0.89551, 1.71962, -0.03567, -0.36383, -0.33771, -0.89588, 0.75656, -0.04577, -0.41820, -0.47833, -0.62455, 0.17221, 0.06393, 2.86278, -0.48228, -0.62499, 1.74656, -0.03673, -0.36469, -0.42603, -0.89517, 1.74866, -0.04232, -0.36391, -0.43525, -0.89610, 0.56780, -0.04577, 0.78695, -0.54952, -0.62347, 0.76501, 0.03476, -0.50200, -0.55102, -0.62482, 1.55504, -0.04130, -0.08844, -0.49133, -0.89462, 1.35185, -0.07823, -0.08976, 0.96781, -0.89624, 0.62119, -0.05030, 0.51986, -0.61059, -0.62315, 0.61770, -0.05826, 0.51586, -0.60873, -0.62521, 1.61370, 0.07179, -0.15967, -0.54537, -0.89532, 1.60809, -0.08038, -0.37343, 0.70508, -0.89625, -0.37701, 1.01266, 0.96887, 0.80815, 1.13253, -0.48363, 2.82402, -0.48390, 0.31301, -0.47442, -0.41641, 2.45890, -0.43089, 0.00000, 0.00000, 0.00000, 1.00000]])
    y_test = np.asarray([[0,0,0,1]])
    out_predicts = eval_model(model_dir, x_test, y_test)
    print (out_predicts)
    expected_predicts = [[3.6518050e-03, 1.1295808e-10, 4.7147728e-06, 9.9634343e-01]]
    assert np.allclose(out_predicts, expected_predicts)

def test_e2e():
    # The INPUT contains a header and 9 rows. Output must match expected output.
    in_csv = op.join(data_dir, 'tiny.fextract.csv')
    in_npz = op.join(data_dir, 'tiny.fextract.npz')
    out_csv = op.join(out_dir, 'test_evalmodel_e2e.csv')
    main([model_dir, in_npz, in_csv, out_csv])
    assert op.exists(out_csv)
    outputs = [r.strip() for r in open(out_csv, 'r')]
    assert len(outputs) == 10
    assert outputs[0].split(',')[-5:] == ["PredictedCigar", "Predicted=", "PredictedI", "PredictedX", "PredictedD"]

    output_predicts = []
    for output in outputs[1:]:
        output_predicts.append( [float(x) for x in output.split(',')[-4:]] )

    expected_output_predicts = np.asarray([
        [3.6519575e-03, 1.1295850e-10, 4.7148396e-06, 9.9634331e-01],
        [5.4353762e-01, 1.0746985e-10, 2.1434633e-05, 4.5644096e-01],
        [7.9275429e-02, 1.7089123e-10, 1.0940847e-06, 9.2072344e-01],
        [1.7322069e-02, 1.2338693e-10, 1.5503143e-05, 9.8266244e-01],
        [7.1538943e-01, 2.3632138e-10, 1.4349086e-08, 2.8461060e-01],
        [9.1203320e-01, 4.2408736e-11, 6.2348967e-08, 8.7966673e-02],
        [4.7905289e-02, 4.2043513e-10, 1.1222858e-05, 9.5208353e-01],
        [2.9666493e-02, 3.0423600e-10, 1.7266879e-03, 9.6860683e-01],
        [3.6593257e-03, 2.3359742e-10, 3.0557914e-07, 9.9634039e-01]])
    assert np.allclose(np.asarray(output_predicts), expected_output_predicts)
