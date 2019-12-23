from tfccs.qvpipe import *
import os.path as op
import pytest

def test_config_io():
    """
    Test Config json
    """
    in_config = op.join(op.dirname(op.dirname(__file__)), 'data', 'qvpipe.json')
    out_dir = op.join(op.dirname(op.dirname(__file__)), 'out', 'qvpipe')
    execute(f'mkdir -p {out_dir}')
    config = CcsQvConfig.from_file(in_config)

    out_json = op.join(out_dir, 'out.json')
    execute(f'rm -f {out_json}')

    config.save_json(out_json)
    assert op.exists(out_json)

    config2 = CcsQvConfig.from_file(out_json)

    param = config2.param_config
    assert param.batch_size == 50
    assert param.epochs == 20
    assert param.early_stop_patience == 4
    assert len(param.layers) == 3
    assert param.layers[0].layer_type == "dense"
    assert param.layers[0].node_size == 64
    assert param.layers[0].activation == "relu"
    assert param.layers[0].kernel_initializer == "random_uniform"
    assert param.layers[0].kernel_regularizer == "l2"
    assert abs(param.layers[0].regularizer_weight - 0.001) <= 1e-7

    assert param.layers[1].layer_type == "dense"
    assert param.layers[1].node_size == 32
    assert param.layers[1].activation == "sigmoid"

    assert param.layers[2].layer_type == "dense"
    assert param.layers[2].activation == "softmax"
    assert param.layers[2].node_size == -1
