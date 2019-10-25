from tfccs.multinomial import train
import numpy as np
import tensorflow as tf
import os.path as op
import pytest

def test_naive_multinomial():
    """
    Create a simple naive multipnomial model where
    - input X is a 1D array of floats in [0, 1.]
    - output Y is one-hot encode of three classes, A, B, C.
    [1,0,0] -> class A
    [0,1,0] -> class B
    [0,0,1] -> class C
    Any x_i in [0, 0.3) belong to class A.
    Any x_i in [0.3, 0.6) belong to class B.
    Any x_i in [0.6, 1) belong to class C.
    """
    out_dir = op.join(op.dirname(op.dirname(__file__)), 'out', 'naive_multinomial')
    x_train = np.fromiter(range(1000), dtype=np.float32) / 1000.  # 0, 0.001, ..., 0.999
    y_train = np.asarray([1,0,0] * 300 + [0,1,0] * 300 + [0,0,1] * 400).reshape(1000, 3)
    model, evl = train(x_train, y_train, out_dir, 'naive_multinomial', epochs=200)
    pred = model.evaluate([0.02, 0.5, 0.8], [[1,0,0], [0, 1, 0], [0, 0, 1]])
    assert pred[1] == 1.0
    # save_dir = '/pbi/dept/secondary/siv/yli/jira/tak-97/naive-multinomial/'
    # new_model = tf.keras.models.load_model(save_dir)
    # new_pred = new_model.evaluate([0.02, 0.5, 0.8], [[1,0,0], [0, 1, 0], [0, 0, 1]])
    # assert pred == new_pred
