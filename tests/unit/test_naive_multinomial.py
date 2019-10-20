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
    y_train = np.asarray([tf.one_hot(0, 3)] * 300 + [tf.one_hot(1, 3)] * 300 + [tf.one_hot(1, 3)] * 400).reshape(1000, 3)
    model, evl = train(x_train, y_train, out_dir, 'naive_multinomial', epochs=100)
    pred = model.evaluate([0.2, 0.5, 0.7], [[1,0,0], [0, 1, 0], [0, 0, 1]])
    assert pred == 1.0


#test_naive_multinomial()
