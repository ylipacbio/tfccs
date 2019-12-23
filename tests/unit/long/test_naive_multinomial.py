from tfccs.train import train
from tfccs.models import multinomial_model_0
import numpy as np
import tensorflow as tf
import os.path as op
import pytest

# Disable this test: this model is not guaranteed to converage to high accuracy within 200 epochs.

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
    def make_y(x):
        if x <= 0.33:
            return [1,0,0]
        elif x > 0.33 and x <= 0.67:
            return [0,1,0]
        else:
            return [0,0,1]

    out_dir = op.join(op.dirname(op.dirname(__file__)), 'out', 'naive_multinomial')
    x_train = np.fromiter(range(1000), dtype=np.float32) / 1000.  # 0, 0.001, ..., 0.999
    np.random.shuffle(x_train)

    # WARNING: batch_size matters, and randomization matters!
    # If input x_train is not randomized, and batch_size is large (e.g., 512), then
    # trained model has poor performance (accuray 0.33).
    # If batch size is small (e.g., 32), the trained model has good accuracy.
    # If x_train is randomly shuffled, the trained model has good accuracy even when batch size is large.

    y_train = [make_y(x) for x in x_train]
    y_train = np.asarray(y_train).reshape(1000, 3)

    model, evl = train(x_train=x_train, y_train=y_train,
                       out_dir=out_dir, name='naive_multinomial',
                       epochs=200, batch_size=32,
                       create_and_compile_model_func=multinomial_model_0)
    pred = model.evaluate([0.02, 0.5, 0.8], [[1,0,0], [0, 1, 0], [0, 0, 1]])
    assert pred[1] == 1.0
    # save_dir = '/pbi/dept/secondary/siv/yli/jira/tak-97/naive-multinomial/'
    # new_model = tf.keras.models.load_model(save_dir)
    # new_pred = new_model.evaluate([0.02, 0.5, 0.8], [[1,0,0], [0, 1, 0], [0, 0, 1]])
    # assert pred == new_pred
