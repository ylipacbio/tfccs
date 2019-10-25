import numpy as np
import tensorflow as tf
import sys
import os
import os.path as op
from tfccs.utils import load_fextract_npz


def train(x_train, y_train, out_dir, name="multinomial", epochs=500):
    """
    x_train - normalized standardized features
    y_train - outputs
    """
    if not op.exists(out_dir):
        os.mkdir(out_dir)
    print("x_train shape: " + str(x_train.shape))
    print("y_train shape: " + str(y_train.shape))
    print('head(x_train, 1): ' + str(x_train[0]))
    print('head(y_train, 1): ' + str(y_train[0]))
    x_nrow = x_train.shape[0]
    x_ncol = 1 if len(x_train.shape) == 1 else x_train.shape[1]
    y_nrow = y_train.shape[0]
    y_ncol = 1 if len(y_train.shape) == 1 else y_train.shape[1]
    if x_nrow != y_nrow:
        raise ValueError("Number of rows in x_train {} and y_train {} diff!".format(x_nrow, y_nrow))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(y_ncol, kernel_initializer="random_uniform", activation="softmax", input_shape=(x_ncol,))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    summary = model.summary()
    print(summary)
    model_png = op.join(out_dir, '{}.model.png'.format(name))
    checkpoint_file = op.join(out_dir, "{}.ckpt".format(name))
    tf.keras.utils.plot_model(model, to_file=model_png)

    # Fit and call back
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=out_dir, verbose=1)
    model.fit(x_train, y_train, epochs=epochs, callbacks=[cp_callback])

    # Evaluate
    evl = model.evaluate(x_train, y_train)
    print(evl)
    tf.saved_model.save(model, out_dir)
    return model, evl


def multinomial_ccs2genome():
    test_npz_filename = "/pbi/dept/secondary/siv/yli/jira/tak-59/multi-ccs2genome/chunk-0.train.npz"
    fextract_input, _, _, ccs2genome_cigars, nrow, ncol = \
        load_fextract_npz(test_npz_filename)
    out_dir = 'multinomial-1M'
    train(fextract_input, ccs2genome_cigars, out_dir, 'lambda', epochs=1000)


if __name__ == "__main__":
    multinomial_ccs2genome()
