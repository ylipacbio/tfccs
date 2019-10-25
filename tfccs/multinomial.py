import numpy as np
import tensorflow as tf
import sys
import os
import os.path as op
from tfccs.utils import load_fextract_npz
import argparse


def train(x_train, y_train, out_dir, name="multinomial", batch_size=32, epochs=500):
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
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[cp_callback])

    # Evaluate
    evl = model.evaluate(x_train, y_train)
    print(evl)
    tf.saved_model.save(model, out_dir)
    return model, evl


def multinomial_ccs2genome(in_npz, out_dir, name, batch_size, epochs):
    fextract_input, _, _, ccs2genome_cigars, nrow, ncol = load_fextract_npz(in_npz)
    train(fextract_input, ccs2genome_cigars, out_dir=out_dir, name=name, batch_size=batch_size, epochs=epochs)


def run(args):
    multinomial_ccs2genome(in_npz=args.in_fextract_npz, out_dir=args.out_dir,
                           name=args.name, batch_size=args.batch_size, epochs=args.epochs)
    return 0


def get_parser():
    """Set up and return argument parser."""
    desc = """Train a multinomial model"""
    p = argparse.ArgumentParser(desc)
    p.add_argument("in_fextract_npz", help="Input fextract standarized npz file")
    p.add_argument("out_dir", help="Output directory for saving model.")
    p.add_argument("--name", help="Model name.")
    p.add_argument("--batch-size", default=32, type=int, help="Batch size")
    p.add_argument("--epochs", default=500, type=int, help="Epochs")
    return p


def main(args=sys.argv[1:]):
    """main"""
    run(get_parser().parse_args(args))


if __name__ == "__main__":
    sys.exit(main(args=sys.argv[1:]))
