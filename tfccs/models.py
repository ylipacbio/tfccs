import tensorflow as tf
import logging
import os.path as op

FORMATTER = op.basename(__file__) + ':%(levelname)s:'+'%(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMATTER)
log = logging.getLogger(__name__)


def multinomial_model_0(x_ncol, y_ncol):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(y_ncol, kernel_initializer="random_uniform",
                              activation="softmax", input_shape=(x_ncol,), name="out_layer")
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def multinomial_model_1(x_ncol, y_ncol):
    # Use one hidden layer, use regularizer
    hidden_layer_nodes = 64
    regularization_weight = 0.001
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_layer_nodes,
                              kernel_initializer="random_uniform",
                              kernel_regularizer=tf.keras.regularizers.l2(regularization_weight),
                              activation="relu",
                              input_shape=(x_ncol,)),
        tf.keras.layers.Dense(y_ncol,
                              kernel_initializer="random_uniform",
                              kernel_regularizer=tf.keras.regularizers.l2(regularization_weight),
                              activation="softmax",
                              input_shape=(hidden_layer_nodes,),
                              name="out_layer")
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def cnn_model(x_ncol, y_ncol):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(int(x_ncol/2), kernel_initializer="random_uniform",
                              activation="relu", input_shape=(x_ncol,)),
        tf.keras.layers.Conv1D(y_ncol, activation="relu", input_shape=(int(x_ncol)/2,)),
        tf.keras.layers.MaxPooling1D((2)),
        tf.keras.layers.Dense(y_ncol, activation="softmax", input_shape=(x_ncol,)),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
