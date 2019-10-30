import tensorflow as tf


def create_and_compile_multinomial_model(x_ncol, y_ncol):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(y_ncol, kernel_initializer="random_uniform", activation="softmax", input_shape=(x_ncol,))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_and_compile_cnn_model(x_ncol, y_ncol):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(int(x_ncol/2), kernel_initializer="random_uniform",
                              activation="relu", input_shape=(x_ncol,)),
        tf.keras.layers.Conv1D(y_ncol, activation="relu", input_shape=(int(x_ncol)/2,)),
        tf.keras.layers.MaxPooling1D((2)),
        tf.keras.layers.Dense(y_ncol, activation="softmax", input_shape=(x_ncol,)),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
