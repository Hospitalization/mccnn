import tensorflow as tf
import numpy as np
from tensorflow import keras
import config

FLAGS = config.FLAGS


def mccnn_model() -> keras.Model:
    inputs = keras.layers.Input((FLAGS.input_shape, FLAGS.input_shape, 3))
    x = keras.layers.Conv2D(filters=16, kernel_size=(9, 9), padding='same', activation='relu')(inputs)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(filters=16, kernel_size=(7, 7), padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(filters=8, kernel_size=(7, 7), padding='same', activation='relu')(x)

    y = keras.layers.Conv2D(filters=20, kernel_size=(7, 7), padding='same', activation='relu')(inputs)
    y = keras.layers.MaxPool2D()(y)
    y = keras.layers.Conv2D(filters=40, kernel_size=(5, 5), padding='same', activation='relu')(y)
    y = keras.layers.MaxPool2D()(y)
    y = keras.layers.Conv2D(filters=20, kernel_size=(5, 5), padding='same', activation='relu')(y)
    y = keras.layers.Conv2D(filters=10, kernel_size=(5, 5), padding='same', activation='relu')(y)

    z = keras.layers.Conv2D(filters=24, kernel_size=(5, 5), padding='same', activation='relu')(inputs)
    z = keras.layers.MaxPool2D()(z)
    z = keras.layers.Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu')(z)
    z = keras.layers.MaxPool2D()(z)
    z = keras.layers.Conv2D(filters=24, kernel_size=(5, 5), padding='same', activation='relu')(z)
    z = keras.layers.Conv2D(filters=12, kernel_size=(5, 5), padding='same', activation='relu')(z)

    conc = keras.layers.Concatenate(axis=-1)([x, y, z])
    final = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(conc)

    # print(final)
    model = keras.Model(inputs=inputs, outputs=final)

    return model


def main():
    temp = np.random.rand(8, FLAGS.input_shape, FLAGS.input_shape, 3)
    print(temp.shape)
    model = mccnn_model()

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.mean_squared_error,
        metrics=['mae', 'mse']
    )

    model.summary()

    # model.fit(
    #     x=,
    #     y=,
    #     batch_size=,
    #     epochs=,
    # )


if __name__ == '__main__':
    main()
