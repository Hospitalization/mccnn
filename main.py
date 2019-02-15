import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras
import config
import data
from keras import backend as K
import cv2

FLAGS = config.FLAGS

scale = 255


def count_mse_loss(y_true, y_pred):
    return K.sqrt(K.mean(
        K.square(K.sum(K.sum(K.sum(y_pred / scale, axis=-1), axis=-1), axis=-1)
                 - K.sum(K.sum(K.sum(y_true / scale, axis=-1), axis=-1), axis=-1)
                 ), axis=-1))


def mccnn_model() -> keras.Model:
    inputs = keras.layers.Input((None, None, 3))
    x = keras.layers.Conv2D(filters=16, kernel_size=(9, 9), padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(filters=16, kernel_size=(7, 7), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(filters=8, kernel_size=(7, 7), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    y = keras.layers.Conv2D(filters=20, kernel_size=(7, 7), padding='same', activation='relu')(inputs)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.MaxPool2D()(y)
    y = keras.layers.Conv2D(filters=40, kernel_size=(5, 5), padding='same', activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.MaxPool2D()(y)
    y = keras.layers.Conv2D(filters=20, kernel_size=(5, 5), padding='same', activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Conv2D(filters=10, kernel_size=(5, 5), padding='same', activation='relu')(y)
    y = keras.layers.BatchNormalization()(y)

    z = keras.layers.Conv2D(filters=24, kernel_size=(5, 5), padding='same', activation='relu')(inputs)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.MaxPool2D()(z)
    z = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same', activation='relu')(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.MaxPool2D()(z)
    z = keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding='same', activation='relu')(z)
    z = keras.layers.BatchNormalization()(z)
    z = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(z)
    z = keras.layers.BatchNormalization()(z)

    conc = keras.layers.Concatenate(axis=-1)([x, y, z])
    final = keras.layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='relu')(conc)

    # print(final)
    model = keras.Model(inputs=inputs, outputs=final)

    return model


def main():
    model = mccnn_model()

    model.compile(
        optimizer=keras.optimizers.Adam(lr=FLAGS.lr),
        loss=keras.losses.mean_squared_error,
        metrics=[count_mse_loss],
    )

    model.summary()
    file = 'data_{}.pkl'.format(FLAGS.input_shape)
    with open(file, 'rb') as f:
        [images_patches, gts_patches, num_of_head] = pickle.load(f)

    train_mode = False
    if train_mode:
        # model.load_weights('model_{}.h5'.format(FLAGS.input_shape))
        model.fit(
            x=images_patches,
            y=gts_patches * scale,
            batch_size=8,
            epochs=FLAGS.epochs,
        )
        model.save_weights('model_{}.h5'.format(FLAGS.input_shape))
    else:
        model.load_weights('model_{}.h5'.format(FLAGS.input_shape))

    test_data = cv2.imread('dataset/test/1_2.jpg')

    pred = model.predict(
        x=np.expand_dims(test_data, 0),
    )
    # pred = model.predict(
    #     x=data.load_data('dataset/ShanghaiTech/part_B/debug/')[0],
    # )

    import matplotlib.pyplot as plt
    for p in pred:
        # p = p.clip(min=0)
        print(np.sum(p) / scale)
        plt.imshow(np.squeeze(p))
        plt.show()


if __name__ == '__main__':
    main()

    # file = 'data_{}.pkl'.format(FLAGS.input_shape)
    # with open(file, 'rb') as f:
    #     [images_patches, gts_patches, num_of_head] = pickle.load(f)
    #
    # print(gts_patches.shape)
    # for gt in gts_patches:
    #     print(np.sum(gt))
