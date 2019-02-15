import tensorflow as tf
import numpy as np
import pickle
from tensorflow import keras
import config
import data
from models import mccnn
from models import mscnn
from keras import backend as K
import cv2

FLAGS = config.FLAGS

scale = 1


def count_mse_loss(y_true, y_pred):
    return K.sqrt(K.mean(
        K.square(K.sum(K.sum(K.sum(y_pred / scale, axis=-1), axis=-1), axis=-1)
                 - K.sum(K.sum(K.sum(y_true / scale, axis=-1), axis=-1), axis=-1)
                 ), axis=-1))


def main():
    if FLAGS.model == 'mcnn':
        model = mccnn.model()
    elif FLAGS.model == 'mscnn':
        model = mscnn.model()

    model.compile(
        optimizer=keras.optimizers.Adam(lr=FLAGS.lr),
        loss=keras.losses.mean_squared_error,
        metrics=[count_mse_loss],
    )

    model.summary()
    file = 'data_{}.pkl'.format(FLAGS.input_shape)
    with open(file, 'rb') as f:
        [images_patches, gts_patches, num_of_head] = pickle.load(f)

    train_mode = True
    if train_mode:
        # model.load_weights('model_{}.h5'.format(FLAGS.model, FLAGS.input_shape))
        model.fit(
            x=images_patches,
            y=gts_patches * scale,
            batch_size=8,
            epochs=FLAGS.epochs,
        )
        model.save_weights('model_{}_{}.h5'.format(FLAGS.model, FLAGS.input_shape))
    else:
        model.load_weights('model_{}_{}.h5'.format(FLAGS.model, FLAGS.input_shape))

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
