import scipy.io
import scipy.misc
from sklearn.preprocessing import normalize
import os
import numpy as np
import cv2
import config

FLAGS = config.FLAGS

dataset_path = 'dataset/ShanghaiTech/part_B/'

train_path = dataset_path + 'train_data/'
test_path = dataset_path + 'test_data/'


def load_mat(path):
    mat = scipy.io.loadmat(path)
    return mat


def make_density_map(xs, sigma=5):
    #
    def get_mean_distance(xs, m=5):
        dists = []
        for xi, yi in xs:
            temp = []
            for xi2, yi2 in xs:
                dist = np.linalg.norm(np.array([xi, yi]) - np.array([xi2, yi2]))
                temp.append(dist)
            dists.append(temp)

        means = []
        for dist in dists:
            dist.sort()
            means.append(np.mean(dist[1:1 + m]))

        return means

    print(get_mean_distance(xs))

    size_of_map = FLAGS.input_shape

    density_map = np.zeros((size_of_map, size_of_map, 1))

    for xi, yi in xs:
        density_map[int(xi)][int(yi)][0] = 1

    density_map = cv2.GaussianBlur(density_map, (sigma, sigma), 0)

    density_map = cv2.resize(density_map, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    return density_map


if __name__ == '__main__':
    mats_path = os.listdir(train_path + 'ground-truth/')

    mat = load_mat(train_path + 'ground-truth/' + mats_path[0])
    # print(mat['image_info'][0][0][0][0][0])

    dmap = make_density_map(((0, 1), (1, 2), (223, 0), (12, 55), (100, 10), (200, 10)))
    print(dmap)
