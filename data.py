import scipy.io
import scipy.misc
from sklearn.preprocessing import normalize
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import config

FLAGS = config.FLAGS

dataset_path = 'dataset/ShanghaiTech/part_B/'

train_path = dataset_path + 'train_data/'
test_path = dataset_path + 'test_data/'


def load_mat(path):
    mat = scipy.io.loadmat(path)
    mat = mat['image_info'][0][0][0][0][0]
    return mat


def make_density_map(xs):
    """
    :param xs : head coordinate. [n, 2]
    :return: density_map
    """

    def get_mean_distance(xs, m=3):
        if len(xs) == 1:
            means = [FLAGS.input_shape / 2.]
        else:
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

    mean_distances = get_mean_distance(xs)
    # print(mean_distances)
    # print(len(xs))

    size_of_map = FLAGS.input_shape

    density_map = np.zeros((size_of_map, size_of_map))

    for i, xyi in enumerate(xs):
        yi = xyi[0]
        xi = xyi[1]
        zero_map = np.zeros((size_of_map, size_of_map))

        zero_map[int(xi)][int(yi)] = 1
        if mean_distances[i] and not np.isnan(mean_distances[i]):
            sigma = int(mean_distances[i] * FLAGS.beta)
            if sigma % 2 == 0:
                sigma += 1
            zero_map = cv2.GaussianBlur(zero_map, (sigma, sigma), 0)
        else:
            sigma = int(FLAGS.input_shape / 8. * FLAGS.beta)
            if sigma % 2 == 0:
                sigma += 1
            zero_map = cv2.GaussianBlur(zero_map, (sigma, sigma), 0)

        density_map = density_map + zero_map

    # for xi, yi in xs:
    #     density_map[int(xi)][int(yi)][0] = 1
    #
    # density_map = cv2.GaussianBlur(density_map, (sigma, sigma), 0)

    # resize 1/4 and total sum recovery
    density_map = cv2.resize(density_map, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) * 16
    density_map = np.expand_dims(density_map, 2)
    return density_map


def make_density_map_2(xs, sigma=15):
    """
    :param xs : head coordinate. [n, 2]
    :param sigma : Gaussian blur sigma
    :return: density_map
    """
    size_of_map = FLAGS.input_shape

    density_map = np.zeros((size_of_map, size_of_map))

    for i, xyi in enumerate(xs):
        yi = xyi[0]
        xi = xyi[1]
        density_map[int(xi)][int(yi)] = 1

    density_map = cv2.GaussianBlur(density_map, (sigma, sigma), 0)
    density_map = cv2.resize(density_map, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    density_map = np.expand_dims(density_map, 2)
    return density_map


def load_data(path):
    # make images data
    images_path = os.listdir(path + 'images/')
    images_path.sort()
    images_patches = []
    # IMAGE_SIZE = (768, 1024)
    IMAGE_SIZE = (1024, 768)
    patch_size = FLAGS.input_shape

    for image_path in images_path:
        image = cv2.imread(path + 'images/' + image_path)
        patch_0_0 = image[0:patch_size,
                    0:patch_size]
        patch_0_1 = image[int((IMAGE_SIZE[1] - patch_size) / 2): int((IMAGE_SIZE[1] - patch_size) / 2) + patch_size,
                    0:patch_size]
        patch_0_2 = image[IMAGE_SIZE[1] - patch_size:,
                    0:patch_size]

        patch_1_0 = image[0:patch_size,
                    int((IMAGE_SIZE[0] - patch_size) / 2):int((IMAGE_SIZE[0] - patch_size) / 2) + patch_size]
        patch_1_1 = image[int((IMAGE_SIZE[1] - patch_size) / 2): int((IMAGE_SIZE[1] - patch_size) / 2) + patch_size,
                    int((IMAGE_SIZE[0] - patch_size) / 2): int((IMAGE_SIZE[0] - patch_size) / 2) + patch_size]
        patch_1_2 = image[IMAGE_SIZE[1] - patch_size:,
                    int((IMAGE_SIZE[0] - patch_size) / 2): int((IMAGE_SIZE[0] - patch_size) / 2) + patch_size]

        patch_2_0 = image[0:patch_size,
                    IMAGE_SIZE[0] - patch_size:]
        patch_2_1 = image[int((IMAGE_SIZE[1] - patch_size) / 2): int((IMAGE_SIZE[1] - patch_size) / 2) + patch_size,
                    IMAGE_SIZE[0] - patch_size:]
        patch_2_2 = image[IMAGE_SIZE[1] - patch_size:,
                    IMAGE_SIZE[0] - patch_size:]

        images_patches.append(patch_0_0)
        images_patches.append(patch_0_1)
        images_patches.append(patch_0_2)
        images_patches.append(patch_1_0)
        images_patches.append(patch_1_1)
        images_patches.append(patch_1_2)
        images_patches.append(patch_2_0)
        images_patches.append(patch_2_1)
        images_patches.append(patch_2_2)

        # plt.imshow(patch_0_0)
        # plt.show()

    gts_path = os.listdir(path + 'ground-truth/')
    gts_path.sort()
    gts_patches = []

    num_of_head = []

    for gt_path in gts_path:
        gt = load_mat(path + 'ground-truth/' + gt_path)
        gt_0_0 = []
        gt_0_1 = []
        gt_0_2 = []
        gt_1_0 = []
        gt_1_1 = []
        gt_1_2 = []
        gt_2_0 = []
        gt_2_1 = []
        gt_2_2 = []

        for x, y in gt:
            if x < patch_size:
                if y < patch_size:
                    gt_0_0.append([x, y])
                if int((IMAGE_SIZE[1] - patch_size) / 2) <= y < int((IMAGE_SIZE[1] - patch_size) / 2) + patch_size:
                    gt_0_1.append([x, y - int((IMAGE_SIZE[1] - patch_size) / 2)])
                if IMAGE_SIZE[1] - patch_size <= y:
                    gt_0_2.append([x, y - IMAGE_SIZE[1] + patch_size])
            if int((IMAGE_SIZE[0] - patch_size) / 2) <= x < int((IMAGE_SIZE[0] - patch_size) / 2) + patch_size:
                if y < patch_size:
                    gt_1_0.append([x - int((IMAGE_SIZE[0] - patch_size) / 2), y])
                if int((IMAGE_SIZE[1] - patch_size) / 2) <= y < int((IMAGE_SIZE[1] - patch_size) / 2) + patch_size:
                    gt_1_1.append(
                        [x - int((IMAGE_SIZE[0] - patch_size) / 2), y - int((IMAGE_SIZE[1] - patch_size) / 2)])
                if IMAGE_SIZE[1] - patch_size <= y < IMAGE_SIZE[1]:
                    gt_1_2.append([x - int((IMAGE_SIZE[0] - patch_size) / 2), y - IMAGE_SIZE[1] + patch_size])
            if IMAGE_SIZE[0] - patch_size <= x < IMAGE_SIZE[0]:
                if y < patch_size:
                    gt_2_0.append([x - IMAGE_SIZE[0] + patch_size, y])
                if int((IMAGE_SIZE[1] - patch_size) / 2) <= y < int((IMAGE_SIZE[1] - patch_size) / 2) + patch_size:
                    gt_2_1.append([x - IMAGE_SIZE[0] + patch_size, y - int((IMAGE_SIZE[1] - patch_size) / 2)])
                if IMAGE_SIZE[1] - patch_size <= y < IMAGE_SIZE[1]:
                    gt_2_2.append([x - IMAGE_SIZE[0] + patch_size, y - IMAGE_SIZE[1] + patch_size])

        gts_patches.append(gt_0_0)
        gts_patches.append(gt_0_1)
        gts_patches.append(gt_0_2)
        gts_patches.append(gt_1_0)
        gts_patches.append(gt_1_1)
        gts_patches.append(gt_1_2)
        gts_patches.append(gt_2_0)
        gts_patches.append(gt_2_1)
        gts_patches.append(gt_2_2)

        num_of_head.append(len(gt_0_0))
        num_of_head.append(len(gt_0_1))
        num_of_head.append(len(gt_0_2))
        num_of_head.append(len(gt_1_0))
        num_of_head.append(len(gt_1_1))
        num_of_head.append(len(gt_1_2))
        num_of_head.append(len(gt_2_0))
        num_of_head.append(len(gt_2_1))
        num_of_head.append(len(gt_2_2))

    gts_patches = np.array(list(map(make_density_map, gts_patches)))
    # gts_patches = np.array(list(map(make_density_map_2, gts_patches)))

    return np.array(images_patches), np.array(gts_patches), np.array(num_of_head)


if __name__ == '__main__':
    ## DEBUG
    images_patches, gts_patches, num_of_head = load_data('dataset/ShanghaiTech/part_B/debug/')
    print(gts_patches.shape)
    for i, gt in enumerate(gts_patches):
        fig = plt.figure(figsize=(int(FLAGS.input_shape / 4), int(FLAGS.input_shape / 4)))
        fig.add_subplot(1, 2, 1)
        plt.imshow(images_patches[i])

        fig.add_subplot(1, 2, 2)
        plt.imshow(np.squeeze(gts_patches[i]))

        print(num_of_head[i])
        print(np.sum(gts_patches[i]))
        plt.show()

    # import pickle
    #
    # file = 'data_{}.pkl'.format(FLAGS.input_shape)
    # images_patches, gts_patches, num_of_head = load_data('dataset/ShanghaiTech/part_B/train_data/')
    # with open(file, 'wb') as f:
    #     pickle.dump([images_patches, gts_patches, num_of_head], f)
    #
    # with open(file, 'rb') as f:
    #     [images_patches, gts_patches, num_of_head] = pickle.load(f)
