import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray


def preprocess(img):
    return resize(rgb2gray(img), (110, 84))[13:110 - 13, :]


def transform_reward(reward):
    return np.sign(reward)


def get_epsilon_for_iteration(iteration):
    if iteration > 1000000:
        return 0.1
    else:
        return 1 - (0.9 / 1000000) * iteration
