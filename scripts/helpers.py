import numpy as np
from skimage.transform import resize


def preprocess(img):
    return resize(np.mean(img, axis=2).astype(np.uint8), (110, 84))[17:110 - 9, :]


def transform_reward(reward):
    return np.sign(reward)


def get_epsilon_for_iteration(iteration):
    if iteration > 1000000:
        return 0.1
    else:
        return 1.0 - (0.9 / 1000000) * iteration


def normalize(frame):
    return frame / 255
