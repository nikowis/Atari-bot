import numpy as np
from skimage.transform import resize
import keras

def preprocess(img):
    return resize(np.mean(img, axis=2).astype(np.uint8), (110, 84), preserve_range=True)[17:110 - 9, :]


def transform_reward(reward):
    return np.sign(reward)


def get_epsilon_for_iteration(iteration):
    if iteration > 100000:
        return 0.1
    else:
        return 1.0 - (0.9 / 100000) * iteration


def normalize(frame):
    return frame / 255


def copy_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model')
    return keras.models.load_model('tmp_model')
