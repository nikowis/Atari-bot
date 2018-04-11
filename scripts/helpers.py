import numpy as np
from skimage.transform import resize
import keras
import math


def preprocess(img):
    return resize(np.mean(img, axis=2).astype(np.uint8), (110, 84), preserve_range=True)[17:110 - 9, :]


def transform_reward(reward):
    return np.sign(reward)


def get_epsilon_for_iteration(iteration):
    if iteration > 1000000:
        return 0.1
    else:
        return 1.0 - (0.9 / 1000000) * iteration


def normalize(frame):
    return frame / 255


def copy_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model')
    return keras.models.load_model('tmp_model')


def get_start_state(frame, img_size, frames_count):
    """
    Creates an array of 4 frames from a single frame.
    :param frame: starting frame
    :return: array of 4 frames
    """
    processed_frame = preprocess(frame)
    start_state = np.empty((frames_count, img_size, img_size))
    start_state[:] = processed_frame
    return start_state


def get_next_state(stte, new_frame, img_size, frames_count):
    """
    Removes first frame from state, and appends new_frame at the end.
    :param stte: current state
    :param new_frame: frame to append
    :return: new state
    """
    new_state = np.empty((frames_count, img_size, img_size))
    processed_frame = preprocess(new_frame)
    new_state[0:frames_count - 1] = stte[1:frames_count]
    new_state[frames_count - 1] = processed_frame
    return new_state


def save_model(model, path):
    model.save(path)

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])
