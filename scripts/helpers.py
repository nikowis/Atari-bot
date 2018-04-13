import math

import keras
import numpy as np
from skimage.transform import resize
import sys


def preprocess(img):
    return resize(np.mean(img, axis=2).astype(np.uint8), (110, 84), preserve_range=True)[17:110 - 9, :]


def transform_reward(reward):
    return np.sign(reward)


def get_epsilon_for_iteration(iteration, treshold, min=0.1):
    if iteration > treshold:
        return min
    else:
        return 1.0 - ((1.0 - min) / treshold) * iteration


def normalize(frame):
    return frame / 255


def copy_model(model):
    """Returns a copy of a keras model."""
    model.save('tmp_model')
    return keras.models.load_model('tmp_model')


def get_start_state(frame, frames_count):
    """
    Creates an array of frames_count frames from a single frame.
    :param frames_count:  frames count in a state
    :param frame: starting frame
    :return: array of frames_count frames
    """
    processed_frame = preprocess(frame)
    start_state = [0] * frames_count
    for i in range(frames_count):
        start_state[i] = processed_frame
    return start_state


def get_next_state(stte, new_frame, frames_count):
    """
    Removes first frame from state, and appends new_frame at the end.
    :param frames_count: frames count in a state
    :param stte: current state
    :param new_frame: frame to append
    :return: new state
    """
    new_state = [0] * frames_count
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


def load_model(path):
    return keras.models.load_model(path)


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
