import numpy as np
import os
from tqdm import tqdm
import tensorflow as tf


def get_unique_path(directory, name, number=0, exists_function=os.path.isfile):
    if not exists_function(os.path.join(directory, name.format(number))):
        return os.path.join(directory, name.format(number))
    else:
        return get_unique_path(directory, name, number + 1, exists_function)


def make_unique_directory(name):
    dir_path = get_unique_path("", name, exists_function=os.path.isdir)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path


def mask_to_weights(mask):
    return mask.astype(np.float32) * len(mask) / np.count_nonzero(mask)


def weight_by_class(y, weights):
    samples = y[weights != 0].sum(axis=0)
    samples = np.true_divide(len(weights), samples * len(samples[samples != 0]), out=0. * samples, where=samples != 0)
    return (y * samples).max(axis=-1) * weights


def gpu_initialise(gpu_list):
    if len(gpu_list) > 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpus = [gpus[gpu_id] for gpu_id in gpu_list]
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        tf.config.experimental.set_visible_devices([], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(logical_gpus), "Logical GPU")
