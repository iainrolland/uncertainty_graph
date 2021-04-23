import numpy as np
import os
from datasets import HoustonDataset, Karate
from tqdm import tqdm
import tensorflow as tf
from models import GCN, S_GCN

supported_models = {"GCN": GCN, "S_GCN": S_GCN}
supported_datasets = {"houston": HoustonDataset, "karate": Karate}


def dissonance_uncertainty(alpha):
    strength = np.sum(alpha, axis=1, keepdims=True)
    belief = (alpha - 1) / strength
    dis_un = np.zeros_like(strength, dtype="float64")

    for i in tqdm(range(belief.shape[0])):
        b = belief[i]
        b[b == 0] += 1e-15  # to prevent divide by any divide by zero errors
        bal = 1 - np.abs(b[:, None] - b[None, :]) / (b[None, :] + b[:, None]) - np.eye(len(b))
        coefficients = b[:, None] * b[None, :] - np.diag(b ** 2)
        denominator = np.sum(b[None, :] * np.ones(belief.shape[1]) - np.diag(b), axis=-1, keepdims=True)
        dis_un[i] = (coefficients * bal / denominator).sum()

    return dis_un


def vacuity_uncertainty(alpha):
    return alpha.shape[-1] / alpha.sum(axis=-1)


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
    return (y * samples).max(axis=-1)


def gpu_initialise(gpu_list):
    if len(gpu_list) > 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(list(np.array(gpus)[gpu_list]), 'GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        tf.config.experimental.set_visible_devices([], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(logical_gpus), "Logical GPU")
