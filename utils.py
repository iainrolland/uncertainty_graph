import numpy as np
import os
import tensorflow as tf
from scipy.ndimage import label
import logging


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


def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)


def masks_from_gt(gt, train_ratio=0.4, val_ratio=0.3):
    mask_tr, mask_va, mask_te = [np.zeros_like(gt).astype("bool") for _ in range(3)]

    for class_id in range(1, gt.max() + 1):  # for each output class (other than "Unclassified")
        labels, numb_objects = label(gt == class_id)  # label each contiguous block with a number from 1,...,N
        num_pixels = (labels != 0).sum()
        # if the labelled area for a class is small, a finer grid chop might be needed to get pixels into each split
        if num_pixels < 1000:
            labels = grid_chop(labels, grid_size=25)
        elif num_pixels < 10000:
            labels = grid_chop(labels, grid_size=150)
        else:
            labels = grid_chop(labels, grid_size=250)
        numb_objects = labels.max()
        objects = np.random.choice(np.arange(1, numb_objects + 1), numb_objects, replace=False)
        numb_per_split = np.ceil(
            np.array([numb_objects * train_ratio, numb_objects * (train_ratio + val_ratio), numb_objects])).astype(int)
        training_objects = objects[:numb_per_split[0]]
        val_objects = objects[numb_per_split[0]:numb_per_split[1]]
        test_objects = objects[numb_per_split[1]:]

        mask_tr = mask_tr | np.isin(labels, training_objects)
        mask_va = mask_va | np.isin(labels, val_objects)
        mask_te = mask_te | np.isin(labels, test_objects)
    return mask_tr.flatten(), mask_va.flatten(), mask_te.flatten()


def grid_chop(labels, grid_size=150):
    x, y = np.array([np.arange(labels.shape[1])] * labels.shape[0]), np.array(
        [np.arange(labels.shape[0])] * labels.shape[1]).T
    div_x, div_y = x // grid_size, y // grid_size
    grid_idxs = div_x + div_y * div_x.max()
    class_mask = labels != 0
    grid_id_nums = set(grid_idxs[class_mask])
    object_numb = 1
    for grid_id in grid_id_nums:
        grid_mask = grid_idxs == grid_id
        object_nums = set(labels[grid_mask]).difference({0})
        for obj_num in object_nums:
            labels[grid_mask & (labels == obj_num)] = object_numb
            object_numb += 1
    return labels


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def log_error(error_type, message):
    logging.critical(message)
    return error_type(message)
