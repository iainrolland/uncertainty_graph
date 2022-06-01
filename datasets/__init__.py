# from .HoustonDataset_k2 import *
# from .HoustonDatasetMini import *
# from .HoustonSpixelMini import *
from .Karate import *
import sys

sys.path.insert(1, "/media/imr27/SharedDataPartition/PythonProjects/damage-assessment/")
from BeirutDataset import BeirutDataset


def get_dataset(data_name):
    supported_datasets = dict(zip(["BeirutDataset"],
                                  [BeirutDataset]))
    try:
        return supported_datasets[data_name]
    except KeyError:
        raise ValueError(
            "{} was not a recognised dataset. Must be one of {}.".format(data_name, "/".join(supported_datasets)))
