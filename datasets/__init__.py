from .HoustonDataset_k2 import *
from .HoustonDatasetMini import *
from .Karate import *


def get_dataset(data_name):
    supported_datasets = dict(zip(["HoustonDataset_k2", "HoustonDatasetMini", "Karate"],
                                  [HoustonDataset_k2, HoustonDatasetMini, Karate]))
    try:
        return supported_datasets[data_name]
    except KeyError:
        raise ValueError(
            "{} was not a recognised dataset. Must be one of {}.".format(data_name, "/".join(supported_datasets)))
