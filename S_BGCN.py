from spektral.datasets import Citation
import os
import numpy as np
from params import Params
from training import train
from evaluation import evaluate
from experiments import S_BGCN
from datasets import Karate, HoustonDatasetMini
from utils import gpu_initialise, set_seeds, set_logger, log_error

params = Params("config_files/default_S_BGCN.json")
set_logger(os.path.join(params.directory, "output.log"))
gpu_initialise(params.gpu_list)
set_seeds(params.seed)
if params.data == "HoustonDatasetMini":
    data = HoustonDatasetMini(transforms=S_BGCN.transforms)
elif params.data == "Karate":
    data = Karate(transforms=S_BGCN.transforms)
elif params.data in ["cora", "citeseer", "pubmed"]:
    data = Citation(name=params.data, transforms=S_BGCN.transforms)
else:
    raise log_error(ValueError, "{} not recognised as a dataset.".format(params.data))

supported_models = ["S-BGCN", "S-GCN"]
if params.model not in supported_models:
    raise ValueError("model was {} but must be one of {}.".format(params.model, "/".join(supported_models)))

if len(params.ood_classes) > 0:
    data.mask_tr[np.argwhere(np.isin(data[0].y.argmax(axis=-1), params.ood_classes)).flatten()] = False
    test_ood = True
else:
    test_ood = False

network = train(S_BGCN, data, params)
evaluate(network, data, params, test_ood_detection=test_ood)
params.save()
