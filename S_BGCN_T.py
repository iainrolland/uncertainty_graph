import os
import numpy as np
from training import train
from evaluation import evaluate
from experiments import S_BGCN_T
from datasets import Karate, HoustonDatasetMini
from spektral.datasets import Citation
from utils import gpu_initialise, set_seeds, set_logger, log_error
from params import Params

params = Params("config_files/default_S_BGCN_T.json")
set_logger(os.path.join(params.directory, "output.log"))
gpu_initialise(params.gpu_list)
set_seeds(params.seed)
if params.data == "HoustonDatasetMini":
    data = HoustonDatasetMini(transforms=S_BGCN_T.transforms)
elif params.data == "Karate":
    data = Karate(transforms=S_BGCN_T.transforms)
elif params.data in ["cora", "citeseer", "pubmed"]:
    data = Citation(name=params.data, transforms=S_BGCN_T.transforms)
else:
    raise log_error(ValueError, "{} not recognised as a dataset.".format(params.data))

supported_models = ["S-BGCN-T"]
if params.model not in supported_models:
    raise ValueError("model was {} but must be one of {}.".format(params.model, "/".join(supported_models)))

if len(params.ood_classes) > 0:
    data.mask_tr[np.argwhere(np.isin(data[0].y.argmax(axis=-1), params.ood_classes)).flatten()] = False
    test_ood = True
else:
    test_ood = False

network = train(S_BGCN_T("experiments/GCN_HoustonDatasetMini_2021_04_29_1/prob.npy", params.teacher_coefficient), data,
                params)
evaluate(network, data, params, test_ood_detection=test_ood)
params.save()
