import os
import numpy as np
from spektral.datasets import Citation
from training import train
from evaluation import evaluate
from models import S_BGCN_T
from datasets import Karate, HoustonDatasetMini
from utils import gpu_initialise, set_seeds, set_logger, log_error
from params import Params
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/test',
                    help="Experiment directory containing params.json")

if __name__ == "__main__":
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.directory = args.model_dir

    # Check that we are not overwriting some previous experiment
    model_dir_has_best_weights = len(glob(os.path.join(args.model_dir, "*.h5"))) > 0
    overwriting = model_dir_has_best_weights
    assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

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
        raise log_error(ValueError,
                        "model was {} but must be one of {}.".format(params.model, "/".join(supported_models)))

    if len(params.ood_classes) > 0:
        data.mask_tr[np.argwhere(np.isin(data[0].y.argmax(axis=-1), params.ood_classes)).flatten()] = False
        test_ood = True
    else:
        test_ood = False

    network = train(S_BGCN_T("experiments/GCN_HoustonDatasetMini_03_05_21_0/prob.npy",
                             teacher_coefficient=params.teacher_coefficient), data, params)
    evaluate(network, data, params, test_ood_detection=test_ood)