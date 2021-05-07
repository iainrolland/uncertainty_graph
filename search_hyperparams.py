import argparse
import os
from subprocess import check_call
import sys

from params import Params

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help="Directory containing params.json")


def launch_training_job(parent_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        parent_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} training.py --model_dir {model_dir}".format(python=PYTHON, model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from .json file
    args = parser.parse_args()
    json_path = "experiments/Sampled_OOD_classes/params.json"
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Perform hypersearch over one parameter
    # learning_rates = [1e-4, 1e-3, 1e-2]
    # channels = [8, 16, 32, 64]
    # l2_loss_coefficients = [1e-4, 1e-3, 1e-2]
    # seeds = [0, 1, 2, 3]
    ood_classes_list = [[7, 12], [1, 10], [13, 1], [5, 4], [9, 13], [8, 12], [4, 2], [2, 13], [16, 13], [16, 11]]
    model_names = ["GCN", "Drop-GCN", "S-GCN", "S-BGCN", "S-BGCN-T", "S-BGCN-T-K"]

    for ood_classes in ood_classes_list:
        for model_name in model_names:
            # Modify the relevant parameter in params
            params.ood_classes = ood_classes
            params.model = model_name

            # Launch job (name has to be unique)
            job_name = "ood_classes_{}{}_model_{}".format(*ood_classes, model_name)
            launch_training_job(args.parent_dir, job_name, params)
