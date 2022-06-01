import argparse
import os
from subprocess import check_call
import sys
from glob import glob

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


def houston_ood():
    # Perform hypersearch over parameters
    seed = 1
    ood_classes = [[4, 2], [16, 13], [1, 10], [8, 12], [2, 13], [16, 11], [5, 4], [9, 13], [13, 1], [7, 12]]
    # alpha_prior_path = "experiments/Sampled_OOD_classes/spixel_{}_{}_prior.npy"
    # teacher_file_path = "experiments/Sampled_OOD_classes/ood_classes_{}{}_model_GCN/prob.npy"

    json_path = "experiments/Sampled_OOD_classes/params.json"
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.model = "S-BMLP"

    for ood_c in ood_classes:
        job_name = "ood_classes_{}{}_model_S-BMLP".format(*ood_c)

        # params.alpha_prior_path = alpha_prior_path.format(*ood_c)
        # params.teacher_file_path = teacher_file_path.format(*ood_c)
        params.ood_classes = ood_c

        # Launch job (name has to be unique)
        launch_training_job(args.parent_dir, job_name, params)

if __name__ == "__main__":
    args = parser.parse_args()

    json_path = sorted(glob("experiments/Beirut/misclassification_tests/*/"))

    for jp in json_path:
        assert os.path.isfile(jp + "params.json"), "No json configuration file found at {}".format(json_path)

        if not os.path.isfile(jp + "prob.npy"):
            # Launch training with this config
            cmd = "{python} training.py --model_dir {model_dir}".format(python=PYTHON, model_dir=jp)
            print(cmd)
            check_call(cmd, shell=True)
