from spektral.models import GCN
from spektral.data import SingleLoader
import tensorflow as tf
import numpy as np
import os
import argparse
from glob import glob

from params import Params
from models import get_model
import datasets
import utils
from evaluation import evaluate


def train(model, dataset, params):
    neural_net = model.get_network(params, dataset.n_node_features, dataset.n_labels)
    model.compile_network(params)

    # Train model
    history = model.fit_network(params, dataset)

    np.save(os.path.join(params.directory, "history"), history.history["val_loss"])
    return neural_net


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help="Experiment directory containing params.json")
    args = parser.parse_args()

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    if not os.path.isfile(json_path):
        raise utils.log_error(AssertionError, "No json configuration file found at {}".format(json_path))
    parameters = Params(json_path)
    parameters.directory = args.model_dir

    # # Check that we are not overwriting some previous experiment
    # if len(glob(os.path.join(args.model_dir, "*.h5"))) > 0:
    #     raise utils.log_error(AssertionError, "Weights found in model_dir, aborting to avoid overwrite")

    # Set to use the specified GPUs
    utils.gpu_initialise(parameters.gpu_list)
    # Set random seeds (affects dataset train/val/test split as well as model weight initialisation)
    utils.set_seeds(parameters.seed)

    # Load the specifics used to train a model of type described by parameters.model
    try:
        model_type = get_model(parameters)
    except (ValueError, AttributeError) as err:
        raise utils.log_error(ValueError, err)

    # Load dataset
    try:
        data = datasets.get_dataset(parameters.data)(transforms=model_type.transforms)
    except ValueError as err:
        raise utils.log_error(ValueError, err)

    # Create boolean indicating whether some classes have been hidden to act as OOD
    if len(parameters.ood_classes) > 0:
        data.mask_tr[np.argwhere(np.isin(data[0].y.argmax(axis=-1), parameters.ood_classes)).flatten()] = False
        test_ood = True
    else:
        test_ood = False

    if len(glob(os.path.join(args.model_dir, "*.h5"))) > 0:  # if model_weights exist in directory -> load
        model = get_model(parameters)
        model.get_network(parameters, data.n_node_features, data.n_labels)
        network = model.network
        # network.predict(np.ones((1, data.n_node_features)))  # dummy predict in order to build correct dims
        network.predict(
            (np.ones((2, data.n_node_features)), np.ones((2, 2))))  # dummy predict in order to build correct dims
        network.load_weights(os.path.join(args.model_dir, model.__name__ + ".h5"))
    elif len(glob(os.path.join(args.model_dir, "*.new_ext"))) > 0:  # if model_weights exist in directory -> load
        model = get_model(parameters)
        model.get_network(parameters, data.n_node_features, data.n_labels)
        network = model.network
        network.predict((np.ones((2, data.n_node_features)), np.ones((2, 2))))  # dummy predict in order to build correct dims
        network.load_weights(os.path.join(args.model_dir, model.__name__ + ".new_ext"))
 
    else:
        network = train(model_type, data, parameters)

    evaluate(network, data, parameters, test_ood_detection=test_ood)
