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


def get_metric(model, weight):
    def gcn_conv_0_l2_reg_loss(y_true, y_pred):
        return tf.nn.l2_loss(model.layers[1].kernel) * weight

    return gcn_conv_0_l2_reg_loss


def train(model, dataset, params):
    weights_va, weights_te = (
        utils.mask_to_weights(mask).astype(np.float32)
        for mask in (dataset.mask_va, dataset.mask_te)
    )
    weights_tr = utils.weight_by_class(dataset[0].y, dataset.mask_tr).astype(np.float32)

    neural_net = GCN(n_labels=dataset.n_labels, channels=params.channels, n_input_channels=dataset.n_node_features,
                     output_activation=model.output_activation, l2_reg=params.l2_loss_coefficient)
    neural_net.compile(
        optimizer=tf.keras.optimizers.Adam(params.learning_rate),
        loss=model.loss,
        weighted_metrics=["acc", get_metric(neural_net, params.l2_loss_coefficient)]
    )

    # Train model
    loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
    loader_va = SingleLoader(dataset, sample_weights=weights_va)
    history = neural_net.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=params.epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=params.patience,
                                                    restore_best_weights=True),
                   tf.keras.callbacks.ModelCheckpoint(os.path.join(params.directory, model.__name__ + ".h5"),
                                                      monitor="val_loss", save_best_only=True,
                                                      save_weights_only=False)],
    )

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

    # Check that we are not overwriting some previous experiment
    if len(glob(os.path.join(args.model_dir, "*.h5"))) > 0:
        raise utils.log_error(AssertionError, "Weights found in model_dir, aborting to avoid overwrite")

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

    network = train(model_type, data, parameters)
    evaluate(network, data, parameters, test_ood_detection=test_ood)
