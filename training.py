import utils
from spektral.models import GCN
from spektral.data import SingleLoader
import tensorflow as tf
import numpy as np
import os
from datetime import date


def train(model, data, learning_rate=1e-2, l2_loss_coefficient=5e-4, epochs=200, patience=10, seed=0,
          gpu_list=None):
    if gpu_list is None:
        gpu_list = []

    if data not in utils.supported_datasets.keys():
        error_message = "Dataset by the name {} is not known, must be one of {}"
        raise ValueError(
            error_message.format(model, ", ".join(list(utils.supported_datasets.keys())))
        )

    utils.gpu_initialise(gpu_list)

    tf.random.set_seed(seed=seed)  # make weight initialization reproducible

    dataset = utils.supported_datasets[data](transforms=model.transforms)

    save_directory = utils.make_unique_directory(
        "models/" + model.__name__ + "_" + data + "_" + date.isoformat(date.today()).replace("-", "_") + "_{}")

    weights_va, weights_te = (
        utils.mask_to_weights(mask)
        for mask in (dataset.mask_va, dataset.mask_te)
    )
    weights_tr = utils.weight_by_class(dataset[0].y, dataset.mask_tr)

    network = GCN(n_labels=dataset.n_labels, n_input_channels=dataset.n_node_features,
                  output_activation=model.output_activation, l2_reg=l2_loss_coefficient)
    network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=model.loss,
        weighted_metrics=["acc"]
    )

    # Train model
    loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
    loader_va = SingleLoader(dataset, sample_weights=weights_va)
    network.fit(
        loader_tr.load(),
        steps_per_epoch=loader_tr.steps_per_epoch,
        validation_data=loader_va.load(),
        validation_steps=loader_va.steps_per_epoch,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
                   tf.keras.callbacks.ModelCheckpoint(os.path.join(save_directory, model.__name__ + ".h5"),
                                                      monitor="val_loss", save_best_only=True,
                                                      save_weights_only=False)],
    )

    # Evaluate model
    print("Evaluating model.")
    loader_te = SingleLoader(dataset, sample_weights=weights_te)
    eval_results = network.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))

    loader_all = SingleLoader(dataset, epochs=1)
    output = network.predict(loader_all.load())
    output_name = "alpha" if model.__name__ != "GCN" else "prob_pred"
    np.save(os.path.join(save_directory, output_name + ".npy"), output)
