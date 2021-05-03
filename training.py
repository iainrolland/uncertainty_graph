from spektral.models import GCN
from spektral.data import SingleLoader
import tensorflow as tf
import numpy as np
import os
from datetime import date

import utils
import uncertainty_utils


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

    network = GCN(n_labels=dataset.n_labels, channels=params.channels, n_input_channels=dataset.n_node_features,
                  output_activation=model.output_activation, l2_reg=params.l2_loss_coefficient)
    network.compile(
        optimizer=tf.keras.optimizers.Adam(params.learning_rate),
        loss=model.loss,
        weighted_metrics=["acc", get_metric(network, params.l2_loss_coefficient)]
    )

    # Train model
    loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
    loader_va = SingleLoader(dataset, sample_weights=weights_va)
    history = network.fit(
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

    np.save("history", history.history["val_loss"])
    return network
