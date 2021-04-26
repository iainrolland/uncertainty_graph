from spektral.models import GCN
from spektral.data import SingleLoader
import tensorflow as tf
import numpy as np
import os
from datetime import date

import utils
import uncertainty_utils


def train(model, dataset, learning_rate=1e-2, l2_loss_coefficient=5e-4, epochs=200, patience=10, seed=0):
    tf.random.set_seed(seed=seed)  # make weight initialization reproducible
    data_name = dataset.__class__.__name__

    save_directory = utils.make_unique_directory(
        "models/" + model.__name__ + "_" + data_name + "_" + date.isoformat(date.today()).replace("-", "_") + "_{}")

    weights_va, weights_te = (
        utils.mask_to_weights(mask).astype(np.float32)
        for mask in (dataset.mask_va, dataset.mask_te)
    )
    weights_tr = utils.weight_by_class(dataset[0].y, dataset.mask_tr).astype(np.float32)

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
    history = network.fit(
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

    if model.__name__ != "GCN":
        output_name = "alpha"
        vacuity = uncertainty_utils.vacuity_uncertainty(output)
        dissonance = uncertainty_utils.dissonance_uncertainty(output)
        np.save(os.path.join(save_directory, "vacuity.npy"), vacuity)
        np.save(os.path.join(save_directory, "dissonance.npy"), dissonance)
        unc_dict = uncertainty_utils.misclassification(output, vacuity, dissonance, dataset)
        auroc = [(unc, unc_dict[unc]["auroc"]) for unc in unc_dict]
        aupr = [(unc, unc_dict[unc]["aupr"]) for unc in unc_dict]

        print("Misclassification AUROC: ", *[unc_name + " = " + str(score) for unc_name, score in auroc])
        print("Misclassification AUPR: ", *[unc_name + " = " + str(score) for unc_name, score in aupr])

        unc_dict = uncertainty_utils.ood_detection(vacuity, dissonance, dataset)
        auroc = [(unc, unc_dict[unc]["auroc"]) for unc in unc_dict]
        aupr = [(unc, unc_dict[unc]["aupr"]) for unc in unc_dict]

        print("OOD Detection AUROC: ", *[unc_name + " = " + str(score) for unc_name, score in auroc])
        print("OOD Detection AUPR: ", *[unc_name + " = " + str(score) for unc_name, score in aupr])
    else:
        output_name = "prob_pred"

    np.save(os.path.join(save_directory, output_name + ".npy"), output)
    np.save("history", history.history["val_loss"])
