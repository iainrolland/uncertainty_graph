import spektral.data.loaders
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from spektral.data import SingleLoader
from spektral.models import GCN
from spektral.utils import normalized_adjacency
import numpy as np
from scipy import sparse
from datetime import date
import networkx as nx
import os

import layers
import losses
from houston_dataset import HoustonDataset
import models

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # tf.config.experimental.set_visible_devices(gpus, 'GPU')
            tf.config.experimental.set_visible_devices([], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # learning_rate = 1e-2
    # seed = 0
    # epochs = 200
    # patience = 15
    #
    # tf.random.set_seed(seed=seed)  # make weight initialization reproducible
    #
    # dataset = HoustonDataset()
    #
    #
    # def mask_to_weights(mask):
    #     # convert binary masks to sample weights to compute average loss over the nodes
    #     return mask.astype(np.float32) / np.count_nonzero(mask)
    #
    #
    # def weight_training_mask(mask, y):
    #     numb_class_samples = y.sum(axis=0)
    #     mask = mask.astype(np.float32)
    #     for i, n in enumerate(numb_class_samples):
    #         if n != 0:
    #             mask[y[:, i] == 1] /= n * 5
    #     return mask / mask.sum()
    #
    #
    # # training, validation and test sample weights (used when calculating average loss) from their respective masks
    # weights_va, weights_te = (
    #     mask_to_weights(mask)
    #     for mask in (dataset.mask_va, dataset.mask_te)
    # )
    # weights_tr = weight_training_mask(dataset.mask_tr, dataset[0].y)
    # print(weights_tr)
    # model = models.GCN(n_labels=dataset.n_labels, n_input_channels=dataset.n_node_features)
    # model.compile(
    #     optimizer=Adam(learning_rate),
    #     loss=losses.masked_square_error_dirichlet(weights_tr),
    #     weighted_metrics=["acc"],
    # )
    # # Loaders iterate over a graph dataset to create mini-batches
    # loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
    # loader_va = SingleLoader(dataset, sample_weights=weights_va)
    #
    #
    # def get_unique_filename(directory, name, number=0):
    #     if not os.path.isfile(os.path.join(directory, name.format(number))):
    #         return os.path.join(directory, name.format(number))
    #     else:
    #         return get_unique_filename(directory, name, number + 1)
    #
    #
    # model_name = "gcn{}_{}.h5".format("{}", date.isoformat(date.today()).replace('-', '_'))
    # model.fit(
    #     loader_tr.load(),
    #     steps_per_epoch=loader_tr.steps_per_epoch,
    #     validation_data=loader_va.load(),
    #     validation_steps=loader_va.steps_per_epoch,
    #     epochs=epochs,
    #     callbacks=[EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
    #                ModelCheckpoint(get_unique_filename("models", model_name))],
    # )
    # # # Evaluate model
    # # print("Evaluating model.")
    # # loader_te = SingleLoader(dataset, sample_weights=weights_va)
    # # eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    # # print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))
    # loader_all = SingleLoader(dataset)
    # predictions = model.predict(loader_all.load(), steps=loader_all.steps_per_epoch)
    # np.save(get_unique_filename("houston_data/np_arrays", "predictions{}.npy"), predictions)

    # a = np.array([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 1, 0], [0, 0, 1, 0, 1]]).astype(
    #     np.float32)
    # a = sparse.csr_matrix(a)
    # a = normalized_adjacency(a)
    # x = np.array([[-.5, -.5], [.5, .5], [.2, .7], [.7, .2], [-.4, -.6]]).astype(np.float32)

    adj = normalized_adjacency(sparse.csr_matrix(np.load("karate_adj.npy")))
    x = np.eye(adj.shape[0])
    yt = tf.keras.utils.to_categorical(np.load("karate_labels.npy"))
    mask_tr = np.array([0.5] + [0] * 32 + [0.5])
    print(yt[0], yt[-1])

    m = models.GCN(2, output_activation="softmax", dropout_rate=.5)
    # m = spektral.models.GCN(2, output_activation="softmax", use_bias=True)
    m.compile(loss=losses.masked_categorical_cross_entropy(mask_tr), metrics=["acc"])
    m.fit((x, adj), yt, epochs=200, batch_size=adj.shape[0],
          callbacks=[EarlyStopping(monitor="acc", patience=30, restore_best_weights=True),
                     ])
    out = m.predict((x, adj), batch_size=adj.shape[0])
    # alpha = np.exp(out) + 1
    # prob = alpha / alpha.sum(axis=1)[:, None]

    print(np.array(adj))
    print(out)
    print(np.vstack([out.argmax(axis=1)[None, :], yt.argmax(axis=1)[None, :]]))
    print(m.losses)
    # dlt = losses.masked_square_error_dirichlet(np.ones(5))
    # yt = tf.keras.utils.to_categorical(np.array([0, 0, 1, 0, 0]))
    # yp = tf.cast(np.log(np.array([[2, 2], [10000, 2], [2, 10000], [2, 2], [2, 2]]) - 1), "float32")
    # print(dlt(yt, yp))
