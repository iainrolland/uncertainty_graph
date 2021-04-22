import os
import numpy as np
import tensorflow as tf
from datetime import date
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from spektral.data.loaders import SingleLoader
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.transforms import AdjToSpTensor, LayerPreprocess

import losses
from datasets import Karate, HoustonDataset
from utils import make_unique_directory, mask_to_weights, class_weights, gpu_initialise

learning_rate = 1e-2
l2_loss_coefficient = 5e-4
gpu_list = []
seed = 0
epochs = 200
patience = 10
data = "houston"

gpu_initialise(gpu_list)

tf.random.set_seed(seed=seed)  # make weight initialization reproducible

if data == "houston":
    dataset = HoustonDataset(transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()])
elif data == "karate":
    dataset = Karate(transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()])
else:
    raise ValueError(
        "Dataset by the name {} is not known, must be one of {}".format(data, ", ".join(["houston", "karate"])
                                                                        )
    )

save_directory = make_unique_directory(
    "models/S_GCN_" + data + "_" + date.isoformat(date.today()).replace("-", "_") + "_{}")

weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)

model = GCN(n_labels=dataset.n_labels, n_input_channels=dataset.n_node_features,
            output_activation=lambda foo: tf.exp(foo) + 1, l2_reg=l2_loss_coefficient)
model.compile(
    optimizer=Adam(learning_rate),
    weighted_metrics=["acc"]
)

# Train model
loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
loader_va = SingleLoader(dataset, sample_weights=weights_va)
model.fit(
    loader_tr.load(),
    class_weight=class_weights(dataset[0].y, dataset.mask_tr),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
               ModelCheckpoint(os.path.join(save_directory, "S_GCN.h5"), monitor="val_loss", save_best_only=True,
                               save_weights_only=False)],
)

# Evaluate model
print("Evaluating model.")
loader_te = SingleLoader(dataset, sample_weights=weights_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))

loader_all = SingleLoader(dataset, sample_weights=np.ones(dataset[0].a.shape[0]) / dataset[0].a.shape[0], epochs=1)
alpha = model.predict(loader_all.load())
np.save(os.path.join(save_directory, "alpha.npy"), alpha)
