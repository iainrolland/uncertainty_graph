import os
import tensorflow as tf
import rasterio
import numpy as np
from uncertainty_utils import vacuity_uncertainty

model_dir = "experiments/S_BGCN_HoustonDataset_k2_2021_04_26_2"
alpha = np.load(os.path.join(model_dir, "alpha.npy"))
vacuity = np.load(os.path.join(model_dir, "vacuity.npy"))
dissonance = np.load(os.path.join(model_dir, "dissonance.npy"))

tif = rasterio.open("houston_data/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif")
profile = tif.profile

with rasterio.open(os.path.join(model_dir, "alpha.tif"), "w", **profile) as op:
    op.write(alpha.argmax(axis=1).reshape(-1, 4768).astype(np.uint8), 1)

profile.update({"dtype": np.float32})

with rasterio.open(os.path.join(model_dir, "vacuity.tif"), "w", **profile) as op:
    op.write(vacuity.reshape(-1, 4768).astype(np.float32), 1)

with rasterio.open(os.path.join(model_dir, "dissonance.tif"), "w", **profile) as op:
    op.write(dissonance.reshape(-1, 4768).astype(np.float32), 1)
