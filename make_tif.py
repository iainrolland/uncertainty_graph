import tensorflow as tf
import rasterio
import numpy as np
from utils import dissonance_uncertainty

alpha = np.load("houston_data/np_arrays/alpha.npy")
vacuity = alpha.shape[-1] / alpha.sum(axis=-1)
dissonance = dissonance_uncertainty(alpha)
# print(a.shape, tf.keras.utils.to_categorical(a.argmax(axis=1)).sum(axis=0), a[0])
tif = rasterio.open("houston_data/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif")
profile = tif.profile
# print(profile)
profile.update({"dtype": np.float32})

with rasterio.open("models/dissonance.tif", "w", **profile) as op:
    # op.write(a.argmax(axis=1).reshape(-1,4768).astype(np.uint8),1)
    op.write(dissonance.reshape(-1, 4768).astype(np.float32), 1)
