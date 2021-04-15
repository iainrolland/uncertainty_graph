import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio

import ground_truth
import rgb
import hyperspectral
import lidar

fig, ax = plt.subplots(nrows=4)
ax[0].matshow(ground_truth.data.array[0])
ax[1].matshow(lidar.data.array[0])
ax[2].imshow(np.rollaxis(rgb.data.array, 0, 3))
ax[3].imshow(np.rollaxis(hyperspectral.data.array[10:13], 0, 3))
plt.show()
