import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
import rgb_mosaic


def tif_list_overlap_bounds(tif_list):
    left, bottom, right, top = tif_list[0].bounds
    for tif in tif_list[1:]:
        left = left if tif.bounds[0] < left else tif.bounds[0]
        bottom = bottom if tif.bounds[1] < bottom else tif.bounds[1]
        right = right if tif.bounds[2] > right else tif.bounds[2]
        top = top if tif.bounds[3] > top else tif.bounds[3]
    return left, bottom, right, top


def tif_list_stack_overlap(tif_list):
    left, bottom, right, top = tif_list_overlap_bounds(tif_list)
    data_list = []
    for tif in tif_list:
        l, t = ~tif.transform * (left, top)
        r, b = ~tif.transform * (right, bottom)
        data_list.append(np.rollaxis(tif.read(), 0, 3)[int(t):int(b), int(l):int(r)])
    return np.concatenate(data_list, -1)


def make_tif_list(file_path_list):
    tif_list = []
    for tif in file_path_list:
        if not isinstance(tif, str):  # Check file path list contains strings
            raise TypeError("List must contain strings but contained element of type %s" % type(tif))
        else:
            if not os.path.isfile(tif):  # Check file path is a valid file
                raise ValueError("List elements must be valid file paths but could not find a " +
                                 "file at the following location: %s" % tif)
            else:
                tif_list.append(rasterio.open(tif))
    return tif_list


files = ['houston_data/Lidar GeoTiff Rasters/DEM+B_C123/UH17_GEM051.tif',
         'houston_data/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif']
tif_list = make_tif_list(files) + [rgb_mosaic.get_tif()]

array = tif_list_stack_overlap(tif_list)

fig, ax = plt.subplots(array.shape[-1])
for i in range(array.shape[-1]):
    print(array[..., i].shape)
    ax[i].matshow(array[..., i])
plt.show()
