from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
import numpy as np

from rs import ground_truth


def get_training_array(tif):
    array = tif.read(window=from_bounds(*ground_truth.data.labelled_bounds + (tif.transform,)),
                     out_shape=(tif.count, ground_truth.data.height, ground_truth.data.width),
                     resampling=Resampling.bilinear)
    return normalise(array, tif.nodata)


def normalise(array, nodata):
    """Sets pixels with nodata value to zero then normalises each channel to between 0 and 1"""
    array[array == nodata] = 0
    return (array - array.min(axis=(1, 2))[:, None, None]) / (
        (array.max(axis=(1, 2)) - array.min(axis=(1, 2)))[:, None, None])

def read_tif_channels(tif, channel_index):
    if not isinstance(channel_index, list):
        channel_index = list(channel_index)
    if channel_index[0] == 0: # rasterio indexes channels starting from 1 not 0...
        channel_index = list(np.array(channel_index) + 1)
    profile = tif.profile
    profile.update({'count': len(channel_index)})
    memory_file = MemoryFile().open(**profile)
    memory_file.write(tif.read(channel_index))
    return memory_file


def resample_tif(tif, scale_factor, mode=Resampling.bilinear):
    data = tif.read(
        out_shape=(
            tif.count,
            int(tif.height * scale_factor),
            int(tif.width * scale_factor)
        ),
        resampling=mode
    )
    # scale image transform
    transform = tif.transform * tif.transform.scale(
        (tif.width / data.shape[-1]),
        (tif.height / data.shape[-2])
    )
    profile = tif.profile
    profile.update({'width': int(tif.width * scale_factor),
                    'height': int(tif.height * scale_factor),
                    'transform': transform})
    memory_file = MemoryFile().open(**profile)
    memory_file.write(data)
    return memory_file
