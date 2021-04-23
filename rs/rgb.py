import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from glob import glob
import rs.ground_truth
from .utils import get_training_array, resample_tif


def mosaic(directory):
    file_paths = glob(directory + "/*.tif")
    tif_list = []
    for path in file_paths:
        tif = rasterio.open(path)
        tif_list.append(resample_tif(tif, tif.transform[0] / rs.ground_truth.data.resolution))
    mosaic_tif, mosaic_transform = merge(tif_list)
    profile = tif_list[0].profile
    profile.update({'width': mosaic_tif.shape[-1],
                    'height': mosaic_tif.shape[-2],
                    'transform': mosaic_transform})
    memory_file = MemoryFile().open(**profile)
    memory_file.write(mosaic_tif)
    return memory_file


def training_array():
    file_directory = "./houston_data/Final RGB HR Imagery"
    return get_training_array(mosaic(file_directory))
