import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from glob import glob
from tif_manip import resample_tif
import ground_truth
from abstract_mode import Data


class RGB(Data):
    def __init__(self):
        super().__init__()
        file_directory = "houston_data/Final RGB HR Imagery"
        tif = mosaic(file_directory)
        self.array = tif.read(window=from_bounds(*ground_truth.data.labelled_bounds + (tif.transform,)),
                              out_shape=(tif.count, ground_truth.data.height, ground_truth.data.width),
                              resampling=Resampling.bilinear)
        self.normalise(tif.nodata)


def mosaic(directory):
    file_paths = glob(directory + "/*.tif")
    tif_list = []
    for path in file_paths:
        tif = rasterio.open(path)
        tif_list.append(resample_tif(tif, tif.transform[0] / ground_truth.data.resolution))
    mosaic_tif, mosaic_transform = merge(tif_list)
    profile = tif_list[0].profile
    profile.update({'width': mosaic_tif.shape[-1],
                    'height': mosaic_tif.shape[-2],
                    'transform': mosaic_transform})
    memory_file = MemoryFile().open(**profile)
    memory_file.write(mosaic_tif)
    return memory_file


data = RGB()
