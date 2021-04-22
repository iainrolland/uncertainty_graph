import rasterio

from .utils import get_training_array


def training_array():
    tif = rasterio.open("./houston_data/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix")
    return get_training_array(tif)
