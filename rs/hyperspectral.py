import rasterio
import numpy as np

from .utils import get_training_array, read_tif_channels


def training_array():
    tif = rasterio.open("./houston_data/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix")
    tif = read_tif_channels(tif, np.arange(48)+1) # only the first 48 channels of the HS tif are actually HS channels (we don't use channel 49/50) 
    return get_training_array(tif)
