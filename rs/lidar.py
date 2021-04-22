import rasterio

from .utils import get_training_array


def training_array():
    tif = rasterio.open("./houston_data/Lidar GeoTiff Rasters/DEM+B_C123/UH17_GEM051.tif")
    return get_training_array(tif)
