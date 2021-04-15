import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import ground_truth
from abstract_mode import Data


class Lidar(Data):
    def __init__(self):
        super().__init__()
        file_path = "houston_data/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix"
        tif = rasterio.open(file_path)
        self.array = tif.read(window=from_bounds(*ground_truth.data.labelled_bounds + (tif.transform,)),
                              out_shape=(tif.count, ground_truth.data.height, ground_truth.data.width),
                              resampling=Resampling.bilinear)
        self.normalise(tif.nodata)


data = Lidar()
