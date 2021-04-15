import rasterio


class GroundTruth:
    def __init__(self):
        file_path = "houston_data/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif"
        tif = rasterio.open(file_path)
        self.labelled_bounds = tif.bounds
        self.height = tif.height
        self.width = tif.width
        self.resolution = tif.transform[0]
        self.array = tif.read()


data = GroundTruth()
