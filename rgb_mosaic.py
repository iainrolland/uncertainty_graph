from glob import glob

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile

from rasterio.merge import merge

rgb_directory = 'houston_data/Final RGB HR Imagery'  # directory containing RGB files
pixels_per_meter = 0.5  # files are 0.05 GSD but down-sampling to 0.5 GSD ensures a match with other mode resolutions


def get_tif():
    file_paths = glob(rgb_directory + "/*.tif")
    tif_list = []
    for tif_path in file_paths:
        tif = rasterio.open(tif_path)
        upscale_factor = tif.transform[0] / pixels_per_meter
        data = tif.read(
            out_shape=(
                tif.count,
                int(tif.height * upscale_factor),
                int(tif.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )
        # scale image transform
        transform = tif.transform * tif.transform.scale(
            (tif.width / data.shape[-1]),
            (tif.height / data.shape[-2])
        )
        profile = tif.profile
        profile.update({'width': int(tif.width * upscale_factor),
                        'height': int(tif.height * upscale_factor),
                        'transform': transform})
        memory_file = MemoryFile().open(**profile)
        memory_file.write(data)
        tif_list.append(memory_file)

    mosaic, mosaic_transform = merge(tif_list)
    profile = tif_list[0].profile
    profile.update({'width': mosaic.shape[-1],
                    'height': mosaic.shape[-2],
                    'transform': mosaic_transform})
    memory_file = MemoryFile().open(**profile)
    memory_file.write(mosaic)
    return memory_file
