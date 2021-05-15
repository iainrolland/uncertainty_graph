import os

import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from skimage.segmentation import slic
from sklearn.preprocessing import OneHotEncoder
from spektral.data import Dataset, Graph

from rs import ground_truth, rgb, hyperspectral, lidar
from utils import masks_from_gt


class HoustonSpixelMini(Dataset):
    def __init__(self, **kwargs):
        self.mask_tr, self.mask_va, self.mask_te = None, None, None
        self.segments, self.spixel_x, self.spixel_adj = None, None, None
        super().__init__(**kwargs)

    def download(self):
        raise NotImplementedError("Not implemented...")

    def read(self):
        data = np.load(os.path.join(self.path, f'graph.npz'), allow_pickle=True)
        x, adj_full, y = data['x'].astype(np.float32), data['a'].tolist(), data['y'].astype(np.uint8)
        self.segments, self.spixel_x, self.spixel_adj = data['segments'], data['spixel_x'], data['spixel_adj'].tolist()

        self.mask_tr, self.mask_va, self.mask_te = masks_from_gt(ground_truth.data.array[0][:, 1800:3600])

        return [Graph(x=x, a=adj_full, y=y)]
