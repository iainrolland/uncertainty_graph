import os

import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder
from spektral.data import Dataset, Graph

from rs import ground_truth, rgb, hyperspectral, lidar
from utils import masks_from_gt


class HoustonDataset_k2(Dataset):
    def __init__(self, **kwargs):
        self.mask_tr, self.mask_va, self.mask_te = None, None, None
        super().__init__(**kwargs)

    def download(self):
        y = ground_truth.data.array[0].reshape(-1, 1)
        x = np.concatenate([rgb.training_array(), hyperspectral.training_array(), lidar.training_array()])
        x = np.rollaxis(x.reshape(x.shape[0], -1), 0, 2)
        print("Computing k neighbors graph...")
        a = kneighbors_graph(x, 15, include_self=False)
        a = a + a.T  # to make graph symmetric (using k neighbours in "either" rather than "mutual" mode)
        a[a > 1] = 1  # get rid of any edges we just made double
        print("Graph computed.")

        # Create the directory
        os.mkdir(self.path)

        filename = os.path.join(self.path, f'graph')
        np.savez(filename, x=x, a=a, y=OneHotEncoder().fit_transform(y).toarray())

    def read(self):
        data = np.load(os.path.join(self.path, f'graph.npz'), allow_pickle=True)
        x, a, y = data['x'].astype(np.float32), data['a'].tolist(), data['y'].astype(np.uint8)

        # Train/valid/test masks
        # self.mask_tr = np.array([True if 1436 <= n % 4768 <= 2000 else False for n in range(len(y))])
        # self.mask_va = np.array([True if n % 4768 < 1436 else False for n in range(len(y))])
        # self.mask_te = np.array([True if n % 4768 > 2000 else False for n in range(len(y))])
        # # Remove 'Unclassified':
        # # - from training so we don't learn to predict it as an output class
        # # - from validation so the classification of unlabelled nodes doesn't contribute to val_loss
        # # - from test because the classification of unlabelled nodes shouldn't contribute to test_acc
        # self.mask_tr[np.argwhere(y[:, 0] == 1).flatten()] = False
        # self.mask_va[np.argwhere(y[:, 0] == 1).flatten()] = False
        # self.mask_te[np.argwhere(y[:, 0] == 1).flatten()] = False
        self.mask_tr, self.mask_va, self.mask_te = masks_from_gt(ground_truth.data.array[0])

        return [Graph(x=x, a=a, y=y)]
