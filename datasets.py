import scipy.sparse
from spektral.data import Dataset, Graph
import os
import numpy as np
from rs import hyperspectral, lidar, rgb, ground_truth
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder

from utils import masks_from_gt


class HoustonDataset_k2(Dataset):
    def __init__(self, **kwargs):
        self.mask_tr = self.mask_va = self.mask_te = None
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


class Karate(Dataset):
    def __init__(self, **kwargs):
        self.mask_tr = self.mask_va = self.mask_te = None
        super().__init__(**kwargs)

    def download(self):
        a = scipy.sparse.csr_matrix(np.load("karate_adj.npy"))
        x = np.eye(a.shape[0])
        y = np.load("karate_labels.npy")

        # Create the directory
        os.mkdir(self.path)

        filename = os.path.join(self.path, f'graph')
        np.savez(filename, x=x, a=a, y=OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray())

    def read(self):
        data = np.load(os.path.join(self.path, f'graph.npz'), allow_pickle=True)
        x, a, y = data['x'].astype(np.float32), data['a'].tolist(), data['y'].astype(np.uint8)

        # Train/valid/test masks
        self.mask_tr = np.array([True] + [False] * 32 + [True])
        self.mask_va = np.array([False if i % 2 == 0 else True for i in range(34)])
        self.mask_va[[0, -1]] = False
        self.mask_te = ~(self.mask_tr + self.mask_va)

        return [Graph(x=x, a=a, y=y)]


class HoustonDatasetMini(Dataset):
    def __init__(self, **kwargs):
        self.mask_tr = self.mask_va = self.mask_te = None
        super().__init__(**kwargs)

    def download(self):
        gt = ground_truth.data.array[0]
        y = gt[:, 1800:3600].reshape(-1, 1)
        x = np.concatenate([rgb.training_array(), hyperspectral.training_array(), lidar.training_array()])
        x = x[:, :, 1800:3600]
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

        self.mask_tr, self.mask_va, self.mask_te = masks_from_gt(ground_truth.data.array[0][:, 1800:3600])

        return [Graph(x=x, a=a, y=y)]
