import os

import numpy as np
import scipy.sparse
from sklearn.preprocessing import OneHotEncoder
from spektral.data import Dataset, Graph


class Karate(Dataset):
    def __init__(self, **kwargs):
        self.mask_tr, self.mask_va, self.mask_te = None, None, None
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
