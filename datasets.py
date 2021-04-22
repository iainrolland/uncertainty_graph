import scipy.sparse
from spektral.data import Dataset, Graph
import os
import numpy as np
from rs import hyperspectral, lidar, rgb, ground_truth
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import OneHotEncoder


class HoustonDataset(Dataset):
    def __init__(self, **kwargs):
        self.mask_tr = self.mask_va = self.mask_te = None
        super().__init__(**kwargs)

    def download(self):
        y = np.rollaxis(ground_truth.data.array, 0, 3).reshape(-1, 1)
        x = np.concatenate([rgb.training_array(), hyperspectral.training_array(), lidar.training_array()])
        x = np.rollaxis(x.reshape(x.shape[0], -1), 0, 2)
        print("Computing k neighbors graph...")
        a = kneighbors_graph(x, 2, include_self=False)
        print("Graph computed.")

        # Create the directory
        os.mkdir(self.path)

        for i in range(1):
            filename = os.path.join(self.path, f'graph_{i}')
            np.savez(filename, x=x, a=a, y=OneHotEncoder().fit_transform(y).toarray())

    def read(self):
        # We must return a list of Graph objects
        output = []

        for i in range(1):
            data = np.load(os.path.join(self.path, f'graph_{i}.npz'), allow_pickle=True)
            output.append(
                Graph(x=data['x'].astype(np.float32), a=data['a'].tolist(), y=data['y'].astype(np.uint8))
            )

        # Train/valid/test masks
        self.mask_tr = np.array([True if 1436 <= n % 4768 <= 2000 else False for n in range(len(output[0].y))])
        self.mask_va = np.array([True if n % 4768 < 1436 else False for n in range(len(output[0].y))])
        self.mask_te = np.array([True if n % 4768 > 2000 else False for n in range(len(output[0].y))])
        self.mask_tr[np.argwhere(output[0].y[:, 0] == 1).flatten()] = False  # remove 'Unclassified' from training
        self.mask_va[np.argwhere(output[0].y[:, 0] == 1).flatten()] = False  # and validation (don't increase loss)
        # self.mask_tr = under_sample(output[0].y, self.mask_tr)
        print(output[0].y[self.mask_tr].sum(axis=0))

        return output


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

        for i in range(1):
            filename = os.path.join(self.path, f'graph_{i}')
            np.savez(filename, x=x, a=a, y=OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray())

    def read(self):
        # We must return a list of Graph objects
        output = []

        for i in range(1):
            data = np.load(os.path.join(self.path, f'graph_{i}.npz'), allow_pickle=True)
            output.append(
                Graph(x=data['x'].astype(np.float32), a=data['a'].tolist(), y=data['y'].astype(np.uint8))
            )

        # Train/valid/test masks
        self.mask_tr = np.array([True] + [False] * 32 + [True])
        self.mask_va = np.array([False if i % 2 == 0 else True for i in range(34)])
        self.mask_va[[0, -1]] = False
        self.mask_te = ~(self.mask_tr + self.mask_va)

        return output


def under_sample(y, mask):
    samples = y[mask].sum(axis=0)[y[mask].sum(axis=0) > 0].min()
    for i in range(y.shape[1]):
        samples_total = (y[mask, i] == 1).sum()
        if samples_total > 0:
            # make samples_total-samples of those False so that only samples number remain
            mask[np.random.choice(np.argwhere(mask & y[:, i] == 1).flatten(), int(samples_total - samples),
                                  replace=False)] = False
    return mask
