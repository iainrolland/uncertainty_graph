from spektral.datasets import Citation
from datasets import HoustonDatasetMini
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tensorflow.keras.utils import to_categorical
import pickle
from scipy import sparse
from tqdm import tqdm
from plotting.utils import close_fig
from utils import set_seeds


def update(state, truth, adjacency, mask, step_size=.2):
    new_state = ((1 - step_size) * sparse.identity(adjacency.shape[0]) + step_size * adjacency.dot(
        np.diag(adjacency.sum(axis=0).A1 ** -1))).dot(state)
    new_state[mask] = truth[mask]  # boundary conditions
    return new_state, np.mean(np.power((new_state - state), 2))


if __name__ == "__main__":
    # Load dataset
    set_seeds(0)
    cora = HoustonDatasetMini()
    adj = cora[0].a
    yt = cora[0].y
    mask_tr = cora.mask_tr

    # Set initial state
    y = np.ones_like(yt) / yt.shape[1]
    y[mask_tr] = yt[mask_tr]

    # Update until convergence
    y, l2 = update(y, yt, adj, mask_tr)
    l2_list = [l2]
    for t in range(500):
        print(t)
        l2_old = l2
        y, l2 = update(y, yt, adj, mask_tr)

        l2_list.append(l2)
        if l2 / l2_old >= 0.99:
            break

    # Compute accuracy
    print(np.sum(yt.argmax(axis=1) == y.argmax(axis=1).A1) / yt.shape[0])
