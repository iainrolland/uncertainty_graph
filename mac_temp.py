from datasets import HoustonDatasetMini
import copy
import warnings
import numpy as np
from scipy import sparse as sp

from utils import set_seeds


def degree_power(A, k):
    r"""
    Computes \(\D^{k}\) from the given adjacency matrix. Useful for computing
    normalised Laplacian.
    :param A: rank 2 array or sparse matrix.
    :param k: exponent to which elevate the degree matrix.
    :return: if A is a dense array, a dense array; if A is sparse, a sparse
    matrix in DIA format.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        degrees = np.power(np.array(A.sum(1)), k).ravel()
    degrees[np.isinf(degrees)] = 0.0
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


def normalized_adjacency_by_degree(A):
    normalized_D = degree_power(A, -1)
    return A.dot(normalized_D)


def diffusion_filter(A):
    out = copy.deepcopy(A)
    if isinstance(A, list) or (isinstance(A, np.ndarray) and A.ndim == 3):
        for i in range(len(A)):
            out[i] = A[i]
            out[i] = normalized_adjacency_by_degree(out[i])
    else:
        if hasattr(out, "tocsr"):
            out = out.tocsr()
        out = normalized_adjacency_by_degree(out)

    if sp.issparse(out):
        out.sort_indices()
    return out


def update(state, truth, propagation_matrix, mask, step_size=.2):
    flow_in = propagation_matrix.dot(state)
    flow_out = state
    update = step_size * (flow_in - flow_out)
    update_l2 = np.mean(np.power(update, 2))
    state += update
    # new_state = ((1 - step_size) * sparse.identity(adjacency.shape[0]) + step_size * adjacency.dot(
    #     np.diag(adjacency.sum(axis=0).A1 ** -1))).dot(state)
    state[mask] = truth[mask]  # boundary conditions
    return state, update_l2


def gauss(x, sigma=1):
    return np.exp(-x ** 2 / 2 / sigma ** 2) * (2 * np.pi * sigma ** 2) ** -0.5


def alpha_prior(adjacency, y_true, training_mask, num_steps=10):
    prior = np.ones(y_true.shape)
    masked_y = 0 * y_true
    masked_y[training_mask] = y_true[training_mask]
    state = masked_y
    for n_step in range(num_steps):
        new_state = adjacency.dot(state)
        step = new_state * gauss(n_step, sigma=1)
        prior += step
        state = new_state
    return prior


if __name__ == "__main__":
    # Load dataset
    set_seeds(0)
    dataset = HoustonDatasetMini()
    adj = dataset[0].a
    yt = dataset[0].y
    mask_tr = dataset.mask_tr

    # # Set initial state
    # y = np.ones_like(yt) / yt.shape[1]
    # y[mask_tr] = yt[mask_tr]
    #
    # flow_matrix = diffusion_filter(adj)
    #
    # # Update until convergence
    # y, l2 = update(y, yt, flow_matrix, mask_tr)
    # l2_list = [l2]
    # for t in range(500):
    #     l2_old = l2
    #     y, l2 = update(y, yt, flow_matrix, mask_tr)
    #     l2_list.append(l2)
    #     print(t, l2 / l2_old)
    #     if l2 / l2_old >= 0.99:
    #         break
    # predictions = y.argmax(axis=1)

    alpha = alpha_prior(adj, yt, dataset.mask_tr)
    predictions = alpha.argmax(axis=1)

    # Compute accuracy
    print((predictions == yt.argmax(axis=1))[dataset.mask_te].mean())
