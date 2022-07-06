import uncertainty_utils as uu
import utils
from datasets import get_dataset
import copy
import warnings
import numpy as np
from scipy import sparse as sp
from utils import set_logger
import logging
from glob import glob
from time import time
import os
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import set_seeds
from plotting.utils import classes_array_to_colormapped_array
from params import Params


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


def alpha_prior(adjacency, y_true, training_mask, num_steps=50, w_scale=1):
    print("Propagating alpha prior...")
    prior = np.ones(y_true.shape, dtype="float32")
    masked_y = 0 * y_true
    masked_y[training_mask] = y_true[training_mask]
    state = masked_y
    prior_update = state * gauss(0, sigma=1)
    prior += prior_update
    for n_step in range(num_steps):
        new_state = adjacency.dot(state)
        prior_update = new_state * gauss(n_step + 1, sigma=w_scale)
        update_l2 = np.mean(np.power(prior_update[prior_update != 0], 2))
        prior += prior_update
        state = new_state
        print(update_l2)
        if update_l2 <= 1e-32:
            return prior
    else:
        raise ArithmeticError("Propagation of alpha prior not converged")


def alpha_prior_from_spixels(spixel_adj, segments, pixel_y_true, training_mask, scale=1):
    masked_y = 0 * pixel_y_true
    masked_y[training_mask] = pixel_y_true[training_mask]

    sp_matrix = shortest_path(spixel_adj)
    flat_seg = segments.flatten()
    h = gauss(sp_matrix, sigma=scale)

    masked_y_seg = np.array(
        [list(masked_y[np.argwhere(flat_seg == i)].sum(axis=0).flatten()) for i in
         range(segments.min(), segments.max() + 1)])

    prior = h.dot(masked_y_seg) + 1
    return prior.astype("float32")  # shape: (#spixels, #classes)


def pixels_from_spixels(array, segments):
    return array[segments.flatten()]


def rows_cols(segments, spixel_adj):
    connections = []
    one = time()
    rows, cols = [], []
    for spixel_idx in tqdm(range(10)):
        pix = np.argwhere((segments == spixel_idx).flatten()).flatten()
        seg_neighb = np.argwhere(spixel_adj[spixel_idx] != 0)[:, 1]
        seg_neighb_pix = np.argwhere(np.isin(segments, seg_neighb).flatten()).flatten()
        rows.extend([a for _ in pix for a in seg_neighb_pix])
        cols.extend([a for a in pix for _ in seg_neighb_pix])
    adj_full = csr_matrix((np.ones(len(rows)), (rows, cols)), dtype="uint8")
    print(time() - one)


def lil_mat(segments, spixel_adj):
    lm = lil_matrix((segments.shape[0] * segments.shape[1], segments.shape[0] * segments.shape[1]), dtype="uint8")
    one = time()
    for spixel_idx in tqdm(range(10)):
        pix = np.argwhere((segments == spixel_idx).flatten()).flatten()
        seg_neighb = np.argwhere(spixel_adj[spixel_idx] != 0)[:, 1]
        seg_neighb_pix = np.argwhere(np.isin(segments, seg_neighb).flatten()).flatten()
        lm[pix][:, seg_neighb_pix] = 1
    print(time() - one)


def csr_mat(segments, spixel_adj):
    lm = csr_matrix((segments.shape[0] * segments.shape[1], segments.shape[0] * segments.shape[1]), dtype="uint8")
    for spixel_idx in tqdm(range(segments.min(), segments.max() + 1)):
        pix = np.argwhere((segments == spixel_idx).flatten()).flatten()
        seg_neighb = np.argwhere(spixel_adj[spixel_idx] != 0)[:, 1]
        seg_neighb_pix = np.argwhere(np.isin(segments, seg_neighb).flatten()).flatten()
        lm[pix][:, seg_neighb_pix] = 1
    return lm


def meshgrid(segments, spixel_adj):
    one = time()
    rows, cols = [], []
    for spixel_idx in tqdm(range(10)):
        pix = np.argwhere((segments == spixel_idx).flatten()).flatten()
        seg_neighb = np.argwhere(spixel_adj[spixel_idx] != 0)[:, 1]
        seg_neighb_pix = np.argwhere(np.isin(segments, seg_neighb).flatten()).flatten()
        r, c = np.meshgrid(pix, seg_neighb_pix)
        rows.extend(r.flatten())
        cols.extend(c.flatten())
    adj_full = csr_matrix((np.ones(len(rows)), (rows, cols)), dtype="uint8")
    print(time() - one)


def houston():
    seed = 1
    utils.set_seeds(seed)
    ood_classes = [[4, 2], [16, 13], [1, 10], [8, 12], [2, 13], [16, 11], [5, 4], [9, 13], [13, 1], [7, 12]]

    # data = get_dataset("HoustonDatasetMini")()
    data = get_dataset("HoustonSpixelMini")()
    all_classes_mask_tr = data.mask_tr.copy()
    # data.read()

    app = "experiments/Sampled_OOD_classes/spixel_{}_{}_prior.npy"

    for ood_c in ood_classes:
        ood_classes_mask_tr = data.mask_tr.copy()
        ood_classes_mask_tr[np.argwhere(np.isin(data[0].y.argmax(axis=-1), ood_c)).flatten()] = False
        # prior = alpha_prior(data[0].a, data[0].y, data.mask_tr, w_scale=scale, num_steps=1000)
        prior = pixels_from_spixels(
            alpha_prior_from_spixels(data.spixel_adj, data.segments, data[0].y, ood_classes_mask_tr, scale=1),
            data.segments)
        np.save(app.format(*ood_c), prior)
        print((prior.argmax(axis=1) == data[0].y.argmax(axis=1))[data.mask_te].mean())
    # prior, vac, dis = [np.load("experiments/all_paths_vary_scale_vertically/seed_0_v_scale_1.9_{}.npy".format(t)) for t in
    #                    ["prior", "vac", "dis"]]
    # unc = uu.get_subjective_uncertainties(prior)
    # foo = uu.misclassification(uu.alpha_to_prob(prior), unc, data[0].y, data.mask_te)
    # print(foo)


def make_alpha_priors(parent_dir, exact=True):
    # any experiment with a K in its name uses an alpha prior, get each of these and make a prior for each
    # (seeds, and therefore priors, usually differ)
    folders = [f for f in glob(os.path.join(parent_dir, "*")) if "K" in f]

    for f in tqdm(folders):
        parameters = Params(os.path.join(f, "params.json"))
        utils.set_seeds(parameters.seed)

        # load dataset
        if parameters.data == "AirQuality":
            data = get_dataset(parameters.data)(parameters.region,
                                                parameters.datatype,
                                                parameters.numb_op_classes,
                                                parameters.seed,
                                                parameters.train_ratio,
                                                parameters.val_ratio)
        else:
            data = get_dataset(parameters.data)()

        all_classes_mask_tr = data.mask_tr.copy()
        alpha_prior_path = os.path.join(parent_dir, "alpha_prior_seed_%s.npy" % parameters.seed)

        sigma = 1
        masked_y = 0 * data[0].y
        masked_y[data.mask_tr] = data[0].y[data.mask_tr]
        if exact:
            prior = np.dot(np.exp((-shortest_path(data[0].a) ** 2) / (2 * sigma ** 2)) / sigma / (2 * np.pi) ** 0.5,
                           masked_y) + 1
        else:
            prior = alpha_prior(data[0].a, data[0].y, data.mask_tr)

        np.save(alpha_prior_path, prior)
        print("seed_%s: " % parameters.seed, np.unique(prior.argmax(axis=1), return_counts=True))


if __name__ == '__main__':
    # houston()
    make_alpha_priors("experiments/AirQuality/Italy/PM25/misclassification_tests", False)
