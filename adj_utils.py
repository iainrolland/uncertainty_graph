import uncertainty_utils
from datasets import get_dataset
import copy
import warnings
import numpy as np
from scipy import sparse as sp
from utils import set_logger
import logging
from time import time
import itertools
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import lil_matrix, csr_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import set_seeds
from plotting.utils import classes_array_to_colormapped_array


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


def alpha_prior(adjacency, y_true, training_mask, num_steps=50):
    print("Propagating alpha prior...")
    prior = np.ones(y_true.shape)
    masked_y = 0 * y_true
    masked_y[training_mask] = y_true[training_mask]
    state = masked_y
    prior_update = state * gauss(0, sigma=1)
    prior += prior_update
    for n_step in range(num_steps):
        new_state = adjacency.dot(state)
        prior_update = new_state * gauss(n_step + 1, sigma=1)
        update_l2 = np.mean(np.power(prior_update[prior_update != 0], 2))
        prior += prior_update
        state = new_state
        if update_l2 <= 1e-32:
            return prior
    else:
        raise ArithmeticError("Propagation of alpha prior not converged")


def alpha_prior_from_spixels(spixel_adj, segments, pixel_y_true, training_mask):
    masked_y = 0 * pixel_y_true
    masked_y[training_mask] = pixel_y_true[training_mask]

    sp_matrix = shortest_path(spixel_adj)
    flat_seg = segments.flatten()
    h = gauss(sp_matrix, sigma=1)

    masked_y_seg = np.array(
        [list(masked_y[np.argwhere(flat_seg == i)].sum(axis=0).flatten()) for i in
         range(segments.min(), segments.max() + 1)])

    prior = h.dot(masked_y_seg) + 1
    return prior  # shape: (#spixels, #classes)


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


# if __name__ == "__main__":
#     ood_classes = [9, 13]
#
#     # Load dataset
#     set_seeds(1)
#     dataset = HoustonDatasetMini()
#     dataset.mask_tr[np.argwhere(np.isin(dataset[0].y.argmax(axis=-1), ood_classes)).flatten()] = False
#     adj = dataset[0].a
#     yt = dataset[0].y
#     mask_tr = dataset.mask_tr
#
#     # # Set initial state
#     # y = np.ones_like(yt) / yt.shape[1]
#     # y[mask_tr] = yt[mask_tr]
#     #
#     # flow_matrix = diffusion_filter(adj)
#     #
#     # # Update until convergence
#     # y, l2 = update(y, yt, flow_matrix, mask_tr)
#     # l2_list = [l2]
#     # for t in range(500):
#     #     l2_old = l2
#     #     y, l2 = update(y, yt, flow_matrix, mask_tr)
#     #     l2_list.append(l2)
#     #     print(t, l2 / l2_old)
#     #     if l2 / l2_old >= 0.99:
#     #         break
#     # predictions = y.argmax(axis=1)
#
#     # alpha = alpha_prior(adj, yt, dataset.mask_tr)
#     # predictions = alpha.argmax(axis=1)
#     #
#     # # Compute accuracy
#     # print((predictions == yt.argmax(axis=1))[dataset.mask_te].mean())
#     #
#     # np.save("experiments/OOD_Detection_seed_1/alpha_prior.npy", alpha)
#
#     set_logger("experiments/Sampled_OOD_classes/ood_classes_{}{}_alpha_prior/train.log".format(*ood_classes))
#     alpha = np.load("experiments/Sampled_OOD_classes/alpha_{}_{}_prior.npy".format(*ood_classes))
#     unc = uncertainty_utils.get_subjective_uncertainties(alpha)
#     prob = uncertainty_utils.alpha_to_prob(alpha)
#
#     misc_results = uncertainty_utils.misclassification(prob, unc, dataset[0].y, dataset.mask_te)
#     ood_results = uncertainty_utils.ood_detection(unc, dataset[0].y, dataset.mask_tr, dataset.mask_te)
#
#     auroc = [(unc, misc_results[unc]["auroc"]) for unc in misc_results]
#     aupr = [(unc, misc_results[unc]["aupr"]) for unc in misc_results]
#
#     logging.info("Misclassification AUROC: " +
#                  ' '.join([unc_name + " = " + str(score) for unc_name, score in auroc]))
#     logging.info("Misclassification AUPR: " + ' '.join([unc_name + " = " + str(score) for unc_name, score in aupr]))
#     auroc = [(unc, misc_results[unc]["auroc"]) for unc in misc_results]
#     aupr = [(unc, misc_results[unc]["aupr"]) for unc in misc_results]
#
#     logging.info("OOD Detection AUROC: " +
#                  ' '.join([unc_name + " = " + str(score) for unc_name, score in auroc]))
#     logging.info("OOD Detection AUPR: " + ' '.join([unc_name + " = " + str(score) for unc_name, score in aupr]))
#     logging.info("Test set accuracy: {}".format((prob.argmax(axis=1) == yt.argmax(axis=1))[dataset.mask_te].mean()))

data = get_dataset("HoustonSpixelMini")()

p = alpha_prior_from_spixels(data.spixel_adj, data.segments, data[0].y, data.mask_tr)
dis, vac = uncertainty_utils.dissonance_uncertainty(p), uncertainty_utils.vacuity_uncertainty(p)
p, vac, dis = [pixels_from_spixels(array, data.segments) for array in [p, vac, dis]]
np.save("experiments/spixel_test/spixel_prior.npy", p)
