import numpy as np
from tqdm import tqdm
from utils import mask_to_weights
from sklearn.metrics import roc_auc_score, average_precision_score


def dissonance_uncertainty(alpha):
    strength = np.sum(alpha, axis=1, keepdims=True)
    belief = (alpha - 1) / strength
    dis_un = np.zeros_like(strength, dtype="float64")

    for i in tqdm(range(belief.shape[0])):
        b = belief[i]
        b[b == 0] += 1e-15  # to prevent divide by any divide by zero errors
        bal = 1 - np.abs(b[:, None] - b[None, :]) / (b[None, :] + b[:, None]) - np.eye(len(b))
        coefficients = b[:, None] * b[None, :] - np.diag(b ** 2)
        denominator = np.sum(b[None, :] * np.ones(belief.shape[1]) - np.diag(b), axis=-1, keepdims=True)
        dis_un[i] = (coefficients * bal / denominator).sum()

    return dis_un


def vacuity_uncertainty(alpha):
    return alpha.shape[-1] / alpha.sum(axis=-1)


def misclassification(alpha, vacuity, dissonance, dataset, n_splits=10):
    mask_te = dataset.mask_te
    # Consider nodes in the Unclassified class (id 0) as neither in nor out of distribution i.e. just remove them
    mask_te[np.argwhere((dataset[0].y.argmax(axis=-1) == 0) & mask_te).flatten()] = False
    mask_te = mask_to_weights(dataset.mask_te)
    uncertainties = {"dissonance": {"values": dissonance, "auroc": np.zeros(n_splits), "aupr": np.zeros(n_splits)},
                     "vacuity": {"values": vacuity, "auroc": np.zeros(n_splits), "aupr": np.zeros(n_splits)}}

    for split_i in range(n_splits):
        node_indices = np.random.choice(np.argwhere(mask_te).flatten(), 1000, replace=False)
        pred_matches_label_bool = np.equal(alpha[node_indices].argmax(axis=-1),
                                           dataset[0].y[node_indices].argmax(axis=-1))
        for unc in uncertainties:
            uncertainties[unc]["auroc"][split_i] = roc_auc_score(~pred_matches_label_bool,
                                                                 uncertainties[unc]["values"][node_indices])
            uncertainties[unc]["aupr"][split_i] = average_precision_score(~pred_matches_label_bool,
                                                                          uncertainties[unc]["values"][node_indices])

    return {uncertainty: {dict_key: unc_dict[dict_key].mean() for dict_key in unc_dict if dict_key != "values"} for
            uncertainty, unc_dict in uncertainties.items()}


def ood_detection(vacuity, dissonance, dataset, n_splits=10):
    mask_te = dataset.mask_te
    # Consider nodes in the Unclassified class (id 0) as neither in nor out of distribution i.e. just remove them
    mask_te[np.argwhere((dataset[0].y.argmax(axis=-1) == 0) & mask_te).flatten()] = False
    mask_te = mask_to_weights(dataset.mask_te)
    ood_classes = np.argwhere(dataset[0].y[dataset.mask_tr].sum(axis=0) == 0).flatten()[1:]
    uncertainties = {"dissonance": {"values": dissonance, "auroc": np.zeros(n_splits), "aupr": np.zeros(n_splits)},
                     "vacuity": {"values": vacuity, "auroc": np.zeros(n_splits), "aupr": np.zeros(n_splits)}}

    for split_i in range(n_splits):
        node_indices = np.random.choice(np.argwhere(mask_te).flatten(), 1000, replace=False)
        is_ood_bool = np.isin(dataset[0].y[node_indices].argmax(axis=-1), ood_classes)
        for unc in uncertainties:
            uncertainties[unc]["auroc"][split_i] = roc_auc_score(is_ood_bool,
                                                                 uncertainties[unc]["values"][node_indices])
            uncertainties[unc]["aupr"][split_i] = average_precision_score(is_ood_bool,
                                                                          uncertainties[unc]["values"][node_indices])

    return {uncertainty: {dict_key: unc_dict[dict_key].mean() for dict_key in unc_dict if dict_key != "values"} for
            uncertainty, unc_dict in uncertainties.items()}
