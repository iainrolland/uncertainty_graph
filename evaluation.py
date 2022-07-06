from spektral.data import SingleLoader
import uncertainty_utils as uu
import numpy as np
import os
import logging
from utils import log_error
from tqdm import tqdm
from spektral.models import GCN
from spektral.data import SingleLoader


def evaluate(network, dataset, params, test_misc_detection=True, test_ood_detection=True):
    supported_models = ["Drop-GCN", "GCN", "S-GCN", "S-BGCN", "S-BGCN-T", "S-BGCN-T-K", "S-BMLP"]

    # # Evaluate model
    # print("Evaluating model.")
    # loader_te = SingleLoader(datasets, sample_weights=weights_te)
    # eval_results = network.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    # print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))

    loader_all = SingleLoader(dataset, epochs=1)
    inputs, outputs = loader_all.__next__()

    if params.model == "GCN":
        prob = np.array(network(inputs, training=False))
        total_entropy, class_entropy = uu.entropy(prob)
        uncertainties = {"entropy": total_entropy}
    elif params.model == "Drop-GCN":
        drop_unc = uu.DropoutUncertainties(100)
        for _ in range(100):
            drop_unc.update(prob=np.array(network(inputs, training=True)))
        uncertainties = drop_unc.get_uncertainties()
        prob = drop_unc.mean_prob
    elif params.model == "S-GCN":
        alpha = np.array(network(inputs, training=False))
        uncertainties = uu.get_subjective_uncertainties(alpha)
        prob = uu.alpha_to_prob(alpha)
    elif params.model in ["S-BGCN", "S-BGCN-T", "S-BGCN-K", "S-BGCN-T-K", "S-BMLP"]:
        sb_unc = uu.SubjectiveBayesianUncertainties(100)
        for _ in tqdm(range(100)):
            if params.model == "S-BMLP":
                alpha = np.array(network(inputs[0], training=True))  # i.e. don't pass in the adjacency matrix to MLP
            else:
                alpha = np.array(network(inputs, training=True))
            sb_unc.update(alpha=alpha)
        uncertainties = sb_unc.get_uncertainties()
        prob = sb_unc.mean_prob
    else:
        raise log_error(ValueError,
                        "model was {} but must be one of {}.".format(params.model, "/".join(supported_models)))

    if params.model == "S-BMLP":
        # don't pass in the adjacency matrix to MLP
        np.save(os.path.join(params.directory, "alpha.npy"), np.array(network(inputs[0])))
    else:
        np.save(os.path.join(params.directory, "alpha.npy"), np.array(network(inputs)))
    np.save(os.path.join(params.directory, "prob.npy"), prob)

    if test_misc_detection:
        misc_results = uu.misclassification(prob, uncertainties, dataset[0].y, dataset.mask_te)

        auroc = [(unc, misc_results[unc]["auroc"]) for unc in misc_results]
        aupr = [(unc, misc_results[unc]["aupr"]) for unc in misc_results]

        logging.info("Misclassification AUROC: " +
                     ' '.join([unc_name + " = " + str(score) for unc_name, score in auroc]))
        logging.info("Misclassification AUPR: " + ' '.join([unc_name + " = " + str(score) for unc_name, score in aupr]))

    if test_ood_detection:
        ood_results = uu.ood_detection(uncertainties, dataset[0].y, dataset.mask_tr, dataset.mask_te)

        auroc = [(unc, ood_results[unc]["auroc"]) for unc in ood_results]
        aupr = [(unc, ood_results[unc]["aupr"]) for unc in ood_results]

        logging.info("OOD Detection AUROC: " + ' '.join([unc_name + " = " + str(score) for unc_name, score in auroc]))
        logging.info("OOD Detection AUPR: " + ' '.join([unc_name + " = " + str(score) for unc_name, score in aupr]))

    test_acc = (prob.argmax(axis=1) == dataset[0].y.argmax(axis=1))[dataset.mask_te].mean()
    logging.info("Test set accuracy: {}".format(test_acc))
