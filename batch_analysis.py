import numpy as np
import os
import re
import argparse
from glob import glob


def batch_analyse(arguments):
    if arguments.type not in ["models", "hyperparams"]:
        raise ValueError("--type must be either 'models' or 'hyperparams' but was given as %s" % type)
    elif arguments.type == "models":
        models_analyse(arguments.parent_dir)
    else:
        hyperparams_analyse(arguments.parent_dir)


def hyperparams_analyse(parent_dir):
    permutations = set(["seed_{}".join(re.split(r"seed_\d", f)) for f in glob(os.path.join(parent_dir, "*/"))])

    validation_acc_dict = {}
    best_permutation = None
    best_val_acc_mean = -1
    for perm in permutations:
        accuracies = []
        for f in glob(perm.format('*')):
            if os.path.isfile(os.path.join(f, "val_loss_history.npy")):
                val_loss = np.load(os.path.join(f, "val_loss_history.npy"))
                val_acc = np.load(os.path.join(f, "val_acc_history.npy"))
                val_acc_min = val_acc[np.argmin(val_loss)]
                accuracies.append(val_acc_min)
        if len(accuracies) > 0:
            validation_acc_dict[perm] = {"mean": np.mean(accuracies), "std": np.std(accuracies)}
            if validation_acc_dict[perm]["mean"] > best_val_acc_mean:
                best_permutation = perm
                best_val_acc_mean = validation_acc_dict[perm]["mean"]
        else:
            validation_acc_dict[perm] = {"mean": np.inf, "std": 0}

    for perm, results in validation_acc_dict.items():
        print("-----------------------------------------------------")
        print("Hyperparameter permutation: %s" % (perm.split('/')[-2]))
        print()
        print("Mean validation acc: %s, Std. validation acc: %s" % (results["mean"], results["std"]))
        print()

    if best_permutation is not None:
        print("-----------------------------------------------------")
        print("Best hyperparameter permutation: %s" % (best_permutation.split('/')[-2]))
        print()
        print("Mean validation acc: %s, Std. validation acc: %s" % (validation_acc_dict[best_permutation]["mean"],
                                                                    validation_acc_dict[best_permutation]["std"]))
        print()


def models_analyse(parent_dir):
    models = np.unique([f.split('/')[-2].split('_')[0] for f in glob(os.path.join(parent_dir, "*/"))])
    for model in models:
        acc_list = []
        for i, exp_dir in enumerate(glob(os.path.join(parent_dir, "%s*" % model))):
            with open(os.path.join(exp_dir, "train.log"), 'r') as f:
                log = f.read()
                j = -1
                while True:
                    try:
                        line = log.split('\n')[j]
                    except IndexError:
                        print(exp_dir)
                        raise IndexError
                    if "accuracy" in line:
                        acc_list.append(float(line.split("accuracy: ")[-1]))
                        break
                    else:
                        j -= 1
                j = -1
                while True:
                    line = log.split('\n')[j]
                    if "AUPR" in line:
                        line = line.split("AUPR: ")[-1]
                        unc_type = line.split(" ")[::3]
                        if i == 0:
                            unc_aupr_list = {unc: [] for unc in unc_type}
                        for unc, n in zip(unc_type, line.split(" ")[2::3]):
                            unc_aupr_list[unc].append(float(n))
                        break
                    else:
                        j -= 1
                j = -1
                while True:
                    line = log.split('\n')[j]
                    if "AUROC" in line:
                        line = line.split("AUROC: ")[-1]
                        unc_type = line.split(" ")[::3]
                        if i == 0:
                            unc_auroc_list = {unc: [] for unc in unc_type}
                        for unc, n in zip(unc_type, line.split(" ")[2::3]):
                            unc_auroc_list[unc].append(float(n))
                        break
                    else:
                        j -= 1
        print("-----------------------------------------------------")
        print("Model: %s" % model)
        print()
        print("Mean test accuracy: %s, Variance test accuracy: %s" % (np.mean(acc_list), np.var(acc_list)))
        print()
        print("Misclassification AUROC:")
        for unc in unc_auroc_list:
            print()
            print(unc)
            print("Mean AUROC: %s, Variance AUROC: %s" % (float("{0:.3f}".format(np.mean(unc_auroc_list[unc]))),
                                                          float("{0:.3f}".format(np.var(unc_auroc_list[unc]) ** .5))))
        print()
        print("Misclassification AUPR:")
        for unc in unc_aupr_list:
            print()
            print(unc)
            print("Mean AUPR: %s, Variance AUPR: %s" % (float("{0:.3f}".format(np.mean(unc_aupr_list[unc]))),
                                                        float("{0:.3f}".format(np.var(unc_aupr_list[unc]) ** .5))))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_dir',
                        help="Directory containing the experiment folders (each with a params.json in it)")
    parser.add_argument('--type', help="Type of analysis to perform: 'models' or 'hyperparams'")
    args = parser.parse_args()
    batch_analyse(args)
