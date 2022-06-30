import numpy as np
import os
import json
from glob import glob


def beirut():
    models = [m.split('/')[-1].split('_')[0] for m in glob("experiments/Beirut/misclassification_tests/*_seed_0")]
    for model in models:
        acc_list = []
        for seed in np.arange(10):
            with open(os.path.join("experiments/Beirut/misclassification_tests/%s_seed_%s/train.log" % (model, seed)), 'r') as f:
                log = f.read()
                i = -1
                while True:
                    line = log.split('\n')[i]
                    if "accuracy" in line:
                        acc_list.append(float(line.split("accuracy: ")[-1]))
                        break
                    else:
                        i -= 1
                i = -1
                while True:
                    line = log.split('\n')[i]
                    if "AUPR" in line:
                        line = line.split("AUPR: ")[-1]
                        unc_type = line.split(" ")[::3]
                        if seed == 0:
                            unc_aupr_list = {unc: [] for unc in unc_type}
                        for unc, n in zip(unc_type, line.split(" ")[2::3]):
                            unc_aupr_list[unc].append(float(n))
                        break
                    else:
                        i -= 1
                i = -1
                while True:
                    line = log.split('\n')[i]
                    if "AUROC" in line:
                        line = line.split("AUROC: ")[-1]
                        unc_type = line.split(" ")[::3]
                        if seed == 0:
                            unc_auroc_list = {unc: [] for unc in unc_type}
                        for unc, n in zip(unc_type, line.split(" ")[2::3]):
                            unc_auroc_list[unc].append(float(n))
                        break
                    else:
                        i -= 1
        print("-----------------------------------------------------")
        print("Model: %s" % model)
        print()
        print("Mean test accuracy: %s, Variance test accuracy: %s" % (np.mean(acc_list), np.var(acc_list)))
        print()
        print("Misclassification AUROC:")
        for unc in unc_auroc_list:
            print()
            print(unc)
            print("Mean AUROC: %s, Variance AUROC: %s" % (float("{0:.3f}".format(np.mean(unc_auroc_list[unc]))), float("{0:.3f}".format(np.var(unc_auroc_list[unc]) ** .5))))
        print()
        print("Misclassification AUPR:")
        for unc in unc_aupr_list:
            print()
            print(unc)
            print("Mean AUPR: %s, Variance AUPR: %s" % (float("{0:.3f}".format(np.mean(unc_aupr_list[unc]))), float("{0:.3f}".format(np.var(unc_aupr_list[unc]) ** .5))))
        print()


if __name__ == "__main__":
    beirut()
