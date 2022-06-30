import numpy as np
import json
import os

import utils
import datasets


def beirut_ood(base_dir="experiments/Beirut/misclassification_tests"):
    if not os.path.exists(base_dir):  # make dir if it doesn't already exist
        os.makedirs(base_dir)

    models = ["GCN", "Drop-GCN", "S-GCN", "S-BGCN", "S-BGCN-T", "S-BGCN-K", "S-BGCN-T-K", "S-BMLP"]
    seeds = np.arange(10)
    path = "experiments/Sampled_OOD_classes/ood_classes_110_model_%s/"
    base_configs = {model: json.load(open(os.path.join(path % model, "params.json"), 'r')) for model in models if os.path.exists(os.path.join(path % model, "params.json"))}
    for model, config in base_configs.items():
        base_configs[model]["ood_classes"] = []
        base_configs[model]["data"] = "BeirutDataset"
        base_configs[model]["epochs"] = 600
        if "T" not in model:
            if "teacher_file_path" in base_configs[model].keys():
                del base_configs[model]["teacher_file_path"]
            if "teacher_coefficient" in base_configs[model].keys():
                del base_configs[model]["teacher_coefficient"]
        if "K" not in model:
            if "alpha_prior_path" in base_configs[model].keys():
                del base_configs[model]["alpha_prior_path"]
            if "alpha_prior_coefficient" in base_configs[model].keys():
                del base_configs[model]["alpha_prior_coefficient"]

    for seed in seeds:
        for model in models:
            if model in base_configs.keys():
                job_name = "%s_seed_%s" % (model, seed)
                base_configs[model]["seed"] = int(seed)
                if "teacher_file_path" in base_configs[model].keys():
                    base_configs[model]["teacher_file_path"] = os.path.join(base_dir, "GCN_seed_%s" % seed, "prob.npy")
                if "alpha_prior_path" in base_configs[model].keys():
                    base_configs[model]["alpha_prior_path"] = os.path.join(base_dir, "alpha_prior_seed_%s.npy" % seed)

                # make dir if it doesn't already exist
                exp_dir = os.path.join(base_dir, job_name)
                if not os.path.exists(exp_dir):
                    os.makedirs(exp_dir)

                # write params file in dir
                with open(os.path.join(exp_dir, "params.json"), 'w') as f:
                    print(base_configs[model])
                    json.dump(base_configs[model], f)


def airquality_misclassifications(base_dir="experiments/AirQuality/Italy/PM25/misclassification_tests", number_of_classes=4):
    if not os.path.exists(base_dir + "_{}classes".format(number_of_classes)):  # make dir if it doesn't already exist
        os.makedirs(base_dir)

    models = ["GCN", "Drop-GCN", "S-GCN", "S-BGCN", "S-BGCN-T", "S-BGCN-K", "S-BGCN-T-K", "S-BMLP"]
    seeds = np.arange(10)
    path = "experiments/Sampled_OOD_classes/ood_classes_110_model_%s/"
    base_configs = {model: json.load(open(os.path.join(path % model, "params.json"), 'r')) for model in models if os.path.exists(os.path.join(path % model, "params.json"))}

    # configure model-specific base_configs
    for model, config in base_configs.items():
        base_configs[model]["ood_classes"] = []  # no ood classes (since doing misclassification tests)
        base_configs[model]["data"] = "AirQuality"
        base_configs[model]["epochs"] = 600
        base_configs[model]["numb_op_classes"] = number_of_classes
        base_configs[model]["region"] = "Italy"
        base_configs[model]["datatype"] = "PM25"
        base_configs[model]["train_ratio"] = 0.5
        base_configs[model]["val_ratio"] = 0.25

        if "T" not in model:
            if "teacher_file_path" in base_configs[model].keys():
                del base_configs[model]["teacher_file_path"]
            if "teacher_coefficient" in base_configs[model].keys():
                del base_configs[model]["teacher_coefficient"]
        if "K" not in model:
            if "alpha_prior_path" in base_configs[model].keys():
                del base_configs[model]["alpha_prior_path"]
            if "alpha_prior_coefficient" in base_configs[model].keys():
                del base_configs[model]["alpha_prior_coefficient"]

    for seed in seeds:
        for model in models:
            if model in base_configs.keys():
                job_name = "%s_seed_%s" % (model, seed)
                base_configs[model]["seed"] = int(seed)
                if "teacher_file_path" in base_configs[model].keys():
                    base_configs[model]["teacher_file_path"] = os.path.join(base_dir, "GCN_seed_%s" % seed, "prob.npy")
                if "alpha_prior_path" in base_configs[model].keys():
                    base_configs[model]["alpha_prior_path"] = os.path.join(base_dir, "alpha_prior_seed_%s.npy" % seed)

                # make dir if it doesn't already exist
                exp_dir = os.path.join(base_dir, job_name)
                if not os.path.exists(exp_dir):
                    os.makedirs(exp_dir)

                # write params file in dir
                if not os.path.isfile(os.path.join(exp_dir, "params.json")):
                    with open(os.path.join(exp_dir, "params.json"), 'w') as f:
                        print(base_configs[model])
                        json.dump(base_configs[model], f)
                else:
                    print("Skipping {}, folder already contains params.json file".format(exp_dir))


if __name__ == "__main__":
    airquality_misclassifications()
