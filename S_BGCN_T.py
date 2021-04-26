import os
from datetime import date
# from spektral.datasets import Citation

from training import train
from models import S_BGCN_T
from datasets import Karate, HoustonDataset_k2
from utils import gpu_initialise

gpu_list = [0]
gpu_initialise(gpu_list)
# data = Citation(name="cora", transforms=S_BGCN_T.transforms)
data = HoustonDataset_k2(transforms=S_BGCN_T.transforms)
teacher_prob_output_path = os.path.join("models",
                                        "GCN_" + data.__class__.__name__ + "_2021_04_24_0",
                                        "prob_pred.npy")
learning_rate = 1e-2
l2_loss_coefficient = 5e-4 * 1
epochs = 200
patience = 20
seed = 0

train(S_BGCN_T(teacher_prob_output_path), data, learning_rate, l2_loss_coefficient, epochs, patience, seed)
