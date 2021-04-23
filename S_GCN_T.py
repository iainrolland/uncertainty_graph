from training import train
from models import S_GCN_T
import os
from datetime import date

data = "karate"
teacher_prob_output_path = os.path.join("models",
                                        "GCN_" + data + "_" + date.isoformat(date.today()).replace("-", "_") + "_1",
                                        "prob_pred.npy")
learning_rate = 1e-2
l2_loss_coefficient = 5e-4
epochs = 200
patience = 10
seed = 0
gpu_list = []

train(S_GCN_T(teacher_prob_output_path), data, learning_rate, l2_loss_coefficient, epochs, patience, seed, gpu_list)
