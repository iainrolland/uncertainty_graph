from training import train
from models import GCN

data = "karate"
learning_rate = 1e-2
l2_loss_coefficient = 5e-4
epochs = 200
patience = 10
seed = 0
gpu_list = []

train(GCN, data, learning_rate, l2_loss_coefficient, epochs, patience, seed, gpu_list)
