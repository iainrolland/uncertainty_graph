# from spektral.datasets import Citation

from training import train
from models import GCN
from datasets import Karate, HoustonDataset_k2
from utils import gpu_initialise

gpu_list = []
gpu_initialise(gpu_list)
# data = Citation(name="cora", transforms=GCN.transforms)
data = HoustonDataset_k2(transforms=GCN.transforms)
learning_rate = 1e-2
l2_loss_coefficient = 5e-4
epochs = 400
patience = 10
seed = 0

train(GCN, data, learning_rate, l2_loss_coefficient, epochs, patience, seed)
