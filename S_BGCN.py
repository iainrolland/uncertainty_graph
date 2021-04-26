from spektral.datasets import Citation

from training import train
from models import S_BGCN
from datasets import Karate, HoustonDataset_k2
from utils import gpu_initialise

gpu_list = []
gpu_initialise(gpu_list)
# data = Citation(name="cora", transforms=S_BGCN.transforms)
data = HoustonDataset_k2(transforms=S_BGCN.transforms)
learning_rate = 2e-2
l2_loss_coefficient = 5e-4
epochs = 200
patience = 10
seed = 0

train(S_BGCN, data, learning_rate, l2_loss_coefficient, epochs, patience, seed)
