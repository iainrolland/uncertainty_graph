from plotting.figures import plot
from utils import set_seeds
from datasets import HoustonDatasetMini
from models import S_BGCN
from spektral.models import GCN
from spektral.data import SingleLoader
from params import Params
from uncertainty_utils import vacuity_uncertainty, dissonance_uncertainty
import numpy as np
import os

set_seeds(0)
model_dir = "experiments/GCN_HoustonDatasetMini_03_05_21_0"
# params = Params(os.path.join(model_dir, "params.json"))
# hd = HoustonDatasetMini(transforms=S_BGCN.transforms)
# network = GCN(n_labels=hd.n_labels, channels=params.channels, n_input_channels=hd.n_node_features,
#               output_activation=S_BGCN.output_activation, l2_reg=params.l2_loss_coefficient)
# network((np.ones((1, 54)), np.ones((1, 1))))
# network.load_weights(os.path.join(model_dir, "S_BGCN.h5"))
# inputs = (hd[0].x, hd[0].a)
# outputs = network(inputs)
# np.save(os.path.join(model_dir, "alpha.npy"), outputs)
# alpha = np.load(os.path.join(model_dir, "alpha.npy"))
# vacuity, dissonance = vacuity_uncertainty(alpha), dissonance_uncertainty(alpha)
# np.save(os.path.join(model_dir, "vacuity.npy"), vacuity)
# np.save(os.path.join(model_dir, "dissonance.npy"), dissonance)
plot(model_dir)

