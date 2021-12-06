from plotting.figures import *
import plotting.utils as pu
from utils import set_seeds
import matplotlib.pyplot as plt
import os
from datasets import HoustonDatasetMini
from models import S_BGCN_T_K
from spektral.models import GCN
from spektral.data import SingleLoader
from params import Params
from uncertainty_utils import vacuity_uncertainty, dissonance_uncertainty
from utils import gpu_initialise
import numpy as np
import os
from glob import glob
from params import Params
from models import get_model


# seed = 0
# set_seeds(seed)
# gpu_initialise([])
# hd = HoustonDatasetMini(transforms=S_BGCN_T_K.transforms)
# model_dir = "experiments/test_SBMLP/"
# parameters = Params(os.path.join(model_dir, "params.json"))

def get_network(m_dir, data, params):
    model = get_model(params)
    model.get_network(params, data.n_node_features, data.n_labels)
    network = model.network
    network(np.ones((1, data.n_node_features)) / 1.)  # dummy predict in order to build correct dims
    network.load_weights(os.path.join(m_dir, model.__name__ + ".h5"))

    return network


# net = get_network(model_dir, hd, parameters)
# # inputs = (hd[0].x, hd[0].a)
# outputs = net.predict(hd[0].x)
# np.save(os.path.join(model_dir, "alpha.npy"), outputs.astype("float32"))
# alpha = np.load(os.path.join(model_dir, "alpha.npy"))
# vacuity, dissonance = vacuity_uncertainty(alpha), dissonance_uncertainty(alpha)
# np.save(os.path.join(model_dir, "vacuity.npy"), vacuity.astype("float32"))
# np.save(os.path.join(model_dir, "dissonance.npy"), dissonance.astype("float32"))
# plot(model_dir)

# draw_dirichlet([[1.] * 3, [5.] * 3, [50] * 3])
beirut_fig()

# spixel_path = "experiments/compare_w_vs_wo_spixel/spixel_{}_seed_{}.npy".format("{}", seed)
# pixel_path = "experiments/compare_w_vs_wo_spixel/alpha_{}_seed_{}.npy".format("{}", seed)
#
#
# # # spixel_path = "experiments/all_paths_vary_scale/seed_{}_scale_3_{}.npy".format(seed, "{}")
# scales = sorted([a.split('/')[-1].split("scale_")[-1].split('_')[0] for a in
#                  glob("experiments/spixel_prior_vary_scale/*_alpha.npy")])
# # scales = sorted([a.split('/')[-1].split("scale_")[-1].split('_')[0] for a in
# #                  glob("experiments/all_paths_vary_scale_vertically/*_prior.npy")])
# # scales = [scale for _, scale in sorted(zip([float(s) for s in scales], scales))]
# # # paths = ["experiments/all_paths_vary_scale/seed_{}_scale_{}_{}.npy".format(seed, scale, "{}") for scale in scales]
# paths = ["experiments/spixel_prior_vary_scale/seed_{}_scale_{}_{}.npy".format(seed, scale, "{}") for scale in scales]
#
#
# # spixel = np.load(spixel_path.format("prior"))
# # pixel = np.load(pixel_path.format("prior"))
#
#
# def make_vac_dis(alpha, alpha_path):
#     if not os.path.isfile(alpha_path.format("vac")):
#         np.save(alpha_path.format("vac"), vacuity_uncertainty(alpha))
#     if not os.path.isfile(alpha_path.format("dis")):
#         np.save(alpha_path.format("dis"), dissonance_uncertainty(alpha))
#
#
# for p in paths:
#     make_vac_dis(np.load(p.format("alpha")), p)
#
# fig, ax = pu.dense_fig(3, len(paths), keep_square=False, height=3)
#
#
# def plot_col(figure, axes, path, i):
#     alpha, vac, dis = [np.load(path.format(a)) for a in ["alpha", "vac", "dis"]]
#     axes[0, i].imshow(pu.classes_array_to_colormapped_array(alpha.argmax(axis=-1).reshape(-1, 1800)))
#     img = axes[1, i].matshow(vac.reshape(-1, 1800), vmin=0, vmax=1)
#     if i == axes.shape[1] - 1:
#         pu.add_colorbar(figure, img, ax[1, i], x_shift=.05, height_scale=.9)
#     img = axes[2, i].matshow(dis.reshape(-1, 1800), vmin=0, vmax=1)
#     if i == axes.shape[1] - 1:
#         pu.add_colorbar(figure, img, ax[2, i], x_shift=.05, height_scale=.9)
#     pu.remove_ax_list_ticks([a for a in axes[:, i]])
#     if i == 0:
#         pu.set_each_ax_y_label([r"$\hat\mathbf{\alpha}$", "Vac.", "Dis."],
#                                [a for a in axes[:, i]])
#
#
# # plot_col(fig, ax, spixel_path, 0)
# # plot_col(fig, ax, pixel_path, 1)
# # ax[0, 1].set_title("Pixel prior\n" + r"(All paths $\hat{\mathbf{\alpha}}$)")
# # ax[0, 0].set_title("Superpixel prior\n" + r"(Shortest path $\hat{\mathbf{\alpha}}$)")
# for i, p in enumerate(paths):
#     ax[0, i].set_title(r"$\sigma=" + p.split("/")[-1].split("scale_")[1].split("_")[0] + '$')
#     plot_col(fig, ax, p, i)
#
# fig.savefig("docs/18_may/spixel_prior_vary_scale.jpg", dpi=300, bbox_inches="tight")
