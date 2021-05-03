import os

from .utils import *
from rs import rgb
from .utils import unflatten_array, read_colormap

set_rcParams()


def load_alpha_vac_dis(directory):
    paths = [os.path.join(directory, file_name) for file_name in ["alpha.npy", "vacuity.npy", "dissonance.npy"]]
    return [np.load(path) for path in paths]


def plot(model_dir="experiments/S_BGCN_HoustonDataset_k2_2021_04_26_2"):
    # alpha, vac, dis = [unflatten_array(array) for array in load_alpha_vac_dis(model_dir)]
    alpha = unflatten_array(np.load(os.path.join(model_dir, "prob_pred.npy")))
    colormap = read_colormap()

    fig, ax = close_fig(3, 1, width=7, height=4, keep_square=False)
    gt = load_gt()
    classification = np.argmax(alpha, axis=-1)
    classification_rgb = np.zeros(classification.shape + (4,), dtype=np.uint8)
    gt_rgb = np.zeros(gt.shape + (4,), dtype=np.uint8)
    for class_id in range(np.max(classification)):
        classification_rgb[classification == class_id] = colormap[class_id]
        gt_rgb[gt == class_id] = colormap[class_id]

    ax[0].imshow(np.rollaxis(rgb.training_array(), 0, 3))
    legend = shade_one_axes_with_splits(ax[0])
    ax[0].legend(handles=legend, loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].imshow(gt_rgb)
    ax[2].imshow(classification_rgb)
    # img = ax[3].matshow(vac)
    # ax[3].set_xticks([])
    # ax[3].set_yticks([])
    # add_colorbar(fig, img, ax[3])
    # img = ax[4].matshow(dis)
    # ax[4].set_xticks([])
    # ax[4].set_yticks([])
    # add_colorbar(fig, img, ax[4])
    set_each_ax_y_label(["Optical", "GT", "GCN\nClassification", "Vacuity", "Dissonance"], ax)

    fig.savefig("docs/26_april/gcn_results.pdf", bbox_inches="tight")
