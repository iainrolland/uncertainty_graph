import os

from .utils import *
from rs import rgb
from uncertainty_utils import vacuity_uncertainty, dissonance_uncertainty
from datasets import HoustonDatasetMini

set_rcParams()


def load_alpha_vac_dis(directory):
    return [np.load(os.path.join(directory, file_name)) for file_name in
            ["alpha.npy", "vacuity.npy", "dissonance.npy"]]


def plot(model_dir="experiments/S_BGCN_HoustonDataset_k2_2021_04_26_2"):
    gt = load_gt()
    dataset = HoustonDatasetMini()

    alpha, vac, dis = [unflatten_array(array) for array in load_alpha_vac_dis(model_dir)]
    # alpha = unflatten_array(np.load(os.path.join(model_dir, "prob.npy")), gt.shape)

    # fig, ax = dense_fig(3, 1, width=7, height=4, keep_square=False)
    fig, ax = dense_fig(3, 2, width=8, height=7, keep_square=False)
    ax[0, 1].axis('off')  # invisible
    classification = np.argmax(alpha, axis=-1)

    gt_rgb = classes_array_to_colormapped_array(gt)
    classification_rgb = classes_array_to_colormapped_array(classification)

    ax[0, 0].imshow(np.rollaxis(rgb.training_array(), 0, 3)[:, 1800:3600])
    legend = shade_one_axes_with_splits(ax[0, 0], dataset.mask_tr, dataset.mask_va, dataset.mask_te)
    ax[0, 0].legend(handles=legend, loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1, 0].imshow(gt_rgb)
    ax[2, 0].imshow(classification_rgb)
    img = ax[1, 1].matshow(vac)
    add_colorbar(fig, img, ax[1, 1], x_shift=.1)
    img = ax[2, 1].matshow(dis)
    add_colorbar(fig, img, ax[2, 1], x_shift=.1)
    set_each_ax_y_label(["Optical", "GT", "S-BMLP\nClassification", "Vacuity", "Dissonance"],
                        [ax[0, 0], ax[1, 0], ax[2, 0], ax[1, 1], ax[2, 1]])
    # set_each_ax_y_label(["Optical", "GT", "GCN\nClassification"], [ax[0, 0], ax[1, 0], ax[1, 1]])
    remove_ax_list_ticks(ax.flatten())

    fig.savefig("docs/25_may/sbmlp.pdf", bbox_inches="tight")
