import os

from .utils import *
from rs import rgb
from .utils import unflatten_array

set_rcParams()


def load_alpha_vac_dis(directory):
    paths = [os.path.join(directory, file_name) for file_name in ["alpha.npy", "vacuity.npy", "dissonance.npy"]]
    return [np.load(path) for path in paths]


def read_colormap(path="houston_data/colormap.clr"):
    with open(path, "r") as f:
        colormap = np.array([[int(value) for value in row.split(" ")[1:]] for row in f.read().split("\n")])[:, :4]
    return colormap


def plot(model_dir="models/S_BGCN_HoustonDataset_k2_2021_04_26_2"):
    alpha, vac, dis = [unflatten_array(array) for array in load_alpha_vac_dis(model_dir)]
    colormap = read_colormap()

    fig, ax = close_fig(5, 1, width=7, height=8, keep_square=False)
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
    img = ax[3].matshow(vac)
    ax[3].set_xticks([])
    ax[3].set_yticks([])
    add_colorbar(fig, img, ax[3])
    img = ax[4].matshow(dis)
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    add_colorbar(fig, img, ax[4])
    set_each_ax_y_label(["Optical", "GT", "Classification", "Vacuity", "Dissonance"], ax)

    plt.show()
