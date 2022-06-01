import os
import math
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

# import rs.ground_truth
import uncertainty_utils
from .simplex_plots import *
from .utils import *
# from rs import rgb, hyperspectral, lidar
from uncertainty_utils import vacuity_uncertainty, dissonance_uncertainty
# from datasets import HoustonDatasetMini

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


def draw_dirichlet(alpha_list):
    fig, axes = plt.subplots(1, len(alpha_list))

    for ax, alpha in zip(axes, alpha_list):
        dist = Dirichlet(alpha)
        draw_pdf_contours(dist, ax)
        title = r'$\boldsymbol{\alpha}$ = [%.3f, %.3f, %.3f]$^T$' % tuple(alpha)
        uncertainties = uncertainty_utils.get_subjective_uncertainties(np.array(alpha).reshape(1, -1))
        # caption = (r'$u_v$ = %.3f, $u_{diss}$ = %.3f,' + '\n' + r'$u_{entropy}$ = %.3f') % (*uncertainties.values(),)
        caption = (r'vacuity = %.3f' + '\n' + 'dissonance = %.3f' + '\n' + r'entropy = %.3f') % (*uncertainties.values(),)
        ax.set_title(title, fontdict={'fontsize': 8})
        ax.set_xlabel(caption, fontdict={'fontsize': 8})
        # plt.subplot(2, len(alpha_list), i + 1 + len(alpha_list))
    plt.savefig("figures/dirichlet_figs.pdf", bbox_inches="tight")


def data_figure_rgb():
    fig, ax = dense_fig(1, 1)
    ax.imshow(np.rollaxis(rgb.training_array()[..., 1800:3600], 0, 3))
    fig.savefig("docs/19_oct/data_figure_rgb.pdf", bbox_inches="tight")


def channels_figure(margin=.1, width=.6):
    # image = [np.arange(50 * 100).reshape(50, 100)] * 4
    # shape: channels, height, width
    image = np.concatenate([rgb.training_array(), hyperspectral.training_array(), lidar.training_array()])[...,
            1800:3600]
    aspect, numb = image.shape[1] / image.shape[2], len(image)
    height = width * aspect
    dx, dy = (1 - 2 * margin - width) / (numb - 1), (1 - 2 * margin - height) / (numb - 1)
    fig = plt.figure(figsize=(8, 4))
    axes = [fig.add_axes([margin + n * dx, margin + (numb - n - 1) * dy, width, height]) for n in
            range(numb - 1 - 2, -1, -1)]  # extra -2 because we'll show 3 channels in one for the rgb
    for i, ax in enumerate(axes):
        if i != len(axes) - 1:
            ax.matshow(image[len(axes) - i - 1], cmap='hsv')
        else:
            ax.imshow(np.rollaxis(image[:3], 0, 3))
        ax.axis('off')
    fig.savefig('docs/19_oct/channels.pdf', bbox_inches='tight')


def gt_figure():
    fig, ax = dense_fig(1, 1, width=6, height=5)
    gt = rs.ground_truth.data.array[0, ..., 1800:3600]
    ax.imshow(classes_array_to_colormapped_array(gt))
    ax.axis('off')
    fig.savefig('docs/19_oct/gt.pdf', bbox_inches='tight')


def graph_figure(radius=0.15, margin=.1, numb=10):
    fig, ax = plt.subplots(figsize=(8, 4))
    np.random.seed(0)
    centers = np.random.uniform(0, 4, (numb, 2)) ** 1.1
    adj = np.random.uniform(0, .8, (numb, numb))
    adj = adj + adj.T
    adj[adj < 1] = 0
    adj[adj > 1] = 1

    for i in range(numb):
        for j in range(i, numb):
            if adj[i, j] == 1:
                ax.plot([centers[i, 0], centers[j, 0]], [centers[i, 1], centers[j, 1]], zorder=-1, color='black')

    circles = [plt.Circle((coord[0], coord[1]), radius, facecolor='r', lw=1, edgecolor='black') for coord in centers]
    for circle in circles:
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlim(centers[:, 0].min() - radius - margin, centers[:, 0].max() + radius + margin)
    ax.set_ylim(centers[:, 1].min() - radius - margin, centers[:, 1].max() + radius + margin)
    ax.axis('off')
    fig.savefig('docs/19_oct/graph_fig.pdf', bbox_inches='tight')


def unc_eq_fig():
    fig, ax = plt.subplots(1, 1, figsize=(4, 1))
    ax.text(.1, .1, r"$u_v = \frac{K}{S}$", fontdict={'fontsize': 16})
    ax.text(.2, .9,
            r"$u_{diss} = \sum_{i=1}^K\left(\frac{\frac{\alpha_i-1}{S}\sum_{j\neq i} \frac{\alpha_j-1}{S}\left(1-\frac{\vert\alpha_j-\alpha_i\vert}{\alpha_j+\alpha_i}\right)}{\sum_{j\neq i} \frac{\alpha_j-1}{S}}\right)$",
            fontdict={'fontsize': 16})
    ax.text(.6, .1, r"$u_{entropy}=\sum_{i=1}^K-\frac{\alpha_i}{S}\log\left(\frac{\alpha_i}{S}\right)/\log\left(K\right)$",
            fontdict={'fontsize': 16})
    ax.axis("off")
    fig.savefig("docs/19_oct/unc_eq.pdf", bbox_inches="tight")


def strength_eq_fig():
    fig, ax = plt.subplots(1, 1, figsize=(1, 1))
    ax.text(.2, .2, r"$S=\sum_{i=1}^K\alpha_i$", fontdict={'fontsize': 18})
    ax.axis("off")
    fig.savefig("docs/19_oct/strength_eq.pdf", bbox_inches="tight")


def beirut_fig():
    alpha = np.load("experiments/Beirut/NoCoords/alpha.npy")
    print(alpha.shape)
