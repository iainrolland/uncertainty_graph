import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from datasets import HoustonDataset_k2


def close_fig(numb_rows, numb_cols, boundary=0.1, width=9, height=6, keep_square=True):
    fig, ax = plt.subplots(numb_rows, numb_cols, figsize=(width, height))
    if isinstance(ax, np.ndarray):
        if ax.ndim == 2:
            ax_list = [b for a in ax for b in a]
        else:
            ax_list = [a for a in ax]
    else:
        ax_list = [ax]
    if keep_square:
        if width / height >= numb_cols / numb_rows:  # if figure wider ratio than grid
            topmost = 1 - boundary
            vertical_step = (1 - 2 * boundary) / numb_rows
            horizontal_step = vertical_step * height / width
            leftmost = (1 - numb_cols * horizontal_step) / 2
        else:  # if figure taller ratio than grid
            leftmost = boundary
            horizontal_step = (1 - 2 * boundary) / numb_cols
            vertical_step = horizontal_step * width / height
            topmost = 0.5 + (numb_rows * vertical_step) / 2
    else:
        topmost, leftmost = 1 - boundary, boundary
        horizontal_step, vertical_step = (1 - 2 * boundary) / numb_cols, (1 - 2 * boundary) / numb_rows
    for i in range(numb_rows):
        for j in range(numb_cols):
            ax_list[i * numb_cols + j].set_yticks([])
            ax_list[i * numb_cols + j].set_xticks([])
            ax_list[i * numb_cols + j].set_position(
                [leftmost + j * horizontal_step, topmost - (i + 1) * vertical_step, horizontal_step, vertical_step])
    if isinstance(ax, np.ndarray):
        return fig, np.array(ax_list).reshape(ax.shape)
    else:
        return fig, ax_list[0]


def load_gt(path="houston_data/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif"):
    return rasterio.open(path).read()[0]


def set_each_ax_y_label(label_list, ax_list):
    for ax, label in zip(ax_list, label_list):
        ax.set_ylabel(label)


def set_rcParams():
    # plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams['font.size'] = 3


def add_colorbar(fig, img, one_ax, x_shift=0.1, height_scale=0.95):
    bounds = one_ax.get_position().bounds
    bounds = (bounds[0] + x_shift, (3 - height_scale) * bounds[1] / 2, bounds[2], bounds[3] * height_scale,)
    cbar = fig.add_axes(bounds)
    cbar.axis("off")
    fig.colorbar(img, ax=cbar)


def get_shaded_data_split(split="Training"):
    supported_splits = ["Training", "Validation", "Test"]
    if split not in supported_splits:
        raise ValueError("split was {} but must be one of {}.".format(split, "/".join(supported_splits)))

    dataset = HoustonDataset_k2()

    for one_split, mask in zip(supported_splits, [dataset.mask_tr, dataset.mask_va, dataset.mask_te]):
        if split == one_split:
            mask = unflatten_array(mask)[..., 0]
            return mask


def shade_one_axes_with_splits(ax):
    legend = []
    for split, color in zip(["Training", "Validation", "Test"],
                            [[1., 0., 0., 0.5], [0., 1., 0., 0.5], [0., 0., 1., 0.5]]):
        shaded = get_shaded_data_split(split)
        shaded_rgb = np.zeros(load_gt().shape + (4,)).astype("float32")
        shaded_rgb[shaded] = color
        ax.imshow(shaded_rgb)
        legend.append(mpatches.Patch(color=color, label=split))
    return legend


def unflatten_array(array, output_shape=load_gt().shape):
    return array.reshape(output_shape + (-1,))
