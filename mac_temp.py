from datasets import HoustonDatasetMini
import numpy as np
from utils import set_seeds


def update(state, truth, adjacency, mask, step_size=.2):
    flow_in = (adjacency / adjacency.sum(axis=0)).dot(state)
    flow_out = state
    update = step_size * (flow_in - flow_out)
    update_l2 = np.mean(np.power(update, 2))
    state += update
    # new_state = ((1 - step_size) * sparse.identity(adjacency.shape[0]) + step_size * adjacency.dot(
    #     np.diag(adjacency.sum(axis=0).A1 ** -1))).dot(state)
    state[mask] = truth[mask]  # boundary conditions
    return state, update_l2


if __name__ == "__main__":
    # Load dataset
    set_seeds(0)
    dataset = HoustonDatasetMini()
    adj = dataset[0].a
    yt = dataset[0].y
    mask_tr = dataset.mask_tr

    # Set initial state
    y = np.ones_like(yt) / yt.shape[1]
    y[mask_tr] = yt[mask_tr]

    # Update until convergence
    y, l2 = update(y, yt, adj, mask_tr)
    l2_list = [l2]
    for t in range(1):
        l2_old = l2
        y, l2 = update(y, yt, adj, mask_tr)
        l2_list.append(l2)
        if l2 / l2_old >= 0.99:
            break

    # Compute accuracy
    print(np.sum(yt.argmax(axis=1) == y.argmax(axis=1)) / yt.shape[0])
