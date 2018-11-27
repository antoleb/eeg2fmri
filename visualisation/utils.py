import numpy as np


def concatenate_collage(tensor):
    col_list = [np.concatenate([
        tensor[:, :, i] for i in range(start_i, start_i + 5)
    ], 0) for start_i in range(0, 30, 5)]
    collage = np.concatenate(col_list, 1)
    return collage