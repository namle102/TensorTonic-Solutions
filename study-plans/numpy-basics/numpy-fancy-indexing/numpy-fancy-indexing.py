import numpy as np

def select_by_index(arr, indices, axis):
    """
    Returns: 2D ndarray of float64
    """
    a = np.array(arr, dtype=np.float64)
    if axis == 0:
        return a[indices, :]
    else:
        return a[:, indices]