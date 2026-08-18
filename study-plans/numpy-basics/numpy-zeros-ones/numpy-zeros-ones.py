import numpy as np

def create_filled_array(shape, kind):
    """
    Returns: 2D numpy array of given shape with dtype float64
    """
    row, col = shape[0], shape[1]
    if kind == 'zeros':
        return np.zeros((row, col))
    return np.ones((row, col))