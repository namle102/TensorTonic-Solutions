import numpy as np

def create_array(data):
    """
    Returns: 2D numpy array with dtype float64
    """
    return np.atleast_2d(np.asarray(data, dtype=np.float64))