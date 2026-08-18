import numpy as np

def reshape_array(data, operation):
    """
    Returns: ndarray of float64 with shape determined by the operation
    """
    arr = np.asarray(data, dtype=np.float64)
    if operation == 'flatten':
        return arr.reshape(-1)
    elif operation == 'transpose':
        return arr.T
    return arr.reshape(1, arr.shape[0], -1)
