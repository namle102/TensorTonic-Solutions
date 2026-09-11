import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.asarray(X, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.int64)
    
    if mode == 'train':
        return ((X * mask) / (1.0 - drop_prob))
    return X