import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.asarray(X, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.int64)

    if mode == 'test':
        return [[round(float(v), 4) for v in row] for row in X]

    out = X * mask / (1.0 - drop_prob)
    return [[round(float(v), 4) for v in row] for row in out]