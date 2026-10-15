import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a float.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    temp = x - y
    res = float(np.sqrt(np.sum(temp**2)))
    return res