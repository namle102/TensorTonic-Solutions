import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array/
    """
    A = np.asarray(A, dtype=int)
    return A.T
