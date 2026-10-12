import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns mean, median, and mode as Python floats in a dictionary.
    """
    x = np.array(x, dtype=float)
    unique_nums, counts = np.unique(x, return_counts=True)
    mode = unique_nums[np.argmax(counts)]
    return {
        'mean': float(np.mean(x)),
        'median': float(np.median(x)),
        'mode': float(mode)
    }