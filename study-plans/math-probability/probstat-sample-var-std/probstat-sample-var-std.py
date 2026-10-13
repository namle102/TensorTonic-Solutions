import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Return sample variance and standard deviation as Python floats.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    x_bar = np.mean(x)
    var = 0.0

    for num in x:
        var += (num - x_bar)**2
    var /= (n - 1)
    std = np.sqrt(var)

    return {
        'variance': var,
        'std_dev': std
    }