import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0

    for _ in range(epochs):
        pred = X @ w + b
        err = pred - y
        dw = 2/m * (X.T @ err)
        db = 2/m * np.sum(err)
        w -= lr * dw
        b -= lr * db

    return w, b
        
