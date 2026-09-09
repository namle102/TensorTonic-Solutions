import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ w + b
        err = y_hat - y
        dw = 2/n * (X.T @ err)
        db = 2/n * np.sum(err)
        w -= lr * dw
        b -= lr * db

    return (w, b)
