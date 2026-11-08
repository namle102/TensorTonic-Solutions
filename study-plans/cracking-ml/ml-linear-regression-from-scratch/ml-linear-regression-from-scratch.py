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

    # compute grad descent
    for i in range(epochs):
        # compute err
        pred = X @ w + b
        err = pred - y

        # compute grad
        dw = (X.T @ err) * (2/m)
        db = np.mean(err) * 2

        # compute grad desc
        w -= lr * dw
        b -= lr * db

    return (w, b)