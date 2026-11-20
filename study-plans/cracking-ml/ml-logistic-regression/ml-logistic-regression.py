import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    m, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(n_iters):
        z = X @ w + b
        y_hat = 1 / (1 + np.exp(-z))
        err = y_hat - y
        dw = (1 / m) * (X.T @ err)
        db = (1 / m) * np.sum(err)
        w -= lr * dw
        b -= lr * db

    return [round(float(v), 4) for v in w], round(float(b), 4)
