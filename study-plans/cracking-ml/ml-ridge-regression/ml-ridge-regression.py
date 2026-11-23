import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        pred = X @ w + b
        err = pred - y
        dw = 2/m * (X.T @ err) + (2*alpha*w)
        db = 2 * np.mean(err)
        w -= lr * dw
        b -= lr * db

    return [round(float(v), 4) for v in w], round(float(b), 4)
