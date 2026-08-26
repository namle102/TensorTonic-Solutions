import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    w, b = np.zeros(d), 0.0

    for _ in range(n_iters):
        z = X@w + b
        y_hat = 1 / (1 + np.exp(-z))
        err = y_hat - y
        dw = (1/n) * (X.T @ err)
        db = (1/n) * np.sum(err)
        w -= lr * dw
        b -= lr * db

    return (w, b)
