import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape[0], X.shape[1]
    w, b = np.zeros(d), 0

    for epoch in range(epochs):
        y_hat = X @ w + b
        grad_w = 2/n * X.T @ (y_hat - y)
        grad_b = 2/n * np.sum(y_hat - y)
        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b
