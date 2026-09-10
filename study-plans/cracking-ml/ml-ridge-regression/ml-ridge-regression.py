import numpy as np

def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ w + b
        err = y_hat - y
        dw = 2/n * (X.T @ err) + 2 * alpha * w
        db = 2/n * np.sum(err)
        w -= lr * dw
        b -= lr * db

    return (w, b)