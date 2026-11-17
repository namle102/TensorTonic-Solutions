import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    m, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        err = X@w + b - y
        dw = (2.0/m) * (X.T@err)
        db = (2.0/m) * np.sum(err)

        w -= lr * dw
        b -= lr * db

    return [round(float(value),4) for value in w], round(float(b),4)