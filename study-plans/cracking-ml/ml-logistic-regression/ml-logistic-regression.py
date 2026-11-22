import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for _ in range(n_iters):
        logits = X @ w + b
        probs = 1 / (1 + np.exp(-logits))
        errs = probs - y
        dw = (1 / m) * (X.T @ errs)
        db = np.mean(errs)
        w -= lr * dw
        b -= lr * db

    return [round(float(v), 4) for v in w], round(float(b), 4)
