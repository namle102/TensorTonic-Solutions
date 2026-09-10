def lasso_regression(X, y, lr, epochs, alpha):
    """
    Perform Lasso Regression using gradient descent with L1 subgradient.
    Returns: tuple of (weights_list, bias_float)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64).reshape(-1)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for i in range(epochs):
        y_hat = X @ w + b
        err = y_hat - y
        dw = 2/n * (X.T @ err) + alpha * np.sign(w)
        db = 2/n * np.sum(err)
        w -= lr * dw
        b -= lr * db

    return (w, b)