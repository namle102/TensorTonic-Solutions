import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    # X dim = n x d
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64).reshape(-1)
    K = n_classes
    n, d = X.shape
    
    # W dim = d x K; b dim = d,
    W = np.zeros((d, K))
    b = np.zeros(K)

    Y_oh = np.zeros((n, K))
    Y_oh[np.arange(n), y] = 1

    for _ in range(n_iters):
        Z = X @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        err = P - Y_oh
        dW = 1/n * (X.T @ err)
        db = 1/n * np.sum(err, axis=0)
        W -= lr * dW
        b -= lr * db

    return (W, b)