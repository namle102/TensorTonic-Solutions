import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    x = float(x)

    if activation == 'sigmoid':
        out = 1.0 / (1.0 + np.exp(-x))
        deriv = out * (1 - out)

    elif activation == 'relu':
        out = max(0.0, x)
        deriv = 1.0 if x > 0 else 0.0

    elif activation == 'leaky_relu':
        out = x if x > 0 else 0.01 * x
        deriv = 1.0 if x > 0 else 0.01

    elif activation == "tanh":
        t = float(np.tanh(x))
        out = t
        deriv = float(1 - t ** 2)

    elif activation == "gelu":
        c = np.sqrt(2.0 / np.pi)
        inner = c * (x + 0.044715 * x ** 3)
        t = float(np.tanh(inner))
        out = float(0.5 * x * (1 + t))
        sech2 = 1 - t ** 2
        inner_deriv = c * (1 + 3 * 0.044715 * x ** 2)
        deriv = float(0.5 * (1 + t) + 0.5 * x * sech2 * inner_deriv)

    elif activation == "swish":
        s = 1.0 / (1.0 + np.exp(-x))
        out = float(x * s)
        deriv = float(s + x * s * (1 - s))

    return [round(out, 4), round(deriv, 4)]
