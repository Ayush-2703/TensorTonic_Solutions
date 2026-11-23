import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    x = np.asarray(x, dtype=float)
    z = np.exp(-np.abs(x))          # always in (0, 1], so exp never overflows
    out = np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
    return float(out) if out.ndim == 0 else out
    pass