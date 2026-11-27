import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    # Write code here
    train = np.asarray(X_train, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)

    # Treat 1D arrays as collections of 1D scalar samples
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)

    n_train = train.shape[0]
    n_test = test.shape[0]

    # Pairwise Euclidean distances of shape (n_test, n_train)
    diff = test[:, None, :] - train[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    # Stable sort preserves ascending index order on tied distances
    sorted_indices = np.argsort(dists, axis=1, kind="mergesort")

    # Pad with -1 if k exceeds the number of training samples
    result = np.full((n_test, k), -1, dtype=np.int64)
    num_neighbors = min(k, n_train)
    result[:, :num_neighbors] = sorted_indices[:, :num_neighbors]

    return result
    pass