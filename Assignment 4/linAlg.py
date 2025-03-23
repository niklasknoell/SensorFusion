from typing import Sequence, Union
import numpy as np


def block_diag(*arrs: Sequence[Union[np.ndarray, float]]):
    """
    Construct a block diagonal matrix from a sequence of 1D or 2D numpy arrays or floats.

    Args:
        arrs (Sequence[Union[np.ndarray, float]]): A sequence of numpy arrays and/or floats.
            Each element can either be a numpy ndarray or a float. If a float is provided,
            it will be treated as a 1x1 matrix.

    Returns:
        np.ndarray: A block diagonal matrix formed from the input arrays.

    Example:
        >>> block_diag(np.array([[1, 2]]), np.array([[3, 4], [5, 6]]))
        array([[1, 2, 0, 0],
               [0, 0, 3, 4],
               [0, 0, 5, 6]])

        >>> block_diag(1.0, np.array([[2, 3]]))
        array([[1., 0.],
               [0., 2., 3.]])
    """
    # Handle empty input
    if len(arrs) == 0:
        return np.array([])

    # Convert 1D arrays to 2D
    arrs = [np.atleast_2d(arr) for arr in arrs]

    # Get shapes of input arrays
    shapes = np.array([a.shape for a in arrs])

    # Calculate output shape
    out_shape = shapes.sum(axis=0)

    # Create output array filled with zeros
    out = np.zeros(out_shape, dtype=arrs[0].dtype)

    # Fill output array with input arrays
    r, c = 0, 0
    for arr in arrs:
        rr, cc = arr.shape
        out[r : r + rr, c : c + cc] = arr
        r += rr
        c += cc

    return out
