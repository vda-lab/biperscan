import numba
import numpy as np


@numba.jit
def minmax(x: np.ndarray):
    """
    Efficiently computes the minimum and maximum of a 1D numpy array.
    """
    maximum = x[0]
    minimum = x[0]
    for i in x[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return minimum, maximum


def normalize(arr):
    i, a = minmax(np.asarray(arr))
    if i == a:
        return arr - i
    return (arr - i) / (a - i)
