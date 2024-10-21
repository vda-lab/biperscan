"""
This module implements several point-cloud measures that can be used as lenses
with BPSCAN. Most of these functions are based on the documentation of the
python implementation of Mapper.
"""

import numpy as np
from scipy.spatial.distance import cdist, squareform

from ._impl import minmax_of


def normalize(values: np.ndarray[np.float64]) -> np.ndarray[np.float32]:
    """Scales values to lie between 0 and 1."""
    i, a = minmax_of(values)
    out = np.empty_like(values, dtype=np.float32)
    if i == a:
        out[:] = values - i
    else:
        out[:] = (values - i) / (a - i)
    return out


def _gauss_similarity(distances: np.ndarray, *, sigma: float = 0.3) -> np.ndarray:
    """Transforms distances into similarity using a Gaussian kernel.

    Parameters
    ----------
    distances : 2D numpy array
      The distance matrix.
    sigma : float, optional (default = 0.3)
      The variance of the Gaussian kernel to apply.

    Returns
    -------
    A numpy array with node similarities.
    """
    return np.exp(-np.multiply(distances, distances) / (2 * sigma * sigma))


def negative_density(
    X: np.ndarray, distance_matrix: np.ndarray, *, sigma: float = 0.3, **kwargs
) -> np.ndarray:
    """Computes point-cloud density

    Parameters
    ----------
    X : 2D NumPy array
        The original data matrix. Not used in this function.
    distances : 1D numpy array
        The condensed distance matrix.
    sigma : float, optional (default = 0.3)
        stddev of Gaussian smoothing kernel.

    Returns
    -------
    N by 1 numpy array containing the negative vertex density values normalized to
    lie between 0 and 1.

    """
    distance_matrix = squareform(distance_matrix)
    return normalize(-np.sum(_gauss_similarity(distance_matrix, sigma=sigma), axis=0))


def negative_eccentricity(
    X: np.ndarray, distance_matrix: np.ndarray, *, power: float = np.inf, **kwargs
) -> np.ndarray:
    """Computes point-cloud eccentricity

    Parameters
    ----------
    X : 2D NumPy array
      The original data matrix. Not used in this function.
    distances : 1D numpy array
      The condensed distance matrix
    power : int, optional (default = np.inf)
      The power to use, may also be infinite.

    Returns
    -------
    N by 1 numpy array containing the negative vertex eccentricity values scaled
    to lie between 0 and 1.
    """
    distance_matrix = squareform(distance_matrix)
    if power == np.inf:
        values = np.max(distance_matrix, axis=0)
    else:
        num_points = distance_matrix.shape[0]
        values = np.power(
            np.sum(np.power(distance_matrix, power), axis=0) / num_points,
            1 / power,
        )
    return normalize(-values)


def negative_distance_to_median(
    X: np.ndarray,
    distance_matrix: np.ndarray,
    *,
    metric: str = "euclidean",
    metric_kws: dict = None,
    **kwargs
) -> np.ndarray:
    """Computes distance to the median centroid.

    Parameters
    ----------
    X : 2D NumPy array
      The original data matrix.
    distances : 1D numpy array
      The condensed distance matrix, not used in this function.

    Returns
    -------
    N by 1 numpy array containing the negative distance to centroid values
    scaled to lie between 0 and 1.
    """
    if metric_kws is None:
        metric_kws = dict()
    if X is None:
        raise ValueError("Raw data must be provided to compute the median centroid.")
    values = cdist(np.median(X, axis=0)[None], X, metric=metric, **metric_kws)[0]
    return normalize(-values)


def negative_distance_to_mean(
    X: np.ndarray,
    distance_matrix: np.ndarray,
    *,
    metric: str = "euclidean",
    metric_kws: dict = None,
    **kwargs
) -> np.ndarray:
    """Computes distance to the mean centroid.

    Parameters
    ----------
    X : 2D NumPy array
      The original data matrix.
    distances : 1D numpy array
      The condensed distance matrix, not used in this function.

    Returns
    -------
    N by 1 numpy array containing the negative distance to centroid values
    scaled to lie between 0 and 1.
    """
    if X is None:
        raise ValueError("Raw data must be provided to compute the mean centroid.")
    if metric_kws is None:
        metric_kws = dict()
    values = cdist(np.mean(X, axis=0)[None], X, metric=metric, **metric_kws)[0]
    return normalize(-values)


available_lenses = dict(
    negative_eccentricity=negative_eccentricity,
    negative_distance_to_median=negative_distance_to_median,
    negative_distance_to_mean=negative_distance_to_mean,
)
