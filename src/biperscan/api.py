import time
import numpy as np
from joblib import Memory
from typing import Callable
from scipy.stats import rankdata
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KDTree, BallTree
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import pdist, squareform
from hdbscan._hdbscan_reachability import mutual_reachability

from .lenses import available_lenses
from ._impl import (
    compute_minimal_presentation,
    compute_minpres_merges,
    compute_linkage_hierarchy,
    mutual_reachability_from_pdist,
)
from .plots import LinkageHierarchy, MinimalPresentation, MergeList, SimplifiedMergeList


KDTREE_VALID_METRICS = [
    "euclidean",
    "l2",
    "minkowski",
    "p",
    "manhattan",
    "cityblock",
    "l1",
    "chebyshev",
    "infinity",
]
BALLTREE_VALID_METRICS = [
    "braycurtis",
    "canberra",
    "dice",
    "hamming",
    "haversine",
    "jaccard",
    "mahalanobis",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
]


class BPSCAN(BaseEstimator, ClusterMixin):
    """
    Perform Bi-Persistence clustering. BPSCAN adapts HDBSCAN* to operate on a
    bi-filtration of the data, where the filtration is defined by a lens
    function and a mutual reachability distance.

    Parameters
    ----------
    min_samples : int, default=None
        The number of samples in a neighborhood for a point to be considered as
        a core point. If None, defaults to min_cluster_size.
    min_cluster_size : int, default=10
        The minimum number of samples to be a cluster.
    distance_fraction : float, default=1.0
        The fraction of the maximum distance grade to use a upper distance limit
        when extracting merges.
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of the
        options allowed by :func:`sklearn.metrics.pairwise_distances` for its
        metric parameter. If metric is "precomputed", X is assumed to be a
        distance matrix. Otherwise, X is passed to the metric function as
        argument(s).
    metric_kws : dict, default=None
        Additional keyword arguments to pass to the metric function.
    lens : str or callable or array of shape (n_samples,)
        The lens function to use when computing the bi-filtration. If a string,
        it must be a key in :func:`biperscan.lenses.available_lenses`. If a
        callable, must return a float32 array of lens values. If an array, must
        be a float32 array of lens values.
    lens_kws : dict, default=None
        Additional keyword arguments to pass to the lens function.
    memory : str or None, default=None
        A path to store the cache or None to disable caching.

    Attributes
    -------
    distances_ : array of shape (n_samples, n_samples)
        The mutual reachability distance matrix in condensed form.
    lens_values_ : array of shape (n_samples,)
        The computed lens values.
    lens_grades_ : array of shape (n_samples,)
        The lens grade for each point.
    minimal_presentation_ : :class:`~biperscan.plots.MinimalPresentation`
        The minimal presentation of the bi-filtration.
    merges_ : :class:`~biperscan.plots.MergeList`
        The detected merges.
    simplified_merges_ : :class:`~biperscan.plots.SimplifiedHierarchy`
        The simplified merges.
    linkage_hierarchy_ : :class:`~biperscan.plots.LinkageHierarchy`
        The linkage hierarchy graph. This property is computed on demand and not
        cached.
    membership_ : array of shape (n_samples,n_groups)
        A binary membership matrix indicating which points belong to which
        groups. Groups can overlap and relate to the child and parent sides of
        the simplified merges.
    labels_ : array of shape (n_samples,)
        The computed cluster labels. The labels identify points with the same
        membership combinations.
    timers_ : dict
        The time spent on each step of the clustering process.
    """

    def __init__(
        self,
        min_samples: int | None = None,
        min_cluster_size: int = 10,
        distance_fraction: float = 1.0,
        metric: str | Callable = "euclidean",
        metric_kws: dict | None = None,
        lens: str | Callable | np.ndarray[np.float32] = "negative_distance_to_median",
        lens_kws: dict | None = None,
        memory: str | None = None,
    ):
        """See :class:`~biperscan.api.BPSCAN` for documentation."""
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.distance_fraction = distance_fraction
        self.metric = metric
        self.metric_kws = metric_kws
        self.lens = lens
        self.lens_kws = lens_kws
        self.memory = memory

    def fit(self, X: np.ndarray[np.float64], y: np.ndarray = None):
        """Performs BPSCAN clustering on the given data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features), or \
            array of shape (1, n_samples * (n_samples - 1) // 2)
          A feature array, or condensed distance array if metric='precomputed'.
        y : None
          Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        (
            self.distances_,
            self.lens_values_,
            self.lens_grades_,
            self._col_to_edge,
            self._row_to_point,
            self._minimal_presentation,
            self._merges,
            self._simplified_merges,
            self.membership_,
            self.labels_,
            self.timers_,
        ) = bpscan(X, **self.get_params())
        return self

    @property
    def minimal_presentation_(self):
        check_is_fitted(self, "_minimal_presentation")
        return MinimalPresentation(
            self.distances_,
            self.lens_values_,
            self.lens_grades_,
            self._col_to_edge,
            self._row_to_point,
            self._minimal_presentation,
        )

    @property
    def merges_(self):
        check_is_fitted(self, "_minimal_presentation")
        return MergeList(
            self.distances_,
            self.lens_values_,
            self._col_to_edge,
            self._row_to_point,
            self._minimal_presentation,
            self._merges,
        )

    @property
    def simplified_merges_(self):
        check_is_fitted(self, "_minimal_presentation")
        return SimplifiedMergeList(
            self.distances_,
            self.lens_values_,
            self._col_to_edge,
            self._row_to_point,
            self._minimal_presentation,
            self._simplified_merges,
        )

    @property
    def linkage_hierarchy_(self):
        check_is_fitted(self, "_minimal_presentation")
        self._linkage_hierarchy, linkage_time = compute_linkage_hierarchy(
            self._minimal_presentation, self._row_to_point
        )
        self.timers_["linkage"] = linkage_time
        return LinkageHierarchy(
            self.distances_,
            self.lens_values_,
            self.lens_grades_,
            self._col_to_edge,
            self._row_to_point,
            self._linkage_hierarchy,
        )


def bpscan(
    X,
    *,
    min_samples: int | None = None,
    min_cluster_size: int = 10,
    distance_fraction: float = 1.0,
    metric: str | Callable = "euclidean",
    metric_kws: dict | None = None,
    lens: (
        str | Callable | np.ndarray[np.float32]
    ) = "negative_distance_to_median",  # Must return float32 array
    lens_kws: dict | None = None,
    memory: str | None = None,
):
    """
    Perform Bi-Persistence clustering on the given data. BPSCAN adapts HDBSCAN*
    to operate on a bi-filtration of the data, where the filtration is defined
    by a lens function and a mutual reachability distance. 
    
    Parameters
    ----------
    X : array of shape (n_samples, n_features), or \
        array of shape (1, n_samples * (n_samples - 1) // 2) A feature array, or
        condensed distance array if metric='precomputed'.
    min_samples : int, default=None
        The number of samples in a neighborhood for a point to be considered as
        a core point. If None, defaults to min_cluster_size.
    min_cluster_size : int, default=10
        The minimum number of samples to be a cluster.
    distance_fraction : float, default=1.0
        The fraction of the maximum distance grade to use a upper distance limit
        when extracting merges.
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of the
        options allowed by :func:`sklearn.metrics.pairwise_distances` for its
        metric parameter. If metric is "precomputed", X is assumed to be a
        distance matrix. Otherwise, X is passed to the metric function as
        argument(s).
    metric_kws : dict, default=None
        Additional keyword arguments to pass to the metric function.
    lens : str or callable or array of shape (n_samples,)
        The lens function to use when computing the bi-filtration. If a string,
        it must be a key in :func:`biperscan.lenses.available_lenses`. If a
        callable, must return a float32 array of lens values. If an array, must
        be a float32 array of lens values.
    lens_kws : dict, default=None
        Additional keyword arguments to pass to the lens function.
    memory : str or None, default=None
        A path to store the cache or None to disable caching.

    Returns
    -------
    distances : array of shape (n_samples, n_samples)
        The mutual reachability distance matrix in condensed form.
    lens_values : array of shape (n_samples,)
        The computed lens values.
    lens_grades : array of shape (n_samples,)
        The lens grade for each point.
    col_to_edge : array of shape (n_edges,)
        Mapping from column index to index in the condensed distance matrix.
    row_to_point : array of shape (n_samples,)
        Mapping from minimal presentation row index to data point index. 
    minimal_presentation : dict
        The minimal presentation of the bi-filtration. Contains the following
        keys: 'lens_grade', 'distance_grade', 'parent', 'child'.
    merges : dict
        The merges extracted from the minimal presentation. Contains the
        following keys: 'start_column', 'end_column', 'lens_grade',
        'distance_grade', 'parent', 'child', 'parent_side', 'child_side'. Merges
        are ordered by increasing end column.
    simplified_merges : dict
        The merges that remain after combining similar merges. Contains the 
        following keys: 'parent', 'child', 'parent_side', 'child_side'.
    membership : array of shape (n_samples,n_groups)
        A binary membership matrix indicating which points belong to which
        groups. Groups can overlap and relate to the child and parent sides
        of the simplified merges.
    labels : array of shape (n_samples,)
        The computed cluster labels.
    timers : dict
        The time spent on each step of the clustering process.
    """
    X = check_array(X, ensure_all_finite=True, ensure_2d=metric != "precomputed")

    # Fill in default parameters
    if min_samples is None:
        min_samples = min_cluster_size
    if metric_kws is None:
        metric_kws = dict()
    if lens_kws is None:
        lens_kws = dict()
    if memory is None:
        memory = Memory(None, verbose=0)
    elif isinstance(memory, str):
        memory = Memory(memory, verbose=0)

    # Check parameter values
    if not isinstance(min_cluster_size, int) or min_cluster_size <= 1:
        raise ValueError("Min cluster size must be a positive integer greater than 1.")
    if not isinstance(min_samples, int) or min_samples <= 0:
        raise ValueError("Min cluster size must be a positive integer.")
    if min_samples >= X.shape[0] or min_cluster_size >= X.shape[0]:
        raise ValueError(
            "Too few data points for the specified min cluster size or min samples."
        )
    if (
        not isinstance(distance_fraction, float)
        or distance_fraction < 0.0
        or distance_fraction > 1.0
    ):
        raise ValueError("Distance fraction must be a real number between 0 and 1.")

    # Compute the distance matrix
    if metric == "precomputed":
        distance_time = 0
        reachability = X.astype(np.float32)
    else:
        if metric in KDTREE_VALID_METRICS:
            compute_reachability = _kdtree_mutual_reachability
            remap_metrics = dict(
                l2="euclidean",
                l1="cityblock",
                manhattan="cityblock",
                p="minkowski",
                infinity="minkowski",
            )
            if metric == "infinity":
                metric_kws["p"] = np.inf
            metric = remap_metrics.get(metric, metric)
        elif metric in BALLTREE_VALID_METRICS:
            compute_reachability = _balltree_mutual_reachability
        else:
            compute_reachability = _slow_mutual_reachability
        reachability, distance_time = memory.cache(_as_float32)(
            compute_reachability, X, metric, min_points=min_samples, **metric_kws
        )

    # Compute the lens values
    if type(lens) is np.ndarray:
        # implementation expects 32bit floats!
        lens_time = 0
        point_lens = check_array(lens, ensure_2d=False).astype(np.float32)
    else:
        if isinstance(lens, str) and lens in available_lenses:
            lens_fun = available_lenses[lens]
        elif isinstance(lens, Callable):
            lens_fun = lens
        point_lens, lens_time = memory.cache(_timed)(
            lens_fun,
            X if metric != "precomputed" else None,
            reachability,
            metric=metric,
            **metric_kws,
            **lens_kws,
        )

    # Compute bi-filtration minimal presentation
    (
        col_to_edge,
        row_to_point,
        point_lens_grades,
        minimum_presentation,
        matrix_time,
        minpres_time,
    ) = memory.cache(compute_minimal_presentation)(reachability, point_lens)

    # Extract merge hierarchy from the minimal presentation
    (
        merges,
        merge_time,
        simplified_merges,
        simplify_time,
    ) = memory.cache(
        _compute_merges
    )(minimum_presentation, len(point_lens), min_cluster_size, distance_fraction)

    # Compute segmentation
    (labels, membership), label_time = memory.cache(_timed)(
        _extract_labels, simplified_merges, row_to_point
    )

    # Collect times
    times = dict(
        distances=distance_time,
        lens_values=lens_time,
        matrix=matrix_time,
        minpres=minpres_time,
        merges=merge_time,
        simplify=simplify_time,
        labels=label_time,
    )

    return (
        reachability,
        point_lens,
        point_lens_grades,
        col_to_edge,
        row_to_point,
        minimum_presentation,
        merges,
        simplified_merges,
        membership,
        labels,
        times,
    )


def _kdtree_mutual_reachability(X, metric, min_points=5, **kwargs):
    start = time.perf_counter()
    tree = KDTree(X, metric=metric, **kwargs)
    min_points = min(X.shape[0] - 1, min_points)
    core_distances = tree.query(X, k=min_points)[0][:, -1]

    dists = pdist(X, metric=metric, **kwargs)
    dists = mutual_reachability_from_pdist(core_distances, dists, X.shape[0])
    duration = time.perf_counter() - start
    return dists, duration


def _balltree_mutual_reachability(X, metric, min_points=5, **kwargs):
    start = time.perf_counter()
    tree = BallTree(X, metric=metric, **kwargs)
    min_points = min(X.shape[0] - 1, min_points)
    core_distances = tree.query(X, k=min_points)[0][:, -1]

    dists = pdist(X, metric=metric, **kwargs)
    dists = mutual_reachability_from_pdist(core_distances, dists, X.shape[0])
    duration = time.perf_counter() - start
    return dists, duration


def _slow_mutual_reachability(X, metric, min_points=5, **kwargs):
    """Computes condensed mutual reachability matrix for the given data."""
    start = time.perf_counter()
    dists = pdist(X, metric=metric, **kwargs) if metric != "precomputed" else X
    mreach = squareform(
        mutual_reachability(squareform(dists), min_points=min_points), checks=False
    )
    duration = time.perf_counter() - start
    return mreach, duration


def _as_float32(fun, *args, **kwargs):
    """Wraps a function to cast its output to float32."""
    res = fun(*args, **kwargs)
    return (res[0].astype(np.float32), res[1])


def _compute_merges(
    minimal_presentation, num_points, min_cluster_size, distance_fraction
):
    """Extracts merges from the minimal presentation and creates a
    hierarchy connecting merges if they evolve into each other."""
    merges, merge_time = compute_minpres_merges(
        minimal_presentation, num_points, min_cluster_size, distance_fraction
    )

    start = time.perf_counter()
    simplified_merges = _combine_merges(merges)
    simplify_time = time.perf_counter() - start

    return (
        merges,
        merge_time,
        simplified_merges,
        simplify_time,
    )


def _combine_merges(merges):
    def isin(sorted_arr, key):
        idx = np.searchsorted(sorted_arr, key)
        return idx < len(sorted_arr) and sorted_arr[idx] == key

    groups = []
    for i in np.lexsort((merges["lens_grade"], merges["distance_grade"])):
        child = merges["child"][i]
        parent = merges["parent"][i]
        child_points = merges["child_side"][i]
        parent_points = merges["parent_side"][i]
        bigrade = (merges["lens_grade"][i], merges["distance_grade"][i])

        for j, ((pr, pl), (cr, cl), gs) in enumerate(groups):
            if isin(child_points, cr) and isin(parent_points, pr):
                groups[j] = (
                    (parent, np.union1d(pl, parent_points)),
                    (child, np.union1d(cl, child_points)),
                    [bigrade] + gs,
                )
                groups = sorted(groups, key=lambda x: x[0][0])
                break
        else:
            groups.append(((parent, parent_points), (child, child_points), [bigrade]))
            groups = sorted(groups, key=lambda x: x[0][0])

    return dict(
        parent=[g[0][0] for g in groups],
        child=[g[1][0] for g in groups],
        parent_side=[g[0][1] for g in groups],
        child_side=[np.setdiff1d(g[1][1], g[0][1], assume_unique=True) for g in groups],
        grade_trace=[
            dict(
                lens_grade=[bg[0] for bg in g[2]],
                distance_grade=[bg[1] for bg in g[2]],
            )
            for g in groups
        ],
    )


def _extract_labels(simplified_merges, row_to_point):
    num_merges = len(simplified_merges["parent"])
    membership = np.zeros((len(row_to_point), num_merges * 2), dtype=np.int32)
    for i, (parent_side, child_side) in enumerate(
        zip(simplified_merges["parent_side"], simplified_merges["child_side"])
    ):
        membership[row_to_point[parent_side], i * 2] = np.int32(1)
        membership[row_to_point[child_side], i * 2 + 1] = np.int32(1)

    bases = np.power(2, np.arange(membership.shape[1]))
    identifiers = (membership * bases).sum(axis=1)

    # uniques, counts = np.unique(identifiers, return_counts=True)
    # for id in uniques[counts < min_cluster_size]:
    #     identifiers[identifiers == id] = 0

    labels = rankdata(identifiers, method="dense") - 2
    return labels, membership


def _timed(fun, *args, **kwargs):
    start = time.perf_counter()
    out = fun(*args, **kwargs)
    duration = time.perf_counter() - start
    return out, duration
