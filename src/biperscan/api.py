import gc
import numpy as np
import networkx as nx
from joblib import Memory
from typing import Callable
from itertools import combinations
from collections import defaultdict
from sklearn.utils import check_array
from sklearn.metrics import hamming_loss
from sklearn.utils.validation import check_is_fitted
from sklearn.neighbors import KDTree, BallTree
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform
from hdbscan._hdbscan_reachability import mutual_reachability

from .lenses import available_lenses
from ._impl import (
    compute_minimal_presentation,
    compute_minpres_merges,
    compute_linkage_hierarchy,
    mutual_reachability_from_pdist,
)
from .plots import (
    LinkageHierarchy,
    MinimalPresentation,
    MergeHierarchy,
    SimplifiedHierarchy,
)


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
    max_label_depth : int, default=None
        The maximum depth to extract labels from the simplified hierarchy.
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
    merge_hierarchy_ : :class:`~biperscan.plots.MergeHierarchy`
        The merge hierarchy graph.
    simplified_hierarchy_ : :class:`~biperscan.plots.SimplifiedHierarchy`
        The simplified hierarchy graph.
    linkage_hierarchy_ : :class:`~biperscan.plots.LinkageHierarchy`
        The linkage hierarchy graph. This property is computed on demand and not
        cached. 
    labels_ : array of shape (n_samples,)
        The computed cluster labels.
    membership_ : array of shape (n_samples, n_clusters)
        A binary matrix indicating which points are members of which clusters.
        Columns are ordered by the merge hierarchy, so that the first non-zero
        column can be used to extract a labelling. Cluster membership overlaps.
    """
    def __init__(
        self,
        min_samples: int | None = None,
        min_cluster_size: int = 10,
        distance_fraction: float = 1.0,
        max_label_depth: int | None = None,
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
        self.max_label_depth = max_label_depth
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
            self._merge_hierarchy,
            self._simplified_hierarchy,
            self.labels_,
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
    def merge_hierarchy_(self):
        check_is_fitted(self, "_minimal_presentation")
        return MergeHierarchy(
            self.distances_,
            self.lens_values_,
            self._col_to_edge,
            self._row_to_point,
            self._minimal_presentation,
            self._merge_hierarchy,
        )

    @property
    def simplified_hierarchy_(self):
        check_is_fitted(self, "_minimal_presentation")
        return SimplifiedHierarchy(
            self.distances_,
            self.lens_values_,
            self._col_to_edge,
            self._row_to_point,
            self._minimal_presentation,
            self._merge_hierarchy,
            self._simplified_hierarchy,
        )

    @property
    def linkage_hierarchy_(self):
        check_is_fitted(self, "_minimal_presentation")
        self._linkage_hierarchy = compute_linkage_hierarchy(
            self._minimal_presentation, self._row_to_point
        )
        return LinkageHierarchy(
            self.distances_,
            self.lens_values_,
            self.lens_grades_,
            self._col_to_edge,
            self._row_to_point,
            self._linkage_hierarchy,
        )

    @property
    def membership_(self):
        check_is_fitted(self, "_minimal_presentation")
        membership = np.zeros(
            (len(self.lens_values_), len(self._simplified_hierarchy) * 2)
        )
        i = 0
        visited = set()
        for root in sorted(
            [
                x
                for x in self._simplified_hierarchy
                if self._simplified_hierarchy.in_degree(x) == 0
            ],
            reverse=True,
        ):
            for node in nx.dfs_postorder_nodes(self._simplified_hierarchy, source=root):
                if node in visited:
                    continue
                visited.add(node)
                for side in [
                    self._simplified_hierarchy.nodes[node]["parent_side"],
                    self._simplified_hierarchy.nodes[node]["child_side"],
                ]:
                    pts_one = [self._row_to_point[pt] for pt in side]
                    membership[pts_one, i] = 1
                    i += 1

        return membership

    def labels_at_depth(self, depth: int | None = None):
        """Recomputes labels from the simplified hierarchy.
        
        Parameters
        ----------
        depth : int or None, default=None
            The maximum depth to extract labels from the simplified hierarchy.
        
        Returns
        -------
        labels : array of shape (n_samples,)
            The computed cluster labels.
        """
        check_is_fitted(self, "_minimal_presentation")
        return _extract_labels(
            self._simplified_hierarchy,
            self._row_to_point,
            self.min_cluster_size,
            len(self.lens_values_),
            depth_limit=depth,
        )

    def first_nonzero_membership(self):
        """Return the first non-zero members column index."""
        check_is_fitted(self, "_minimal_presentation")

        def first_nonzero_index(array):
            fnzi = -1  # first non-zero index
            indices = np.flatnonzero(array)
            if len(indices) > 0:
                fnzi = indices[0]
            return fnzi

        return np.apply_along_axis(first_nonzero_index, 1, self.membership_)


def bpscan(
    X,
    *,
    min_samples: int | None = None,
    min_cluster_size: int = 10,
    distance_fraction: float = 1.0,
    max_label_depth: int | None = None,
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
        array of shape (1, n_samples * (n_samples - 1) // 2)
        A feature array, or condensed distance array if metric='precomputed'.
    min_samples : int, default=None
        The number of samples in a neighborhood for a point to be considered as
        a core point. If None, defaults to min_cluster_size.
    min_cluster_size : int, default=10
        The minimum number of samples to be a cluster.
    distance_fraction : float, default=1.0
        The fraction of the maximum distance grade to use a upper distance limit
        when extracting merges.
    max_label_depth : int, default=None
        The maximum depth to extract labels from the simplified hierarchy.
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
        following keys: 'start_column', 'end_column', 'root_one', 'root_two',
        'side_one', 'side_two', 'lens_grade', 'distance_grade'. Merges are
        ordered with increasing end column. Rows with the same 'start_column'
        and 'end_column' pairs indicate merges that originate from the same
        edges. Take the data point union over 'side_one' and 'side_two' for all
        rows with the same 'start_column' and 'end_column' up to the row being
        processed to find all points included in the row being processed!
    merge_hierarchy : networkx.DiGraph
        The merge hierarchy graph.
    simplified_hierarchy : networkx.DiGraph
        The simplified hierarchy graph.
    labels : array of shape (n_samples,)
        The computed cluster labels.
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
        reachability = memory.cache(_as_float32)(
            compute_reachability, X, metric, min_points=min_samples, **metric_kws
        )

    # Compute the lens values
    if type(lens) is np.ndarray:
        # implementation expects 32bit floats!
        point_lens = check_array(lens, ensure_2d=False).astype(np.float32)
    else:
        if isinstance(lens, str) and lens in available_lenses:
            lens_fun = available_lenses[lens]
        elif isinstance(lens, Callable):
            lens_fun = lens
        point_lens = memory.cache(lens_fun)(
            X if metric != "precomputed" else None,
            reachability,
            metric=metric,
            **metric_kws,
            **lens_kws,
        )

    # Compute bi-filtration minimal presentation
    (col_to_edge, row_to_point, point_lens_grades, minimum_presentation) = memory.cache(
        compute_minimal_presentation
    )(reachability, point_lens)

    # Extract merge hierarchy from the minimal presentation
    simplified_hierarchy, merge_hierarchy, merges = memory.cache(
        _compute_merge_hierarchy
    )(minimum_presentation, len(point_lens), min_cluster_size, distance_fraction)

    # Compute segmentation
    labels = memory.cache(_extract_labels)(
        simplified_hierarchy,
        row_to_point,
        min_cluster_size,
        len(point_lens),
        depth_limit=max_label_depth,
    )

    return (
        reachability,
        point_lens,
        point_lens_grades,
        col_to_edge,
        row_to_point,
        minimum_presentation,
        merges,
        merge_hierarchy,
        simplified_hierarchy,
        labels,
    )


def _kdtree_mutual_reachability(X, metric, min_points=5, **kwargs):
    tree = KDTree(X, metric=metric, **kwargs)
    min_points = min(X.shape[0] - 1, min_points)
    core_distances = tree.query(X, k=min_points)[0][:, -1]
    del tree
    gc.collect()

    dists = pdist(X, metric=metric, **kwargs)
    dists = mutual_reachability_from_pdist(core_distances, dists, X.shape[0])
    return dists


def _balltree_mutual_reachability(X, metric, min_points=5, **kwargs):
    tree = BallTree(X, metric=metric, **kwargs)
    min_points = min(X.shape[0] - 1, min_points)
    core_distances = tree.query(X, k=min_points)[0][:, -1]
    del tree
    gc.collect()

    dists = pdist(X, metric=metric, **kwargs)
    dists = mutual_reachability_from_pdist(core_distances, dists, X.shape[0])
    return dists


def _slow_mutual_reachability(X, metric, min_points=5, **kwargs):
    """Computes condensed mutual reachability matrix for the given data."""
    dists = pdist(X, metric=metric, **kwargs) if metric != "precomputed" else X
    return squareform(
        mutual_reachability(squareform(dists), min_points=min_points), checks=False
    )


def _as_float32(fun, *args, **kwargs):
    """Wraps a function to cast its output to float32."""
    return fun(*args, **kwargs).astype(np.float32)


def _compute_merge_hierarchy(
    minimal_presentation, num_points, min_cluster_size, distance_fraction
):
    """Extracts merges from the minimal presentation and creates a
    hierarchy connecting merges if they evolve into each other."""
    merges = compute_minpres_merges(
        minimal_presentation, num_points, min_cluster_size, distance_fraction
    )
    hierarchy = _merges_to_graph(merges, num_points)
    simplified_hierarchy = _simplify_merge_hierarchy(hierarchy)
    return simplified_hierarchy, hierarchy, merges


def _merges_to_graph(merges, num_points):
    """Converts a merge dictionary to a directed graph."""
    # allocate hierarchy data structures
    edge_to_node = dict()
    hierarchy = nx.DiGraph()
    fore_graph = defaultdict(set)
    back_graph = defaultdict(set)

    # find boundaries for merges originating from the same column
    column_groups = [0] + (
        np.nonzero(np.diff(merges["start_column"]) + np.diff(merges["end_column"]))[0]
        + 1
    ).tolist()

    # add merges as nodes, iterates over merges backwards to keep only the
    # highest lens + distance merge between the same points
    cnt = 0
    end = len(merges["start_column"]) - 1
    for start in column_groups[::-1]:
        for idx in range(end, start - 1, -1):
            parent, child, parent_side, child_side = _points_of_merge(
                merges, start, idx
            )
            if (parent, child) in edge_to_node:
                continue

            edge_to_node[(parent, child)] = cnt
            fore_graph[parent].add(child)
            back_graph[child].add(parent)
            hierarchy.add_node(
                cnt,
                lens_grade=merges["lens_grade"][idx],
                distance_grade=merges["distance_grade"][idx],
                parent=parent,
                child=child,
                parent_side=set.union(*[set(pts) for pts in parent_side]),
                child_side=set.union(*[set(pts) for pts in child_side]),
            )
            cnt += 1
        end = start - 1

    # add same parent and same child edges
    for graph, relation in zip([fore_graph, back_graph], ["parent", "child"]):
        prefix = "parent" if relation == "child" else "child"
        for point, neighbors in graph.items():
            for neighbor_one, neighbor_two in combinations(neighbors, 2):
                if neighbor_two < neighbor_one:
                    neighbor_one, neighbor_two = neighbor_two, neighbor_one
                if relation == "parent":
                    node_lower = edge_to_node[(point, neighbor_one)]
                    node_higher = edge_to_node[(point, neighbor_two)]
                else:
                    node_lower = edge_to_node[(neighbor_one, point)]
                    node_higher = edge_to_node[(neighbor_two, point)]
                if neighbor_two in hierarchy.nodes[node_lower][f"{prefix}_side"]:
                    edge_type = f"same_{relation}"
                else:
                    edge_type = f"into_{relation}"
                hierarchy.add_edge(node_higher, node_lower, type=edge_type)

    # add parent as child edges
    for parent, descendants in fore_graph.items():
        for grand_parent in back_graph[parent]:
            parent_node = edge_to_node[(grand_parent, parent)]
            for child in descendants:
                child_node = edge_to_node[(parent, child)]
                hierarchy.add_edge(child_node, parent_node, type="into_child")

    # connect the remaining components through most similar merges.
    # (implemented as naive MST construction, could be sped up)
    num_nodes = hierarchy.number_of_nodes()
    vectors = np.zeros((num_nodes, num_points), dtype=np.uint32)
    for parent, descendants in fore_graph.items():
        for child in descendants:
            node = edge_to_node[(parent, child)]
            vectors[node, list(hierarchy.nodes[node]["parent_side"])] = 1
            vectors[node, list(hierarchy.nodes[node]["child_side"])] = 2

    while True:
        components = list(
            nx.connected_components(hierarchy.to_undirected(as_view=True))
        )
        if len(components) == 1:
            break

        for component_ids in components:
            other_ids = set(range(num_nodes)) - component_ids
            in_rows = list(component_ids)
            out_rows = list(other_ids)
            closest_column, distances = pairwise_distances_argmin_min(
                vectors[in_rows], vectors[out_rows], metric=hamming_loss
            )
            row = distances.argmin()
            col = closest_column[row]
            if in_rows[row] in hierarchy[out_rows[col]]:
                continue
            hierarchy.add_edge(in_rows[row], out_rows[col], type="duplicate")

    return hierarchy


def _points_of_merge(merges, start, idx):
    """
    Returns the points on both sides of the merge at index `idx`. Parent and
    child sides are ordered by root point. Points are stored incrementally,
    `start` indicates the first index of the column from which merge `idx`
    originates.
    """
    parent = merges["root_one"][idx]
    child = merges["root_two"][idx]
    parent_side = merges["side_one"][start : idx + 1]
    child_side = merges["side_two"][start : idx + 1]
    if child < parent:
        parent, child = child, parent
        parent_side, child_side = child_side, parent_side
    return parent, child, parent_side, child_side


def _simplify_merge_hierarchy(hierarchy):
    # extract connected components
    components = list(
        nx.connected_components(
            nx.subgraph_view(
                hierarchy,
                filter_edge=lambda u, v: hierarchy[u][v]["type"]
                not in ["into_child", "into_parent"],
            ).to_undirected()
        )
    )
    comp_labels = np.empty(len(hierarchy), dtype=np.uint32)
    for i, component in enumerate(components):
        comp_labels[list(component)] = i

    # aggregate components into nodes
    simplified_hierarchy = nx.DiGraph()
    for i, component in enumerate(components):
        simplified_hierarchy.add_node(
            i,
            merges=component,
            root=min(hierarchy.nodes[pt]["parent"] for pt in component),
            parent_side=set.union(
                *[hierarchy.nodes[pt]["parent_side"] for pt in component]
            ),
            child_side=set.union(
                *[hierarchy.nodes[pt]["child_side"] for pt in component]
            ),
        )

    # aggregate hierarchy edges between components
    for u, v in hierarchy.edges():
        edge_type = hierarchy[u][v]["type"]
        if edge_type not in ["into_child", "into_parent"]:
            continue

        # skip edges between the same component
        comp_u = comp_labels[u]
        comp_v = comp_labels[v]
        if comp_u == comp_v:
            continue

        # take lowest root as parent (highest lens as tiebreaker)
        root_u = simplified_hierarchy.nodes[comp_u]["root"]
        root_v = simplified_hierarchy.nodes[comp_v]["root"]
        if root_u == root_v:
            lens_u = hierarchy.nodes[u]["lens_grade"]
            lens_v = hierarchy.nodes[v]["lens_grade"]
            if lens_u < lens_v:
                comp_u, comp_v = comp_v, comp_u
        elif root_u > root_v:
            comp_u, comp_v = comp_v, comp_u

        # add the edge
        if comp_u in simplified_hierarchy[comp_v]:
            continue
        if (
            comp_v in simplified_hierarchy[comp_u]
            and simplified_hierarchy[comp_u][comp_v]["type"] != edge_type
        ):
            edge_type = "both"
        simplified_hierarchy.add_edge(comp_u, comp_v, type=edge_type)

    return simplified_hierarchy


def _extract_labels(
    simplified_hierarchy, row_to_point, min_cluster_size, num_points, depth_limit=None
):
    if depth_limit is None:
        depth_limit = len(simplified_hierarchy)

    visited = set()

    def traverse_down(g, node, assigned_points, depth):
        # extract the sides
        parent_side = g.nodes[node]["parent_side"]
        child_side = g.nodes[node]["child_side"]

        # process the children, let them update the current child & parent sides!
        groups = []
        last_child = None
        last_parent = None
        if depth < depth_limit:
            for child in g[node]:
                if child in visited:
                    continue
                visited.add(child)
                child_groups, assigned_points, prev_group, _ = traverse_down(
                    g, child, assigned_points, depth + 1
                )
                groups.extend(child_groups)
                if g[node][child]["type"] == "into_child":
                    last_child = prev_group
                else:
                    last_parent = prev_group

        # create new groups from the remaining points
        new_group = child_side - assigned_points
        if last_child is not None and len(new_group) < min_cluster_size:
            last_child.update(new_group)
        else:
            groups.append(new_group)
            last_child = new_group
        assigned_points.update(new_group)

        new_group = parent_side - assigned_points
        if last_parent is not None and len(new_group) < min_cluster_size:
            last_parent.update(new_group)
        else:
            groups.append(new_group)
            last_parent = new_group

        assigned_points.update(new_group)

        # return detected groups and remaining points
        return groups, assigned_points, new_group, last_child

    # actual traversal
    roots = sorted(
        [x for x in simplified_hierarchy if simplified_hierarchy.in_degree(x) == 0],
        reverse=True,
    )
    for root in roots:
        groups, assigned_points, last_parent, last_child = traverse_down(
            simplified_hierarchy, root, set(), 0
        )

        remaining_group = (
            simplified_hierarchy.nodes[root]["child_side"] - assigned_points
        )
        if len(remaining_group) > min_cluster_size:
            groups.append(remaining_group)
        else:
            last_child.update(remaining_group)

        remaining_group = (
            simplified_hierarchy.nodes[root]["parent_side"] - assigned_points
        )
        if len(remaining_group) > min_cluster_size:
            groups.append(remaining_group)
        else:
            last_parent.update(remaining_group)

    labels = np.full(num_points, -1, dtype=np.int32)
    for i, group in enumerate(groups):
        labels[row_to_point[list(group)]] = i
    return labels
