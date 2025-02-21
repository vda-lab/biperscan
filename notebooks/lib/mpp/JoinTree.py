import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from dataclasses import dataclass, field
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from .util import minmax


@dataclass
class JoinTree:
    """
    A class representing a Join Tree for signals on graphs.

    Attributes
    ----------
        segments: list[joinTree.Segment]
            The segments of the tree.
        arcs: list[JoinTree.Arc]
            The arcs of the tree.
        root: JoinTree.Segment
            A reference to the root of the tree. The root is also in
            `tree.segments`.
    """

    @dataclass
    class Segment:
        """
        A class representing a birth-death pair in a merge tree.

        Attributes
        ----------
            id: int
                The unique id (and index) of this segment.
            birth_value: float
                The lens value at which this segment begins.
            death_value: float
                The lens value at which this segment dies. Consistent with
                functional persistence, the root segment dies at the maximum
                lens value. (This is different from 0-dimensional persistence,
                where the root keeps existing to infinity). The value -1 is used
                during construction but is not a valid value.
            point_ids: list[int]
                The vertex ids of all vertices (or data points) within the
                segment at the moment of death. So vertex ids of child segments
                are included! Use `tree.unique_points_of(segment_id)` to get the
                vertex ids unique to a particular segment.
            child_ids: list[int]
                The ids of segments that merge into this segment. This
                information is also encoded in `tree.arcs`.
        """

        id: int = 0
        birth_value: float = 0
        death_value: float = -1
        point_ids: list[int] = field(default_factory=list)
        child_ids: list[int] = field(default_factory=list)

        @property
        def persistence(self):
            """
            Returns the persistence of the segment: `death - birth`
            """
            if self.death_value == -1:
                return np.inf
            return self.death_value - self.birth_value

        def is_important(self, threshold: float):
            """
            Returns whether the segment's persistence is higher or equal to
            given threshold.
            """
            return self.persistence >= threshold

    @dataclass
    class Arc:
        """
        A class representing edges in a join tree.

        Attributes
        ----------
        elder_id: int
            The index (and id) of the elder segment in tree.segments.
        child_id: int
            The index (and id) of the child segment in tree.segments.
        """

        elder_id: int = 0
        child_id: int = 0
        lens_value: float = 0

        def __iter__(self):
            """
            Yields the elder_id and child_id in sequence. Allows easy tuple
            construction from Arcs: `tuple(arc_object)`
            """
            yield self.elder_id
            yield self.child_id

    segments: list[Segment] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    root: Segment = None

    def __init__(
        self,
        network: coo_matrix = None,
        lens: list[float] = None,
        num_values: int = None,
    ):
        """
        Constructs the JoinTree from a graph and a lens (signal with a value for
        each vertex).

        Params
        ------
        network: coo_matrix
            A sparse adjacency matrix specifying the edges in the network.
        lens: list[float]
            A list with lens values for each vertex.
        num_values: int
            The number of lens_threshold to evaluate. If not specified *all*
            unique values in the lens are evaluated. Reducing the number of
            thresholds increases performance at the cost of accuracy.
        """
        self.segments = []
        self.arcs = []

        # Input checks
        if network is None:
            return
        if lens is None:
            raise Exception("`lens` must be specified when a network is given")

        # Construct using the builder class
        self._construct_join_tree(network, np.asarray(lens), num_values)

    @property
    def maximum_persistence(self):
        """
        Returns the maximum persistence of a segment in the tree.
        """
        return self.root.persistence

    @property
    def traversal_order(self):
        children = [self.segments[c] for c in self.root.child_ids]
        order = [self.root.id]
        nodes = children[::-1]
        while len(nodes) > 0:
            node = nodes.pop()
            order.append(node.id)

            for child in [self.segments[c] for c in node.child_ids]:
                nodes.append(child)
        return order

    def unique_points_of(self, segment_id):
        """
        Returns the data-points that are in the given segment but not in any of
        its children.
        """
        point_ids = set(self.segments[segment_id].point_ids)
        for child_id in self.segments[segment_id].child_ids:
            point_ids = point_ids - set(self.segments[child_id].point_ids)
        return list(point_ids)

    def simplify(self, persistence_factor: float = 0.05):
        """
        Applies a topological simplification to the tree: all segments with a
        persistence lower than `persistence_factor * max_persistence` are
        removed.
        """

        def _copy_node(node: JoinTree.Segment, idx: int) -> JoinTree.Segment:
            """
            Copies a node without the child_id list. Sets the new node's id to
            the given value.
            """
            return JoinTree.Segment(
                id=idx,
                birth_value=node.birth_value,
                death_value=node.death_value,
                point_ids=node.point_ids,
            )

        def _process_noise(noiseTree: JoinTree, node: JoinTree.Segment, parent_id: int):
            subtree_nodes = [node]
            while len(subtree_nodes) > 0:
                subtree_node = subtree_nodes.pop(0)
                idx = len(noiseTree.segments)
                noiseTree.segments.append(_copy_node(subtree_node, idx))
                noiseTree.arcs.append(
                    JoinTree.Arc(
                        elder_id=parent_id,
                        child_id=idx,
                        lens_value=subtree_node.death_value,
                    )
                )
                for child_id in subtree_node.child_ids:
                    subtree_nodes.append(self.segments[child_id])

        def _process_signal(
            signalTree: JoinTree, node: JoinTree.Segment, parent_id: int
        ) -> int:
            # Add current node to result, maintain arcs with parent
            idx = len(signalTree.segments)
            signalTree.segments.append(_copy_node(node, idx))
            if parent_id is not None:
                signalTree.segments[parent_id].child_ids.append(idx)
                signalTree.arcs.append(
                    JoinTree.Arc(
                        elder_id=parent_id, child_id=idx, lens_value=node.death_value
                    )
                )
            else:
                signalTree.root = signalTree.segments[idx]
            return idx

        signalTree = JoinTree()
        noiseTree = JoinTree()

        threshold = self.maximum_persistence * persistence_factor
        nodes = [(None, self.root)]

        # Traverse the tree
        while len(nodes) > 0:
            parent_id, node = nodes.pop(0)
            if node.persistence < threshold:
                # Add entire subtree to noiseTree.
                _process_noise(noiseTree, node, parent_id)
            else:
                # Add segment and arc with parent to signalTree.
                idx = _process_signal(signalTree, node, parent_id)
                # process children
                for child_id in node.child_ids:
                    nodes.append((idx, self.segments[child_id]))

        return signalTree, noiseTree

    def segment_distances(self):
        rows = []
        cols = []
        distance = []
        nodes = [self.root]
        while len(nodes) > 0:
            segment = nodes.pop()

            # Add edges between children
            child_deaths = [
                self.segments[child_id].death_value for child_id in segment.child_ids
            ]
            order = np.argsort(np.asarray(child_deaths))
            for idx in range(len(segment.child_ids) - 1):
                sibling1 = self.segments[segment.child_ids[order[idx + 1]]]
                sibling2 = self.segments[segment.child_ids[order[idx]]]
                rows.append(sibling1.id)
                cols.append(sibling2.id)
                distance.append(sibling1.death_value - sibling2.death_value)

            # Add edges to children
            for child_id in segment.child_ids:
                child = self.segments[child_id]
                rows.append(segment.id)
                cols.append(child.id)
                distance.append(segment.death_value - child.death_value)
                nodes.append(child)

        # Compute all shortest paths
        #  can't do a shortcut by breadthfirst traversal
        n_nodes = len(self.segments)
        sparse_network = coo_matrix((distance, (rows, cols)), shape=(n_nodes, n_nodes))
        return shortest_path(sparse_network.tocsr(), directed=False)

    def as_coo_matrix(self):
        """
        Converts the JoinTree to an unweighted coo_matrix.
        """
        n_points = len(self.segments)
        (
            rows,
            cols,
        ) = map(np.asarray, zip(*self.arcs))
        data = np.ones(len(rows), dtype="bool")
        return coo_matrix((data, (rows, cols)), shape=(n_points, n_points))

    def as_nx(self):
        """
        Converts the JoinTree to an unweighted networkx Graph.
        """
        g = nx.Graph()
        for n in self.segments:
            g.add_node(n.id)
        for arc in self.arcs:
            g.add_edge(arc.elder_id, arc.child_id)
        return g

    def _construct_join_tree(
        self, network: coo_matrix, lens: np.ndarray, num_values: int = None
    ):
        """
        A utility function that constructs a JoinTree from a network and lens.

        The function constructs a table with rows for each data-point in the
        network. Then at every to-be-evaluated level of the lens, it filters out
        the data-points with a value higher than the threshold and detects the
        connected component in that filtered network. The table is filled with
        the (arbitrary) id of the connected component that each point belongs to.
        The value -1 is used when a point is not in the filtered network.

        The final JoinTree is build up throughout this process.
        """

        @dataclass
        class State:
            """
            A utility class to pass each iteration's state to helper functions
            in one variable.
            """

            tree: JoinTree = None
            threshold: float = 0
            segment_counter: int = 0
            segment_of_point: np.ndarray = None

        def _determine_thresholds(
            lens: np.ndarray, num_values: int = None
        ) -> np.ndarray:
            # Determine thresholds
            if num_values is None:
                lens_thresholds = np.unique(lens)
            else:
                (minimum, maximum) = minmax(lens)
                lens_thresholds = np.linspace(minimum, maximum, num_values)
            return lens_thresholds

        def _evaluate_threshold(state: State, network: coo_matrix, threshold: float):
            # Apply the threshold
            state.threshold = threshold
            point_mask = lens <= threshold
            point_ids = np.where(point_mask)[0]
            masked_network = (
                network.copy().tocsc()[:, point_mask][point_mask, :].tocoo()
            )
            # Check for edges
            if len(masked_network.data) == 0:
                _update_tree_without_edges(state, point_ids)
            else:
                _update_join_tree_with_edges(state, masked_network, point_ids)

        def _update_tree_without_edges(state: State, point_ids: np.ndarray):
            # give every point it's own segment
            for point_id in point_ids:
                previous_id = state.segment_of_point[point_id, 0]
                # segment exists?
                if previous_id > -1:
                    _continue_existing_segment(state, point_id, previous_id)
                else:
                    _add_new_segment(state, point_id)

        def _update_join_tree_with_edges(
            state: State, masked_network: coo_matrix, point_ids: np.ndarray
        ):
            # compute connected components and assign to points
            nx_network = nx.Graph(masked_network)

            for component in nx.connected_components(nx_network):
                component_point_ids = np.asarray([point_ids[idx] for idx in component])
                previous_segment_ids = np.unique(
                    state.segment_of_point[component_point_ids, 0]
                )
                previous_segment_ids = previous_segment_ids[previous_segment_ids != -1]

                # check for merges
                if len(previous_segment_ids) == 1:
                    _continue_existing_segment(
                        state, component_point_ids, previous_segment_ids[0]
                    )
                elif len(previous_segment_ids) > 1:
                    _merge_segments(state, component_point_ids, previous_segment_ids)
                else:
                    _add_new_segment(state, component_point_ids)

        def _add_new_segment(state: State, point_ids: np.ndarray):
            """
            Function that adds a new segment to the join tree.
            """
            state.tree.segments.append(
                JoinTree.Segment(id=state.segment_counter, birth_value=state.threshold)
            )
            state.segment_of_point[point_ids, 1] = state.segment_counter
            state.segment_counter = state.segment_counter + 1

        def _continue_existing_segment(
            state: State, point_ids: np.ndarray, previous_id: int
        ):
            """
            Function that updates the data-point table to continue an existing
            segment.
            """
            state.segment_of_point[point_ids, 1] = previous_id

        def _merge_segments(
            state: State, point_ids: np.ndarray, previous_ids: np.ndarray
        ):
            """
            Function that merges multiple segments. The eldest segment is
            determined, with a tie-break by minimum data-point id (row index) in
            the case of equal age.
            """
            elder_id = _determine_elder_segment(state, previous_ids)
            _continue_existing_segment(state, point_ids, elder_id)

            dying_ids = previous_ids[previous_ids != elder_id]
            for dying_id in dying_ids:
                dying_point_ids = np.where(state.segment_of_point[:, 0] == dying_id)[
                    0
                ].tolist()
                state.tree.segments[dying_id].death_value = state.threshold
                state.tree.segments[dying_id].point_ids = dying_point_ids
                state.tree.segments[elder_id].child_ids.append(dying_id)
                state.tree.arcs.append(
                    JoinTree.Arc(
                        elder_id=elder_id, child_id=dying_id, lens_value=state.threshold
                    )
                )

        def _determine_elder_segment(state: State, segment_ids: np.ndarray) -> int:
            """
            Determines which segment is the eldest from the given ids.
            """
            segment_births = np.asarray(
                [state.tree.segments[i].birth_value for i in segment_ids]
            )
            most_elder_ids = segment_ids[
                np.where(segment_births == np.min(segment_births))[0]
            ]

            # tie-break by min(row_id)
            if len(most_elder_ids) > 1:
                min_row_id = [
                    np.argmax(state.segment_of_point[:, 0] == i) for i in most_elder_ids
                ]
                elder_id = most_elder_ids[np.argmin(min_row_id)]
            else:
                elder_id = most_elder_ids[0]
            return elder_id

        def _finalize_root(state, network: coo_matrix):
            for s in state.tree.segments:
                if s.death_value > -1:
                    continue
                dying_point_ids = np.where(state.segment_of_point[:, 0] == s.id)[
                    0
                ].tolist()
                state.tree.segments[s.id].point_ids = dying_point_ids

        # Construct state
        state = State(
            tree=self, segment_of_point=np.full((network.shape[0], 2), -1, dtype="int")
        )

        # Evaluate thresholds
        lens_thresholds = _determine_thresholds(lens, num_values)
        for threshold in lens_thresholds:
            _evaluate_threshold(state, network, threshold)

            state.segment_of_point[:, 0] = state.segment_of_point[:, 1]
            state.segment_of_point[:, 1] = -1

        # Fill in the root segment
        _finalize_root(state, network)

        # assert np.asarray([segment.death_value == -1 for segment in self.segments]).sum() == 1, 'The tree did not converge to 1 component!'
        if len(self.arcs) > 0:
            self.root = self.segments[self.arcs[-1].elder_id]
        else:
            self.root = self.segments[0]
        self.root.death_value = threshold
        self.root.point_ids = np.arange(network.shape[0]).tolist()


def join_tree_layout(
    tree: JoinTree,
    noise_tree: JoinTree = None,
    width=1,
    x_offset=0,
    non_important_gap=0.15,
):
    """
    Computes a planar layout for the given JoinTree.

    Params
    ------
    tree: JoinTree
        The main JoinTree to visualize
    noise_tree: JoinTree
        An optional JoinTree specifying the noise removed by tree.simplify().
    width: float
        The width to locate the tree in.
    x_offset: float
        An offset to apply to the x coordinate.
    non_important_gap: float
        The x-offset of noise segments from their signal parent, as a ratio
        of segment x-separation.

    Returns
    -------
    pos: 2D numpy array
        The x, y coordinates of the segments' birth and death points.
    edges: [[(x1, y1), (x2, y2)], ...]
        The edges between birth and death points, to be used with
        LineCollections.
    connections: [[(x1, y1), (x2, y2)], ...]
        The edges between the death points and parent segments, to be used
        with LineCollections.
    index: 1D numpy array
        The order in which segments are present in `pos`.
    noise_pos:
        same as `pos` but for the noise_tree.
    noise_edges:
        same as `edges` but for the noise_tree.
    noise_connections:
        same as `connections` but for the noise_tree.
    """
    # Compute layout for signal tree
    pos = []
    index = {}
    dummy = []
    connections = []
    nodes = [(None, tree.root)]
    offset_counter = 0
    while len(nodes) > 0:
        parent_idx, node = nodes.pop()
        pos.append([offset_counter, node.birth_value])
        pos.append([offset_counter, node.death_value])

        if parent_idx is not None:
            dummy_idx = len(dummy)
            dummy.append([pos[parent_idx][0], node.death_value])
            connections.append((dummy_idx, len(pos) - 1))

        offset_counter = offset_counter + 1

        parent_idx = len(pos) - 1
        index[node.id] = parent_idx
        for child_id in node.child_ids[::-1]:
            nodes.append((parent_idx, tree.segments[child_id]))

    # Compute layout for noise tree
    noise_pos = []
    noise_dummy = []
    noise_connections = []
    if noise_tree is not None:
        for arc in noise_tree.arcs:
            elder_id = arc.elder_id
            child_id = arc.child_id

            elder_x = pos[index[elder_id]][0]
            noise_pos.append(
                [elder_x - non_important_gap, noise_tree.segments[child_id].birth_value]
            )
            noise_pos.append(
                [elder_x - non_important_gap, noise_tree.segments[child_id].death_value]
            )
            noise_dummy_idx = len(noise_dummy)
            noise_dummy.append([elder_x, noise_tree.segments[child_id].death_value])
            noise_connections.append((noise_dummy_idx, len(noise_pos) - 1))

    # rescale to specified size
    pos = np.asarray(pos)
    dummy = np.asarray(dummy)
    noise_pos = np.asarray(noise_pos)
    noise_dummy = np.asarray(noise_dummy)

    i_min, x_max = minmax(pos[:, 0])
    if len(noise_pos) > 0:
        u_min = noise_pos[:, 0].min()
        x_min = min(i_min, u_min)
    else:
        x_min = i_min

    if x_min != x_max:
        pos[:, 0] = (pos[:, 0] - x_min) / (x_max - x_min) * width - width / 2 + x_offset
        if len(dummy) > 0:
            dummy[:, 0] = (
                (dummy[:, 0] - x_min) / (x_max - x_min) * width - width / 2 + x_offset
            )
        if len(noise_pos) > 0:
            noise_pos[:, 0] = (
                (noise_pos[:, 0] - x_min) / (x_max - x_min) * width
                - width / 2
                + x_offset
            )
            noise_dummy[:, 0] = (
                (noise_dummy[:, 0] - x_min) / (x_max - x_min) * width
                - width / 2
                + x_offset
            )
    else:
        pos[:, 0] = x_offset
        if len(dummy) > 0:
            dummy[:, 0] = x_offset
        if len(noise_pos) > 0:
            noise_pos[:, 0] = x_offset
            noise_dummy[:, 0] = x_offset

    # extract edges
    edges = [
        [tuple(pos[i, :]), tuple(pos[i + 1, :])] for i in range(0, pos.shape[0], 2)
    ]
    noise_edges = [
        [tuple(noise_pos[i, :]), tuple(noise_pos[i + 1, :])]
        for i in range(0, noise_pos.shape[0], 2)
    ]

    # extract connections
    connections = [[tuple(dummy[s[0], :]), tuple(pos[s[1], :])] for s in connections]
    noise_connections = [
        [tuple(noise_dummy[s[0], :]), tuple(noise_pos[s[1], :])]
        for s in noise_connections
    ]

    return pos, edges, connections, index, noise_pos, noise_edges, noise_connections


def hierarchical_join_tree_layout(
    tree: JoinTree,
    width=1,
    x_offset=0,
    leaf_vs_root_factor=0.5,
    vert_gap=0.2,
    vert_loc=0,
):
    """
    Constructs a hierarchical tree layout for the tree.

    Params
    ------
    tree: JoinTree
        The join tree to layout.
    width: float
        The width to locate the tree in.
    x_offset: float
        An offset to apply to the x coordinate.
    leaf_vs_root_factor: float
        A parameter influencing the layout of leaves.
    vert_gap: float
        The vertical distance between depths layers.
    vert_loc: float
        The vertical offset.

    Returns
    -------
    pos:
        2D numpy array (N x 2)  array specifying x-y coordinates of each
        segment.
    edges:
        The coordinates of the edges to be used with a LineCollection:
            [[(1x, y1), (x2, y2)], ....]

    Adapted from:
    https://epidemicsonnetworks.readthedocs.io/en/latest/_modules/EoN/auxiliary.html#hierarchy_pos
    """
    x_center = width / 2.0

    def _is_leaf(node):
        return len(node.child_ids) == 0

    leaf_count = len([node for node in tree.segments if _is_leaf(node)])

    def _hierarchy_pos(
        root_id,
        left_most,
        width,
        leaf_dx=0.2,
        vert_gap=0.2,
        vert_loc=0,
        x_center=0.5,
        root_pos=None,
        leaf_pos=None,
    ):
        # Fill in default values if None
        if leaf_pos is None:
            leaf_pos = {}

        # Fill in root_pos dictionary with current position
        if root_pos is None:
            root_pos = {root_id: x_center}
        else:
            root_pos[root_id] = x_center

        leaf_count = 0
        child_ids = tree.segments[root_id].child_ids
        if len(child_ids) != 0:
            root_dx = width / len(child_ids)
            next_x = x_center - width / 2 - root_dx / 2
            for child_id in child_ids:
                next_x += root_dx
                root_pos, leaf_pos, new_leaves = _hierarchy_pos(
                    child_id,
                    left_most + leaf_count * leaf_dx,
                    width=root_dx,
                    leaf_dx=leaf_dx,
                    vert_gap=vert_gap,
                    vert_loc=vert_loc - vert_gap,
                    x_center=next_x,
                    root_pos=root_pos,
                    leaf_pos=leaf_pos,
                )
            leaf_count += new_leaves

            left_most = min(
                (x for x, _ in [leaf_pos[child_id] for child_id in child_ids])
            )
            right_most = max(
                (x for x, _ in [leaf_pos[child_id] for child_id in child_ids])
            )
            leaf_pos[root_id] = ((left_most + right_most) / 2, vert_loc)
        else:
            leaf_count = 1
            leaf_pos[root_id] = (left_most, vert_loc)
        return root_pos, leaf_pos, leaf_count

    root_pos, leaf_pos, leaf_count = _hierarchy_pos(
        tree.root.id,
        0,
        width,
        leaf_dx=width * 1.0 / leaf_count,
        vert_gap=vert_gap,
        vert_loc=vert_loc,
        x_center=x_center,
    )

    # preallocate the result
    pos = np.zeros((len(tree.segments), 2))

    # fill preliminary values
    for s in tree.segments:
        pos[s.id, 0] = (
            leaf_vs_root_factor * leaf_pos[s.id][0]
            + (1 - leaf_vs_root_factor) * root_pos[s.id]
        )
        pos[s.id, 1] = leaf_pos[s.id][1]

    # normalize the x-coordinate
    x_min, x_max = minmax(pos[:, 0])
    if x_min == x_max:
        pos[:, 0] = x_offset
    else:
        pos[:, 0] = (pos[:, 0] - x_min) / (x_max - x_min) * width - width / 2 + x_offset

    # fill the edges
    edges = []
    for node_id in range(len(tree.segments)):
        for child_id in tree.segments[node_id].child_ids:
            edges.append([tuple(pos[node_id, :]), tuple(pos[child_id, :])])
    return pos, edges


def plot_join_tree(
    tree: JoinTree,
    noiseTree: JoinTree = None,
    width=1,
    x_offset=0,
    non_important_gap=0.15,
    ax=None,
    clim=None,
    cmap=None,
    text_offset=None,
    labels=None,
):
    """
    Plots the JoinTree using a planar layout.

    Params
    ------
    tree: JoinTree
        The main JoinTree to visualize
    noise_tree: JoinTree
        An optional JoinTree specifying the noise removed by tree.simplify().
    width: float
        The width to locate the tree in.
    x_offset: float
        An offset to apply to the x coordinate.
    non_important_gap: float
        The x-offset of noise segments from their signal parent, as a ratio
        of segment x-separation.
    ax: matplotlib axis
        The axis to plot to, if not given the current axis is used.
    clim: list
        The limits of the color dimension to use. By default it uses,
        the minimum and maximum value of the color dimension (persistence).
    cmap: matplotlib colormap
        The colormap to apply.
    text_offset: tuple
        A tuple specifying the x-y offsets to apply to the text annotation.
    labels: list
        A list of labels to use for the segments.
    """
    layout = join_tree_layout(
        tree,
        noiseTree,
        width=width,
        x_offset=x_offset,
        non_important_gap=non_important_gap,
    )
    (i_pos, i_edges, i_connections, i_index, u_pos, u_edges, u_connections) = layout

    i_persistence = np.repeat(i_pos[1::2, 1], 2) - np.repeat(i_pos[0::2, 1], 2)
    if clim is None:
        clim = [tree.root.birth_value, tree.root.death_value]

    # Get the current figure and axis, or create a new one
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    fig = plt.gcf()

    # Plot edges
    lc = LineCollection(i_edges, array=i_persistence[0::2], linewidths=1, cmap=cmap)
    lc.set_clim(clim)
    ax.add_collection(lc)
    lc = LineCollection(u_edges, colors="#ddd", linewidths=0.55)
    ax.add_collection(lc)

    # Plot connections
    lc = LineCollection(
        i_connections, array=i_persistence[2::2], linewidths=0.75, cmap=cmap
    )
    lc.set_clim(clim)
    ax.add_collection(lc)
    lc = LineCollection(u_connections, colors="#ddd", linewidths=0.55)
    ax.add_collection(lc)

    # Plot points
    plt.scatter(
        i_pos[:, 0],
        i_pos[:, 1],
        20,
        i_persistence,
        zorder=2,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        edgecolors="none",
        linewidths=0,
    )
    if len(u_pos) > 0:
        plt.scatter(
            u_pos[:, 0],
            u_pos[:, 1],
            10,
            "#ddd",
            zorder=2,
            linewidths=0,
            edgecolors="none",
        )

    # Plot node ids
    if text_offset is None:
        text_offset = (width / 25, -0.015)
    for s in tree.segments:
        idx = i_index[s.id] - 1
        if labels is None:
            label = s.id
        elif s.id in labels:
            label = labels[s.id]
        else:
            continue
        plt.annotate(
            label,
            (i_pos[idx, 0] + text_offset[0], i_pos[idx, 1] + text_offset[1]),
            va="center",
            ha="center",
            fontsize=6,
        )
    return fig, ax, layout


def plot_hierarchical_join_tree(
    tree: JoinTree,
    width=1,
    x_offset=0,
    ax=None,
    clim=None,
    cmap=None,
    text_offset=None,
    labels=None,
):
    """
    Plots the JoinTree with an hierarchical layout using matplotlib.

    Params
    ------
    width: float
        The width to locate the tree in.
    x_offset: float
        An offset to apply to the x coordinate.
    ax: matplotlib axis
        The axis to plot to, if not given the current axis is used.
    clim: list
        The limits of the color dimension to use. By default it uses,
        the minimum and maximum value of the color dimension (persistence).
    cmap: matplotlib colormap
        The colormap to apply.
    text_offset: tuple
        A tuple specifying the x-y offsets to apply to the text annotation.
    labels: list
        A list of labels, one for each segment.
    """
    layout = hierarchical_join_tree_layout(tree, width=width, x_offset=x_offset)
    (pos, edges) = layout
    # Get the current figure and axis, or create a new one
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    fig = plt.gcf()

    if clim is None:
        clim = [tree.root.birth_value, tree.root.death_value]

    # Plot the tree
    lc = LineCollection(edges, colors="#ddd", linewidths=0.75)
    ax.add_collection(lc)
    color = np.asarray([s.persistence for s in tree.segments])
    plt.scatter(
        pos[:, 0],
        pos[:, 1],
        20,
        color,
        zorder=2,
        cmap=cmap,
        vmin=clim[0],
        vmax=clim[1],
        linewidths=0,
        edgecolors="none",
    )

    if text_offset is None:
        text_offset = (width / 25, 0.005)
    for i, s in enumerate(tree.segments):
        if labels is None:
            label = s.id
        elif s.id in labels:
            label = labels[s.id]
        else:
            continue
        plt.annotate(
            label,
            (pos[i, 0] + text_offset[0], pos[i, 1] + text_offset[1]),
            fontsize=6,
            va="center",
            ha="center",
        )
    return fig, ax, layout
