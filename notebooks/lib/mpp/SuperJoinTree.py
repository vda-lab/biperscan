import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection, PatchCollection
from dataclasses import dataclass, field

from .JoinTree import plot_join_tree, plot_hierarchical_join_tree
from .JoinTreeSequence import JoinTreeSequence
from .util import minmax


@dataclass
class SuperJoinTree:
    """
    A class that represents the tree structure across levels in a
    JoinTreeSequence.

    Attributes
    ----------
    sequence: JoinTreeSequence
    values: list[float]
    segments: list[SuperJoinTree.Segment]
    root: SuperJoinTree.Segment
    """

    @dataclass
    class Segment:
        """
        A class representing a chain of JoinTree.Segments across the
        sequence.

        In the most basic form, Super Segments are a single undisrupted chain of
        Sequence Segments over the levels. In that case a segment has a single birth
        and death level. In some cases, the chain is disrupted, where the Segment
        merges with another Segment for some levels but re-emerges at higher levels.
        In that case the Super segment contains multiple birth-death pairs.


        Attributes
        ----------
        id: int
            The index / id of this segment in the SuperJoinTree.
        birth_levels: list[float]
            The birth level for each part of this super segment.
        birth_parents: list[SuperJoinTree.Segment]
            The super segments this super segment splits of from at births, None
            in the case of a leaf.
        birth_children: list[SuperJoinTree.Segment]
            The super segments that split of from this super segment at some
            level.

        death_levels: list[float]
            The death level for each part of this super segment.
        death_parents: list[SuperJoinTree.Segment]
            The super segment into which this super segment merges at deaths.
        death_children: list[SuperJoinTree.Segment]
            The children that merge into this super segment at some level.

        segment: dict[level_id] = segment_id
            The JoinTree Segments belonging to this SuperSegment at the levels
            this SuperSegment exists.
        """

        id: int = 0

        birth_levels: list[float] = field(default_factory=list)
        birth_parents: list = field(default_factory=list)
        birth_children: set = field(default_factory=set)

        death_levels: list[float] = field(default_factory=list)
        death_parents: list = field(default_factory=list)
        death_children: set = field(default_factory=set)

        segments: dict[int, JoinTreeSequence.Segment] = field(default_factory=dict)

        def __eq__(self, other):
            return self.id == other.id

        def __hash__(self):
            return hash(self.id)

        @property
        def birth(self):
            """The lowest birth level."""
            return self.birth_levels[-1]

        @property
        def death(self):
            """The highest death level."""
            return self.death_levels[0]

        @property
        def final_death_parent(self):
            return self.death_parents[0]

        @property
        def total_persistence(self):
            total = 0
            for _, segment in self.segments.items():
                total = total + segment.persistence
            return total

        def death_children_ordered(self, super_tree, by=lambda c: c.total_persistence):
            # TODO would rather pay the cost of having children in order during
            # construction than during retrieval...
            children = sorted(self.death_children, key=by, reverse=True)
            if self != super_tree.root:
                return children
            order = []
            marked = set()
            nodes = children[::-1]
            while len(nodes) > 0:
                node = nodes.pop()
                if node in marked:
                    continue
                marked.add(node)
                if node.final_death_parent == super_tree.root:
                    order.append(node)

                # Postpone processing partial children to after all full
                # children have been processed. So they have to be added
                # to the stack before the full children!
                full_children = []
                for child in node.death_children_ordered(super_tree, by=by):
                    if child.final_death_parent == node:
                        full_children.append(child)
                    else:
                        # Add the top-level node of child's subtree
                        parent = child
                        while parent.final_death_parent != super_tree.root:
                            parent = parent.final_death_parent
                        nodes.append(parent)
                for child in full_children:
                    nodes.append(child)
            return order

        def segment_at_level(self, level_id):
            """
            The Join Tree segment of this SuperTree (or its parents if this
            SuperSegment does not exist at the level) at a particular level.
            """
            # Within the life of this segment
            if level_id in self.segments:
                return self.segments[level_id].join_tree_segment
            # level_id is somewhere outside of birth-death values
            # check parts from high to low
            for i, death in enumerate(self.death_levels):
                # if above the death, then it is between this part and the
                # one above, so return the death_parent of this part
                if level_id > death:
                    return self.death_parents[i].segment_at_level(level_id)
            # Now, level_id can only be below the first birth, so return
            # the corresponding birth_parent.
            return self.birth_parents[i]

        def segment_per_level(self, n_levels):
            """
            The Join Tree Segment that belongs to this SuperSegment (or its
            parents if this segment does not exist) for each level.
            """
            return [self.segment_at_level(level_id) for level_id in range(n_levels)]

    sequence: JoinTreeSequence
    values: list[float] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    root: Segment = None

    def __init__(self, sequence: JoinTreeSequence, values: list[float]):
        """
        Constructs the SuperJoinTree from a JoinTreeSequence and the value
        of each level in the sequence.

        Params
        ------
        trees: list[JoinTree]
            A sequence of JoinTrees
        values: list[float]
            The level-values of the sequence.
        """
        self.sequence = sequence
        self.values = values
        self.segments = []
        self.arcs = []

        if sequence is not None:
            self._construct_super_tree()

    def persistence_of(self, segment_id: int):
        n_levels = len(self.values)
        total = self.segments[segment_id].total_persistence
        value_range = self.values[-1] - self.values[0]
        return (total * value_range) / (n_levels - 1)

    @property
    def traversal_order(self):
        children = sorted(
            self.root.death_children, key=lambda c: c.total_persistence, reverse=True
        )
        order = []
        marked = set()
        nodes = children[::-1]
        while len(nodes) > 0:
            node = nodes.pop()
            if node in marked:
                continue
            marked.add(node)
            order.append(node)

            # Postpone processing partial children to after all full
            # children have been processed. So they have to be added
            # to the stack before the full children!
            full_children = []
            for child in node.death_children_ordered(self, by=lambda c: c.death):
                if child.final_death_parent == node:
                    full_children.append(child)
                else:
                    # Add the top-level node of child's subtree
                    parent = child
                    while parent.final_death_parent != self.root:
                        parent = parent.final_death_parent
                    nodes.append(parent)
            for child in full_children:
                nodes.append(child)
        return order

    def simplify(self, persistence_factor=0.05):
        # TODO should properly process the JoinTree segments that are removed here
        # i.e., put them in the noise trees, put their data-points in their parents
        persistences = np.asarray(
            [self.persistence_of(i) for i in range(len(self.segments))]
        )
        max_persistence = np.max(persistences)
        persistence_threshold = persistence_factor * max_persistence

        # Create new tree to fill with only signal
        signal_super_tree = SuperJoinTree(None, None)
        signal_super_tree.sequence = self.sequence
        signal_super_tree.values = self.values
        signal_super_tree.segments.append(SuperJoinTree.Segment())

        # Add the root, use index to map ids between the trees
        index = {}
        signal_super_tree.root = signal_super_tree.segments[0]
        index[self.root.id] = 0

        # Traverse the tree
        nodes = [
            (signal_super_tree.root, s)
            for s in self.root.death_children_ordered(self)[::-1]
        ]
        counter = 1
        while len(nodes) > 0:
            parent, node = nodes.pop()

            if persistences[node.id] < persistence_threshold:
                continue
            if node.id in index:
                continue

            # Add segments without reference to their parents yet
            index[node.id] = counter
            signal_segment = SuperJoinTree.Segment(
                id=counter,
                birth_levels=node.birth_levels.copy(),
                death_levels=node.death_levels.copy(),
                segments=node.segments.copy(),
            )
            counter = counter + 1
            signal_super_tree.segments.append(signal_segment)

            # Add children of parents
            parent.death_children.add(signal_segment)

            # Queue children
            for child in node.death_children_ordered(self)[::-1]:
                nodes.append((signal_segment, child))

        # Traverse the tree again to fill in parents
        nodes = [s for s in self.root.death_children_ordered(self)[::-1]]
        masked = set()
        while len(nodes) > 0:
            node = nodes.pop()
            if node.id in masked:
                continue
            masked.add(node.id)
            if persistences[node.id] < persistence_threshold:
                continue

            signal_segment = signal_super_tree.segments[index[node.id]]

            for b, d in zip(node.birth_parents, node.death_parents):
                signal_segment.birth_parents.append(
                    signal_super_tree.segments[index[b.id]]
                )
                signal_segment.death_parents.append(
                    signal_super_tree.segments[index[d.id]]
                )

            # Queue children
            for child in node.death_children_ordered(self)[::-1]:
                nodes.append(child)
        return signal_super_tree

    def _construct_super_tree(self):
        @dataclass
        class State:
            index: dict = field(default_factory=dict)
            replacement_arcs: list[JoinTreeSequence.Arc] = field(default_factory=list)
            split_arcs: list = field(default_factory=list)
            segment_counter: int = 0

        def _add_new_segment(
            super_tree: SuperJoinTree,
            state: State,
            parent_id: int,
            segment: JoinTreeSequence.Segment,
        ):
            new_parent_id = state.segment_counter
            death_parent = super_tree.segments[parent_id]
            # Append the new segment
            super_segment = SuperJoinTree.Segment(
                id=new_parent_id,
                birth_levels=[segment.level_id],
                birth_parents=[super_tree.root],
                death_levels=[segment.level_id],
                death_parents=[death_parent],
            )
            super_segment.segments[segment.level_id] = segment
            super_tree.segments.append(super_segment)

            # Append the new arc, including in the parent's child_ids
            super_tree.segments[parent_id].death_children.add(super_segment)

            # Update counters and index
            state.segment_counter = state.segment_counter + 1
            state.index[segment.id] = new_parent_id
            return new_parent_id

        def _append_segment(
            super_tree: SuperJoinTree, parent_id: int, segment: JoinTreeSequence.Segment
        ):
            super_segment = super_tree.segments[parent_id]
            super_segment.segments[segment.level_id] = segment
            super_segment.birth_levels[0] = segment.level_id
            state.index[segment.id] = parent_id

        # Append dummy root.
        self.segments.append(SuperJoinTree.Segment())
        self.root = self.segments[0]

        # Traverse the chains
        state = State(segment_counter=1)
        n_roots = len(self.sequence.trees[-1].segments)
        roots = self.sequence.segments[:n_roots]
        for root in roots:
            parent_id = _add_new_segment(self, state, 0, root)

            arcs = [(parent_id, arc) for arc in root.in_arcs]
            while len(arcs) > 0:
                parent_id, arc = arcs.pop(0)
                child = self.sequence.segments[arc.child_id]

                if arc.type == "core":
                    if arc.replaces is not None:
                        state.replacement_arcs.append(arc)
                    _append_segment(self, parent_id, child)
                    arcs = arcs + [(parent_id, arc) for arc in child.in_arcs]
                if arc.type == "split":  # birth
                    # _append_segment(self, parent_id, child)
                    state.split_arcs.append((parent_id, arc.child_id))
                    # Can't add birth-parent yet because we may not yet know which
                    # super segment the sequence segment belongs to...
                    # No need to continue with the in-arcs of this node.
                if arc.type == "merge":  # death
                    new_parent_id = _add_new_segment(self, state, parent_id, child)
                    arcs = arcs + [(new_parent_id, arc) for arc in child.in_arcs]

        # Add the birth_parents
        for parent_id, child_id in state.split_arcs:
            self.segments[parent_id].birth_parents[0] = self.segments[
                state.index[child_id]
            ]

        # process the arcs that span more than one level.
        for arc in state.replacement_arcs:
            (arc1, arc2) = arc.replaces
            disappearing_segment = self.segments[state.index[arc.elder_id]]
            maintaining_segment = self.segments[state.index[arc1.elder_id]]
            birth_segment = self.segments[state.index[arc2.child_id]]
            # Store levels at which the segment is missing
            begin_level = self.sequence.segments[arc1.child_id].level_id
            end_level = self.sequence.segments[arc2.elder_id].level_id

            disappearing_segment.birth_levels.insert(-1, end_level)
            disappearing_segment.birth_parents.insert(-1, birth_segment)
            disappearing_segment.death_levels.append(begin_level)
            disappearing_segment.death_parents.append(maintaining_segment)
            birth_segment.birth_children.add(disappearing_segment)
            maintaining_segment.death_children.add(disappearing_segment)


def plot_super_join_tree(
    super_tree: SuperJoinTree, height=1, text_offset=None, ax=None
):
    """
    Plots a SuperJoinTree as a planar merge tree.

    Params
    ------
    tree: SuperJoinTree
        The SuperJoinTree to visualize
    width: float
        The width to locate the tree in.
    x_offset: float
        An offset to apply to the x coordinate.

    Returns
    -------
        fig, ax
    """
    # Get the current figure and axis, or create a new one
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    fig = plt.gcf()

    # Allocate variables for the coordinates
    segment_order = super_tree.traversal_order
    # Reverse the relation so that we can find the x-location of each segment
    index = {segment.id: i for i, segment in enumerate(segment_order)}

    # Create a lookup for the super tree segment ids from join tree ids
    super_id_lookup = {}
    for s in super_tree.segments[1:]:
        for level_id, sequence_segment in s.segments.items():
            super_id_lookup[(level_id, sequence_segment.join_tree_segment.id)] = s.id

    def _y(i):
        return i / (len(segment_order) - 1) * height - height / 2

    def _interpolate(ls, xs, ys, cs):
        l = np.linspace(ls[0], ls[-1], 100)
        xs = np.interp(l, ls, xs)
        ys = np.interp(l, ls, ys)
        cs = np.interp(l, ls, cs)
        coords = np.asarray([xs, ys]).T.reshape((-1, 1, 2))
        segments = np.concatenate([coords[:-1], coords[1:]], axis=1)
        return segments, cs

    # Combine all the coordinates
    point_coords = []
    point_colors = []
    point_labels = []
    edge_segments = []
    edge_colors = []
    spanning_segments = []
    death_segments = []
    birth_segments = []
    partial_death_segments = []
    lens_lines = []

    for i, segment in enumerate(segment_order):
        y_location = _y(i)
        for i, (birth, death, birth_parent, death_parent) in enumerate(
            zip(
                segment.birth_levels,
                segment.death_levels,
                segment.birth_parents,
                segment.death_parents,
            )
        ):
            birth_value = super_tree.values[birth]
            birth_persistence = segment.segment_at_level(birth).persistence
            death_value = super_tree.values[death]
            death_persistence = segment.segment_at_level(death).persistence

            # Points and label
            point_coords.append([birth_value, y_location])
            point_colors.append(birth_persistence)
            point_coords.append([death_value, y_location])
            point_colors.append(death_persistence)
            if i == 0:
                point_labels.append((segment.id, len(point_coords) - 1))

            # Edge
            ls = list(range(birth, death + 1))
            xs = []
            ys = []
            cs = []
            for level in ls:
                xs.append(super_tree.values[level])
                ys.append(y_location)
                cs.append(segment.segment_at_level(level).persistence)
            segments, colors = _interpolate(ls, xs, ys, cs)
            edge_segments.append(segments)
            edge_colors = edge_colors + list(colors[:-1])

            # # Spanning edge
            if i + 1 < len(segment.birth_levels):
                lower_death = segment.death_levels[i + 1]
                lower_death_value = super_tree.values[lower_death]
                spanning_segments.append(
                    ([birth_value, y_location], [lower_death_value, y_location])
                )

            # Death edge
            if death_parent.id > 0:
                parent_x = super_tree.values[death + 1]
                parent_y = _y(index[death_parent.id])
                segments = ([parent_x, parent_y], [death_value, y_location])
                if i == 0:
                    death_segments.append(segments)
                else:
                    partial_death_segments.append(segments)

            # Birth edge
            if birth_parent.id > 0:
                parent_x = super_tree.values[birth - 1]
                parent_y = _y(index[birth_parent.id])
                segments = ([parent_x, parent_y], [birth_value, y_location])
                birth_segments.append(segments)

        # Find lens parent--child relations
        for level, sequence_segment in segment.segments.items():
            x = super_tree.values[level]
            tree = super_tree.sequence.trees[level]
            original_segment_id = sequence_segment.join_tree_segment.id
            for arc in tree.arcs:
                if arc.child_id == original_segment_id:
                    y_parent = _y(index[super_id_lookup[(level, arc.elder_id)]])
                    lens_lines.append(([x, y_parent], [x, y_location]))

    # Process colors
    edge_colors = np.asarray(edge_colors)
    mi, ma = minmax(edge_colors)

    # Plot the points
    point_coords = np.asarray(point_coords)
    plt.scatter(
        point_coords[:, 0],
        point_coords[:, 1],
        20,
        point_colors,
        vmin=mi,
        vmax=ma,
        zorder=5,
        linewidths=0,
        edgecolors="none",
    )

    # Plot the connections
    lc = LineCollection(death_segments, linewidth=0.55, color="#ddd")
    ax.add_collection(lc)
    lc = LineCollection(
        partial_death_segments, linewidth=0.55, linestyle="--", color="#ddd"
    )
    ax.add_collection(lc)
    lc = LineCollection(birth_segments, linewidth=0.55, linestyle="--", color="#ddd")
    ax.add_collection(lc)
    lc = LineCollection(spanning_segments, linewidth=0.55, linestyle=":", color="#ddd")
    ax.add_collection(lc)
    lc = LineCollection(np.concatenate(edge_segments), linewidth=2, clim=[mi, ma])
    lc.set_array(edge_colors)
    lines = ax.add_collection(lc)
    pc = PatchCollection(
        [
            FancyArrowPatch(
                line[1],
                line[0],
                shrinkA=0.04,
                shrinkB=0.04,
                arrowstyle="Simple, tail_width=0.0015, head_width=0.015, head_length=0.05",
                connectionstyle="arc3,rad=-0.05",
            )
            for line in lens_lines
        ],
        lw=0,
        color="silver",
        edgecolor='none',
        zorder=-1,
    )
    ax.add_collection(pc)

    # Plot the labels
    if text_offset is None:
        text_offset = (0, 0)
    for label, idx in point_labels:
        plt.annotate(
            label,
            (
                point_coords[idx, 0] + text_offset[0],
                point_coords[idx, 1] + text_offset[1],
            ),
            va="center",
            ha="center",
        )

    plt.xticks(super_tree.values)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return fig, ax, lines


def plot_super_join_tree_2(
    super_tree: SuperJoinTree, height=1, text_offset=None, ax=None
):
    """
    Plots a SuperJoinTree as a planar merge tree.

    Params
    ------
    tree: SuperJoinTree
        The SuperJoinTree to visualize
    width: float
        The width to locate the tree in.
    x_offset: float
        An offset to apply to the x coordinate.

    Returns
    -------
        fig, ax
    """
    # Get the current figure and axis, or create a new one
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    fig = plt.gcf()

    # Allocate variables for the coordinates
    segment_order = super_tree.traversal_order
    # Reverse the relation so that we can find the x-location of each segment
    index = {}
    for i, segment in enumerate(segment_order):
        index[segment.id] = i

    def _y(i):
        return i / len(segment_order) * height - height / 2

    # Combine all the coordinates
    point_coords = []
    part_colors = []
    point_labels = []
    death_connection_coords = []
    death_connection_colors = []
    partial_death_connection_coords = []
    partial_death_connection_colors = []
    birth_connection_coords = []
    birth_connection_colors = []
    spanning_connection_coords = []
    spanning_connection_colors = []
    for i, segment in enumerate(segment_order):
        persistence = super_tree.persistence_of(segment.id)
        y_location = _y(i)
        for i, (birth, death, birth_parent, death_parent) in enumerate(
            zip(
                segment.birth_levels,
                segment.death_levels,
                segment.birth_parents,
                segment.death_parents,
            )
        ):
            p1 = [super_tree.values[birth], y_location]
            p2 = [super_tree.values[death], y_location]

            point_coords.append(p1)
            point_coords.append(p2)
            part_colors.append(persistence)
            if i == 0:
                point_labels.append((segment.id, len(point_coords) - 1))

            if i + 1 < len(segment.birth_levels):
                p3 = [super_tree.values[segment.death_levels[i + 1]], y_location]
                spanning_connection_coords.append((p1, p3))
                spanning_connection_colors.append(persistence)

            if death_parent.id > 0:
                death_x = super_tree.values[death + 1]
                parent_y = _y(index[death_parent.id])
                connection_segment = (p2, [death_x, parent_y])
                if i == 0:
                    death_connection_coords.append(connection_segment)
                    death_connection_colors.append(persistence)
                else:
                    partial_death_connection_coords.append(connection_segment)
                    partial_death_connection_colors.append(persistence)
            if birth_parent.id > 0:
                death_x = super_tree.values[birth - 1]
                parent_y = _y(index[birth_parent.id])
                connection_segment = (p1, [death_x, parent_y])
                birth_connection_coords.append(connection_segment)
                birth_connection_colors.append(persistence)

    # Process colors
    part_colors = np.asarray(part_colors)
    death_connection_colors = np.asarray(death_connection_colors)
    birth_connection_colors = np.asarray(birth_connection_colors)
    partial_death_connection_colors = np.asarray(partial_death_connection_colors)
    spanning_connection_colors = np.asarray(spanning_connection_colors)
    mi, ma = minmax(part_colors)

    # Plot the points
    point_coords = np.asarray(point_coords)
    plt.scatter(
        point_coords[:, 0],
        point_coords[:, 1],
        20,
        np.repeat(part_colors, 2),
        vmin=mi,
        vmax=ma,
        zorder=5,
        linewidths=0,
        edgecolors="none",
    )

    # Plot the connections
    connections = np.asarray(death_connection_coords)
    lc = LineCollection(connections, linewidth=0.55, clim=[mi, ma])
    lc.set_array(death_connection_colors)
    ax.add_collection(lc)
    connections = np.asarray(partial_death_connection_coords)
    lc = LineCollection(connections, linewidth=0.55, clim=[mi, ma], linestyle="--")
    lc.set_array(partial_death_connection_colors)
    ax.add_collection(lc)
    connections = np.asarray(birth_connection_coords)
    lc = LineCollection(connections, linewidth=0.55, clim=[mi, ma], linestyle="--")
    lc.set_array(birth_connection_colors)
    ax.add_collection(lc)
    connections = np.asarray(spanning_connection_coords)
    lc = LineCollection(connections, linewidth=0.55, clim=[mi, ma], linestyle=":")
    lc.set_array(spanning_connection_colors)
    ax.add_collection(lc)

    # Plot the edges
    point_coords = point_coords.reshape((-1, 1, 2))
    segments = np.concatenate([point_coords[:-1:2], point_coords[1::2]], axis=1)
    lc = LineCollection(segments, linewidth=2, clim=[mi, ma])
    lc.set_array(part_colors)
    ax.add_collection(lc)

    # Plot the labels
    if text_offset is None:
        text_offset = (0.1, 0)
    for label, idx in point_labels:
        plt.annotate(
            label,
            (
                point_coords[idx, 0, 0] + text_offset[0],
                point_coords[idx, 0, 1] + text_offset[1],
            ),
            ha="center",
            va="center",
        )

    plt.xticks(super_tree.values)
    return fig, ax


def hierarchical_super_join_tree_layout(
    super_tree: SuperJoinTree,
    width=1,
    x_offset=0,
    leaf_vs_root_factor=0.5,
    vert_gap=0.2,
    vert_loc=0,
):
    """
    Constructs a hierarchical tree layout for the super tree.

    Params
    ------
    tree: SuperJoinTree
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
        return len(node.death_children) == 0 or all(
            [
                child.final_death_parent == super_tree.root
                for child in node.death_children
            ]
        )

    leaf_count = len([node for node in super_tree.segments if _is_leaf(node)])

    def _hierarchy_pos(
        root,
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
            root_pos = {root.id: x_center}
        else:
            root_pos[root.id] = x_center

        # Order by decreasing death value
        leaf_count = 0
        children = [
            c
            for c in root.death_children_ordered(super_tree)
            if c.final_death_parent == root
        ]
        if len(children) != 0:
            root_dx = width / len(children)
            next_x = x_center - width / 2 - root_dx / 2

            for child in children:
                next_x += root_dx
                root_pos, leaf_pos, new_leaves = _hierarchy_pos(
                    child,
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

            left_most, right_most = minmax(
                np.asarray([leaf_pos[child.id][0] for child in children])
            )
            leaf_pos[root.id] = ((left_most + right_most) / 2, vert_loc)
        else:
            leaf_count = 1
            leaf_pos[root.id] = (left_most, vert_loc)
        return root_pos, leaf_pos, leaf_count

    root_pos, leaf_pos, leaf_count = _hierarchy_pos(
        super_tree.root,
        0,
        width,
        leaf_dx=width * 1.0 / leaf_count,
        vert_gap=vert_gap,
        vert_loc=vert_loc,
        x_center=x_center,
    )

    # preallocate the result
    pos = np.zeros((len(super_tree.segments), 2))

    # fill preliminary values
    for s in super_tree.segments:
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
    partial_edges = []
    for node in super_tree.segments[1:]:
        for child in node.death_children:
            coords = [tuple(pos[node.id, :]), tuple(pos[child.id, :])]
            if child.final_death_parent == node:
                edges.append(coords)
            else:
                partial_edges.append(coords)

    return pos[1:, :], edges, partial_edges


def plot_hierarchical_super_join_tree(
    super_tree: SuperJoinTree, width=1, x_offset=0, ax=None, text_offset=None
):
    """
    Plots the SuperJoinTree with an hierarchical layout using matplotlib.
    Dashed edges indicate partial arcs, i.e., the segments are merged in some
    of the levels but not at the highest level in the sequence.

    Params
    ------
    width: float
        The width to locate the tree in.
    x_offset: float
        An offset to apply to the x coordinate.
    ax: matplotlib axis
        The axis to plot to, if not given the current axis is used.
    text_offset: tuple
        A tuple specifying the x-y offsets to apply to the text annotation.
    """
    layout = hierarchical_super_join_tree_layout(
        super_tree, width=width, x_offset=x_offset
    )
    (pos, edges, partial_edges) = layout
    # Get the current figure and axis, or create a new one
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    fig = plt.gcf()

    # Plot the tree
    lc = LineCollection(edges, colors="#ddd", linewidths=1)
    ax.add_collection(lc)
    lc = LineCollection(partial_edges, colors="#ddd", linewidths=1, linestyle="--")
    ax.add_collection(lc)
    colors = [
        super_tree.persistence_of(idx) for idx in range(1, len(super_tree.segments))
    ]
    plt.scatter(pos[:, 0], pos[:, 1], 50, colors, zorder=2)

    if text_offset is None:
        text_offset = (width / 25, 0.005)
    for i, s in enumerate(super_tree.segments[1:]):
        plt.annotate(s.id, (pos[i, 0] + text_offset[0], pos[i, 1] + text_offset[1]))
    return fig, ax, layout


def plot_super_join_tree_sequence(super_tree: SuperJoinTree, ax=None):
    # Derive values from input
    sequence = super_tree.sequence
    values = super_tree.values
    # clim = [
    #     sequence.trees[0].root.birth_value,
    #     sequence.trees[0].root.death_value
    # ]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        plt.sca(ax)

    fig = plt.gcf()

    for idx, tree in enumerate(sequence.trees):
        label_map = {}
        for super_segment in super_tree.segments[1:]:
            if idx in super_segment.segments:
                segment = super_segment.segments[idx].join_tree_segment
                label_map[segment.id] = super_segment.id

        if sequence.noise_trees is None:
            noise = None
        else:
            noise = sequence.noise_trees[idx]

        plot_join_tree(
            tree, noise, width=0.75, x_offset=idx, clim=None, ax=ax, labels=label_map
        )

    plt.xticks(range(len(sequence.trees)))
    plt.gca().set_xticklabels(values)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.set_figwidth(20)
    fig.set_figheight(3)
    return fig, ax
