import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from scipy.interpolate import griddata

from dataclasses import dataclass, field

from .JoinTree import JoinTree, plot_join_tree, plot_hierarchical_join_tree


@dataclass
class JoinTreeSequence:
    """
    A class that represents a sequence of JoinTrees. It computes the data-point
    overlap between segments across levels and provides a simplification mehtod
    to extract the 'core' connections.

    Attributes
    ----------
    trees: list[JoinTree]
    segments: list[JoinTreeSequence.Segment]
    arcs: list[JoinTreeSequence.Arc]
    """

    @dataclass
    class Segment:
        """
        A class pointing to a JoinTree.Segments in the sequence. It keeps track
        of the in and out arcs of the segment across the levels in the sequence.

        Attributes
        ----------
        id: int
            The index / id of this segment in the JoinTreeSequence.
        level_id: int
            The level in which this segment occurs.
        segment_id: int
            The id of the segment in its tree.
        in_arcs: list[JoinTreeSequence.Arc]
            The arcs connecting to this segment.
        out_arcs: list[JoinTreeSequence.Arc]
            The arcs connecting from this segment.
        """

        id: int = 0
        level_id: int = 0
        in_arcs: list = field(default_factory=list)
        out_arcs: list = field(default_factory=list)
        join_tree_segment: JoinTree.Segment = None

        @property
        def persistence(self):
            return self.join_tree_segment.persistence

        @property
        def segment_id(self):
            return self.join_tree_segment.id

    @dataclass
    class Arc:
        """
        A class representing the connections between segments
        accross trees in the sequence.

        Attributes
        ----------
        elder_id: int
            The Segment id of the elder in the JoinTreeSequence.
        child_id: int
            The Segment id of the child in the JoinTreeSequence.
        elder_overlap: float
            The ratio of data-points in both segments with the data-points in
            the elder segment.
        child_overlap: float
            The ratio of data-points in both segments with the data-points in
            the child segment.
        jaccard_overlap: float
            the ratio intersection / union between both segments' data-points.
        """

        type: str = ""  # 'core', 'split', 'merge'
        elder_id: int = 0
        child_id: int = 0
        replaces: tuple = None
        elder_overlap: float = 0
        child_overlap: float = 0
        jaccard_overlap: float = 0

        def __iter__(self):
            """
            Yields the elder_id and child_id in sequence. Allows easy tuple
            construction from Arcs: `tuple(arc_object)`
            """
            yield self.elder_id
            yield self.child_id

        def __eq__(self, other):
            """
            Identity only by elder and child id, assumes that connections between
            two segments are unique, i.e., only one arc connects the same two
            segments.
            """
            return other and tuple(self) == tuple(other)

        def __hash__(self):
            """
            Allows arcs to be put in a set, assumes that connections between
            two segments are unique, i.e., only one arc connects the same two
            segments.
            """
            return hash(tuple(self))

    trees: list[JoinTree] = field(default_factory=list)
    noise_trees: list[JoinTree] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)

    def __init__(
        self, trees: list[JoinTree] = None, noise_trees: list[JoinTree] = None
    ):
        """
        Constructs the JoinTreeSequence from a sequence of JoinTrees.

        Builds a table indicating which segment each data-point belongs to in
        each level and constructs JoinTreeSequence from that table.

        Params
        ------
        trees: list[JoinTree]
            A sequence of JoinTrees.
        noise_trees: list[JoinTree]
            An optional sequence of noise trees as given by `JoinTree.simplify`.
            Only used in the plotting functions.
        """
        self.trees = trees
        self.noise_trees = noise_trees
        self.segments = []
        self.arcs = []

        if trees is not None:
            self._construct_tree_sequence(trees)

    def simplify(self):
        """
        Simplifies the JoinTreeSequency by keeping only the strongest in and out
        arc of each segment. Segments that are the strongest on both sides are
        labelled as 'core', segments that are only strongest on the elder side
        are labelled as 'split', and segments that are only strongest on the
        child side are labelled as 'merge'.
        """

        def _copy_node(node):
            return JoinTreeSequence.Segment(
                id=node.id,
                level_id=node.level_id,
                join_tree_segment=node.join_tree_segment,
            )

        signal = JoinTreeSequence()
        signal.trees = self.trees
        signal.noise_trees = self.noise_trees
        signal.segments = [_copy_node(s) for s in self.segments]

        # Extract only strongest in and out arc on each segment
        in_arcs = set()
        out_arcs = set()
        for segment in self.segments:
            if len(segment.out_arcs) > 0:
                overlaps = [a.jaccard_overlap for a in segment.out_arcs]
                arc = segment.out_arcs[np.argmax(overlaps)]
                out_arcs.add(arc)
            if len(segment.in_arcs) > 0:
                overlaps = [a.jaccard_overlap for a in segment.in_arcs]
                arc = segment.in_arcs[np.argmax(overlaps)]
                in_arcs.add(arc)

        # Set the remaining arcs in the data structure
        signal.arcs = list(in_arcs | out_arcs)
        for arc in signal.arcs:
            signal.segments[arc.elder_id].in_arcs.append(arc)
            signal.segments[arc.child_id].out_arcs.append(arc)

        # Set arc types
        intersection = in_arcs & out_arcs
        for arc in intersection:
            arc.type = "core"
        for arc in out_arcs - intersection:
            arc.type = "merge"
        for arc in in_arcs - intersection:
            arc.type = "split"
            elder = self.segments[arc.elder_id]
            elder_points = set(
                self.trees[elder.level_id].unique_points_of(elder.segment_id)
            )

            arcs = signal.segments[arc.child_id].in_arcs.copy()
            while len(arcs) > 0:
                arc_ = arcs.pop(0)
                incoming = signal.segments[arc_.child_id]
                if arc_.type == "merge":
                    child_points = set(
                        self.trees[incoming.level_id].unique_points_of(
                            incoming.segment_id
                        )
                    )
                    intersection = float(len(elder_points & child_points))
                    union = len(elder_points) + len(child_points) - intersection
                    child_overlap = intersection / len(child_points)
                    elder_overlap = intersection / len(elder_points)
                    jaccard = intersection / union
                    if jaccard > arc_.jaccard_overlap:
                        # remove old arcs
                        signal.arcs.remove(arc)
                        signal.arcs.remove(arc_)
                        signal.segments[arc.elder_id].in_arcs.remove(arc)
                        signal.segments[arc_.elder_id].in_arcs.remove(arc_)
                        signal.segments[arc.child_id].out_arcs.remove(arc)
                        signal.segments[arc_.child_id].out_arcs.remove(arc_)

                        # add new arc
                        new_arc = JoinTreeSequence.Arc(
                            type="core",
                            elder_id=arc.elder_id,
                            child_id=arc_.child_id,
                            replaces=(arc_, arc),
                            elder_overlap=child_overlap,
                            child_overlap=elder_overlap,
                            jaccard_overlap=jaccard,
                        )
                        signal.arcs.append(new_arc)
                        signal.segments[new_arc.elder_id].in_arcs.append(new_arc)
                        signal.segments[new_arc.child_id].out_arcs.append(new_arc)
                        break
                else:
                    arcs = arcs + incoming.in_arcs.copy()

        return signal

    def segment_distances(self):
        return [tree.segment_distances() for tree in self.trees]

    def _construct_tree_sequence(self, trees: list[JoinTree]):

        def _create_segment_table(trees: list[JoinTree]) -> np.ndarray:
            num_points = len(trees[0].root.point_ids)
            num_levels = len(trees)
            table = np.zeros((num_points, num_levels), dtype="int")

            for idx, tree in enumerate(trees):
                for segment in tree.segments:
                    table[tree.unique_points_of(segment.id), idx] = segment.id

            return table

        # Add top level segments.
        index = {}
        segment_counter = 0
        level_idx = len(trees) - 1
        for segment in trees[-1].segments:
            self.segments.append(
                JoinTreeSequence.Segment(
                    id=segment_counter, level_id=level_idx, join_tree_segment=segment
                )
            )
            index[(level_idx, segment.id)] = segment_counter
            segment_counter = segment_counter + 1

        # Link segments of lower levels together
        segment_of_point = _create_segment_table(trees)
        level_indices = np.arange(len(trees) - 2, -1, -1)
        for level_idx in level_indices:
            tree = trees[level_idx]
            parent_tree = trees[level_idx + 1]
            for segment in tree.segments:
                # append current segment
                self.segments.append(
                    JoinTreeSequence.Segment(
                        id=segment_counter,
                        level_id=level_idx,
                        join_tree_segment=segment,
                    )
                )
                index[(level_idx, segment.id)] = segment_counter

                # Evaluate overlaps
                mask = segment_of_point[:, level_idx] == segment.id
                points = segment_of_point[mask, level_idx + 1]
                parent_ids, counts = np.unique(points, return_counts=True)
                for idx, parent_id in enumerate(parent_ids):
                    parent = parent_tree.segments[parent_id]
                    parent_sequence_id = index[(level_idx + 1, parent_id)]
                    intersection = float(counts[idx])
                    union = (
                        len(segment.point_ids) + len(parent.point_ids) - intersection
                    )
                    arc = JoinTreeSequence.Arc(
                        elder_id=parent_sequence_id,
                        child_id=segment_counter,
                        elder_overlap=intersection / len(parent.point_ids),
                        child_overlap=intersection / len(segment.point_ids),
                        jaccard_overlap=intersection / union,
                    )
                    self.segments[segment_counter].out_arcs.append(arc)
                    self.segments[parent_sequence_id].in_arcs.append(arc)
                    self.arcs.append(arc)

                segment_counter = segment_counter + 1


def _overlap_colormap(xy_data):
    def _clamp(x):
        return max(min(x, 1), 0)

    key_xy_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    key_xy_RGBs = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype="float")

    reds = griddata(key_xy_points, key_xy_RGBs.T[0], xy_data)
    greens = griddata(key_xy_points, key_xy_RGBs.T[1], xy_data)
    blues = griddata(key_xy_points, key_xy_RGBs.T[2], xy_data)

    res = np.vstack((reds, greens, blues, np.ones(len(blues)))).T
    return np.vectorize(_clamp)(res)


def _type_colormap(arc):
    if arc.type == "core":
        return "k"
    if arc.type == "split":
        return "c"
    return "orchid"


def plot_join_tree_sequence_colormap():
    x = np.arange(0, 1 + 0.05, 0.05)
    y = np.arange(0, 1 + 0.05, 0.05)
    X, Y = np.meshgrid(x, y)
    xy_data = np.vstack((X.flatten(), Y.flatten())).T
    res = _overlap_colormap(xy_data)
    plt.imshow(
        res.reshape((len(x), len(y), 4)),
        extent=[0, 1, 0, 1],
        origin="lower",
        interpolation="spline16",
    )
    plt.xticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=6)
    plt.yticks([0, 0.5, 1], ["0", "0.5", "1"], fontsize=6)
    plt.ylabel("parent", fontsize=6, labelpad=0)
    plt.xlabel("child", fontsize=6, labelpad=0)
    return plt.gcf()


def plot_join_tree_sequence(
    sequence: JoinTreeSequence, x_values: list[float], clim=None, text_offset=(0, 0)
):
    if clim is None:
        clim = [sequence.trees[0].root.birth_value, sequence.trees[0].root.death_value]

    width = (x_values[1] - x_values[0]) * 0.65
    for idx, tree in enumerate(sequence.trees):
        if sequence.noise_trees is None:
            noise = None
        else:
            noise = sequence.noise_trees[idx]
        plot_join_tree(
            tree,
            noise,
            width=width,
            x_offset=x_values[idx],
            clim=clim,
            text_offset=text_offset,
        )

    plt.xticks(x_values)


def plot_join_tree_hierarchy_sequence(
    sequence: JoinTreeSequence,
    x_values: list[float],
    text_offset=(0, 0),
):
    if sequence.arcs[0].type == "":
        color = lambda a: _overlap_colormap([a.child_overlap, a.elder_overlap])
    else:
        color = lambda a: _type_colormap(a)

    width = (x_values[1] - x_values[0]) * 0.65

    positions = []
    for idx, tree in enumerate(sequence.trees):
        _, _, layout = plot_hierarchical_join_tree(
            tree, width=width, x_offset=x_values[idx], text_offset=text_offset
        )
        positions.append(layout[0])

    ax = plt.gca()
    Path = mpath.Path
    for arc in sequence.arcs:
        elder = sequence.segments[arc.elder_id]
        elder_pos = tuple(positions[elder.level_id][elder.segment_id])
        child = sequence.segments[arc.child_id]
        child_pos = tuple(positions[child.level_id][child.segment_id])

        if elder_pos[1] == child_pos[1]:
            mid_pos = ((elder_pos[0] + child_pos[0]) / 2, child_pos[1] + 0.2)
        else:
            mid_pos = (
                (elder_pos[0] + child_pos[0]) / 2,
                (elder_pos[1] + child_pos[1]) / 2,
            )

        p = mpatches.PathPatch(
            Path(
                [child_pos, mid_pos, elder_pos], [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            ),
            fc="none",
            transform=ax.transData,
            linewidth=0.55,
            color=color(arc),
        )
        ax.add_patch(p)

    plt.xticks(x_values)
    plt.yticks([])
