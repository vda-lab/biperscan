import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from scipy.stats import rankdata
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform

from biperscan import BPSCAN

five_points = np.array(
    [
        [0, 1],
        [1.5, 1],
        [1.5, 0],
        [0, 0],
        [0.75, 0.5],
    ]
)
dist = pdist(five_points)
lens = np.array([1, 1, 1, 1, 2], dtype=np.float32) - 1
num_points = len(lens)


def plot_graph(g, data):
    # Use the data coordinates as node positions
    node_size = 100
    pos = {i: data[i] for i in range(data.shape[0])}
    nx.draw_networkx_nodes(
        g,
        pos=pos,
        node_size=node_size,
        node_color="blue",
    )
    nx.draw_networkx_labels(g, pos=pos, font_color="w", font_size=6)

    # Count and group edges between the same two points so they can be plotted with
    # different radii and are all visible.
    count_keys = defaultdict(int)
    per_radius = defaultdict(dict)
    for k, v in nx.get_edge_attributes(g, "grade").items():
        ordered_k = min(k[:2]), max(k[:2])
        count_keys[ordered_k] += 1
        k2 = count_keys[ordered_k]
        per_radius[k2][k] = v

    # Plot the edges
    max_radius = max(per_radius.keys())
    for r, edges in enumerate(per_radius.values()):
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=edges.keys(),
            edge_color="black",
            arrows=True,
            node_size=node_size,
            connectionstyle=f"arc3,rad={0.4 * r/max_radius + 0.1}",
        )
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=edges,
            node_size=node_size,
            font_size=6,
            connectionstyle=f"arc3,rad={0.4 * r/max_radius + 0.1}",
        )
    return pos


def plot_five_points_all():
    # Create a graph with all possible edges
    g = nx.MultiDiGraph()
    dist_grades = squareform(rankdata(dist, method="ordinal") - 1)
    for i, j in product(range(num_points), range(num_points)):
        if i != j:
            g.add_edge(
                i, j, grade=f"({int(max(lens[i], lens[j]))}, {int(dist_grades[i, j])})"
            )

    plot_graph(g, five_points)


def plot_five_points_minpres():
    # Convert the minimal presentation to the same format
    minpres = (
        BPSCAN(metric="precomputed", lens=lens, min_cluster_size=2)
        .fit(dist)
        .minimal_presentation_
    ).as_networkx()
    nx.set_edge_attributes(
        minpres,
        values={
            (i, j, c): f'({attrs["lens_grade"]}, {attrs["distance_grade"]})'
            for i, j, c, attrs in minpres.edges(data=True, keys=True)
        },
        name="grade",
    )
    plot_graph(minpres, five_points)
