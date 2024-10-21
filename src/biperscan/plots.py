import numpy as np
import networkx as nx
from collections import defaultdict


def plot_network(
    graph,
    layout: str | dict = "sfdp",
    nodes: bool = True,
    edges: bool = True,
    labels: bool = True,
    node_kws: dict | None = None,
    line_kws: dict | None = None,
    font_kws: dict | None = None,
    hide_ticks: bool = True,
):
    import matplotlib.pyplot as plt

    if line_kws == None:
        line_kws = dict()
    if node_kws == None:
        node_kws = dict()
    if font_kws == None:
        font_kws = dict()

    # Compute the layout
    layouts = dict(
        spring=nx.spring_layout,
        spectral=nx.spectral_layout,
        forceatlas=nx.forceatlas2_layout,
    )
    if type(layout) is dict:
        pos = layout
    elif layout in ["sfdp", "neato", "dot"]:
        pos = nx.nx_agraph.pygraphviz_layout(graph, prog=layout)
    elif layout in layouts:
        pos = layouts[layout](graph)
    else:
        raise ValueError(f'Unknown layout "{layout}"')

    # Create the figure
    fig = plt.gcf()
    ax = plt.gca()

    # Node size
    if "node_size" not in node_kws:
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        area = width * height * fig.dpi
        node_kws["node_size"] = np.pi * area / graph.number_of_nodes()

    # Draw the nodes
    if nodes:
        nodes = nx.draw_networkx_nodes(
            graph, pos=pos, hide_ticks=hide_ticks, **node_kws
        )

    # Draw the edges
    if "arrows" not in line_kws:
        line_kws["arrows"] = True
    if edges:
        edges = nx.draw_networkx_edges(
            graph,
            pos=pos,
            node_size=node_kws["node_size"],
            hide_ticks=hide_ticks,
            **line_kws,
        )

    # Draw the labels
    if labels:
        labels = nx.draw_networkx_labels(
            graph, pos=pos, hide_ticks=hide_ticks, **font_kws
        )

    # Remove the axes
    ax.set_frame_on(False)
    return pos, nodes, edges, labels, node_kws["node_size"]


class LinkageHierarchy:
    """A class for plotting and transforming linkage hierarchies."""

    def __init__(
        self,
        distances: np.ndarray[np.float32],
        point_lens_values: np.ndarray[np.float32],
        point_lens_grades: np.ndarray[np.uint32],
        col_to_edge: np.ndarray[np.uint32],
        row_to_point: np.ndarray[np.uint32],
        linkage_hierarchy: dict[str, np.ndarray],
    ):
        import pandas as pd

        self.num_points = len(point_lens_grades)
        self.point_lens_values = point_lens_values
        self.point_lens_grades = point_lens_grades
        self.hierarchy = pd.DataFrame.from_dict(linkage_hierarchy)
        self.hierarchy["lens_value"] = point_lens_values[
            row_to_point[self.hierarchy.lens_grade]
        ]
        self.hierarchy["distance_value"] = distances[
            col_to_edge[self.hierarchy.distance_grade]
        ]

        self.max_lens_grade = linkage_hierarchy["lens_grade"][-1]

        # Caches
        self._G = None

    def as_pandas(self):
        """Returns the hierarchy as a pandas DataFrame."""
        return self.hierarchy.copy()

    def as_networkx(self):
        """Returns the hierarchy as a networkx graph."""
        if self._G is None:
            import pandas as pd

            self._G = nx.compose(
                *(
                    nx.from_pandas_edgelist(
                        pd.DataFrame(
                            dict(
                                parent=self.hierarchy.index + self.num_points,
                                child=self.hierarchy[side],
                                root=self.hierarchy[f"{side}_root"],
                                lens_grade=self.hierarchy.lens_grade,
                                distance_grade=self.hierarchy.distance_grade,
                                lens_value=self.hierarchy.lens_value,
                                distance_value=self.hierarchy.distance_value,
                            )
                        ),
                        source="child",
                        target="parent",
                        edge_attr=True,
                        create_using=nx.DiGraph,
                    )
                    for side in ["parent", "child"]
                )
            )

            id_grade_map = self._id_grade_map()
            id_value_map = self._id_value_map()
            self._G.add_nodes_from(
                (
                    n,
                    dict(
                        lens_grade=id_grade_map[n, 0],
                        distance_grade=id_grade_map[n, 1],
                        lens_value=id_value_map[n, 0],
                        distance_value=id_value_map[n, 1],
                    ),
                )
                for n in range(self.num_points + self.hierarchy.shape[0])
            )

        return self._G

    def plot_network(
        self,
        *,
        layout: str | dict = "sfdp",
        nodes: bool = True,
        edges: bool = True,
        labels: bool = True,
        node_kws: dict | None = None,
        line_kws: dict | None = None,
        font_kws: dict | None = None,
        hide_ticks: bool = True,
    ):
        """
        Plots the hierarchy as a network.

        Parameters
        ----------
        layout : str or dict
            The layout of the network. If a string, it should be one of "sfdp",
            "neato", "dot", "spring", or "spectral". If a dictionary, it should
            be a mapping from node to position.
        nodes : bool
            Whether to plot the nodes.
        edges : bool
            Whether to plot the edges.
        labels : bool
            Whether to plot the labels.
        node_kws : dict, optional
            Additional keyword arguments for plotting the nodes.
        line_kws : dict, optional
            Additional keyword arguments for plotting the edges.
        font_kws : dict, optional
            Additional keyword arguments for plotting the labels.
        hide_ticks : bool
            Whether to hide ticks.
        """
        import matplotlib.pyplot as plt

        if font_kws == None:
            font_kws = dict(font_color="w", font_size=8)

        id_grade_map = self._id_grade_map()
        node_kws = dict(
            nodelist=range(id_grade_map.shape[0]),
            node_color=id_grade_map[:, 0],
            cmap=plt.cm.viridis,
            vmin=0,
            vmax=self.max_lens_grade,
            **(node_kws or dict()),
        )

        G = self.as_networkx()
        edge_dists = nx.get_edge_attributes(G, "distance_grade")
        line_kws = dict(
            edgelist=edge_dists.keys(),
            edge_color=edge_dists.values(),
            edge_cmap=plt.cm.Purples,
            **(line_kws or dict()),
        )

        pos, nodes, edges, labels, node_size = plot_network(
            G,
            layout,
            nodes,
            edges,
            labels,
            node_kws,
            line_kws,
            font_kws,
            hide_ticks=hide_ticks,
        )
        if nodes:
            plt.colorbar(nodes).set_label("Lens grade")
        return pos

    def plot_persistence_areas(
        self,
        *,
        view_type: str = "grade",
        transposed: bool = False,
        labels: bool = True,
        offset_x: float = 0.02,
        offset_y: float = 0.0,
        node_kws: dict | None = None,
        line_kws: dict | None = None,
        text_kws: dict | None = None,
    ):
        """
        Plots the distance and lens grade (or values) of the hierarchy.

        Parameters
        ----------
        view_type : str
            The type of view to plot. Either "grade" or "value".
        transposed : bool
            Whether to transpose the plot.
        labels : bool
            Whether to plot the labels.
        offset_x : float
            The x offset for the labels.
        offset_y : float
            The y offset for the labels.
        node_kws : dict, optional
            Additional keyword arguments for plotting the nodes
        line_kws : dict, optional
            Additional keyword arguments for plotting the lines.
        text_kws : dict, optional
            Additional keyword arguments for plotting the labels.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        # Fill in default parameters
        if line_kws is None:
            line_kws = dict(linestyle="-", linewidths=0.5, color="k")
        if text_kws is None:
            text_kws = dict(fontsize=8, ha="left", va="center")
        if node_kws is None:
            node_kws = dict(s=60, edgecolors="none", linewidths=0)

        # Compute the line segments
        if view_type == "value":
            base_map = self._id_value_map()
        elif view_type == "grade":
            base_map = self._id_grade_map()
        else:
            raise ValueError(f'Unknown view type "{view_type}"')
        grade_columns = [f"lens_{view_type}", f"distance_{view_type}"]

        curves = np.concatenate(
            (
                base_map[self.hierarchy.child, :],
                base_map[self.num_points :, :],
                base_map[self.hierarchy.parent, :],
                np.full((self.hierarchy.shape[0], 2), np.nan),
            ),
            axis=1,
            dtype=np.float32,
        ).reshape(-1, 2)

        if transposed:
            curves = [curve[::-1] for curve in curves]

        # Create figure
        fig = plt.gcf()
        ax = plt.gca()

        # Plot the curves
        ax.add_collection(LineCollection([curves], **line_kws))
        ax.autoscale()

        # Extract link counts
        grouped = self.hierarchy.reset_index().groupby(grade_columns)
        grade_counts = (
            grouped.index.agg(len).reset_index().rename(columns={"index": "count"})
        )

        # Plot the links
        if transposed:
            grade_columns = grade_columns[::-1]
        xs, ys = grade_counts[grade_columns[0]], grade_counts[grade_columns[1]]
        plt.scatter(
            xs,
            ys,
            c=grade_counts["count"],
            zorder=2,
            cmap="viridis",
            vmin=1,
            vmax=grade_counts["count"].max(),
            **node_kws,
        )
        if view_type != "grade":
            cb = plt.colorbar()
            cb.set_label("Num. links")

        # Annotate the links
        if labels:
            annotate_ids = (
                grouped.index.apply(
                    lambda x: ", ".join([f"{v + self.num_points}" for v in x])
                )
                .reset_index()
                .rename(columns=dict(index="id"))
            )
            for _, lg, dg, id in annotate_ids.itertuples():
                plt.text(lg + offset_x, dg + offset_y, id, **text_kws)

        if transposed:
            plt.xlabel("Distance " + view_type)
            plt.ylabel("Lens " + view_type)
        else:
            plt.xlabel("Lens " + view_type)
            plt.ylabel("Distance " + view_type)

    def _id_grade_map(self):
        id_grade_map = np.empty(
            (self.num_points + self.hierarchy.shape[0], 2), dtype=np.int32
        )
        id_grade_map[: self.num_points, 0] = self.point_lens_grades
        id_grade_map[: self.num_points, 1] = -1
        id_grade_map[self.num_points :, 0] = self.hierarchy.lens_grade
        id_grade_map[self.num_points :, 1] = self.hierarchy.distance_grade
        return id_grade_map

    def _id_value_map(self):
        id_value_map = np.empty(
            (self.num_points + self.hierarchy.shape[0], 2), dtype=np.int32
        )
        id_value_map[: self.num_points, 0] = self.point_lens_values
        id_value_map[: self.num_points, 1] = 0
        id_value_map[self.num_points :, 0] = self.hierarchy.lens_value
        id_value_map[self.num_points :, 1] = self.hierarchy.distance_value
        return id_value_map


class MinimalPresentation:
    """A class for plotting and transforming minimal presentations."""

    def __init__(
        self,
        distances: np.ndarray[np.float32],
        point_lens_values: np.ndarray[np.float32],
        point_lens_grades: np.ndarray[np.uint32],
        col_to_edge: np.ndarray[np.uint32],
        row_to_point: np.ndarray[np.uint32],
        minpres: dict[str, np.ndarray],
    ):
        import pandas as pd

        self.num_points = len(point_lens_values)
        self.point_lens_values = point_lens_values
        self.point_lens_grades = point_lens_grades
        self.minpres = pd.DataFrame.from_dict(minpres)
        self.minpres["lens_value"] = self.point_lens_values[
            row_to_point[self.minpres.lens_grade]
        ]
        self.minpres["distance_value"] = distances[
            col_to_edge[self.minpres.distance_grade]
        ]

        self.max_lens = point_lens_values.max()
        self.max_lens_grade = minpres["lens_grade"][-1]

        # Caches
        self._G = None

    def as_pandas(self):
        """Returns the minimal presentation as a pandas DataFrame."""
        return self.minpres.copy()

    def as_networkx(self):
        """Returns the minimal presentation as a networkx graph."""
        if self._G is None:
            self._G = nx.from_pandas_edgelist(
                self.minpres,
                source="child",
                target="parent",
                edge_attr=True,
                create_using=nx.MultiDiGraph,
            )
        return self._G

    def compute_value_death_curves(self):
        def _traces_to_curve(traces):
            """go from [x1, x2, ...] and [y1, y2, ...] to [(x1, y1), (x2, y1), ...]"""

            result = np.empty((2 * len(traces) + 3, 2), dtype=np.float32)
            result[0, 0] = traces.lens_value.iloc[0]
            result[1:-2, 0] = np.repeat(traces.lens_value, 2)
            result[-2:, 0] = self.max_lens

            result[:2, 1] = 0
            result[2:-1, 1] = np.repeat(traces.distance_value, 2)
            result[-1, 1] = 0
            return result

        return self.minpres.groupby("child")[["lens_value", "distance_value"]].apply(
            _traces_to_curve
        )

    def compute_grade_death_curves(self):
        def _traces_to_curve(traces):
            """go from [x1, x2, ...] and [y1, y2, ...] to [(x1, y1), (x2, y1), ...]"""

            result = np.empty((2 * len(traces) + 3, 2), dtype=np.uint32)
            result[0, 0] = traces.lens_grade.iloc[0]
            result[1:-2, 0] = np.repeat(traces.lens_grade, 2)
            result[-2:, 0] = self.max_lens_grade

            result[:2, 1] = 0
            result[2:-1, 1] = np.repeat(traces.distance_grade, 2)
            result[-1, 1] = 0
            return result

        return (
            self.minpres.groupby("child")[["lens_grade", "distance_grade"]]
            .apply(_traces_to_curve)
            .to_numpy()
        )

    def plot_network(
        self,
        *,
        layout: str | dict = "sfdp",
        nodes: bool = True,
        edges: bool = True,
        labels: bool = True,
        node_kws: dict | None = None,
        line_kws: dict | None = None,
        font_kws: dict | None = None,
        hide_ticks: bool = True,
    ):
        """
        Plots the hierarchy as a network.

        Parameters
        ----------
        layout : str or dict
            The layout of the network. If a string, it should be one of "sfdp",
            "neato", "dot", "spring", or "spectral". If a dictionary, it should
            be a mapping from node to position.
        nodes : bool
            Whether to plot the nodes.
        edges : bool
            Whether to plot the edges.
        labels : bool
            Whether to plot the labels.
        node_kws : dict, optional
            Additional keyword arguments for plotting the nodes.
        line_kws : dict, optional
            Additional keyword arguments for plotting the edges.
        font_kws : dict, optional
            Additional keyword arguments for plotting the labels.
        hide_ticks : bool
            Whether to hide ticks.
        """
        import matplotlib.pyplot as plt

        if font_kws == None:
            font_kws = dict(font_color="w", font_size=8)

        node_kws = dict(
            nodelist=np.arange(len(self.point_lens_values)),
            node_color=self.point_lens_grades,
            cmap=plt.cm.viridis,
            vmin=0,
            vmax=self.max_lens_grade,
            **(node_kws or dict()),
        )

        G = self.as_networkx()
        pos, nodes, _, labels, node_size = plot_network(
            G,
            layout,
            nodes,
            False,
            labels,
            node_kws,
            line_kws,
            font_kws,
            hide_ticks=hide_ticks,
        )
        if nodes:
            c = plt.colorbar(nodes, shrink=0.8)
            c.set_label("Lens grade")

        if edges:
            edge_dists = nx.get_edge_attributes(G, "distance_grade")
            edges_per_radius = defaultdict(dict)
            for k, v in edge_dists.items():
                edges_per_radius[k[2]][k] = v
            max_radius = max(edges_per_radius.keys())
            for i in range(len(edges_per_radius)):
                edges = edges_per_radius[i]
                nx.draw_networkx_edges(
                    G,
                    pos=pos,
                    edgelist=edges.keys(),
                    edge_color=edges.values(),
                    edge_cmap=plt.cm.Purples,
                    node_size=node_size,
                    connectionstyle=f"arc3, rad={0.3 * i / max_radius + 0.1}",
                    **(line_kws or dict()),
                )

        return pos

    def plot_persistence_areas(
        self,
        *,
        view_type: str = "grade",
        transposed: bool = False,
        line_kws: dict | None = None,
    ):
        """
        Plots the distance and lens grade (or values) of the minimal presentation.

        Parameters
        ----------
        view_type : str
            The type of view to plot. Either "grade" or "value".
        transposed : bool
            Whether to transpose the plot.
        line_kws : dict, optional
            Additional keyword arguments for plotting the lines.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        # Fill in default parameters
        if line_kws is None:
            line_kws = dict(linestyle="--", linewidths=0.5)

        # Compute the persistence areas
        if view_type == "value":
            curves = self.compute_value_death_curves()
        elif view_type == "grade":
            curves = self.compute_grade_death_curves()
        else:
            raise ValueError(f'Unknown view type "{view_type}"')

        if transposed:
            curves = [curve[:, ::-1] for curve in curves]

        # Create figure
        fig = plt.gcf()
        ax = plt.gca()

        # Plot the death curves
        ax.add_collection(LineCollection(curves, **line_kws))
        ax.autoscale()

        if transposed:
            plt.xlabel("Distance " + view_type)
            plt.ylabel("Lens " + view_type)
        else:
            plt.xlabel("Lens " + view_type)
            plt.ylabel("Distance " + view_type)


class MergeHierarchy:
    """A class for plotting and transforming merge hierarchies."""

    def __init__(
        self,
        distances: np.ndarray[np.float32],
        point_lens_values: np.ndarray[np.float32],
        col_to_edge: np.ndarray[np.uint32],
        row_to_point: np.ndarray[np.uint32],
        minpres: dict[str, np.ndarray],
        merge_hierarchy: nx.DiGraph,
    ):
        self.minpres = minpres
        self.graph = merge_hierarchy
        self.row_to_point = row_to_point
        nx.set_node_attributes(
            self.graph,
            {
                n: dict(
                    lens_value=point_lens_values[
                        row_to_point[self.graph.nodes[n]["lens_grade"]]
                    ],
                    distance_value=distances[
                        col_to_edge[self.graph.nodes[n]["distance_grade"]]
                    ],
                )
                for n in self.graph.nodes
            },
        )
        self.minpres["lens_value"] = point_lens_values[
            row_to_point[self.minpres["lens_grade"]]
        ]
        self.minpres["distance_value"] = distances[
            col_to_edge[self.minpres["distance_grade"]]
        ]

    def as_pandas(self):
        """
        Transform the merge hierarchy into pandas DataFrames.

        Returns
        -------
        nodes : pd.DataFrame
            One row for each merge in the merge hierarchy, can be used to
            extract data points contained in each aggregated node.

        edges : pd.DataFrame
            Edges in the merge hierarchy.
        """
        import pandas as pd

        nodes = pd.DataFrame.from_records(
            [record for _, record in self.graph.nodes(data=True)]
        )
        edges = nx.to_pandas_edgelist(self.graph, source="child", target="parent")
        return nodes, edges

    def as_networkx(self):
        """Returns the merge hierarchy as a networkx graph."""
        return self.graph.copy()

    def plot_network(
        self,
        *,
        layout: str | dict = "dot",
        nodes: bool = False,
        edges: bool = True,
        labels: bool = True,
        node_kws: dict | None = None,
        line_kws: dict | None = None,
        font_kws: dict | None = None,
        hide_ticks: bool = True,
    ):
        """
        Plots the hierarchy as a network.

        Parameters
        ----------
        layout : str or dict
            The layout of the network. If a string, it should be one of "sfdp",
            "neato", "dot", "spring", or "spectral". If a dictionary, it should
            be a mapping from node to position.
        nodes : bool
            Whether to plot the nodes.
        edges : bool
            Whether to plot the edges.
        labels : bool
            Whether to plot the labels.
        node_kws : dict, optional
            Additional keyword arguments for plotting the nodes.
        line_kws : dict, optional
            Additional keyword arguments for plotting the edges.
        font_kws : dict, optional
            Additional keyword arguments for plotting the labels.
        hide_ticks : bool
            Whether to hide ticks.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if node_kws == None:
            node_kws = dict()
        if font_kws == None:
            font_kws = dict(font_color="k", font_size=8)

        color_map = dict(
            same_parent="C0",
            same_child="C1",
            into_parent="C2",
            into_child="C3",
            duplicate="C4",
        )
        line_kws = dict(
            edgelist=self.graph.edges(),
            edge_color=[
                color_map[self.graph[u][v]["type"]] for u, v in self.graph.edges()
            ],
            **(line_kws or dict()),
        )
        pos, nodes, edges, labels, node_size = plot_network(
            self.graph,
            layout,
            nodes,
            edges,
            labels,
            node_kws,
            line_kws,
            font_kws,
            hide_ticks,
        )

        legend = plt.legend(
            title="Merge relation",
            handles=[
                Line2D([0], [0], color=color, lw=line_kws.get("width", 1), label=label)
                for label, color in color_map.items()
            ],
        )

        return pos, legend

    def plot_persistence_areas(
        self,
        *,
        view_type: str = "grade",
        transposed: bool = False,
        distance_offset: float = 1.05,
        line_kws: dict | None = None,
        font_kws: dict | None = None,
    ):
        """
        Plots the distance and lens grade (or values) of the merge hierarchy.

        Parameters
        ----------
        view_type : str
            The type of view to plot. Either "grade" or "value".
        transposed : bool
            Whether to transpose the plot.
        distance_offset : float
            A factor that controls the upper distance limit.
        line_kws : dict, optional
            Additional keyword arguments for plotting the lines.
        font_kws : dict, optional
            Additional keyword arguments for plotting the labels.
        """
        import matplotlib.pyplot as plt

        pos, x_attr, y_attr = self._persistent_positions(view_type, transposed)
        plt.scatter(
            self.minpres[x_attr],
            self.minpres[y_attr],
            s=1,
            color="silver",
            label="Minimal presentation",
            edgecolors="none",
            linewidths=0,
        )

        _, legend = self.plot_network(
            layout=pos,
            hide_ticks=False,
            line_kws=line_kws,
            font_kws=font_kws,
        )
        legend.set_loc("lower left")
        self._persistent_axes(
            x_attr,
            y_attr,
            view_type,
            distance_offset,
            transposed,
        )

    def plot_merges(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        s: int = 2,
        title_y: float = 0.9,
        arrowsize: int = 10,
        linewidth: float = 1,
    ):
        """
        Plots the points in each merges in the merge hierarchy.

        Parameters
        ----------
        xs : np.ndarray
            The x-coordinates of the points.
        ys : np.ndarray
            The y-coordinates of the points.
        s : int
            The size of the points.
        title_y : float
            The y-coordinate of the title.
        arrowsize : int
            The size of the arrows.
        linewidth : float
            The width of the arrows.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        num_plots = self.graph.number_of_nodes()
        n_rows = int(np.ceil(np.sqrt(num_plots)))
        n_cols = n_rows
        while n_rows * (n_cols - 1) > num_plots:
            n_cols -= 1

        cmap = ListedColormap(["silver", "C0", "C1", "C2"])

        for n, attr in self.graph.nodes(data=True):
            child = self.row_to_point[attr["child"]]
            parent = self.row_to_point[attr["parent"]]
            parent_side = self.row_to_point[list(attr["parent_side"])]
            child_side = self.row_to_point[list(attr["child_side"])]

            labels = np.zeros(len(xs), dtype=np.uint32)
            labels[parent_side] += 1
            labels[child_side] += 2
            plt.subplot(n_rows, n_cols, n + 1)
            plt.title(f"{n}", fontsize=6, y=title_y)
            plt.scatter(
                xs,
                ys,
                s=s,
                c=labels,
                cmap=cmap,
                vmin=0,
                vmax=3,
                linewidths=0,
                edgecolors="none",
            )
            plt.annotate(
                "",
                xytext=(xs[child], ys[child]),
                xy=(xs[parent], ys[parent]),
                arrowprops=dict(arrowstyle="->", color="k", linewidth=linewidth),
                size=arrowsize,
            )
            plt.axis("off")

    def _persistent_positions(self, view_type: str = "grade", transposed: bool = False):
        x_attr = f"lens_{view_type}"
        y_attr = f"distance_{view_type}"
        if transposed:
            x_attr, y_attr = y_attr, x_attr

        pos = {
            n: (attrs[x_attr], attrs[y_attr])
            for n, attrs in self.graph.nodes(data=True)
        }
        return pos, x_attr, y_attr

    def _persistent_axes(
        self,
        x_attr: str,
        y_attr: str,
        view_type: str = "grade",
        distance_offset: float = 1.05,
        transposed: bool = False,
    ):
        import matplotlib.pyplot as plt

        if transposed:
            plt.xlabel("Distance " + view_type)
            plt.ylabel("Lens " + view_type)
        else:
            plt.xlabel("Lens " + view_type)
            plt.ylabel("Distance " + view_type)

        if transposed:
            max_dist = distance_offset * max(
                attr[x_attr] for attr in self.graph.nodes.values()
            )
            plt.xlim(0, max_dist)
        else:
            max_dist = distance_offset * max(
                attr[y_attr] for attr in self.graph.nodes.values()
            )
            plt.ylim(0, max_dist)


class SimplifiedHierarchy:
    """A class for plotting and transforming simplified merge hierarchies."""

    def __init__(
        self,
        distances: np.ndarray[np.float32],
        point_lens_values: np.ndarray[np.float32],
        col_to_edge: np.ndarray[np.uint32],
        row_to_point: np.ndarray[np.uint32],
        minpres: dict[str, np.ndarray],
        merge_hierarchy: nx.DiGraph,
        simplified_hierarchy: nx.DiGraph,
    ):
        self.hierarchy = MergeHierarchy(
            distances,
            point_lens_values,
            col_to_edge,
            row_to_point,
            minpres,
            merge_hierarchy,
        )
        self.tree = simplified_hierarchy

    def as_pandas(self):
        """
        Transform the simplified hierarchy into pandas DataFrames.

        Returns
        -------
        nodes : pd.DataFrame
            One node for each connected component in the merge hierarchy. Lists
            which merges are part of the component.

        edges : pd.DataFrame
            Edges in the simplified hierarchy.

        merges : pd.DataFrame
            One row for each merge in the merge hierarchy, can be used to
            extract data points contained in each aggregated node.
        """
        import pandas as pd

        nodes = pd.DataFrame.from_records(
            [record for _, record in self.tree.nodes(data=True)]
        )
        merges = pd.DataFrame.from_records(
            [record for _, record in self.hierarchy.graph.nodes(data=True)]
        )
        edges = nx.to_pandas_edgelist(self.tree, source="child", target="parent")
        return nodes, edges, merges

    def as_networkx(self):
        """
        Transforms the simplified hierarchy into a networkx DiGraph.

        Returns
        -------
        simplified_hierarchy : nx.DiGraph
            The simplified hierarchy.

        merge_hierarchy : nx.DiGraph
            The merge hierarchy. Needed to map nodes in the simplified hierarchy
            to data points contained in the merge hierarchy.
        """
        return self.tree.copy(), self.hierarchy

    def plot_network(
        self,
        *,
        layout: str | dict = "dot",
        nodes: bool = False,
        edges: bool = True,
        labels: bool = True,
        node_kws: dict | None = None,
        line_kws: dict | None = None,
        font_kws: dict | None = None,
        hide_ticks=True,
    ):
        """
        Plots the hierarchy as a network.

        Parameters
        ----------
        layout : str or dict
            The layout of the network. If a string, it should be one of "sfdp",
            "neato", "dot", "spring", or "spectral". If a dictionary, it should
            be a mapping from node to position.
        nodes : bool
            Whether to plot the nodes.
        edges : bool
            Whether to plot the edges.
        labels : bool
            Whether to plot the labels.
        node_kws : dict, optional
            Additional keyword arguments for plotting the nodes.
        line_kws : dict, optional
            Additional keyword arguments for plotting the edges.
        font_kws : dict, optional
            Additional keyword arguments for plotting the labels.
        hide_ticks : bool
            Whether to hide ticks.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if font_kws == None:
            font_kws = dict(font_color="k", font_size=8)
        color_map = dict(into_parent="C0", into_child="C1", both="C2")
        line_kws = dict(
            edgelist=self.tree.edges(),
            edge_color=[
                color_map[self.tree[u][v]["type"]] for u, v in self.tree.edges()
            ],
            **(line_kws or dict()),
        )
        pos, nodes, edges, labels, node_size = plot_network(
            self.tree,
            layout,
            nodes,
            edges,
            labels,
            node_kws,
            line_kws,
            font_kws,
            hide_ticks=hide_ticks,
        )
        plt.legend(
            title="Descend from",
            handles=[
                Line2D(
                    [0], [0], color=f"C{i}", lw=line_kws.get("width", 1), label=label
                )
                for i, label in enumerate(["parent", "child", "both"])
            ],
        )
        return pos

    def plot_persistence_areas(
        self,
        view_type: str = "grade",
        transposed: bool = False,
        distance_offset: float = 1.05,
        node_kws: dict | None = None,
        line_kws: dict | None = None,
        font_kws: dict | None = None,
    ):
        """
        Plots the distance and lens grade (or values) of the simplified merge hierarchy.

        Parameters
        ----------
        view_type : str
            The type of view to plot. Either "grade" or "value".
        transposed : bool
            Whether to transpose the plot.
        distance_offset : float
            A factor that controls the upper distance limit.
        line_kws : dict, optional
            Additional keyword arguments for plotting the lines.
        font_kws : dict, optional
            Additional keyword arguments for plotting the labels.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if font_kws == None:
            font_kws = dict(font_color="k", font_size=8)

        pos, x_attr, y_attr = self.hierarchy._persistent_positions(
            view_type, transposed
        )
        plt.scatter(
            self.hierarchy.minpres[x_attr],
            self.hierarchy.minpres[y_attr],
            s=1,
            color="silver",
            label="Minimal presentation",
            edgecolors="none",
            linewidths=0,
        )

        comp_label = np.empty(self.hierarchy.graph.number_of_nodes(), dtype=np.uint32)
        for n, attrs in self.tree.nodes(data=True):
            comp_label[list(attrs["merges"])] = np.uint32(n)

        node_kws = dict(
            nodelist=self.hierarchy.graph.nodes(),
            node_color=comp_label % 10,
            cmap="tab10",
            vmin=0,
            vmax=9,
            **(node_kws or dict(node_size=4)),
        )
        plot_network(
            self.hierarchy.graph,
            layout=pos,
            nodes=True,
            edges=True,
            labels=False,
            node_kws=node_kws,
            line_kws=dict(
                width=0.1,
                arrows=False,
                edge_color="k",
                edgelist=[
                    (u, v)
                    for u, v in self.hierarchy.graph.edges()
                    if self.hierarchy.graph[u][v]["type"]
                    not in ["into_child", "into_parent"]
                ],
            ),
            font_kws=None,
            hide_ticks=False,
        )

        def _agg_xy(nodes):
            xs = np.array([pos[n][0] for n in nodes])
            ys = np.array([pos[n][1] for n in nodes])
            return np.average(xs), np.average(ys)

        pos = {n: _agg_xy(attrs["merges"]) for n, attrs in self.tree.nodes(data=True)}
        color_map = dict(into_parent="C0", into_child="C1", both="C2")
        line_kws = dict(
            edgelist=self.tree.edges(),
            edge_color=[
                color_map[self.tree[u][v]["type"]] for u, v in self.tree.edges()
            ],
            **(line_kws or dict()),
        )
        plot_network(
            self.tree,
            layout=pos,
            nodes=False,
            edges=True,
            labels=True,
            node_kws=dict(node_size=10),
            line_kws=line_kws,
            font_kws=font_kws,
            hide_ticks=False,
        )
        self.hierarchy._persistent_axes(
            x_attr,
            y_attr,
            view_type,
            distance_offset,
            transposed,
        )
        plt.legend(
            title="Descend from",
            loc="lower left",
            handles=[
                Line2D(
                    [0], [0], color=f"C{i}", lw=line_kws.get("width", 1), label=label
                )
                for i, label in enumerate(["parent", "child", "both"])
            ],
        )

    def plot_merges(
        self, xs: np.ndarray, ys: np.ndarray, *, s: int = 2, title_y: float = 0.9
    ):
        """
        Plots the points in each merges in the merge hierarchy.

        Parameters
        ----------
        xs : np.ndarray
            The x-coordinates of the points.
        ys : np.ndarray
            The y-coordinates of the points.
        s : int
            The size of the points.
        title_y : float
            The y-coordinate of the title.
        linewidth : float
            The width of the arrows.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        num_plots = self.tree.number_of_nodes()
        n_cols = int(np.ceil(np.sqrt(num_plots)))
        n_rows = n_cols
        while n_cols * (n_rows - 1) > num_plots:
            n_rows -= 1

        cmap = ListedColormap(["silver", "C0", "C1", "C2"])

        for n, attr in self.tree.nodes(data=True):
            parent_side = self.hierarchy.row_to_point[list(attr["parent_side"])]
            child_side = self.hierarchy.row_to_point[list(attr["child_side"])]

            labels = np.zeros(len(xs), dtype=np.uint32)
            labels[parent_side] += 1
            labels[child_side] += 2
            plt.subplot(n_rows, n_cols, n + 1)
            plt.title(f"{n}", fontsize=6, y=title_y)
            plt.scatter(
                xs,
                ys,
                s=s,
                c=labels,
                cmap=cmap,
                vmin=0,
                vmax=3,
                linewidths=0,
                edgecolors="none",
            )
            plt.axis("off")
