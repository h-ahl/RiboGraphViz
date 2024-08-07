import itertools
from collections.abc import Collection
from copy import copy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from networkx.drawing.nx_agraph import graphviz_layout

from viserna import utils
from viserna.utils import LEFT_DELIMITERS, RIGHT_DELIMITERS, get_strand_id

ALL_POSSIBLE_CONSEQUTIVE_DELIMITERS = [x[0] + x[1] for x in list(itertools.product(RIGHT_DELIMITERS, LEFT_DELIMITERS))]
IDENTICAL_CONSEQUTIVE_LEFT_DELIMITERS = [x + x for x in LEFT_DELIMITERS]
IDENTICAL_CONSEQUTIVE_RIGHT_DELIMITERS = [x + x for x in RIGHT_DELIMITERS]

class RNAGraph:
    """Represents an RNA secondary structure as a directed graph.

    Attributes:
        structure: RNA secondary structure in dot-bracket notation.
        sequence: Optional RNA sequence.
        fiveprime_coords_x: X-coordinate of the 5' end.
        fiveprime_coords_y: Y-coordinate of the 5' end.
        threeprime_coords_x: X-coordinate of the 3' end.
        threeprime_coords_y: Y-coordinate of the 3' end.
    """

    structure: str
    sequence: str | None = None
    fiveprime_coords_x: float | None = None
    fiveprime_coords_y: float | None = None
    threeprime_coords_x: float | None = None
    threeprime_coords_y: float | None = None

    def __init__(self, structure: str, sequence: str | None = None):
        """Initializes a directed graph representing an RNA secondary structure.

        Args:
            structure: RNA secondary structure in dot-bracket notation.
            sequence: Optional RNA sequence.

        Note:
            Nodes represent non-helix elements (hairpin loops, internal loops, etc.), and edges represent helices with
            weights corresponding to base pairs.
        """

        self.graph = nx.DiGraph()
        self.sequence = sequence
        self.structure = structure

        self._stems = sorted(utils.stems_from_pairs(utils.pairs_from_dotbracket(structure)))
        self._stem_assignment = utils.get_stem_assignment(self.structure)
        self._pairmap = utils.get_pairmap(self.structure)
        self._n_bases = len(self.structure)

        self.create_graph()

    def _edges_from_pairmap(self):
        """Process the _pairmap to add the edges to the graph and get the loop sizes."""
        jj = 0
        while jj < len(self._pairmap):
            if self._pairmap[jj] > jj:  # in structure
                self.add_edges(jj, self._pairmap[jj], 0, 0)
                jj = self._pairmap[jj] + 1
            else:  # external loop
                jj += 1

    def _remove_self_referencing_edges(self):
        """Remove edges that point to the same node."""
        self.graph.remove_edges_from(
            (nod, nod) for nod in range(len(self._stem_assignment)) if (nod, nod) in self.graph.edges)


    def _process_helix_nodes(self):
        """Processes helix nodes in the graph, adding edges between nucleotide nodes and adjacent helix nodes.

        Adds edges based on the stem assignments and updates the graph with helix connections.
        """

        stem_assignment_left = np.concatenate([np.array([-1]), self._stem_assignment[:-1]])
        stem_assignment_right = np.concatenate([self._stem_assignment[1:], np.array([-1])])
        for i, pair in enumerate(self._pairmap):
            if pair != -1:
                continue

            self.graph.add_node(f"n{i}")
            if stem_assignment_left[i] > 0:
                strand_id = get_strand_id(self.structure[i - 1])
                self.graph.add_edge(f'n{i}', f'h{self._stem_assignment[i - 1]}{strand_id}', len=1.25, mld_weight=0)

            if stem_assignment_right[i] > 0:
                strand_id = get_strand_id(self.structure[i + 1])
                self.graph.add_edge(f'n{i}', f'h{self._stem_assignment[i + 1]}{strand_id}', len=1.25, mld_weight=0)
            # TODO: add helix node a and b

    def _process_nucleotide_nodes(self):
        """Processes nucleotide nodes in the graph and adds edges between consecutive nodes.

        Edges are added between nodes "n{i-1}" and "n{i}" if both exist.
        """

        nucleotide_nodes = [n for n in list(self.graph.nodes) if isinstance(n, str) and n.startswith("n")]  # hacky
        for nuc in nucleotide_nodes:
            ind = int(nuc.replace("n", ""))
            if f"n{ind - 1}" in nucleotide_nodes:
                self.graph.add_edge("n%d" % (ind - 1), "n%d" % ind, len=1, mld_weight=0)

    def create_graph(self):
        """Creates a graph by reading the _pairmap array and recursively creating edges. Performs the following steps:
            - Initializes the graph with a starting node.
            - Reads the _pairmap array to add edges.
            - Removes self-referencing edges.
            - Processes helix and nucleotide nodes.
            - Adds edges between helix nodes based on stem length.
            - Adds edges between structure segments based on delimiters.
        """

        self.graph.add_node(0)
        self._edges_from_pairmap()
        self._remove_self_referencing_edges()
        self._process_helix_nodes()
        self._process_nucleotide_nodes()

        for stem_ind in range(1, len(self._stems) + 1):
            stem_length = len(self._stems[stem_ind - 1])

            self.graph.add_edge(
                f"h{stem_ind}a",
                f"h{stem_ind}b",
                len=(stem_length - 0.99),
                mld_weight=stem_length,
            )

        for i in range(self._n_bases - 1):
            if not ((self._stem_assignment[i + 1] != self._stem_assignment[i]) and
                    (self._stem_assignment[i + 1] != 0.0 and self._stem_assignment[i] != 0.0)):
                continue

            # Add edges based on the structure segment
            stem_ind_1 = self._stem_assignment[i]
            stem_ind_2 = self._stem_assignment[i + 1]
            structure_segment = self.structure[i: i + 2]

            if structure_segment in IDENTICAL_CONSEQUTIVE_LEFT_DELIMITERS:
                self.graph.add_edge(f"h{stem_ind_1}a", f"h{stem_ind_2}b", len=1, weight=1, mld_weight=0)
            elif structure_segment in ALL_POSSIBLE_CONSEQUTIVE_DELIMITERS:
                self.graph.add_edge(f"h{stem_ind_1}b", f"h{stem_ind_2}b", len=1, weight=1, mld_weight=0)
            elif structure_segment in IDENTICAL_CONSEQUTIVE_RIGHT_DELIMITERS:
                self.graph.add_edge(f"h{stem_ind_1}b", f"h{stem_ind_2}a", len=1, weight=1, mld_weight=0)

    def add_edges(self, start_index, end_index, last_helix, last_loop):
        """Recursively adds edges to the graph based on the pairing map.

        Args:
            start_index: Starting index of the sequence.
            end_index: Ending index of the sequence.
            last_helix: Last helix index used in the graph.
            last_loop: Last loop index used in the graph.

        Raises:
            ValueError: If start_index is greater than end_index.
        """

        if start_index > end_index:
            raise ValueError("start_index > end_index")

        if self._pairmap[start_index] == end_index:
            self.add_edges(start_index + 1, end_index - 1, self._stem_assignment[start_index], last_loop)

        else:
            jj = start_index
            while jj <= end_index:
                if self._pairmap[jj] > jj:
                    self.graph.add_edge(
                        int(last_loop), int(last_helix)
                    )
                    last_loop = copy(last_helix)

                    self.add_edges(jj, self._pairmap[jj], self._stem_assignment[jj], last_loop)
                    jj = self._pairmap[jj] + 1  # leaving helix
                else:
                    if last_helix not in list(self.graph.nodes):
                        self.graph.add_edge(
                            int(last_loop),
                            int(last_helix),
                            len=len(self._stems[int(last_helix - 1)]),
                            weight=len(self._stems[int(last_helix - 1)]),
                            mld_weight=0,
                        )
                        last_loop = copy(last_helix)
                    jj += 1

    def _update_flank_coordinates(self, node_positions_x, node_positions_y):
        """Updates the coordinates of the 5' and 3' flanking coordinates.

        Args:
            node_positions_x: Dictionary of x coordinates of nodes.
            node_positions_y: Dictionary of y coordinates of nodes.
        """
        if self.threeprime_coords_x is not None:
            self.threeprime_coords_x = self.threeprime_coords_x - node_positions_x[0]
            self.threeprime_coords_y = self.threeprime_coords_y - node_positions_y[0]

        if self.fiveprime_coords_x is not None:
            self.fiveprime_coords_x = self.fiveprime_coords_x - node_positions_x[0]
            self.fiveprime_coords_y = self.fiveprime_coords_y - node_positions_y[0]

    def _update_stem_coordinates(self, node_positions_x, node_positions_y, positions, bond_width):
        """Updates the coordinates of stem nodes based on their positions and bond width.

        Args:
            node_positions_x: Dictionary of x coordinates of nodes.
            node_positions_y: Dictionary of y coordinates of nodes.
            positions: Dictionary of initial node positions.
            bond_width: Bond width used for calculating positions.

        Returns:
            Updated dictionaries of x and y coordinates of nodes.
        """
        for i, stem in enumerate(self._stems):
            start_x, start_y = positions[f"h{i + 1}a"]
            fin_x, fin_y = positions[f"h{i + 1}b"]

            x_dist = fin_x - start_x
            y_dist = fin_y - start_y

            np.sqrt((start_x - fin_x) ** 2 + (start_y - fin_y) ** 2)
            stem_angle = np.arctan2(y_dist, x_dist)

            x_diff = np.cos(stem_angle + np.pi / 2) * 0.5 * bond_width
            y_diff = np.sin(stem_angle + np.pi / 2) * 0.5 * bond_width

            # Treat as equidistant divide stem length, and scatter points along.
            for j in range(len(stem)):
                x_midpoint = start_x + j * x_dist / (len(stem) - 0.99)
                y_midpoint = start_y + j * y_dist / (len(stem) - 0.99)

                if stem_angle < 0:
                    node_positions_x[stem[j][1]] = x_midpoint + x_diff
                    node_positions_x[stem[j][0]] = x_midpoint - x_diff
                    node_positions_y[stem[j][1]] = y_midpoint + y_diff
                    node_positions_y[stem[j][0]] = y_midpoint - y_diff

                else:
                    node_positions_x[stem[j][0]] = x_midpoint + x_diff
                    node_positions_x[stem[j][1]] = x_midpoint - x_diff
                    node_positions_y[stem[j][0]] = y_midpoint + y_diff
                    node_positions_y[stem[j][1]] = y_midpoint - y_diff
        return node_positions_x, node_positions_y

    def _configure_positions(self, subgraph: nx.Graph):
        """Configures positions of nodes in the subgraph using Graphviz layout.

        Args:
            subgraph: Subgraph for which to configure positions.

        Returns:
            Tuple containing:
                - graph_positions: Dictionary of node positions.
                - bond_width: Computed bond width between nodes.
                - node_positions_x: Dictionary of x coordinates of nodes.
                - node_positions_y: Dictionary of y coordinates of nodes.
        """
        graph_positions = graphviz_layout(subgraph, prog="neato")

        # If single-molecule, get the 5 and 3' end coordinates
        self.fiveprime_coords_x, self.fiveprime_coords_y = graph_positions["5'"]
        if "3'" in list(subgraph.nodes()):
            self.threeprime_coords_x, self.threeprime_coords_y = graph_positions["3'"]

        # Compute the bond width
        for u, v in list(subgraph.edges()):
            if not (u.startswith('n') and v.startswith('n')):
                continue
            x1, x2, y1, y2 = graph_positions[u][0], graph_positions[v][0], graph_positions[u][1], graph_positions[v][1]
            break
        bond_width = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        node_positions_x, node_positions_y = {}, {}
        for node in list(subgraph.nodes()):
            if node.startswith('n'):
                seq_ind = int(node[1:])
                node_positions_x[seq_ind] = graph_positions[node][0]
                node_positions_y[seq_ind] = graph_positions[node][1]

        return graph_positions, bond_width, node_positions_x, node_positions_y

    def _add_flank_nodes(self, subgraph: nx.Graph):
        """Adds nodes for the 5' and 3' ends to the subgraph.

        Args:
            subgraph: Subgraph to which the flank nodes are added.

        Returns:
            Modified subgraph with added flank nodes.
        """

        # 5' end
        if "n0" in list(subgraph.nodes()):
            subgraph.add_edge("n0", "5'", len=2)
        else:
            subgraph.add_edge("h1b", "5'", len=2)

        # 3' end
        if "n%d" % (self._n_bases - 1) in list(subgraph.nodes()):
            subgraph.add_edge("n%d" % (self._n_bases - 1), "3'", len=2)

        return subgraph

    def get_coordinates(self):
        """Calculates and returns the x and y coordinates of nodes in the graph.

        Returns:
            Tuple of arrays: (x_coords, y_coords).
        """
        plot_nodes = [n for n in list(self.graph.nodes) if isinstance(n, str)]
        subgraph = self.graph.subgraph(plot_nodes).to_undirected()

        # Add nodes specifically for the 5' and 3' ends
        subgraph = self._add_flank_nodes(subgraph)

        graph_positions, bond_width, node_positions_x, node_positions_y = self._configure_positions(subgraph)
        node_positions_x, node_positions_y = self._update_stem_coordinates(node_positions_x, node_positions_y,
                                                                                   graph_positions, bond_width)
        self._update_flank_coordinates(node_positions_x, node_positions_y)

        node_pos_list_x = [node_positions_x[i] - node_positions_x[0] for i in range(self._n_bases)]
        node_pos_list_y = [node_positions_y[i] - node_positions_y[0] for i in range(self._n_bases)]

        # Normalize to bond width
        node_pos_list_x /= bond_width
        node_pos_list_y /= bond_width

        if self.threeprime_coords_x is not None:
            self.threeprime_coords_x /= bond_width
            self.threeprime_coords_y /= bond_width

        if self.fiveprime_coords_x is not None:
            self.fiveprime_coords_x /= bond_width
            self.fiveprime_coords_y /= bond_width

        return node_pos_list_x, node_pos_list_y

    def _get_colors(self, c: list[float | str] | str | None, vmin: float | None = None, vmax: float | None = None,
                    cmap: str = "plasma"):
        """Processes the input to generate a list of colors.

        Args:
            c: List of floats or strings, string, or None.
            vmin: Minimum value for normalization.
            vmax: Maximum value for normalization.
            cmap: Colormap.

        Returns:
            List of colors.
        """

        # TODO: Add color modes
        if c is None:
            return ["k"] * self._n_bases

        if isinstance(c, str):
            if len(c) == 1:
                colors = [c] * self._n_bases
            else:
                assert len(c) == self._n_bases
                colors = c
        else:
            assert len(c) == self._n_bases
            colormap = plt.get_cmap(cmap)
            vmin = vmin or np.min(c)
            vmax = vmax or np.max(c)
            color_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            scalar_map = cm.ScalarMappable(norm=color_norm, cmap=colormap)
            colors = [scalar_map.to_rgba(val) for val in c]

        return colors

    def _get_alpha(self, alpha: float | list[float] | None) -> list[float]:
        """Processes the input alpha to ensure it is a list of appropriate length.

        Args:
            alpha: A float, list of floats, or None.

        Returns:
            List of floats.

        Raises:
            TypeError: If alpha is not a float, list of floats, or None.
        """
        if alpha is None:
            return [1] * self._n_bases

        if isinstance(alpha, float):
            alpha = [alpha] * self._n_bases
        elif isinstance(alpha, list[float]):
            assert len(alpha) == self._n_bases
        else:
            raise TypeError("Invalid alpha type. Must be float, list of floats or None.")
        return alpha

    def draw(
        self,
        c: str | Collection[str, ...] = None,
        cmap: str = "plasma",  # TODO: parse cmap
        alpha: float | Collection[float, ...] | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        nt_labels: Collection[str, ...] = None,
        nt_labels_offset: float | Collection[float, float] = 0.0,
        show_ends: bool = True,
        five_prime_label: str = "5'",
        three_prime_label: str = "3'",
        fontsize: float = 12,
        ax: plt.Axes | None = None,
        pad: int = 5
    ) -> plt.Axes | None:
        """Draws the structure using a GraphViz layout.

        Args:
            c: Color specification, which can be a single color, a collection of colors, or a list-like object of
                values to be used with `cmap`.
            nt_labels: Collection of nucleotide labels.
            nt_labels_offset: Offset for labels, either a float or a collection of two floats.
            cmap: Colormap to use for coloring.
            alpha: Transparency value, either a single float or a collection of floats.
            vmin: Minimum data value for colormap scaling.
            vmax: Maximum data value for colormap scaling.
            ax: Matplotlib Axes object to use for drawing. If None, a new Axes object is created.
            fontsize: Font size for labels.
            show_ends: Whether to show labels for the 5' and 3' ends.
            five_prime_label: Label for the 5' end.
            three_prime_label: Label for the 3' end.
            pad: Padding for the margins.

        Returns:
            Matplotlib Axes object if `ax` is set. Otherwise, plots the structure in the current axis and returns None.

        """
        if isinstance(nt_labels_offset, float):
            nt_labels_offset = [nt_labels_offset, nt_labels_offset]

        x_coords, y_coords = self.get_coordinates()

        ax = ax or plt.gca()
        ax.set_aspect("equal")
        ax.set_xlim([np.min(x_coords) - pad + nt_labels_offset[0], np.max(x_coords) + pad + nt_labels_offset[0]])
        ax.set_ylim([np.min(y_coords) - pad + nt_labels_offset[1], np.max(y_coords) + pad + nt_labels_offset[1]])

        colors = self._get_colors(c, cmap, vmin, vmax)
        alpha = self._get_alpha(alpha)

        # Plot nucleotides
        for i in range(self._n_bases):
            circ = plt.Circle(
                (x_coords[i] + nt_labels_offset[0], y_coords[i] + nt_labels_offset[1]), radius=1 / 2, color=colors[i],
                alpha=alpha[i], linewidth=0
            )
            ax.add_artist(circ)
            if nt_labels:
                plt.text(
                    x_coords[i] + nt_labels_offset[0],
                    y_coords[i] + nt_labels_offset[1],
                    nt_labels[i],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=fontsize,
                )

        if show_ends:
            if self.fiveprime_coords_x and self.fiveprime_coords_y:
                plt.text(self.fiveprime_coords_x + nt_labels_offset[0], self.fiveprime_coords_y + nt_labels_offset[1],
                         five_prime_label, fontsize=fontsize)
            if self.threeprime_coords_x and self.threeprime_coords_y:
                plt.text(self.threeprime_coords_x + nt_labels_offset[0], self.threeprime_coords_y + nt_labels_offset[1],
                         three_prime_label, fontsize=fontsize)

        ax.axis("off")
        return ax
