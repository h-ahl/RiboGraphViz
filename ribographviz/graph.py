import itertools
from copy import copy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from networkx.drawing.nx_agraph import graphviz_layout

from ribographviz import utils
from ribographviz.utils import LEFT_DELIMITERS, RIGHT_DELIMITERS


class RNAGraph:
    """TODO: """
    structure: str
    sequence: str | None = None
    fiveprime_x: float | None = None
    fiveprime_y: float | None = None
    threeprime_x: float | None = None
    threeprime_y: float | None = None

    def __init__(self, structure: str, sequence: str | None = None):
        """Create a directed graph representing an RNA secondary structure, where nodes represent hairpin loops,
        internal loops, bulges, anything not a helix, and _stems represent helices.

        Edge weights (lengths) are equal to the number of base pairs present in the helix. Nodes are sized according to
        the number of unpaired bases in the loop.

        Args:
            Secondary structure in dot-bracket notation. Currently cannot handle pseudoknots.
        """

        self.graph = nx.DiGraph()
        self.sequence = sequence
        self.structure = structure

        self._stems = sorted(utils.stems_from_pairs(utils.pairs_from_dotbracket(structure)))
        self._stem_assignment = utils.get_stem_assignment(self.structure)
        self._pairmap = utils.get_pairmap(self.structure)
        self._loop_sizes = {0: 0}
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
                self._loop_sizes[0] += 1

    def _remove_self_referencing_edges(self):
        """Remove edges that point to the same node."""
        for nod in range(len(self._stem_assignment)):
            if (nod, nod) in self.graph.edges:
                self.graph.remove_edge(nod, nod)

    def _process_helix_nodes(self):
        stem_assignment_left = np.concatenate([np.array([-1]), self._stem_assignment[:-1]])
        stem_assignment_right = np.concatenate([self._stem_assignment[1:], np.array([-1])])
        for i, pair in enumerate(self._pairmap):
            if pair != -1:
                continue

            self.graph.add_node("n%d" % i)
            if stem_assignment_left[i] > 0:
                if self.structure[i - 1] in LEFT_DELIMITERS:
                    letter = "a"
                elif self.structure[i - 1] in RIGHT_DELIMITERS:
                    letter = "b"
                else:
                    raise ValueError(f"Unexpected character in structure: {self.structure[i - 1]}")

                self.graph.add_edge(
                    f"n{i}",
                    f"h{self._stem_assignment[i - 1]}{letter}",
                    len=1.25,
                    mld_weight=0
                )
                # TODO: add helix_a node here

            if stem_assignment_right[i] > 0:
                if self.structure[i + 1] in RIGHT_DELIMITERS:
                    letter = "a"
                elif self.structure[i + 1] in LEFT_DELIMITERS:
                    letter = "b"
                else:
                    raise ValueError(f"Unexpected character in structure: {self.structure[i - 1]}")

                self.graph.add_edge(
                    f"n{i}",
                    f"h{self._stem_assignment[i + 1]}{letter}",
                    len=1.25,
                    mld_weight=0
                )
                    # TODO: add helix_b node here

    def _process_nucleotide_nodes(self):
        """TODO: """
        nucleotide_nodes = [n for n in list(self.graph.nodes) if isinstance(n, str) and n.startswith("n")]  # hacky
        for nuc in nucleotide_nodes:
            ind = int(nuc.replace("n", ""))
            if f"n{ind - 1}" in nucleotide_nodes:
                self.graph.add_edge("n%d" % (ind - 1), "n%d" % ind, len=1, mld_weight=0)

    def create_graph(self):
        """Create graph by reading _pairmap array and recursively creating edges."""
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

            left_left_same = [x + x for x in LEFT_DELIMITERS]
            right_left = [x[0] + x[1] for x in list(itertools.product(RIGHT_DELIMITERS, LEFT_DELIMITERS))]
            right_right_same = [x + x for x in RIGHT_DELIMITERS]

            stem_ind_1 = self._stem_assignment[i]
            stem_ind_2 = self._stem_assignment[i + 1]
            if self.structure[i: i + 2] in left_left_same:
                self.graph.add_edge(f"h{stem_ind_1}a", f"h{stem_ind_2}b", len=1, weight=1,
                                    mld_weight=0)
            elif self.structure[i : i + 2] in right_left:
                self.graph.add_edge(f"h{stem_ind_1}b", f"h{stem_ind_2}b", len=1, weight=1,
                                    mld_weight=0)
            elif self.structure[i : i + 2] in right_right_same:
                self.graph.add_edge(f"h{stem_ind_1}b", f"h{stem_ind_2}a", len=1, weight=1,
                                    mld_weight=0)

    def add_edges(self, start_index, end_index, last_helix, last_loop):
        """Recursive method to add edges to graph."""

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

                    if int(last_helix) not in self._loop_sizes:
                        self._loop_sizes[int(last_helix)] = 0

                    self._loop_sizes[int(last_helix)] += 1

                    jj += 1

    def _update_flank_coordinates(self, node_positions_x, node_positions_y):
        if self.threeprime_x is not None:
            self.threeprime_x = self.threeprime_x - node_positions_x[0]
            self.threeprime_y = self.threeprime_y - node_positions_y[0]

        if self.fiveprime_x is not None:
            self.fiveprime_x = self.fiveprime_x - node_positions_x[0]
            self.fiveprime_y = self.fiveprime_y - node_positions_y[0]

    def _update_stem_coordinates(self, node_positions_x, node_positions_y, pos, bond_width):
        for i, stem in enumerate(self._stems):
            start_x, start_y = pos["h%da" % (i + 1)]
            fin_x, fin_y = pos["h%db" % (i + 1)]

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

    def _configure_positions(self, subgraph):
        graph_positions = graphviz_layout(subgraph, prog="neato")

        # If single-molecule, get the 5 and 3' end coordinates
        self.fiveprime_x, self.fiveprime_y = graph_positions["5'"]
        if "3'" in list(subgraph.nodes()):
            self.threeprime_x, self.threeprime_y = graph_positions["3'"]

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

    def _add_flank_nodes(self, subgraph):
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
        """
        Return: array of x_coords, array of y_coords
        """
        plot_nodes = [n for n in list(self.graph.nodes) if isinstance(n, str)]
        subgraph = self.graph.subgraph(plot_nodes)
        subgraph = subgraph.to_undirected()

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

        if self.threeprime_x is not None:
            self.threeprime_x /= bond_width
            self.threeprime_y /= bond_width

        if self.fiveprime_x is not None:
            self.fiveprime_x /= bond_width
            self.fiveprime_y /= bond_width

        return node_pos_list_x, node_pos_list_y

    def draw(
        self,
        label=None,
        struct_label=None,
        x0=0,
        y0=0,
        c=None,
        cmap="plasma",
        alpha=None,
        ax=None,
        vmin=None,
        vmax=None,
        fontsize=12,
    ):
        """Draw structure using GraphViz layout.

        Input: RNAGraph object or list of objects.

        label (str): list of labels for nucleotides.
        c: color. Can be single matplotlib color letter, string of matplotlib color letters,
                        or list-like object of values (to be used with cmap).
        alpha: Transparency value. can be single value or vector.

        ax: axis object to use for drawing.
        figsize: figure size.
        fontsize (int): fontsize for labels.

        Returns: ax object.
        """
        if struct_label is not None:
            raise RuntimeError("struct_label present and not multiple structures.")

        x_coords, y_coords = self.get_coordinates()

        ax = ax or plt.gca()
        ax.set_aspect("equal")

        if isinstance(c, str):
            if len(c) == 1:
                colors = [c] * self._n_bases
            else:
                assert len(c) == self._n_bases
                colors = c

        elif c is None:
            colors = ["k"] * self._n_bases

        else:
            assert len(c) == self._n_bases
            colormap = plt.get_cmap(cmap)
            if vmin is None:
                vmin = np.min(c)
            if vmax is None:
                vmax = np.max(c)
            color_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  # 3 for reac
            scalar_map = cm.ScalarMappable(norm=color_norm, cmap=colormap)
            colors = [scalar_map.to_rgba(val) for val in c]

        ax.set_xlim([np.min(x_coords) - 5 + x0, np.max(x_coords) + 5 + x0])
        ax.set_ylim([np.min(y_coords) - 5 + y0, np.max(y_coords) + 5 + y0])

        if alpha is None:
            alpha = [1] * self._n_bases
        elif isinstance(alpha, float):
            alpha = [alpha] * self._n_bases
        elif isinstance(alpha, list):
            assert len(alpha) == self._n_bases

        for i in range(self._n_bases):
            circ = plt.Circle(
                (x_coords[i] + x0, y_coords[i] + y0), radius=1 / 2, color=colors[i], alpha=alpha[i], linewidth=0
            )
            ax.add_artist(circ)
            if label:
                plt.text(
                    x_coords[i] + x0,
                    y_coords[i] + y0,
                    label[i],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=fontsize,
                )

        if self.fiveprime_x is not None:
            plt.text(self.fiveprime_x + x0, self.fiveprime_y + y0, "5'")

        if self.threeprime_x is not None:
            plt.text(self.threeprime_x + x0, self.threeprime_y + y0, "3'")

        ax.axis("off")
        return ax
