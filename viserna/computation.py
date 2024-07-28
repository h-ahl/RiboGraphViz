
import networkx as nx

from viserna import RNAGraph


def compute_max_ladder_distance(rna_graph: RNAGraph) -> float:
    """Computes the maximum ladder distance. This is defined as the end-to-end distance of helices present in
    structure, not counting lengths of loops."""
    nodes = list(rna_graph.graph.nodes)
    subgraph = rna_graph.graph.subgraph(nodes).to_undirected()

    first_helix_node = next((n for n in nodes if isinstance(n, str) and n.startswith("h")), None)

    if first_helix_node is None:
        return 0

    node1 = list(nx.traversal.bfs_edges(subgraph, first_helix_node))[-1][-1]
    node2 = list(nx.traversal.bfs_edges(subgraph, node1))[-1][-1]

    return nx.shortest_path_length(subgraph, node1, node2, weight="mld_weight")


def count_loops(rna_graph: RNAGraph) -> tuple[int, int, int, int, int]:
    degree_counts = {1: 0, 2: 0, 3: 0, 4: 0, 'more_than_4': 0}
    nodes = [n for n in rna_graph.graph.nodes if not isinstance(n, str)]
    subgraph = rna_graph.graph.subgraph(nodes).to_undirected()

    for _, degree in subgraph.degree:
        if degree in degree_counts:
            degree_counts[degree] += 1
        else:
            degree_counts['more_than_4'] += 1

    # Subtract one from the count of nodes with degree 1 to account for the exterior loop
    degree_counts[1] = max(0, degree_counts[1] - 1)

    return (degree_counts[1], degree_counts[2], degree_counts[3], degree_counts[4], degree_counts['more_than_4'])
