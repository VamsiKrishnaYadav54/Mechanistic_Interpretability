"""Build a circuit graph from edge scores."""
import networkx as nx


def build_circuit_graph(edge_scores: dict, threshold: float = 0.1) -> nx.DiGraph:
    """Keep only edges above threshold, return as a directed graph.
    
    Args:
        edge_scores: dict mapping "layer_N_head_M -> layer_P_head_Q" to float score
        threshold:   edges with score below this are pruned (not in the circuit)
    
    Returns:
        nx.DiGraph where nodes are model components and edges are important connections
    """
    G = nx.DiGraph()
    for edge_name, score in edge_scores.items():
        if abs(score) >= threshold:
            src, dst = edge_name.split(" -> ")
            G.add_edge(src, dst, weight=float(score))
    return G
