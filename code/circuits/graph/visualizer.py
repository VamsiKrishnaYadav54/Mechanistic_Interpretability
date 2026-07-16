"""Visualize a circuit graph."""
import networkx as nx
import plotly.graph_objects as go


def visualize_circuit(G: nx.DiGraph, title: str = "Circuit") -> go.Figure:
    """Render circuit as an interactive Plotly DAG."""
    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y,
                            mode='lines',
                            line=dict(width=1, color='#888'))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_trace = go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            text=list(G.nodes()),
                            textposition="top center",
                            marker=dict(size=10, color='#4a9eff'))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title=title, showlegend=False))
    return fig
