# src/data_loader.py
import networkx as nx
from pathlib import Path
from .config import RAW_DIR

def load_ego_graph(ego_id: int) -> nx.Graph:
    """
    Đọc file {ego_id}.edges trong data/raw và build Graph NetworkX.
    """
    path = Path(RAW_DIR) / f"{ego_id}.edges"
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G
