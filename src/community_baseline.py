# src/community_baseline.py
import networkx as nx
import community as community_louvain  # python-louvain

def louvain_communities(G: nx.Graph) -> dict:
    """
    Chạy Louvain, trả về partition: node -> community_id
    """
    partition = community_louvain.best_partition(G)
    return partition
