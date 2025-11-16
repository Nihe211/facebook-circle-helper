# src/community_leiden.py

import networkx as nx
import igraph as ig
import leidenalg as la


def nx_to_igraph(G: nx.Graph) -> tuple[ig.Graph, dict]:
    """
    Chuyển một NetworkX Graph sang igraph.Graph.
    Trả về:
      - g_ig: graph igraph
      - idx_to_node: mapping index -> original node
    """
    # Đảm bảo node là list cố định
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Tạo graph igraph với đúng số node
    g_ig = ig.Graph()
    g_ig.add_vertices(len(nodes))

    # Thêm cạnh (dùng chỉ số)
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    g_ig.add_edges(edges)

    return g_ig, {i: n for n, i in node_to_idx.items()}


def leiden_communities(G: nx.Graph) -> dict:
    """
    Chạy Leiden trên graph G, trả về partition: node -> community_id.
    Sử dụng modularity làm hàm chất lượng.
    """
    g_ig, idx_to_node = nx_to_igraph(G)

    # Chạy Leiden, dùng CPM hoặc modularity; ở đây dùng modularity
    # la.find_partition trả về một partition (danh sách cộng đồng)
    partition = la.find_partition(
        g_ig,
        la.ModularityVertexPartition
    )

    # partition là list các community, mỗi community là list index node
    node_to_comm = {}
    for comm_id, community in enumerate(partition):
        for idx in community:
            node = idx_to_node[idx]
            node_to_comm[node] = comm_id

    return node_to_comm
