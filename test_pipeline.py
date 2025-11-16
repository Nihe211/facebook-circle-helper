# test_community_louvain_leiden.py

from src.data_loader import load_ego_graph
from src.community_baseline import louvain_communities
from src.community_leiden import leiden_communities


def print_partition_stats(name, partition):
    """
    In thống kê đơn giản về partition: số cộng đồng, size từng cộng đồng.
    partition: dict node -> community_id
    """
    comm_to_nodes = {}
    for n, cid in partition.items():
        comm_to_nodes.setdefault(cid, []).append(n)

    print(f"--- {name} ---")
    print(f"Số cộng đồng: {len(comm_to_nodes)}")
    sizes = sorted([len(v) for v in comm_to_nodes.values()], reverse=True)
    print("Kích thước các cộng đồng (top 10):", sizes[:10])
    print()


def main():
    ego_id = 0  # m có thể đổi sang 107, 1912,... nếu đã có file .edges
    G = load_ego_graph(ego_id)
    print(f"Ego {ego_id}: nodes = {G.number_of_nodes()}, edges = {G.number_of_edges()}")
    print()

    # Louvain
    part_louvain = louvain_communities(G)
    print_partition_stats("Louvain", part_louvain)

    # Leiden
    part_leiden = leiden_communities(G)
    print_partition_stats("Leiden", part_leiden)


if __name__ == "__main__":
    main()
