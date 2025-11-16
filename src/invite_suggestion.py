# src/invite_suggestion.py

import networkx as nx
import numpy as np
from .config import ALPHA_SCORE, BETA_SCORE


def common_neighbors_score(G: nx.Graph, x, circle_nodes):
    """
    Điểm classical: trung bình số common neighbors giữa x và
    các node trong circle.
    """
    scores = []
    for u in circle_nodes:
        cn = len(list(nx.common_neighbors(G, x, u)))
        scores.append(cn)
    return float(np.mean(scores)) if scores else 0.0


def embedding_similarity_score(x_idx: int, circle_indices, emb: np.ndarray) -> float:
    """
    Điểm embedding: trung bình cosine similarity giữa embedding của x
    và embedding của các node trong circle.
    """
    x_vec = emb[x_idx]
    circle_vecs = emb[circle_indices]
    num = circle_vecs @ x_vec
    denom = np.linalg.norm(circle_vecs, axis=1) * np.linalg.norm(x_vec)
    sim = num / (denom + 1e-8)
    return float(np.mean(sim))


def suggest_invites(G: nx.Graph, circle_nodes, all_nodes, emb: np.ndarray, top_k: int = 5):
    """
    Gợi ý top_k node chưa nằm trong circle:
    trả về list (candidate, total_score, cn_score, emb_score).
    """
    circle_set = set(circle_nodes)
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    candidates = [n for n in all_nodes if n not in circle_set]

    circle_indices = [node_to_idx[n] for n in circle_nodes]

    results = []
    for x in candidates:
        idx = node_to_idx[x]
        cn_score = common_neighbors_score(G, x, circle_nodes)
        emb_score = embedding_similarity_score(idx, circle_indices, emb)
        total = ALPHA_SCORE * emb_score + BETA_SCORE * cn_score
        results.append((x, total, cn_score, emb_score))

    # sort giảm dần theo tổng điểm
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
