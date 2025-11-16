import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from .config import (
    EMBEDDING_DIM,
    NODE2VEC_WALK_LENGTH,
    NODE2VEC_NUM_WALKS,
    NODE2VEC_WINDOW,
    DEFAULT_K,
)

def learn_node2vec_embeddings(G: nx.Graph):
    node2vec = Node2Vec(
        G,
        dimensions=EMBEDDING_DIM,
        walk_length=NODE2VEC_WALK_LENGTH,
        num_walks=NODE2VEC_NUM_WALKS,
        workers=2,
    )
    model = node2vec.fit(window=NODE2VEC_WINDOW, min_count=1)
    nodes = list(G.nodes())
    emb = np.array([model.wv[str(n)] for n in nodes])
    return nodes, emb

def node2vec_kmeans(G: nx.Graph, k: int = DEFAULT_K) -> dict:
    nodes, emb = learn_node2vec_embeddings(G)
    km = KMeans(n_clusters=k, random_state=0).fit(emb)
    labels = km.labels_
    return {nodes[i]: int(labels[i]) for i in range(len(nodes))}
