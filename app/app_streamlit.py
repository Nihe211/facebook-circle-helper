import streamlit as st
from pathlib import Path
import pandas as pd

from src.data_loader import load_ego_graph
from src.community_baseline import louvain_communities
from src.community_leiden import leiden_communities
from src.community_modern import learn_node2vec_embeddings
from src.invite_suggestion import suggest_invites


# ==== Tiện ích: tự tìm các ego có sẵn trong data/raw ====
def get_available_ego_ids():
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return [0]  # fallback

    ego_ids = []
    for p in raw_dir.glob("*.edges"):
        try:
            ego_ids.append(int(p.stem))  # "0.edges" -> 0
        except ValueError:
            continue
    ego_ids = sorted(set(ego_ids))
    return ego_ids or [0]


# ==== Chuyển partition node -> comm thành dict comm -> list node ====
def build_circles_from_partition(partition: dict):
    comm_to_nodes = {}
    for n, cid in partition.items():
        comm_to_nodes.setdefault(cid, []).append(n)
    return comm_to_nodes


def main():
    st.set_page_config(page_title="Facebook Circle Helper", layout="wide")
    st.title("Facebook Circle Helper 👥")
    st.write(
        "Prototype hỗ trợ phát hiện circle và gợi ý lời mời trong ego-Facebook network."
    )

    ego_ids = get_available_ego_ids()

    # ========== SIDEBAR: CẤU HÌNH ==========
    st.sidebar.header("Cấu hình community detection")

    ego_id = st.sidebar.selectbox("Chọn Ego user", ego_ids, index=0)

    algo = st.sidebar.selectbox(
        "Thuật toán cộng đồng",
        [
            "Louvain (baseline)",
            "Leiden (modern)",
        ],
    )

    run_btn = st.sidebar.button("⚙️ Run circle detection")

    # ========== KHI BẤM RUN: CHẠY CHIA CỘNG ĐỒNG ==========
    if run_btn:
        st.sidebar.success("Đang chạy trên ego {} ...".format(ego_id))

        # 1) Load ego graph
        G = load_ego_graph(ego_id)

        # 2) Chạy thuật toán cộng đồng
        if "Louvain" in algo:
            partition = louvain_communities(G)
        else:  # "Leiden"
            partition = leiden_communities(G)

        circles = build_circles_from_partition(partition)

        # 3) Học embedding node2vec (để dành cho phần gợi ý)
        nodes, emb = learn_node2vec_embeddings(G)

        # 4) Lưu vào session_state để các phần khác dùng
        st.session_state["G"] = G
        st.session_state["partition"] = partition
        st.session_state["circles"] = circles
        st.session_state["ego_id"] = ego_id
        st.session_state["algo"] = algo
        st.session_state["nodes"] = nodes
        st.session_state["emb"] = emb

        st.success("Hoàn thành community detection cho ego {} bằng {}.".format(ego_id, algo))

    # ========== PHẦN 1: DASHBOARD CIRCLE ==========
    st.markdown("## 1. Circle Detection Dashboard")

    if "G" not in st.session_state or "partition" not in st.session_state:
        st.info("Hãy chọn Ego + thuật toán ở sidebar rồi bấm **Run circle detection** trước.")
        return

    G = st.session_state["G"]
    partition = st.session_state["partition"]
    circles = st.session_state["circles"]
    ego_id = st.session_state["ego_id"]
    algo = st.session_state["algo"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Tóm tắt ego-network")
        st.write(f"**Ego ID:** `{ego_id}`")
        st.write(f"**Thuật toán:** {algo}")
        st.write(f"**Số node:** {G.number_of_nodes()}")
        st.write(f"**Số cạnh:** {G.number_of_edges()}")
        st.write(f"**Số cộng đồng:** {len(circles)}")

    with col2:
        st.subheader("Phân bố kích thước cộng đồng")
        rows = [
            {"Circle ID": cid, "Size": len(nodes)}
            for cid, nodes in circles.items()
        ]
        df_sizes = pd.DataFrame(rows).sort_values("Size", ascending=False)
        st.dataframe(df_sizes, use_container_width=True)

    # ========== PHẦN 2: KHÁM PHÁ 1 CIRCLE ==========
    st.markdown("---")
    st.markdown("## 2. Khám phá chi tiết một circle")

    circle_ids = sorted(circles.keys())
    selected_circle_id = st.selectbox("Chọn Circle ID", circle_ids)

    circle_nodes = circles[selected_circle_id]

    col3, col4 = st.columns(2)
    with col3:
        st.write(f"**Circle {selected_circle_id}**")
        st.write(f"Số node trong circle: **{len(circle_nodes)}**")
        st.write("Một vài node đầu tiên:")
        st.write(circle_nodes[:15])
    with col4:
        st.write("Gợi ý: sau này có thể thêm biểu đồ con (subgraph) hoặc thống kê độ bậc tại đây.")

    # ========== PHẦN 3: GỢI Ý LỜI MỜI (INVITE SUGGESTION) ==========
    st.markdown("---")
    st.markdown("## 3. Invite Suggestion (gợi ý lời mời)")

    nodes = st.session_state["nodes"]
    emb = st.session_state["emb"]

    top_k = st.slider("Số lượng gợi ý (top-k)", min_value=3, max_value=30, value=10)

    if st.button("🚀 Suggest invites cho circle này"):
        with st.spinner("Đang tính điểm gợi ý..."):
            suggestions = suggest_invites(G, circle_nodes, nodes, emb, top_k=top_k)

        # suggestions: list (candidate, total_score, cn_score, emb_score)
        table_rows = []
        for cand, score, cn, emb_score in suggestions:
            table_rows.append(
                {
                    "Candidate": cand,
                    "Score (tổng)": round(score, 4),
                    "Common Neighbors": round(cn, 2),
                    "Embedding similarity": round(emb_score, 4),
                }
            )

        st.subheader("Danh sách gợi ý lời mời (top-k)")
        df_sug = pd.DataFrame(table_rows)
        st.dataframe(df_sug, use_container_width=True)
        st.caption(
            "Score = α · similarity_embedding + β · common_neighbors (α, β khai báo trong config.py)."
        )


if __name__ == "__main__":
    main()
