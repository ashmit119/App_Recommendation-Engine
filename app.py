"""
app.py
------
Streamlit web interface. Loads precomputed data instantly from disk.
Run precompute.py first if data/items.pkl or data/sim_cache.pkl don't exist.

Run with:
    streamlit run app.py
"""

import pickle
import streamlit as st

from recommender import hybrid_recommend, evaluate

import os
if not os.path.exists("data/items.pkl"):
    from utils import load_and_preprocess, precompute_similarities
    import pickle
    os.makedirs("data", exist_ok=True)
    items = load_and_preprocess()
    sim_cache = precompute_similarities(items)
    with open("data/items.pkl", "wb") as f:
        pickle.dump(items, f)
    with open("data/sim_cache.pkl", "wb") as f:
        pickle.dump(sim_cache, f)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="App & Game Recommender",
    page_icon="🎮",
    layout="centered",
)


# ─────────────────────────────────────────────────────────────
# INSTANT LOAD — just reading pickle files from disk
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_precomputed():
    """Load preprocessed data and similarity matrices from disk (~instant)."""
    try:
        with open("data/items.pkl", "rb") as f:
            items = pickle.load(f)
        with open("data/sim_cache.pkl", "rb") as f:
            sim_cache = pickle.load(f)
        return items, sim_cache
    except FileNotFoundError:
        st.error(
            "⚠️ Precomputed data not found. "
            "Please run `python precompute.py` first."
        )
        st.stop()


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────

def main():
    st.title("🎮 Hybrid App & Game Recommender")
    st.caption(
        "Combines **content-based similarity** (item metadata) with "
        "**review-text similarity** (TF-IDF on user reviews) into a "
        "tunable weighted hybrid score."
    )

    items, sim_cache = load_precomputed()

    st.divider()

    # ── Controls ─────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        item_type = st.radio("Type", ["app", "game"], horizontal=True)

    with col2:
        alpha = st.slider(
            "Content weight (alpha)",
            min_value=0.0, max_value=1.0, value=0.6, step=0.05,
            help=(
                "alpha=1.0 → pure metadata similarity  |  "
                "alpha=0.0 → pure review-text similarity"
            ),
        )

    type_items    = items[items["type"] == item_type]["name"].sort_values().unique()
    selected_item = st.selectbox(f"Select a {item_type}:", type_items)
    top_n         = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5)

    # ── Recommend ─────────────────────────────────────────────
    if st.button("🔍 Recommend", use_container_width=True):
        recs = hybrid_recommend(
            selected_item, items, sim_cache,
            top_n=int(top_n), alpha=alpha,
        )

        st.subheader(f"Top {top_n} recommendations for **{selected_item}**")
        for i, name in enumerate(recs, 1):
            row = items[items["name"] == name]
            if not row.empty:
                r          = row.iloc[0]
                avg_score  = r.get("avg_review_score", 0)
                r_count    = r.get("review_count", 0)
                categories = r.get("categories", "N/A")
                with st.container(border=True):
                    st.markdown(f"**{i}. {name}**")
                    st.caption(
                        f"📂 {categories}  |  "
                        f"⭐ {avg_score:.2f}  |  "
                        f"💬 {int(r_count)} reviews"
                    )
            else:
                st.write(f"{i}. {name}")

    st.divider()

    # ── Evaluation panel ─────────────────────────────────────
    with st.expander("📊 Evaluate model — Precision@K & Hit Rate@K"):
        st.caption(
            "Offline evaluation on a random sample. "
            "Relevant = same type AND avg review score ≥ threshold."
        )
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            sample_size = st.number_input("Sample size", 10, 500, 100)
        with ec2:
            eval_k = st.number_input("K", 1, 20, 5)
        with ec3:
            threshold = st.slider("Relevance threshold", 1.0, 5.0, 3.5, 0.5)

        if st.button("Run Evaluation", use_container_width=True):
            with st.spinner("Evaluating…"):
                metrics = evaluate(
                    items, sim_cache,
                    sample_size=int(sample_size),
                    top_n=int(eval_k),
                    alpha=alpha,
                    relevance_threshold=threshold,
                )
            m1, m2 = st.columns(2)
            m1.metric(f"Precision@{eval_k}", metrics[f"precision_at_{eval_k}"])
            m2.metric(f"Hit Rate@{eval_k}",  metrics[f"hit_rate_at_{eval_k}"])


if __name__ == "__main__":
    main()
