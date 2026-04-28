"""
recommender.py
--------------
Hybrid recommendation engine combining:
  - Content-based similarity  (item metadata: description, categories, sentiment)
  - Review-text similarity    (TF-IDF on aggregated user review language)

Also includes Precision@K and Hit-Rate evaluation.

Usage
-----
    python recommender.py
"""

import numpy as np
import pandas as pd
from utils import load_and_preprocess, precompute_similarities


# ─────────────────────────────────────────────────────────────
# 1.  HYBRID RECOMMENDATION
# ─────────────────────────────────────────────────────────────

def hybrid_recommend(
    item_name: str,
    items: pd.DataFrame,
    sim_cache: dict,
    top_n: int = 5,
    alpha: float = 0.6,
) -> list[str]:
    """
    Return the top-N most similar items using a weighted hybrid score:

        hybrid_score = alpha * content_similarity
                     + (1 - alpha) * review_text_similarity

    Parameters
    ----------
    item_name : str   — exact name of the query item
    items     : DataFrame — full items DataFrame (used for fallback lookup)
    sim_cache : dict  — precomputed similarity matrices from utils.precompute_similarities()
    top_n     : int   — number of recommendations to return  (default 5)
    alpha     : float — weight for content similarity in [0, 1] (default 0.6)
                        higher alpha → more metadata-driven recommendations
                        lower  alpha → more community-review-driven recommendations

    Returns
    -------
    list[str]  — recommended item names, ranked best-first
    """

    # ── Validate query item ──────────────────────────────────
    item_rows = items[items["name"] == item_name]
    if item_rows.empty:
        print(f"[WARN] '{item_name}' not found. Returning popularity-based fallback.")
        return _popularity_fallback(items, top_n)

    item      = item_rows.iloc[0]
    item_type = item["type"]
    item_cat  = item["categories"]

    # ── Retrieve precomputed matrices for this type ──────────
    entry        = sim_cache[item_type]
    subset       = entry["subset"]
    content_sim  = entry["content_sim"]
    review_sim   = entry["review_sim"]

    # ── Locate query item inside the type-subset ────────────
    item_idx_series = subset[subset["name"] == item_name].index
    if item_idx_series.empty:
        return _popularity_fallback(subset, top_n)
    idx = item_idx_series[0]

    # ── Filter to items sharing at least one category ────────
    def _shares_category(cat_str: str) -> bool:
        query_cats = {c.strip().lower() for c in str(item_cat).split(",")}
        cand_cats  = {c.strip().lower() for c in str(cat_str).split(",")}
        return bool(query_cats & cand_cats)

    cat_mask    = subset["categories"].apply(_shares_category)
    cat_indices = subset[cat_mask].index.tolist()

    # Fallback: use the whole type if no category overlap found
    if len(cat_indices) < 2:
        cat_indices = subset.index.tolist()

    # ── Slice precomputed similarity vectors ─────────────────
    content_vec = content_sim[idx, cat_indices]
    review_vec  = review_sim[idx,  cat_indices]

    # ── Weighted hybrid score ────────────────────────────────
    hybrid_scores = alpha * content_vec + (1.0 - alpha) * review_vec

    # ── Rank and exclude query item itself ───────────────────
    sorted_positions = np.argsort(hybrid_scores)[::-1]
    recommended_indices = [
        cat_indices[p]
        for p in sorted_positions
        if cat_indices[p] != idx
    ][:top_n]

    recommendations = subset.iloc[recommended_indices]["name"].tolist()

    # ── Popularity fill-up (edge case: too few category matches) ─
    if len(recommendations) < top_n:
        recommendations = _fill_with_popular(
            recommendations, subset, item_name, top_n
        )

    return recommendations


def _popularity_fallback(df: pd.DataFrame, top_n: int) -> list[str]:
    """Return top-N items ranked by avg_review_score × log(review_count)
    — a Bayesian-style popularity score that balances rating with volume."""
    df = df.copy()
    df["pop_score"] = df["avg_review_score"] * np.log1p(df["review_count"])
    return df.nlargest(top_n, "pop_score")["name"].tolist()


def _fill_with_popular(
    existing: list[str],
    df: pd.DataFrame,
    exclude_name: str,
    top_n: int,
) -> list[str]:
    """Top up recommendations with popular items not already included."""
    popular = _popularity_fallback(df, top_n + len(existing) + 1)
    for name in popular:
        if name not in existing and name != exclude_name:
            existing.append(name)
        if len(existing) >= top_n:
            break
    return existing


# ─────────────────────────────────────────────────────────────
# 2.  OFFLINE EVALUATION  (Precision@K and Hit Rate)
# ─────────────────────────────────────────────────────────────

def evaluate(
    items: pd.DataFrame,
    sim_cache: dict,
    sample_size: int = 100,
    top_n: int = 5,
    alpha: float = 0.6,
    relevance_threshold: float = 3.5,
) -> dict:
    """
    Evaluate recommendation quality on a random sample of items using:

      • Precision@K  — fraction of top-K recommendations that are "relevant"
                        (relevant = same type AND avg_review_score ≥ threshold)
      • Hit Rate@K   — fraction of query items that have at least one relevant
                        recommendation in the top-K list

    Parameters
    ----------
    sample_size          : number of random query items to evaluate
    top_n                : K in Precision@K and Hit Rate@K
    relevance_threshold  : min avg_review_score to consider an item relevant

    Returns
    -------
    dict with keys 'precision_at_k' and 'hit_rate_at_k'
    """

    sample = items.sample(min(sample_size, len(items)), random_state=42)

    precisions = []
    hits        = 0

    for _, row in sample.iterrows():
        recs = hybrid_recommend(
            row["name"], items, sim_cache, top_n=top_n, alpha=alpha
        )

        # Build relevance set: items of same type with high avg score
        relevant_set = set(
            items[
                (items["type"] == row["type"]) &
                (items["avg_review_score"] >= relevance_threshold) &
                (items["name"] != row["name"])
            ]["name"]
        )

        n_relevant_in_recs = sum(1 for r in recs if r in relevant_set)
        precision          = n_relevant_in_recs / top_n if top_n > 0 else 0.0
        precisions.append(precision)

        if n_relevant_in_recs > 0:
            hits += 1

    results = {
        f"precision_at_{top_n}": round(np.mean(precisions), 4),
        f"hit_rate_at_{top_n}":  round(hits / len(sample), 4),
    }
    return results


# ─────────────────────────────────────────────────────────────
# 3.  MAIN  — CLI demo + evaluation report
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading and preprocessing data…")
    items = load_and_preprocess()

    print("Precomputing similarity matrices…")
    sim_cache = precompute_similarities(items)

    # ── Demo recommendation ──────────────────────────────────
    query = "MONOPOLY"
    print(f"\n── Top-5 recommendations for '{query}' (alpha=0.6) ──")
    recs = hybrid_recommend(query, items, sim_cache, top_n=5, alpha=0.6)
    for i, name in enumerate(recs, 1):
        print(f"  {i}. {name}")

    # ── Evaluation report ────────────────────────────────────
    print("\n── Offline Evaluation (sample=100, K=5) ──")
    metrics = evaluate(items, sim_cache, sample_size=100, top_n=5)
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
