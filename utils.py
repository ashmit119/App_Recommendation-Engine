"""
utils.py
--------
Shared data loading, preprocessing, and similarity computation.
"""

import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def load_and_preprocess() -> pd.DataFrame:

    # --- Load ---
    apps_info     = pd.read_csv("apps_info.csv")
    games_info    = pd.read_csv("games_info.csv")
    apps_reviews  = pd.read_parquet("apps_reviews.parquet")
    games_reviews = pd.read_parquet("games_reviews.parquet")


    # --- Drop unused columns ---
    apps_info     = apps_info.drop(columns=["score","ratings_count","downloads","content_rating","section"], errors="ignore")
    games_info    = games_info.drop(columns=["score","ratings_count","downloads","content_rating","section"], errors="ignore")
    apps_reviews  = apps_reviews.drop(columns=["review_date","helpful_count"], errors="ignore")
    games_reviews = games_reviews.drop(columns=["review_date","helpful_count"], errors="ignore")

    # --- Rename info columns ---
    apps_info  = apps_info.rename(columns={"app_id":  "id", "app_name":  "name"})
    games_info = games_info.rename(columns={"game_id": "id", "game_name": "name"})

    # --- Rename review id columns individually before concat ---
    apps_reviews.columns  = [c if c != "app_id"  else "id" for c in apps_reviews.columns]
    games_reviews.columns = [c if c != "game_id" else "id" for c in games_reviews.columns]

    # --- Combine reviews ---
    reviews = pd.concat([apps_reviews, games_reviews], ignore_index=True)

    # --- DEBUG: print columns so we can confirm 'id' exists ---
    print(f"  reviews columns after concat: {reviews.columns.tolist()}")
    print(f"  total review rows: {len(reviews):,}")

    if "id" not in reviews.columns:
        raise ValueError(
            f"'id' column missing. apps_reviews cols were: {apps_reviews.columns.tolist()}, "
            f"games_reviews cols were: {games_reviews.columns.tolist()}"
        )

    # --- Cap reviews per item using groupby().head() ---
    # Avoids .apply() which has breaking changes in pandas 2.x
    MAX_REVIEWS_PER_ITEM = 40
    reviews = (
        reviews
        .sort_values("id")
        .groupby("id", group_keys=False)
        .head(MAX_REVIEWS_PER_ITEM)
        .reset_index(drop=True)
    )
    print(f"  reviews after cap: {len(reviews):,} rows")

    # --- VADER Sentiment ---
    sia = SentimentIntensityAnalyzer()

    def vader_sentiment(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return "neutral"
        compound = sia.polarity_scores(text)["compound"]
        if compound >= 0.05:   return "positive"
        if compound <= -0.05:  return "negative"
        return "neutral"

    print(f"  Running VADER...")
    reviews["sentiment"] = reviews["review_text"].apply(vader_sentiment)

    # --- Aggregate per item ---
    reviews_agg = (
        reviews.groupby("id")
        .agg(
            review_text      =("review_text",  lambda x: " ".join(x.dropna())),
            sentiment        =("sentiment",    lambda x: " ".join(x)),
            avg_review_score =("review_score", "mean"),
            review_count     =("review_score", "count"),
        )
        .reset_index()
    )

    # --- Merge with item info ---
    apps_info["type"]  = "app"
    games_info["type"] = "game"
    items_info = pd.concat([apps_info, games_info], ignore_index=True)
    items      = items_info.merge(reviews_agg, on="id", how="left")

    items["review_text"]      = items["review_text"].fillna("")
    items["sentiment"]        = items["sentiment"].fillna("")
    items["avg_review_score"] = items["avg_review_score"].fillna(0.0)
    items["review_count"]     = items["review_count"].fillna(0).astype(int)

    items["tags"] = (
        items["description"].fillna("") + " " +
        items["categories"].fillna("") + " " +
        items["sentiment"]
    ).str.lower()

    print(f"  Final items: {len(items):,}")
    return items.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# SIMILARITY COMPUTATION
# ─────────────────────────────────────────────────────────────

def _normalize(matrix: np.ndarray) -> np.ndarray:
    return MinMaxScaler().fit_transform(matrix)

def _content_similarity(subset: pd.DataFrame) -> np.ndarray:
    cv = CountVectorizer(max_features=5_000, stop_words="english")
    return cosine_similarity(cv.fit_transform(subset["tags"]))

def _review_text_similarity(subset: pd.DataFrame) -> np.ndarray:
    tfidf = TfidfVectorizer(max_features=5_000, stop_words="english")
    return cosine_similarity(tfidf.fit_transform(subset["review_text"]))


# ─────────────────────────────────────────────────────────────
# PRECOMPUTE & CACHE
# ─────────────────────────────────────────────────────────────

def precompute_similarities(items: pd.DataFrame) -> dict:
    MAX_ITEMS_PER_TYPE = 3_000
    cache = {}
    for item_type in items["type"].unique():
        subset = (
            items[items["type"] == item_type]
            .sort_values("review_count", ascending=False)
            .head(MAX_ITEMS_PER_TYPE)
            .reset_index(drop=True)
        )
        print(f"  [{item_type}] content similarity ({len(subset)} items)...")
        content_sim = _normalize(_content_similarity(subset))
        print(f"  [{item_type}] review-text similarity...")
        review_sim  = _normalize(_review_text_similarity(subset))
        cache[item_type] = {
            "subset":      subset,
            "content_sim": content_sim,
            "review_sim":  review_sim,
        }
        print(f"  [{item_type}] done")
    return cache
