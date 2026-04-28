Live link: https://apprecommendation-engine-119.streamlit.app/
# Hybrid App & Game Recommendation System

A hybrid recommendation engine built on the **Play Market 2025** dataset that combines content-based and review-text similarity to recommend apps and games.

## How It Works

| Signal | Method | Source |
|---|---|---|
| Content-based | Bag-of-Words + Cosine Similarity | Item description, categories, sentiment labels |
| Review-text | TF-IDF + Cosine Similarity | Aggregated user review text |
| Sentiment | VADER (NLP lexicon analysis) | Review text → positive / neutral / negative |

The final hybrid score is a weighted combination:

```
hybrid_score = α × content_similarity + (1 − α) × review_text_similarity
```

`α` is tunable at query time (default `0.6`).

## Architecture

```
recommender_project/
├── utils.py          # Data loading, preprocessing, similarity precomputation
├── recommender.py    # Hybrid recommendation logic + offline evaluation
├── app.py            # Streamlit web interface
└── requirements.txt
```

Similarity matrices are **precomputed once at startup** and cached — queries are O(1) slice operations, not O(n²) recomputations.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**CLI (with evaluation output):**
```bash
python recommender.py
```

**Web UI:**
```bash
streamlit run app.py
```

## Evaluation

The system is evaluated using offline metrics on a random held-out sample:

- **Precision@K** — fraction of top-K recommendations rated ≥ 3.5 stars
- **Hit Rate@K** — fraction of query items with at least one relevant recommendation in top-K

## Dataset

[Play Market 2025 — Kaggle](https://www.kaggle.com/)  
Contains apps and games metadata, user reviews, ratings, and download counts.

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `NLTK (VADER)` · `Streamlit` · `NumPy`
