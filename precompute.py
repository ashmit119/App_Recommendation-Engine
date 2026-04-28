"""
precompute.py
-------------
Run this ONCE offline to process data and save similarity matrices to disk.
The Streamlit app then loads these instantly instead of recomputing every time.

Usage:
    python precompute.py

Outputs:
    data/items.pkl       — preprocessed items DataFrame
    data/sim_cache.pkl   — precomputed similarity matrices
"""

import os
import pickle
import time
from utils import load_and_preprocess, precompute_similarities

os.makedirs("data", exist_ok=True)

print("=" * 50)
print("  PRECOMPUTATION — run once, loads instantly after")
print("=" * 50)

t0 = time.time()
print("\n[1/2] Loading and preprocessing data…")
items = load_and_preprocess()
print(f"      Done — {len(items):,} items  ({time.time()-t0:.1f}s)")

t1 = time.time()
print("\n[2/2] Computing similarity matrices…")
sim_cache = precompute_similarities(items)
print(f"      Done  ({time.time()-t1:.1f}s)")

print("\nSaving to disk…")
with open("data/items.pkl", "wb") as f:
    pickle.dump(items, f)
with open("data/sim_cache.pkl", "wb") as f:
    pickle.dump(sim_cache, f)

total = time.time() - t0
print(f"\n✅  Saved to data/items.pkl and data/sim_cache.pkl")
print(f"    Total time: {total:.1f}s — app will now load in under 1 second.")
