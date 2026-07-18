"""
Quick script to run all Phase 2 logic, serialize features, and print summary stats.
Run after Phase 1 is done: python3 run_phase2.py
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils import ensure_dirs
from src.features import get_bow_vectorizer, get_tfidf_vectorizer, fit_transform

ensure_dirs("data/processed", "results/figures", "results/metrics")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── 1. Load ────────────────────────────────────────────────────────────────
data_path = Path("data/processed/news_cleaned.csv")
if not data_path.exists():
    raise FileNotFoundError(f"Cleaned dataset not found at {data_path}. Run Phase 1 first.")

df = pd.read_csv(data_path)
df["clean_text"] = df["clean_text"].fillna("")
print(f"Loaded dataset: {len(df):,} rows")

# ── 2. Split ───────────────────────────────────────────────────────────────
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,}")
print(f"Test set:     {len(X_test):,}")

# ── 3. BoW ─────────────────────────────────────────────────────────────────
bow_vec = get_bow_vectorizer(max_features=5000)
X_train_bow, X_test_bow, bow_vec = fit_transform(bow_vec, X_train, X_test)

# ── 4. TF-IDF ──────────────────────────────────────────────────────────────
tfidf_vec = get_tfidf_vectorizer(max_features=5000)
X_train_tfidf, X_test_tfidf, tfidf_vec = fit_transform(tfidf_vec, X_train, X_test)

# ── 5. Sparsity ────────────────────────────────────────────────────────────
def get_sparsity_stats(matrix):
    total_elements = matrix.shape[0] * matrix.shape[1]
    non_zero = matrix.nnz
    sparsity = 1.0 - (non_zero / total_elements)
    return non_zero, total_elements, sparsity

bow_nz, bow_tot, bow_sp = get_sparsity_stats(X_train_bow)
tfidf_nz, tfidf_tot, tfidf_sp = get_sparsity_stats(X_train_tfidf)

print(f"\nBag-of-Words (Train):\n  Sparsity: {bow_sp*100:.4f}% ({bow_nz:,} / {bow_tot:,})")
print(f"TF-IDF (Train):\n  Sparsity: {tfidf_sp*100:.4f}% ({tfidf_nz:,} / {tfidf_tot:,})")

# ── 6. Top features per class ──────────────────────────────────────────────
feature_names = np.array(tfidf_vec.get_feature_names_out())

def get_top_tfidf_terms(X_matrix, feature_names, labels, class_val, n=15):
    class_indices = np.where(labels.values == class_val)[0]
    class_subset = X_matrix[class_indices]
    mean_weights = np.asarray(class_subset.mean(axis=0)).flatten()
    top_idx = mean_weights.argsort()[::-1][:n]
    return pd.DataFrame({
        "term": feature_names[top_idx],
        "weight": mean_weights[top_idx]
    })

fake_top = get_top_tfidf_terms(X_train_tfidf, feature_names, y_train, class_val=0)
real_top = get_top_tfidf_terms(X_train_tfidf, feature_names, y_train, class_val=1)

print("\n=== Top 10 TF-IDF in Fake ===")
print(fake_top.head(10).to_string(index=False))
print("\n=== Top 10 TF-IDF in Real ===")
print(real_top.head(10).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(data=fake_top, x="weight", y="term", ax=axes[0], color="#e05c5c")
axes[0].set_title("Top TF-IDF Weights — Fake News")
axes[0].set_xlabel("Mean TF-IDF Weight")

sns.barplot(data=real_top, x="weight", y="term", ax=axes[1], color="#5c8ee0")
axes[1].set_title("Top TF-IDF Weights — Real News")
axes[1].set_xlabel("Mean TF-IDF Weight")

plt.tight_layout()
plt.savefig("results/figures/top_tfidf_weights.png", dpi=150)
plt.close()
print("\nSaved: results/figures/top_tfidf_weights.png")

# ── 7. Save sets ───────────────────────────────────────────────────────────
joblib.dump(y_train, "data/processed/y_train.joblib")
joblib.dump(y_test, "data/processed/y_test.joblib")

joblib.dump(X_train_bow, "data/processed/X_train_bow.joblib")
joblib.dump(X_test_bow, "data/processed/X_test_bow.joblib")
joblib.dump(bow_vec, "data/processed/bow_vectorizer.joblib")

joblib.dump(X_train_tfidf, "data/processed/X_train_tfidf.joblib")
joblib.dump(X_test_tfidf, "data/processed/X_test_tfidf.joblib")
joblib.dump(tfidf_vec, "data/processed/tfidf_vectorizer.joblib")

stats = {
    "train_size": len(X_train),
    "test_size": len(X_test),
    "max_features": 5000,
    "bow_sparsity": float(bow_sp),
    "tfidf_sparsity": float(tfidf_sp)
}
joblib.dump(stats, "results/metrics/phase2_stats.joblib")
print("Saved features and stats dictionary.")
