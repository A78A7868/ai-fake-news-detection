"""
Quick script to run all Phase 1 logic and print the summary stats.
Run after venv is ready: python3 run_phase1.py
"""
import sys
import os
from pathlib import Path

# Make sure src/ is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils import check_raw_data, ensure_dirs
from src.preprocessing import clean_text, build_corpus, strip_reuters_dateline

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

ensure_dirs("data/processed", "results/figures", "results/metrics", "results/notebook_exports")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── 1. Load ────────────────────────────────────────────────────────────────
fake_path, true_path = check_raw_data("data/raw")
fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)
print(f"Fake articles : {len(fake_df):,}")
print(f"Real articles : {len(true_df):,}")
print(f"Total raw     : {len(fake_df) + len(true_df):,}")
print()
print("Fake.csv columns:", fake_df.columns.tolist())
print("True.csv columns:", true_df.columns.tolist())

# ── 2. Merge & label ───────────────────────────────────────────────────────
fake_df["label"] = 0
true_df["label"] = 1
df = pd.concat([fake_df, true_df], ignore_index=True)
# Clean datelines from body text before concatenating with the title
df["text"] = df["text"].fillna("").apply(strip_reuters_dateline)
df["text"] = df["title"].fillna("") + " " + df["text"]
df["text"] = df["text"].str.strip()

# ── 3. Quality checks ──────────────────────────────────────────────────────
print(f"\nNull counts:\n{df.isnull().sum()}")
n_dupes = df.duplicated(subset="text").sum()
print(f"\nDuplicate rows: {n_dupes}")
if n_dupes > 0:
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

empty_mask = df["text"].str.strip().eq("")
print(f"Empty-text rows: {empty_mask.sum()}")
df = df[~empty_mask].reset_index(drop=True)

# ── 4. EDA — class balance ─────────────────────────────────────────────────
cc = df["label"].value_counts()
print(f"\nClass balance:")
print(f"  Fake (0): {cc[0]:,}  ({100*cc[0]/len(df):.1f}%)")
print(f"  Real (1): {cc[1]:,}  ({100*cc[1]/len(df):.1f}%)")

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Fake", "Real"], [cc[0], cc[1]], color=["#e05c5c", "#5c8ee0"])
ax.set_ylabel("Article count")
ax.set_title("Class Balance: Fake vs. Real")
for i, v in enumerate([cc[0], cc[1]]):
    ax.text(i, v + 100, f"{v:,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/class_balance.png", dpi=150)
plt.close()
print("Saved: results/figures/class_balance.png")

# ── 5. Article length ──────────────────────────────────────────────────────
df["word_count_raw"] = df["text"].str.split().apply(len)
print(f"\nWord count stats (raw):")
print(df.groupby("label")["word_count_raw"].describe().round(1))

fig, ax = plt.subplots(figsize=(8, 4))
for label, color, name in [(0, "#e05c5c", "Fake"), (1, "#5c8ee0", "Real")]:
    ax.hist(df[df["label"]==label]["word_count_raw"].clip(upper=2000),
            bins=60, alpha=0.6, color=color, label=name)
ax.set_xlabel("Word count (clipped at 2000)")
ax.set_ylabel("Number of articles")
ax.set_title("Article Length Distribution")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/length_distribution.png", dpi=150)
plt.close()
print("Saved: results/figures/length_distribution.png")

# ── 6. Clean corpus ────────────────────────────────────────────────────────
print("\nCleaning corpus (may take ~1–2 min)...")
df["clean_text"] = build_corpus(df, text_col="text")
df["word_count_clean"] = df["clean_text"].str.split().apply(len)

empty_after = df["clean_text"].str.strip().eq("")
print(f"Empty after cleaning: {empty_after.sum()}")
df = df[~empty_after].reset_index(drop=True)
print(f"Final dataset size: {len(df):,}")

# ── 7. Top words per class ─────────────────────────────────────────────────
def top_words(series, n=20):
    tokens = []
    for t in series:
        tokens.extend(t.split())
    return Counter(tokens).most_common(n)

fake_top = top_words(df[df["label"]==0]["clean_text"])
real_top = top_words(df[df["label"]==1]["clean_text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, top, title, color in [
    (axes[0], fake_top, "Top 20 Words — Fake", "#e05c5c"),
    (axes[1], real_top, "Top 20 Words — Real", "#5c8ee0"),
]:
    words, counts = zip(*top)
    ax.barh(words[::-1], counts[::-1], color=color)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig("results/figures/top_words_per_class.png", dpi=150)
plt.close()
print("Saved: results/figures/top_words_per_class.png")

# ── 8. Subject distribution ────────────────────────────────────────────────
if "subject" in df.columns:
    print(f"\nSubject distribution:\n{df['subject'].value_counts()}")
    fig, ax = plt.subplots(figsize=(9, 4))
    df["subject"].value_counts().plot(kind="bar", ax=ax, color="#7b7bc8")
    ax.set_xlabel("Subject category")
    ax.set_ylabel("Article count")
    ax.set_title("Article Counts by Subject Category")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig("results/figures/subject_distribution.png", dpi=150)
    plt.close()
    print("Saved: results/figures/subject_distribution.png")

# ── 9. Save processed CSV ──────────────────────────────────────────────────
out_cols = (["text", "clean_text", "label", "subject"]
            if "subject" in df.columns else ["text", "clean_text", "label"])
df[out_cols].to_csv("data/processed/news_cleaned.csv", index=False)
print(f"\nSaved: data/processed/news_cleaned.csv")

# ── 10. Summary ────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("PHASE 1 SUMMARY")
print("=" * 50)
print(f"  Raw fake articles      : {len(fake_df):,}")
print(f"  Raw real articles      : {len(true_df):,}")
print(f"  Total raw              : {len(fake_df)+len(true_df):,}")
print(f"  Final (after cleaning) : {len(df):,}")
print(f"  Fake % (final)         : {100*(df['label']==0).sum()/len(df):.1f}%")
print(f"  Real % (final)         : {100*(df['label']==1).sum()/len(df):.1f}%")
print(f"  Avg words/article(raw) : {df['word_count_raw'].mean():.0f}")
print(f"  Avg words/art.(clean)  : {df['word_count_clean'].mean():.0f}")
print("=" * 50)
