"""
Quick script to run Phase 3 training headlessly inside the activated environment.
"""
import sys
import os
from pathlib import Path
import time
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils import ensure_dirs
from src.models import get_models, train_all, save_models, tune_hyperparameters

ensure_dirs("data/processed", "results/figures", "results/metrics")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── 1. Load Data ───────────────────────────────────────────────────────────
X_train_path = Path("data/processed/X_train_tfidf.joblib")
y_train_path = Path("data/processed/y_train.joblib")

if not X_train_path.exists() or not y_train_path.exists():
    raise FileNotFoundError("TF-IDF features not found. Run Phase 2 first.")

X_train = joblib.load(X_train_path)
y_train = joblib.load(y_train_path)
print(f"Loaded TF-IDF training matrix of shape: {X_train.shape}")

# ── 2. Train Models ────────────────────────────────────────────────────────
models = get_models()
print("Starting training...")
fitted_models = train_all(models, X_train, y_train)
print("Training complete.")

# ── 3. Time Comparison ─────────────────────────────────────────────────────
times_dict = {name: info[1] for name, info in fitted_models.items()}
times_df = pd.DataFrame(list(times_dict.items()), columns=["Model", "Training Time (s)"])
print("\n=== Training Times ===")
print(times_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=times_df, x="Model", y="Training Time (s)", ax=ax, hue="Model", palette="viridis", legend=False)
ax.set_title("Training Time Comparison")
ax.set_ylabel("Time (seconds)")
for i, v in enumerate(times_df["Training Time (s)"]):
    ax.text(i, v + (max(times_df["Training Time (s)"]) * 0.01), f"{v:.2f}s", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("results/figures/training_times.png", dpi=150)
plt.close()
print("\nSaved: results/figures/training_times.png")

# ── 4. Serialize ───────────────────────────────────────────────────────────
save_models(fitted_models)
joblib.dump(times_dict, "results/metrics/phase3_training_times.joblib")
print("Saved all baseline models and metrics.")

# ── 5. Hyperparameter Tuning ────────────────────────────────────────────────
tuned_info = tune_hyperparameters(X_train, y_train)
joblib.dump(tuned_info["LogReg"], "results/LogReg_tuned.joblib")
joblib.dump(tuned_info["RandomForest"], "results/RandomForest_tuned.joblib")
joblib.dump(tuned_info["LogReg_params"], "results/metrics/logreg_best_params.joblib")
joblib.dump(tuned_info["RandomForest_params"], "results/metrics/randomforest_best_params.joblib")
print("Saved tuned models and parameters successfully.")
