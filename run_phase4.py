"""
run_phase4.py - Comprehensive evaluation script including baseline evaluation,
hyperparameter comparison, statistical significance testing (McNemar's & Bootstrap),
and out-of-domain generalization testing on recent 2023+ news.
"""
import sys
import os
from pathlib import Path
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils import ensure_dirs
from src.models import load_model
from src.preprocessing import clean_text, strip_reuters_dateline
from src.evaluate import (
    compute_metrics,
    save_metrics,
    plot_confusion_matrix,
    plot_metrics_comparison,
    mcnemar_test,
    bootstrap_accuracy_ci
)

ensure_dirs("data/processed", "results/figures", "results/metrics")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120

# ── 1. Load Data ───────────────────────────────────────────────────────────
X_test_path = Path("data/processed/X_test_tfidf.joblib")
y_test_path = Path("data/processed/y_test.joblib")

if not X_test_path.exists() or not y_test_path.exists():
    raise FileNotFoundError("TF-IDF features not found. Run Phase 2 first.")

X_test = joblib.load(X_test_path)
y_test = joblib.load(y_test_path)
print(f"Loaded TF-IDF test matrix of shape: {X_test.shape}")

# Load TF-IDF Vectorizer for OOD testing
vectorizer_path = Path("data/processed/tfidf_vectorizer.joblib")
if not vectorizer_path.exists():
    raise FileNotFoundError("Vectorizer not found. Run Phase 2 first.")
vectorizer = joblib.load(vectorizer_path)

# ── 2. Load Models and Evaluate In-Domain Baselines ──────────────────────────
model_names = ["KNN", "LogReg", "RandomForest", "NeuralNet"]
metrics_list = []
predictions = {}

print("\n=== In-Domain Baseline Evaluation ===")
for name in model_names:
    model = load_model(name)
    print(f"Running inference for baseline {name}...")
    
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0
    
    predictions[name] = y_pred
    metrics = compute_metrics(y_test, y_pred, name)
    metrics["inference_time_s"] = round(elapsed, 4)
    metrics_list.append(metrics)
    
    # Save confusion matrix plot
    plot_confusion_matrix(y_test, y_pred, name)

# Save baseline metrics and plot comparison
save_metrics(metrics_list)
plot_metrics_comparison(metrics_list)

metrics_df = pd.DataFrame(metrics_list)
print("\n--- Baseline Performance Metrics ---")
print(metrics_df.to_string(index=False))

# ── 3. Evaluate Tuned Models ───────────────────────────────────────────────
print("\n=== Tuned vs. Baseline Performance ===")
tuned_metrics_list = []
tuned_predictions = {}

tuned_models_to_check = ["LogReg", "RandomForest"]
for name in tuned_models_to_check:
    tuned_model_path = Path(f"results/{name}_tuned.joblib")
    if tuned_model_path.exists():
        tuned_model = joblib.load(tuned_model_path)
        print(f"Running inference for tuned {name}...")
        
        y_pred = tuned_model.predict(X_test)
        tuned_predictions[name] = y_pred
        
        metrics = compute_metrics(y_test, y_pred, f"{name}_tuned")
        tuned_metrics_list.append(metrics)
    else:
        print(f"Warning: Tuned model {name} not found at {tuned_model_path}")

# Compare tuned vs baseline side-by-side
compare_rows = []
for name in tuned_models_to_check:
    # Baseline
    base_m = metrics_df[metrics_df["model"] == name].iloc[0]
    # Tuned
    tuned_m = [m for m in tuned_metrics_list if m["model"] == f"{name}_tuned"][0]
    
    # Load best params
    best_params = joblib.load(f"results/metrics/{name.lower()}_best_params.joblib")
    
    compare_rows.append({
        "Model": name,
        "Baseline Acc (%)": round(base_m["accuracy"] * 100, 2),
        "Tuned Acc (%)": round(tuned_m["accuracy"] * 100, 2),
        "Delta Acc (%)": round((tuned_m["accuracy"] - base_m["accuracy"]) * 100, 2),
        "Best Params": str(best_params)
    })

comparison_df = pd.DataFrame(compare_rows)
print("\n--- Hyperparameter Optimization Results ---")
print(comparison_df.to_string(index=False))

# ── 4. Statistical Significance Testing ─────────────────────────────────────
print("\n=== Statistical Significance Testing (RF vs. LogReg) ===")
rf_pred = predictions["RandomForest"]
lr_pred = predictions["LogReg"]

# McNemar's Test
mc_res = mcnemar_test(y_test, rf_pred, lr_pred)
print("\nMcNemar's Test Results:")
print(f"  Discordant cells (b - RF correct, LR incorrect): {mc_res['b']}")
print(f"  Discordant cells (c - RF incorrect, LR correct): {mc_res['c']}")
print(f"  Yates' Chi-squared Statistic: {mc_res['statistic']:.4f}")
print(f"  p-value: {mc_res['p_value']:.4e}")
if mc_res['p_value'] < 0.05:
    print("  Result is STATISTICALLY SIGNIFICANT (p < 0.05).")
else:
    print("  Result is NOT statistically significant (p >= 0.05).")

# Bootstrap CIs
print("\nBootstrap Confidence Intervals (Accuracy, 95% CI):")
for name in ["LogReg", "RandomForest"]:
    lower, upper = bootstrap_accuracy_ci(y_test, predictions[name])
    print(f"  {name}: {lower*100:.2f}% to {upper*100:.2f}%")

# Save statistical outputs for report/appendix
sig_stats = {
    "mcnemar": mc_res,
    "logreg_ci": bootstrap_accuracy_ci(y_test, lr_pred),
    "rf_ci": bootstrap_accuracy_ci(y_test, rf_pred)
}
joblib.dump(sig_stats, "results/metrics/statistical_significance.joblib")

# ── 5. Out-of-Domain Generalization Test ──────────────────────────────────
print("\n=== Out-of-Domain Generalization Testing ===")
ood_path = Path("data/raw/recent_news_2023.csv")
if ood_path.exists():
    ood_df = pd.read_csv(ood_path)
    print(f"Loaded out-of-domain dataset: {len(ood_df)} articles (2023+)")
    
    # Preprocess
    ood_df["text"] = ood_df["text"].fillna("").apply(strip_reuters_dateline)
    ood_df["text"] = ood_df["title"].fillna("") + " " + ood_df["text"]
    ood_df["clean_text"] = ood_df["text"].apply(clean_text)
    
    X_ood = vectorizer.transform(ood_df["clean_text"])
    y_ood = ood_df["label"]
    
    ood_results = []
    for name in model_names:
        model = load_model(name)
        y_pred_ood = model.predict(X_ood)
        metrics_ood = compute_metrics(y_ood, y_pred_ood, name)
        
        # Get corresponding in-domain test accuracy
        in_domain_acc = metrics_df[metrics_df["model"] == name].iloc[0]["accuracy"]
        
        ood_results.append({
            "Model": name,
            "In-Domain Acc (%)": round(in_domain_acc * 100, 2),
            "Out-of-Domain Acc (%)": round(metrics_ood["accuracy"] * 100, 2),
            "F1 (OOD)": round(metrics_ood["f1"], 4)
        })
        
    ood_results_df = pd.DataFrame(ood_results)
    print("\n--- Out-of-Domain Generalization Metrics (Smoke Test, n=10) ---")
    print(ood_results_df.to_string(index=False))
    
    # Save OOD metrics
    joblib.dump(ood_results, "results/metrics/out_of_domain_metrics.joblib")
else:
    print("Warning: Out-of-domain test file not found at data/raw/recent_news_2023.csv")

print("\nEvaluation metrics, plots, and significance tests saved successfully.")
