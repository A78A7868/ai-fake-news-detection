"""
evaluate.py - metrics, confusion matrices, and comparison plots.

Used in week4_evaluation.ipynb and the IEEE report generation.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

FIGURES_DIR = Path("results/figures")
METRICS_DIR = Path("results/metrics")


def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    """
    Compute accuracy, precision, recall, and F1 for a single model.
    Returns a dict that can be serialised to JSON.
    """
    return {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }


def save_metrics(metrics_list: list[dict], out_dir: str | Path = METRICS_DIR) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "all_models_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics_list, f, indent=2)
    print(f"Metrics saved to {path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_dir: str | Path = FIGURES_DIR) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    path = out / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_metrics_comparison(metrics_list: list[dict], out_dir: str | Path = FIGURES_DIR) -> None:
    """Bar chart comparing accuracy, precision, recall, and F1 across all models."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    models = [m["model"] for m in metrics_list]
    metric_names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metric_names):
        vals = [m[metric] for m in metrics_list]
        ax.bar(x + i * width, vals, width, label=metric.capitalize())

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - All Metrics")
    ax.legend()
    plt.tight_layout()

    path = out / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def mcnemar_test(y_true, y_pred1, y_pred2) -> dict:
    """
    Perform McNemar's test on paired predictions from two models.
    y_pred1: prediction array of model 1
    y_pred2: prediction array of model 2
    """
    from scipy.stats import chi2
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    # Contingency table cells:
    # b: Model 1 correct, Model 2 incorrect
    # c: Model 1 incorrect, Model 2 correct
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    
    # McNemar's chi-squared with Yates' continuity correction
    total_discordant = b + c
    if total_discordant > 0:
        stat = ((abs(b - c) - 1.0) ** 2) / total_discordant
        p_val = chi2.sf(stat, 1)
    else:
        stat = 0.0
        p_val = 1.0
        
    return {
        "b": int(b),
        "c": int(c),
        "statistic": float(stat),
        "p_value": float(p_val)
    }


def bootstrap_accuracy_ci(y_true, y_pred, n_bootstraps: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """
    Calculate the bootstrap confidence interval for model accuracy.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    
    rng = np.random.default_rng(42)
    boot_accuracies = []
    
    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        acc = np.mean(y_true[indices] == y_pred[indices])
        boot_accuracies.append(acc)
        
    boot_accuracies = np.sort(boot_accuracies)
    lower = boot_accuracies[int(n_bootstraps * (alpha / 2))]
    upper = boot_accuracies[int(n_bootstraps * (1.0 - alpha / 2))]
    
    return float(lower), float(upper)
