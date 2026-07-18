"""
models.py - training wrappers for all four classifiers.

Hyperparameters match the brief's own Python skeleton exactly as the baseline.
Any deviations from those defaults will be documented in docs/phase3_model_report.md.
"""

import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

RESULTS_DIR = Path("results")


def get_models() -> dict:
    """
    Return a dict of model name -> unfitted estimator, using the brief's baseline
    hyperparameters. random_state is fixed at 42 where applicable.
    """
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
    }


def train_all(models: dict, X_train, y_train) -> dict:
    """
    Fit each model. Returns a dict of name -> (fitted_model, training_time_seconds).
    Prints a one-line status for each model.
    """
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}...", end=" ", flush=True)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        fitted[name] = (model, elapsed)
        print(f"done ({elapsed:.1f}s)")
    return fitted


def save_models(fitted: dict, out_dir: str | Path = RESULTS_DIR) -> None:
    """Serialize each fitted model to results/<name>.joblib."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, (model, _) in fitted.items():
        path = out / f"{name}.joblib"
        joblib.dump(model, path)
        print(f"Saved: {path}")


def load_model(name: str, out_dir: str | Path = RESULTS_DIR):
    """Load a previously saved model from disk."""
    path = Path(out_dir) / f"{name}.joblib"
    return joblib.load(path)


def tune_hyperparameters(X_train, y_train) -> dict:
    """
    Perform Grid Search Cross-Validation for Logistic Regression and Random Forest.
    Returns a dict containing best estimators and their best parameters.
    """
    from sklearn.model_selection import GridSearchCV
    
    # 1. Logistic Regression tuning
    logreg = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
    logreg_param_grid = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"]
    }
    print("Tuning Logistic Regression hyper-parameters using 3-fold CV...")
    t0 = time.perf_counter()
    grid_logreg = GridSearchCV(logreg, logreg_param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_logreg.fit(X_train, y_train)
    logreg_time = time.perf_counter() - t0
    print(f"LogReg tuning complete in {logreg_time:.1f}s. Best params: {grid_logreg.best_params_}")
    
    # 2. Random Forest tuning
    rf = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 20, None]
    }
    print("Tuning Random Forest hyper-parameters using 3-fold CV...")
    t0 = time.perf_counter()
    grid_rf = GridSearchCV(rf, rf_param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    rf_time = time.perf_counter() - t0
    print(f"Random Forest tuning complete in {rf_time:.1f}s. Best params: {grid_rf.best_params_}")
    
    return {
        "LogReg": grid_logreg.best_estimator_,
        "RandomForest": grid_rf.best_estimator_,
        "LogReg_params": grid_logreg.best_params_,
        "RandomForest_params": grid_rf.best_params_
    }
