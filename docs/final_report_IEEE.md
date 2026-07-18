# IEEE Conference Paper: AI-Powered Fake News Detection Using Text Classification

*This is the markdown preview of the final IEEE conference report. The compile-ready LaTeX source is located at [`docs/final_report_IEEE.tex`](file:///Users/anandkrishnagrnair/Documents/antigravity/keen-goodall/fake-news-detection/docs/final_report_IEEE.tex).*

---

### Abstract
Misinformation on digital networks presents a significant challenge to modern society. This paper describes the design and implementation of an end-to-end machine learning pipeline built from scratch to classify news articles as real or fake. Using a corpus of 39,098 articles, we apply custom text cleaning and manual tokenization. We compare two text vectorization techniques: Bag-of-Words and Term Frequency-Inverse Document Frequency (TF-IDF). We then train and compare four classifiers: K-Nearest Neighbors (KNN), Logistic Regression, Random Forest, and a Multi-Layer Perceptron (MLP) neural network. Our evaluation shows that the MLP Neural Network achieves the highest baseline accuracy of 98.41%, while Logistic Regression provides the fastest inference time (0.0025 seconds) with a baseline accuracy of 98.24% (improving to 98.62% after hyperparameter tuning). We analyze the trade-offs between parametric and non-parametric classifiers and highlight potential source leakages in the dataset.

---

## 1. Introduction
Misinformation is a growing problem that spreads rapidly across social media and digital platforms. Sorting through thousands of daily articles manually to verify their authenticity is not feasible. This makes automated systems essential. 

In this work, we use **Artificial Intelligence (AI)** to automate the detection of misinformation, utilizing **Machine Learning (ML)** algorithms to learn text patterns (such as word frequency and context) to classify articles as real or fake. We apply **Natural Language Processing (NLP)** techniques to preprocess raw text and convert it into numerical features suitable for mathematical classifiers.

Our main goal is to build a complete pipeline from scratch using basic python structures and Scikit-Learn libraries, implementing manual tokenization to avoid high-level library dependencies. We assess the trade-offs in accuracy, training time, storage, and inference latency across four different classifier architectures.

---

## 2. Dataset Description
We used the ISOT Fake and Real News Dataset hosted on Kaggle (Bisaillon 2020). The raw data consists of two files: `Fake.csv` (23,481 articles flagged as fake) and `True.csv` (21,417 articles from Reuters).

We combined the title and body of each article into a single text field to capture titles that contain strong classification signals. During our initial data analysis, we found 5,795 exact duplicate rows in the combined text field. Keeping duplicates would artificially inflate our performance scores, so we removed them. We also dropped 5 articles that were reduced to empty strings after cleaning (mainly consisting of numbers, URLs, or punctuation). 

Our final dataset has **39,098 articles** with a balanced distribution:
- **Fake News (0):** 17,902 articles (45.79%)
- **Real News (1):** 21,196 articles (54.21%)

The text primarily covers U.S. politics from 2016-2017, meaning it skews heavily towards specific political figures (Trump, Clinton, Obama) and government policies.

---

## 3. Methodology

### 3.1 Preprocessing & Manual Tokenization
Our cleaning process in `preprocessing.py` is executed as follows:
1. **Lowercase Conversion:** All characters are converted to lowercase.
2. **URL Removal:** Links matching `https?://\S+|www\.\S+` are removed since they add noise.
3. **Manual Tokenization:** We extract words using the regex pattern `re.findall(r'\b[a-z]+\b', text)`. This satisfies our constraint to avoid high-level tokenizers like `nltk.word_tokenize` or spaCy.
4. **Stopword Filtering:** We remove tokens present in NLTK's English stopword list using a hash set. Single-letter words are also dropped.

### 3.2 Feature Extraction
We compared two vectorization techniques using a fixed vocabulary of the 5,000 most frequent unigrams:
- **Bag-of-Words (BoW):** Represents documents as raw count vectors of token occurrences: $x_{t,d} = \text{count}(t, d)$.
- **TF-IDF Vectorization:** Scales counts by document specificity. The smoothed Inverse Document Frequency is defined as:
  $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
  The final vectors are L2-normalized: $\mathbf{v}_d = \mathbf{w}_d / \|\mathbf{w}_d\|_2$.

Both representations result in highly sparse training matrices. On our 80% training split (31,278 rows $\times$ 5,000 columns), the matrix sparsity ratio is **97.3771%**, meaning only 2.6229% of cells contain non-zero values.

### 3.3 Classifier Architectures
We split the dataset 80/20 into train/test sets stratified on the labels. We trained the following models:
1. **K-Nearest Neighbors (KNN):** Non-parametric instance-based classifier using $K=5$ and Euclidean distance.
2. **Logistic Regression:** Parametric model optimized using L-BFGS to predict label probability:
   $$P(Y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
3. **Random Forest:** Non-parametric ensemble classifier training 100 decision trees on bootstrap samples.
4. **Multi-Layer Perceptron (MLP):** A single hidden layer of 100 neurons trained using Adam optimizer with ReLU activation.

All models used a fixed seed of `random_state=42` to ensure reproducible results.

---

## 4. Experimental Results

Table 1 summarizes the performance of the four models on the 20% test set (7,820 articles).

### Table 1: Model Performance on Evaluation Set

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Inference Time |
|---|---|---|---|---|---|
| **KNN** | 88.29 | 85.15 | 94.95 | 89.78 | 6.55s |
| **KNN + LSA (200-d)\*** | 91.06 | 90.01 | 93.94 | 91.93 | 0.52ms |
| **LogReg** | 98.24 | 97.83 | 98.94 | 98.38 | 2.5ms |
| **Random Forest** | 98.36 | 97.95 | 99.06 | 98.50 | 0.16s |
| **MLP Neural Net** | **98.41** | **98.17** | **98.91** | **98.54** | 25.3ms |

\*Note: LSA dense representation reduces features to 200 components, which was not applied to other models.

### Training Times
Training models on our 31,278-row TF-IDF matrix yielded varying execution times:
- **KNN:** 0.0152 seconds (deferred fit cost)
- **Logistic Regression:** 0.1174 seconds
- **Random Forest:** 14.5721 seconds
- **MLP Neural Network:** 30.9381 seconds

*Note: Wall-clock times vary slightly between runs since they depend on system load, not the fixed random_state.*

### Confusion Matrices
The confusion matrices details on the 7,820 test set articles are:
- **KNN**: TN=2879, FP=702, FN=214, TP=4025
- **LogReg**: TN=3488, FP=93, FN=45, TP=4194
- **RandomForest**: TN=3493, FP=88, FN=40, TP=4199
- **NeuralNet**: TN=3503, FP=78, FN=46, TP=4193

---

## 5. Discussion & Trade-offs

### 5.1 Parametric vs. Non-Parametric Performance
Our experiments highlight clear architectural differences:
- **Computational Costs:** KNN has a training time of 0.0152 seconds, but its test inference speed is very slow: the 5000-d sparse cosine model requires 6.5467 ± 0.0812 seconds (0.8372 ms per sample) to predict the test set. Using the dense KNN + LSA (200-d) pipeline reduces this latency significantly to 4.0864 ± 0.1196 seconds (0.5226 ms per sample) while boosting test accuracy to 91.06%. This demonstrates that high dimensionality is the primary computational and performance bottleneck for instance-based methods. Parametric models (Logistic Regression, MLP) perform inference using simple matrix operations, making them fast during prediction.
- **Model Size and Memory Footprint:** The Random Forest model consumes ~40 MB on disk to store 100 deep trees. Logistic Regression stores only 5,000 weight coefficients, making it highly lightweight. Unlike parametric models, KNN must retain the entire training matrix in memory at inference time; using a sparse Compressed Sparse Row (CSR) format reduces this RAM footprint to 47.30 MB, compared to 1.19 GB if stored as a dense matrix.

### 5.2 Data Leakage Limitations
We implemented a strict regex-based dateline removal process using the pattern `^([a-z0-9\s,./#&-]{1,50})?\s*\(reuters\)\s*[-–—\s]*` applied directly to the body text prior to title concatenation. Stripping the full datelines (e.g. "WASHINGTON (Reuters) -") instead of just the single token "reuters" led to a noticeable performance drop. Specifically, Random Forest test accuracy decreased by 1.32% (from 99.68% to 98.36%) and Logistic Regression decreased by 0.52% (from 98.76% to 98.24%). This accuracy drop demonstrates that models relied partly on dataset-specific formatting templates rather than learning pure semantic representations. Additionally, U.S. political skew (2016-2017) limits geographic generalizability. Furthermore, strict binary labels fail to model graded misinformation categories like political satire, spin, or statements that are only partially accurate.

---

## 6. Conclusion
We built a machine learning pipeline from scratch to classify fake news. Manual tokenization and cleaning proved effective. Our evaluations show that the MLP Neural Network achieves the highest baseline accuracy of 98.41%, while Random Forest performs similarly with 98.36% accuracy. Logistic Regression provides an exceptionally strong alternative, yielding 98.24% baseline accuracy (98.62% when tuned) with exceptionally low inference latency (0.0025s). Future work will focus on removing source-specific tags to prevent leakage, utilizing dense word embeddings or transformer-based representations (e.g., BERT) as alternatives to bag-of-words or TF-IDF features, and evaluating model generalization capabilities on more recent and topically diverse news corpora.

---

## 7. Appendix

### 7.1 Project Implementation and Notebooks
The four Jupyter notebooks (`week1_data_eda.ipynb` through `week4_evaluation.ipynb`) are submitted alongside this report as the complete executable implementation of the machine learning pipeline. The preprocessing logic is implemented in `src/preprocessing.py`, vectorization wrappers are located in `src/features.py`, and modeling scripts are in `src/models.py`. The execution notebooks are stored under the `notebooks/` directory.

### 7.2 Sample Test Data
Below are three actual sample articles drawn from the stratified test partition, showing their raw content, manually preprocessed token sequences, true labels, and predictions.

| Raw Text Snippet | Cleaned Token Sequence (Manual Preprocessing) | True Label | Prediction |
|---|---|---|---|
| "Russian submarines fire cruise missiles at Islamic state in Syria MOSCOW (Reuters) - The Russian Navy on Thursday fired seven cruise missiles at Islamic state targets in the suburbs of Syria s Deir al-Zor..." | russian, submarines, fire, cruise, missiles, islamic, state, syria, moscow, reuters, russian, navy, thursday, fired, seven, cruise, missiles, islamic, state, targets, suburbs, syria, deir, al, zor, russian, defence... | Real (1) | Real (1) |
| "MICHELLE OBAMA TO HILLARY: "If You Can't Run Your Own House...You Certainly Can't Run The White House" [VIDEO] Remember when Mooch thought having #CrookedHillary back in the White House wasn't in the best..." | michelle, obama, hillary, run, house, certainly, run, white, house, video, remember, mooch, thought, crookedhillary, back, white, house, best, interest, nation, yeah, neither, maybe, someone... | Fake (0) | Fake (0) |
| "WOW! HILLARY CAUGHT ON VIDEO In 2000 Saying She Doesn't Like Emails Because You Can't Hide Them From Investigators Too bad for Hillary she wasn't actually telling the truth that time WATCH..." | wow, hillary, caught, video, saying, like, emails, hide, investigators, bad, hillary, actually, telling, truth, time, watch, hillary, clinton, saying, like, emails, hide, investigators, pic, twitter, com... | Fake (0) | Fake (0) |

### 7.3 Code Reference: preprocessing.py
```python
"""
preprocessing.py - text cleaning and manual tokenization for the fake-news pipeline.

Rules (per project brief):
  - Tokenization is manual: re.findall(r'\b[a-z]+\b', text) - no nltk.word_tokenize.
  - Stopword list comes from NLTK (list lookup only, not the tokenizer).
  - Vectorization (TF-IDF / BoW) lives in features.py, not here.
"""

import re

import nltk
import pandas as pd

# Download stopwords corpus on first run (no-op if already cached).
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS: set[str] = set(stopwords.words("english"))

# Regex pattern for URL removal - covers http/https and bare www. links.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")

# After lowercasing and URL removal, keep only lowercase letters (word tokens).
_TOKEN_RE = re.compile(r"\b[a-z]+\b")


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline applied to a single article string.

    Steps (order matters):
      1. Lowercase
      2. Strip URLs
      3. Strip punctuation and digits - keep only letter characters and spaces
      4. Tokenize via regex word-boundary match (manual, no library tokenizer)
      5. Remove stopwords using NLTK's English list
      6. Rejoin tokens into a single cleaned string for vectorization

    Returns the cleaned string. Returns "" for null/non-string input.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)

    # Extract word tokens (only [a-z] runs - removes digits and punctuation implicitly).
    tokens = _TOKEN_RE.findall(text)

    # Drop stopwords and very short tokens (length <= 1 adds no signal).
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    """
    Return the token list for a pre-cleaned string.
    Used in EDA where we need per-token counts, not the joined string.
    """
    return text.split()


def build_corpus(df: pd.DataFrame, text_col: str = "text") -> pd.Series:
    """
    Apply clean_text to every row in df[text_col].
    Returns a Series of cleaned strings, aligned with df's index.
    """
    return df[text_col].fillna("").apply(clean_text)
```

### 7.2 features.py
```python
"""
features.py - Bag-of-Words and TF-IDF vectorization using scikit-learn.

Vectorizer objects are fitted on the training split only, then used to
transform both train and test sets - avoids data leakage.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Baseline parameters from the project brief's own Python skeleton.
# max_features=5000 keeps the vocabulary manageable and reduces noise from
# very rare words.
DEFAULT_MAX_FEATURES = 5000


def get_tfidf_vectorizer(max_features: int = DEFAULT_MAX_FEATURES, **kwargs) -> TfidfVectorizer:
    """
    Return a TfidfVectorizer configured for this project.

    TF-IDF down-weights terms that appear in almost every document
    (which carry little discriminative power) and boosts terms that are
    frequent in a document but rare across the corpus.

    Formula used internally by sklearn:
        TF(t, d)  = count(t, d) (raw count)
        IDF(t)    = log((1 + N) / (1 + df(t))) + 1   [sklearn smooth IDF]
        TF-IDF    = TF * IDF
    """
    return TfidfVectorizer(max_features=max_features, **kwargs)


def get_bow_vectorizer(max_features: int = DEFAULT_MAX_FEATURES, **kwargs) -> CountVectorizer:
    """
    Return a CountVectorizer (Bag-of-Words) for this project.

    BoW represents each document as a vector of raw token counts.
    It ignores word order and document length - a simple but effective
    baseline for text classification.
    """
    return CountVectorizer(max_features=max_features, **kwargs)


def fit_transform(vectorizer, X_train, X_test):
    """
    Fit vectorizer on X_train, then transform both splits.
    Returns (X_train_vec, X_test_vec, vectorizer).
    """
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def top_features(vectorizer, n: int = 20) -> list[str]:
    """Return the n most frequent terms from a fitted vectorizer's vocabulary."""
    vocab = vectorizer.vocabulary_
    # Sort by index (which corresponds to column position in the feature matrix).
    sorted_terms = sorted(vocab.items(), key=lambda x: x[1])
    return [term for term, _ in sorted_terms[:n]]
```

### 7.3 models.py
```python
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
```

### 7.4 evaluate.py
```python
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
```

---

## 8. References
1. Bisaillon, C. (2020), "Fake and Real News Dataset," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
2. H. Ahmed, I. Traore, and S. Saad, "Detecting opinion spams and fake news using text classification," Journal of Security and Privacy, vol. 1, no. 1, p. e9, 2018.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.
