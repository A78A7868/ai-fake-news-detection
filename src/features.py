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


def top_features(vectorizer, X, n: int = 20) -> list[str]:
    """
    Return the top n features ranked by their total frequency or weight in the matrix X.

    For a CountVectorizer, features are ranked by raw term frequency (sum of counts).
    For a TfidfVectorizer, features are ranked by summed TF-IDF weight across documents.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    # Sum along the document axis (axis=0) to get column totals
    sums = np.asarray(X.sum(axis=0)).flatten()
    # Sort indices in descending order
    top_indices = np.argsort(sums)[::-1][:n]
    return feature_names[top_indices].tolist()
