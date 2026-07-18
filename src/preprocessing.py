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

# Regex to match Reuters datelines like 'washington (reuters) -' at the start of text
_REUTERS_DATELINE_RE = re.compile(r"^([a-z0-9\s,./#&-]{1,50})?\s*\(reuters\)\s*[-–—\s]*")


def strip_reuters_dateline(text: str) -> str:
    """
    Strips Reuters dateline (e.g. 'washington (reuters) -') from the start of the lowercased text.
    """
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    return _REUTERS_DATELINE_RE.sub("", text_lower)


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline applied to a single article string.

    Steps (order matters):
      1. Lowercase and strip Reuters datelines
      2. Strip URLs
      3. Strip punctuation and digits - keep only letter characters and spaces
      4. Tokenize via regex word-boundary match (manual, no library tokenizer)
      5. Remove stopwords using NLTK's English list
      6. Rejoin tokens into a single cleaned string for vectorization

    Returns the cleaned string. Returns "" for null/non-string input.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Strip Reuters dateline from the beginning
    text = strip_reuters_dateline(text)
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
