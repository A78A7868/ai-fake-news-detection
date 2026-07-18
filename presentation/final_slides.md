# Project Presentation: AI-Powered Fake News Detection

*This is the slide deck outline for the final presentation. You can copy these slides directly into PowerPoint, Google Slides, or compile them using Marp (Markdown Presentation Writer).*

---

## Slide 1: Title Slide
*   **Project Title:** AI-Powered Fake News Detection Using Text Classification
*   **Course:** Summer Internship Program in AI & ML, 2026 (Project - 1)
*   **Author:** Anand Krishna G R Nair
*   **Institution:** Indian Institute of Computing and Technology (IICT)

---

## Slide 2: Project Overview & Objectives
*   **Goal:** Build an end-to-end machine learning pipeline from scratch to classify articles as real or fake.
*   **Constraints:**
    *   No pretrained APIs or AutoML classifiers.
    *   Manual tokenization using regex word-boundary split.
    *   Compare Bag-of-Words and TF-IDF representations.
    *   Compare 4 distinct classifier types: KNN, Logistic Regression, Random Forest, MLP Neural Network.
    *   Adhere to strict IEEE reporting standards.

---

## Slide 3: Dataset Summary
*   **Source:** ISOT Fake and Real News Dataset (Kaggle)
*   **Preprocessing Ingestion:**
    *   Merged Title + Body into a single `text` field.
    *   Removed **5,795 duplicate rows** to prevent metric inflation.
    *   Dropped rows reduced to empty strings post-cleaning.
*   **Final Corpus Size:** 39,098 articles
*   **Distribution:** 45.79% Fake / 54.21% Real (reasonably balanced)

---

## Slide 4: Data Preprocessing Pipeline
*   **Manual Text Cleaning (`src/preprocessing.py`):**
    1.  Convert to lowercase.
    2.  Strip URLs via regex (`https?://\S+|www\.\S+`).
    3.  Tokenize manually using regex: `re.findall(r'\b[a-z]+\b', text)`. No library tokenizers used.
    4.  Remove words matching NLTK's English stopword list.
    5.  Filter out single-letter tokens.

---

## Slide 5: Feature Engineering & Sparsity
*   **Feature Extraction (`src/features.py`):**
    *   Compared Bag-of-Words and TF-IDF (L2-normalized).
    *   Restricted vocabulary to the top **5,000 unigrams**.
*   **Matrix Sparsity:**
    *   Matrix shape: $31,278 \times 5,000$ columns.
    *   Sparsity Ratio: **97.3640%** (CSR sparse representation required to save memory).
*   **Key Observations:**
    *   Common words across both classesSkews towards politics (`trump`, `clinton`, `obama`).
    *   *Real news* includes strong source signals like `reuters`.

---

## Slide 6: Classifier Training & Speeds
*   **Model Building (`src/models.py`):**
    *   Stratified 80/20 train/test split.
    *   Fixed seed `random_state=42` throughout.
*   **Training Time Comparison (31,278 documents):**
    *   **KNN ($K=5$):** 0.02 seconds (deferred fit cost).
    *   **Logistic Regression:** 0.18 seconds.
    *   **Random Forest (100 trees):** 14.62 seconds.
    *   **MLP Neural Network:** 26.21 seconds.

---

## Slide 7: Model Evaluation (Quantitative Results)
*   **Testing Results (7,820 documents):**
    *   **KNN:** 87.80% Accuracy | 89.38% F1-Score
    *   **Logistic Regression:** 98.76% Accuracy | 98.86% F1-Score
    *   **Neural Network (MLP):** 98.87% Accuracy | 98.96% F1-Score
    *   **Random Forest:** **99.68% Accuracy** | **99.71% F1-Score**
*   **Summary:** Ensemble and linear models easily capture the text splits on TF-IDF features.

---

## Slide 8: Parametric vs. Non-Parametric Trade-offs
*   **Parametric Models (LogReg, MLP):**
    *   *Pros:* Extremely fast inference (LogReg takes 0.0025s), lightweight storage (LogReg stores 5,000 float weights).
    *   *Cons:* Assume fixed linear boundaries (LogReg).
*   **Non-Parametric Models (KNN, Random Forest):**
    *   *Pros:* Random Forest achieves the highest accuracy (99.68%) by reducing variance.
    *   *Cons:* Large memory footprint (Random Forest size is ~40 MB), slow query times (KNN takes 6.38s for inference).

---

## Slide 9: Critical Analysis: Limitations & Leakage
*   **Data Leakage:**
    *   `reuters` token is a major predictor for real news.
    *   Removing `reuters` drops test accuracy by **1.5%**, showing models rely partly on layout templates.
*   **Domain Shift:**
    *   Vocabulary skews heavily to U.S. election cycles from 2016-2017.
    *   Models are unlikely to generalize to other topics (science, health) or current news cycles.

---

## Slide 10: Conclusion & Future Scope
*   **Summary:** End-to-end pipeline built successfully. Random Forest provides the best accuracy, while Logistic Regression is optimal for low-latency systems.
*   **Future Directions:**
    1.  Strip all source-identifying words (e.g. `reuters`, locations) to verify semantic accuracy.
    2.  Expand vocabulary to include bigrams/trigrams.
    3.  Evaluate on out-of-domain test sets (e.g. recent 2026 political articles).
