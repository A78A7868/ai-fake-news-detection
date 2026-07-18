# Phase 4 Evaluation Report — AI-Powered Fake News Detection

**Project:** IICT Summer Internship Program in AI & ML, 2026  
**Phase:** 4 of 4 — Evaluation  
**Period:** Days 22–30

---

## 1. Quantitative Performance Comparison

We evaluated the four trained classifiers on the 20% test partition (7,820 articles) using the fixed 5,000 TF-IDF features. Performance metrics were calculated using the inference outputs:

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time |
|---|---|---|---|---|---|
| **KNN** ($K=5$) | 87.80% | 84.60% | 94.74% | 89.38% | 6.3782s |
| **Logistic Regression** | 98.76% | 98.57% | 99.15% | 98.86% | 0.0025s |
| **Random Forest** | **99.68%** | **99.60%** | **99.81%** | **99.71%** | 0.1382s |
| **NeuralNet (MLP)** | 98.87% | 98.82% | 99.10% | 98.96% | 0.0261s |

*Note: Inference time measures the total seconds taken to predict class labels for all 7,820 test documents on the evaluation hardware.*

### Metric Highlights
- **Random Forest** achieved the highest overall score, reaching **99.68% accuracy** and a **99.71% F1-score**. 
- **Logistic Regression** and **NeuralNet (MLP)** performed almost identically, both exceeding **98.7% accuracy**.
- **KNN** performed worst, peaking at **87.80% accuracy** and taking **6.38 seconds** to complete predictions, which is over 2,500 times slower than Logistic Regression.

---

## 2. Confusion Matrices & Error Analysis

The confusion matrix for each classifier shows where predictions went wrong on the 7,820 test articles (3,581 Fake, 4,239 Real):

### K-Nearest Neighbors (KNN)
- **True Fake (Predicted Fake):** 2,850
- **False Real (Type I Error):** 731
- **False Fake (Type II Error):** 223
- **True Real (Predicted Real):** 4,016
*Analysis:* KNN shows a high rate of false positives (labeling fake articles as real). Because it relies on simple geometric distance in a sparse 5,000-dimensional space, documents with similar lengths or common political terms are easily misclassified as nearby neighbors.

### Logistic Regression
- **True Fake (Predicted Fake):** 3,520
- **False Real:** 61
- **False Fake:** 36
- **True Real (Predicted Real):** 4,203
*Analysis:* Logistic Regression is highly accurate, making only 97 errors out of 7,820 predictions. The errors are balanced between false negatives and false positives.

### Random Forest
- **True Fake (Predicted Fake):** 3,564
- **False Real:** 17
- **False Fake:** 8
- **True Real (Predicted Real):** 4,231
*Analysis:* Random Forest made only **25 total errors** across the entire test set. The ensemble of 100 trees minimizes classification variance, leading to near-perfect splits on the TF-IDF representation.

### Multi-Layer Perceptron (NeuralNet)
- **True Fake (Predicted Fake):** 3,531
- **False Real:** 50
- **False Fake:** 38
- **True Real (Predicted Real):** 4,201
*Analysis:* The neural network committed 88 errors. It performs slightly better than Logistic Regression but does not outperform the Random Forest ensemble.

Individual confusion matrix plots have been saved to `results/figures/confusion_matrix_{Model}.png`.

---

## 3. Parametric vs. Non-Parametric Trade-offs

This project demonstrates clear trade-offs between parametric and non-parametric classifiers:

### 1. Training vs. Inference Speeds
- **KNN** is a non-parametric model with zero training cost, but it is slow during prediction (6.38s) because it must calculate Euclidean distances against all 31,278 stored training documents for every test instance.
- **Logistic Regression** (parametric) is extremely fast for both training (0.18s) and prediction (0.0025s). Prediction is a simple linear dot product, making it ideal for low-latency production setups.
- **Random Forest** (non-parametric ensemble) takes longer to train (14.62s) and requires more inference time (0.14s) than linear models, but it provides the highest accuracy.

### 2. Generalization & Memory Space
- **Random Forest** requires **~40 MB** of storage on disk because it stores 100 deep decision trees.
- **Logistic Regression** is extremely lightweight, storing only 5,000 coefficient weights and a single intercept value.
- The high performance of the linear models suggests that the cleaned text representation is largely linearly separable under TF-IDF weighting.

---

## 4. Project Limitations & Failure Cases

1. **Source Leakage:** The term `reuters` emerged as a top feature for real news. Since all real articles were scraped from Reuters and often contain the source tag, the classifiers are likely utilizing this single token as a shortcut. When we tested a model with `reuters` stripped, validation accuracy fell by roughly 1.5%, highlighting that the models rely partly on crawler formatting rather than semantic validity.
2. **Political Bias:** The dataset is heavily focused on U.S. politics from 2016-2017. As a result, terms like `trump`, `clinton`, and `obama` have high weights. The classifiers will likely fail if evaluated on current news cycles (e.g. artificial intelligence or science news) where these specific entities are not mentioned.

---

*Phase 4 complete. Next: Assembly of the final IEEE conference report and project presentation slides (Days 22–30).*
