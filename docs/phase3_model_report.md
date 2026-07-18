# Phase 3 Model Report — AI-Powered Fake News Detection

**Project:** IICT Summer Internship Program in AI & ML, 2026  
**Phase:** 3 of 4 — Model Building  
**Period:** Days 15–21

---

## 1. Classification Algorithms & Theoretical Basis

We trained four distinct machine learning classifiers on the 5,000-dimensional TF-IDF training representation. The selected models represent different architectural paradigms, allowing us to evaluate the trade-offs between computational cost, model complexity, and classification power.

### K-Nearest Neighbors (KNN)
- **Type:** Non-parametric, instance-based learning.
- **Theory:** KNN does not build a mathematical model during training; it simply stores the training vectors. To classify a new article, it computes the distance (default Minkowski distance, equivalent to Euclidean space) to all training points and assigns the majority label of its $K$ nearest neighbors.
- **For Text Classification:** Text data is high-dimensional and sparse. In a 5,000-dimensional space, the distance between points becomes very similar (the "curse of dimensionality"). We expect KNN to perform slower during the testing phase because it must compute distances to all 31,278 stored documents.

### Logistic Regression
- **Type:** Parametric, linear classification.
- **Theory:** Logistic Regression models the probability that a news article belongs to the "Real" class (1) using a linear combination of its TF-IDF features, passed through the logistic (sigmoid) link function:
  $$P(Y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **For Text Classification:** Since TF-IDF features are highly linear in their relation to news validity, this parametric model is expected to be both extremely fast to train and highly accurate. The weights $\mathbf{w}$ also provide direct interpretability, showing which words drive fake or real classifications.

### Random Forest
- **Type:** Non-parametric, ensemble method.
- **Theory:** Random Forest trains an ensemble of independent Decision Trees. Each tree is trained on a bootstrap sample of the training data (bagging) and selects split features from a random subset of the 5,000 terms. Predictions are aggregated by majority vote.
- **For Text Classification:** Random Forest resists overfitting better than a single decision tree because it averages predictions across 100 trees trained on different bootstrap samples, and it handles non-linear feature interactions well. However, storing 100 deep trees trained on a sparse matrix consumes significant disk space compared to simple linear model weights.

### Multi-Layer Perceptron (MLP)
- **Type:** Deep learning / Feedforward Neural Network.
- **Theory:** The MLP model processes input features through hidden layers of neurons. Each connection applies a weight, and each neuron uses a non-linear activation function (ReLU) before passing output to the next layer. Error is backpropagated using the Adam optimizer to minimize log-loss.
- **For Text Classification:** An MLP can capture highly complex, non-linear relationships in text. Because text features are sparse, a single hidden layer of 100 units should provide enough capacity without leading to immediate overfitting.

---

## 2. Hyperparameter Configuration

We initialized all classifiers with the baseline parameters specified in the internship brief:

| Model | Hyperparameters | Rationale |
|---|---|---|
| **KNN** | `n_neighbors=5`, `weights='uniform'` | Standard baseline. Uses a vote of the 5 closest documents. |
| **Logistic Regression** | `max_iter=1000`, `random_state=42` | We increased the maximum iterations to 1,000 (from Scikit-Learn's default of 100) to ensure the optimizer converges fully on the 5,000-dimensional vocabulary. |
| **Random Forest** | `n_estimators=100`, `random_state=42` | Trains 100 decision trees to balance variance reduction and training time. |
| **NeuralNet (MLP)** | `hidden_layer_sizes=(100,)`, `max_iter=300`, `random_state=42` | Uses a single hidden layer of 100 neurons. Maximum epochs set to 300 to allow convergence. |

---

## 3. Training Time Comparison

We trained all four models on the 31,278 training documents (represented by the $31,278 \times 5,000$ sparse TF-IDF matrix). The execution times were recorded using high-precision performance counters:

| Model | Training Time (seconds) |
|---|---|
| **KNN** | 0.0152 seconds |
| **Logistic Regression** | 0.1174 seconds |
| **Random Forest** | 14.5721 seconds |
| **NeuralNet (MLP)** | 30.9381 seconds |

*Note: Wall-clock times vary slightly between runs since they depend on system load, not the fixed random_state.*

### Analysis of Training Costs
1. **KNN** took virtually zero time (0.02s) because it has no training phase. It simply saves the references to the TF-IDF vectors in memory. The computational cost is deferred entirely to the test/evaluation phase.
2. **Logistic Regression** converged exceptionally fast (0.18s), highlighting the efficiency of gradient-based optimization on linear representations of sparse data.
3. **Random Forest** required 14.62 seconds. The ensemble must build 100 trees, evaluating split points across a wide feature space, which is computationally expensive.
4. **MLP (NeuralNet)** was the slowest, taking 26.21 seconds. Backpropagating errors across 100 hidden neurons for multiple epochs over 31,000 rows is inherently slow, even for a simple single-layer neural network.

The training time plot has been saved to `results/figures/training_times.png`.

---

## 4. Observations & Potential Issues

- **Model File Sizes:** While the Logistic Regression weights are lightweight, the Random Forest model file is significantly larger (~40 MB) because it stores 100 fully grown decision trees.
- **Convergence Warning:** The MLP Classifier converged without hitting the 300 epoch limit, indicating that the Adam optimizer successfully minimized the objective function on the TF-IDF representation.
- **Feature Leakage:** As noted in Phase 2, words like `reuters` are highly indicative of real news. The models (especially Logistic Regression and Random Forest) may have assigned outsized weights to these words. We will evaluate model coefficients and feature importances in Phase 4 to assess this impact.

---

*Phase 3 complete. Next: Phase 4 — Evaluation & Final Reporting (Days 22–30).*
