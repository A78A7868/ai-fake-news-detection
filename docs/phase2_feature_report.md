# Phase 2 Feature Report — AI-Powered Fake News Detection

**Project:** IICT Summer Internship Program in AI & ML, 2026  
**Phase:** 2 of 4 — Feature Engineering  
**Period:** Days 8–14

---

## 1. Feature Representation Methods

We transformed the cleaned text corpus into numerical vectors using two classic natural language processing techniques: **Bag-of-Words (BoW)** and **Term Frequency-Inverse Document Frequency (TF-IDF)**. While we utilized Scikit-Learn's optimized library implementations, we detail the underlying mathematical formulations below.

### Bag-of-Words (BoW)
The Bag-of-Words model represents each news article as a flat vector of token frequencies. It discards word order, grammar, and syntax, focusing entirely on token occurrence. 

For a document $d$ and a term $t$ in a predefined vocabulary $V$, the BoW feature value $x_{t,d}$ is simply the raw count of occurrences of term $t$ in document $d$:
$$x_{t,d} = \text{count}(t, d)$$

The entire document becomes a vector $\mathbf{x}_d \in \mathbb{R}^{|V|}$ where each dimension corresponds to a specific vocabulary term.

### Term Frequency-Inverse Document Frequency (TF-IDF)
While BoW treats all terms equally, TF-IDF down-weights words that are common across the entire dataset (such as "said" or "would") and scales up words that are highly specific to individual documents.

The value $w_{t,d}$ of a term $t$ in a document $d$ is calculated by multiplying its Term Frequency (TF) with its Inverse Document Frequency (IDF). 

In `scikit-learn`'s implementation, the components are defined as:

1. **Term Frequency (TF):** The raw count of the term in the document.
   $$\text{TF}(t, d) = \text{count}(t, d)$$

2. **Inverse Document Frequency (IDF):** Calculated with smoothing to prevent division-by-zero errors:
   $$\text{IDF}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$
   where $N$ is the total number of documents in the training set and $\text{df}(t)$ is the number of documents containing term $t$.

3. **TF-IDF Weighting:** 
   $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

4. **Normalization:** The resulting vectors are L2-normalized to ensure that document length differences do not distort the distance metrics during modeling:
   $$\mathbf{v}_d = \frac{\mathbf{w}_d}{\|\mathbf{w}_d\|_2}$$

---

## 2. Parameter Selection & Rationale

We used `scikit-learn`'s `CountVectorizer` and `TfidfVectorizer` with the following parameters:

- **`max_features=5000`**: We limited the vocabulary size to the 5,000 most frequent tokens across the training set. This baseline was chosen to match the project brief's skeleton. It keeps the feature space manageable, prevents out-of-memory errors during model training, and filters out highly rare typos or OCR errors that add noise without contributing generalizable signal.
- **`ngram_range=(1, 1)`**: We used unigrams (single words) as our core features. While bigrams (e.g. "white house") capture local context, they exponentially increase the vocabulary size and sparsity. We chose to start with a pure unigram representation as our baseline.
- **Stopwords**: Already handled during the Phase 1 cleaning process.
- **`token_pattern`**: We left this default because we had already tokenized and cleaned the text in Phase 1, rejoining the tokens with single spaces. The default pattern `(?u)\b\w\w+\b` splits the pre-joined text cleanly back into individual words.

---

## 3. Dimensionality & Sparsity Analysis

We executed an 80/20 train/test split stratified by the labels to ensure both sets contained proportional representation of fake and real articles.

- **Total clean dataset size:** 39,098 articles
- **Training partition ($X_{\text{train}}$):** 31,278 articles
- **Test partition ($X_{\text{test}}$):** 7,820 articles

### Sparsity Statistics
Since most news articles only contain a tiny fraction of the 5,000 words in our vocabulary, the resulting feature matrices are extremely sparse. The sparsity stats for both Bag-of-Words and TF-IDF matrices on the training partition are identical because they share the same zero/non-zero structure:

- **Matrix Dimensions:** 31,278 rows $\times$ 5,000 columns
- **Total Possible Cells:** 156,390,000
- **Non-Zero Elements:** 4,122,390
- **Matrix Sparsity Ratio:** **97.3640%**

This high sparsity ratio (only 2.636% of cells contain non-zero values) means we must store these features as Compressed Sparse Row (CSR) matrices to avoid wasting system memory. Standard dense arrays would require storing over 156 million floating-point numbers per representation, which is unnecessary.

---

## 4. Feature Weights Analysis

We computed the mean TF-IDF weights across the training partition to see which words dominate the two classes.

### Top 10 TF-IDF Terms in Fake News
| Term | Mean Weight |
|---|---|
| trump | 0.0894 |
| video | 0.0335 |
| clinton | 0.0275 |
| obama | 0.0267 |
| people | 0.0255 |
| hillary | 0.0251 |
| president | 0.0236 |
| one | 0.0219 |
| like | 0.0211 |
| donald | 0.0206 |

### Top 10 TF-IDF Terms in Real News
| Term | Mean Weight |
|---|---|
| said | 0.0666 |
| trump | 0.0521 |
| reuters | 0.0304 |
| president | 0.0269 |
| would | 0.0260 |
| house | 0.0240 |
| state | 0.0235 |
| government | 0.0218 |
| republican | 0.0202 |
| washington | 0.0188 |

### Key Observations
1. **Source Identifiers:** The term `reuters` appears as the third most significant feature in real news. This is a potential source of data leakage, since verified articles in this dataset were scraped directly from Reuters and frequently include a location/source tag (e.g. "WASHINGTON (Reuters) -"). In Phase 3, we should evaluate how heavily models rely on this specific token.
2. **Subject Focus:** Both classes are heavily dominated by political entities (`trump`, `clinton`, `obama`, `house`, `government`). This matches the subject category distribution we found in Phase 1.
3. **Writing Style Clues:** Fake news articles show high frequency of terms like `video` and `like`, indicating a more conversational or multimedia-dependent reporting style compared to the formal language of real articles.

A comparison chart visualising these weights has been saved to `results/figures/top_tfidf_weights.png`.

---

## 5. Outputs and Serialization

To ensure the downstream modeling phase is reproducible and fast, we serialized the following outputs to `data/processed/` using `joblib`:

1. Labels: `y_train.joblib`, `y_test.joblib`
2. Bag-of-Words matrices: `X_train_bow.joblib`, `X_test_bow.joblib`
3. TF-IDF matrices: `X_train_tfidf.joblib`, `X_test_tfidf.joblib`
4. Fitted Vectorizers: `bow_vectorizer.joblib`, `tfidf_vectorizer.joblib`

These files can be directly loaded in Phase 3 without repeating the fitting or transformation process.

---

*Phase 2 complete. Next: Phase 3 — Model Building (Days 15–21).*
