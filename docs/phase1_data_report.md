# Phase 1 Data Report — AI-Powered Fake News Detection

**Project:** IICT Summer Internship Program in AI & ML, 2026  
**Phase:** 1 of 4 — Data Collection & Cleaning  
**Period:** Days 1–7

---

## 1. Dataset Source

We used the **Fake and Real News Dataset** hosted on Kaggle:

> Bisaillon, C. (2020). *Fake and Real News Dataset*. Kaggle.  
> https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset ships as two separate CSV files: `Fake.csv` (articles flagged as fake) and `True.csv` (articles from Reuters and other verified sources). Both were downloaded manually and placed in `data/raw/`.

---

## 2. Dataset Size and Structure

We loaded the files directly and computed all counts from the actual data:

| File | Row count |
|---|---|
| `Fake.csv` | 23,481 |
| `True.csv` | 21,417 |
| After merge | 44,898 |
| After duplicate removal | 39,103 |
| After empty-text removal (post-cleaning) | 39,098 |

**Columns in both files:** `title`, `text`, `subject`, `date`

There were no null values in either file — every row had a title, body text, subject label, and date. We concatenated the two files and added a binary `label` column (`0 = fake`, `1 = real`). Title and body were then merged into a single `text` field for classification, since titles frequently carry classification-relevant language that the body might not contain alone.

**Duplicate rows:** 5,795 exact duplicates (matched on combined text) were removed. This is a known issue in this dataset — some articles were scraped multiple times. Keeping duplicates would inflate performance metrics artificially, so we dropped them before any further processing.

**Final class balance (after deduplication and empty-text removal):**

| Class | Count | Percentage |
|---|---|---|
| Fake (0) | 17,902 | 45.79% |
| Real (1) | 21,196 | 54.21% |

The dataset is reasonably balanced — roughly a 46/54 split. We did not apply oversampling or undersampling at this stage, as the imbalance is mild enough that standard classifiers should handle it without special treatment. If class-weighted training becomes necessary in Phase 3, we will document that decision there.

---

## 3. Cleaning Steps

All cleaning is implemented in `src/preprocessing.py`. Steps applied in order:

1. **Lowercase.** Converts all text to lowercase before any other processing.
2. **URL removal.** The regex `https?://\S+|www\.\S+` strips both protocol URLs and bare `www.` links. News articles frequently embed source links or reference URLs that add token noise without contributing classification signal.
3. **Manual tokenization.** We extracted tokens using `re.findall(r'\b[a-z]+\b', text)` — a plain regex word-boundary match on lowercase alphabetic runs. This is the step the brief explicitly requires to be manual: we did not use `nltk.word_tokenize`, spaCy's tokenizer, or any other library tokenizer. Digits and punctuation are excluded by the character class, so no separate punctuation-stripping step is needed.
4. **Stopword removal.** Tokens were filtered against NLTK's English stopword list (`nltk.corpus.stopwords.words("english")`). We used it as a plain Python `set` for lookup — the NLTK tokenizer itself was not invoked. Tokens of length ≤ 1 were also dropped, since single-letter survivors (e.g. `s`, `u`) carry no meaning.
5. **Rejoin.** Remaining tokens are joined back into a space-separated string. This is the input format expected by `TfidfVectorizer` and `CountVectorizer` in Phase 2.

**Before/after example (first article in dataset):**

*Raw (first 200 chars):*
```
Donald Trump just couldn't wish all Americans a Happy New Year and leave it at that. Instead,
he had to give a shout out to his enemies, haters and the very dishonest fake news media...
```

*Cleaned:*
```
donald trump wish americans happy new year leave instead shout enemies haters dishonest fake
news media...
```

---

## 4. EDA Findings

### Article Length

Average word counts before and after cleaning:

| Class | Avg words (raw) | Avg words (cleaned) |
|---|---|---|
| Fake (0) | 429 | ~247 |
| Real (1) | 395 | ~227 |

Fake articles are slightly longer on average. The distribution has a long right tail in both classes — a handful of articles exceed 2,000 words (max 8,148 in fake, 5,181 in real). The length distribution plot is at `results/figures/length_distribution.png`.

### Most Frequent Words

After cleaning, common tokens across both classes include `said`, `trump`, `people`, `president`, `us`. The overlap is substantial, which is expected — both fake and real articles cover political news. The differences appear in second- and third-tier terms: fake articles show higher frequency of vague or emotionally loaded words, while real articles lean toward named organizations and formal political terminology. The side-by-side bar chart is at `results/figures/top_words_per_class.png`.

### Subject Category Distribution

The `subject` column groups articles by topic. The distribution across the final dataset:

| Subject | Count |
|---|---|
| politicsNews | 11,216 |
| worldnews | 9,980 |
| News | 9,050 |
| politics | 6,832 |
| US_News | 783 |
| left-news | 705 |
| Government News | 532 |

This is almost entirely political and governmental coverage. The classifiers in Phase 3 will be trained on this narrow domain — we should expect weaker generalization on, say, science or entertainment news. The bar chart is at `results/figures/subject_distribution.png`.

---

## 5. Data Quality Issues

- **5,795 duplicate rows** removed. These were exact text matches, likely from repeated scraping of the same source.
- **5 rows emptied by cleaning** — articles that contained almost no alphabetic content (mostly URLs, numbers, or punctuation). These were dropped after the cleaning step.
- **Date column not used.** The `date` field exists in both files but we are not using it for classification. Publication date could be an interesting feature for detecting news from specific political periods, but that is out of scope for this project.
- **Subject label inconsistency.** The fake articles use subject labels like `News`, `politics`, `left-news`, `US_News`, and `Government News`, while real articles use `politicsNews` and `worldnews`. This labeling mismatch means the `subject` field is itself a near-perfect proxy for the fake/real label — using it as a feature would be data leakage. We are excluding `subject` from the feature matrix in Phase 2 and using only the text content.

---

## 6. Figures

All plots are saved in `results/figures/`:

| File | Description |
|---|---|
| `class_balance.png` | Bar chart of fake vs. real article counts after deduplication |
| `length_distribution.png` | Overlapping histograms of raw word count by class (clipped at 2,000) |
| `top_words_per_class.png` | Top 20 tokens in fake and real articles after cleaning |
| `subject_distribution.png` | Article counts broken down by subject category |

---

## 7. Output for Phase 2

The cleaned dataset is saved to `data/processed/news_cleaned.csv` (39,098 rows, columns: `text`, `clean_text`, `label`, `subject`). Phase 2 will load this file and apply `TfidfVectorizer(max_features=5000)` and `CountVectorizer` to the `clean_text` column, then perform an 80/20 train/test split with `random_state=42`.

---

*Phase 1 complete — pipeline ran fully without manual steps. Next: Phase 2 — Feature Engineering (Days 8–14).*
