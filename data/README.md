# Dataset

This project uses the **ISOT Fake and Real News Dataset**.

## How to get the data

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

2. Place both CSV files in this directory:
   ```
   data/raw/Fake.csv
   data/raw/True.csv
   ```

3. Run the Phase 1 preprocessing script from the project root:
   ```bash
   python run_phase1.py
   ```
   This will clean the text, strip source-tag leakage (Reuters datelines), remove duplicates,
   and save the processed data under `data/processed/`.

## Out-of-domain test set

A small hand-curated set of 10 recent (2023+) news articles is included at
`data/raw/recent_news_2023.csv` for out-of-domain generalization testing.

## Dataset stats (after cleaning)

| Split | Fake | Real | Total |
|-------|------|------|-------|
| Full  | 17,902 | 21,196 | 39,098 |
| Train (80%) | 14,322 | 16,956 | 31,278 |
| Test  (20%) | 3,580  | 4,240  | 7,820  |
