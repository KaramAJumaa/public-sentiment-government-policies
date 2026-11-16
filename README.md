# Analyzing Public Sentiment on Government Policies

**Course:** Data Mining (A0443402) — Academic Year 2026–2027  
**Team:** Group 10  
**Phase:** Phase 1 — Data Understanding, Preprocessing & Feature Engineering  

---

## 1. Overview & Objectives

This project analyzes public sentiment and societal reactions surrounding government policies and political leadership during the 2020 United States Presidential Election. Using social media discourse from Twitter, our goal is to uncover sentiment trends, public engagement metrics, and geographical patterns across distinct policy subjects.

### Key Objectives
- Consolidate multi-source political discourse into a unified sentiment analysis corpus.
- Implement a reproducible data cleaning and lexical normalization pipeline.
- Perform feature reduction by eliminating non-informative identifiers and high-cardinality metadata.
- Prepare structured, clean text attributes for subsequent NLP tasks (sentiment classification, topic modeling, spatial analysis).

---

## 2. Data Sources & Attribution

The raw data originates from the Kaggle 2020 US Election Twitter collection:
- **Kaggle Source:** [US Election 2020 Tweets Dataset](https://www.kaggle.com/datasets/manchunhui/us-election-2020-tweets?select=hashtag_donaldtrump.csv)
- **Raw Files:**
  - `hashtag_donaldtrump.csv`: 970,919 raw tweets (~483 MB)
  - `hashtag_joebiden.csv`: 776,886 raw tweets (~380 MB)
  - *Combined Total:* 1,747,805 raw observations across 21 schema columns.

> *Note on Raw Files:* Due to GitHub repository size limits, raw datasets (>800 MB combined) are not stored directly in this repository. Download raw files directly from the Kaggle link above and place them in the project root to run the end-to-end preprocessing script.

---

## 3. Preprocessing & Data Engineering Pipeline

The end-to-end processing logic is implemented in `preprocess_pipeline.py`. The execution workflow follows these stages:

```
[Raw Trump & Biden CSVs]
           │
           ▼
[Entity Labeling: 'candidate' Tag]
           │
           ▼
[Corpus Concatenation (1.75M rows)]
           │
           ▼
[Reproducible Sampling: 110,000 rows (seed=42)]
           │
           ▼
[Integrity Auditing: Null Text Removal & Exact Deduplication]
           │
           ▼
[Feature Reduction: Drop 5 High-Cardinality Identifiers]
           │
           ▼
[Regex Text Preprocessing -> 'cleaned_tweet']
           │
           ▼
[Export Cleaned Dataset: cleaned_sentiment_data.csv]
```

### Preprocessing Actions:
1. **Candidate Tagging:** Added a categorical `candidate` column (`Donald Trump` vs. `Joe Biden`) before merging to preserve target context.
2. **Merging & Sampling:** Concatenated raw collections and extracted a pseudo-random sample of 110,000 rows with a fixed random seed (`42`).
3. **Null & Duplicate Removal:** Verified and removed any records missing tweet text. Purged exact duplicate records.
4. **Feature Reduction:** Dropped uninformative identifier columns to achieve the target 15–20 feature count.
5. **Text Cleaning:** Generated `cleaned_tweet` via regex operations:
   - Stripped web URLs (`https?://\S+|www\.\S+`).
   - Stripped user handles (`@\w+`).
   - Removed special characters, symbols, and punctuation (`[^a-zA-Z0-9\s]`).
   - Standardized multiple whitespace sequences to single space.
   - Standardized characters to lowercase.
   - Discarded records that collapsed to empty strings post-cleaning.

---

## 4. Dataset Dimensions & Statistics

| Metric | Raw Combined Input | Processed Sample |
| :--- | :--- | :--- |
| **Total Rows** | 1,747,805 | **109,989** |
| **Total Features** | 21 | **18** |
| **Donald Trump Records** | 970,919 (55.5%) | 60,888 (55.36%) |
| **Joe Biden Records** | 776,886 (44.5%) | 49,101 (44.64%) |
| **Clean Output Size** | ~864 MB (Raw) | ~57.1 MB (`cleaned_sentiment_data.csv`) |

---

## 5. Feature Dictionary & Engineering Rationales

### Retained Features (18)

| # | Feature | Data Type | Analytical Role & Justification |
| :--- | :--- | :--- | :--- |
| 1 | `created_at` | datetime/str | Temporal analysis, event studies (debates, election events), and sentiment trends over time. |
| 2 | `tweet` | string | Unaltered source text preserved for data auditing and future lexical feature extraction. |
| 3 | `cleaned_tweet` | string | Normalized text ready for tokenization, TF-IDF vectorization, and sentiment classifiers. |
| 4 | `likes` | numeric | Endorsement metric; used for weighting sentiment scores by public resonance. |
| 5 | `retweet_count` | numeric | Amplification and virality metric measuring message diffusion across networks. |
| 6 | `source` | string | Device/platform metadata; utilized for bot detection and platform segmentation. |
| 7 | `user_description` | string | Profile biography; secondary text attribute for assessing user political orientation. |
| 8 | `user_join_date` | datetime/str | Account longevity metric; helps distinguish organic voters from newly formed astroturfing accounts. |
| 9 | `user_followers_count` | numeric | Quantifies social reach and influence tier of the expressing user. |
| 10 | `user_location` | string | User-provided free-text location for geographic fallback extraction. |
| 11 | `lat` | float | Continuous latitude coordinate for geospatial clustering and heatmap density plotting. |
| 12 | `long` | float | Continuous longitude coordinate for coordinate-level spatial analysis. |
| 13 | `city` | string | Urban administrative division; evaluates urban vs. rural sentiment variance. |
| 14 | `state` | string | State-level division; key for examining policy sentiment across key battleground states. |
| 15 | `state_code` | string | Standardized two-letter state code for choropleth mapping. |
| 16 | `country` | string | Sovereign nation filter; distinguishes domestic from international discourse. |
| 17 | `continent` | string | Macro-regional classification for global public sentiment comparison. |
| 18 | `candidate` | string | Target entity category (`Donald Trump` vs. `Joe Biden`) for comparative evaluation. |

### Dropped Features (5)

- **`tweet_id`**: Dropped. Unique 64-bit identifier from Twitter. Lacks semantic or sentiment value; risks overfitting.
- **`user_id`**: Dropped. Numeric user surrogate key. Provides no generalizable linguistic or sentiment insight.
- **`user_name`**: Dropped. Display name containing unstructured text and emojis. Carries PII risk without predictive value.
- **`user_screen_name`**: Dropped. Unique user handle (`@user`). PII identifier with zero sentiment signal.
- **`collected_at`**: Dropped. Technical scraper timestamp. Redundant with `created_at` which tracks actual publication time.

---

## 6. Project Structure

```
├── .gitignore                      # Excludes large raw CSVs and cache files
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python environment dependencies
├── preprocess_pipeline.py          # Data ingestion and cleaning script
└── cleaned_sentiment_data.csv      # Exported cleaned dataset (109,989 rows, 18 columns)
```

---

## 7. How to Reproduce

### 1. Clone Repository
```bash
git clone https://github.com/KaramAJumaa/public-sentiment-government-policies.git
cd public-sentiment-government-policies
```

### 2. Set Up Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Pipeline
Place `hashtag_donaldtrump.csv` and `hashtag_joebiden.csv` into the project root directory, then run:
```bash
python preprocess_pipeline.py
```
The output file `cleaned_sentiment_data.csv` will be generated with 109,989 rows and 18 columns.
