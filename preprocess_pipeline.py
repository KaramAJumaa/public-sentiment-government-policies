import os
import re
import pandas as pd
import numpy as np

def clean_tweet_text(text: str) -> str:
    """
    Standardize and clean tweet text for NLP sentiment mining:
    1. Remove URLs (http, https, www)
    2. Remove user mentions (@username)
    3. Remove special characters, emojis, and punctuation (retain alphanumeric and spaces)
    4. Normalize whitespace
    5. Convert to lowercase
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove mentions (@username)
    text = re.sub(r"@\w+", "", text)
    # Remove special characters, symbols, emojis (keep letters, numbers, spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Collapse multiple whitespace sequences into single space
    text = re.sub(r"\s+", " ", text)
    # Lowercase and strip leading/trailing whitespace
    return text.lower().strip()

def run_data_pipeline():
    print("=== Phase 1 Lead Data Engineering Pipeline: Group 10 ===")
    
    # 1. Load Datasets
    print("\n[Step 1] Loading raw datasets...")
    trump_path = "hashtag_donaldtrump.csv"
    biden_path = "hashtag_joebiden.csv"
    
    df_trump = pd.read_csv(trump_path, lineterminator="\n", low_memory=False)
    df_biden = pd.read_csv(biden_path, lineterminator="\n", low_memory=False)
    
    print(f"Loaded Donald Trump dataset: {df_trump.shape[0]:,} rows, {df_trump.shape[1]} columns")
    print(f"Loaded Joe Biden dataset:     {df_biden.shape[0]:,} rows, {df_biden.shape[1]} columns")
    
    # Tag candidate / target entity to maintain political context upon merge
    df_trump["candidate"] = "Donald Trump"
    df_biden["candidate"] = "Joe Biden"
    
    # 2. Merge into single dataframe
    print("\n[Step 2] Merging datasets for balanced sentiment representation...")
    df_merged = pd.concat([df_trump, df_biden], ignore_index=True)
    print(f"Merged corpus total shape: {df_merged.shape[0]:,} rows, {df_merged.shape[1]} columns")
    
    # 3. Sample a subset of 110,000 rows
    print("\n[Step 3] Sampling 110,000 rows (random_state=42 for reproducibility)...")
    df_sampled = df_merged.sample(n=110000, random_state=42).copy()
    print(f"Sampled shape: {df_sampled.shape}")
    
    # 4. Handle Missing Values in Text Column
    print("\n[Step 4] Checking and dropping null values in 'tweet' text column...")
    null_count = df_sampled["tweet"].isnull().sum()
    print(f"Null entries in 'tweet': {null_count}")
    df_sampled = df_sampled.dropna(subset=["tweet"])
    
    # 5. Drop Exact Full-Row Duplicates
    initial_exact_dups = df_sampled.duplicated().sum()
    print(f"Exact full-row duplicates detected: {initial_exact_dups}")
    if initial_exact_dups > 0:
        df_sampled = df_sampled.drop_duplicates()
        
    # 6. Feature Reduction: Drop uninformative identifiers
    print("\n[Step 5] Performing Feature Reduction (target 15-20 features)...")
    drop_columns = [
        "tweet_id",           # Arbitrary 64-bit Twitter ID; zero predictive/semantic signal
        "user_id",            # Arbitrary user account ID; high-cardinality identifier
        "user_name",          # User display name; unstructured noise, PII risk
        "user_screen_name",   # Twitter handle; unique identifier, PII risk
        "collected_at"        # Scraper execution timestamp; technical scraping artifact
    ]
    df_reduced = df_sampled.drop(columns=drop_columns)
    print(f"Dropped {len(drop_columns)} columns: {drop_columns}")
    print(f"Shape after feature reduction: {df_reduced.shape}")
    
    # Check for duplicate rows revealed after dropping unique identifiers
    post_drop_dups = df_reduced.duplicated().sum()
    print(f"Exact duplicate records across remaining features: {post_drop_dups}")
    if post_drop_dups > 0:
        df_reduced = df_reduced.drop_duplicates()
        print(f"Shape after removing post-drop duplicate records: {df_reduced.shape}")
    
    # 7. Text Preprocessing
    print("\n[Step 6] Performing text cleaning and standardization...")
    df_reduced["cleaned_tweet"] = df_reduced["tweet"].apply(clean_tweet_text)
    
    # Filter out empty or whitespace-only strings resulting from cleaning
    empty_cleaned_mask = df_reduced["cleaned_tweet"] == ""
    empty_count = empty_cleaned_mask.sum()
    print(f"Tweets becoming empty post-cleaning (e.g. only URLs/handles): {empty_count}")
    df_final = df_reduced[~empty_cleaned_mask].copy()
    
    # 8. Final Shape Verification
    final_rows, final_cols = df_final.shape
    print(f"\n[Verification] Final Shape: {final_rows:,} rows, {final_cols} columns")
    assert final_rows > 100000, f"Row count {final_rows} is not > 100,000!"
    assert 15 <= final_cols <= 20, f"Feature count {final_cols} is outside [15, 20] range!"
    print("PASS: Final shape conforms strictly to project constraints (>100k rows, 15-20 features).")
    
    # 9. Export Cleaned Dataset
    output_filename = "cleaned_sentiment_data.csv"
    print(f"\n[Step 7] Exporting cleaned dataframe to {output_filename}...")
    df_final.to_csv(output_filename, index=False, encoding="utf-8")
    
    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"Successfully exported {output_filename} ({file_size_mb:.2f} MB)")
    
    # 10. Print Column Overview
    print("\n=== Final Features List & Data Types ===")
    for idx, (col, dtype) in enumerate(df_final.dtypes.items(), 1):
        sample_val = str(df_final[col].iloc[0])
        if len(sample_val) > 40:
            sample_val = sample_val[:37] + "..."
        print(f"{idx:2d}. {col:<22} | Type: {str(dtype):<10} | Sample: {sample_val}")

if __name__ == "__main__":
    run_data_pipeline()
