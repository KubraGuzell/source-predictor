import os
import hashlib
import pandas as pd
from data_utils import load_dataset, basic_cleaning


DATA_PATH = "data/dataset.json"
OUTPUT_DIR = "eda_outputs"


def normalize_text(text: str) -> str:
    """
    Light normalization for duplicate checks.
    Do not use aggressive normalization for modeling yet.
    """
    return " ".join(text.lower().split())


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset(DATA_PATH)
    df = basic_cleaning(df)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total records           : {len(df)}")
    print(f"Unique sources          : {df['source'].nunique()}")
    print(f"Unique categories       : {df['category'].nunique()}")
    print()

    # Text length statistics
    df["text_len_chars"] = df["text"].str.len()
    df["text_len_words"] = df["text"].str.split().str.len()

    print("=" * 60)
    print("TEXT LENGTH STATS")
    print("=" * 60)
    print(df["text_len_chars"].describe())
    print()
    print(df["text_len_words"].describe())
    print()

    # Class distribution
    source_counts = df["source"].value_counts()
    category_counts = df["category"].value_counts()

    print("=" * 60)
    print("TOP 20 SOURCES")
    print("=" * 60)
    print(source_counts.head(20))
    print()

    print("=" * 60)
    print("TOP 20 CATEGORIES")
    print("=" * 60)
    print(category_counts.head(20))
    print()

    # Exact duplicates
    exact_duplicate_count = df.duplicated(subset=["text"]).sum()
    exact_duplicate_text_source_count = df.duplicated(subset=["text", "source"]).sum()

    print("=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)
    print(f"Exact duplicate texts               : {exact_duplicate_count}")
    print(f"Exact duplicate (text, source) rows : {exact_duplicate_text_source_count}")

    # Same text with different sources
    text_to_source_count = df.groupby("text")["source"].nunique()
    ambiguous_texts = text_to_source_count[text_to_source_count > 1]

    print(f"Texts appearing with multiple sources: {len(ambiguous_texts)}")
    print()

    # Normalized duplicates
    df["normalized_text"] = df["text"].apply(normalize_text)
    normalized_dup_count = df.duplicated(subset=["normalized_text"]).sum()
    print(f"Normalized duplicate texts          : {normalized_dup_count}")
    print()

    # Hash groups
    df["text_hash"] = df["normalized_text"].apply(text_hash)

    # Group-level leakage risk proxy
    group_source_counts = df.groupby("text_hash")["source"].nunique()
    suspicious_groups = group_source_counts[group_source_counts > 1]
    print(f"Normalized text groups with multiple sources: {len(suspicious_groups)}")
    print()

    # Save useful reports
    source_counts.to_csv(os.path.join(OUTPUT_DIR, "source_distribution.csv"))
    category_counts.to_csv(os.path.join(OUTPUT_DIR, "category_distribution.csv"))
    df[["text", "source", "category", "text_len_chars", "text_len_words", "text_hash"]].to_csv(
        os.path.join(OUTPUT_DIR, "dataset_profile.csv"), index=False
    )

    if len(ambiguous_texts) > 0:
        ambiguous_df = df[df["text"].isin(ambiguous_texts.index)]
        ambiguous_df.to_csv(os.path.join(OUTPUT_DIR, "same_text_multiple_sources.csv"), index=False)

    print(f"EDA outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()