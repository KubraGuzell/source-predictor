import os
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from data_utils import load_dataset, basic_cleaning


DATA_PATH = "data/dataset.json"
OUTPUT_DIR = "eda_outputs"

# Her text için kaç komşu bakacağız
N_NEIGHBORS = 6   # kendisi dahil gelir, yani kendisi + en yakın 5 komşu

# Çok kısa textler similarity'de yanıltıcı olabilir, istersek filtreleyebiliriz
MIN_WORDS = 5

# Similarity threshold'ları
HIGH_SIM_THRESHOLD = 0.90
MEDIUM_SIM_THRESHOLD = 0.80


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset(DATA_PATH)
    df = basic_cleaning(df)

    df["text_len_words"] = df["text"].str.split().str.len()
    df = df[df["text_len_words"] >= MIN_WORDS].reset_index(drop=True)

    print("=" * 70)
    print("NEAR-DUPLICATE ANALYSIS")
    print("=" * 70)
    print(f"Records after min-word filter ({MIN_WORDS}): {len(df)}")

    # Daha hızlı ve dengeli olsun diye word TF-IDF kullanıyoruz
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df["text"])

    print(f"TF-IDF matrix shape: {X.shape}")

    # cosine distance kullanacağız; similarity = 1 - distance
    nn = NearestNeighbors(
        n_neighbors=N_NEIGHBORS,
        metric="cosine",
        algorithm="brute"
    )
    nn.fit(X)

    distances, indices = nn.kneighbors(X)

    pairs = []
    seen = set()

    for i in range(len(df)):
        for rank in range(1, N_NEIGHBORS):  # 0 kendisi olur, onu atlıyoruz
            j = indices[i, rank]

            # Aynı çifti iki kere yazmamak için sırala
            a, b = sorted((i, j))
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))

            sim = 1.0 - distances[i, rank]

            pairs.append({
                "idx_a": a,
                "idx_b": b,
                "similarity": sim,
                "same_source": df.loc[a, "source"] == df.loc[b, "source"],
                "source_a": df.loc[a, "source"],
                "source_b": df.loc[b, "source"],
                "category_a": df.loc[a, "category"],
                "category_b": df.loc[b, "category"],
                "text_a": df.loc[a, "text"],
                "text_b": df.loc[b, "text"],
                "words_a": df.loc[a, "text_len_words"],
                "words_b": df.loc[b, "text_len_words"],
            })

    pairs_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False).reset_index(drop=True)

    # Kaydet
    all_pairs_path = os.path.join(OUTPUT_DIR, "near_duplicate_pairs_all.csv")
    pairs_df.to_csv(all_pairs_path, index=False)

    high_sim_df = pairs_df[pairs_df["similarity"] >= HIGH_SIM_THRESHOLD].copy()
    medium_sim_df = pairs_df[pairs_df["similarity"] >= MEDIUM_SIM_THRESHOLD].copy()

    high_sim_path = os.path.join(OUTPUT_DIR, "near_duplicate_pairs_high_sim.csv")
    medium_sim_path = os.path.join(OUTPUT_DIR, "near_duplicate_pairs_medium_sim.csv")

    high_sim_df.to_csv(high_sim_path, index=False)
    medium_sim_df.to_csv(medium_sim_path, index=False)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"All candidate pairs saved      : {all_pairs_path}")
    print(f"Pairs with sim >= {MEDIUM_SIM_THRESHOLD:.2f}: {len(medium_sim_df)}")
    print(f"Pairs with sim >= {HIGH_SIM_THRESHOLD:.2f}: {len(high_sim_df)}")
    print()

    if len(medium_sim_df) > 0:
        print("Top 20 most similar pairs:")
        cols = ["similarity", "same_source", "source_a", "source_b", "category_a", "category_b"]
        print(medium_sim_df[cols].head(20).to_string(index=False))
        print()

        same_source_ratio = medium_sim_df["same_source"].mean()
        print(f"Among sim >= {MEDIUM_SIM_THRESHOLD:.2f}, same-source ratio: {same_source_ratio:.4f}")

    # Farklı source ama çok benzer örnekleri özellikle incelemek isteyebiliriz
    cross_source_high = high_sim_df[high_sim_df["same_source"] == False].copy()
    cross_source_path = os.path.join(OUTPUT_DIR, "near_duplicate_cross_source_high_sim.csv")
    cross_source_high.to_csv(cross_source_path, index=False)

    print(f"Cross-source high-sim pairs saved: {cross_source_path}")
    print()

    # İlk birkaç örneği terminale yazdır
    preview_df = high_sim_df.head(10)

    if len(preview_df) == 0:
        print("No very high similarity pairs found.")
    else:
        print("=" * 70)
        print("TOP HIGH-SIMILARITY EXAMPLES")
        print("=" * 70)
        for k, row in preview_df.iterrows():
            print(f"\nPair #{k+1}")
            print(f"Similarity : {row['similarity']:.4f}")
            print(f"Same source: {row['same_source']}")
            print(f"A: source={row['source_a']} | category={row['category_a']} | words={row['words_a']}")
            print(f"B: source={row['source_b']} | category={row['category_b']} | words={row['words_b']}")
            print("-" * 40)
            print("TEXT A:")
            print(row["text_a"][:500])
            print("-" * 40)
            print("TEXT B:")
            print(row["text_b"][:500])
            print("=" * 70)


if __name__ == "__main__":
    main()