import pandas as pd
from data_utils import load_dataset, basic_cleaning

DATA_PATH = "data/dataset.json"

def main():

    df = load_dataset(DATA_PATH)
    df = basic_cleaning(df)

    print("\n===== CATEGORY vs SOURCE DISTRIBUTION =====\n")

    table = pd.crosstab(df["category"], df["source"])

    print(table)

    print("\n===== NORMALIZED DISTRIBUTION =====\n")

    normalized = pd.crosstab(
        df["category"],
        df["source"],
        normalize="index"
    )

    print(normalized)

    print("\n===== CATEGORY COUNTS =====\n")
    print(df["category"].value_counts())

    print("\n===== SOURCE COUNTS =====\n")
    print(df["source"].value_counts())

if __name__ == "__main__":
    main()