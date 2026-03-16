import json
import pandas as pd


def load_dataset(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    expected_cols = {"text", "category", "source"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(subset=["text", "source"])

    df["text"] = df["text"].astype(str).str.strip()
    df["source"] = df["source"].astype(str).str.strip()
    df["category"] = df["category"].fillna("").astype(str).str.strip()

    df = df[df["text"] != ""]
    df = df[df["source"] != ""]

    # exact duplicate (text, source) temizliği
    df = df.drop_duplicates(subset=["text", "source"]).reset_index(drop=True)

    return df