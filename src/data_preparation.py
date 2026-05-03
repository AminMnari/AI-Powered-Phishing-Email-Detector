import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "cleaned_emails.csv"

TEXT_CANDIDATE_COLUMNS = ["text", "email_text", "email", "body", "message", "content"]
LABEL_CANDIDATE_COLUMNS = ["label", "target", "class", "is_spam", "is_phishing"]


def find_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a {kind} column. Available columns: {df.columns.tolist()}"
    )


def normalize_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    text_col = find_column(df, TEXT_CANDIDATE_COLUMNS, "text")
    label_col = find_column(df, LABEL_CANDIDATE_COLUMNS, "label")

    normalized = df[[text_col, label_col]].copy()
    normalized.columns = ["text", "label"]

    label_map = {
        "phishing": 1,
        "spam": 1,
        "malicious": 1,
        "legitimate": 0,
        "ham": 0,
        "benign": 0,
    }

    if normalized["label"].dtype == object:
        normalized["label"] = (
            normalized["label"].astype(str).str.strip().str.lower().replace(label_map)
        )

    normalized["label"] = pd.to_numeric(normalized["label"], errors="coerce")
    before = len(normalized)
    normalized = normalized.dropna(subset=["text", "label"]).copy()
    normalized = normalized[normalized["label"].isin([0, 1])].copy()
    removed = before - len(normalized)

    print(f"{name}: shape={normalized.shape}, removed_invalid_rows={removed}")
    print(f"{name}: label distribution:\n{normalized['label'].value_counts(dropna=False)}")

    normalized["label"] = normalized["label"].astype(int)
    normalized["text"] = normalized["text"].astype(str)
    return normalized


def clean_email(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> None:
    df1 = pd.read_csv(DATA_DIR / "phishing_legitimate.csv")
    df2 = pd.read_csv(DATA_DIR / "spam_assassin.csv")

    print("Dataset 1 columns:", df1.columns.tolist())
    print("Dataset 2 columns:", df2.columns.tolist())

    d1 = normalize_dataset(df1, "Phishing-Legitimate")
    d2 = normalize_dataset(df2, "SpamAssassin")

    df = pd.concat([d1, d2], ignore_index=True)
    print(f"Merged dataset shape: {df.shape}")
    print("Merged label distribution:\n", df["label"].value_counts())

    print("Cleaning email text...")
    df["cleaned_text"] = df["text"].apply(clean_email)
    df = df[df["cleaned_text"].str.len() > 10].copy()

    print(f"Final dataset shape after cleaning: {df.shape}")
    print(df[["cleaned_text", "label"]].head(3))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
