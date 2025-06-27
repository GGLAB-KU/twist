from glob import glob
import os
import re
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_PATH: str = "./"
DATA_PATH: str = os.path.join(PROJECT_PATH, "inputs")
OUTPUT_PATH: str = os.path.join(PROJECT_PATH, "outputs", "yok_dataset_V1.xlsx")

SPECIAL_CHARS: set[str] = {"¨", "ï", "¸", "?", "$"}
TURKISH_LETTERS: set[str] = set("ıüğüşçö")

UNIVERSITY_ABBREV: dict[str, str] = {
    "Orta Doğu Teknik Üniversitesi": "ODTÜ",
    "Boğaziçi Üniversitesi": "BOUN",
    "İstanbul Teknik Üniversitesi": "İTÜ",
    "İhsan Doğramacı Bilkent Üniversitesi": "BİLKENT",
    "Koç Üniversitesi": "KOÇ",
    "Sabancı Üniversitesi": "SABANCI",
}

DEPARTMENT_ABBREV: dict[str, str] = {
    (
        "Elektrik ve Elektronik Mühendisliği = "
        "Electrical and Electronics Engineering"
    ): "ELEC",
    (
        "Bilgisayar Mühendisliği Bilimleri-Bilgisayar ve Kontrol = "
        "Computer Engineering and Computer Science and Control"
    ): "COMP",
    "Makine Mühendisliği = Mechanical Engineering": "MECH",
    "Endüstri ve Endüstri Mühendisliği = Industrial and Industrial Engineering": "INDR",
    "Fizik ve Fizik Mühendisliği = Physics and Physics Engineering": "PHYS",
    "Matematik = Mathematics": "MATH",
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def starts_with_two_capitals(text: str) -> bool:
    """Return True if the first two characters are uppercase (e.g., 'IT')."""
    return text[:2].isupper()


def count_periods(text: str) -> int:
    """Count '.' characters in *text*."""
    return text.count(".")


def contains_turkish_letters(text: str) -> bool:
    """Detect at least one Turkish-specific letter."""
    return any(letter in text for letter in TURKISH_LETTERS)


def contains_any_special_char(text: str, chars: set[str]) -> bool:
    """Return True if *text* has any char in *chars*."""
    return any(char in text for char in chars)


def starts_with_capital(text: str) -> bool:
    """Return True if the first character is uppercase."""
    return text[0].isupper()


def ends_with_dot(text: str) -> bool:
    """Return True if *text* ends with a period."""
    return text.endswith(".")


def contains_single_letter_word(text: str) -> bool:
    """Return True if *text* contains a single-letter word (e.g., 'a', 'I')."""
    return any(len(word) == 1 and word.isalpha() for word in text.split())


def length_difference_within_limit(text1: str, text2: str, limit: float = 0.10) -> bool:
    """
    Compare lengths of *text1* and *text2*.
    Return True when the relative difference is ≤ *limit*.
    """
    len1, len2 = len(text1), len(text2)
    return abs(len1 - len2) / max(len1, len2) <= limit


def transform_dashes(text: str) -> str:
    """
    Replace patterns like 'a-a' → 'aa' and 'aa- aa' → 'aa'.
    Keeps behaviour identical to the original regexes.
    """
    if isinstance(text, str):
        text = re.sub(r"([a-zA-Z])-([a-zA-Z])", r"\1\1", text)
        text = re.sub(r"([a-zA-Z]{2})- ([a-zA-Z]{2})", r"\1\2", text)
    return text


def merge_csv_files(data_path: str) -> pd.DataFrame:
    """Load and concatenate all CSV files in *data_path*."""
    csv_files = glob(os.path.join(data_path, "*.csv"))
    return pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Read & combine data
    merged_df = merge_csv_files(DATA_PATH)

    # 2. Ensure 'tr' and 'en' are strings
    merged_df["tr"] = merged_df["tr"].astype(str)
    merged_df["en"] = merged_df["en"].astype(str)

    # 3. Dash transformations
    merged_df["tr"] = merged_df["tr"].apply(transform_dashes)
    merged_df["en"] = merged_df["en"].apply(transform_dashes)

    # 4. Filtering rules
    mask = (
        (~merged_df["en"].apply(starts_with_two_capitals))
        & (~merged_df["tr"].apply(starts_with_two_capitals))
        & (merged_df["tr"].apply(count_periods) == merged_df["en"].apply(count_periods))
        & merged_df["tr"].apply(starts_with_capital)
        & merged_df["en"].apply(starts_with_capital)
        & merged_df["tr"].apply(contains_turkish_letters)
        & ~merged_df["tr"].apply(contains_any_special_char, chars=SPECIAL_CHARS)
        & ~merged_df["en"].apply(contains_any_special_char, chars=SPECIAL_CHARS)
        & merged_df["tr"].apply(ends_with_dot)
        & merged_df["en"].apply(ends_with_dot)
        & ~merged_df["tr"].apply(contains_single_letter_word)
        & merged_df.apply(
            lambda row: length_difference_within_limit(row["tr"], row["en"]), axis=1
        )
    )

    filtered_df = (
        merged_df.dropna()
        .drop_duplicates(subset=["tr", "en"])
        .loc[mask]
        .copy()
    )

    # 5. Derived columns
    filtered_df["n_characters_en"] = filtered_df["en"].str.len()
    filtered_df["n_words_en"] = filtered_df["en"].apply(lambda x: len(x.split()))
    filtered_df["n_sentences_en"] = filtered_df["en"].apply(
        lambda x: len(re.split(r"[.!?]", x))
    )

    # 6. Abbreviations
    filtered_df["university"] = filtered_df["university"].replace(UNIVERSITY_ABBREV)
    filtered_df["konu"] = filtered_df["konu"].replace(DEPARTMENT_ABBREV)

    # 7. Unique identifier
    filtered_df["yok_id"] = range(1, len(filtered_df) + 1)

    # 8. Column order & export
    final_cols = [
        "yok_id",
        "university",
        "konu",
        "tr",
        "en",
        "n_characters_en",
        "n_words_en",
        "n_sentences_en",
    ]
    filtered_df = filtered_df[final_cols]

    filtered_df.to_excel(OUTPUT_PATH, index=False, engine="xlsxwriter")
    print(f"✅ Dataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()