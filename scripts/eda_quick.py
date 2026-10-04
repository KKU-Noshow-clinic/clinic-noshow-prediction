"""Quick, read-only EDA for the Traffy Fondue CSV.

Prints a summary to the terminal. Does not modify the data or anything in the repo.

Usage:
    uv run python scripts/eda_quick.py                      # default: data/raw/traffy.csv
    uv run python scripts/eda_quick.py path/to/file.csv
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("data/raw/traffy.csv")
BKK_LAT = (13.45, 14.00)  # rough Bangkok bounding box
BKK_LON = (100.30, 100.95)
RARE_CLASS_THRESHOLD = 200  # categories with fewer tickets are candidates for "อื่นๆ"
EMOJI_RE = "[\U0001f300-\U0001faff]"


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_types(s: pd.Series) -> pd.Series:
    """'{ถนน,น้ำท่วม}' or "['ถนน']" -> ['ถนน', ...];  '{}' / "['']" / NaN -> []"""
    return (
        s.fillna("")
        .astype(str)
        .str.strip("{}[] ")
        .str.split(",")
        .apply(lambda xs: [x.strip(" '\"") for x in xs if x.strip(" '\"")])
    )


def parse_coords(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'100.53,13.72' or "['100.53', '13.72']" -> (lon, lat). Per row: in Thailand
    lon (~97-106) > lat (~5-21), so the larger value is lon even if a row stores them as lat,lon."""
    clean = s.astype(str).str.replace(r"[\[\]' ]", "", regex=True)
    parts = clean.str.split(",", n=1, expand=True).reindex(columns=[0, 1])
    a = pd.to_numeric(parts[0], errors="coerce")
    b = pd.to_numeric(parts[1], errors="coerce")
    lon = pd.concat([a, b], axis=1).max(axis=1, skipna=False)
    lat = pd.concat([a, b], axis=1).min(axis=1, skipna=False)
    swapped = (a < b).sum()
    print(f"rows stored as lat,lon (auto-swapped): {swapped:,}")
    return lon, lat


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_colwidth", 60)

    section("0. File / data version")
    print(f"path   : {path}")
    print(f"size   : {path.stat().st_size / 1e6:.1f} MB")
    print(f"sha256 : {sha256(path)}")

    df = pd.read_csv(path, low_memory=False)
    # Traffy exports differ: text is `comment` or `description`, label is `type` or
    # `problem_type_abdul`. Pick whichever exists and actually has values.
    text_col = next((c for c in ["comment", "description"] if c in df), None)
    candidates = [c for c in ["type", "problem_type_abdul"] if c in df]
    has_labels = [c for c in candidates if parse_types(df[c]).str.len().any()]
    label_col = next(
        iter(has_labels),
        "type" if "type" in df else None,
    )
    print(f"text column: {text_col}   label column: {label_col}")

    # 1 --------------------------------------------------------------------
    section("1. Structure")
    print(f"rows = {len(df):,}   columns = {df.shape[1]}\n")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head(5).to_string())

    # 2 --------------------------------------------------------------------
    section("2. Missing values & duplicates")
    missing = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    print("% missing per column:")
    print(missing.to_string())
    if "ticket_id" in df:
        print(f"\nduplicate ticket_id      : {df['ticket_id'].duplicated().sum():,}")
    if text_col:
        text = df[text_col].fillna("").astype(str).str.strip()
        print(f"empty text               : {(text == '').sum():,}")
        dup_text = text[text != ""].duplicated().sum()
        print(f"duplicate text           : {dup_text:,}")
    if label_col:
        empty_type = parse_types(df[label_col]).str.len().eq(0).sum()
        print(f"empty type (no label)    : {empty_type:,}")

    # 3 --------------------------------------------------------------------
    section(f"3. Labels ({label_col})")
    if label_col:
        labels = parse_types(df[label_col])
        print("labels per ticket:")
        print(labels.str.len().value_counts().sort_index().to_string())
        counts = labels.explode().dropna().value_counts()
        print(f"\nunique categories: {len(counts)}\n")
        print(counts.to_string())
        rare = counts[counts < RARE_CLASS_THRESHOLD]
        print(f"\ncategories with < {RARE_CLASS_THRESHOLD} tickets: {list(rare.index)}")
    else:
        print("no label column found")

    # 4 --------------------------------------------------------------------
    section("4. Time")
    if "timestamp" in df:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True, format="mixed")
        print(f"unparseable timestamps: {ts.isna().sum():,}")
        print(f"first: {ts.min()}   last: {ts.max()}")
        month = ts.dt.tz_convert("Asia/Bangkok").dt.tz_localize(None).dt.to_period("M")
        monthly = month.value_counts().sort_index().rename("tickets").to_frame()
        if label_col:
            has_flood = parse_types(df[label_col]).apply(lambda xs: "น้ำท่วม" in xs)
            monthly["flood_share_%"] = (has_flood.groupby(month).mean() * 100).round(1)
        print("\ntickets per month (flood_share = drift evidence by season):")
        print(monthly.to_string())
    else:
        print("column 'timestamp' not found")

    # 5 --------------------------------------------------------------------
    section(f"5. Text ({text_col})")
    if text_col:
        text = df[text_col].fillna("").astype(str)
        length = text.str.len()
        print("character length:")
        print(length.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(1).to_string())
        print(f"\nshort (< 10 chars)   : {(length < 10).sum():,}")
        print(f"long (> 1000 chars)  : {(length > 1000).sum():,}")
        print(f"contains URL         : {text.str.contains(r'https?://').sum():,}")
        print(f"contains emoji       : {text.str.contains(EMOJI_RE).sum():,}")
        print("\n5 random comments:")
        non_empty = text[text.str.strip() != ""]
        for c in non_empty.sample(min(5, len(non_empty)), random_state=42):
            print(" -", c[:150].replace("\n", " "))
    else:
        print("no text column found")

    # 6 --------------------------------------------------------------------
    section("6. Location & metadata")
    if "address" in df and "district" not in df:
        df["district"] = df["address"].str.extract(r"เขต\s*(\S+)")[0]
    for col in ["province", "district", "state", "organization", "org"]:
        if col in df:
            vc = df[col].value_counts(dropna=False)
            print(f"\n{col}: {df[col].nunique():,} unique values (top 10)")
            print(vc.head(10).to_string())
    if "coords" in df:
        lon, lat = parse_coords(df["coords"])
        in_bkk = lat.between(*BKK_LAT) & lon.between(*BKK_LON)
        print(f"\ncoords unparseable      : {lat.isna().sum():,}")
        print(f"coords inside Bangkok   : {in_bkk.sum():,} ({in_bkk.mean() * 100:.1f}%)")
        print(f"lat range: {lat.min()} .. {lat.max()}   lon range: {lon.min()} .. {lon.max()}")
    for col in ["star", "count_reopen"]:
        if col in df:
            print(f"\n{col}:")
            print(pd.to_numeric(df[col], errors="coerce").describe().round(2).to_string())

    print("\nDone. (read-only: nothing was modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
