from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

HISTORY_CSV = RAW_DIR / "history.csv"
TODAY_CSV = RAW_DIR / "today_entries.csv"
MODEL_PATH = MODEL_DIR / "win_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURE_COLS = [
    "car_no",
    "age",
    "score",
    "win_rate",
    "place2_rate",
    "place3_rate",
    "back_count",
    "recent_avg_finish",
    "days_since_last_race",
    "venue_win_rate",
    "odds_win",
    "style_code",
]

STYLE_MAP = {
    "逃": 0,
    "捲": 1,
    "ま": 1,
    "差": 2,
    "追": 3,
    "両": 4,
    "自在": 4,
}


def ensure_dirs():
    for p in [RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, MODEL_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def add_style_code(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "style" not in df.columns:
        df["style"] = "追"
    df["style_code"] = df["style"].map(STYLE_MAP).fillna(-1).astype(float)
    return df


def prepare_features(df: pd.DataFrame, fill_values=None):
    df = add_style_code(df)

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

    X = df[FEATURE_COLS].copy()

    for col in FEATURE_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if fill_values is None:
        fill_values = {}
        for col in FEATURE_COLS:
            med = X[col].median()
            if pd.isna(med):
                med = 0.0
            fill_values[col] = float(med)

    X = X.fillna(fill_values)
    return X, fill_values


def normalize_race_prob(df: pd.DataFrame, raw_col="p_raw", out_col="p_win") -> pd.DataFrame:
    df = df.copy()
    df[raw_col] = np.clip(pd.to_numeric(df[raw_col], errors="coerce").fillna(1e-6), 1e-6, 1.0)
    sums = df.groupby("race_id")[raw_col].transform("sum")
    df[out_col] = df[raw_col] / sums.replace(0, np.nan)
    df[out_col] = df[out_col].fillna(1.0 / df.groupby("race_id")["race_id"].transform("count"))
    return df
