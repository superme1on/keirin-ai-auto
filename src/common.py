from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

HISTORY_CSV = RAW_DIR / "history.csv"
TODAY_CSV = RAW_DIR / "today_entries.csv"
TODAY_ODDS_CSV = RAW_DIR / "today_odds.csv"
HISTORY_ODDS_CSV = RAW_DIR / "history_odds.csv"
HISTORY_TRIFECTA_ODDS_CSV = RAW_DIR / "history_trifecta_odds.csv"
MODEL_PATH = MODEL_DIR / "win_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURE_COLS = [
    "race_no",
    "car_no",
    "age",
    "score",
    "rider_strength",
    "rider_strength_rank",
    "rider_strength_gap_to_best",
    "rider_strength_vs_field",
    "score_rank",
    "score_gap_to_best",
    "score_vs_field",
    "player_prior_races",
    "player_prior_win_rate",
    "player_prior_place2_rate",
    "player_prior_place3_rate",
    "player_prior_avg_finish",
    "player_prior_strength",
    "player_prior_strength_rank",
    "player_prior_strength_gap_to_best",
    "player_prior_strength_vs_field",
    "win_rate",
    "win_rate_rank",
    "win_rate_gap_to_best",
    "place2_rate",
    "place2_rate_rank",
    "place3_rate",
    "place3_rate_rank",
    "back_count",
    "recent_avg_finish",
    "recent_avg_finish_rank",
    "days_since_last_race",
    "venue_win_rate",
    "odds_win",
    "distance",
    "style_code",
    "venue_code",
    "race_class_code",
    "race_type_code",
    "player_id_code",
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


def stable_bucket(value, modulo=1000):
    if pd.isna(value):
        return -1.0
    text = str(value).strip()
    if not text:
        return -1.0
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16) % modulo)


def add_categorical_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = add_style_code(df)
    df = df.copy()
    df["venue_code"] = df.get("venue", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 200))
    df["race_class_code"] = df.get("race_class", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 50))
    df["race_type_code"] = df.get("race_type", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 100))
    df["player_id_code"] = df.get("player_id", pd.Series(index=df.index, dtype=object)).map(lambda x: stable_bucket(x, 2000))
    return df


def add_player_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"player_id", "date", "finish_pos"}
    if not required.issubset(df.columns):
        return df

    original_index = df.index
    work = df.copy()
    work["_original_order"] = np.arange(len(work))
    work["_date_dt"] = pd.to_datetime(work["date"], errors="coerce")
    sort_cols = [c for c in ["_date_dt", "race_id", "race_no", "car_no", "_original_order"] if c in work.columns]
    work = work.sort_values(sort_cols, kind="mergesort")

    player = work["player_id"].astype(str)
    finish = pd.to_numeric(work["finish_pos"], errors="coerce")
    observed = finish.notna().astype(float)
    prior_races = observed.groupby(player, dropna=False).cumsum() - observed

    win = finish.eq(1).astype(float) * observed
    place2 = finish.le(2).astype(float) * observed
    place3 = finish.le(3).astype(float) * observed
    finish_sum = finish.fillna(0) * observed

    prior_wins = win.groupby(player, dropna=False).cumsum() - win
    prior_place2 = place2.groupby(player, dropna=False).cumsum() - place2
    prior_place3 = place3.groupby(player, dropna=False).cumsum() - place3
    prior_finish_sum = finish_sum.groupby(player, dropna=False).cumsum() - finish_sum

    def safe_rate(values):
        return np.where(prior_races > 0, values / prior_races, np.nan)

    work["player_prior_races"] = prior_races
    work["player_prior_win_rate"] = safe_rate(prior_wins)
    work["player_prior_place2_rate"] = safe_rate(prior_place2)
    work["player_prior_place3_rate"] = safe_rate(prior_place3)
    work["player_prior_avg_finish"] = safe_rate(prior_finish_sum)

    event_dates = work["_date_dt"].where(observed.astype(bool))
    prev_dates = event_dates.groupby(player, dropna=False).transform(lambda s: s.ffill().shift(1))
    work["days_since_last_race"] = (work["_date_dt"] - prev_dates).dt.days

    has_prior = work["player_prior_races"] > 0
    for source, target in [
        ("player_prior_win_rate", "win_rate"),
        ("player_prior_place2_rate", "place2_rate"),
        ("player_prior_place3_rate", "place3_rate"),
        ("player_prior_avg_finish", "recent_avg_finish"),
    ]:
        if target not in work.columns:
            work[target] = np.nan
        work[target] = work[source].where(has_prior, pd.to_numeric(work[target], errors="coerce"))

    work["player_prior_strength"] = (
        work["player_prior_win_rate"].fillna(0) * 10.0
        + work["player_prior_place2_rate"].fillna(0) * 4.0
        + work["player_prior_place3_rate"].fillna(0) * 2.0
        - work["player_prior_avg_finish"].fillna(work["player_prior_avg_finish"].median()) * 0.45
        + np.log1p(work["player_prior_races"].fillna(0)) * 0.1
    )

    if "race_id" in work.columns:
        race_ids = work["race_id"]
        values = pd.to_numeric(work["player_prior_strength"], errors="coerce")
        group = values.groupby(race_ids, dropna=False)
        work["player_prior_strength_rank"] = group.rank(ascending=False, method="average")
        work["player_prior_strength_gap_to_best"] = group.transform("max") - values
        work["player_prior_strength_vs_field"] = values - group.transform("mean")

    work = work.sort_values("_original_order", kind="mergesort").set_index(original_index)
    return work.drop(columns=["_original_order", "_date_dt"], errors="ignore")


def add_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    race_key = "race_id" if "race_id" in df.columns else None

    def num(col, default=np.nan):
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    score = num("score")
    win_rate = num("win_rate")
    place2_rate = num("place2_rate")
    place3_rate = num("place3_rate")
    recent_avg_finish = num("recent_avg_finish")

    # Compact, human-readable rider strength prior. The model still learns the
    # final weights, but this gives it a direct "who is strong" signal.
    df["rider_strength"] = (
        score.fillna(score.median()) * 0.08
        + win_rate.fillna(0) * 10.0
        + place2_rate.fillna(0) * 4.0
        + place3_rate.fillna(0) * 2.0
        - recent_avg_finish.fillna(recent_avg_finish.median()) * 0.45
    )

    if race_key:
        race_ids = df[race_key]
        for col, ascending in [
            ("rider_strength", False),
            ("score", False),
            ("win_rate", False),
            ("place2_rate", False),
            ("place3_rate", False),
            ("recent_avg_finish", True),
        ]:
            values = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            ranks = values.groupby(race_ids, dropna=False).rank(ascending=ascending, method="average")
            df[f"{col}_rank"] = ranks

        strength_group = df["rider_strength"].groupby(race_ids, dropna=False)
        score_group = score.groupby(race_ids, dropna=False)
        win_rate_group = win_rate.groupby(race_ids, dropna=False)
        df["rider_strength_gap_to_best"] = strength_group.transform("max") - df["rider_strength"]
        df["rider_strength_vs_field"] = df["rider_strength"] - strength_group.transform("mean")
        df["score_gap_to_best"] = score_group.transform("max") - score
        df["score_vs_field"] = score - score_group.transform("mean")
        df["win_rate_gap_to_best"] = win_rate_group.transform("max") - win_rate
    else:
        df["rider_strength_rank"] = np.nan
        df["rider_strength_gap_to_best"] = np.nan
        df["rider_strength_vs_field"] = np.nan
        df["score_rank"] = np.nan
        df["score_gap_to_best"] = np.nan
        df["score_vs_field"] = np.nan
        df["win_rate_rank"] = np.nan
        df["win_rate_gap_to_best"] = np.nan
        df["place2_rate_rank"] = np.nan
        df["place3_rate_rank"] = np.nan
        df["recent_avg_finish_rank"] = np.nan

    return df


def prepare_features(df: pd.DataFrame, fill_values=None):
    df = add_categorical_codes(df)
    df = add_strength_features(df)

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
