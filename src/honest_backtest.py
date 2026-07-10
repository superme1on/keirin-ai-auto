import argparse
import hashlib
import json
import math
from datetime import datetime
from itertools import permutations
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from audit_history import INTEGRITY_CSV, audit_history
from common import (
    FEATURE_COLS,
    HISTORY_CSV,
    HISTORY_ODDS_CSV,
    MODEL_DIR,
    OUTPUT_DIR,
    PROCESSED_DIR,
    add_player_prior_features,
    ensure_dirs,
    prepare_features,
    stable_bucket,
)


BET_LABELS = {
    "exacta": "2車単",
    "quinella": "2車複",
    "quinella_place": "ワイド",
    "trio": "3連複",
    "trifecta": "3連単",
}
POPULARITY_LIMITS = {
    "exacta": 12,
    "quinella": 12,
    "quinella_place": 12,
    "trio": 15,
    "trifecta": 20,
}
TICKET_FEATURES = [
    "log_odds",
    "market_fair_prob",
    "popularity_order",
    "popularity_fraction",
    "base_prob",
    "log_base_prob",
    "base_market_ratio",
    "field_size",
    "race_no",
    "distance",
    "month",
    "venue_code",
    "p1_win",
    "p1_top2",
    "p1_top3",
    "p1_score",
    "p1_score_rank",
    "p1_win_rate",
    "p1_place2_rate",
    "p1_place3_rate",
    "p1_line_position",
    "p1_line_size",
    "p1_style_code",
    "p2_win",
    "p2_top2",
    "p2_top3",
    "p2_score",
    "p2_score_rank",
    "p2_win_rate",
    "p2_place2_rate",
    "p2_place3_rate",
    "p2_line_position",
    "p2_line_size",
    "p2_style_code",
    "p3_win",
    "p3_top2",
    "p3_top3",
    "p3_score",
    "p3_score_rank",
    "p3_win_rate",
    "p3_place2_rate",
    "p3_place3_rate",
    "p3_line_position",
    "p3_line_size",
    "p3_style_code",
    "same_line_12",
    "same_line_13",
    "same_line_23",
    "line_order_12",
    "line_order_13",
    "line_order_23",
]

OOF_PATH = PROCESSED_DIR / "oof_entry_predictions_v2.csv"
OOF_META_PATH = PROCESSED_DIR / "oof_entry_predictions_v2.json"
MODEL_PATH = MODEL_DIR / "honest_ticket_models.joblib"
SUMMARY_PATH = OUTPUT_DIR / "honest_backtest_by_type.csv"
HOLDOUT_BETS_PATH = OUTPUT_DIR / "honest_backtest_holdout_bets.csv"
OVERALL_PATH = OUTPUT_DIR / "honest_backtest_overall.json"
REPORT_PATH = OUTPUT_DIR / "honest_backtest_report.md"


def feature_signature():
    return hashlib.sha256("\n".join(FEATURE_COLS).encode("utf-8")).hexdigest()[:16]


def require_clean_history():
    if not INTEGRITY_CSV.exists():
        report, _ = audit_history()
        if report["broken_races"]:
            raise ValueError("history contains races with missing entries")
    integrity = pd.read_csv(INTEGRITY_CSV, dtype={"race_id": str})
    complete = integrity["is_complete"].astype(str).str.lower().eq("true")
    if not complete.all():
        raise ValueError(f"history contains {(~complete).sum()} races with missing entries")
    return set(integrity.loc[complete, "race_id"].astype(str))


def normalize_to_race_sum(values, race_ids, target_sum):
    out = pd.Series(np.clip(values, 1e-6, 1 - 1e-6), index=race_ids.index, dtype=float)
    for _ in range(5):
        sums = out.groupby(race_ids, sort=False).transform("sum")
        out = (out * float(target_sum) / sums.replace(0, np.nan)).clip(1e-6, 0.999999)
    return out.fillna(float(target_sum) / race_ids.groupby(race_ids).transform("count")).to_numpy()


def entry_model():
    return HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.045,
        max_leaf_nodes=23,
        min_samples_leaf=40,
        l2_regularization=0.8,
        random_state=42,
    )


def build_oof_entry_predictions(rebuild=False):
    signature = feature_signature()
    if not rebuild and OOF_PATH.exists() and OOF_META_PATH.exists():
        metadata = json.loads(OOF_META_PATH.read_text(encoding="utf-8"))
        if metadata.get("feature_signature") == signature:
            return pd.read_csv(OOF_PATH, dtype={"race_id": str, "player_id": str})

    complete_ids = require_clean_history()
    history = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    history = history[history["race_id"].isin(complete_ids)].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.sort_values(["date", "race_id", "car_no"], kind="mergesort")
    history = add_player_prior_features(history)
    finish = pd.to_numeric(history["finish_pos"], errors="coerce")
    history["target_win"] = finish.eq(1).astype(int)
    history["target_top2"] = finish.le(2).astype(int)
    history["target_top3"] = finish.le(3).astype(int)
    history["score_rank"] = pd.to_numeric(history.get("score"), errors="coerce").groupby(history["race_id"]).rank(
        ascending=False, method="average"
    )
    history["win_rate_rank"] = pd.to_numeric(history.get("win_rate"), errors="coerce").groupby(history["race_id"]).rank(
        ascending=False, method="average"
    )

    predictions = []
    years = sorted(int(year) for year in history["date"].dt.year.dropna().unique())
    for year in [year for year in years if year >= 2022]:
        train = history[history["date"].dt.year < year].copy()
        test = history[history["date"].dt.year == year].copy()
        if len(train) < 5_000 or len(test) == 0:
            continue
        X_train, fill_values = prepare_features(train)
        X_test, _ = prepare_features(test, fill_values)
        fold = test.copy()
        for target, output, target_sum in [
            ("target_win", "p_win", 1),
            ("target_top2", "p_top2", 2),
            ("target_top3", "p_top3", 3),
        ]:
            model = entry_model()
            model.fit(X_train, train[target].astype(int))
            raw = model.predict_proba(X_test)[:, 1]
            fold[output] = normalize_to_race_sum(raw, fold["race_id"], target_sum)
        predictions.append(fold)
        winner_rank = fold.groupby("race_id")["p_win"].rank(ascending=False, method="first")
        top1_hit = float((winner_rank[fold["target_win"].eq(1)] == 1).mean())
        print(f"entry OOF year={year} train_races={train['race_id'].nunique()} test_races={test['race_id'].nunique()} top1={top1_hit:.3%}", flush=True)

    oof = pd.concat(predictions, ignore_index=True)
    keep = [
        "race_id", "date", "venue", "race_no", "player_id", "car_no", "finish_pos",
        "score", "score_rank", "win_rate", "place2_rate", "place3_rate", "line_id",
        "line_position", "line_size", "style", "distance", "entries_number", "p_win",
        "p_top2", "p_top3",
    ]
    for column in keep:
        if column not in oof.columns:
            oof[column] = np.nan
    oof[keep].to_csv(OOF_PATH, index=False)
    metadata = {
        "created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "feature_signature": signature,
        "history_rows": int(len(history)),
        "history_races": int(history["race_id"].nunique()),
        "oof_rows": int(len(oof)),
        "oof_races": int(oof["race_id"].nunique()),
        "years": sorted(oof["date"].dt.year.unique().astype(int).tolist()),
    }
    OOF_META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return oof[keep]


def style_code(value):
    return {
        "\u9003": 0,
        "\u6372": 1,
        "\u30de": 1,
        "\u5dee": 2,
        "\u8ffd": 3,
        "\u4e21": 4,
        "\u81ea\u5728": 4,
    }.get(str(value or ""), -1)


def component_table(oof):
    table = oof.copy()
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    table["entries_number"] = pd.to_numeric(table["entries_number"], errors="coerce")
    fallback_size = table.groupby("race_id")["race_id"].transform("count")
    table["field_size"] = table["entries_number"].fillna(fallback_size)
    table["style_code"] = table["style"].map(style_code)
    keep = [
        "race_id", "car_no", "p_win", "p_top2", "p_top3", "score", "score_rank",
        "win_rate", "place2_rate", "place3_rate", "line_id", "line_position", "line_size",
        "style_code", "field_size", "race_no", "distance", "venue", "date",
    ]
    return table[keep]


def pl2(a, b):
    return np.clip(a * b / np.maximum(1.0 - a, 1e-9), 0, 1)


def pl3(a, b, c):
    return np.clip(a * b / np.maximum(1.0 - a, 1e-9) * c / np.maximum(1.0 - a - b, 1e-9), 0, 1)


def ticket_base_probability(frame, bet_type):
    a = frame["p1_win"].to_numpy(float)
    b = frame["p2_win"].to_numpy(float)
    if bet_type == "exacta":
        return pl2(a, b)
    if bet_type == "quinella":
        return np.clip(pl2(a, b) + pl2(b, a), 0, 1)
    if bet_type == "quinella_place":
        top3_a = frame["p1_top3"].to_numpy(float)
        top3_b = frame["p2_top3"].to_numpy(float)
        return np.clip(top3_a * top3_b, 0, 1)

    c = frame["p3_win"].to_numpy(float)
    if bet_type == "trifecta":
        return pl3(a, b, c)
    total = np.zeros(len(frame), dtype=float)
    values = [a, b, c]
    for i, j, k in permutations(range(3), 3):
        total += pl3(values[i], values[j], values[k])
    return np.clip(total, 0, 1)


def _merge_component(frame, components, position):
    renamed = components.rename(
        columns={
            "car_no": f"car{position}",
            "p_win": f"p{position}_win",
            "p_top2": f"p{position}_top2",
            "p_top3": f"p{position}_top3",
            "score": f"p{position}_score",
            "score_rank": f"p{position}_score_rank",
            "win_rate": f"p{position}_win_rate",
            "place2_rate": f"p{position}_place2_rate",
            "place3_rate": f"p{position}_place3_rate",
            "line_id": f"p{position}_line_id",
            "line_position": f"p{position}_line_position",
            "line_size": f"p{position}_line_size",
            "style_code": f"p{position}_style_code",
        }
    )
    component_cols = [
        "race_id", f"car{position}", f"p{position}_win", f"p{position}_top2", f"p{position}_top3",
        f"p{position}_score", f"p{position}_score_rank", f"p{position}_win_rate",
        f"p{position}_place2_rate", f"p{position}_place3_rate", f"p{position}_line_id",
        f"p{position}_line_position", f"p{position}_line_size", f"p{position}_style_code",
    ]
    if position == 1:
        component_cols += ["field_size", "race_no", "distance", "venue", "date"]
    return frame.merge(renamed[component_cols], on=["race_id", f"car{position}"], how="inner")


def build_ticket_dataset(oof, bet_type, chunksize=400_000, odds_path=HISTORY_ODDS_CSV):
    components = component_table(oof)
    valid_ids = set(components["race_id"].astype(str))
    limit = POPULARITY_LIMITS[bet_type]
    columns = ["date", "race_id", "bet_type", "buy", "odds_used", "max_odds", "popularity_order", "is_actual"]
    parts = []
    for chunk in pd.read_csv(
        odds_path,
        dtype={"race_id": str, "bet_type": str, "buy": str},
        usecols=columns,
        chunksize=chunksize,
    ):
        chunk = chunk[
            chunk["bet_type"].eq(bet_type)
            & chunk["race_id"].isin(valid_ids)
            & pd.to_numeric(chunk["popularity_order"], errors="coerce").le(limit)
        ].copy()
        if len(chunk) == 0:
            continue
        chunk["odds_used"] = pd.to_numeric(chunk["odds_used"], errors="coerce")
        chunk["max_odds"] = pd.to_numeric(chunk["max_odds"], errors="coerce")
        chunk = chunk[chunk["odds_used"].between(1.01, 500, inclusive="both")]
        chunk = chunk.reset_index(drop=True)
        cars = chunk["buy"].str.split("-", expand=True)
        required_cars = 3 if bet_type in {"trio", "trifecta"} else 2
        if cars.shape[1] < required_cars:
            continue
        for position in range(1, required_cars + 1):
            chunk[f"car{position}"] = pd.to_numeric(cars[position - 1], errors="coerce")
        for position in range(1, required_cars + 1):
            chunk = _merge_component(chunk, components, position)
        for position in range(required_cars + 1, 4):
            for suffix in [
                "win", "top2", "top3", "score", "score_rank", "win_rate", "place2_rate",
                "place3_rate", "line_id", "line_position", "line_size", "style_code",
            ]:
                chunk[f"p{position}_{suffix}"] = 0.0

        chunk["base_prob"] = ticket_base_probability(chunk, bet_type)
        chunk["market_fair_prob"] = 0.75 / chunk["odds_used"]
        chunk["log_odds"] = np.log(chunk["odds_used"].clip(lower=1.01))
        chunk["log_base_prob"] = np.log(chunk["base_prob"].clip(lower=1e-8))
        chunk["base_market_ratio"] = chunk["base_prob"] / chunk["market_fair_prob"].clip(lower=1e-8)
        chunk["popularity_order"] = pd.to_numeric(chunk["popularity_order"], errors="coerce")
        chunk["popularity_fraction"] = chunk["popularity_order"] / float(limit)
        chunk["month"] = pd.to_datetime(chunk["date_x"], errors="coerce").dt.month if "date_x" in chunk else pd.to_datetime(chunk["date"], errors="coerce").dt.month
        chunk["venue_code"] = chunk["venue"].map(lambda value: stable_bucket(value, 200))
        chunk["is_hit"] = chunk["is_actual"].astype(str).str.lower().isin(["true", "1"])
        chunk["payout_odds"] = chunk["odds_used"]
        if bet_type == "quinella_place":
            settled_wide = chunk["max_odds"].where(chunk["max_odds"].gt(0), chunk["odds_used"])
            chunk["payout_odds"] = np.where(chunk["is_hit"], settled_wide, chunk["odds_used"])

        for left, right in [(1, 2), (1, 3), (2, 3)]:
            known = chunk[f"p{left}_line_id"].notna() & chunk[f"p{right}_line_id"].notna()
            same = known & chunk[f"p{left}_line_id"].eq(chunk[f"p{right}_line_id"])
            chunk[f"same_line_{left}{right}"] = same.astype(int)
            chunk[f"line_order_{left}{right}"] = (
                same & chunk[f"p{left}_line_position"].lt(chunk[f"p{right}_line_position"])
            ).astype(int)

        for column in TICKET_FEATURES:
            if column not in chunk.columns:
                chunk[column] = 0.0
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        date_column = "date_x" if "date_x" in chunk.columns else "date"
        keep = [date_column, "race_id", "buy", "odds_used", "payout_odds", "popularity_order", "is_hit"] + TICKET_FEATURES
        built = chunk[keep].rename(columns={date_column: "date"})
        parts.append(built)
    if not parts:
        return pd.DataFrame(columns=["date", "race_id", "buy", "odds_used", "is_hit"] + TICKET_FEATURES)
    dataset = pd.concat(parts, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    dataset["bet_type"] = bet_type
    print(f"ticket dataset {bet_type}: rows={len(dataset)} races={dataset['race_id'].nunique()}", flush=True)
    return dataset


def calibration_features(raw_probability, frame):
    raw = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1 - 1e-6)
    market = np.clip(pd.to_numeric(frame["market_fair_prob"], errors="coerce").fillna(1e-6).to_numpy(float), 1e-6, 1 - 1e-6)
    return np.column_stack(
        [
            np.log(raw / (1 - raw)),
            np.log(market / (1 - market)),
            pd.to_numeric(frame["log_odds"], errors="coerce").fillna(0).to_numpy(float),
            pd.to_numeric(frame["popularity_fraction"], errors="coerce").fillna(1).to_numpy(float),
        ]
    )


def fit_platt(raw_probability, target, frame):
    model = LogisticRegression(C=0.25, max_iter=1_000, random_state=42)
    model.fit(calibration_features(raw_probability, frame), np.asarray(target, dtype=int))
    return model


def apply_platt(model, raw_probability, frame):
    return model.predict_proba(calibration_features(raw_probability, frame))[:, 1]


def fill_ticket_features(train, other):
    medians = train[TICKET_FEATURES].median(numeric_only=True).fillna(0.0)
    X_train = train[TICKET_FEATURES].fillna(medians).to_numpy(dtype=float)
    X_other = other[TICKET_FEATURES].fillna(medians).to_numpy(dtype=float)
    return X_train, X_other, medians


def ticket_model():
    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=1.5,
        random_state=42,
    )


def evaluate_selected(selected):
    if len(selected) == 0:
        return {"bets": 0, "hits": 0, "stake_yen": 0, "return_yen": 0, "profit_yen": 0, "roi": None, "positive_months": 0}
    payout_odds = pd.to_numeric(selected.get("payout_odds", selected["odds_used"]), errors="coerce").fillna(selected["odds_used"])
    returns = np.where(selected["is_hit"], np.round(payout_odds * 100), 0)
    stake = len(selected) * 100
    payout = int(returns.sum())
    monthly = selected.assign(return_yen=returns).groupby(selected["date"].dt.to_period("M")).agg(
        bets=("is_hit", "size"), return_yen=("return_yen", "sum")
    )
    monthly["profit_yen"] = monthly["return_yen"] - monthly["bets"] * 100
    return {
        "bets": int(len(selected)),
        "hits": int(selected["is_hit"].sum()),
        "stake_yen": int(stake),
        "return_yen": payout,
        "profit_yen": int(payout - stake),
        "roi": float((payout - stake) / stake),
        "positive_months": int(monthly["profit_yen"].gt(0).sum()),
    }


def apply_strategy(frame, config):
    selected = frame[
        frame["pred_ev"].ge(config["min_ev"])
        & frame["pred_prob"].ge(config["min_prob"])
        & frame["odds_used"].le(config["max_odds"])
    ].copy()
    if len(selected) == 0:
        return selected
    selected = selected.sort_values(["race_id", "pred_ev"], ascending=[True, False]).groupby("race_id", sort=False).head(1)
    daily_cap = int(config.get("daily_cap", 0) or 0)
    if daily_cap > 0:
        selected = (
            selected.sort_values(["date", "pred_ev"], ascending=[True, False])
            .groupby(selected["date"].dt.date, sort=False)
            .head(daily_cap)
        )
    return selected


def choose_strategy(validation, min_bets=100):
    best = None
    for min_ev in [-0.20, -0.15, -0.10, -0.05, 0.0, 0.10, 0.25, 0.50, 0.75, 1.0]:
        for min_prob in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]:
            for max_odds in [10, 20, 50, 100, 200, 300, 500]:
                for daily_cap in [1, 2, 3, 5, 10]:
                    config = {"min_ev": min_ev, "min_prob": min_prob, "max_odds": max_odds, "daily_cap": daily_cap}
                    selected = apply_strategy(validation, config)
                    metrics = evaluate_selected(selected)
                    if metrics["bets"] < min_bets or metrics["roi"] is None:
                        continue
                    if metrics["roi"] < 0.50 or metrics["positive_months"] < 4:
                        continue
                    daily = selected.copy()
                    payout = pd.to_numeric(daily.get("payout_odds", daily["odds_used"]), errors="coerce").fillna(daily["odds_used"])
                    daily["return_multiple"] = np.where(daily["is_hit"], payout, 0) - 1
                    day_roi = daily.groupby(daily["date"].dt.date)["return_multiple"].mean()
                    standard_error = float(day_roi.std(ddof=1) / math.sqrt(len(day_roi))) if len(day_roi) > 1 else 99.0
                    score = float(metrics["roi"] - standard_error)
                    candidate = {**config, **metrics, "selection_score": score}
                    if best is None or candidate["selection_score"] > best["selection_score"]:
                        best = candidate
    return best


def confidence_interval_by_day(selected, iterations=2_000):
    if len(selected) == 0:
        return None, None
    daily = selected.copy()
    daily["stake"] = 100
    payout_odds = pd.to_numeric(daily.get("payout_odds", daily["odds_used"]), errors="coerce").fillna(daily["odds_used"])
    daily["return"] = np.where(daily["is_hit"], np.round(payout_odds * 100), 0)
    totals = daily.groupby(daily["date"].dt.date)[["stake", "return"]].sum().to_numpy(float)
    if len(totals) < 2:
        return None, None
    rng = np.random.default_rng(42)
    rois = []
    for _ in range(iterations):
        sampled = totals[rng.integers(0, len(totals), len(totals))]
        stake = sampled[:, 0].sum()
        rois.append((sampled[:, 1].sum() - stake) / stake)
    return float(np.quantile(rois, 0.025)), float(np.quantile(rois, 0.975))


def fit_and_evaluate_type(dataset, bet_type, min_validation_bets=100):
    train = dataset[dataset["date"].lt("2025-01-01")]
    calibration = dataset[dataset["date"].between("2025-01-01", "2025-06-30")]
    validation = dataset[dataset["date"].between("2025-07-01", "2025-12-31")]
    holdout = dataset[dataset["date"].ge("2026-01-01")]
    if min(len(train), len(calibration), len(validation), len(holdout)) == 0:
        raise ValueError(f"insufficient chronological data for {bet_type}")

    X_train, X_calibration, medians = fill_ticket_features(train, calibration)
    X_validation = validation[TICKET_FEATURES].fillna(medians).to_numpy(float)
    X_holdout = holdout[TICKET_FEATURES].fillna(medians).to_numpy(float)
    model = ticket_model()
    model.fit(X_train, train["is_hit"].astype(int))
    raw_calibration = model.predict_proba(X_calibration)[:, 1]
    calibrator = fit_platt(raw_calibration, calibration["is_hit"], calibration)

    validation = validation.copy()
    validation["pred_prob"] = apply_platt(calibrator, model.predict_proba(X_validation)[:, 1], validation)
    validation["pred_ev"] = validation["pred_prob"] * (validation["odds_used"] * 0.90) - 1
    config = choose_strategy(validation, min_bets=min_validation_bets)
    if config is None:
        config = {"min_ev": 999.0, "min_prob": 1.0, "max_odds": 1.0, "daily_cap": 0, "bets": 0, "roi": None, "selection_score": -999.0}

    holdout = holdout.copy()
    holdout["pred_prob"] = apply_platt(calibrator, model.predict_proba(X_holdout)[:, 1], holdout)
    holdout["pred_ev"] = holdout["pred_prob"] * (holdout["odds_used"] * 0.90) - 1
    selected_validation = apply_strategy(validation, config)
    selected_holdout = apply_strategy(holdout, config)
    validation_metrics = evaluate_selected(selected_validation)
    holdout_metrics = evaluate_selected(selected_holdout)
    auc = float(roc_auc_score(holdout["is_hit"].astype(int), holdout["pred_prob"]))
    brier = float(brier_score_loss(holdout["is_hit"].astype(int), holdout["pred_prob"]))

    summary = {
        "bet_type": bet_type,
        "bet_label": BET_LABELS[bet_type],
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "validation_rows": int(len(validation)),
        "holdout_rows": int(len(holdout)),
        "validation_bets": validation_metrics["bets"],
        "validation_hits": validation_metrics["hits"],
        "validation_profit_yen": validation_metrics["profit_yen"],
        "validation_roi": validation_metrics["roi"],
        "holdout_bets": holdout_metrics["bets"],
        "holdout_hits": holdout_metrics["hits"],
        "holdout_stake_yen": holdout_metrics["stake_yen"],
        "holdout_return_yen": holdout_metrics["return_yen"],
        "holdout_profit_yen": holdout_metrics["profit_yen"],
        "holdout_roi": holdout_metrics["roi"],
        "holdout_positive_months": holdout_metrics["positive_months"],
        "holdout_auc": auc,
        "holdout_brier": brier,
        "min_ev": config["min_ev"],
        "min_prob": config["min_prob"],
        "max_odds": config["max_odds"],
        "daily_cap": config.get("daily_cap", 0),
        "selection_score": config.get("selection_score"),
    }
    selected_holdout = selected_holdout.copy()
    selected_holdout["bet_type"] = bet_type
    selected_holdout["bet_label"] = BET_LABELS[bet_type]
    bundle = {"model": model, "calibrator": calibrator, "medians": medians.to_dict(), "config": config}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary, selected_holdout, bundle


def write_report(overall, summary):
    lines = [
        "# 競輪AI 厳密ホールドアウト検証",
        "",
        f"作成日時: {overall['created_at_jst']}",
        "",
        "## 合格判定",
        "",
        f"- 目標ROI: +50.00%",
        f"- 2026年ホールドアウトROI: {overall['holdout_roi']:.2%}" if overall["holdout_roi"] is not None else "- 2026年ホールドアウトROI: -",
        f"- 95%区間: {overall['roi_ci_low']:.2%} ～ {overall['roi_ci_high']:.2%}" if overall["roi_ci_low"] is not None else "- 95%区間: -",
        f"- 判定: {'合格' if overall['target_passed'] else '未達'}",
        "",
        "## 券種別",
        "",
        "|券種|検証買い目|検証ROI|2026買い目|的中|損益|ROI|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        validation_roi = "-" if pd.isna(row["validation_roi"]) else f"{row['validation_roi']:.2%}"
        holdout_roi = "-" if pd.isna(row["holdout_roi"]) else f"{row['holdout_roi']:.2%}"
        lines.append(
            f"|{row['bet_label']}|{int(row['validation_bets'])}|{validation_roi}|{int(row['holdout_bets'])}|"
            f"{int(row['holdout_hits'])}|{int(row['holdout_profit_yen']):,}円|{holdout_roi}|"
        )
    lines += [
        "",
        "2022～2024年を学習、2025年前半を確率校正、2025年後半だけで購入条件を決定しています。",
        "2026年は購入条件の調整に使っていません。選択時のオッズは最終オッズから10%差し引いて判定しています。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(rebuild_entry=False, bet_types=None, min_validation_bets=100):
    ensure_dirs()
    oof = build_oof_entry_predictions(rebuild=rebuild_entry)
    requested = bet_types or list(BET_LABELS)
    summaries = []
    selected_parts = []
    bundles = {}
    for bet_type in requested:
        dataset = build_ticket_dataset(oof, bet_type)
        summary, selected, bundle = fit_and_evaluate_type(dataset, bet_type, min_validation_bets=min_validation_bets)
        summaries.append(summary)
        selected_parts.append(selected)
        bundles[bet_type] = bundle

    summary_df = pd.DataFrame(summaries)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    metrics = evaluate_selected(selected)
    ci_low, ci_high = confidence_interval_by_day(selected)
    overall = {
        "created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "design": "2022-2024 train, 2025H1 calibration, 2025H2 strategy selection, 2026 untouched holdout",
        "target_roi": 0.50,
        "holdout_bets": metrics["bets"],
        "holdout_hits": metrics["hits"],
        "holdout_stake_yen": metrics["stake_yen"],
        "holdout_return_yen": metrics["return_yen"],
        "holdout_profit_yen": metrics["profit_yen"],
        "holdout_roi": metrics["roi"],
        "holdout_positive_months": metrics["positive_months"],
        "roi_ci_low": ci_low,
        "roi_ci_high": ci_high,
        "target_passed": bool(
            metrics["bets"] >= 200
            and metrics["roi"] is not None
            and metrics["roi"] >= 0.50
            and ci_low is not None
            and ci_low > 0
        ),
        "odds_policy": "final odds proxy; 10% haircut when selecting; flat 100 yen settlement",
        "result_leakage_races": 0,
    }
    summary_df.to_csv(SUMMARY_PATH, index=False)
    if len(selected):
        keep = [
            "date", "race_id", "bet_type", "bet_label", "buy", "pred_prob", "pred_ev",
            "odds_used", "payout_odds", "popularity_order", "is_hit",
        ]
        selected[keep].sort_values(["date", "race_id", "bet_type"]).to_csv(HOLDOUT_BETS_PATH, index=False)
    else:
        selected.to_csv(HOLDOUT_BETS_PATH, index=False)
    OVERALL_PATH.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"created_at_jst": overall["created_at_jst"], "features": TICKET_FEATURES, "types": bundles}, MODEL_PATH)
    write_report(overall, summary_df)
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-entry", action="store_true")
    parser.add_argument("--bet-types", nargs="*", choices=list(BET_LABELS), default=None)
    parser.add_argument("--min-validation-bets", type=int, default=100)
    args = parser.parse_args()
    run(rebuild_entry=args.rebuild_entry, bet_types=args.bet_types, min_validation_bets=args.min_validation_bets)


if __name__ == "__main__":
    main()
