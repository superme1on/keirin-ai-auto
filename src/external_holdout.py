import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from build_history import ODDS_COLUMNS, collect_all_race_urls, collect_cups, fetch_history_races
from common import FEATURE_COLS, HISTORY_CSV, MODEL_DIR, OUTPUT_DIR, ROOT, add_player_prior_features, prepare_features
from honest_backtest import (
    BET_LABELS,
    MODEL_PATH,
    POPULARITY_LIMITS,
    TICKET_FEATURES,
    apply_platt,
    apply_strategy,
    build_ticket_dataset,
    confidence_interval_by_day,
    entry_model,
    evaluate_selected,
    normalize_to_race_sum,
)


HOLDOUT_DIR = ROOT / "data" / "external_holdout"
ENTRIES_PATH = HOLDOUT_DIR / "entries.csv"
ODDS_PATH = HOLDOUT_DIR / "odds_top.csv"
FETCH_METADATA_PATH = HOLDOUT_DIR / "fetch_metadata.json"
ENTRY_PREDICTIONS_PATH = HOLDOUT_DIR / "entry_predictions.csv"
OPENED_PATH = HOLDOUT_DIR / "opened.json"
SUMMARY_PATH = OUTPUT_DIR / "external_holdout_summary.csv"
BETS_PATH = OUTPUT_DIR / "external_holdout_bets.csv"
OVERALL_PATH = OUTPUT_DIR / "external_holdout_overall.json"
REPORT_PATH = OUTPUT_DIR / "external_holdout_report.md"


def append_csv(frame, path):
    if len(frame) == 0:
        return
    frame.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def existing_urls(path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    return set(pd.read_csv(path, usecols=["source_url"])["source_url"].dropna().astype(str))


def fetch_external_holdout(months_back=19, workers=24, chunk_size=200, max_races=None):
    HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    main_urls = set(pd.read_csv(HISTORY_CSV, usecols=["source_url"])["source_url"].dropna().astype(str))
    done_urls = existing_urls(ENTRIES_PATH)
    cups = collect_cups(months_back=months_back)
    urls, cup_failures = collect_all_race_urls(cups, workers=workers, progress_every=100)
    unseen = sorted(set(urls) - main_urls - done_urls)
    if max_races:
        unseen = unseen[: int(max_races)]

    fetched = 0
    saved_races = 0
    saved_rows = 0
    saved_odds = 0
    failures = []
    for batch in chunked(unseen, chunk_size):
        rows, odds_rows, batch_failures = fetch_history_races(
            batch,
            sleep_sec=0,
            use_cache=True,
            workers=workers,
            progress_every=100,
        )
        failures.extend(batch_failures)
        entries = pd.DataFrame(rows)
        odds = pd.DataFrame(odds_rows, columns=ODDS_COLUMNS)
        if len(entries):
            entries = entries[pd.to_datetime(entries["date"], errors="coerce").ge("2025-01-01")]
            entries = entries.drop_duplicates(["race_id", "player_id"])
        if len(odds):
            odds = odds[pd.to_datetime(odds["date"], errors="coerce").ge("2025-01-01")]
            odds["popularity_order"] = pd.to_numeric(odds["popularity_order"], errors="coerce")
            limit = odds["bet_type"].map(POPULARITY_LIMITS)
            odds = odds[limit.notna() & odds["popularity_order"].le(limit)]
            odds = odds.drop_duplicates(["race_id", "bet_type", "buy"])
        append_csv(entries, ENTRIES_PATH)
        append_csv(odds, ODDS_PATH)
        fetched += len(batch)
        saved_races += int(entries["race_id"].nunique()) if len(entries) else 0
        saved_rows += len(entries)
        saved_odds += len(odds)
        print(
            f"external fetch: {fetched}/{len(unseen)} races={saved_races} entries={saved_rows} odds={saved_odds} failures={len(failures)}",
            flush=True,
        )

    metadata = {
        "created_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "purpose": "never-used external holdout; excluded every race URL in the development history",
        "months_back": int(months_back),
        "candidate_urls": int(len(urls)),
        "main_urls_excluded": int(len(main_urls)),
        "previously_downloaded_urls": int(len(done_urls)),
        "new_urls_fetched": int(fetched),
        "saved_races": int(saved_races),
        "saved_entry_rows": int(saved_rows),
        "saved_top_odds_rows": int(saved_odds),
        "cup_failures": cup_failures,
        "race_failures": failures,
    }
    FETCH_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def build_external_entry_predictions(rebuild=False):
    if ENTRY_PREDICTIONS_PATH.exists() and not rebuild:
        return pd.read_csv(ENTRY_PREDICTIONS_PATH, dtype={"race_id": str, "player_id": str})
    main = pd.read_csv(HISTORY_CSV, dtype={"race_id": str, "player_id": str})
    external_true = pd.read_csv(ENTRIES_PATH, dtype={"race_id": str, "player_id": str})
    external_true = external_true.drop_duplicates(["race_id", "player_id"], keep="last")
    true_finish = external_true.set_index(["race_id", "player_id"])["finish_pos"]

    main["_external"] = 0
    external = external_true.copy()
    external["_external"] = 1
    external["finish_pos"] = np.nan
    combined = pd.concat([main, external], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.sort_values(["date", "race_id", "car_no"], kind="mergesort")
    combined = add_player_prior_features(combined)
    combined["score_rank"] = pd.to_numeric(combined["score"], errors="coerce").groupby(combined["race_id"]).rank(
        ascending=False, method="average"
    )
    train_all = combined[combined["_external"].eq(0)].copy()
    finish = pd.to_numeric(train_all["finish_pos"], errors="coerce")
    train_all["target_win"] = finish.eq(1).astype(int)
    train_all["target_top2"] = finish.le(2).astype(int)
    train_all["target_top3"] = finish.le(3).astype(int)
    test_all = combined[combined["_external"].eq(1)].copy()

    folds = []
    for year in sorted(test_all["date"].dt.year.dropna().unique().astype(int)):
        train = train_all[train_all["date"].dt.year < year]
        test = test_all[test_all["date"].dt.year == year].copy()
        if len(train) < 5_000 or len(test) == 0:
            continue
        X_train, fill_values = prepare_features(train)
        X_test, _ = prepare_features(test, fill_values)
        for target, output, target_sum in [
            ("target_win", "p_win", 1),
            ("target_top2", "p_top2", 2),
            ("target_top3", "p_top3", 3),
        ]:
            model = entry_model()
            model.fit(X_train, train[target].astype(int))
            test[output] = normalize_to_race_sum(model.predict_proba(X_test)[:, 1], test["race_id"], target_sum)
        folds.append(test)
        print(f"external entry predictions year={year} train={train['race_id'].nunique()} test={test['race_id'].nunique()}", flush=True)

    predicted = pd.concat(folds, ignore_index=True)
    keys = pd.MultiIndex.from_frame(predicted[["race_id", "player_id"]])
    predicted["finish_pos"] = true_finish.reindex(keys).to_numpy()
    keep = [
        "race_id", "date", "venue", "race_no", "player_id", "car_no", "finish_pos", "score",
        "score_rank", "win_rate", "place2_rate", "place3_rate", "line_id", "line_position",
        "line_size", "style", "distance", "entries_number", "p_win", "p_top2", "p_top3",
    ]
    predicted[keep].to_csv(ENTRY_PREDICTIONS_PATH, index=False)
    return predicted[keep]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_external(rebuild_entries=False, force=False):
    model_hash = file_sha256(MODEL_PATH)
    if OPENED_PATH.exists():
        opened = json.loads(OPENED_PATH.read_text(encoding="utf-8"))
        if opened.get("model_sha256") != model_hash and not force:
            raise ValueError("external holdout has already been opened with another model; refusing post-hoc retuning")

    entry_predictions = build_external_entry_predictions(rebuild=rebuild_entries)
    frozen = joblib.load(MODEL_PATH)
    summaries = []
    selected_parts = []
    for bet_type, bundle in frozen["types"].items():
        config = bundle["config"]
        if float(config.get("min_ev", 999)) >= 900:
            continue
        dataset = build_ticket_dataset(entry_predictions, bet_type, odds_path=ODDS_PATH)
        medians = pd.Series(bundle["medians"])
        X = dataset[TICKET_FEATURES].fillna(medians).to_numpy(float)
        raw = bundle["model"].predict_proba(X)[:, 1]
        dataset["pred_prob"] = apply_platt(bundle["calibrator"], raw, dataset)
        dataset["pred_ev"] = dataset["pred_prob"] * (dataset["odds_used"] * 0.90) - 1
        selected = apply_strategy(dataset, config)
        metrics = evaluate_selected(selected)
        summaries.append(
            {
                "bet_type": bet_type,
                "bet_label": BET_LABELS[bet_type],
                **metrics,
                "min_ev": config.get("min_ev"),
                "min_prob": config.get("min_prob"),
                "max_odds": config.get("max_odds"),
                "daily_cap": config.get("daily_cap"),
                "validation_selection_score": config.get("selection_score"),
            }
        )
        selected["bet_type"] = bet_type
        selected["bet_label"] = BET_LABELS[bet_type]
        selected_parts.append(selected)

    summary = pd.DataFrame(summaries)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    metrics = evaluate_selected(selected)
    ci_low, ci_high = confidence_interval_by_day(selected)
    overall = {
        "opened_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "model_sha256": model_hash,
        "model_created_at_jst": frozen.get("created_at_jst"),
        "external_races": int(entry_predictions["race_id"].nunique()),
        "bets": metrics["bets"],
        "hits": metrics["hits"],
        "stake_yen": metrics["stake_yen"],
        "return_yen": metrics["return_yen"],
        "profit_yen": metrics["profit_yen"],
        "roi": metrics["roi"],
        "positive_months": metrics["positive_months"],
        "roi_ci_low": ci_low,
        "roi_ci_high": ci_high,
        "target_roi": 0.50,
        "target_passed": bool(
            metrics["bets"] >= 200
            and metrics["roi"] is not None
            and metrics["roi"] >= 0.50
            and ci_low is not None
            and ci_low > 0
        ),
    }
    summary.to_csv(SUMMARY_PATH, index=False)
    if len(selected):
        selected[
            ["date", "race_id", "bet_type", "bet_label", "buy", "pred_prob", "pred_ev", "odds_used", "payout_odds", "is_hit"]
        ].sort_values(["date", "race_id"]).to_csv(BETS_PATH, index=False)
    OVERALL_PATH.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    OPENED_PATH.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        "# 外部未使用レース検証",
        "",
        f"- 対象レース: {overall['external_races']:,}",
        f"- 購入: {overall['bets']:,}点",
        f"- 的中: {overall['hits']:,}点",
        f"- 損益: {overall['profit_yen']:,}円",
        f"- ROI: {overall['roi']:.2%}" if overall["roi"] is not None else "- ROI: -",
        f"- 95%区間: {overall['roi_ci_low']:.2%} ～ {overall['roi_ci_high']:.2%}" if overall["roi_ci_low"] is not None else "- 95%区間: -",
        f"- 目標判定: {'合格' if overall['target_passed'] else '未達'}",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--months-back", type=int, default=19)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--max-races", type=int, default=None)
    parser.add_argument("--rebuild-entries", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.fetch and not args.evaluate:
        args.fetch = True
        args.evaluate = True
    if args.fetch:
        fetch_external_holdout(args.months_back, args.workers, args.chunk_size, args.max_races)
    if args.evaluate:
        evaluate_external(rebuild_entries=args.rebuild_entries, force=args.force)


if __name__ == "__main__":
    main()
