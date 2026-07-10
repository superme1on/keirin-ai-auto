import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


JST = ZoneInfo("Asia/Tokyo")


def _number(value, default=np.nan):
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) else float(parsed)


def _rate(value):
    parsed = _number(value)
    return parsed / 100.0 if pd.notna(parsed) else np.nan


def _race_date_from_id(race_id):
    match = re.search(r"(20\d{6})", str(race_id or ""))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def prior_results(record, race_date, current_race_id=None, limit=12):
    current = datetime.strptime(str(race_date), "%Y-%m-%d").date()
    seen = set()
    results = []

    sources = [record.get("currentCupResults", []), record.get("previousCupResults", [])]
    for cup in record.get("latestCupResults", []) or []:
        sources.append(cup.get("raceResults", []))

    for source in sources:
        for result in source or []:
            race_id = str(result.get("raceId", ""))
            if not race_id or race_id == str(current_race_id) or race_id in seen:
                continue
            event_date = _race_date_from_id(race_id)
            if event_date is None or event_date >= current:
                continue
            seen.add(race_id)
            results.append((event_date, result))

    results.sort(key=lambda item: (item[0], str(item[1].get("raceId", ""))), reverse=True)
    return [result for _, result in results[:limit]]


def recent_avg_finish(record, race_date, current_race_id=None):
    orders = []
    for result in prior_results(record, race_date, current_race_id=current_race_id, limit=10):
        order = _number(result.get("order"))
        if pd.notna(order) and order > 0:
            orders.append(order)
    return float(np.mean(orders)) if orders else np.nan


def current_cup_avg_finish(record):
    orders = []
    for result in record.get("currentCupResults", []) or []:
        order = _number(result.get("order"))
        if pd.notna(order) and order > 0:
            orders.append(order)
    return float(np.mean(orders)) if orders else np.nan


def days_since_last_race(record, race_date, current_race_id=None):
    results = prior_results(record, race_date, current_race_id=current_race_id, limit=1)
    if not results:
        return np.nan
    previous = _race_date_from_id(results[0].get("raceId"))
    current = datetime.strptime(str(race_date), "%Y-%m-%d").date()
    return max((current - previous).days, 0) if previous else np.nan


def _summary_rates(summary):
    summary = summary if isinstance(summary, dict) else {}
    return {
        "win": _rate(summary.get("firstPercentage")),
        "place2": _rate(summary.get("secondPercentage")),
        "place3": _rate(summary.get("thirdPercentage")),
        "races": _number(summary.get("total"), 0.0),
    }


def _track_summary_key(distance):
    distance = _number(distance)
    if pd.isna(distance):
        return None
    if distance < 375:
        return "trackDistance333"
    if distance >= 450:
        return "trackDistance500"
    return "trackDistance400"


def _weather_summary_key(weather):
    return {
        "\u6674": "weatherSunny",
        "\u6674\u308c": "weatherSunny",
        "\u66c7": "weatherCloudy",
        "\u66c7\u308a": "weatherCloudy",
        "\u96e8": "weatherRainy",
        "\u96ea": "weatherSnowy",
    }.get(str(weather or ""))


def _race_type_summary_key(race_type):
    race_type = str(race_type or "")
    if "\u4e88\u9078" in race_type:
        return "raceTypeQualifyingRound"
    if "\u6e96\u6c7a" in race_type:
        return "raceTypeSemifinal"
    if "\u6c7a\u52dd" in race_type:
        return "raceTypeFinal"
    if "\u4e00\u822c" in race_type:
        return "raceTypeLoserRound"
    return "raceTypeSpecial"


def _hour_summary_key(start_at):
    start_at = _number(start_at)
    if pd.isna(start_at):
        return None
    hour = datetime.fromtimestamp(start_at, JST).hour
    if hour < 11:
        return "hourTypeMorning"
    if hour >= 21:
        return "hourTypeMidnight"
    if hour >= 16:
        return "hourTypeNight"
    return "hourTypeNormal"


def line_features(line_prediction):
    prediction = line_prediction if isinstance(line_prediction, dict) else {}
    lines = prediction.get("lines", []) or []
    mapped = {}
    for line_id, line in enumerate(lines, start=1):
        groups = line.get("entries", []) or []
        cars = []
        for group in groups:
            for number in group.get("numbers", []) or []:
                try:
                    cars.append(int(number))
                except (TypeError, ValueError):
                    continue
        for position, car_no in enumerate(cars, start=1):
            mapped[car_no] = {
                "line_id": line_id,
                "line_position": position,
                "line_size": len(cars),
                "is_line_leader": int(position == 1),
            }
    return mapped, len(lines), prediction.get("lineType", "")


def _line_history_key(position, size):
    if size == 1:
        return "lineSingleHorseman"
    if position == 1:
        return "linePositionFirst"
    if position == 2:
        return "linePositionSecond"
    if pd.notna(position):
        return "linePositionThird"
    return None


def build_entry_rows(race_data, race_date, venue, race_no, race_id, source_url, include_results):
    race = race_data.get("race", {})
    schedule = race_data.get("schedule", {})
    players = {str(player.get("id")): player for player in race_data.get("players", [])}
    records = {str(record.get("playerId")): record for record in race_data.get("records", [])}
    results = {str(result.get("playerId")): result for result in race_data.get("results", []) or []}
    entries = [entry for entry in race_data.get("entries", []) if not entry.get("absent")]
    field_size = len(entries)
    lines, number_of_lines, line_type = line_features(race_data.get("linePrediction"))

    weather = race.get("weather", "")
    race_type = race.get("raceType", "")
    race_type_short = race.get("raceType3", race_type)
    track_key = _track_summary_key(race.get("distance"))
    weather_key = _weather_summary_key(weather)
    race_type_key = _race_type_summary_key(race_type_short)
    hour_key = _hour_summary_key(race.get("startAt"))

    rows = []
    for entry in entries:
        player_id = str(entry.get("playerId", ""))
        player = players.get(player_id, {})
        record = records.get(player_id, {})
        car_no = int(entry.get("number"))
        line = lines.get(car_no, {})
        line_history_key = _line_history_key(line.get("line_position"), line.get("line_size"))

        track = _summary_rates(record.get(track_key)) if track_key else _summary_rates({})
        weather_stats = _summary_rates(record.get(weather_key)) if weather_key else _summary_rates({})
        race_type_stats = _summary_rates(record.get(race_type_key))
        hour_stats = _summary_rates(record.get(hour_key)) if hour_key else _summary_rates({})
        line_stats = _summary_rates(record.get(line_history_key)) if line_history_key else _summary_rates({})

        result = results.get(player_id, {}) if include_results else {}
        raw_order = _number(result.get("order")) if result else np.nan
        result_available = int(include_results and bool(result))
        if include_results:
            finish_pos = int(raw_order) if pd.notna(raw_order) and raw_order > 0 else field_size + 1
        else:
            finish_pos = np.nan

        recent = prior_results(record, race_date, current_race_id=race_id, limit=10)
        rows.append(
            {
                "race_id": str(race_id),
                "date": race_date,
                "venue": venue,
                "race_no": int(race_no),
                "player_id": player_id,
                "car_no": car_no,
                "bracket_no": entry.get("bracketNumber", np.nan),
                "age": player.get("age", np.nan),
                "gender": player.get("gender", ""),
                "term": player.get("term", np.nan),
                "prefecture": player.get("prefecture", ""),
                "region_id": player.get("regionId", np.nan),
                "player_class": player.get("class", np.nan),
                "player_group": player.get("group", np.nan),
                "score": record.get("racePoint", np.nan),
                "win_rate": _rate(record.get("firstRate")),
                "place2_rate": _rate(record.get("secondRate")),
                "place3_rate": _rate(record.get("thirdRate")),
                "back_count": record.get("back", np.nan),
                "standing_count": record.get("standing", np.nan),
                "front_runner_count": record.get("frontRunner", np.nan),
                "stalker_count": record.get("stalker", np.nan),
                "deep_closer_count": record.get("deepCloser", np.nan),
                "marker_count": record.get("marker", np.nan),
                "gear_ratio": record.get("gearRatio", np.nan),
                "prediction_mark": record.get("predictionMark", np.nan),
                "style": record.get("style", ""),
                "recent_avg_finish": recent_avg_finish(record, race_date, current_race_id=race_id),
                "recent_races_count": len(recent),
                "current_cup_avg_finish": current_cup_avg_finish(record),
                "days_since_last_race": days_since_last_race(record, race_date, current_race_id=race_id),
                "venue_win_rate": np.nan,
                "track_win_rate": track["win"],
                "track_place2_rate": track["place2"],
                "track_place3_rate": track["place3"],
                "track_races": track["races"],
                "weather_win_rate": weather_stats["win"],
                "weather_place2_rate": weather_stats["place2"],
                "weather_place3_rate": weather_stats["place3"],
                "weather_races": weather_stats["races"],
                "race_type_win_rate": race_type_stats["win"],
                "race_type_place2_rate": race_type_stats["place2"],
                "race_type_place3_rate": race_type_stats["place3"],
                "race_type_races": race_type_stats["races"],
                "hour_win_rate": hour_stats["win"],
                "hour_place2_rate": hour_stats["place2"],
                "hour_place3_rate": hour_stats["place3"],
                "hour_races": hour_stats["races"],
                "line_role_win_rate": line_stats["win"],
                "line_role_place2_rate": line_stats["place2"],
                "line_role_place3_rate": line_stats["place3"],
                "line_role_races": line_stats["races"],
                "line_id": line.get("line_id", np.nan),
                "line_position": line.get("line_position", np.nan),
                "line_size": line.get("line_size", np.nan),
                "is_line_leader": line.get("is_line_leader", np.nan),
                "number_of_lines": number_of_lines,
                "line_type": line_type,
                "weather": weather,
                "wind_speed": _number(race.get("windSpeed")),
                "meeting_day": schedule.get("day", schedule.get("index", np.nan)),
                "entries_number": race.get("entriesNumber", field_size),
                "is_grade_race": int(bool(race.get("isGradeRace", False))),
                "start_at": race.get("startAt", np.nan),
                "close_at": race.get("closeAt", np.nan),
                "current_term_class": entry.get("playerCurrentTermClass", np.nan),
                "current_term_group": entry.get("playerCurrentTermGroup", np.nan),
                "previous_term_class": entry.get("playerPreviousTermClass", np.nan),
                "previous_term_group": entry.get("playerPreviousTermGroup", np.nan),
                "odds_win": np.nan,
                "finish_pos": finish_pos,
                "official_finish_pos": raw_order,
                "result_available": result_available,
                "is_dnf": int(include_results and (pd.isna(raw_order) or raw_order <= 0)),
                "result_status": result.get("accidentName") or result.get("accident") or "",
                "source_url": source_url,
                "player_name": player.get("name", ""),
                "race_class": race.get("class", ""),
                "race_type": race_type,
                "distance": race.get("distance", np.nan),
            }
        )
    return rows
