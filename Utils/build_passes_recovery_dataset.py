import argparse
import json
import os
from typing import Any

import pandas as pd

from Utils.config import DATA_ROOT

MAX_RECOVERY_GAP_SECONDS = 15.0

def _safe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _extract_home_ball(game_event_obj: Any) -> Any:
    game_event = _safe_dict(game_event_obj)
    return game_event.get("home_ball", pd.NA)


def _extract_team_name(game_event_obj: Any) -> Any:
    game_event = _safe_dict(game_event_obj)
    return game_event.get("team_name", pd.NA)

def _extract_start_frame(game_event_obj: Any) -> Any:
    game_event = _safe_dict(game_event_obj)
    return game_event.get("start_frame", pd.NA)

def _extract_possession_start_frame(possession_event_obj: Any) -> Any:
    possession_event = _safe_dict(possession_event_obj)
    return possession_event.get("start_frame", pd.NA)


def build_base_dataset(base_path: str) -> str:
    events_dir = os.path.join(base_path, "eventdata")
    output_dir = os.path.join(base_path, "dataset_passes_recovery")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "dataset_passes_recovery_all_matches.csv")

    rows: list[dict] = []
    event_files = sorted([f for f in os.listdir(events_dir) if f.endswith(".json")])

    for file_name in event_files:
        game_file_id = os.path.splitext(file_name)[0]
        file_path = os.path.join(events_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            events = json.load(f)
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue
            possession_event = _safe_dict(event.get("possessionEvents"))
            game_event = _safe_dict(event.get("gameEvents"))
            if possession_event.get("passOutcomeType") != "D":
                continue

            rows.append(
                {
                    "source_game_file": game_file_id,
                    "gameId": event.get("gameId"),
                    "gameEventId": event.get("gameEventId"),
                    "possessionEventId": event.get("possessionEventId"),
                }
            )

    dataset = pd.DataFrame(
        rows,
        columns=[
            "source_game_file",
            "gameId",
            "gameEventId",
            "possessionEventId",
        ],
    )
    dataset.to_csv(output_path, index=False)
    print(f"Base dataset saved: {output_path} ({len(dataset)} rows)")
    return output_path


def enrich_with_tracking(dataset_path: str, base_path: str) -> None:
    dataset = pd.read_csv(dataset_path)

    dataset["t0_startFrame_nextGameEvent"] = pd.NA
    dataset["periodGameClockTime_possessionEventId"] = pd.NA
    dataset["periodGameClockTime_t0_nextGameEvent"] = pd.NA
    dataset["startFrame_possessionEventId"] = pd.NA
    dataset["homeTeam"] = pd.NA
    dataset["teamName"] = pd.NA
    dataset["homeBall_t0_nextGameEvent"] = pd.NA
    dataset["teamName_t0_nextGameEvent"] = pd.NA

    for game_file_id, idxs in dataset.groupby("source_game_file").groups.items():
        tracking_path = os.path.join(base_path, "trackingdata_parquet", f"{game_file_id}.parquet")
        tracking_df = pd.read_parquet(
            tracking_path,
            columns=[
                "game_event_id",
                "possession_event_id",
                "frameNum",
                "periodGameClockTime",
                "game_event",
                "possession_event",
            ],
        )
        tracking_df = tracking_df.sort_values("frameNum").reset_index(drop=True)

        for idx in idxs:
            row = dataset.loc[idx]
            pos_id = row["possessionEventId"]
            game_event_id = row["gameEventId"]

            pos_rows = tracking_df[tracking_df["possession_event_id"] == pos_id]
            if pos_rows.empty:
                continue

            start_frame_val = _extract_possession_start_frame(pos_rows.iloc[0].get("possession_event"))
            if pd.isna(start_frame_val):
                continue
            start_frame = int(start_frame_val)
            dataset.at[idx, "startFrame_possessionEventId"] = start_frame

            pos_start_rows = pos_rows[pos_rows["frameNum"] == start_frame]
            if not pos_start_rows.empty:
                pos_start_row = pos_start_rows.iloc[0]
                dataset.at[idx, "periodGameClockTime_possessionEventId"] = pos_start_row["periodGameClockTime"]
                dataset.at[idx, "homeTeam"] = _extract_home_ball(pos_start_row["game_event"])
                dataset.at[idx, "teamName"] = _extract_team_name(pos_start_row["game_event"])

            future_rows = tracking_df[tracking_df["frameNum"] >= start_frame]
            next_event_rows = future_rows[
                (future_rows["game_event_id"].notna())
                & (future_rows["game_event_id"] != "")
                & (future_rows["game_event_id"] != game_event_id)
            ]
            if next_event_rows.empty:
                continue

            t0_row = next_event_rows.iloc[0]
            dataset.at[idx, "t0_startFrame_nextGameEvent"] = _extract_start_frame(t0_row["game_event"])
            dataset.at[idx, "periodGameClockTime_t0_nextGameEvent"] = t0_row["periodGameClockTime"]
            dataset.at[idx, "homeBall_t0_nextGameEvent"] = _extract_home_ball(t0_row["game_event"])
            dataset.at[idx, "teamName_t0_nextGameEvent"] = _extract_team_name(t0_row["game_event"])

    # Keep only recovery candidates where possession team changes at the next game event.
    dataset = dataset[
        dataset["teamName"].notna()
        & dataset["teamName_t0_nextGameEvent"].notna()
        & (dataset["teamName"] != dataset["teamName_t0_nextGameEvent"])
        & (dataset["homeTeam"].notna())
        & (dataset["homeBall_t0_nextGameEvent"].notna())
    ].copy()

    # Keep only recoveries where the next game event happens quickly enough.
    dataset["periodGameClockTime_possessionEventId"] = pd.to_numeric(
        dataset["periodGameClockTime_possessionEventId"], errors="coerce"
    )
    dataset["periodGameClockTime_t0_nextGameEvent"] = pd.to_numeric(
        dataset["periodGameClockTime_t0_nextGameEvent"], errors="coerce"
    )
    dataset["delta_seconds_to_next_game_event"] = (
        dataset["periodGameClockTime_t0_nextGameEvent"]
        - dataset["periodGameClockTime_possessionEventId"]
    )
    dataset = dataset[
        dataset["delta_seconds_to_next_game_event"].notna()
        & (dataset["delta_seconds_to_next_game_event"] < MAX_RECOVERY_GAP_SECONDS)
    ].copy()

    dataset.to_csv(dataset_path, index=False)
    print(f"Enriched dataset saved: {dataset_path} ({len(dataset)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build D-pass dataset, then enrich with tracking-based start frame/t0/periodGameClockTime."
    )
    parser.add_argument(
        "--base-path",
        default=DATA_ROOT,
        help="Root path containing eventdata and trackingdata_parquet (default: DATA_ROOT).",
    )
    args = parser.parse_args()

    dataset_path = build_base_dataset(args.base_path)
    enrich_with_tracking(dataset_path, args.base_path)


if __name__ == "__main__":
    main()
