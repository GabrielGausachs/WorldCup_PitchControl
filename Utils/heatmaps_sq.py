from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.config import DATA_ROOT
from Utils.feature_extraction import space_quality
from Utils.loading import load_game_from_pff
from Utils.visualizations import save_space_quality_heatmaps

HOME_TEAM_COLOR = "#225ea8"
AWAY_TEAM_COLOR = "#cb181d"
BALL_COLOR = "#ffffff"
BALL_EDGE_COLOR = "#000000"


def _parse_bool_like(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
    raise ValueError(f"Could not parse boolean value: {value}")


def run_random_space_quality_analysis(
    dataset_recovery: pd.DataFrame,
    base_path: str = DATA_ROOT,
    seed: int = 42,
    output_dir: str = "Outputs",
) -> dict:
    required = {"source_game_file", "startFrame_possessionEventId", "homeTeam"}
    missing = required - set(dataset_recovery.columns)
    if missing:
        raise ValueError(f"dataset_recovery missing required columns: {sorted(missing)}")

    if dataset_recovery.empty:
        raise ValueError("dataset_recovery is empty.")

    valid_rows = dataset_recovery.dropna(subset=["source_game_file", "startFrame_possessionEventId", "homeTeam"]).copy()
    valid_rows["homeTeam_bool"] = valid_rows["homeTeam"].apply(_parse_bool_like)
    valid_rows = valid_rows[valid_rows["homeTeam_bool"]]
    if valid_rows.empty:
        raise ValueError("No valid home-team rows with non-null source_game_file/startFrame_possessionEventId/homeTeam.")

    rng = random.Random(seed)
    sample_idx = rng.choice(valid_rows.index.tolist())
    row = valid_rows.loc[sample_idx]

    game_id = str(row["source_game_file"])
    frame_num = int(row["startFrame_possessionEventId"])
    home_ball = True

    game = load_game_from_pff(base_path, game_id)
    idx = game.tracking_data.index[game.tracking_data["frame"] == frame_num]
    if len(idx) == 0:
        raise ValueError(f"Frame {frame_num} not found in game {game_id}.")
    frame_idx = int(idx[0])
    frame_row = game.tracking_data.loc[frame_idx]

    possession_team = row["teamName"] if "teamName" in row.index else None
    next_event_team = row["teamName_t0_nextGameEvent"] if "teamName_t0_nextGameEvent" in row.index else None
    print("Selected sample for SQ heatmaps:")
    print(f"- game_id: {game_id}")
    print(f"- frame_num: {frame_num}")
    print(f"- home_team_in_possession: {home_ball}")
    print(f"- possession_team_at_frame: {possession_team}")
    print(f"- next_event_team: {next_event_team}")
    print("Heatmap overlay color mapping:")
    print(f"- ball fill: {BALL_COLOR}")
    print(f"- ball edge: {BALL_EDGE_COLOR}")

    pc, pv, sq = space_quality(
        frame_row=frame_row,
        game=game,
        frame_idx=frame_idx,
        home_team_in_possession=home_ball,
        base_path=base_path,
    )

    prefix = f"{game_id}_frame_{frame_num}"
    files = save_space_quality_heatmaps(
        pitch_control=pc,
        pitch_value=pv,
        space_quality=sq,
        out_dir=output_dir,
        prefix=prefix,
        pitch_length=float(game.pitch_dimensions[0]),
        pitch_width=float(game.pitch_dimensions[1]),
        frame_row=frame_row,
        player_ids=game.get_column_ids(),
        home_team_in_possession=home_ball,
    )

    return {
        "sample_index": int(sample_idx),
        "game_id": game_id,
        "frame_num": frame_num,
        "home_ball_possession": home_ball,
        "saved_files": [str(Path(p)) for p in files],
        "shape": pc.shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PC/PV/SQ heatmaps for one random home-team recovery event."
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional full path to dataset_recovery CSV. Defaults to DATA_ROOT/dataset_passes_recovery/dataset_passes_recovery_all_matches.csv",
    )
    parser.add_argument(
        "--base-path",
        default=DATA_ROOT,
        help="Data root containing metadata/rosters/tracking files (default: DATA_ROOT).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting one recovery row.",
    )
    parser.add_argument(
        "--output-dir",
        default="Outputs",
        help="Directory where heatmaps are saved.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = str(
            Path(args.base_path)
            / "dataset_passes_recovery"
            / "dataset_passes_recovery_all_matches.csv"
        )

    dataset_recovery = pd.read_csv(dataset_path)
    result = run_random_space_quality_analysis(
        dataset_recovery=dataset_recovery,
        base_path=args.base_path,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
