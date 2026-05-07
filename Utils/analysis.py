from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.config import DATA_ROOT
from Utils.feature_extraction import space_quality
from Utils.loading import load_game_from_pff
from Utils.visualizations import save_space_quality_heatmaps


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
    required = {"source_game_file", "startFrame_possessionEventId", "homeBall_possessionEventId"}
    missing = required - set(dataset_recovery.columns)
    if missing:
        raise ValueError(f"dataset_recovery missing required columns: {sorted(missing)}")

    if dataset_recovery.empty:
        raise ValueError("dataset_recovery is empty.")

    rng = random.Random(seed)
    sample_idx = rng.choice(dataset_recovery.index.tolist())
    row = dataset_recovery.loc[sample_idx]

    game_id = str(row["source_game_file"])
    frame_num = int(row["startFrame_possessionEventId"])
    home_ball = _parse_bool_like(row["homeBall_possessionEventId"])

    game = load_game_from_pff(base_path, game_id)
    idx = game.tracking_data.index[game.tracking_data["frame"] == frame_num]
    if len(idx) == 0:
        raise ValueError(f"Frame {frame_num} not found in game {game_id}.")
    frame_idx = int(idx[0])
    frame_row = game.tracking_data.loc[frame_idx]

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
