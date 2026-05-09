from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from Utils.config import DATA_ROOT
from Utils.feature_extraction import space_quality
from Utils.loading import load_files, load_game_from_pff


REQUIRED_COLUMNS = [
    "source_game_file",
    "t0_startFrame_nextGameEvent",
    "periodGameClockTime_t0_nextGameEvent",
    "teamName_t0_nextGameEvent",
    "homeBall_t0_nextGameEvent",
]


def _parse_bool_like(value: Any) -> bool:
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


def avg_recovery_space_gain(
    recovery_df: pd.DataFrame,
    base_path: str = DATA_ROOT,
    model_rel_path: str = "pitch_value_model/models/pv_mlp.pkl",
) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in recovery_df.columns]
    if missing:
        raise ValueError(f"recovery_df missing required columns: {missing}")

    results: list[dict[str, Any]] = []
    grouped = recovery_df.groupby("source_game_file", dropna=False)

    for game_id_raw, group in grouped:
        game_id = str(game_id_raw)
        if game_id.lower() == "nan":
            for _, row in group.iterrows():
                results.append(
                    {
                        "source_game_file": row.get("source_game_file"),
                        "t0_startFrame_nextGameEvent": row.get("t0_startFrame_nextGameEvent"),
                        "periodGameClockTime_t0_nextGameEvent": row.get("periodGameClockTime_t0_nextGameEvent"),
                        "teamName_t0_nextGameEvent": row.get("teamName_t0_nextGameEvent"),
                        "homeBall_t0_nextGameEvent": row.get("homeBall_t0_nextGameEvent"),
                        "frame_idx": pd.NA,
                        "pc_mean": pd.NA,
                        "pv_mean": pd.NA,
                        "sq_mean": pd.NA,
                        "sq_max": pd.NA,
                        "status": "skipped",
                        "error": "missing_game_id",
                    }
                )
            continue

        try:
            game = load_game_from_pff(base_path=base_path, game_id=game_id)
            _, tracking_df = load_files(base_path=base_path, game_id=game_id)
        except Exception as exc:
            for _, row in group.iterrows():
                results.append(
                    {
                        "source_game_file": row.get("source_game_file"),
                        "t0_startFrame_nextGameEvent": row.get("t0_startFrame_nextGameEvent"),
                        "periodGameClockTime_t0_nextGameEvent": row.get("periodGameClockTime_t0_nextGameEvent"),
                        "teamName_t0_nextGameEvent": row.get("teamName_t0_nextGameEvent"),
                        "homeBall_t0_nextGameEvent": row.get("homeBall_t0_nextGameEvent"),
                        "frame_idx": pd.NA,
                        "pc_mean": pd.NA,
                        "pv_mean": pd.NA,
                        "sq_mean": pd.NA,
                        "sq_max": pd.NA,
                        "status": "skipped",
                        "error": f"game_load_failed: {exc}",
                    }
                )
            continue

        for _, row in group.iterrows():
            base_result = {
                "source_game_file": row.get("source_game_file"),
                "t0_startFrame_nextGameEvent": row.get("t0_startFrame_nextGameEvent"),
                "periodGameClockTime_t0_nextGameEvent": row.get("periodGameClockTime_t0_nextGameEvent"),
                "teamName_t0_nextGameEvent": row.get("teamName_t0_nextGameEvent"),
                "homeBall_t0_nextGameEvent": row.get("homeBall_t0_nextGameEvent"),
            }
            try:
                frame_num = int(float(row["t0_startFrame_nextGameEvent"]))
            except Exception:
                results.append(
                    {
                        **base_result,
                        "frame_idx": pd.NA,
                        "pc_mean": pd.NA,
                        "pv_mean": pd.NA,
                        "sq_mean": pd.NA,
                        "sq_max": pd.NA,
                        "status": "skipped",
                        "error": "invalid_t0_frame",
                    }
                )
                continue

            try:
                home_team_in_possession = _parse_bool_like(row["homeBall_t0_nextGameEvent"])
            except Exception as exc:
                results.append(
                    {
                        **base_result,
                        "frame_idx": pd.NA,
                        "pc_mean": pd.NA,
                        "pv_mean": pd.NA,
                        "sq_mean": pd.NA,
                        "sq_max": pd.NA,
                        "status": "skipped",
                        "error": f"invalid_homeBall_t0_nextGameEvent: {exc}",
                    }
                )
                continue

            idx = game.tracking_data.index[game.tracking_data["frame"] == frame_num]
            if len(idx) == 0:
                results.append(
                    {
                        **base_result,
                        "frame_idx": pd.NA,
                        "pc_mean": pd.NA,
                        "pv_mean": pd.NA,
                        "sq_mean": pd.NA,
                        "sq_max": pd.NA,
                        "status": "skipped",
                        "error": "frame_not_found_in_game_tracking_data",
                    }
                )
                continue

            frame_idx = int(idx[0])
            frame_row = game.tracking_data.loc[frame_idx]

            if "frameNum" in tracking_df.columns:
                _ = tracking_df[tracking_df["frameNum"] == frame_num]

            try:
                pc_att, pv, sq = space_quality(
                    frame_row=frame_row,
                    game=game,
                    frame_idx=frame_idx,
                    home_team_in_possession=home_team_in_possession,
                    base_path=base_path,
                    model_rel_path=model_rel_path,
                )
            except Exception as exc:
                results.append(
                    {
                        **base_result,
                        "frame_idx": frame_idx,
                        "pc_mean": pd.NA,
                        "pv_mean": pd.NA,
                        "sq_mean": pd.NA,
                        "sq_max": pd.NA,
                        "status": "skipped",
                        "error": f"space_quality_failed: {exc}",
                    }
                )
                continue

            results.append(
                {
                    **base_result,
                    "frame_idx": frame_idx,
                    "pc_mean": float(np.mean(pc_att)),
                    "pv_mean": float(np.mean(pv)),
                    "sq_mean": float(np.mean(sq)),
                    "sq_max": float(np.max(sq)),
                    "status": "ok",
                    "error": pd.NA,
                }
            )

    return pd.DataFrame(results)
