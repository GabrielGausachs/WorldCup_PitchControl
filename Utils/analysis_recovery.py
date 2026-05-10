from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from Utils.config import DATA_ROOT
from Utils.loading import load_files, load_game_from_pff


REQUIRED_COLUMNS = [
    "source_game_file",
    "t0_startFrame_nextGameEvent",
    "periodGameClockTime_t0_nextGameEvent",
    "teamName_t0_nextGameEvent",
    "homeBall_t0_nextGameEvent",
]

WINDOW_SECONDS = 5.0
END_CLOCK_TOLERANCE_SECONDS = 0.5
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RADIUS_METERS = 20.0


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


def _append_skipped_result(
    results: list[dict[str, Any]],
    base_result: dict[str, Any],
    error: str,
    frame_num: Any = pd.NA,
    frame_idx: Any = pd.NA,
) -> None:
    results.append(
        {
            **base_result,
            "frame_num": frame_num,
            "frame_idx": frame_idx,
            "pc_mean": pd.NA,
            "pv_mean": pd.NA,
            "sq_mean": pd.NA,
            "sq_max": pd.NA,
            "sq_t0_mean": pd.NA,
            "sq_max_5s": pd.NA,
            "recovery_space_gain": pd.NA,
            "pc_mean_r20": pd.NA,
            "pv_mean_r20": pd.NA,
            "sq_mean_r20": pd.NA,
            "sq_t0_mean_r20": pd.NA,
            "sq_max_5s_r20": pd.NA,
            "recovery_space_gain_r20": pd.NA,
            "window_start_clock": pd.NA,
            "window_target_clock": pd.NA,
            "window_end_clock": pd.NA,
            "window_n_frames": pd.NA,
            "status": "skipped",
            "frame_error": pd.NA,
            "error": error,
        }
    )


def _build_meter_grid(pitch_length: float, pitch_width: float) -> tuple[np.ndarray, np.ndarray]:
    n_x = int(round(pitch_length))
    n_y = int(round(pitch_width))
    x = np.linspace(-pitch_length / 2.0, pitch_length / 2.0, n_x)
    y = np.linspace(-pitch_width / 2.0, pitch_width / 2.0, n_y)
    return np.meshgrid(x, y, indexing="ij")


def _resize_2d_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    x_idx = np.linspace(0, arr.shape[0] - 1, target_shape[0]).astype(int)
    y_idx = np.linspace(0, arr.shape[1] - 1, target_shape[1]).astype(int)
    return arr[np.ix_(x_idx, y_idx)]


def avg_recovery_space_gain(
    recovery_df: pd.DataFrame,
    base_path: str = DATA_ROOT,
    model_rel_path: str = "pitch_value_model/models/pv_mlp.pkl",
) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in recovery_df.columns]
    if missing:
        raise ValueError(f"recovery_df missing required columns: {missing}")

    print(f"[analysis_recovery] Starting run with {len(recovery_df)} recovery rows")
    results: list[dict[str, Any]] = []
    grouped = recovery_df.groupby("source_game_file", dropna=False)
    total_games = len(grouped)
    print(f"[analysis_recovery] Grouped into {total_games} game(s)")

    for game_idx, (game_id_raw, group) in enumerate(grouped, start=1):
        game_id = str(game_id_raw)
        print(
            f"[analysis_recovery] Game {game_idx}/{total_games}: game_id={game_id}, recoveries={len(group)}"
        )
        if game_id.lower() == "nan":
            for _, row in group.iterrows():
                _append_skipped_result(
                    results=results,
                    base_result={
                        "source_game_file": row.get("source_game_file"),
                        "t0_startFrame_nextGameEvent": row.get("t0_startFrame_nextGameEvent"),
                        "periodGameClockTime_t0_nextGameEvent": row.get("periodGameClockTime_t0_nextGameEvent"),
                        "teamName_t0_nextGameEvent": row.get("teamName_t0_nextGameEvent"),
                        "homeBall_t0_nextGameEvent": row.get("homeBall_t0_nextGameEvent"),
                    },
                    error="missing_game_id",
                )
            continue

        try:
            game = load_game_from_pff(base_path=base_path, game_id=game_id)
            _, tracking_df = load_files(base_path=base_path, game_id=game_id)
            print(f"[analysis_recovery] Loaded game resources for {game_id}")
        except Exception as exc:
            print(f"[analysis_recovery] Failed loading game {game_id}: {exc}")
            for _, row in group.iterrows():
                _append_skipped_result(
                    results=results,
                    base_result={
                        "source_game_file": row.get("source_game_file"),
                        "t0_startFrame_nextGameEvent": row.get("t0_startFrame_nextGameEvent"),
                        "periodGameClockTime_t0_nextGameEvent": row.get("periodGameClockTime_t0_nextGameEvent"),
                        "teamName_t0_nextGameEvent": row.get("teamName_t0_nextGameEvent"),
                        "homeBall_t0_nextGameEvent": row.get("homeBall_t0_nextGameEvent"),
                    },
                    error=f"game_load_failed: {exc}",
                )
            continue

        tracking_cols = ["frameNum", "periodGameClockTime"]
        missing_tracking_cols = [c for c in tracking_cols if c not in tracking_df.columns]
        if missing_tracking_cols:
            for _, row in group.iterrows():
                _append_skipped_result(
                    results=results,
                    base_result={
                        "source_game_file": row.get("source_game_file"),
                        "t0_startFrame_nextGameEvent": row.get("t0_startFrame_nextGameEvent"),
                        "periodGameClockTime_t0_nextGameEvent": row.get("periodGameClockTime_t0_nextGameEvent"),
                        "teamName_t0_nextGameEvent": row.get("teamName_t0_nextGameEvent"),
                        "homeBall_t0_nextGameEvent": row.get("homeBall_t0_nextGameEvent"),
                    },
                    error=f"tracking_df_missing_columns: {missing_tracking_cols}",
                )
            continue

        game_frame_to_idx = pd.Series(game.tracking_data.index.values, index=game.tracking_data["frame"]).to_dict()
        tracking_clock_df = tracking_df[tracking_cols].copy()
        tracking_clock_df["frameNum"] = pd.to_numeric(tracking_clock_df["frameNum"], errors="coerce")
        tracking_clock_df["periodGameClockTime"] = pd.to_numeric(
            tracking_clock_df["periodGameClockTime"], errors="coerce"
        )
        tracking_clock_df = tracking_clock_df.dropna(subset=["frameNum", "periodGameClockTime"])
        tracking_clock_df = (
            tracking_clock_df.sort_values("frameNum")
            .drop_duplicates(subset=["frameNum"], keep="first")
            .reset_index(drop=True)
        )
        model_path = Path(model_rel_path)
        if not model_path.is_absolute():
            model_path = REPO_ROOT / model_path
        model = joblib.load(model_path)
        pitch_length, pitch_width = float(game.pitch_dimensions[0]), float(game.pitch_dimensions[1])
        X, Y = _build_meter_grid(pitch_length=pitch_length, pitch_width=pitch_width)
        grid_shape = X.shape
        print(f"[analysis_recovery] Model/grid ready for {game_id}: grid_shape={grid_shape}")

        for recovery_idx, (_, row) in enumerate(group.iterrows(), start=1):
            if recovery_idx == 1 or recovery_idx % 25 == 0 or recovery_idx == len(group):
                print(
                    f"[analysis_recovery] {game_id}: processing recovery {recovery_idx}/{len(group)}"
                )
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
                _append_skipped_result(
                    results=results,
                    base_result=base_result,
                    error="invalid_t0_frame",
                )
                continue

            try:
                t0_clock = float(row["periodGameClockTime_t0_nextGameEvent"])
            except Exception:
                _append_skipped_result(
                    results=results,
                    base_result=base_result,
                    error="invalid_periodGameClockTime_t0_nextGameEvent",
                    frame_num=frame_num,
                )
                continue

            if np.isnan(t0_clock):
                _append_skipped_result(
                    results=results,
                    base_result=base_result,
                    error="invalid_periodGameClockTime_t0_nextGameEvent",
                    frame_num=frame_num,
                )
                continue

            try:
                home_team_in_possession = _parse_bool_like(row["homeBall_t0_nextGameEvent"])
            except Exception as exc:
                _append_skipped_result(
                    results=results,
                    base_result=base_result,
                    error=f"invalid_homeBall_t0_nextGameEvent: {exc}",
                    frame_num=frame_num,
                )
                continue

            if frame_num not in game_frame_to_idx:
                _append_skipped_result(
                    results=results,
                    base_result=base_result,
                    error="frame_not_found_in_game_tracking_data",
                    frame_num=frame_num,
                )
                continue

            t_target = t0_clock + WINDOW_SECONDS
            closest_idx = (tracking_clock_df["periodGameClockTime"] - t_target).abs().idxmin()
            closest_row = tracking_clock_df.loc[closest_idx]
            t_end_clock = float(closest_row["periodGameClockTime"])
            t_end_frame = int(float(closest_row["frameNum"]))
            if abs(t_end_clock - t_target) > END_CLOCK_TOLERANCE_SECONDS:
                _append_skipped_result(
                    results=results,
                    base_result=base_result,
                    error="t_end_not_found_within_tolerance",
                    frame_num=frame_num,
                )
                continue

            window_rows = tracking_clock_df[
                (tracking_clock_df["frameNum"] >= frame_num) & (tracking_clock_df["frameNum"] <= t_end_frame)
            ].copy()
            if window_rows.empty:
                _append_skipped_result(
                    results=results,
                    base_result=base_result,
                    error="empty_5s_window",
                    frame_num=frame_num,
                )
                continue

            frame_results: list[dict[str, Any]] = []
            frame_nums = window_rows["frameNum"].astype(int).tolist()
            frame_idxs_raw = [game_frame_to_idx.get(fn) for fn in frame_nums]
            if any(idx is None for idx in frame_idxs_raw):
                _append_skipped_result(
                    results=results,
                    base_result=base_result,
                    error="frame_not_found_in_game_tracking_data",
                    frame_num=frame_num,
                )
                print(
                    f"[analysis_recovery] {game_id}: skipped recovery at t0_frame={frame_num} "
                    "(frame_not_found_in_game_tracking_data)"
                )
                continue
            frame_idxs = [int(idx) for idx in frame_idxs_raw]
            n_frames = len(frame_idxs)
            frame_rows = game.tracking_data.loc[frame_idxs]
            ball_x = frame_rows["ball_x"].to_numpy(dtype=float)
            ball_y = frame_rows["ball_y"].to_numpy(dtype=float)

            pc_att = np.full((n_frames, grid_shape[0], grid_shape[1]), np.nan, dtype=float)
            try:
                pc_raw = game.tracking_data.get_pitch_control(
                    game.pitch_dimensions,
                    grid_shape[0],
                    grid_shape[1],
                    frame_idxs[0],
                    frame_idxs[-1],
                )
                pc_arr = np.asarray(pc_raw)
                if pc_arr.ndim == 2:
                    pc_arr = pc_arr[np.newaxis, :, :]
                if pc_arr.ndim != 3:
                    raise ValueError(f"Unsupported pitch-control shape: {pc_arr.shape}")
                pc_home = np.transpose(pc_arr, (0, 2, 1))
                if pc_home.shape[1:] != grid_shape:
                    pc_home = np.stack(
                        [_resize_2d_to_shape(pc_home[i], grid_shape) for i in range(pc_home.shape[0])],
                        axis=0,
                    )
                pc_home = 1.0 - pc_home
                pc_att_batch = pc_home if home_team_in_possession else (1.0 - pc_home)
                if pc_att_batch.shape[0] == n_frames:
                    pc_att = pc_att_batch
                else:
                    raise ValueError(
                        f"Pitch-control frame count mismatch (pc={pc_att_batch.shape[0]}, expected={n_frames})"
                    )
            except Exception:
                # Fallback: try frame-by-frame pitch control; keep failing frames as NaN.
                for i, curr_idx in enumerate(frame_idxs):
                    try:
                        pc_raw_single = game.tracking_data.get_pitch_control(
                            game.pitch_dimensions,
                            grid_shape[0],
                            grid_shape[1],
                            curr_idx,
                            curr_idx,
                        )
                        pc_single = np.asarray(pc_raw_single)
                        if pc_single.ndim == 3:
                            pc_single = pc_single[0]
                        if pc_single.ndim != 2:
                            raise ValueError(f"Unsupported pitch-control shape: {pc_single.shape}")
                        pc_single = pc_single.T
                        if pc_single.shape != grid_shape:
                            pc_single = _resize_2d_to_shape(pc_single, grid_shape)
                        pc_single = 1.0 - pc_single
                        pc_att[i] = pc_single if home_team_in_possession else (1.0 - pc_single)
                    except Exception:
                        continue

            pv = np.full((n_frames, grid_shape[0], grid_shape[1]), np.nan, dtype=float)
            valid_ball_mask = np.isfinite(ball_x) & np.isfinite(ball_y)
            valid_idxs = np.where(valid_ball_mask)[0]
            if len(valid_idxs) > 0:
                n_cells = grid_shape[0] * grid_shape[1]
                feats = np.column_stack(
                    [
                        np.repeat(ball_x[valid_idxs], n_cells),
                        np.repeat(ball_y[valid_idxs], n_cells),
                        np.tile(X.ravel(), len(valid_idxs)),
                        np.tile(Y.ravel(), len(valid_idxs)),
                    ]
                )
                pv_flat = model.predict(feats)
                pv_valid = np.clip(pv_flat.reshape(len(valid_idxs), grid_shape[0], grid_shape[1]), 0.0, 1.0)
                if not home_team_in_possession:
                    pv_valid = np.flip(pv_valid, axis=(1, 2))
                pv[valid_idxs] = pv_valid

            sq = np.clip(pc_att * pv, 0.0, 1.0)
            radius_sq = float(LOCAL_RADIUS_METERS**2)
            for i in range(n_frames):
                curr_frame_result: dict[str, Any] = {
                    **base_result,
                    "frame_num": frame_nums[i],
                    "frame_idx": frame_idxs[i],
                    "pc_mean": pd.NA,
                    "pv_mean": pd.NA,
                    "sq_mean": pd.NA,
                    "sq_max": pd.NA,
                    "pc_mean_r20": pd.NA,
                    "pv_mean_r20": pd.NA,
                    "sq_mean_r20": pd.NA,
                    "frame_error": pd.NA,
                }
                try:
                    bx = float(ball_x[i])
                    by = float(ball_y[i])
                    if not np.isfinite(bx) or not np.isfinite(by):
                        raise ValueError("invalid_ball_coordinates")
                    local_mask = ((X - bx) ** 2 + (Y - by) ** 2) <= radius_sq
                    if not np.any(local_mask):
                        raise ValueError("empty_radius_mask_r20")
                    if not np.isfinite(pc_att[i]).any():
                        raise ValueError("pc_frame_failed")
                    if not np.isfinite(pv[i]).any():
                        raise ValueError("pv_frame_failed")

                    curr_frame_result["pc_mean"] = float(np.nanmean(pc_att[i]))
                    curr_frame_result["pv_mean"] = float(np.nanmean(pv[i]))
                    curr_frame_result["sq_mean"] = float(np.nanmean(sq[i]))
                    curr_frame_result["sq_max"] = float(np.nanmax(sq[i]))
                    curr_frame_result["pc_mean_r20"] = float(np.nanmean(pc_att[i][local_mask]))
                    curr_frame_result["pv_mean_r20"] = float(np.nanmean(pv[i][local_mask]))
                    curr_frame_result["sq_mean_r20"] = float(np.nanmean(sq[i][local_mask]))
                except Exception as exc:
                    curr_frame_result["frame_error"] = str(exc)
                frame_results.append(curr_frame_result)

            if not frame_results:
                _append_skipped_result(
                    results=results,
                    base_result=base_result,
                    error="empty_5s_window",
                    frame_num=frame_num,
                )
                continue

            sq_series = pd.to_numeric(pd.Series([fr["sq_mean"] for fr in frame_results]), errors="coerce").to_numpy()
            valid_sq = np.isfinite(sq_series)
            sq_t0_mean = float(sq_series[0]) if np.isfinite(sq_series[0]) else pd.NA
            sq_max_5s = float(np.nanmax(sq_series)) if valid_sq.any() else pd.NA
            recovery_space_gain = (
                float(sq_max_5s - sq_t0_mean) if (pd.notna(sq_t0_mean) and pd.notna(sq_max_5s)) else pd.NA
            )
            sq_series_r20 = pd.to_numeric(
                pd.Series([fr["sq_mean_r20"] for fr in frame_results]), errors="coerce"
            ).to_numpy()
            valid_sq_r20 = np.isfinite(sq_series_r20)
            sq_t0_mean_r20 = float(sq_series_r20[0]) if np.isfinite(sq_series_r20[0]) else pd.NA
            sq_max_5s_r20 = float(np.nanmax(sq_series_r20)) if valid_sq_r20.any() else pd.NA
            recovery_space_gain_r20 = (
                float(sq_max_5s_r20 - sq_t0_mean_r20)
                if (pd.notna(sq_t0_mean_r20) and pd.notna(sq_max_5s_r20))
                else pd.NA
            )
            window_n_frames = int(len(frame_results))
            window_start_clock = t0_clock
            window_target_clock = t_target
            window_end_clock = t_end_clock
            recovery_status = "ok" if all(pd.isna(fr["frame_error"]) for fr in frame_results) else "partial"
            recovery_error = pd.NA if recovery_status == "ok" else "one_or_more_frame_errors"

            for fr in frame_results:
                fr["sq_t0_mean"] = sq_t0_mean
                fr["sq_max_5s"] = sq_max_5s
                fr["recovery_space_gain"] = recovery_space_gain
                fr["sq_t0_mean_r20"] = sq_t0_mean_r20
                fr["sq_max_5s_r20"] = sq_max_5s_r20
                fr["recovery_space_gain_r20"] = recovery_space_gain_r20
                fr["window_start_clock"] = window_start_clock
                fr["window_target_clock"] = window_target_clock
                fr["window_end_clock"] = window_end_clock
                fr["window_n_frames"] = window_n_frames
                fr["status"] = recovery_status
                fr["error"] = recovery_error
            results.extend(frame_results)
            if recovery_idx == 1 or recovery_idx % 25 == 0 or recovery_idx == len(group):
                gain_txt = f"{float(recovery_space_gain):.4f}" if pd.notna(recovery_space_gain) else "NA"
                print(
                    f"[analysis_recovery] {game_id}: completed recovery {recovery_idx}/{len(group)} "
                    f"(window_frames={window_n_frames}, gain={gain_txt})"
                )

    print(f"[analysis_recovery] Finished run. Output rows: {len(results)}")
    return pd.DataFrame(results)
