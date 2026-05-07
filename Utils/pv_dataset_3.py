import os
import time
import argparse
import random
import warnings

import numpy as np
import pandas as pd

from Utils.pc_functions import get_team_influence
from Utils.loading import load_files, load_game_from_pff
from Utils.config import DATA_ROOT


def _parse_bool_like(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
        return None
    try:
        return bool(int(value))
    except Exception:
        return None


def identify_home_possession_situations(tracking_df: pd.DataFrame):
    df = tracking_df.copy()
    n_input_rows = len(df)
    invalid_tracking_entities = 0

    def _extract_home_team(x):
        if isinstance(x, dict):
            return x.get("home_team")
        return None

    df["home_team_in_possession_raw"] = df["game_event"].apply(_extract_home_team)
    df["home_team_in_possession"] = df["home_team_in_possession_raw"].apply(_parse_bool_like)

    parsed_mask = df["home_team_in_possession"].notnull()
    n_parsed_possession_rows = int(parsed_mask.sum())
    parsed_df = df[parsed_mask].copy()

    home_pos_df = parsed_df[parsed_df["home_team_in_possession"]].copy()
    n_home_possession_rows = len(home_pos_df)

    def valid_ball(x):
        if not isinstance(x, dict):
            return False
        x_val = x.get("x")
        y_val = x.get("y")
        return x_val is not None and y_val is not None and np.isfinite(x_val) and np.isfinite(y_val)

    def valid_players(arr):
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return False
        for player in arr:
            if not isinstance(player, dict):
                return False
            x = player.get("x")
            y = player.get("y")
            if x is None or y is None or not np.isfinite(x) or not np.isfinite(y):
                return False
        return True

    # Validate entities before time binning.
    valid_mask = (
        home_pos_df["ballsSmoothed"].apply(valid_ball)
        & home_pos_df["homePlayersSmoothed"].apply(valid_players)
        & home_pos_df["awayPlayersSmoothed"].apply(valid_players)
    )
    invalid_tracking_entities = int((~valid_mask).sum())
    valid_home_pos_df = home_pos_df[valid_mask].copy()

    # Keep one frame per 3-second bin after validity filtering.
    valid_home_pos_df["time_bin_3s"] = (valid_home_pos_df["periodGameClockTime"] // 3).astype(int)
    rows = (
        valid_home_pos_df.sort_values(["period", "periodGameClockTime"])
        .groupby(["period", "time_bin_3s"])
        .first()
        .reset_index()
    )

    stats = {
        "input_rows": int(n_input_rows),
        "parsed_possession_rows": int(n_parsed_possession_rows),
        "home_possession_rows": int(n_home_possession_rows),
        "kept_rows_after_validity": int(len(valid_home_pos_df)),
        "binned_rows_3s": int(len(rows)),
        "invalid_tracking_entities": int(invalid_tracking_entities),
    }
    return rows, stats


def get_grid_centers(n_x=21, n_y=15, pitch_length=105, pitch_width=68):
    x_edges = np.linspace(-pitch_length / 2, pitch_length / 2, n_x + 1)
    y_edges = np.linspace(-pitch_width / 2, pitch_width / 2, n_y + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    return x_centers, y_centers


def get_grid(n_x=21, n_y=15, pitch_length=105, pitch_width=68):
    x_centers, y_centers = get_grid_centers(n_x, n_y, pitch_length, pitch_width)
    return np.meshgrid(x_centers, y_centers, indexing="ij")


def _get_team_column_ids(game):
    all_col_ids = game.get_column_ids()
    home_col_ids = [col_id for col_id in all_col_ids if str(col_id).startswith("home_")]
    away_col_ids = [col_id for col_id in all_col_ids if str(col_id).startswith("away_")]
    return home_col_ids, away_col_ids


def creation_pv_dataset_home_possession(tracking_df: pd.DataFrame, game) -> pd.DataFrame:
    X, Y = get_grid(21, 15, game.pitch_dimensions[0], game.pitch_dimensions[1])
    _, away_col_ids = _get_team_column_ids(game)

    tracking_rows, selection_stats = identify_home_possession_situations(tracking_df)
    print("Selection summary:")
    print(
        f"input_rows={selection_stats['input_rows']} "
        f"parsed_possession_rows={selection_stats['parsed_possession_rows']} "
        f"home_possession_rows={selection_stats['home_possession_rows']} "
        f"binned_rows_3s={selection_stats['binned_rows_3s']} "
        f"kept_rows_after_validity={selection_stats['kept_rows_after_validity']}"
    )

    data_rows = []
    skipped_missing_frame = 0
    skipped_invalid_possession = selection_stats["home_possession_rows"] - selection_stats["parsed_possession_rows"]
    if skipped_invalid_possession < 0:
        skipped_invalid_possession = selection_stats["input_rows"] - selection_stats["parsed_possession_rows"]
    skipped_no_away_ids = 0
    skipped_influence_error = 0
    skipped_invalid_influence_shape = 0
    skipped_invalid_tracking_entities = selection_stats["invalid_tracking_entities"]

    if len(away_col_ids) == 0:
        print("WARNING: No away player column IDs found. Returning empty dataset.")
        skipped_no_away_ids = len(tracking_rows)
        print(
            "Frame processing summary:",
            f"input_rows={len(tracking_rows)}",
            "kept_frames=0",
            f"skipped_missing_frame={skipped_missing_frame}",
            f"skipped_invalid_possession={skipped_invalid_possession}",
            f"skipped_no_away_ids={skipped_no_away_ids}",
            f"skipped_influence_error={skipped_influence_error}",
            f"skipped_invalid_influence_shape={skipped_invalid_influence_shape}",
            f"skipped_invalid_tracking_entities={skipped_invalid_tracking_entities}",
        )
        return pd.DataFrame(
            columns=["game_id", "frameNum", "ball_x", "ball_y", "cell_x", "cell_y", "defending_value"]
        )

    for _, row in tracking_rows.iterrows():
        frame_num = row["frameNum"]
        idx = game.tracking_data.index[game.tracking_data["frame"] == frame_num]
        if len(idx) == 0:
            skipped_missing_frame += 1
            continue

        frame = game.tracking_data.loc[idx[0]]

        try:
            team_influence = get_team_influence(frame=frame, col_ids=away_col_ids, grid=[X, Y])
        except Exception:
            skipped_influence_error += 1
            continue

        if not isinstance(team_influence, np.ndarray) or team_influence.ndim != 2:
            skipped_invalid_influence_shape += 1
            continue

        defending_value = np.clip(team_influence, 0.0, 1.0)

        ball_x = row["ballsSmoothed"]["x"]
        ball_y = row["ballsSmoothed"]["y"]
        X_flat = X.ravel()
        Y_flat = Y.ravel()
        V_flat = defending_value.ravel()

        df_frame = pd.DataFrame(
            {
                "game_id": row["gameRefId"],
                "frameNum": frame_num,
                "ball_x": np.full(X_flat.shape, ball_x, dtype=float),
                "ball_y": np.full(Y_flat.shape, ball_y, dtype=float),
                "cell_x": np.round(X_flat, 3),
                "cell_y": np.round(Y_flat, 3),
                "defending_value": np.round(V_flat, 3),
            }
        )
        data_rows.append(df_frame)

    print(
        "Frame processing summary:",
        f"input_rows={len(tracking_rows)}",
        f"kept_frames={len(data_rows)}",
        f"skipped_missing_frame={skipped_missing_frame}",
        f"skipped_invalid_possession={skipped_invalid_possession}",
        f"skipped_no_away_ids={skipped_no_away_ids}",
        f"skipped_influence_error={skipped_influence_error}",
        f"skipped_invalid_influence_shape={skipped_invalid_influence_shape}",
        f"skipped_invalid_tracking_entities={skipped_invalid_tracking_entities}",
    )

    if not data_rows:
        print("WARNING: No valid frames produced. Returning empty dataset.")
        return pd.DataFrame(
            columns=["game_id", "frameNum", "ball_x", "ball_y", "cell_x", "cell_y", "defending_value"]
        )

    dataset = pd.concat(data_rows, ignore_index=True)
    print(
        f"Dataset created with {len(dataset)} rows and "
        f"{dataset['frameNum'].nunique()} unique frames."
    )
    return dataset


def build_pv_datasets_for_all_games(
    base_path: str, max_games: int | None = None, random_seed: int = 42
) -> tuple[int, int, list[str]]:
    game_ids = sorted(
        [gid.split(".")[0] for gid in os.listdir(os.path.join(base_path, "eventdata"))]
    )
    rng = random.Random(random_seed)
    rng.shuffle(game_ids)
    output_dir = os.path.join(base_path, "datasets_pitch_value")
    os.makedirs(output_dir, exist_ok=True)

    total_frames = 0
    total_rows = 0
    failed_games = []
    warning_games = []
    processed_games = 0

    for game_id in game_ids:
        if max_games is not None and processed_games >= max_games:
            break
        print(f"Processing game {game_id}...")
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                game = load_game_from_pff(base_path, game_id)
            if len(caught_warnings) > 0:
                warning_games.append(game_id)
                print(f"Skipping game {game_id} due to load warnings ({len(caught_warnings)} warnings).")
                print("-" * 50)
                continue

            _, tracking_df = load_files(base_path, game_id)
            dataset = creation_pv_dataset_home_possession(tracking_df, game)

            game_frames = int(dataset["frameNum"].nunique()) if not dataset.empty else 0
            game_rows = int(len(dataset))
            total_frames += game_frames
            total_rows += game_rows
            processed_games += 1

            out_path = os.path.join(output_dir, f"dataset_{game_id}.csv")
            dataset.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")
            print(f"Game {game_id} totals: frames_saved={game_frames}, rows_saved={game_rows}")
        except Exception as e:
            failed_games.append(game_id)
            print(f"Failed game {game_id}: {e}")

        print("-" * 50)
        time.sleep(5)

    print(f"Total frames processed: {total_frames}")
    print(f"Total rows in all datasets: {total_rows}")
    print(f"Processed games count: {processed_games}")
    print(f"Skipped due to load warnings: {len(warning_games)}")
    print(f"Warning-skipped games: {warning_games}")
    print(f"Failed games count: {len(failed_games)}")
    print(f"Failed games: {failed_games}")
    return total_frames, total_rows, failed_games


def _build_pv_dataset_for_single_game(base_path: str, game_id: str) -> tuple[int, int]:
    output_dir = os.path.join(base_path, "datasets_pitch_value")
    os.makedirs(output_dir, exist_ok=True)

    _, tracking_df = load_files(base_path, game_id)
    game = load_game_from_pff(base_path, game_id)
    dataset = creation_pv_dataset_home_possession(tracking_df, game)

    out_path = os.path.join(output_dir, f"dataset_{game_id}.csv")
    dataset.to_csv(out_path, index=False)

    total_frames = int(dataset["frameNum"].nunique()) if not dataset.empty else 0
    total_rows = int(len(dataset))
    print(f"Saved: {out_path}")
    print(f"Game {game_id} totals: frames_saved={total_frames}, rows_saved={total_rows}")
    return total_frames, total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Build home-possession-only pitch-value datasets from tracking data."
    )
    parser.add_argument(
        "--base-path",
        default=DATA_ROOT,
        help="Root path containing eventdata/trackingdata folders (default: DATA_ROOT).",
    )
    parser.add_argument(
        "--game-id",
        default=None,
        help="Optional single game id. If omitted, process all games in eventdata.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional maximum number of warning-free games to process (random order).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used to shuffle game order.",
    )
    args = parser.parse_args()

    if args.game_id:
        _build_pv_dataset_for_single_game(args.base_path, args.game_id)
    else:
        build_pv_datasets_for_all_games(
            args.base_path,
            max_games=args.max_games,
            random_seed=args.random_seed,
        )


if __name__ == "__main__":
    main()
