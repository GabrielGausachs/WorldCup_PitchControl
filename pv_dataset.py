import os
import time
import argparse
import random
import warnings

import numpy as np
import pandas as pd

from pitch_control_functions.pc_functions import get_team_influence
from Utils.loading import load_files, load_game_from_pff
from Utils.config import DATA_ROOT


def identify_defensive_situations(tracking_df):
    tracking_df = tracking_df.copy()

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

    tracking_rows = tracking_df[
        (tracking_df["ballsSmoothed"].apply(valid_ball))
        & (tracking_df["homePlayersSmoothed"].apply(valid_players))
        & (tracking_df["awayPlayersSmoothed"].apply(valid_players))
    ].copy()

    # Keep possession context for downstream logic
    tracking_rows["team_in_possession"] = tracking_rows["game_event"].apply(
        lambda x: x.get("team_name") if isinstance(x, dict) else None
    )
    tracking_rows["home_team_in_possession"] = tracking_rows["game_event"].apply(
        lambda x: x.get("home_team") if isinstance(x, dict) else None
    )

    # Event-based sampling from possession events
    def _event_type(x):
        return x.get("possession_event_type") if isinstance(x, dict) else None

    tracking_passes = tracking_rows[tracking_rows["possession_event"].apply(lambda x: _event_type(x) == "PA")].copy()
    tracking_shots = tracking_rows[tracking_rows["possession_event"].apply(lambda x: _event_type(x) == "SH")].copy()
    tracking_crosses = tracking_rows[tracking_rows["possession_event"].apply(lambda x: _event_type(x) == "CR")].copy()
    tracking_ball_carries = tracking_rows[tracking_rows["possession_event"].apply(lambda x: _event_type(x) == "BC")].copy()

    print(
        "Selected possession events:",
        f"PA={len(tracking_passes)}",
        f"SH={len(tracking_shots)}",
        f"CR={len(tracking_crosses)}",
        f"BC={len(tracking_ball_carries)}",
    )

    tracking_event_rows = pd.concat(
        [tracking_passes, tracking_shots, tracking_crosses, tracking_ball_carries],
        ignore_index=True,
    )

    tracking_event_rows["possession_start_frame"] = tracking_event_rows["possession_event"].apply(
        lambda x: x.get("start_frame") if isinstance(x, dict) else None
    )
    before_dropna = len(tracking_event_rows)
    tracking_event_rows = tracking_event_rows[tracking_event_rows["possession_start_frame"].notnull()].copy()
    tracking_event_rows["possession_start_frame"] = tracking_event_rows["possession_start_frame"].astype(int)
    after_dropna = len(tracking_event_rows)
    before_dedup = len(tracking_event_rows)
    tracking_event_rows = tracking_event_rows.drop_duplicates(subset=["possession_start_frame"]).copy()
    after_dedup = len(tracking_event_rows)

    print(
        "Event frame filtering:",
        f"before_start_frame_filter={before_dropna}",
        f"after_start_frame_filter={after_dropna}",
        f"before_dedup={before_dedup}",
        f"after_dedup_unique_frames={after_dedup}",
    )

    return tracking_event_rows


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


def creation_pv_dataset(tracking_df: pd.DataFrame, game) -> pd.DataFrame:
    X, Y = get_grid(21, 15, game.pitch_dimensions[0], game.pitch_dimensions[1])
    home_col_ids, away_col_ids = _get_team_column_ids(game)

    tracking_rows = identify_defensive_situations(tracking_df)
    print(f"identify_defensive_situations returned {len(tracking_rows)} tracking rows")
    data_rows = []
    skipped_missing_frame = 0
    skipped_non_bool_possession = 0
    skipped_no_defending_ids = 0
    skipped_influence_error = 0
    skipped_invalid_influence_shape = 0

    for _, row in tracking_rows.iterrows():
        frame_num = row["possession_start_frame"]
        idx = game.tracking_data.index[game.tracking_data["frame"] == frame_num]
        if len(idx) == 0:
            skipped_missing_frame += 1
            continue

        frame = game.tracking_data.loc[idx[0]]

        home_team_in_possession_raw = row["home_team_in_possession"]
        if pd.isna(home_team_in_possession_raw):
            skipped_non_bool_possession += 1
            continue
        try:
            if isinstance(home_team_in_possession_raw, str):
                value = home_team_in_possession_raw.strip().lower()
                if value in {"1", "true", "t", "yes", "y"}:
                    home_team_in_possession = True
                elif value in {"0", "false", "f", "no", "n"}:
                    home_team_in_possession = False
                else:
                    skipped_non_bool_possession += 1
                    continue
            else:
                home_team_in_possession = bool(int(home_team_in_possession_raw))
        except Exception:
            skipped_non_bool_possession += 1
            continue

        defenders_are_home = not home_team_in_possession
        defending_col_ids = home_col_ids if defenders_are_home else away_col_ids
        if len(defending_col_ids) == 0:
            skipped_no_defending_ids += 1
            continue

        try:
            team_influence = get_team_influence(frame=frame, col_ids=defending_col_ids, grid=[X, Y])
        except Exception:
            skipped_influence_error += 1
            continue

        if not isinstance(team_influence, np.ndarray) or team_influence.ndim != 2:
            skipped_invalid_influence_shape += 1
            continue
        
        defending_value = np.clip(team_influence, 0.0, 1.0)

        ball_x = row["ballsSmoothed"]["x"]
        ball_y = row["ballsSmoothed"]["y"]
        X_frame = X
        Y_frame = Y
        V_frame = defending_value

        # Home team always attacks left-to-right in this game object.
        # So if defenders are home team, flip to normalize possessions to left-to-right attacking.
        if defenders_are_home:
            ball_x = -ball_x
            ball_y = -ball_y
            X_frame = -X_frame
            Y_frame = -Y_frame
            V_frame = defending_value[::-1, ::-1]

        X_flat = X_frame.ravel()
        Y_flat = Y_frame.ravel()
        V_flat = V_frame.ravel()

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
        f"skipped_non_bool_possession={skipped_non_bool_possession}",
        f"skipped_no_defending_ids={skipped_no_defending_ids}",
        f"skipped_influence_error={skipped_influence_error}",
        f"skipped_invalid_influence_shape={skipped_invalid_influence_shape}",
    )

    if not data_rows:
        print("WARNING: No valid frames produced. Returning empty dataset.")
        return pd.DataFrame(
            columns=["game_id", "frameNum", "ball_x", "ball_y", "cell_x", "cell_y", "defending_value"]
        )

    dataset = pd.concat(data_rows, ignore_index=True)
    print(f"Dataset created with {len(dataset)} rows and {dataset['frameNum'].nunique()} unique frames.")
    return dataset


def build_pv_datasets_for_all_games(base_path: str) -> tuple[int, int, list[str]]:
    game_ids_all = sorted(
        [gid.split(".")[0] for gid in os.listdir(os.path.join(base_path, "eventdata"))]
    )
    rng = random.Random(42)
    rng.shuffle(game_ids_all)
    output_dir = os.path.join(base_path, "datasets_pitch_value")
    os.makedirs(output_dir, exist_ok=True)
    existing_files = [
        f for f in os.listdir(output_dir) if f.startswith("dataset_") and f.endswith(".csv")
    ]
    existing_game_ids = {
        f.replace("dataset_", "").replace(".csv", "") for f in existing_files
    }

    total_frames = 0
    total_rows = 0
    failed_games = []
    warning_games = []
    processed_games = []
    target_games = 10
    remaining_games_to_process = max(0, target_games - len(existing_game_ids))

    print(
        f"Existing datasets found: {len(existing_game_ids)}",
        f"Remaining games to process: {remaining_games_to_process}",
    )
    if remaining_games_to_process == 0:
        print("Target already reached. Nothing to process.")
        return total_frames, total_rows, failed_games

    for game_id in game_ids_all:
        if len(processed_games) >= remaining_games_to_process:
            break
        if game_id in existing_game_ids:
            continue
        print(f"Processing game {game_id}...")
        try:
            _, tracking_df = load_files(base_path, game_id)
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                game = load_game_from_pff(base_path, game_id)
            if len(caught_warnings) > 0:
                warning_games.append(game_id)
                print(f"Skipping game {game_id} due to load warnings ({len(caught_warnings)} warnings).")
                print("-" * 50)
                continue
            dataset = creation_pv_dataset(tracking_df, game)

            total_frames += len(dataset["frameNum"].unique())
            total_rows += len(dataset)
            processed_games.append(game_id)

            dataset.to_csv(
                os.path.join(output_dir, f"dataset_{game_id}.csv"), index=False
            )
            if dataset.empty:
                print(f"WARNING: Dataset for game {game_id} is empty (header only).")
            print(f"Dataset for game {game_id} saved with {len(dataset)} rows.")
        except Exception as e:
            failed_games.append(game_id)
            print(f"Failed game {game_id}: {e}")

        print("-" * 50)
        time.sleep(5)

    if len(processed_games) < remaining_games_to_process:
        print(
            f"WARNING: Only {len(processed_games)} clean games processed "
            f"(needed {remaining_games_to_process} to reach target {target_games})."
        )

    print(f"Previously existing datasets ({len(existing_game_ids)}): {sorted(existing_game_ids)}")
    print(f"Newly processed clean games ({len(processed_games)}): {processed_games}")
    print(f"Skipped due to warnings ({len(warning_games)}): {warning_games}")
    print(f"Total frames processed: {total_frames}")
    print(f"Total rows in all datasets: {total_rows}")
    print(f"Failed games count: {len(failed_games)}")
    print(f"Failed games: {failed_games}")

    return total_frames, total_rows, failed_games


def _build_pv_dataset_for_single_game(base_path: str, game_id: str) -> tuple[int, int]:
    output_dir = os.path.join(base_path, "datasets_pitch_value")
    os.makedirs(output_dir, exist_ok=True)

    _, tracking_df = load_files(base_path, game_id)
    game = load_game_from_pff(base_path, game_id)
    dataset = creation_pv_dataset(tracking_df, game)
    dataset.to_csv(os.path.join(output_dir, f"dataset_{game_id}.csv"), index=False)

    total_frames = len(dataset["frameNum"].unique())
    total_rows = len(dataset)
    print(f"Dataset for game {game_id} saved.")
    print(f"Total frames processed: {total_frames}")
    print(f"Total rows in dataset: {total_rows}")
    return total_frames, total_rows


def main():
    parser = argparse.ArgumentParser(description="Build pitch-value datasets from tracking data.")
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
    args = parser.parse_args()

    if args.game_id:
        _build_pv_dataset_for_single_game(args.base_path, args.game_id)
    else:
        build_pv_datasets_for_all_games(args.base_path)


if __name__ == "__main__":
    main()



    # They split data into 10 folds Train on 9, validate on 1 Rotate Pick best hyperparameters
    # MSE LOSS
    # ONCE the train is done, they apply normalization for the distance to the goal
    # Then they do heatmaps to evaluate model
    
