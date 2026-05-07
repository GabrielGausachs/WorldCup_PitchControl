import os
import json
import bz2
import pandas as pd
import numpy as np

def normalize_players(players):
    if not isinstance(players, list):
        return None
    
    return tuple(sorted([
        (
            p.get("jerseyNum"),
            round(p.get("x", 0), 4),
            round(p.get("y", 0), 4)
        )
        for p in players
    ]))

def clean_tracking_data(base_path):
    tracking_path = os.path.join(base_path, "trackingdata")
    tracking_files = sorted([f for f in os.listdir(tracking_path) if f.endswith('.jsonl.bz2')])

    clean_path = os.path.join(base_path, "trackingdata_clean")

    # Make sure the clean folder exists
    os.makedirs(clean_path, exist_ok=True)

    for file in tracking_files:
        input_file = os.path.join(tracking_path, file)
        output_file = os.path.join(clean_path, file.replace(".jsonl.bz2", "_clean.jsonl.bz2"))
        
        print("-" * 50)

        print(f"\nProcessing {file}...")

        # --- LOAD FILE INTO DATAFRAME ---
        with bz2.open(input_file, "rt") as f:
            rows = [json.loads(line) for line in f]

        tracking_df = pd.DataFrame(rows)

        # --- FIND DUPLICATES ---
        duplicate_frames = tracking_df[
            tracking_df.duplicated(subset=['frameNum'], keep=False)
        ]

        print(f"Duplicate rows: {duplicate_frames.shape[0]}")
        print(f"Duplicate ratio: {duplicate_frames.shape[0] / tracking_df.shape[0]:.4f}")

        if duplicate_frames.empty:
            print("✅ No duplicates → saving directly")
            tracking_df_clean = tracking_df.copy()

        else:
            # --- NORMALIZE ---
            tracking_df["homePlayers_norm"] = tracking_df["homePlayers"].apply(normalize_players)
            tracking_df["awayPlayers_norm"] = tracking_df["awayPlayers"].apply(normalize_players)

            # --- CHECK DUPLICATES ---
            problematic_frames = []
            identical_frames = []

            for frame in duplicate_frames['frameNum'].unique():
                frame_rows = tracking_df[tracking_df['frameNum'] == frame]

                home_unique = frame_rows["homePlayers_norm"].nunique()
                away_unique = frame_rows["awayPlayers_norm"].nunique()

                if home_unique > 1 or away_unique > 1:
                    problematic_frames.append(frame)
                else:
                    identical_frames.append(frame)

            # --- CLEAN ---
            if len(problematic_frames) == 0:
                print("✅ All duplicates identical → safe drop")
                tracking_df_clean = tracking_df.drop_duplicates(subset=["frameNum"]).copy()

            else:
                print(f"⚠️ Problematic frames: {len(problematic_frames)}")

                mask_safe = tracking_df["frameNum"].isin(identical_frames)
                mask_problem = tracking_df["frameNum"].isin(problematic_frames)

                clean_safe = tracking_df[mask_safe].drop_duplicates(subset=["frameNum"])
                keep_problem = tracking_df[mask_problem]

                tracking_df_clean = pd.concat([clean_safe, keep_problem], ignore_index=True)

            # --- BALL COORDINATES CLEANING ---
            tracking_df_clean["ball_x"] = tracking_df_clean["ballsSmoothed"].apply(
                lambda b: b["x"] if isinstance(b, dict) else None
            )
            tracking_df_clean["ball_y"] = tracking_df_clean["ballsSmoothed"].apply(
                lambda b: b["y"] if isinstance(b, dict) else None
            )

            # Clip to pitch
            tracking_df_clean["ball_x"] = tracking_df_clean["ball_x"].clip(-54.5, 54.5)
            tracking_df_clean["ball_y"] = tracking_df_clean["ball_y"].clip(-36, 36)

            # Drop temp cols
            tracking_df_clean = tracking_df_clean.drop(
                columns=["homePlayers_norm", "awayPlayers_norm", "ball_x", "ball_y"]
            )

        # --- SAVE BACK ---
        with bz2.open(output_file, "wt") as f:
            for row in tracking_df_clean.to_dict(orient="records"):
                f.write(json.dumps(row) + "\n")

        print(f"✅ Saved: {output_file}")

def bz2_to_parquet(base_path):
    for file in os.listdir(os.path.join(base_path, "trackingdata_clean")):
        print("-" * 50)
        print(f"\nProcessing {file}...")
        # Load tracking data
        tracking_file = os.path.join(base_path, "trackingdata_clean", file)
        df = pd.read_json(tracking_file, lines=True, compression='bz2')

        # Save it as parquet
        os.makedirs(os.path.join(base_path, "trackingdata_parquet"), exist_ok=True)
        parquet_file = os.path.join(base_path, "trackingdata_parquet", file.replace("_clean.jsonl.bz2", ".parquet"))
        df.to_parquet(parquet_file, index=False)
        print(f"✅ Converted to parquet: {parquet_file}")



def goal_distance_factor(cell_x, cell_y, pitch_length=105.0, pitch_width=68.0):
    """
    Deterministic [0, 1] factor based on Euclidean distance to opponent goal center.
    Assumes attacking direction normalized left-to-right.
    """
    cell_x = np.asarray(cell_x, dtype=float)
    cell_y = np.asarray(cell_y, dtype=float)

    goal_x = pitch_length / 2.0
    goal_y = 0.0
    dist = np.sqrt((goal_x - cell_x) ** 2 + (goal_y - cell_y) ** 2)

    # Max relevant distance within the pitch from opponent goal center to far side corners.
    max_dist = np.sqrt((pitch_length) ** 2 + (pitch_width / 2.0) ** 2)
    return np.clip(1.0 - (dist / max_dist), 0.0, 1.0)


def normalize_pitch_value_by_goal_distance(
    pv_raw, cell_x, cell_y, pitch_length=105.0, pitch_width=68.0
):
    """
    Apply distance-to-goal normalization to raw pitch value predictions.
    """
    pv_raw = np.asarray(pv_raw, dtype=float)
    factor = goal_distance_factor(
        cell_x=cell_x,
        cell_y=cell_y,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )
    return np.clip(pv_raw * factor, 0.0, 1.0)


def get_grid(n_x=105, n_y=68, pitch_length=105.0, pitch_width=68.0):
    x_coords = np.linspace(-pitch_length / 2.0, pitch_length / 2.0, n_x)
    y_coords = np.linspace(-pitch_width / 2.0, pitch_width / 2.0, n_y)
    return np.meshgrid(x_coords, y_coords, indexing="ij")


def predict_surface(model, ball_x, ball_y, X, Y):
    feats = np.column_stack(
        [
            np.full(X.size, ball_x, dtype=float),
            np.full(Y.size, ball_y, dtype=float),
            X.ravel(),
            Y.ravel(),
        ]
    )
    pred = model.predict(feats)
    return np.clip(pred.reshape(X.shape), 0.0, 1.0)
