import os
import json
import bz2
import pandas as pd

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

            # Drop temp cols
            tracking_df_clean = tracking_df_clean.drop(
                columns=["homePlayers_norm", "awayPlayers_norm"]
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