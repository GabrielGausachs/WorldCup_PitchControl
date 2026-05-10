import os
import json
from pathlib import Path

import pandas as pd

from Utils.config import DATA_ROOT
from Utils.analysis_recovery import avg_recovery_space_gain

# input recovery dataset (built earlier)
in_path = os.path.join(DATA_ROOT, "dataset_passes_recovery", "dataset_passes_recovery_all_matches.csv")
recovery_df = pd.read_csv(in_path)

# Keep only quarterfinals, semifinals, third-place, and final based on metadata week.
# World Cup 2022 mapping in PFF metadata:
# week=5 (quarterfinals), week=6 (semifinals), week=7 (third place), week=8 (final)
metadata_dir = Path(DATA_ROOT) / "metadata"
keep_weeks = {5, 6, 7, 8}
keep_game_ids: set[str] = set()

for meta_file in metadata_dir.glob("*.json"):
    try:
        payload = json.loads(meta_file.read_text(encoding="utf-8"))
        meta = payload[0] if isinstance(payload, list) and payload else payload
        week = int(meta.get("week")) if isinstance(meta, dict) and meta.get("week") is not None else None
        game_id = str(meta.get("id")) if isinstance(meta, dict) and meta.get("id") is not None else None
        if week in keep_weeks and game_id:
            keep_game_ids.add(game_id)
    except Exception:
        continue

recovery_df["source_game_file"] = recovery_df["source_game_file"].astype(str)
filtered_df = recovery_df[recovery_df["source_game_file"].isin(keep_game_ids)].copy()

print(
    "Filtered recoveries:",
    len(filtered_df),
    "/",
    len(recovery_df),
    "rows across",
    filtered_df["source_game_file"].nunique(),
    "game(s)",
)

# run analysis
out_df = avg_recovery_space_gain(filtered_df, base_path=DATA_ROOT)

# save results
out_path = os.path.join(DATA_ROOT, "dataset_passes_recovery", "recovery_space_metrics_5s_r20_late_knockout.csv")
out_df.to_csv(out_path, index=False)
print("Saved:", out_path, "rows:", len(out_df))
