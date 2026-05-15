import os

import pandas as pd

from Utils.analysis_recovery import label_box_entry_next_20s
from Utils.config import DATA_ROOT


in_path = os.path.join(
    DATA_ROOT,
    "dataset_passes_recovery",
    "recovery_space_metrics_5s_r20_late_knockout.csv",
)
df = pd.read_csv(in_path)
print("Loaded:", in_path, "rows:", len(df))

out_df = label_box_entry_next_20s(
    recovery_df=df,
    base_path=DATA_ROOT,
    window_seconds=20.0,
)

out_path = os.path.join(
    DATA_ROOT,
    "dataset_passes_recovery",
    "recovery_space_metrics_5s_r20_late_knockout_with_box_entry.csv",
)
out_df.to_csv(out_path, index=False)
print("Saved:", out_path, "rows:", len(out_df))
print("box_entry_20s counts:")
print(out_df["box_entry_20s"].value_counts(dropna=False))
print("box_entry_20s_status counts:")
print(out_df["box_entry_20s_status"].value_counts(dropna=False))
