import os
import pandas as pd
from Utils.config import DATA_ROOT
from Utils.analysis_recovery import avg_recovery_space_gain

# input recovery dataset (built earlier)
in_path = os.path.join(DATA_ROOT, "dataset_passes_recovery", "dataset_passes_recovery_all_matches.csv")
recovery_df = pd.read_csv(in_path)


# run analysis
out_df = avg_recovery_space_gain(recovery_df, base_path=DATA_ROOT)

# save results
out_path = os.path.join(DATA_ROOT, "dataset_passes_recovery", "recovery_space_metrics_5s_r20.csv")
out_df.to_csv(out_path, index=False)
print("Saved:", out_path, "rows:", len(out_df))
