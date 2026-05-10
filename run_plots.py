import os
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.config import DATA_ROOT
from Utils.visualizations import (
    plot_team_avg_recovery_gain,
    plot_team_positive_rate,
    plot_team_space_quality_curve,
)

INPUT_CSV = os.path.join(
    DATA_ROOT,
    "dataset_passes_recovery",
    "recovery_space_metrics_5s_r20_late_knockout.csv",
)
FALLBACK_INPUT_CSV = os.path.join(
    DATA_ROOT,
    "dataset_passes_recovery",
    "recovery_space_metrics_5s_r20.csv",
)
OUTPUT_DIR = Path(__file__).resolve().parent / "Outputs"


def _load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [
        "source_game_file",
        "t0_startFrame_nextGameEvent",
        "periodGameClockTime_t0_nextGameEvent",
        "teamName_t0_nextGameEvent",
        "status",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["status"] != "skipped"].copy()
    df["recovery_key"] = (
        df["source_game_file"].astype(str)
        + "|"
        + df["t0_startFrame_nextGameEvent"].astype(str)
        + "|"
        + df["periodGameClockTime_t0_nextGameEvent"].astype(str)
        + "|"
        + df["teamName_t0_nextGameEvent"].astype(str)
    )
    return df


def run_plot_1_avg_recovery_space_gain(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["recovery_space_gain_r20"] = pd.to_numeric(plot_df["recovery_space_gain_r20"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["recovery_space_gain_r20"])].copy()
    dedup = plot_df.drop_duplicates(subset=["recovery_key"]).copy()
    print(f"[plot1] unique recoveries: {len(dedup)}")
    if dedup.empty:
        raise ValueError("Plot 1 has no valid rows after filtering.")

    agg = (
        dedup.groupby("teamName_t0_nextGameEvent", dropna=False)["recovery_space_gain_r20"]
        .mean()
        .reset_index(name="avg_recovery_space_gain_r20")
        .sort_values("avg_recovery_space_gain_r20", ascending=False)
    )
    plot_team_avg_recovery_gain(
        agg,
        str(output_dir / "plot1_avg_recovery_space_gain_r20.png"),
    )


def run_plot_2_space_quality_curve(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["sq_mean_r20"] = pd.to_numeric(plot_df["sq_mean_r20"], errors="coerce")
    plot_df["frame_idx"] = pd.to_numeric(plot_df["frame_idx"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["sq_mean_r20"]) & np.isfinite(plot_df["frame_idx"])].copy()
    if plot_df.empty:
        raise ValueError("Plot 2 has no valid frame rows after filtering.")

    plot_df = plot_df.sort_values(["recovery_key", "frame_idx"]).copy()
    plot_df["rank_index"] = plot_df.groupby("recovery_key").cumcount()
    plot_df["n_frames"] = plot_df.groupby("recovery_key")["frame_idx"].transform("count")
    denom = (plot_df["n_frames"] - 1).clip(lower=1)
    plot_df["seconds_since_t0"] = np.where(
        plot_df["n_frames"] <= 1,
        0.0,
        (plot_df["rank_index"] / denom) * 5.0,
    )
    plot_df["second_bin"] = plot_df["seconds_since_t0"].round().clip(0, 5).astype(int)

    per_recovery_second = (
        plot_df.groupby(["recovery_key", "teamName_t0_nextGameEvent", "second_bin"], dropna=False)["sq_mean_r20"]
        .mean()
        .reset_index(name="sq_mean_r20_recovery_second")
    )
    agg = (
        per_recovery_second.groupby(["teamName_t0_nextGameEvent", "second_bin"], dropna=False)[
            "sq_mean_r20_recovery_second"
        ]
        .mean()
        .reset_index(name="avg_sq_mean_r20")
        .sort_values(["teamName_t0_nextGameEvent", "second_bin"])
    )

    sec_values = set(agg["second_bin"].unique().tolist())
    if not sec_values.issubset({0, 1, 2, 3, 4, 5}):
        raise ValueError(f"Plot 2 invalid second bins found: {sorted(sec_values)}")
    if agg.empty:
        raise ValueError("Plot 2 aggregate is empty.")

    plot_team_space_quality_curve(
        agg,
        str(output_dir / "plot2_space_quality_curve_r20.png"),
    )


def run_plot_3_positive_exploitation_rate(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["recovery_space_gain_r20"] = pd.to_numeric(plot_df["recovery_space_gain_r20"], errors="coerce")
    plot_df = plot_df[np.isfinite(plot_df["recovery_space_gain_r20"])].copy()
    dedup = plot_df.drop_duplicates(subset=["recovery_key"]).copy()
    print(f"[plot3] unique recoveries: {len(dedup)}")
    if dedup.empty:
        raise ValueError("Plot 3 has no valid rows after filtering.")

    dedup["positive_gain"] = dedup["recovery_space_gain_r20"] > 0.0
    agg = (
        dedup.groupby("teamName_t0_nextGameEvent", dropna=False)["positive_gain"]
        .mean()
        .mul(100.0)
        .reset_index(name="positive_exploitation_rate_pct")
        .sort_values("positive_exploitation_rate_pct", ascending=False)
    )
    plot_team_positive_rate(
        agg,
        str(output_dir / "plot3_positive_exploitation_rate_r20.png"),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = INPUT_CSV if os.path.exists(INPUT_CSV) else FALLBACK_INPUT_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Input CSV not found. Checked:\n- {INPUT_CSV}\n- {FALLBACK_INPUT_CSV}"
        )
    print(f"Using input CSV: {csv_path}")
    df = _load_and_prepare(csv_path)
    run_plot_1_avg_recovery_space_gain(df, OUTPUT_DIR)
    run_plot_2_space_quality_curve(df, OUTPUT_DIR)
    run_plot_3_positive_exploitation_rate(df, OUTPUT_DIR)
    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
