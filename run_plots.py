import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.config import DATA_ROOT
from Utils.helpers import compute_game_minutes
from Utils.visualizations import (
    plot_team_avg_recovery_gain,
    plot_team_positive_count_vs_avg_positive_gain,
    plot_team_space_quality_curve,
    plot_team_style_matrix,
)

INPUT_CSV_WITH_BOX_ENTRY = os.path.join(
    DATA_ROOT,
    "dataset_passes_recovery",
    "recovery_space_metrics_5s_r20_late_knockout_with_box_entry.csv",
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
    positive = dedup[dedup["positive_gain"]].copy()
    if positive.empty:
        raise ValueError("Plot 3 has no positive-gain recoveries after filtering.")
    positive_agg = (
        positive.groupby("teamName_t0_nextGameEvent", dropna=False)["recovery_space_gain_r20"]
        .agg(
            n_positive_recoveries="count",
            avg_positive_recovery_space_gain_r20="mean",
        )
        .reset_index()
    )
    total_agg = (
        dedup.groupby("teamName_t0_nextGameEvent", dropna=False)["recovery_key"]
        .count()
        .reset_index(name="n_total_recoveries")
    )
    agg = positive_agg.merge(total_agg, on="teamName_t0_nextGameEvent", how="left")
    agg["positive_recovery_rate_pct"] = (
        agg["n_positive_recoveries"] / agg["n_total_recoveries"] * 100.0
    )
    agg = agg.sort_values(
        ["positive_recovery_rate_pct", "avg_positive_recovery_space_gain_r20"],
        ascending=False,
    )
    plot_team_positive_count_vs_avg_positive_gain(
        agg,
        str(output_dir / "plot3_positive_count_vs_avg_positive_gain_r20.png"),
    )


def run_plot_4_team_style_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    if "box_entry_20s" not in df.columns:
        raise ValueError("Plot 4 requires column 'box_entry_20s' in the input dataset.")

    plot_df = df.copy()
    plot_df["recovery_space_gain_r20"] = pd.to_numeric(plot_df["recovery_space_gain_r20"], errors="coerce")
    plot_df["box_entry_20s"] = plot_df["box_entry_20s"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    plot_df = plot_df[np.isfinite(plot_df["recovery_space_gain_r20"]) & plot_df["box_entry_20s"].notna()].copy()
    dedup = plot_df.drop_duplicates(subset=["recovery_key"]).copy()
    print(f"[plot4] unique recoveries: {len(dedup)}")
    if dedup.empty:
        raise ValueError("Plot 4 has no valid rows after filtering.")

    game_minutes_df = compute_game_minutes(
        base_path=DATA_ROOT,
        game_ids=dedup["source_game_file"].astype(str).unique().tolist(),
    )
    if game_minutes_df.empty:
        raise ValueError("Plot 4 could not compute game minutes from eventdata.")

    agg = (
        dedup.groupby("teamName_t0_nextGameEvent", dropna=False)
        .agg(
            n_recoveries=("recovery_key", "count"),
            avg_recovery_space_gain_r20=("recovery_space_gain_r20", "mean"),
            box_entry_rate_pct=("box_entry_20s", lambda s: float(s.mean()) * 100.0),
        )
        .reset_index()
    )

    team_game_minutes = (
        dedup[["teamName_t0_nextGameEvent", "source_game_file"]]
        .astype({"source_game_file": str})
        .drop_duplicates()
        .merge(game_minutes_df, on="source_game_file", how="left")
        .groupby("teamName_t0_nextGameEvent", dropna=False)["total_match_minutes"]
        .sum()
        .reset_index(name="team_match_minutes")
    )

    agg = agg.merge(team_game_minutes, on="teamName_t0_nextGameEvent", how="left")
    agg["recoveries_per90"] = np.where(
        pd.to_numeric(agg["team_match_minutes"], errors="coerce") > 0,
        agg["n_recoveries"] / (agg["team_match_minutes"] / 90.0),
        np.nan,
    )
    agg = agg.sort_values(["box_entry_rate_pct", "avg_recovery_space_gain_r20"], ascending=False)
    print("[plot4] team summary:")
    print(agg.to_string(index=False))

    plot_team_style_matrix(
        agg,
        str(output_dir / "plot4_team_style_matrix_r20_box_entry.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate recovery analysis plots.")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["1", "2", "3", "4"],
        default=["1", "2", "3", "4"],
        help="Select which plots to run (choices: 1 2 3 4). Default: all.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if os.path.exists(INPUT_CSV_WITH_BOX_ENTRY):
        csv_path = INPUT_CSV_WITH_BOX_ENTRY
    elif os.path.exists(INPUT_CSV):
        csv_path = INPUT_CSV
    else:
        csv_path = FALLBACK_INPUT_CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Input CSV not found. Checked:\n"
            f"- {INPUT_CSV_WITH_BOX_ENTRY}\n"
            f"- {INPUT_CSV}\n"
            f"- {FALLBACK_INPUT_CSV}"
        )
    print(f"Using input CSV: {csv_path}")
    df = _load_and_prepare(csv_path)

    selected = set(args.plots)
    if "1" in selected:
        run_plot_1_avg_recovery_space_gain(df, OUTPUT_DIR)
    if "2" in selected:
        run_plot_2_space_quality_curve(df, OUTPUT_DIR)
    if "3" in selected:
        run_plot_3_positive_exploitation_rate(df, OUTPUT_DIR)
    if "4" in selected:
        run_plot_4_team_style_matrix(df, OUTPUT_DIR)

    print(f"Saved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
