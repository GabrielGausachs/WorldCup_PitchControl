import argparse
import json
import random
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch

from Utils.config import DATA_ROOT

DEFAULT_TRACKING_DIR = Path(DATA_ROOT) / "trackingdata_parquet"
DEFAULT_EVENT_DIR = Path(DATA_ROOT) / "eventdata"
REQUIRED_BASE_COLUMNS = [
    "gameRefId",
    "frameNum",
    "period",
    "periodGameClockTime",
    "game_event_id",
    "possession_event_id",
    "game_event",
    "possession_event",
]
OUTPUT_COLUMNS = [
    "gameRefId",
    "frameNum",
    "period",
    "periodGameClockTime",
    "game_event_id",
    "possession_event_id",
    "game_event_type",
    "team_name",
    "home_team",
    "home_ball",
    "game_event_start_frame",
    "game_event_end_frame",
    "game_event_start_time",
    "game_event_end_time",
    "possession_event_type",
    "possession_start_frame",
    "possession_end_frame",
    "possession_start_time",
]


def _extract_from_dict(value, key):
    if isinstance(value, dict):
        return value.get(key)
    return None


def _resolve_parquet_path(tracking_dir: Path, parquet_arg: str | None) -> Path:
    if parquet_arg:
        parquet_path = Path(parquet_arg)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        return parquet_path

    if not tracking_dir.exists():
        raise FileNotFoundError(f"Tracking directory not found: {tracking_dir}")

    parquet_files = sorted(tracking_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {tracking_dir}")

    return parquet_files[0]


def _validate_required_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")


def _add_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["game_event_type"] = df["game_event"].apply(lambda x: _extract_from_dict(x, "game_event_type"))
    df["team_name"] = df["game_event"].apply(lambda x: _extract_from_dict(x, "team_name"))
    df["home_team"] = df["game_event"].apply(lambda x: _extract_from_dict(x, "home_team"))
    df["home_ball"] = df["game_event"].apply(lambda x: _extract_from_dict(x, "home_ball"))
    df["game_event_start_frame"] = df["game_event"].apply(lambda x: _extract_from_dict(x, "start_frame"))
    df["game_event_end_frame"] = df["game_event"].apply(lambda x: _extract_from_dict(x, "end_frame"))
    df["game_event_start_time"] = df["game_event"].apply(lambda x: _extract_from_dict(x, "start_time"))
    df["game_event_end_time"] = df["game_event"].apply(lambda x: _extract_from_dict(x, "end_time"))

    df["possession_event_type"] = df["possession_event"].apply(
        lambda x: _extract_from_dict(x, "possession_event_type")
    )
    df["possession_start_frame"] = df["possession_event"].apply(
        lambda x: _extract_from_dict(x, "start_frame")
    )
    df["possession_end_frame"] = df["possession_event"].apply(
        lambda x: _extract_from_dict(x, "end_frame")
    )
    df["possession_start_time"] = df["possession_event"].apply(
        lambda x: _extract_from_dict(x, "start_time")
    )

    return df


def _extract_d_pass_possession_id(events: list[dict], seed: int | None) -> tuple[int, dict]:
    candidates = []
    for event in events:
        if not isinstance(event, dict):
            continue
        possession_event = event.get("possessionEvents")
        if not isinstance(possession_event, dict):
            continue
        if possession_event.get("passOutcomeType") != "D":
            continue
        possession_event_id = event.get("possessionEventId")
        if possession_event_id is None:
            continue
        candidates.append((int(possession_event_id), event))

    if not candidates:
        raise ValueError("No pass with passOutcomeType='D' found in event json.")

    rng = random.Random(seed)
    return rng.choice(candidates)


def _pick_window_around_possession_event(df: pd.DataFrame, possession_event_id: int, max_frames: int) -> pd.DataFrame:
    matches = df.index[df["possession_event_id"] == possession_event_id].tolist()
    if not matches:
        raise ValueError(f"possession_event_id={possession_event_id} not found in tracking parquet.")

    chosen_idx = matches[0]
    half = max_frames // 2
    start = max(0, chosen_idx - half)
    end = min(len(df), start + max_frames)
    start = max(0, end - max_frames)

    subset = df.iloc[start:end].copy()
    if subset.empty:
        raise ValueError("Subset is empty after window selection.")
    return subset


def _extract_xy_from_players(players):
    if isinstance(players, np.ndarray):
        players = players.tolist()
    if not isinstance(players, list):
        return [], []
    x_vals = []
    y_vals = []
    for player in players:
        if not isinstance(player, dict):
            continue
        x = player.get("x")
        y = player.get("y")
        if x is None or y is None:
            continue
        x_vals.append(float(x))
        y_vals.append(float(y))
    return x_vals, y_vals


def _extract_xy_from_ball(ball):
    if isinstance(ball, dict):
        x = ball.get("x")
        y = ball.get("y")
        if x is not None and y is not None:
            return float(x), float(y)
    return None, None


def create_animation(subset: pd.DataFrame, output_video: Path, fps: int = 10):
    required_anim_cols = ["homePlayersSmoothed", "awayPlayersSmoothed", "ballsSmoothed", "frameNum"]
    missing = [col for col in required_anim_cols if col not in subset.columns]
    if missing:
        raise ValueError(f"Missing required columns for animation: {missing}")

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=105.0,
        pitch_width=68.0,
        line_color="black",
        pitch_color="#2e8b57",
    )
    fig, ax = pitch.draw(figsize=(10, 6))
    home_scatter = ax.scatter([], [], c="royalblue", s=60, edgecolors="white", linewidths=0.7, label="Home")
    away_scatter = ax.scatter([], [], c="orangered", s=60, edgecolors="white", linewidths=0.7, label="Away")
    ball_scatter = ax.scatter([], [], c="white", s=45, edgecolors="black", linewidths=1.0, label="Ball", zorder=5)
    frame_text = ax.text(2, 66, "", fontsize=10, color="black")
    ax.legend(loc="upper right")

    def _to_pitch_coords(xs, ys):
        # Convert centered coordinates to mplsoccer custom pitch coordinates.
        return [x + 52.5 for x in xs], [y + 34.0 for y in ys]

    def update(i):
        row = subset.iloc[i]
        hx, hy = _extract_xy_from_players(row["homePlayersSmoothed"])
        axh_x, axh_y = _to_pitch_coords(hx, hy)
        if axh_x:
            home_scatter.set_offsets(np.column_stack((axh_x, axh_y)))
        else:
            home_scatter.set_offsets(np.empty((0, 2)))

        axx, ayy = _extract_xy_from_players(row["awayPlayersSmoothed"])
        axa_x, axa_y = _to_pitch_coords(axx, ayy)
        if axa_x:
            away_scatter.set_offsets(np.column_stack((axa_x, axa_y)))
        else:
            away_scatter.set_offsets(np.empty((0, 2)))

        bx, by = _extract_xy_from_ball(row["ballsSmoothed"])
        if bx is not None and by is not None:
            ball_scatter.set_offsets([[bx + 52.5, by + 34.0]])
        else:
            ball_scatter.set_offsets(np.empty((0, 2)))

        frame_text.set_text(f"frameNum={int(row['frameNum'])}")
        return home_scatter, away_scatter, ball_scatter, frame_text

    ani = animation.FuncAnimation(fig, update, frames=len(subset), interval=1000 / fps, blit=False)

    try:
        ani.save(output_video, writer="ffmpeg", fps=fps)
        saved_path = output_video
    except Exception:
        gif_path = output_video.with_suffix(".gif")
        ani.save(gif_path, writer="pillow", fps=fps)
        saved_path = gif_path
    finally:
        plt.close(fig)

    return saved_path


def build_subset(
    tracking_dir: Path,
    event_dir: Path,
    parquet: str | None,
    event_json: str | None,
    max_frames: int,
    output_csv: Path,
    output_video: Path,
    seed: int | None,
):
    if max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    parquet_path = _resolve_parquet_path(tracking_dir=tracking_dir, parquet_arg=parquet)
    df = pd.read_parquet(parquet_path)

    if event_json:
        event_path = Path(event_json)
    else:
        event_path = event_dir / f"{parquet_path.stem}.json"
    if not event_path.exists():
        raise FileNotFoundError(f"Event json not found: {event_path}")

    with open(event_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    if not isinstance(events, list):
        raise ValueError("Event json top-level must be a list.")

    selected_possession_event_id, selected_event = _extract_d_pass_possession_id(events, seed)

    _validate_required_columns(df)
    df = _add_nested_columns(df)
    subset = _pick_window_around_possession_event(
        df=df,
        possession_event_id=selected_possession_event_id,
        max_frames=max_frames,
    )

    output_df = subset[OUTPUT_COLUMNS].copy()
    output_df.to_csv(output_csv, index=False)
    saved_video = create_animation(subset=subset, output_video=output_video)

    print(f"Parquet used: {parquet_path}")
    print(f"Event json used: {event_path}")
    print(f"Selected D-pass gameEventId: {selected_event.get('gameEventId')}")
    print(f"Selected D-pass possessionEventId: {selected_possession_event_id}")
    print(f"Subset frameNum range: {int(output_df['frameNum'].min())} -> {int(output_df['frameNum'].max())}")
    print(f"Rows exported: {len(output_df)}")
    print(f"Output CSV: {output_csv.resolve()}")
    print(f"Output animation: {saved_video.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export subset containing PA/CH possession event followed by home_ball flip within 10 frames."
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=DEFAULT_TRACKING_DIR,
        help="Directory containing tracking parquet files.",
    )
    parser.add_argument(
        "--event-dir",
        type=Path,
        default=DEFAULT_EVENT_DIR,
        help="Directory containing event json files.",
    )
    parser.add_argument(
        "--parquet",
        default=None,
        help="Optional explicit parquet file path. If omitted, first parquet in tracking-dir is used.",
    )
    parser.add_argument(
        "--event-json",
        default=None,
        help="Optional explicit event json path. If omitted, <parquet_stem>.json in event-dir is used.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=250,
        help="Maximum number of contiguous frames to export.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("tmp_tracking_subset.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("tmp_tracking_subset_animation.mp4"),
        help="Output animation video path (mp4 preferred, falls back to gif if needed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for window selection among valid home_ball flips.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_subset(
        tracking_dir=args.tracking_dir,
        event_dir=args.event_dir,
        parquet=args.parquet,
        event_json=args.event_json,
        max_frames=args.max_frames,
        output_csv=args.output_csv,
        output_video=args.output_video,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
