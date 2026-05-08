import argparse
from pathlib import Path

import joblib

from generate_pv_heatmaps import get_grid, predict_surface, save_heatmap
from Utils.helpers import goal_distance_factor, normalize_pitch_value_by_goal_distance


def run(
    base_path: str,
    output_dir: str = "Outputs",
    model_rel_path: str = "processed_pitch_value_new/models/pv_mlp.pkl",
    n_x: int = 105,
    n_y: int = 68,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> None:
    model_path = Path(base_path) / model_rel_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)
    X, Y = get_grid(n_x=n_x, n_y=n_y, pitch_length=pitch_length, pitch_width=pitch_width)

    norm_surface = goal_distance_factor(X, Y, pitch_length=pitch_length, pitch_width=pitch_width)
    save_heatmap(
        X,
        Y,
        norm_surface,
        "Distance-to-goal normalization surface",
        out_dir / "01_goal_distance_normalization_surface.png",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    scenarios_raw = [
        ("02_raw_ball_q1_centered.png", "Raw PV: ball Q1 centered (-26.25, 0.00)", -26.25, 0.0),
        ("03_raw_ball_center.png", "Raw PV: ball center (0.00, 0.00)", 0.0, 0.0),
        ("04_raw_ball_q3_top_lane.png", "Raw PV: ball Q3 top lane (26.25, 22.67)", 26.25, 22.67),
        ("05_raw_ball_q4_centered.png", "Raw PV: ball Q4 centered (39.38, 0.00)", 39.375, 0.0),
    ]
    for file_name, title, ball_x, ball_y in scenarios_raw:
        raw = predict_surface(model=model, ball_x=ball_x, ball_y=ball_y, X=X, Y=Y)
        save_heatmap(
            X,
            Y,
            raw,
            title,
            out_dir / file_name,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            ball_x=ball_x,
            ball_y=ball_y,
        )

    scenarios_norm = [
        ("06_norm_ball_q1_centered.png", "Normalized PV: ball Q1 centered (-26.25, 0.00)", -26.25, 0.0),
        ("07_norm_ball_center.png", "Normalized PV: ball center (0.00, 0.00)", 0.0, 0.0),
        ("08_norm_ball_q4_centered.png", "Normalized PV: ball Q4 centered (39.38, 0.00)", 39.375, 0.0),
    ]
    for file_name, title, ball_x, ball_y in scenarios_norm:
        raw = predict_surface(model=model, ball_x=ball_x, ball_y=ball_y, X=X, Y=Y)
        norm = normalize_pitch_value_by_goal_distance(
            pv_raw=raw,
            cell_x=X,
            cell_y=Y,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        save_heatmap(
            X,
            Y,
            norm,
            title,
            out_dir / file_name,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            ball_x=ball_x,
            ball_y=ball_y,
        )

    print(f"[INFO] Saved 8 heatmaps to: {out_dir}")
    print(f"[INFO] Model used: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporary heatmap generator that loads PV model from processed_pitch_value_new."
    )
    parser.add_argument(
        "--base-path",
        default=r"C:\Users\gausachsfernandezg\OneDrive - TNO\project_football\pff_data",
        help="Base path that contains processed_pitch_value_new/models.",
    )
    parser.add_argument(
        "--model-rel-path",
        default="processed_pitch_value_new/models/pv_mlp.pkl",
        help="Model path relative to base-path.",
    )
    parser.add_argument("--output-dir", default="Outputs", help="Folder to save heatmap PNGs.")
    parser.add_argument("--grid-x", type=int, default=105, help="Number of x coordinates for inference grid.")
    parser.add_argument("--grid-y", type=int, default=68, help="Number of y coordinates for inference grid.")
    args = parser.parse_args()

    run(
        base_path=args.base_path,
        model_rel_path=args.model_rel_path,
        output_dir=args.output_dir,
        n_x=args.grid_x,
        n_y=args.grid_y,
    )


if __name__ == "__main__":
    main()
