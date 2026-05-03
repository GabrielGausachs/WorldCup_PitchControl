import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

from Utils.config import DATA_ROOT
from pv_goal_normalization import goal_distance_factor, normalize_pitch_value_by_goal_distance
from pv_heatmap_plotting import draw_pitch, plot_heatmap


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


def save_heatmap(
    X,
    Y,
    Z,
    title,
    out_path,
    pitch_length=105.0,
    pitch_width=68.0,
    ball_x=None,
    ball_y=None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    draw_pitch(ax=ax, pitch_length=pitch_length, pitch_width=pitch_width, fill_pitch=True)
    heat = plot_heatmap(
        ax=ax,
        X=X,
        Y=Y,
        Z=Z,
        title=title,
        vmin=0.0,
        vmax=1.0,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )
    if ball_x is not None and ball_y is not None:
        bx = ball_x + (pitch_length / 2.0)
        by = ball_y + (pitch_width / 2.0)
        ax.scatter(bx, by, s=40, c="white", edgecolors="black", linewidths=0.8, zorder=5)
    # Redraw pitch lines on top so all markings stay black and visible.
    draw_pitch(ax=ax, pitch_length=pitch_length, pitch_width=pitch_width, fill_pitch=False)
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_heatmaps(base_path, output_dir, n_x=105, n_y=68, pitch_length=105.0, pitch_width=68.0):
    model_path = Path(base_path) / "processed_pitch_value" / "models" / "pv_mlp.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)
    X, Y = get_grid(n_x=n_x, n_y=n_y, pitch_length=pitch_length, pitch_width=pitch_width)

    norm_surface = goal_distance_factor(X, Y, pitch_length=pitch_length, pitch_width=pitch_width)
    save_heatmap(
        X,
        Y,
        norm_surface,
        "Distance-to-goal normalization surface",
        output_dir / "01_goal_distance_normalization_surface.png",
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
            output_dir / file_name,
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
            output_dir / file_name,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            ball_x=ball_x,
            ball_y=ball_y,
        )

    print(f"[INFO] Saved 8 heatmaps to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper-style PV heatmaps.")
    parser.add_argument("--base-path", default=DATA_ROOT, help="Root path containing processed_pitch_value/models.")
    parser.add_argument("--output-dir", default="Outputs", help="Folder to save heatmap PNGs.")
    parser.add_argument("--grid-x", type=int, default=105, help="Number of x coordinates for inference grid.")
    parser.add_argument("--grid-y", type=int, default=68, help="Number of y coordinates for inference grid.")
    args = parser.parse_args()

    generate_heatmaps(
        base_path=args.base_path,
        output_dir=args.output_dir,
        n_x=args.grid_x,
        n_y=args.grid_y,
    )


if __name__ == "__main__":
    main()
