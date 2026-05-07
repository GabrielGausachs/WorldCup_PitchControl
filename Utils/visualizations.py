import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm, Normalize
import joblib

from Utils.config import DATA_ROOT
from Utils.helpers import (
    get_grid,
    goal_distance_factor,
    normalize_pitch_value_by_goal_distance,
    predict_surface,
)


PV_PAPER_CMAP = LinearSegmentedColormap.from_list(
    "pv_paper_style",
    [
        (0.00, "#ffffff"),
        (0.18, "#d9f2d9"),
        (0.45, "#ffe680"),
        (0.70, "#ffb347"),
        (1.00, "#c91515"),
    ],
)


def _to_pitch_plot_coords(X, Y, pitch_length=105.0, pitch_width=68.0):
    return X + (pitch_length / 2.0), Y + (pitch_width / 2.0)


def _render_base_heatmap(
    ax,
    X,
    Y,
    Z,
    title,
    pitch_length=105.0,
    pitch_width=68.0,
    cmap=None,
    vmin=None,
    vmax=None,
    norm=None,
    alpha=0.9,
):
    X_plot, Y_plot = _to_pitch_plot_coords(X, Y, pitch_length=pitch_length, pitch_width=pitch_width)
    heat = ax.pcolormesh(
        X_plot,
        Y_plot,
        Z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        shading="auto",
        alpha=alpha,
        zorder=2,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return heat


def draw_pitch(ax, pitch_length=105.0, pitch_width=68.0, fill_pitch=True):
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        line_color="black",
        pitch_color="white" if fill_pitch else "None",
        line_zorder=5,
    )
    pitch.draw(ax=ax)
    return pitch


def plot_heatmap(ax, X, Y, Z, title, vmin=0.0, vmax=1.0, pitch_length=105.0, pitch_width=68.0):
    heat = _render_base_heatmap(
        ax=ax,
        X=X,
        Y=Y,
        Z=np.clip(Z, vmin, vmax),
        title=title,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        cmap=PV_PAPER_CMAP,
        vmin=vmin,
        vmax=vmax,
        alpha=0.9,
    )
    return heat


def save_heatmaps(
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
        bx, by = _to_pitch_plot_coords(
            ball_x,
            ball_y,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        ax.scatter(bx, by, s=40, c="white", edgecolors="black", linewidths=0.8, zorder=5)
    draw_pitch(ax=ax, pitch_length=pitch_length, pitch_width=pitch_width, fill_pitch=False)
    fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def generate_heatmaps(
    base_path=DATA_ROOT,
    output_dir="Outputs",
    n_x=105,
    n_y=68,
    pitch_length=105.0,
    pitch_width=68.0,
):
    model_path = Path(base_path) / "processed_pitch_value" / "models" / "pv_mlp.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)
    X, Y = get_grid(n_x=n_x, n_y=n_y, pitch_length=pitch_length, pitch_width=pitch_width)

    norm_surface = goal_distance_factor(X, Y, pitch_length=pitch_length, pitch_width=pitch_width)
    save_heatmaps(
        X,
        Y,
        norm_surface,
        "Distance-to-goal normalization surface",
        output_dir / "01_goal_distance_normalization_surface.png",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    scenarios = [
        ("02_raw_ball_q1_centered.png", "Raw PV: ball Q1 centered (-26.25, 0.00)", -26.25, 0.0, False),
        ("03_raw_ball_center.png", "Raw PV: ball center (0.00, 0.00)", 0.0, 0.0, False),
        ("04_raw_ball_q3_top_lane.png", "Raw PV: ball Q3 top lane (26.25, 22.67)", 26.25, 22.67, False),
        ("05_raw_ball_q4_centered.png", "Raw PV: ball Q4 centered (39.38, 0.00)", 39.375, 0.0, False),
        ("06_norm_ball_q1_centered.png", "Normalized PV: ball Q1 centered (-26.25, 0.00)", -26.25, 0.0, True),
        ("07_norm_ball_center.png", "Normalized PV: ball center (0.00, 0.00)", 0.0, 0.0, True),
        ("08_norm_ball_q4_centered.png", "Normalized PV: ball Q4 centered (39.38, 0.00)", 39.375, 0.0, True),
    ]

    for file_name, title, ball_x, ball_y, use_normalization in scenarios:
        raw = predict_surface(model=model, ball_x=ball_x, ball_y=ball_y, X=X, Y=Y)
        surface = raw
        if use_normalization:
            surface = normalize_pitch_value_by_goal_distance(
                pv_raw=raw,
                cell_x=X,
                cell_y=Y,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )
        save_heatmaps(
            X,
            Y,
            surface,
            title,
            output_dir / file_name,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            ball_x=ball_x,
            ball_y=ball_y,
        )

    print(f"[INFO] Saved 8 heatmaps to: {output_dir}")


def save_space_quality_heatmaps(
    pitch_control: np.ndarray,
    pitch_value: np.ndarray,
    space_quality: np.ndarray,
    out_dir: str,
    prefix: str,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    frame_row=None,
    player_ids: list[str] | None = None,
    home_team_in_possession: bool | None = None,
) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-pitch_length / 2.0, pitch_length / 2.0, pitch_control.shape[0])
    y = np.linspace(-pitch_width / 2.0, pitch_width / 2.0, pitch_control.shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")

    pv_vmax = float(np.nanmax(pitch_value)) if np.isfinite(np.nanmax(pitch_value)) else 1.0
    if pv_vmax <= 0:
        pv_vmax = 1.0
    sq_vmax = float(np.nanmax(space_quality)) if np.isfinite(np.nanmax(space_quality)) else 1.0
    if sq_vmax <= 0:
        sq_vmax = 1.0

    specs = [
        (
            "pitch_control",
            "Pitch Control",
            pitch_control,
            LinearSegmentedColormap.from_list("pc_rwb", ["#d73027", "#ffffff", "#2166ac"]),
            TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
            "black",
        ),
        (
            "pitch_value",
            "Pitch Value",
            pitch_value,
            LinearSegmentedColormap.from_list("pv_gy", ["#1a9850", "#ffff66"]),
            Normalize(vmin=0.0, vmax=pv_vmax),
            "white",
        ),
        (
            "space_quality",
            "Space Quality",
            space_quality,
            LinearSegmentedColormap.from_list("sq_rg", ["#d73027", "#1a9850"]),
            Normalize(vmin=0.0, vmax=sq_vmax),
            "white",
        ),
    ]
    saved_files: list[Path] = []

    for suffix, title, arr, cmap, norm, line_color in specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        heat = _render_base_heatmap(
            ax=ax,
            X=X,
            Y=Y,
            Z=np.asarray(arr, dtype=float),
            title=title,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            cmap=cmap,
            norm=norm,
            alpha=0.92,
        )

        # Draw thin pitch lines.
        pitch = Pitch(
            pitch_type="custom",
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            line_color=line_color,
            pitch_color="None",
            line_zorder=6,
            linewidth=1.0,
        )
        pitch.draw(ax=ax)

        if frame_row is not None and player_ids is not None and home_team_in_possession is not None:
            _plot_players_and_ball(
                ax=ax,
                frame_row=frame_row,
                player_ids=player_ids,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                home_team_in_possession=home_team_in_possession,
            )

        fig.colorbar(heat, ax=ax, fraction=0.03, pad=0.02)
        fig.tight_layout()

        file_path = out_path / f"{prefix}_{suffix}.png"
        fig.savefig(file_path, dpi=200)
        plt.close(fig)
        saved_files.append(file_path)

    return saved_files


def _plot_players_and_ball(
    ax,
    frame_row,
    player_ids: list[str],
    pitch_length: float,
    pitch_width: float,
    home_team_in_possession: bool,
) -> None:
    home_face = "#225ea8"
    away_face = "#cb181d"
    text_color = "white"

    def _norm_xy(x_val: float, y_val: float) -> tuple[float, float]:
        if home_team_in_possession:
            return x_val, y_val
        return -x_val, -y_val

    for player_id in player_ids:
        x_key = f"{player_id}_x"
        y_key = f"{player_id}_y"
        if x_key not in frame_row.index or y_key not in frame_row.index:
            continue
        x_val = frame_row[x_key]
        y_val = frame_row[y_key]
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            continue

        x_use, y_use = _norm_xy(float(x_val), float(y_val))
        x_plot, y_plot = _to_pitch_plot_coords(
            x_use,
            y_use,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        team_is_home = str(player_id).startswith("home_")
        face = home_face if team_is_home else away_face
        jersey = str(player_id).split("_")[-1]

        ax.scatter(
            x_plot,
            y_plot,
            s=70,
            c=face,
            edgecolors="black",
            linewidths=0.5,
            zorder=8,
        )
        ax.text(
            x_plot,
            y_plot,
            jersey,
            ha="center",
            va="center",
            color=text_color,
            fontsize=6.5,
            zorder=9,
            fontweight="bold",
        )

    if "ball_x" in frame_row.index and "ball_y" in frame_row.index:
        b_x = frame_row["ball_x"]
        b_y = frame_row["ball_y"]
        if np.isfinite(b_x) and np.isfinite(b_y):
            bnx, bny = _norm_xy(float(b_x), float(b_y))
            bx, by = _to_pitch_plot_coords(
                bnx,
                bny,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
            )
            ax.scatter(
                bx,
                by,
                s=24,
                c="#ffffff",
                edgecolors="black",
                linewidths=0.7,
                zorder=10,
            )
