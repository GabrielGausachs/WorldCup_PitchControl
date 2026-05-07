import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm, Normalize


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
    cmap = LinearSegmentedColormap.from_list(
        "pv_paper_style",
        [
            (0.00, "#ffffff"),  # 0.0 -> white
            (0.18, "#d9f2d9"),  # very light green
            (0.45, "#ffe680"),  # yellow
            (0.70, "#ffb347"),  # orange
            (1.00, "#c91515"),  # strong red
        ],
    )

    # Convert centered coordinates ([-L/2, L/2], [-W/2, W/2]) to mplsoccer custom pitch coordinates ([0, L], [0, W]).
    X_plot = X + (pitch_length / 2.0)
    Y_plot = Y + (pitch_width / 2.0)

    heat = ax.pcolormesh(
        X_plot,
        Y_plot,
        np.clip(Z, vmin, vmax),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        alpha=0.9,
        zorder=2,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    return heat


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
        X_plot = X + (pitch_length / 2.0)
        Y_plot = Y + (pitch_width / 2.0)
        heat = ax.pcolormesh(
            X_plot,
            Y_plot,
            np.asarray(arr, dtype=float),
            cmap=cmap,
            norm=norm,
            shading="auto",
            alpha=0.92,
            zorder=2,
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

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
        x_plot = x_use + (pitch_length / 2.0)
        y_plot = y_use + (pitch_width / 2.0)
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
            bx = bnx + (pitch_length / 2.0)
            by = bny + (pitch_width / 2.0)
            ax.scatter(
                bx,
                by,
                s=24,
                c="#ffffff",
                edgecolors="black",
                linewidths=0.7,
                zorder=10,
            )
