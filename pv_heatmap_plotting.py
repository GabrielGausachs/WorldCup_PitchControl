import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch


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
