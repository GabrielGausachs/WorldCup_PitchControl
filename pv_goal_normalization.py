import numpy as np


def goal_distance_factor(cell_x, cell_y, pitch_length=105.0, pitch_width=68.0):
    """
    Deterministic [0, 1] factor based on Euclidean distance to opponent goal center.
    Assumes attacking direction normalized left-to-right.
    """
    cell_x = np.asarray(cell_x, dtype=float)
    cell_y = np.asarray(cell_y, dtype=float)

    goal_x = pitch_length / 2.0
    goal_y = 0.0
    dist = np.sqrt((goal_x - cell_x) ** 2 + (goal_y - cell_y) ** 2)

    # Max relevant distance within the pitch from opponent goal center to far side corners.
    max_dist = np.sqrt((pitch_length) ** 2 + (pitch_width / 2.0) ** 2)
    return np.clip(1.0 - (dist / max_dist), 0.0, 1.0)


def normalize_pitch_value_by_goal_distance(
    pv_raw, cell_x, cell_y, pitch_length=105.0, pitch_width=68.0
):
    """
    Apply distance-to-goal normalization to raw pitch value predictions.
    """
    pv_raw = np.asarray(pv_raw, dtype=float)
    factor = goal_distance_factor(
        cell_x=cell_x,
        cell_y=cell_y,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )
    return np.clip(pv_raw * factor, 0.0, 1.0)
