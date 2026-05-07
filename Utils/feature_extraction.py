from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from Utils.config import DATA_ROOT


def _build_meter_grid(pitch_length: float, pitch_width: float) -> tuple[np.ndarray, np.ndarray]:
    n_x = int(round(pitch_length))
    n_y = int(round(pitch_width))
    x = np.linspace(-pitch_length / 2.0, pitch_length / 2.0, n_x)
    y = np.linspace(-pitch_width / 2.0, pitch_width / 2.0, n_y)
    return np.meshgrid(x, y, indexing="ij")


def _predict_pitch_value_surface(model, ball_x: float, ball_y: float, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    feats = np.column_stack(
        [
            np.full(X.size, float(ball_x), dtype=float),
            np.full(Y.size, float(ball_y), dtype=float),
            X.ravel(),
            Y.ravel(),
        ]
    )
    pred = model.predict(feats)
    return np.clip(pred.reshape(X.shape), 0.0, 1.0)


def _extract_home_pitch_control(pitch_control_raw) -> np.ndarray:
    pc = np.asarray(pitch_control_raw)
    if pc.ndim == 4:
        # Expected: [frames, teams, x, y] or [frames, teams, y, x]
        pc = pc[0, 0]
    elif pc.ndim == 3:
        # Expected: [frames, x, y] or [teams, x, y]
        pc = pc[0]
    elif pc.ndim != 2:
        raise ValueError(f"Unsupported pitch-control shape: {pc.shape}")

    # Ensure orientation [x, y] to match PV grid.
    if pc.shape[0] < pc.shape[1]:
        pc = pc.T
    return np.clip(pc, 0.0, 1.0)


def _resize_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    x_idx = np.linspace(0, arr.shape[0] - 1, target_shape[0]).astype(int)
    y_idx = np.linspace(0, arr.shape[1] - 1, target_shape[1]).astype(int)
    return arr[np.ix_(x_idx, y_idx)]


def space_quality(
    frame_row,
    game,
    frame_idx: int,
    home_team_in_possession: bool,
    base_path: str = DATA_ROOT,
    model_rel_path: str = "processed_pitch_value/models/pv_mlp.pkl",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_path = Path(base_path) / model_rel_path
    if not model_path.exists():
        raise FileNotFoundError(f"PV model not found: {model_path}")

    pitch_length, pitch_width = float(game.pitch_dimensions[0]), float(game.pitch_dimensions[1])
    X, Y = _build_meter_grid(pitch_length=pitch_length, pitch_width=pitch_width)

    ball_x_raw = float(frame_row["ball_x"])
    ball_y_raw = float(frame_row["ball_y"])

    # Normalize frame context first: always model attacking left-to-right.
    if home_team_in_possession:
        ball_x = ball_x_raw
        ball_y = ball_y_raw
    else:
        ball_x = -ball_x_raw
        ball_y = -ball_y_raw

    model = joblib.load(model_path)
    pv = _predict_pitch_value_surface(model=model, ball_x=ball_x, ball_y=ball_y, X=X, Y=Y)

    pc_raw = game.tracking_data.get_pitch_control(
        game.pitch_dimensions,
        X.shape[0],
        Y.shape[1],
        frame_idx,
        frame_idx,
    )
    pc_home = _extract_home_pitch_control(pc_raw)
    pc_home = _resize_to_shape(pc_home, pv.shape)

    # DataBallPy game object is fixed to home attacking left-to-right.
    # Convert pitch control to attacking-team control in the same normalized frame.
    if home_team_in_possession:
        pc_att = pc_home
    else:
        pc_att = 1.0 - pc_home
        pc_att = pc_att[::-1, ::-1]

    sq = np.clip(pc_att * pv, 0.0, 1.0)
    return np.clip(pc_att, 0.0, 1.0), np.clip(pv, 0.0, 1.0), sq
