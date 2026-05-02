import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
def get_pitch_control_surface_radius(
    distance_to_ball: float, min_r: float = 4.0, max_r: float = 10.0
) -> float:
    """
    Calculate the pitch control surface radius based on the distance to the ball.
    Note that the article does not provide the mathematical function for this formula,
    only a figure (Figure 9 in Appendix 1.). The constants (972 and 3) are
    obtained by visual inspection. The value is refered to as R_i(t) in the article.

    Args:
        distance_to_ball (float): Distance from the player to the ball.
        min_r (float, optional): The minimal influence radius of a player.
            Defaults to 4.0.
        max_r (float, optional): The maximal influence radius of a player.
            Defaults to 10.0

    Returns:
        float: Pitch control surface radius.
    """
    return min(min_r + distance_to_ball**3 / 972, max_r)

def calculate_scaling_matrix(
    speed_magnitude: float,
    distance_to_ball: float,
    max_speed: float = 13.0,
) -> np.ndarray:
    """
    Calculate the scaling factors using the statistical technique presented in the
    article "Wide Open Spaces" by Fernandez & Born (2018).
    Determines the scaling factors based on the player's speed magnitude and the
    distance to the ball. The range of the player's pitch control surface radius is
    defined, and scaling factors are calculated. The scaling matrix is then expanded
    in the x direction and contracted in the y direction using these factors.

    This is formula 19 in the appendix, where S_i(t) is defined.

    Args:
        speed_magnitude (float): Magnitude of the player's speed in m/s.
        distance_to_ball (float): Distance from the player to the ball in meters.
        max_speed (float, optional): Max speed a player can have in m/s.
            Defaults to 13.0.

    Returns:
        np.ndarray: Scaling matrix.
    """

    # Refered to as R_i(t) in the article
    influence_radius = get_pitch_control_surface_radius(distance_to_ball)
    ratio_of_max_speed = np.power(min(speed_magnitude, max_speed), 2) / np.power(max_speed, 2)

    return np.array(
        [
            [(influence_radius + (influence_radius * ratio_of_max_speed)), 0],
            [0, (influence_radius - (influence_radius * ratio_of_max_speed))],
        ]
    )

def calculate_covariance_matrix(
    vx_val: float,
    vy_val: float,
    scaling_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate the covariance matrix using the statistical technique presented in the
    article "Wide Open Spaces" by Fernandez & Born (2018).
    It dynamically adjusts the covariance matrix to provide a player dominance
    distribution that considers both location and velocity. The method involves the
    singular value decomposition (SVD) algorithm, expressing the covariance matrix
    in terms of eigenvectors and eigenvalues. The rotation matrix and scaling
    matrix are then derived, incorporating the rotation angle of the speed vector
    and scaling factors in the x and y directions.

    The calculated value is COV_i(t) as defined in formula 20 of the appendix.

    Args:
        vx_val (float): Velocity of the player in the x-direction in m/s.
        vy_val (float): Velocity of the player in the y-direction in m/s.
        scaling_matrix (np.ndarray): 2 by 2 array based on the velocity of a
            player and its distance to the ball. Determines the spread af the
            covariance matrix.

    Returns:
        np.ndarray: Covariance matrix.
    """
    rotation_angle = np.arctan2(vy_val, vx_val)
    rotation_matrix = np.array(
        [
            [np.cos(rotation_angle), -np.sin(rotation_angle)],
            [np.sin(rotation_angle), np.cos(rotation_angle)],
        ]
    )

    # covariance_matrix = np.dot(np.dot(np.dot(rotation_matrix, scaling_matrix), scaling_matrix), np.linalg.inv(rotation_matrix))
    covariance_matrix = rotation_matrix @ (scaling_matrix @ scaling_matrix) @ rotation_matrix.T

    return covariance_matrix

def get_mean_position_of_influence(
    x_val: float,
    y_val: float,
    vx_val: float,
    vy_val: float,
) -> list[float]:
    """
    Calculate the mean position of player influence over time using the statistical
    technique presented in the article "Wide Open Spaces" by Fernandez & Born (2018).
    It considers the player's current position, velocity, and a specified time step.
    The mean position is obtained by translating the player's location at a given
    time by half the magnitude of the speed vector.

    This refers to u_i(t) definedin formula 21 of the appendix.

    Args:
        x_val (float): x-coordinate of the player's current position in meters.
        y_val (float): y-coordinate of the player's current position in meters.
        vx_val (float): Velocity of the player in the x-direction in m/s.
        vy_val (float): Velocity of the player in the y-direction in m/s.

    Returns:
        List[float]: Mean position [x, y] of the player's influence.
    """
    return np.array([x_val + 0.5 * vx_val, y_val + 0.5 * vy_val])


def get_player_influence(
    x_val: float,
    y_val: float,
    vx_val: float,
    vy_val: float,
    distance_to_ball: float,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Calculate player influence across the grid based on the statistical technique
    presented in the article "Wide Open Spaces" by Fernandez & Born (2018).
    It incorporates the position, velocity, and distance to the ball of a given
    player to determine the influence degree at each location on the field. The
    bivariate normal distribution is utilized to model the player's influence,
    and the result is normalized so that the sum of the players influence over
    all cells in the grid is 1. Thus, the value in a cell in the grid is the ratio
    of influence of that player in that cell.

    Args:
        x_val (float): x-coordinate of the player's current position in meters.
        y_val (float): y-coordinate of the player's current position in meters.
        vx_val (float): Velocity of the player in the x-direction in m/s.
        vy_val (float): Velocity of the player in the y-direction in m/s.
        distance_to_ball (float): distance between the ball and the (x_val, y_val) in
            meters.
        grid (np.ndarray]): Grid created with np.meshgrid.

    Returns:
        np.ndarray: Player influence values across the grid.
    """
    
    mean = get_mean_position_of_influence(x_val, y_val, vx_val, vy_val)
    scaling_matrix = calculate_scaling_matrix(
        np.hypot(vx_val, vy_val), distance_to_ball
    )
    covariance_matrix = calculate_covariance_matrix(vx_val, vy_val, scaling_matrix)
        

    grid_size = grid[0].shape
    positions = np.vstack([grid[0].ravel(), grid[1].ravel()]).T
    distribution = multivariate_normal(mean=mean, cov=covariance_matrix)
    influence_values = distribution.pdf(positions)
    influence_values = influence_values / np.max(influence_values)
    return influence_values.reshape(grid_size[0], grid_size[1])

def get_team_influence(
    frame: pd.Series, col_ids: list, grid: list, player_ball_distances: pd.Series | None = None
) -> np.ndarray:
    """
    Calculate the team influence of a given team at a given frame. The team influence
    is the sum of the individual player influences of the team.

    Args:
        frame (pd.Series): Row of the tracking data.
        col_ids (list): List of column ids of the players in the team.
        grid (list): Grid created with np.meshgrid.
        player_ball_distances (pd.Series, optional): Precomputed player ball distances.

    Returns:
        np.ndarray: Team influence values across the grid.
    """
    player_influence = []
    for col_id in col_ids:
        if pd.isnull(frame[f"{col_id}_vx"]):
            continue

        if player_ball_distances is not None:
            distance_to_ball = player_ball_distances.loc[col_id]
        else:
            distance_to_ball = np.linalg.norm(
                frame[[f"{col_id}_x", f"{col_id}_y"]].values
                - frame[["ball_x", "ball_y"]].values
            )

        player_influence.append(
            get_player_influence(
                x_val=frame[f"{col_id}_x"],
                y_val=frame[f"{col_id}_y"],
                vx_val=frame[f"{col_id}_vx"],
                vy_val=frame[f"{col_id}_vy"],
                distance_to_ball=distance_to_ball,
                grid=grid,
            )
        )
    team_influence = np.sum(player_influence, axis=0)
    return team_influence

