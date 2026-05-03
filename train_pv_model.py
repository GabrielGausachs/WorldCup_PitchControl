import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Utils.config import DATA_ROOT


FEATURES = ["ball_x", "ball_y", "cell_x", "cell_y"]
TARGET = "defending_value"


def _extract_game_id(csv_path: Path) -> str:
    name = csv_path.stem  # dataset_<game_id>
    return name.replace("dataset_", "", 1)


def _list_dataset_files(dataset_dir: Path) -> list[Path]:
    files = sorted(dataset_dir.glob("dataset_*.csv"))
    if not files:
        raise FileNotFoundError(f"No dataset files found in: {dataset_dir}")
    return files


def _sample_games(files: list[Path], n_games: int, seed: int) -> tuple[list[str], dict[str, Path]]:
    game_to_file = {_extract_game_id(path): path for path in files}
    game_ids = sorted(game_to_file.keys())
    if len(game_ids) < n_games:
        raise ValueError(f"Requested {n_games} games but found only {len(game_ids)}.")
    rng = random.Random(seed)
    selected_games = sorted(rng.sample(game_ids, n_games))
    return selected_games, game_to_file


def _load_game_df(path: Path) -> pd.DataFrame:
    cols = ["game_id", "frameNum", *FEATURES, TARGET]
    return pd.read_csv(path, usecols=cols)


def _build_game_data(selected_games: list[str], game_to_file: dict[str, Path]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    game_data = {}
    for game_id in selected_games:
        df = _load_game_df(game_to_file[game_id])
        x = df[FEATURES].to_numpy(dtype=np.float32)
        y = df[TARGET].to_numpy(dtype=np.float32)
        game_data[game_id] = (x, y)
    return game_data


def _train_and_eval(train_x, train_y, val_x, val_y, hidden_size, seed, max_iter):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(hidden_size,),
                    activation="logistic",
                    solver="adam",
                    random_state=seed,
                    batch_size=4096,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=10,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    mlp = model.named_steps["mlp"]
    last_loss = getattr(mlp, "loss_", getattr(mlp, "loss", None))
    print("epochs:", mlp.n_iter_)
    print("best validation score:", mlp.best_validation_score_)
    print("last training loss:", last_loss)
    print("last 10 validation scores:", mlp.validation_scores_[-10:])
    print("last 10 training losses:", mlp.loss_curve_[-10:])
    pred = model.predict(val_x)
    mse = float(mean_squared_error(val_y, pred))
    diagnostics = {
        "epochs": mlp.n_iter_,
        "best_validation_score": mlp.best_validation_score_,
        "last_training_loss": last_loss,
        "last_10_validation_scores": mlp.validation_scores_[-10:],
        "last_10_training_losses": mlp.loss_curve_[-10:],
    }
    return mse, diagnostics


def train_model(
    base_path: str,
    n_games: int = 10,
    seed: int = 42,
    hidden_sizes: tuple[int, ...] = (8, 16, 32),
    cv_folds: int = 5,
    max_iter: int = 100,
    diagnostics_file: str | None = None,
):
    dataset_dir = Path(base_path) / "datasets_pitch_value"
    output_dir = Path(base_path) / "processed_pitch_value" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset directory: {dataset_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    files = _list_dataset_files(dataset_dir)
    print(f"[INFO] Found {len(files)} dataset files.")
    selected_games, game_to_file = _sample_games(files=files, n_games=n_games, seed=seed)
    print(f"[INFO] Random seed: {seed}")
    print(f"[INFO] Selected {len(selected_games)} games: {selected_games}")
    if cv_folds < 2 or cv_folds > len(selected_games):
        raise ValueError(f"cv_folds must be in [2, {len(selected_games)}], got {cv_folds}.")
    print(f"[INFO] CV folds: {cv_folds}")
    game_data = _build_game_data(selected_games=selected_games, game_to_file=game_to_file)
    selected_rows = int(sum(game_data[g][1].shape[0] for g in selected_games))
    print(f"[INFO] Loaded selected games. Total rows: {selected_rows}")
    diag_path = None
    if diagnostics_file:
        diag_path = Path(diagnostics_file)
    else:
        diag_path = output_dir / "pv_training_diagnostics.txt"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("Pitch Value Training Diagnostics\n")
        f.write(f"seed={seed}\n")
        f.write(f"selected_games={selected_games}\n")
        f.write(f"cv_folds={cv_folds}\n")
        f.write(f"hidden_sizes={list(hidden_sizes)}\n")
        f.write(f"max_iter={max_iter}\n")
        f.write("\n")

    hidden_size_results = {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    game_ids_array = np.array(selected_games)
    for hidden_size in hidden_sizes:
        print(f"[INFO] CV start for hidden_size={hidden_size}")
        fold_mse = []
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(game_ids_array), start=1):
            train_games = game_ids_array[train_idx].tolist()
            val_games = game_ids_array[val_idx].tolist()

            train_x = np.concatenate([game_data[g][0] for g in train_games], axis=0)
            train_y = np.concatenate([game_data[g][1] for g in train_games], axis=0)
            val_x = np.concatenate([game_data[g][0] for g in val_games], axis=0)
            val_y = np.concatenate([game_data[g][1] for g in val_games], axis=0)

            mse, fold_diag = _train_and_eval(
                train_x,
                train_y,
                val_x,
                val_y,
                hidden_size=hidden_size,
                seed=seed,
                max_iter=max_iter,
            )
            fold_mse.append({"val_game_ids": val_games, "mse": mse})
            print(
                f"[INFO] hidden_size={hidden_size} fold={fold_idx}/{cv_folds} "
                f"val_games={val_games} mse={mse:.6f}"
            )
            with open(diag_path, "a", encoding="utf-8") as f:
                f.write(f"hidden_size={hidden_size} fold={fold_idx}/{cv_folds}\n")
                f.write(f"val_games={val_games}\n")
                f.write(f"mse={mse}\n")
                f.write(f"epochs={fold_diag['epochs']}\n")
                f.write(f"best_validation_score={fold_diag['best_validation_score']}\n")
                f.write(f"last_training_loss={fold_diag['last_training_loss']}\n")
                f.write(f"last_10_validation_scores={fold_diag['last_10_validation_scores']}\n")
                f.write(f"last_10_training_losses={fold_diag['last_10_training_losses']}\n")
                f.write("\n")

        mean_mse = float(np.mean([m["mse"] for m in fold_mse]))
        hidden_size_results[str(hidden_size)] = {"fold_mse": fold_mse, "mean_mse": mean_mse}
        print(f"[INFO] CV done for hidden_size={hidden_size}. mean_mse={mean_mse:.6f}")

    best_hidden_size = min(
        (int(h) for h in hidden_size_results.keys()),
        key=lambda h: hidden_size_results[str(h)]["mean_mse"],
    )
    print(f"[INFO] Best hidden size selected: {best_hidden_size}")

    full_x = np.concatenate([game_data[g][0] for g in selected_games], axis=0)
    full_y = np.concatenate([game_data[g][1] for g in selected_games], axis=0)
    print("[INFO] Training final model on all selected games...")
    final_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=(best_hidden_size,),
                    activation="logistic",
                    solver="adam",
                    random_state=seed,
                    batch_size=4096,
                    max_iter=max_iter,
                    early_stopping=True,
                    n_iter_no_change=10,
                ),
            ),
        ]
    )
    final_model.fit(full_x, full_y)
    print("[INFO] Final model training complete.")
    final_mlp = final_model.named_steps["mlp"]
    final_last_loss = getattr(final_mlp, "loss_", getattr(final_mlp, "loss", None))
    print("epochs:", final_mlp.n_iter_)
    print("best validation score:", final_mlp.best_validation_score_)
    print("last training loss:", final_last_loss)
    print("last 10 validation scores:", final_mlp.validation_scores_[-10:])
    print("last 10 training losses:", final_mlp.loss_curve_[-10:])
    with open(diag_path, "a", encoding="utf-8") as f:
        f.write("final_model\n")
        f.write(f"best_hidden_size={best_hidden_size}\n")
        f.write(f"epochs={final_mlp.n_iter_}\n")
        f.write(f"best_validation_score={final_mlp.best_validation_score_}\n")
        f.write(f"last_training_loss={final_last_loss}\n")
        f.write(f"last_10_validation_scores={final_mlp.validation_scores_[-10:]}\n")
        f.write(f"last_10_training_losses={final_mlp.loss_curve_[-10:]}\n")
        f.write("\n")

    model_path = output_dir / "pv_mlp.pkl"
    metrics_path = output_dir / "pv_train_metrics.json"
    joblib.dump(final_model, model_path)

    metrics = {
        "seed": seed,
        "n_games_sampled": n_games,
        "selected_games": selected_games,
        "hidden_sizes_tested": list(hidden_sizes),
        "cv_folds": cv_folds,
        "max_iter": max_iter,
        "best_hidden_size": best_hidden_size,
        "cv_results": hidden_size_results,
        "n_rows_selected": int(full_y.shape[0]),
        "n_files_available": len(files),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Selected rows: {metrics['n_rows_selected']}")
    print(f"[INFO] Model saved: {model_path}")
    print(f"[INFO] Metrics saved: {metrics_path}")
    print(f"[INFO] Diagnostics saved: {diag_path}")


def main():
    parser = argparse.ArgumentParser(description="Train pitch value model on a random 10-game subset.")
    parser.add_argument("--base-path", default=DATA_ROOT, help="Root path containing datasets_pitch_value.")
    parser.add_argument("--n-games", type=int, default=5, help="Number of games to sample randomly.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for game sampling and model init.")
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=[8, 16, 32],
        help="Hidden layer sizes to evaluate.",
    )
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum training iterations for MLP.")
    parser.add_argument(
        "--diagnostics-file",
        default=None,
        help="Optional path to save training diagnostics text file.",
    )
    args = parser.parse_args()

    train_model(
        base_path=args.base_path,
        n_games=args.n_games,
        seed=args.seed,
        hidden_sizes=tuple(args.hidden_sizes),
        cv_folds=args.cv_folds,
        max_iter=args.max_iter,
        diagnostics_file=args.diagnostics_file,
    )


if __name__ == "__main__":
    main()
