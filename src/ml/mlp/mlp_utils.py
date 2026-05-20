"""mlp_utils.py — Dataset, model, training loop, and evaluation helpers for MLP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.ml.xgboost.xgb_utils import (
    engineer_features,
    make_event_level_test_split,
    prepare_features,
)
from src.utils.geometry_utils import apply_corrected_distances

# ── Event-level validation split ─────────────────────────────────────────────


def make_event_level_val_split(
    df: pd.DataFrame,
    group_col: str,
    val_fraction: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split train_val DataFrame into train / val at the event level."""
    rng = np.random.default_rng(random_seed + 1)  # different seed from test split
    all_events = df[group_col].unique()
    n_val = max(1, int(len(all_events) * val_fraction))
    val_events = set(rng.choice(all_events, size=n_val, replace=False))
    mask_val = df[group_col].isin(val_events)
    return df[~mask_val].reset_index(drop=True), df[mask_val].reset_index(drop=True)


# ── Feature scaling ───────────────────────────────────────────────────────────


def fit_imputer(X_train: np.ndarray) -> SimpleImputer:
    """Fit a median imputer on training data only (handles NaN in features)."""
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    return imputer


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ── PyTorch Dataset ───────────────────────────────────────────────────────────


def make_tensor_dataset(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> TensorDataset:
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return TensorDataset(X_t, y_t)


# ── Model definition ──────────────────────────────────────────────────────────

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Training loop ─────────────────────────────────────────────────────────────


def train_one_epoch(
    model: MLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Run one training epoch, return mean train loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def evaluate(
    model: MLP,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluate model on a DataLoader, return mean loss."""
    model.eval()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


@torch.no_grad()
def predict_numpy(model: MLP, loader: DataLoader) -> np.ndarray:
    """Collect raw model predictions (in log-space if target was transformed)."""
    model.eval()
    preds = []
    for X_batch, _ in loader:
        preds.append(model(X_batch).cpu().numpy())
    return np.concatenate(preds, axis=0).squeeze()


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    epoch: int,
    model: MLP,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: MLP,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ── Full training run ─────────────────────────────────────────────────────────


def train_mlp(
    model: MLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg_train: Any,
    build_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Train the MLP with early stopping and TensorBoard logging.

    Returns a summary dict with best epoch, best val_loss, history lists.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_train.learning_rate,
        weight_decay=cfg_train.weight_decay,
    )

    # Optional LR scheduler — activated when lr_scheduler=True in config
    use_scheduler = getattr(cfg_train, "lr_scheduler", False)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
        )
        if use_scheduler
        else None
    )

    ckpt_dir = build_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    tb_dir = build_dir / cfg_train.tensorboard_subdir
    writer = SummaryWriter(log_dir=str(tb_dir))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    print(
        f"\nTraining for up to {cfg_train.epochs} epochs "
        f"(patience={cfg_train.patience}) …"
    )
    print(f'  TensorBoard: tensorboard --logdir "{tb_dir}"')

    for epoch in range(1, cfg_train.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Step scheduler on val loss
        if scheduler is not None:
            scheduler.step(val_loss)

        # Best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                build_dir / "best_model.pt", epoch, model, optimizer, val_loss
            )
        else:
            epochs_no_improve += 1

        # Periodic checkpoint
        if epoch % cfg_train.checkpoint_every_n_epochs == 0:
            save_checkpoint(
                ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                epoch,
                model,
                optimizer,
                val_loss,
            )

        # Console progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            flag = " <-- best" if epoch == best_epoch else ""
            print(
                f"  Epoch {epoch:4d}/{cfg_train.epochs} | "
                f"train={train_loss:.5f}  val={val_loss:.5f}{flag}"
            )

        # Early stopping
        if epochs_no_improve >= cfg_train.patience:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(no improvement for {cfg_train.patience} epochs)"
            )
            break

    writer.close()
    print(f"\n  Best val loss: {best_val_loss:.5f} at epoch {best_epoch}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ── RMSE in original units ────────────────────────────────────────────────────


def rmse_original_units(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    log_transform: bool,
) -> float:
    """Compute RMSE in mm/s (undoing log1p if used)."""
    if log_transform:
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
    else:
        y_true = y_true_log
        y_pred = y_pred_log
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ── Full pipeline (shared across all MLP versions) ────────────────────────────


def run_training(CONFIG: Any) -> None:
    """End-to-end training pipeline driven by a Config dataclass.

    Shared by all train_mlp_vN.py entry points — only the CONFIG import differs.
    """
    import dataclasses
    import json
    from datetime import datetime

    import pandas as pd

    torch.manual_seed(CONFIG.data.random_seed)
    np.random.seed(CONFIG.data.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    build_dir = CONFIG.models_root / f"{CONFIG.version_name}_{ts}"
    build_dir.mkdir(parents=True, exist_ok=True)
    print(f"Build folder: {build_dir}")

    # Load data
    print("\nLoading parquet ...")
    parquet_path = Path(CONFIG.parquet_path)
    if not parquet_path.exists():
        parquet_root = Path(r"P:\11210978-erju-ai\holten_parquet")
        v2_builds = sorted(parquet_root.glob("parquet_v002_*"), key=lambda p: p.name)
        if not v2_builds:
            raise FileNotFoundError("No parquet_v002_* builds found.")
        parquet_path = v2_builds[-1] / "dataset.parquet"
        print(f"  (auto-discovered latest v2: {parquet_path})")
    df = pd.read_parquet(parquet_path)
    # Apply geometry correction (no-op if new v2 build already has the column)
    if "effective_distance_to_active_track_m" not in df.columns:
        df = apply_corrected_distances(df)
        print("  Distance correction applied from holten.json.")
    print(f"  {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    # Splits (event-level — no leakage)
    train_val_df, test_df = make_event_level_test_split(
        df,
        group_col=CONFIG.data.group_col,
        test_fraction=CONFIG.data.test_size,
        random_seed=CONFIG.data.random_seed,
    )
    train_df, val_df = make_event_level_val_split(
        train_val_df,
        group_col=CONFIG.data.group_col,
        val_fraction=CONFIG.data.val_size,
        random_seed=CONFIG.data.random_seed,
    )
    print(f"\nSplit summary (by event):")
    print(
        f"  Train : {train_df[CONFIG.data.group_col].nunique():>4} events | {len(train_df):>6,} rows"
    )
    print(
        f"  Val   : {val_df[CONFIG.data.group_col].nunique():>4} events | {len(val_df):>6,} rows"
    )
    print(
        f"  Test  : {test_df[CONFIG.data.group_col].nunique():>4} events | {len(test_df):>6,} rows"
    )

    # Feature preparation
    def _prep(frame: pd.DataFrame):
        X, y, _ = prepare_features(
            frame,
            target_col=CONFIG.data.target_col,
            identifier_cols=CONFIG.data.identifier_cols,
            string_cols=CONFIG.data.string_cols,
        )
        X = engineer_features(X, CONFIG.fe)
        return X, y

    X_train, y_train = _prep(train_df)
    X_val, y_val = _prep(val_df)
    X_test, y_test = _prep(test_df)
    print(f"\nFeature matrix shape: {X_train.shape[1]} features")

    # Target transform
    if CONFIG.fe.log_transform_target:
        y_train_t = np.log1p(y_train.values)
        y_val_t = np.log1p(y_val.values)
        y_test_t = np.log1p(y_test.values)
    else:
        y_train_t = y_train.values
        y_val_t = y_val.values
        y_test_t = y_test.values

    # Impute + scale (fit on train only)
    imputer = fit_imputer(X_train.values)
    X_train_i = imputer.transform(X_train.values)
    X_val_i = imputer.transform(X_val.values)
    X_test_i = imputer.transform(X_test.values)

    scaler = fit_scaler(X_train_i)
    X_train_s = scaler.transform(X_train_i)
    X_val_s = scaler.transform(X_val_i)
    X_test_s = scaler.transform(X_test_i)
    print(f"Scaler fitted on training data only ({X_train.shape[1]} features).")

    # DataLoaders
    train_ds = make_tensor_dataset(X_train_s, y_train_t, device)
    val_ds = make_tensor_dataset(X_val_s, y_val_t, device)
    test_ds = make_tensor_dataset(X_test_s, y_test_t, device)
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG.train.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=CONFIG.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG.train.batch_size, shuffle=False)

    # Model
    model = MLP(
        input_size=X_train.shape[1],
        hidden_sizes=CONFIG.model.hidden_sizes,
        activation=CONFIG.model.activation,
        dropout=CONFIG.model.dropout,
        batch_norm=getattr(CONFIG.model, "batch_norm", False),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model}")
    print(f"  Trainable parameters: {n_params:,}")

    # Train
    history = train_mlp(
        model, train_loader, val_loader, CONFIG.train, build_dir, device
    )

    # Evaluate best model on test set
    print("\nLoading best model checkpoint for test evaluation ...")
    best_ckpt = torch.load(
        build_dir / "best_model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    y_pred_test_log = predict_numpy(model, test_loader)
    y_pred_val_log = predict_numpy(model, val_loader)
    test_rmse = rmse_original_units(
        y_test_t, y_pred_test_log, CONFIG.fe.log_transform_target
    )
    val_rmse = rmse_original_units(
        y_val_t, y_pred_val_log, CONFIG.fe.log_transform_target
    )
    print(f"\n  Val  RMSE: {val_rmse:.4f} mm/s  (best epoch {history['best_epoch']})")
    print(f"  Test RMSE: {test_rmse:.4f} mm/s")

    # Save artefacts
    cfg_dict = dataclasses.asdict(CONFIG)
    cfg_dict["parquet_path"] = str(CONFIG.parquet_path)
    cfg_dict["models_root"] = str(CONFIG.models_root)
    with open(build_dir / "config_snapshot.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    summary = {
        "version_name": CONFIG.version_name,
        "build_dir": str(build_dir),
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
        "n_params": n_params,
        "best_epoch": history["best_epoch"],
        "best_val_loss_log": history["best_val_loss"],
        "val_rmse_mms": val_rmse,
        "test_rmse_mms": test_rmse,
        "train_events": int(train_df[CONFIG.data.group_col].nunique()),
        "val_events": int(val_df[CONFIG.data.group_col].nunique()),
        "test_events": int(test_df[CONFIG.data.group_col].nunique()),
    }
    with open(build_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    pred_df = pd.concat(
        [
            val_df[[CONFIG.data.group_col, "sensor_id", CONFIG.data.target_col]].assign(
                y_pred_log=y_pred_val_log,
                y_pred=(
                    np.expm1(y_pred_val_log)
                    if CONFIG.fe.log_transform_target
                    else y_pred_val_log
                ),
                split="val",
            ),
            test_df[
                [CONFIG.data.group_col, "sensor_id", CONFIG.data.target_col]
            ].assign(
                y_pred_log=y_pred_test_log,
                y_pred=(
                    np.expm1(y_pred_test_log)
                    if CONFIG.fe.log_transform_target
                    else y_pred_test_log
                ),
                split="test",
            ),
        ],
        ignore_index=True,
    )
    pred_df.to_parquet(build_dir / "predictions.parquet", index=False)

    hist_df = pd.DataFrame(
        {
            "epoch": range(1, len(history["train_losses"]) + 1),
            "train_loss": history["train_losses"],
            "val_loss": history["val_losses"],
        }
    )
    hist_df.to_csv(build_dir / "training_history.csv", index=False)

    print(f"\nArtefacts saved to: {build_dir}")
    print(f"  config_snapshot.json  summary.json  predictions.parquet")
    print(f"  training_history.csv  best_model.pt")
    print(f"  checkpoints/  (every {CONFIG.train.checkpoint_every_n_epochs} epochs)")
    print(f"  runs/          (TensorBoard logs)")
    print(f"\nDone.")
