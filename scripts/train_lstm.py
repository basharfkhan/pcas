"""Train the single-aircraft LSTM baseline.

    python scripts/train_lstm.py --subset data/trajair/111_days/111_days

Days are split three ways, chronologically: train, then validation, then test. Validation
comes from days the model trains near but not on, and the test days are the same final 22
used by every other evaluation in this repo, so the comparison is like for like.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pcas.data.adsb import day_to_scenes, read_raw_day
from pcas.data.scenes import build_windows
from pcas.data.subsets import raw_day_files
from pcas.models.lstm import LSTMConfig, build_module, make_samples

log = logging.getLogger("pcas.train")


def windows_for(subset: Path, dates: list[str], stride: int, min_agents: int) -> list:
    files = raw_day_files(subset)
    windows = []
    for date in dates:
        day = read_raw_day(files[date])
        for scene in day_to_scenes(day, min_agents=min_agents):
            windows += build_windows(scene, stride=stride, min_agents=min_agents)
    return windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="data/trajair/111_days/111_days")
    parser.add_argument("--test-days", type=int, default=22)
    parser.add_argument("--val-days", type=int, default=11)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--no-wind", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out", default="artifacts/lstm")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device: %s", device)

    subset = Path(args.subset)
    dates = list(raw_day_files(subset))
    test_dates = dates[-args.test_days :]
    val_dates = dates[-(args.test_days + args.val_days) : -args.test_days]
    train_dates = dates[: -(args.test_days + args.val_days)]
    log.info(
        "days: %d train (%s to %s), %d val, %d test (held back)",
        len(train_dates),
        train_dates[0],
        train_dates[-1],
        len(val_dates),
        len(test_dates),
    )

    config = LSTMConfig(
        use_wind=not args.no_wind,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    log.info("building training windows...")
    train_windows = windows_for(subset, train_dates, args.stride, 1)
    x_train, y_train = make_samples(train_windows, config)
    del train_windows

    log.info("building validation windows...")
    val_windows = windows_for(subset, val_dates, args.stride, 1)
    x_val, y_val = make_samples(val_windows, config)
    del val_windows

    log.info("samples: %d train, %d val, %d features", len(x_train), len(x_val), x_train.shape[-1])

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=1024,
    )

    module = build_module(config).to(device)
    n_params = sum(p.numel() for p in module.parameters())
    log.info("parameters: %d", n_params)

    optimiser = torch.optim.Adam(module.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=2)
    # Huber, with delta set to 100 m expressed in the target's scaled units: ADS-B has
    # outliers, and squared error would let a handful of bad tracks dominate the gradient.
    loss_fn = nn.HuberLoss(delta=100.0 / config.target_scale)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, history = float("inf"), -1, []

    for epoch in range(args.epochs):
        module.train()
        started, total, seen = time.time(), 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = loss_fn(module(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), 5.0)
            optimiser.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        train_loss = total / max(seen, 1)

        module.eval()
        errors = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = module(xb.to(device)).cpu()
                # Report metres, not loss units: mean final displacement error.
                delta = (pred[:, -1] - yb[:, -1]) * config.target_scale
                errors.append(torch.linalg.norm(delta, dim=-1))
        val_fde = float(torch.cat(errors).mean())
        schedule.step(val_fde)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_fde_m": val_fde})
        log.info(
            "epoch %2d  train loss %8.2f  val FDE %7.1f m  (%.0fs)",
            epoch,
            train_loss,
            val_fde,
            time.time() - started,
        )

        if val_fde < best_val:
            best_val, best_epoch = val_fde, epoch
            torch.save(
                {"state_dict": module.state_dict(), "config": vars(config)},
                out_dir / "lstm.pt",
            )
        elif epoch - best_epoch >= args.patience:
            log.info("no improvement for %d epochs, stopping", args.patience)
            break

    (out_dir / "history.json").write_text(
        json.dumps(
            {
                "history": history,
                "best_epoch": best_epoch,
                "best_val_fde_m": best_val,
                "parameters": n_params,
                "train_days": [train_dates[0], train_dates[-1]],
                "val_days": [val_dates[0], val_dates[-1]],
                "test_days": [test_dates[0], test_dates[-1]],
                "config": vars(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("best val FDE %.1f m at epoch %d, saved to %s", best_val, best_epoch, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
