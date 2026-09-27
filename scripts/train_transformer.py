"""Train the multi-agent Transformer, or its social ablation.

    python scripts/train_transformer.py --out artifacts/transformer
    python scripts/train_transformer.py --no-social --out artifacts/transformer_nosocial

The ablation is the point of running this twice. `--no-social` keeps the architecture, the
data, the loss and the schedule and removes only the neighbours, so a difference between the
two runs is attributable to interaction between aircraft rather than to learning in general.
Days are split the same way as every other run here, so results stay comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from pcas.data.adsb import day_to_scenes
from pcas.data.scenes import build_windows
from pcas.data.sources import iter_days, open_source
from pcas.models.transformer import TransformerConfig, build_module, make_samples

log = logging.getLogger("pcas.train_transformer")


def samples_for(source, dates, stride: int, config: TransformerConfig, min_agents: int):
    """Build model inputs for the given dates, one session at a time.

    Windows are converted to tensors and discarded per session rather than held for the whole
    split: the neighbour tensor is (N, T, F) per aircraft, so keeping every window alive
    costs several gigabytes across 300 sessions.
    """
    chunks = []
    for day in iter_days(source, dates):
        windows = []
        for scene in day_to_scenes(day, min_agents=min_agents):
            windows += build_windows(
                scene, stride=stride, min_agents=min_agents, include_dense=False
            )
        if windows:
            chunks.append(make_samples(windows, config))
        log.info("%s: %d windows", day.date, len(windows))

    if not chunks:
        raise ValueError("no windows built")
    return tuple(np.concatenate([chunk[i] for chunk in chunks]) for i in range(4))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["trajair", "tartan"], default="tartan")
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--airport", default="kbtp")
    parser.add_argument("--test-days", type=int, default=40)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--min-agents", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--social-layers", type=int, default=1)
    parser.add_argument("--max-neighbours", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--max-train-days", type=int, default=0, help="Cap training days (0 = all); for quick runs."
    )
    parser.add_argument("--no-social", action="store_true", help="Ablation: hide neighbours.")
    parser.add_argument("--no-wind", action="store_true")
    parser.add_argument(
        "--cache",
        default="",
        help="Directory for built samples. The social run and its ablation take identical\n"
        "inputs, so caching means the second run does not rebuild them.",
    )
    parser.add_argument("--out", default="artifacts/transformer")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = TransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_temporal_layers=args.temporal_layers,
        n_social_layers=args.social_layers,
        max_neighbours=args.max_neighbours,
        dropout=args.dropout,
        use_social=not args.no_social,
        use_wind=not args.no_wind,
    )
    log.info("device %s | social=%s | d_model=%d", device, config.use_social, config.d_model)

    source = open_source(args.source, args.root, args.airport)
    dates = source.dates()
    test_dates = dates[-args.test_days :]
    val_dates = dates[-(args.test_days + args.val_days) : -args.test_days]
    train_dates = dates[: -(args.test_days + args.val_days)]
    if args.max_train_days:
        # Most recent N, so a capped run still trains on days near the validation split.
        train_dates = train_dates[-args.max_train_days :]
    log.info(
        "%d days: %d train (%s to %s), %d val, %d test held back",
        len(dates),
        len(train_dates),
        train_dates[0],
        train_dates[-1],
        len(val_dates),
        len(test_dates),
    )

    def cached(split: str, dates: list[str]):
        """Build a split, or reuse it from disk.

        The cache key covers everything that changes the inputs. `use_social` deliberately
        does not: the ablation must see exactly the same data as the social run.
        """
        if not args.cache:
            return samples_for(source, dates, args.stride, config, args.min_agents)

        key = (
            f"{args.source}_{args.airport}_{split}_{len(dates)}d"
            f"_n{config.max_neighbours}_w{int(config.use_wind)}"
            f"_s{args.stride}_a{args.min_agents}"
        )
        path = Path(args.cache) / f"{key}.npz"
        if path.exists():
            log.info("loading %s samples from %s", split, path.name)
            blob = np.load(path)
            return tuple(blob[name] for name in ("x", "n", "m", "y"))

        built = samples_for(source, dates, args.stride, config, args.min_agents)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=built[0], n=built[1], m=built[2], y=built[3])
        log.info("cached %s samples to %s", split, path.name)
        return built

    log.info("building training samples...")
    x_train, n_train, m_train, y_train = cached("train", train_dates)
    log.info("building validation samples...")
    x_val, n_val, m_val, y_val = cached("val", val_dates)
    log.info(
        "samples: %d train, %d val | with neighbours: %.1f%%",
        len(x_train),
        len(x_val),
        100 * m_train.any(axis=1).mean(),
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(n_train),
            torch.from_numpy(m_train),
            torch.from_numpy(y_train),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val),
            torch.from_numpy(n_val),
            torch.from_numpy(m_val),
            torch.from_numpy(y_val),
        ),
        batch_size=512,
    )

    module = build_module(config).to(device)
    n_params = sum(p.numel() for p in module.parameters())
    log.info("parameters: %d", n_params)

    optimiser = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=3)
    loss_fn = nn.HuberLoss(delta=100.0 / config.target_scale)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val, best_epoch, history = float("inf"), -1, []

    for epoch in range(args.epochs):
        module.train()
        started, total, seen = time.time(), 0.0, 0
        for xb, nb, mb, yb in train_loader:
            xb, nb, mb, yb = xb.to(device), nb.to(device), mb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = loss_fn(module(xb, nb, mb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), 5.0)
            optimiser.step()
            total += loss.item() * len(xb)
            seen += len(xb)
        train_loss = total / max(seen, 1)

        module.eval()
        errors = []
        with torch.no_grad():
            for xb, nb, mb, yb in val_loader:
                pred = module(xb.to(device), nb.to(device), mb.to(device)).cpu()
                delta = (pred[:, -1] - yb[:, -1]) * config.target_scale
                errors.append(torch.linalg.norm(delta, dim=-1))
        val_fde = float(torch.cat(errors).mean())
        schedule.step(val_fde)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_fde_m": val_fde})
        log.info(
            "epoch %3d  train loss %8.4f  val FDE %7.1f m  (%.0fs)",
            epoch,
            train_loss,
            val_fde,
            time.time() - started,
        )

        if val_fde < best_val:
            best_val, best_epoch = val_fde, epoch
            torch.save(
                {"state_dict": module.state_dict(), "config": vars(config), "model": "transformer"},
                out_dir / "model.pt",
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
                "use_social": config.use_social,
                "train_days": [train_dates[0], train_dates[-1]],
                "test_days": [test_dates[0], test_dates[-1]],
                "config": vars(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("best val FDE %.1f m at epoch %d -> %s", best_val, best_epoch, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
