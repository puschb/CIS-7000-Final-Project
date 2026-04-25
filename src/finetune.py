"""Fine-tuning loop for Aurora extended with soil-moisture variables.

Run with:
    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python -m src.finetune
"""

import argparse
import time

import torch

from aurora import AuroraPretrained
from aurora.normalisation import locations, scales

from src.data import EXTRA_SURF_ERA5_TO_AURORA, SURF_VAR_NAMES, make_random_batch

ALL_SURF_VARS = SURF_VAR_NAMES + tuple(EXTRA_SURF_ERA5_TO_AURORA.values())

# ---------------------------------------------------------------------------
# Normalisation statistics for the new surface variables.
#
# These MUST be computed from your ERA5 training data before real training.
# Run ``python -m scripts.compute_norm_stats --data-dir /mnt/data/era5`` to
# compute them, then paste the values here.
#
# The placeholder values below (mean=0, std=1) will let the code run but will
# produce poor training dynamics.  Replace them with real values.
# ---------------------------------------------------------------------------
_NEW_VAR_NORM: dict[str, tuple[float, float]] = {
    # (location/mean, scale/std)
    "swvl1": (0.0, 1.0),  # TODO: replace with real stats
    "stl1": (0.0, 1.0),   # TODO: replace with real stats
    "sd": (0.0, 1.0),     # TODO: replace with real stats
}


def _register_new_var_normalisation() -> None:
    """Insert normalisation statistics for new variables into Aurora's globals."""
    for name, (loc, sc) in _NEW_VAR_NORM.items():
        locations[name] = loc
        scales[name] = sc


def compute_loss(pred, target):
    """MAE loss over all surface variables, matching Aurora's training objective."""
    loss = torch.tensor(0.0, device=next(iter(pred.surf_vars.values())).device)
    n = 0
    for name in pred.surf_vars:
        if name in target.surf_vars:
            loss = loss + torch.nn.functional.l1_loss(
                pred.surf_vars[name],
                target.surf_vars[name][:, -1:, :, :],
            )
            n += 1
    return loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Aurora fine-tuning (soil moisture)")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lr-new", type=float, default=1e-3, help="LR for new patch embeddings")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--n-lat", type=int, default=17, help="Number of latitude points")
    parser.add_argument("--n-lon", type=int, default=32, help="Number of longitude points")
    parser.add_argument("--n-levels", type=int, default=4, help="Number of pressure levels")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training steps: {args.steps}")
    print(f"Grid: {args.n_lat} x {args.n_lon}, {args.n_levels} levels")
    print(f"Surface vars: {ALL_SURF_VARS}")
    print()

    _register_new_var_normalisation()

    model = AuroraPretrained(
        autocast=True,
        surf_vars=ALL_SURF_VARS,
    )
    print("Loading checkpoint (strict=False for new variables)...")
    t0 = time.time()
    model.load_checkpoint(strict=False)
    print(f"Checkpoint loaded in {time.time() - t0:.1f}s")

    model = model.to(device)
    model.train()
    model.configure_activation_checkpointing()

    new_var_names = set(EXTRA_SURF_ERA5_TO_AURORA.values())
    new_embed_params = []
    base_params = []
    for name, param in model.named_parameters():
        is_new = any(v in name for v in new_var_names)
        if is_new:
            new_embed_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr},
            {"params": new_embed_params, "lr": args.lr_new},
        ],
    )

    print(f"\nStarting fine-tuning for {args.steps} steps...")
    print(f"  Base params: {len(base_params)}, lr={args.lr}")
    print(f"  New embed params: {len(new_embed_params)}, lr={args.lr_new}\n")

    for step in range(1, args.steps + 1):
        t0 = time.time()

        batch = make_random_batch(
            n_lat=args.n_lat,
            n_lon=args.n_lon,
            n_levels=args.n_levels,
        )
        target = make_random_batch(
            n_lat=args.n_lat,
            n_lon=args.n_lon,
            n_levels=args.n_levels,
        )

        optimizer.zero_grad()
        pred = model.forward(batch)
        loss = compute_loss(pred, target)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        elapsed = time.time() - t0
        print(
            f"Step {step}/{args.steps} | "
            f"loss={loss.item():.6f} | "
            f"grad_norm={grad_norm:.4f} | "
            f"time={elapsed:.1f}s"
        )

    print("\nFine-tuning complete.")


if __name__ == "__main__":
    main()
