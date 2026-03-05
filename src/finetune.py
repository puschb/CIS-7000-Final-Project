"""Basic fine-tuning loop for Aurora, based on the official example.

Run with:
    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python -m src.finetune
"""

import argparse
import time

import torch

from aurora import AuroraPretrained

from src.data import make_random_batch


def compute_loss(pred, target):
    """Simple MSE loss over surface variables.

    In a real application, you'd design this loss to match your objective,
    potentially weighting different variables differently.
    """
    loss = torch.tensor(0.0, device=next(iter(pred.surf_vars.values())).device)
    for name in pred.surf_vars:
        if name in target.surf_vars:
            loss = loss + torch.nn.functional.mse_loss(
                pred.surf_vars[name],
                target.surf_vars[name][:, -1:, :, :],
            )
    return loss


def main():
    parser = argparse.ArgumentParser(description="Aurora fine-tuning test")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--n-lat", type=int, default=17, help="Number of latitude points")
    parser.add_argument("--n-lon", type=int, default=32, help="Number of longitude points")
    parser.add_argument("--n-levels", type=int, default=4, help="Number of pressure levels")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Training steps: {args.steps}")
    print(f"Grid: {args.n_lat} x {args.n_lon}, {args.n_levels} levels")
    print()

    model = AuroraPretrained(autocast=True)
    print("Loading checkpoint...")
    t0 = time.time()
    model.load_checkpoint()
    print(f"Checkpoint loaded in {time.time() - t0:.1f}s")

    model = model.to(device)
    model.train()
    model.configure_activation_checkpointing()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nStarting fine-tuning for {args.steps} steps...\n")

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

    print("\nFine-tuning test complete.")


if __name__ == "__main__":
    main()
