"""Run Aurora inference on test data."""

import argparse
import time

import torch

from aurora import AuroraPretrained, AuroraSmallPretrained, rollout

from src.data import make_random_batch


def main():
    parser = argparse.ArgumentParser(description="Aurora inference test")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--small", action="store_true", help="Use small model (for testing)")
    parser.add_argument("--rollout-steps", type=int, default=0, help="Autoregressive steps (0 = single step)")
    parser.add_argument("--n-lat", type=int, default=17, help="Number of latitude points")
    parser.add_argument("--n-lon", type=int, default=32, help="Number of longitude points")
    parser.add_argument("--n-levels", type=int, default=4, help="Number of pressure levels")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Model: {'AuroraSmallPretrained' if args.small else 'AuroraPretrained'}")
    print(f"Grid: {args.n_lat} x {args.n_lon}, {args.n_levels} levels")
    print()

    if args.small:
        model = AuroraSmallPretrained()
    else:
        model = AuroraPretrained()

    print("Loading checkpoint...")
    t0 = time.time()
    model.load_checkpoint()
    print(f"Checkpoint loaded in {time.time() - t0:.1f}s")

    model.eval()
    model = model.to(args.device)

    batch = make_random_batch(
        n_lat=args.n_lat,
        n_lon=args.n_lon,
        n_levels=args.n_levels,
    )

    if args.rollout_steps > 0:
        print(f"\nRunning {args.rollout_steps}-step autoregressive rollout...")
        t0 = time.time()
        with torch.inference_mode():
            preds = [p.to("cpu") for p in rollout(model, batch, steps=args.rollout_steps)]
        elapsed = time.time() - t0
        print(f"Rollout complete in {elapsed:.1f}s ({elapsed / args.rollout_steps:.1f}s/step)")

        for i, pred in enumerate(preds):
            print(f"  Step {i + 1}: 2t shape={pred.surf_vars['2t'].shape}, "
                  f"mean={pred.surf_vars['2t'].mean():.4f}")
    else:
        print("\nRunning single-step forward pass...")
        t0 = time.time()
        with torch.inference_mode():
            pred = model.forward(batch)
        elapsed = time.time() - t0
        print(f"Forward pass complete in {elapsed:.1f}s")

        print(f"\nPrediction shapes:")
        for name, tensor in pred.surf_vars.items():
            print(f"  surf_vars['{name}']: {tensor.shape}")
        for name, tensor in pred.atmos_vars.items():
            print(f"  atmos_vars['{name}']: {tensor.shape}")

        print(f"\nSample values (2t):")
        print(f"  {pred.surf_vars['2t'][0, 0, :3, :3]}")

    print("\nInference test passed.")


if __name__ == "__main__":
    main()
