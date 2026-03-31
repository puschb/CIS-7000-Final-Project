"""Quick test of Aurora small model on CPU with random data.

Based on the Aurora docs usage example:
https://microsoft.github.io/aurora/usage.html

Run locally:   uv run python scripts/test_aurora_cpu.py
Run in a pod:  python scripts/test_aurora_cpu.py
"""

import time
from datetime import datetime

import torch

from aurora import AuroraSmallPretrained, Batch, Metadata, rollout

print("=" * 60)
print("Aurora Small Model — CPU Test")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"Device:          cpu")
print()

# Step 1: Build a small random batch (from the docs)
print("Step 1: Constructing random batch (17x32 grid, 4 pressure levels)...")
batch = Batch(
    surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl")},
    static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
    atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
    metadata=Metadata(
        lat=torch.linspace(90, -90, 17),
        lon=torch.linspace(0, 360, 32 + 1)[:-1],
        time=(datetime(2020, 6, 1, 12, 0),),
        atmos_levels=(100, 250, 500, 850),
    ),
)
print(f"  surf_vars:  {list(batch.surf_vars.keys())}")
print(f"  static_vars: {list(batch.static_vars.keys())}")
print(f"  atmos_vars: {list(batch.atmos_vars.keys())}")
print(f"  shape (2t):  {batch.surf_vars['2t'].shape}")
print()

# Step 2: Load the small pretrained model
print("Step 2: Loading AuroraSmallPretrained checkpoint...")
t0 = time.time()
model = AuroraSmallPretrained()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
model.eval()
print(f"  Loaded in {time.time() - t0:.1f}s")
print()

# Step 3: Single-step forward pass
print("Step 3: Running single-step forward pass on CPU...")
t0 = time.time()
with torch.inference_mode():
    pred = model.forward(batch)
elapsed = time.time() - t0

print(f"  Completed in {elapsed:.1f}s")
print(f"  Output 2t shape: {pred.surf_vars['2t'].shape}")
print(f"  Output 2t sample:")
print(f"    {pred.surf_vars['2t'][0, 0, :3, :3]}")
print()

# Step 4: 3-step autoregressive rollout
print("Step 4: Running 3-step autoregressive rollout on CPU...")
t0 = time.time()
with torch.inference_mode():
    preds = [p.to("cpu") for p in rollout(model, batch, steps=3)]
elapsed = time.time() - t0

print(f"  Completed in {elapsed:.1f}s ({elapsed / 3:.1f}s per step)")
for i, p in enumerate(preds):
    print(f"  Step {i + 1}: 2t mean={p.surf_vars['2t'].mean():.4f}, "
          f"std={p.surf_vars['2t'].std():.4f}")
print()

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
