"""Test Aurora full model on GPU with random data.

Based on the Aurora docs usage example:
https://microsoft.github.io/aurora/usage.html

Run in a GPU pod:  python scripts/test_aurora_gpu.py
"""

import time
from datetime import datetime

import torch

from aurora import AuroraPretrained, AuroraSmallPretrained, Batch, Metadata, rollout

device = "cuda"

print("=" * 60)
print("Aurora Model — GPU Test")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"GPU:             {torch.cuda.get_device_name(0)}")
print(f"GPU memory:      {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
print()

# Step 1: Small model smoke test
print("Step 1: Small model smoke test on GPU...")
batch_small = Batch(
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

t0 = time.time()
model_small = AuroraSmallPretrained()
model_small.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
model_small.eval()
model_small = model_small.to(device)
print(f"  AuroraSmallPretrained loaded in {time.time() - t0:.1f}s")

t0 = time.time()
with torch.inference_mode():
    pred = model_small.forward(batch_small.to(device))
print(f"  Forward pass: {time.time() - t0:.1f}s")
print(f"  Output 2t shape: {pred.surf_vars['2t'].shape}")

del model_small
torch.cuda.empty_cache()
print(f"  GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print()

# Step 2: Full model load
print("Step 2: Loading AuroraPretrained (full 0.25° model)...")
t0 = time.time()
model = AuroraPretrained()
model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt")
model.eval()
model = model.to(device)
print(f"  Loaded in {time.time() - t0:.1f}s")
print(f"  GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print()

# Step 3: Full model forward pass (small grid to stay within memory)
print("Step 3: Full model forward pass (17x32 grid)...")
t0 = time.time()
with torch.inference_mode():
    pred = model.forward(batch_small.to(device))
elapsed = time.time() - t0

print(f"  Completed in {elapsed:.1f}s")
print(f"  Output 2t shape: {pred.surf_vars['2t'].shape}")
print(f"  Output 2t sample:")
print(f"    {pred.surf_vars['2t'][0, 0, :3, :3].cpu()}")
print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print()

# Step 4: 3-step autoregressive rollout
print("Step 4: Running 3-step autoregressive rollout on GPU...")
t0 = time.time()
with torch.inference_mode():
    preds = [p.to("cpu") for p in rollout(model, batch_small.to(device), steps=3)]
elapsed = time.time() - t0

print(f"  Completed in {elapsed:.1f}s ({elapsed / 3:.1f}s per step)")
for i, p in enumerate(preds):
    print(f"  Step {i + 1}: 2t mean={p.surf_vars['2t'].mean():.4f}, "
          f"std={p.surf_vars['2t'].std():.4f}")
print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print()

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
