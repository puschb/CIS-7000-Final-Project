# Aurora 0.25° Pretrained on GCP — Setup Plan

## Overview

Set up Microsoft's Aurora 0.25° Pretrained foundation model for atmospheric prediction
on Google Cloud Platform, using Terraform for infrastructure-as-code. The setup supports
both inference and fine-tuning on custom data, starting with test/sample data as a POC.

```
┌─────────────────────────────────────────────────────────┐
│                    Local Machine                        │
│  - Terraform CLI (provisions GCP resources)             │
│  - gcloud CLI (auth, SSH, file transfer)                │
│  - VS Code + Remote-SSH (connect to VM)                 │
│  - uv (local development / testing)                     │
└──────────────────────┬──────────────────────────────────┘
                       │ SSH / gcloud
                       ▼
┌─────────────────────────────────────────────────────────┐
│              GCP (us-east1) - Terraform Managed         │
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │  Compute Engine VM (preemptible/spot)         │       │
│  │  - a2-ultragpu-1g (1x A100 80GB)             │       │
│  │  - NVIDIA GPU drivers pre-installed           │       │
│  │  - Docker + NVIDIA Container Toolkit          │       │
│  │  - Boot disk: 256 GB SSD                      │       │
│  │                                               │       │
│  │  ┌─────────────────────────────────┐          │       │
│  │  │  Docker Container               │          │       │
│  │  │  - NVIDIA PyTorch base image    │          │       │
│  │  │  - microsoft-aurora (pip)       │          │       │
│  │  │  - Project code mounted         │          │       │
│  │  └─────────────────────────────────┘          │       │
│  └──────────────────────────────────────────────┘       │
│                                                         │
│  ┌──────────────────────────────────────────────┐       │
│  │  GCS Bucket                                   │       │
│  │  - Training data                              │       │
│  │  - Model checkpoints                          │       │
│  │  - Static variables (from HuggingFace)        │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## Project Directory Structure

```
CIS-7000-Final-Project/
├── terraform/                  # GCP infrastructure-as-code
│   ├── main.tf                 # Provider, core resources
│   ├── variables.tf            # Configurable inputs (GPU type, region, disk size, etc.)
│   ├── outputs.tf              # VM IP, GCS bucket name, SSH command
│   ├── startup.sh              # VM startup script (install Docker, NVIDIA toolkit, clone repo)
│   └── terraform.tfvars        # Your specific variable values (gitignored)
├── docker/
│   └── Dockerfile              # CUDA + PyTorch + Aurora environment (inference & fine-tuning)
├── src/
│   ├── inference.py            # Run Aurora inference on a batch
│   ├── finetune.py             # Fine-tuning loop (based on Aurora's example)
│   └── data.py                 # Data loading / batch construction utilities
├── scripts/
│   ├── setup_vm.sh             # Post-provision VM setup (pull Docker image, sync data)
│   ├── run_inference.sh        # Convenience script to run inference in Docker
│   └── run_finetune.sh         # Convenience script to run fine-tuning in Docker
├── pyproject.toml              # Project metadata + dependencies (managed by uv)
├── .python-version             # Pin Python version for uv
├── .gitignore                  # Updated with terraform, data, checkpoint ignores
├── PLAN.md                     # This file
└── README.md                   # Project README
```

---

## Step-by-Step Plan

### Phase 1: Local Prerequisites (manual, one-time)

These are things you must do manually before Terraform can run.

#### 1.1 Install tools locally

- **Terraform**: https://developer.hashicorp.com/terraform/install
- **gcloud CLI**: https://cloud.google.com/sdk/docs/install
- **uv**: https://docs.astral.sh/uv/getting-started/installation/

```bash
# Verify installations
terraform --version
gcloud --version
uv --version
```

#### 1.2 Create a GCP project and enable billing

1. Go to https://console.cloud.google.com/
2. Create a new project (e.g., `cis7000-aurora`)
3. Link a billing account to the project
4. Note the **project ID** (not the project name)

#### 1.3 Authenticate and configure gcloud

```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>
gcloud auth application-default login   # Terraform uses this
```

#### 1.4 Enable required GCP APIs

```bash
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable iam.googleapis.com
```

#### 1.5 Request GPU quota (CRITICAL — often takes 24-48 hours)

By default GCP gives you **zero** GPU quota. You must request it:

1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter for: `NVIDIA A100 80GB GPUs` in region `us-east1`
3. Request an increase to **4** (this is a ceiling, not a commitment — you can use 1, 2,
   3, or 4 at any time and only pay for what you provision)
4. Provide a justification (e.g., "ML research — atmospheric model fine-tuning")

> **Note**: This is the most common blocker. Start this immediately. If `us-east1` is
> slow to approve, `us-central1` sometimes has faster approval.
>
> **You can continue all other setup steps while waiting for quota approval.** The only
> thing that requires approved quota is actually creating the GPU VM (`terraform apply`
> for the compute instance). Everything else — Terraform configs, Dockerfile, Python code,
> GCS bucket creation, local testing with the small model on CPU — works without it.

#### 1.6 Create an SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# The public key at ~/.ssh/id_ed25519.pub will be added to the VM by Terraform
```

---

### Phase 2: Terraform Infrastructure Setup

#### 2.1 Create Terraform configuration files

We create 4 files in `terraform/`:

**`variables.tf`** — Configurable inputs with sensible defaults:

| Variable | Default | Description |
|---|---|---|
| `project_id` | (required) | Your GCP project ID |
| `region` | `us-east1` | GCP region |
| `zone` | `us-east1-b` | GCP zone (A100s available here) |
| `machine_type` | `a2-ultragpu-1g` | 1x A100 80GB. Change to `a2-highgpu-1g` for A100 40GB, `a2-ultragpu-2g` for 2x A100 80GB, or `n1-standard-8` + `nvidia-tesla-t4` for testing |
| `gpu_type` | `nvidia-tesla-a100` | GPU accelerator type |
| `gpu_count` | `1` | Number of GPUs |
| `disk_size_gb` | `256` | Boot disk size (easy to change — just update this value and `terraform apply`) |
| `preemptible` | `true` | Use spot/preemptible pricing (~60-70% cheaper) |
| `gcs_bucket_name` | `cis7000-aurora-data` | Bucket for data and checkpoints |
| `ssh_user` | (required) | Your SSH username |
| `ssh_pub_key_path` | `~/.ssh/id_ed25519.pub` | Path to your SSH public key |

> **GPU scaling**: To switch GPU types, you just change `machine_type` in
> `terraform.tfvars` and run `terraform apply`. For multi-GPU sharding (FSDP),
> change to a multi-GPU machine type like `a2-ultragpu-2g` (2x A100 80GB).

**`main.tf`** — Core resources:

- **Google provider** configuration
- **GCS bucket** for data and checkpoints (with `lifecycle { prevent_destroy = true }` so
  `terraform destroy` does not accidentally delete your data — it persists across VM lifecycles)
- **Service account** with GCS read/write permissions (attached to the VM)
- **VPC firewall rule** allowing SSH (port 22)
- **Compute Engine instance** with:
  - The selected GPU machine type
  - Preemptible/spot scheduling
  - Boot image: NVIDIA Deep Learning VM (comes with NVIDIA drivers pre-installed)
  - Startup script that installs Docker + NVIDIA Container Toolkit
  - Service account with GCS read/write access
  - SSH key metadata

**`outputs.tf`** — Useful outputs after `terraform apply`:

- VM external IP address
- SSH command to connect
- GCS bucket URL
- VS Code Remote-SSH config snippet

**`startup.sh`** — Runs on VM boot:

- Installs Docker and NVIDIA Container Toolkit
- Configures Docker to use NVIDIA runtime by default
- Installs `gcsfuse` for mounting GCS buckets
- Pulls the Aurora Docker image (or builds from the repo)

#### 2.2 GPU machine type reference

For easy reconfiguration, here's the mapping:

| Machine Type | GPU | VRAM | Use Case | ~Cost/hr (spot) |
|---|---|---|---|---|
| `n1-standard-8` + `nvidia-tesla-t4` | 1x T4 | 16 GB | Quick testing (small model only) | ~$0.35 |
| `g2-standard-8` | 1x L4 | 24 GB | Testing inference (small model) | ~$0.70 |
| `a2-highgpu-1g` | 1x A100 | 40 GB | Inference (full model) | ~$3.50 |
| `a2-ultragpu-1g` | 1x A100 | 80 GB | Fine-tuning (full model) | ~$5.50 |
| `a2-ultragpu-2g` | 2x A100 | 160 GB | Multi-GPU fine-tuning (FSDP) | ~$11.00 |
| `a2-megagpu-16g` | 16x A100 | 640 GB | Large-scale distributed training | ~$44.00 |

#### 2.3 Provision the infrastructure

```bash
cd terraform/
terraform init
terraform plan          # Review what will be created
terraform apply         # Create the resources (type 'yes' to confirm)
```

#### 2.4 Connect to the VM

```bash
# SSH directly
ssh -i ~/.ssh/id_ed25519 <user>@<VM_EXTERNAL_IP>

# Or use gcloud (handles SSH keys automatically)
gcloud compute ssh <instance-name> --zone us-east1-b

# VS Code: Add to ~/.ssh/config (Terraform outputs this)
# Host aurora-vm
#     HostName <VM_EXTERNAL_IP>
#     User <user>
#     IdentityFile ~/.ssh/id_ed25519
# Then in VS Code: Remote-SSH -> Connect to Host -> aurora-vm
```

#### 2.5 Manage VM lifecycle (important for cost!)

There are two approaches for managing the VM when you're not using it:

**Option A: Stop/Start (preserves local disk data)**

```bash
# Stop the VM (no compute charges, but disk charges continue ~$0.04/GB/month)
gcloud compute instances stop aurora-vm --zone us-east1-b

# Start it again later (same disk, same data)
gcloud compute instances start aurora-vm --zone us-east1-b
```

**Option B: Destroy/Recreate (cheapest, but local data is lost)**

```bash
# Destroy just the VM (keeps the GCS bucket)
terraform destroy -target=google_compute_instance.aurora_vm

# Recreate later
terraform apply
```

> **What happens to local data on destroy?** When you `terraform destroy` (or destroy
> just the VM), the boot disk and all local data on it are **permanently deleted**. This
> includes any downloaded datasets, model checkpoints, training logs, etc. that are only
> on the VM. **Always sync important files to GCS before destroying.**
>
> **GCS bucket persists independently.** The bucket is a separate resource — it survives
> VM destruction. We configure Terraform with `prevent_destroy` on the bucket so you
> don't accidentally delete it. Cost is negligible ($0.02/GB/month).
>
> **Recommended workflow:**
> 1. `terraform apply` — spin up VM
> 2. `gsutil rsync gs://bucket/data ./data` — pull data from GCS to local disk
> 3. Do your work (training, inference)
> 4. `gsutil rsync ./checkpoints gs://bucket/checkpoints` — push results to GCS
> 5. `terraform destroy` or `gcloud compute instances stop` — shut down
>
> **Cost comparison (A100 80GB):**
> - Running VM: ~$5.50/hr (spot) or ~$14/hr (on-demand)
> - Stopped VM: ~$10/month (256GB disk)
> - Destroyed VM: $0 (only GCS storage costs remain)
> - GCS: ~$0.02/GB/month

---

### Phase 3: Docker Environment

#### 3.1 Dockerfile

A single Dockerfile for both inference and fine-tuning:

- **Base image**: `nvcr.io/nvidia/pytorch:24.01-py3` (CUDA 12.x + PyTorch pre-installed)
- **Install**: `microsoft-aurora` via pip, plus any project dependencies
- **Working directory**: `/app` (project code mounted here at runtime)

The Docker container is run with `--gpus all` to access the A100.

```bash
# Build
docker build -t aurora:latest -f docker/Dockerfile .

# Run inference
docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/app \
    aurora:latest \
    python src/inference.py

# Run fine-tuning
docker run --rm -it \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/app \
    aurora:latest \
    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync python src/finetune.py
```

#### 3.2 Why Docker for both inference and fine-tuning?

- Guarantees identical CUDA/PyTorch/driver versions every time
- No dependency conflicts on the host VM
- Reproducible environment — same results on any machine
- The Deep Learning VM image provides the NVIDIA drivers; Docker provides everything else

---

### Phase 4: Python Project Setup (uv)

#### 4.1 Initialize the project with uv

```bash
uv init
uv python pin 3.10    # Aurora is tested with Python 3.10
```

#### 4.2 Dependencies in `pyproject.toml`

Core dependencies:
- `microsoft-aurora` — the Aurora model
- `torch` — PyTorch (comes in the Docker image, but listed for local dev)
- `google-cloud-storage` — for GCS data transfer
- `numpy`
- `xarray` — for working with atmospheric data (NetCDF)

Dev dependencies:
- `ruff` — linting
- `ipykernel` — for Jupyter notebooks if needed

#### 4.3 Local development

uv can be used locally for lightweight development (data prep scripts, testing with
the small model on CPU), while the Docker container is used on the VM for GPU work.

---

### Phase 5: Aurora Code — Inference

#### 5.1 `src/inference.py` — Test inference with random data

A self-contained script that:
1. Constructs a random `aurora.Batch` (for POC testing)
2. Loads `AuroraSmallPretrained` (for quick testing) or `AuroraPretrained` (for real runs)
3. Runs a forward pass
4. Prints output shape and sample values

```python
from datetime import datetime
import torch
from aurora import AuroraSmallPretrained, Batch, Metadata

model = AuroraSmallPretrained()
model.load_checkpoint()
model.eval()

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

model = model.to("cuda")
with torch.inference_mode():
    pred = model.forward(batch)

print(f"Prediction 2t shape: {pred.surf_vars['2t'].shape}")
print(f"Prediction 2t sample: {pred.surf_vars['2t'][0, 0, :3, :3]}")
```

#### 5.2 Autoregressive rollout

For multi-step predictions:

```python
from aurora import rollout

with torch.inference_mode():
    preds = [p.to("cpu") for p in rollout(model, batch, steps=10)]
```

---

### Phase 6: Aurora Code — Fine-Tuning

#### 6.1 `src/finetune.py` — Basic fine-tuning loop

Based on Aurora's official fine-tuning example:

1. Load `AuroraPretrained` with `autocast=True` (required for fitting gradients in 80GB)
2. Enable `configure_activation_checkpointing()`
3. Set model to `train()` mode
4. Forward pass on batch, compute loss, backward pass
5. Gradient clipping to prevent exploding gradients
6. Save checkpoints to GCS

Key settings:
- `autocast=True` — enables AMP with bfloat16 for the backbone
- `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` — CUDA async memory allocator
- Gradient clipping recommended
- Monitor gradient values for explosion detection

#### 6.2 Multi-GPU (future)

When scaling to multi-GPU, use PyTorch FSDP:
- Change machine type to `a2-ultragpu-2g` (or larger) in Terraform
- Wrap the model with FSDP in `finetune.py`
- No other infrastructure changes needed

---

### Phase 7: Data Utilities

#### 7.1 `src/data.py` — Batch construction utilities

Helper functions to:
- Load static variables from HuggingFace (`aurora-0.25-static.pickle`)
- Convert custom data (NetCDF, numpy, etc.) into `aurora.Batch` format
- Required batch structure:
  - `surf_vars`: dict of `(batch, 2, H, W)` tensors — `2t`, `10u`, `10v`, `msl`
  - `static_vars`: dict of `(H, W)` tensors — `lsm`, `z`, `slt`
  - `atmos_vars`: dict of `(batch, 2, levels, H, W)` tensors — `z`, `u`, `v`, `t`, `q`
  - `metadata`: lat (decreasing), lon (increasing, 0-360), time, atmos_levels

#### 7.2 Key data requirements from Aurora docs

- All variables are **unnormalized** (normalization is internal)
- History dimension `t=2`: index 0 = previous step, index 1 = current step
- Latitudes must be **decreasing** (90 to -90)
- Longitudes must be **increasing** in range **[0, 360)**
- Pressure levels as a **tuple** (not list): `(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)`

---

## Execution Order

### Before GPU quota approval (do all of this now)

| Step | Action | Time Estimate |
|------|--------|---------------|
| 1 | Install Terraform, gcloud, uv locally | 15 min |
| 2 | Create GCP project + enable billing | 10 min |
| 3 | **Request GPU quota for 4x A100 80GB** (do this ASAP) | 5 min to request, 24-48 hrs to approve |
| 4 | Authenticate gcloud + enable APIs | 5 min |
| 5 | Write Terraform configs, Dockerfile, Python code | (we do this together) |
| 6 | `terraform apply` for GCS bucket only (no GPU needed) | 2 min |
| 7 | Test locally: `uv run python src/inference.py --device cpu --small` | 5 min |

### After GPU quota approval

| Step | Action | Time Estimate |
|------|--------|---------------|
| 8 | `terraform apply` to provision GPU VM | 5 min |
| 9 | SSH into VM, build Docker image | 10 min |
| 10 | Run test inference with random data on GPU | 5 min |
| 11 | Run test fine-tuning loop with random data | 5 min |
| 12 | Sync results to GCS, then `terraform destroy` | 5 min |

---

## Important Warnings from Aurora Docs

1. **Sensitivity to data**: Aurora is very sensitive to input data format. Variables must
   be the right ones, at the right pressure levels, from the right source. Regridding
   method matters.

2. **Exploding gradients**: During fine-tuning, monitor gradient values. If they blow up,
   set `stabilise_level_agg=True` in the model constructor (requires `strict=False` for
   checkpoint loading).

3. **Checkpoint loading with modifications**: If you add/remove variables or change the
   model, use `model.load_checkpoint(strict=False)`.

4. **Deterministic output**: For reproducibility, set `torch.use_deterministic_algorithms(True)`
   and `model.eval()`.

5. **Memory**: The full 0.25° model on global data needs ~40GB VRAM for inference. Fine-tuning
   needs an A100 80GB with `autocast=True` and activation checkpointing.

---

## Questions to Resolve Later

- [ ] What custom data format will you be using? (determines `src/data.py` implementation)
- [ ] Do you need to add new variables to Aurora? (requires custom normalisation stats)
- [ ] How many fine-tuning steps / what loss function?
- [ ] Do you want Weights & Biases or TensorBoard for experiment tracking?
