# CIS-7000 Final Project — Aurora Atmospheric Model

Inference and fine-tuning of Microsoft's [Aurora 0.25° Pretrained](https://microsoft.github.io/aurora/) foundation model for atmospheric prediction, running on the [NRP Nautilus](https://nrp.ai/) Kubernetes cluster.

## Project Structure

```
docker/Dockerfile              # Docker image: NVIDIA PyTorch + Aurora (deps from pyproject.toml)
k8s/                           # Kubernetes manifests for Nautilus
  aurora-test-cpu-job.yaml     #   Batch: CPU test (small model, no GPU)
  aurora-test-job.yaml         #   Batch: GPU test (small model, A100 80GB)
  aurora-inference-job.yaml    #   Batch: inference (A100 80GB)
  aurora-finetune-job.yaml     #   Batch: fine-tuning (A100 80GB)
  aurora-interactive-cpu-pod.yaml  # Interactive: CPU dev pod
  aurora-interactive-pod.yaml      # Interactive: GPU dev pod (A100 80GB)
scripts/                       # Shell scripts
  docker-build-push.sh         #   Build & push Docker image to Docker Hub
  k8s-setup-github-secret.sh   #   One-time: store GitHub credentials on Nautilus
  k8s-test-cpu.sh              #   Run CPU test job
  k8s-test-gpu.sh              #   Run GPU test job
  k8s-inference.sh             #   Run inference job
  k8s-finetune.sh              #   Run fine-tuning job
  k8s-interactive-cpu.sh       #   Start CPU interactive session
  k8s-interactive-gpu.sh       #   Start GPU interactive session
  test_aurora_cpu.py            #   Standalone CPU test script
src/                           # Python source code
  data.py                      #   Batch construction utilities
  inference.py                 #   Inference CLI
  finetune.py                  #   Fine-tuning CLI
notebooks/                     # Jupyter notebooks
pyproject.toml                 # Dependencies (used by both uv locally and Docker)
```

## Prerequisites

You need the following tools installed on your local machine:

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager (for local development)
- [Docker](https://docs.docker.com/get-docker/) — container runtime (for building the image)
- A [Docker Hub](https://hub.docker.com/) account (free, for hosting the image)
- [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) — Kubernetes CLI
- `curl`, `jq`, `unzip` — standard CLI tools (for installing kubelogin)

## Setup

### Step 1: Get access to Nautilus

Instructions from Email, already completed.

### Step 2: Install kubectl

Follow the [official kubectl install instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl/) for your OS.


### Step 3: Install kubelogin plugin

This is **required** — your kubeconfig will not work without it.

On Linux (Ubuntu/Debian):

```bash
OS_ARCHITECTURE="$(dpkg --print-architecture)"
OS_NAME="linux"

KUBELOGIN_VERSION="$(curl -fsSL "https://api.github.com/repos/int128/kubelogin/releases/latest" | jq -r '.tag_name')"
curl -o kubelogin.zip -fSL "https://github.com/int128/kubelogin/releases/download/${KUBELOGIN_VERSION}/kubelogin_${OS_NAME}_${OS_ARCHITECTURE}.zip"
unzip kubelogin.zip kubelogin
chmod +x ./kubelogin
sudo mv ./kubelogin /usr/local/bin/kubectl-oidc_login
sudo chown root: /usr/local/bin/kubectl-oidc_login
rm -f kubelogin.zip
```

On macOS, change `OS_NAME="darwin"` and use `uname -m` for architecture. See the [kubelogin repo](https://github.com/int128/kubelogin) for other install methods.

### Step 4: Download the Nautilus kubeconfig

```bash
mkdir -p ~/.kube
curl -o ~/.kube/config -fSL "https://nrp.ai/config"
```

#### WSL-specific fixes

If you're on Windows WSL, edit `~/.kube/config` and add these args to the `users.user.exec.args` section to fix browser and port issues:

```yaml
args:
  - oidc-login
  - get-token
  - --oidc-issuer-url=https://authentik.nrp-nautilus.io/application/o/k8s/
  - --oidc-client-id=xrxBIaWxeRmGJUwSvaLjUzMEFZzQu2b4nk9I0B2W
  - --oidc-extra-scope=profile,offline_access
  - --browser-command="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"
  - --listen-address=0.0.0.0:18000
  - --token-cache-storage=disk
```

The `--browser-command` points to your Windows Chrome so it can open for authentication. The `--listen-address` uses port 18000 to avoid conflicts with the default port 8000.

### Step 5: Set namespace and verify access

```bash
# Set the default namespace
kubectl config set-context nautilus --namespace=upenn-dyer-lab

# Verify access (will open browser for CILogon authentication)
kubectl get pods
```

If you see "No resources found in upenn-dyer-lab namespace", that's correct — it means you have access but no pods are running yet.

To check your context is configured correctly:

```bash
kubectl config get-contexts
# Should show:
# CURRENT   NAME       CLUSTER    AUTHINFO   NAMESPACE
# *         nautilus   nautilus   oidc       upenn-dyer-lab
```

### Step 6: Create the GitHub secret (one-time, per user)

Since this is a private repo, each user stores their GitHub credentials on Nautilus so pods can clone it.

1. Create a [GitHub Personal Access Token](https://github.com/settings/tokens) with `repo` scope
2. Run the setup script:

```bash
bash scripts/k8s-setup-github-secret.sh
```

It will prompt for your GitHub username and token.

### Step 7: Set up CDS API key (for ERA5 data)

To download ERA5 weather data, you need a Climate Data Store account:

1. Register at the [Climate Data Store](https://cds.climate.copernicus.eu/)
2. Accept the [ERA5 terms of use](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
3. Copy your API key from your [CDS account page](https://cds.climate.copernicus.eu/profile) and save it locally:

```bash
cat > ~/.cdsapirc << 'EOF'
url: https://cds.climate.copernicus.eu/api
key: <your API key>
EOF
```

For Nautilus, the CDS key is stored as a Kubernetes secret so all pods have access automatically:

```bash
kubectl create secret generic cds-api-key \
    --from-literal=url="https://cds.climate.copernicus.eu/api" \
    --from-literal=key="<your API key>"
```

### Step 8: Build and push the Docker image

This builds an image with all dependencies pre-installed (from `pyproject.toml`) and pushes it to Docker Hub. Only one person needs to do this initially, or when dependencies change.

```bash
docker login
bash scripts/docker-build-push.sh
```

The image is pushed to `puschb/aurora-dev:latest` on Docker Hub (public).

### Step 9: Set up local development environment

```bash
uv sync
```

This creates a `.venv` virtual environment in the project root and installs all dependencies from `pyproject.toml` into it.

## Usage

### Local development (CPU, no cluster needed)

You can run code locally in two ways:

```bash
# Option A: use uv run (no activation needed)
uv run python scripts/test_aurora_cpu.py

# Option B: activate the venv directly, then use python as normal
source .venv/bin/activate
python scripts/test_aurora_cpu.py
```

Example commands (shown with `uv run`; drop the prefix if the venv is activated):

```bash
# Run the CPU test script (small model, random data)
uv run python scripts/test_aurora_cpu.py

# Run inference with the small model on CPU
uv run python -m src.inference --small --device cpu

# Run a quick fine-tuning test on CPU (slow but works)
uv run python -m src.finetune --steps 2
```

### Nautilus batch jobs

These submit a job, stream logs to your terminal, and clean up when done:

```bash
bash scripts/k8s-test-cpu.sh       # CPU test — small model, no GPU
bash scripts/k8s-test-gpu.sh       # GPU test — small model on A100 80GB
bash scripts/k8s-inference.sh      # Run inference on A100 80GB
bash scripts/k8s-finetune.sh       # Run fine-tuning on A100 80GB
```

To manually manage jobs:

```bash
kubectl create -f k8s/aurora-test-cpu-job.yaml    # submit
kubectl get pods                                   # check status
kubectl logs -f <pod-name>                         # stream logs
kubectl delete job <job-name>                      # clean up
```

### Nautilus interactive sessions

These start a pod with the repo pre-cloned and aurora installed, give you a shell, and clean up when you exit:

```bash
bash scripts/k8s-interactive-cpu.sh   # CPU pod (no GPU)
bash scripts/k8s-interactive-gpu.sh   # GPU pod (A100 80GB)
```

Inside the pod, the working directory is `/opt/repo` with the full project:

```bash
python scripts/test_aurora_cpu.py
python -m src.inference --small
python -m src.finetune --steps 5
```

To pull the latest code changes inside a running pod:

```bash
cd /opt/repo && git pull
```

To manually manage interactive pods:

```bash
kubectl create -f k8s/aurora-interactive-cpu-pod.yaml
kubectl exec -it aurora-dev-cpu -- /bin/bash -c 'cd /opt/repo && bash'
kubectl delete pod aurora-dev-cpu
```

### VS Code on Nautilus

You can attach VS Code to a running interactive pod for a full IDE experience:

1. Install the [Kubernetes](https://marketplace.visualstudio.com/items?itemName=ms-kubernetes-tools.vscode-kubernetes-tools) and [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) VS Code extensions
2. Start an interactive pod (`bash scripts/k8s-interactive-cpu.sh` in a separate terminal — don't exit)
3. In VS Code, open the Kubernetes sidebar (ship wheel icon)
4. Navigate to: Nautilus > Workloads > Pods > your pod name
5. Right-click the pod > **Attach Visual Studio Code**

Requirements:
- `~/.kube/config` must be at the default path (not a custom `--kubeconfig` path)
- Both `kubectl` and `kubectl-oidc_login` must be in your `PATH`

## Development Workflow

```
Edit locally  -->  git push  -->  Nautilus pod: git pull  -->  Run
```

1. Edit code locally in your IDE
2. Test quick things locally: `uv run python scripts/test_aurora_cpu.py`
3. Push to GitHub: `git push`
4. On Nautilus: start an interactive session or submit a batch job (both clone the latest code automatically)
5. In an already-running interactive session, use `git pull` to get updates without restarting the pod

### Adding dependencies

1. Add the package to `pyproject.toml` under `dependencies`
2. Locally: `uv sync`
3. Rebuild the Docker image: `bash scripts/docker-build-push.sh`

### Token expiry

The Nautilus access token expires after 30 minutes and is automatically refreshed. If you need to force-refresh (e.g., after being added to a new namespace):

```bash
kubectl oidc-login clean
kubectl get pods    # triggers new token
```

## Aurora Model Reference

- [Aurora Documentation](https://microsoft.github.io/aurora/)
- [Aurora GitHub](https://github.com/microsoft/aurora)
- [HuggingFace Checkpoints](https://huggingface.co/microsoft/aurora)

Key facts:
- **Model**: Aurora 0.25° Pretrained — general-purpose atmospheric foundation model
- **Inference**: ~40 GB GPU VRAM for full 0.25° global data (721x1440 grid)
- **Fine-tuning**: Requires A100 80GB with `autocast=True` and activation checkpointing
- **Small model**: `AuroraSmallPretrained` — for debugging, works on CPU or any GPU
- **Batch format**: `aurora.Batch` with surface vars, static vars, atmospheric vars, and metadata
- **Variables**: `2t`, `10u`, `10v`, `msl` (surface); `lsm`, `z`, `slt` (static); `z`, `u`, `v`, `t`, `q` (atmospheric)
- **Pressure levels**: (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000) hPa

## Nautilus Resources

- [NRP Nautilus Portal](https://nrp.ai/)
- [Nautilus Documentation](https://nrp.ai/documentation/)
- [Getting Started Guide](https://nrp.ai/documentation/userdocs/start/getting-started/)
- [Cluster Policies](https://nrp.ai/documentation/userdocs/start/policies/)
- [kubectl Cheatsheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
