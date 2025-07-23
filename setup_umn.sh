#!/usr/bin/env bash
# setup_umn.sh
set -euo pipefail

MAMBA="$HOME/micromamba/micromamba"
ENV_NAME="e3ti"
PY_VER="3.11"

# init micromamba for this shell
eval "$($MAMBA shell hook --shell bash)"

echo ">>> Creating / activating env: ${ENV_NAME}"
if ! micromamba env list | grep -q "^${ENV_NAME}\b"; then
    micromamba create -y -n "${ENV_NAME}" -c conda-forge python="${PY_VER}"
fi
micromamba activate "${ENV_NAME}"

echo ">>> Micromamba packages..."
micromamba install -y -c conda-forge \
  numpy matplotlib scipy omegaconf hydra-core pytorch-lightning \
  parmed wandb tqdm mdtraj

echo ">>> Pip packages (Torch + PyG stack + e3nn)..."
python -m pip install --upgrade pip build wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
pip install --upgrade e3nn

echo ">>> Build & install your project (dev mode)..."
python -m build --wheel   # if you're in the repo you want to build
pip install -e .          # or: pip install dist/*.whl

echo ">>> Done. Activate later with:  micromamba activate ${ENV_NAME}"
