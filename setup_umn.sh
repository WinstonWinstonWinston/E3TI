#!/usr/bin/env bash
# setup_umn.sh
# -------------------------------------------------
# Creates a Conda environment called `e3ti`,
# installs packages like torch, numpy, matplotlib, etc.
# builds the e3ti wheel, and
# installs that wheel with pip.
# -------------------------------------------------

set -e

module purge
module load gcc/8.2.0
module load ompi/3.1.6/gnu-8.2.0
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2

ENV_NAME="e3ti"
PY_VER="3.11"

echo ">>> Creating / activating Conda env: ${ENV_NAME}"
$HOME/micromamba/micromamba create -n "${ENV_NAME}" python="${PY_VER}" -y
$HOME/micromamba/micromamba activate "${ENV_NAME}"

echo ">>> Upgrading pip and adding build helpers..."
python -m pip install --upgrade pip
python -m pip install build wheel
pip install numpy
pip install matplotlib
pip install scipy
pip install omegaconf
pip install hydra-core
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install torch-geometric
pip install pytorch_lightning
pip install parmed
pip install wandb
pip install tqdm
pip install e3nn
pip install mdtraj

echo ">>> Building kim_convergence distribution..."
# Creates dist/<name>-<version>-py3-none-any.whl (and a source tarball)
python -m build --wheel

echo ">>> Installing e3ti from the freshly built wheel..."
pip install -e .          # dev mode – live edits
# pip install dist/*.whl  # frozen wheel – production test

echo ">>> Environment ${ENV_NAME} is ready."
echo "    Activate later with:  conda activate ${ENV_NAME}"