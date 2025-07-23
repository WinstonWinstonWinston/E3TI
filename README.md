This repository separates **models**, **experiments**, **data**, and **configs** so you can swap architectures, datasets, and hyper‑parameters with minimal friction.

---

## 📁 Repository structure

```text
.
├── models/          # PyTorch modules, losses, utilities
│   └── __init__.py
├── experiments/     # Entry‑points: train.py, inference.py, evaluate.py, etc.
│   ├── train.py
│   ├── inference.py
│   └── evaluate.py
├── data/            # Raw or processed datasets (git‑ignored by default)
│   ├── README.md
│   └── ...
└── configs/         # Hydra configuration files (YAML)
    ├── config.yaml  # master config (defaults list)
    ├── model/       # architecture‑specific configs
    ├── data/        # dataset‑specific configs
    └── experiment/  # training, inference, evaluation settings
```

### Folder breakdown

| Folder           | Purpose                                                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **models/**      | All PyTorch `nn.Module`s and sub‑modules. Keep each architecture in its own file for clarity.                          |
| **experiments/** | Scripts that *do* things—training, inference, evaluation, hyper‑sweeps. Each should be Hydra‑aware.                    |
| **data/**        | Storage for datasets or links to external sources. Add a `README.md` explaining how to download or pre‑process.        |
| **configs/**     | A composable config tree for Hydra. Organise by domain (`model/`, `data/`, `experiment/`) so overrides stay intuitive. |

---

## 🚀 Quick start

Please inspect `setup_umn.sh` before running to ensure it is correct for your machine (paths, modules, etc).

```bash
# 1. Clone and install
git clone https://github.com/SAMPEL-Group/E3-Tensor-Interpolants.git
cd E3-Tensor-Interpolants
chmod +x setup_umn.sh
./setup_umn.sh
```

# 2. Load modules 
```bash
module load gcc/8.2.0
module load ompi/3.1.6/gnu-8.2.0
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2
```

# 3. Train (override any Hydra key on the CLI)
```python
python experiments/train.py \
  model=resnet18 \
  data=cifar10 \
  experiment=baseline \
  trainer.max_epochs=100 \
  optimizer.lr=1e-3
```

# 4. Inference
```python
python experiments/inference.py \
  checkpoint=outputs/2025-07-21/ckpt_best.pt \
  data=test_set
```

# 5. Evaluation
```python
python experiments/evaluate.py \
  checkpoint=outputs/2025-07-21/ckpt_best.pt \
  metrics=accuracy,f1
```

Hydra creates a timestamped sub‑folder inside `outputs/` for every run, logging configs, stdout, and model checkpoints for perfect reproducibility.

---

## 🛠️ Customising configs

1. **Create a new YAML** (e.g. `configs/model/wideresnet.yaml`).
2. Add it to the `defaults:` list in `configs/config.yaml` or pass `model=wideresnet` on the CLI.
3. Profit—no code changes required.

---