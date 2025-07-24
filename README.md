This repository separates **models**, **experiments**, **data**, and **configs** so you can swap architectures, datasets, and hyper‑parameters with minimal friction.

---

### Folder breakdown

| Folder           | Purpose                                                                                                                |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **model/**      | All PyTorch `nn.Module`s and sub‑modules. Keep each architecture in its own file for clarity.                          |
| **experiment/** | Scripts that *do* things—training, inference, evaluation, hyper‑sweeps. Each should be Hydra‑aware.                    |
| **data/**        | Storage for datasets or links to external sources. Add a `README.md` explaining how to download or pre‑process.        |
| **configs/**     | A composable config tree for Hydra. Organise by domain (`model/`, `data/`, `experiment/`) so overrides stay intuitive. |

---

## 🚀 Quick start

Please inspect `setup_umn.sh` before running to ensure it is correct for your machine (paths, modules, etc).

# 1. Load modules 
```bash
module purge
module load gcc/8.2.0
module load ompi/3.1.6/gnu-8.2.0
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2
```

# 2. Clone and install
```bash
git clone https://github.com/SAMPEL-Group/E3-Tensor-Interpolants.git
cd E3-Tensor-Interpolants
chmod +x setup_umn.sh
./setup_umn.sh
micromamba activate e3ti
```

# 3. Train
To run training use
```python
python experiments/train.py
```
Consider modifying `configs.experiment.train.yaml` before executing, not all configurations work on all machines.  

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