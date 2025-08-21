import json
import os
import sys
# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# print(project_root)
sys.path.insert(0, project_root)
import numpy as np
from tqdm import tqdm
import hydra
from hydra.utils import instantiate
from hydra.utils import get_class
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from utils import batch_to_numpy
# from ito import data, utils
# from ito.model import ddpm

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(f"[Hydra] cfg keys: {list(cfg.keys())}")
    print(f"[Hydra] Working directory: {os.getcwd()}")
    print(type(cfg))
    print(OmegaConf.to_yaml(cfg))
    
    sample_cfg = cfg.sample
    ## Set up paths where sampled trajectories are stored
    samples_dir = os.path.join(sample_cfg.root, sample_cfg.checkpoint_dir)
    samples_path = os.path.join(samples_dir, sample_cfg.traj_name)

    ## Get an initial batch to send into model.sample
    dataset = instantiate(cfg.data, _convert_="all")
    loader = DataLoader(
                        dataset,
                        batch_size=int(sample_cfg.samples),
                        shuffle=True,
                        drop_last=True,
                    )
    batch_dict = next(iter(loader))             # {'batch_0': DataBatch(...), 'batch_t': DataBatch(...)}
    batch = batch_dict["batch_0"]               # use t=0 as the sampler's initial condition
    trajectory = [batch_to_numpy(batch)]
    
    ## Load the trained model
    model = instantiate(sample_cfg.model_loader, _convert_="all")

    ## Begin sampling
    for _ in tqdm(range(sample_cfg.traj_length)):
        batch = model.sample(batch, ode_steps=sample_cfg.ode_steps)
        trajectory.append(batch_to_numpy(batch))
   
    trajectory = np.stack(trajectory, axis=1)

    os.makedirs(samples_dir, exist_ok=True)
    np.save(samples_path, trajectory)
    # Destination path for the symbolic link
#     latest_symlink = os.path.join(samples_root, "latest")
#     # Check if the symbolic link or file exists and remove it
#     if os.path.islink(latest_symlink) or os.path.exists(latest_symlink):
#         os.remove(latest_symlink)
#     os.symlink(src=os.path.abspath(samples_path), dst=latest_symlink)
    print(f"samples saved at {samples_path}")


if __name__ == "__main__":
    main()
