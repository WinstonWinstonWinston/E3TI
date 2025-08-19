import os
import sys
# Add src/ito (project root) to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
import json
import wandb
import glob
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split
from utils import get_timestamp, get_latest_timestamp_directory
from data.dataloader import get_dataloaders

from hydra.utils import get_class
import inspect

@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(f"[Hydra] cfg keys: {list(cfg.keys())}")
    print(f"[Hydra] Working directory: {os.getcwd()}")
    print(type(cfg))
    print(OmegaConf.to_yaml(cfg))
    train_cfg = cfg.experiments

    wandb.login()
    run = wandb.init(
        project="ito-hydra",
        config={
            "learning_rate": train_cfg.lr,
            "epochs": train_cfg.epochs,
        },
    )
    timestamp = get_timestamp()

    if train_cfg.checkpoint:
        print(f"Using checkpoint file: {train_cfg.checkpoint}")
        train_dir = get_latest_timestamp_directory(train_cfg.root)
        checkpoint_file_link = train_cfg.checkpoint
        if train_dir:
            print(f"Using newest existing directory: {train_dir}")
        else:
            print("No existing timestamp directory found.")
            return
    else:
        train_dir = os.path.join(train_cfg.root, "train", timestamp)

    checkpoint_dir = os.path.join(train_dir, "checkpoints")
    train_dir_link = os.path.join(train_cfg.root, "train", "latest")
    best_checkpoint_link = os.path.join(train_dir, "best")

    print(f"saving checkpoints to {checkpoint_dir}")
    os.makedirs(train_dir, exist_ok=True)

    with open(os.path.join(train_dir, "args.json"), "w") as f:
        json.dump(OmegaConf.to_container(train_cfg, resolve=True), f, indent=4)

    ## Instantiate the TLDDPM model. model.yaml (default:ito.yaml) file has configurable options for DDPMs, score model used and their params
    model = instantiate(cfg.model, _convert_="all")
 
    ## Get the dataloaders. dataloader.yaml has dataloader configurable options. data.yaml has the dataset information, molecules to be used for training etc. get_dataloaders function has the dataset loading inside it.
    train_loader, val_loader = get_dataloaders(cfg)
    data_iter = iter(train_loader)
    batch = next(data_iter)
    print('Dataloader batch iter:', batch)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1, dirpath=checkpoint_dir, filename="{epoch}"
    )
    print('Max Epochs: {}'.format(train_cfg.epochs))
    trainer = pl.Trainer(
        max_epochs=train_cfg.epochs,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
    )

    if train_cfg.checkpoint:
        print(f"Using checkpoint file: {train_cfg.checkpoint}")
        trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint_file_link)
    else:
        print('No checkpoint file')
        trainer.fit(model, train_loader, val_loader)
        
    os.symlink(
        src=os.path.abspath(checkpoint_callback.best_model_path),
        dst=best_checkpoint_link,
    )
    if os.path.exists(train_dir_link):
        os.unlink(train_dir_link)
    os.symlink(src=os.path.abspath(train_dir), dst=train_dir_link)
    print(f"best model checkpoint: {best_checkpoint_link}")

if __name__ == "__main__":
    main()
