# management 
import hydra
import logging
import os
import GPUtil
from omegaconf import DictConfig, OmegaConf
import wandb

# lightning
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# e3ti
from e3ti.model.module import E3tiModule
from e3ti.data.dataset import MNISTDataModule
from e3ti.utils import flatten_dict, set_seed

logger = logging.getLogger(__name__)
logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
for level in logging_levels:
    setattr(logger, level, rank_zero_only(getattr(logger, level)))


class Experiment:

    def __init__(self, cfg: DictConfig):
        # Split configuration up
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._train_cfg = cfg.exp_train
        self._model_cfg = cfg.model

        # Grab dataset
        self._datamodule: LightningDataModule = MNISTDataModule(data_cfg=self._data_cfg, data_dir=cfg.paths.data_dir)

        # Determine available gpus
        self._train_device_ids = GPUtil.getAvailable(order='memory', limit = 8)[:self._cfg.num_device]
        logger.info(f"Training with devices: {self._train_device_ids}")

        # Set up lightning module
        self._module: LightningModule = E3tiModule(self._model_cfg, self._train_cfg.optim)

        # Set seed (if provided)
        if self._cfg.seed is not None:
            logger.info(f'Setting seed to {self._cfg.seed}')
            set_seed(self._cfg.seed)
        

    def train(self):
        callbacks = []
       
        # Setup lightning wandb connection
        wbLogger = WandbLogger(
            **self._train_cfg.wandb,
        )
        
        wbLogger.watch(
            self._module,
            log=self._train_cfg.wandb_watch.log,
            log_freq=self._train_cfg.wandb_watch.log_freq
        )

        # Checkpoint directory.
        ckpt_dir = self._train_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"Checkpoints saved to {ckpt_dir}")
        
        # Model checkpoints
        callbacks.append(ModelCheckpoint(**self._train_cfg.checkpointer))

        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        # Save config only for main process.
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(flatten_dict(cfg_dict))
            if isinstance(wbLogger.experiment.config, wandb.sdk.wandb_config.Config):
                wbLogger.experiment.config.update(flat_cfg, allow_val_change=True)
        
        trainer = Trainer(
            **self._train_cfg.trainer,
            callbacks=callbacks,
            logger=wbLogger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
        )

        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._train_cfg.warm_start
        )

@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.exp_train.warm_start is not None and cfg.expeexp_trainriment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.exp_train.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        logger.info(f'Loaded warm start config from {warm_start_cfg_path}')

    exp = Experiment(cfg)
    exp.train()

if __name__ == "__main__":
    main()