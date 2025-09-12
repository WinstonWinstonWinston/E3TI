# management 
import hydra
import logging
import os
import GPUtil
from omegaconf import DictConfig, OmegaConf
import wandb

# lightning
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.rank_zero import rank_zero_only # type: ignore
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# mcf
from e3ti.module import E3TIModule
from e3ti.utils import flatten_dict, set_seed
from e3ti.experiment.abstract import Experiment

logger = logging.getLogger(__name__)
logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
for level in logging_levels:
    setattr(logger, level, rank_zero_only(getattr(logger, level)))

class Train(Experiment):

    def __init__(self, cfg: DictConfig) -> None:
        # Split configuration up

        super().__init__(cfg)

        self.data_cfg = cfg.data
        self.train_cfg = cfg.exp_train
        self.model_cfg = cfg.model

        # Grab datasets
        cfg.data_cfg.split = "train"
        self.train_dataset = hydra.utils.instantiate(self.data_cfg.dataset)
        cfg.data_cfg.split = "test"
        self.test_dataset = hydra.utils.instantiate(self.data_cfg.dataset)
        cfg.data_cfg.split = "valid"
        self.valid_dataset = hydra.utils.instantiate(self.data_cfg.dataset)
    
        self.datamodule =  hydra.utils.instantiate(self.data_cfg.loader,
                                                                        train_dataset = self.train_dataset, 
                                                                        valid_dataset = self.valid_dataset, 
                                                                        test_dataset  = self.test_dataset)

        # Determine available gpus
        self.train_device_ids = GPUtil.getAvailable(order='memory', limit = 8)[:self.cfg.num_device]
        logger.info(f"Training with devices: {self.train_device_ids}")

        # Set up lightning module
        self.module = E3TIModule(self.model_cfg)

        # Set seed (if provided)
        if self.cfg.seed is not None:
            logger.info(f'Setting seed to {self.cfg.seed}')
            set_seed(self.cfg.seed)

    def run(self):
        callbacks = []
       
        # Setup lightning wandb connection
        wbLogger = WandbLogger(
            **self.train_cfg.wandb,
        )
        
        wbLogger.watch(
            self.module,
            log=self.train_cfg.wandb_watch.log,
            log_freq=self.train_cfg.wandb_watch.log_freq
        )

        # Checkpoint directory.
        ckpt_dir = self.train_cfg.checkpointer.dirpath
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.info(f"Checkpoints saved to {ckpt_dir}")
        
        # Model checkpoints
        callbacks.append(ModelCheckpoint(**self.train_cfg.checkpointer))

        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        
        # Save config only for main process.
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self.cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            flat_cfg = dict(flatten_dict(cfg_dict))
            if isinstance(wbLogger.experiment.config, wandb.sdk.wandb_config.Config): # type: ignore
                wbLogger.experiment.config.update(flat_cfg, allow_val_change=True)
        
        trainer = Trainer(
            **self.train_cfg.trainer,
            callbacks=callbacks,
            logger=wbLogger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self.train_device_ids,
            # detect_anomaly=True
        )

        trainer.fit(
            model=self.module,
            datamodule=self.datamodule,
            ckpt_path=self.train_cfg.warm_start
        )

    def summarize_cfg(self):

        self.train_dataset.summaraize_cfg()
        self.test_dataset.summaraize_cfg()
        self.valid_dataset.summaraize_cfg()

        self.datamodule.summarize_cfg()
        
        self.module.summarize_cfg()

