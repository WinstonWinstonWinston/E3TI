

from omegaconf import DictConfig
from abc import ABC, abstractmethod
import hydra

class Experiment(ABC):
    """
    Abstract base experiment class. 
    
    All experiments must
    1: Save their configuration
    2: Contain a "run"
    3: Be able to give a descriptive print statement describing the config
    

    TODO: Comment me
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.experiments = []
        for element in cfg.elements:
            self.experiments.append(hydra.utils.instantiate(element))

    def run(self):
        
        for experiment in self.experiments:
            experiment.run()
        raise NotImplementedError
    
    def summarize_cfg(self):
        raise NotImplementedError