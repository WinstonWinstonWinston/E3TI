from omegaconf import DictConfig
from abc import ABC, abstractmethod

class Experiment(ABC):
    """
    Abstract base experiment class. 
    
    All experiments must
    1: Save their configuration
    2: Contain a "run"
    3: Be able to give a descriptive print statement describing the config
    
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def run(self):
        raise NotImplementedError
    
    @abstractmethod
    def summarize_cfg(self):
        raise NotImplementedError