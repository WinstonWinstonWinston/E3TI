from abc import ABC, abstractmethod

class E3TIPrior(ABC):

    def __init__(self,cfg):
        self.cfg = cfg
        pass

    @abstractmethod
    def summarize_cfg(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self,batch,stratified):
        raise NotImplementedError
    
    @abstractmethod
    def log_prob(self,batch):
        raise NotImplementedError
