from abc import ABC, abstractmethod
import torch

class E3TIModel(torch.nn.Module, ABC):

    def __init__(self,**kwargs):
        super().__init__(self)

    @abstractmethod
    def forward(batch):
        """
        TODO: Finish return param typing here
        Implements a forward pass through the model.

        :param batch:
            A torch batch of geometric data objects which come from a data loader. 
        :type batch: torch_geometric.data.Data

        :return:
            A new batch object with modified keys containing velocity, score, denoised point etc.
        :rtype: torch_geometric.data.Data??
        """
        raise NotImplementedError
    
    def summarize_cfg(self):
        """
        Prints details about the model. 

        TODO: Modify this to print self.cfg details instead?
        """
        print(f"[{self.__class__.__name__}]")
        print("="*50)
        print("Model Parameter Count Breakdown")
        print("="*50)

        child_total = 0
        for name, submodule in self.named_children():
            sub_params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
            print(f"Submodule '{name}': {sub_params:,} parameters")
            child_total += sub_params

        grand_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if grand_total != child_total:
            print(f"Unassigned / top-level: {grand_total - child_total:,} parameters")

        print("-"*50)
        print(f"Total Trainable Parameters: {grand_total:,}")
        print("="*50)
        print()
        print()
        print()

