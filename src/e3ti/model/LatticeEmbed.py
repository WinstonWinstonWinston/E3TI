import torch
from e3nn import io

#TODO: Maybe add parameters in this? For now this seems fine.
class LatticeEmbed(torch.nn.Module):
    def __init__(self, latcfg) -> None:
        super().__init__()
        self.irreps_output = io.CartesianTensor('ij=ij')

    def forward(self, ell):
        return self.irreps_output.from_cartesian(ell) # type: ignore