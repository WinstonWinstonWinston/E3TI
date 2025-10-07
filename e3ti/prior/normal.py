from typing import Any
from torch import Tensor
from torch_geometric.data import Data
from e3ti.prior.abstract import E3TIPrior
import torch
import math

class NormalPrior(E3TIPrior):
    """
    Abstract interface for prior samplers used in E3TI workflows.

    :param cfg:
        Configuration object (dataclass/dict/omegaconf) holding prior hyperparameters.
    :type cfg: Any
    """

    def __init__(self, mean:float, std: float) -> None:
        super().__init__()
        self.std = std
        self.mean = mean

    def sample(self, batch: Data, stratified: bool = False) -> Data:
        """
        Draw samples from the prior and return the same batch object type with updated keys.

        :param batch:
            A torch batch of geometric data objects coming from a data loader.
        :type batch: torch_geometric.data.Data
        :param stratified:
            Whether to use stratified sampling over the time variable
        :type stratified: bool

        :return:
            The input batch type with modified keys.
        :rtype: torch_geometric.data.Data
        """
        x_base = torch.randn_like(batch['x'])*self.std + self.mean
        B = max(batch['batch']) + 1 # shift by 1, zero indexing
        device = batch['x'].device
        t_interpolant = torch.rand(B, device=device) if not stratified else torch.cat([(i + torch.rand(B//4 + (i < B%4), device=device))/4 for i in range(4)])

        batch['x_base'] = x_base
        batch['t_interpolant'] = t_interpolant
        return batch

    def log_prob(self, batch: Data) -> Tensor:
        """
        Compute log-probabilities under the prior for variables referenced by batch.

        :param batch:
            A torch batch of geometric data objects coming from a data loader.
        :type batch: torch_geometric.data.Data

        :return:
            Per-example log p(x) with a leading batch dimension.
        :rtype: torch.Tensor
        """
        x = batch["x_base"] # (B*N, 3)
        B = batch.num_graphs
        z = (x - self.mean) / self.std
        z = z.view(B, -1, 3) # requires contiguous nodes per graph
        B, N, D = z.shape
        return -0.5*torch.sum(z*z, dim=(1,2)) - 0.5*N*D*math.log(2*math.pi) - N*D*math.log(float(self.std))

    def summarize_cfg(self) -> None:
        print(f"[{self.__class__.__name__}] mean={self.mean}, std={self.std}")
