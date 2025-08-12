import torch
import torch.nn as nn
import e3nn.o3 as o3

class RotationEmbed(nn.Module):
    r"""Encode rotation matrices into irreducible-tensor features.

    Parameters
    ----------
    cfg : Dict
        Expected keys
        * "L_max" (int, default = 3) - highest ell to include.
        * "alternating_parity" (bool, default = True) - if False, forces all
          blocks to even parity (+1).  (Parity only matters if you mix this encoder with reflections.)
    """
    def __init__(self, cfg):
        super().__init__()
        L_max = cfg.l_max
        alt_par = cfg.alternating_parity

        # Build irreps list once
        self.L_max  = L_max
        irreps_list = [(1, (ell, (-1)**ell if alt_par else 1))
                       for ell in range(L_max + 1)]
        self.irreps = o3.Irreps(irreps_list)

        # Geometry of block‑diagonal D matrix
        sizes       = torch.tensor([2*ell + 1 for ell in range(L_max + 1)])
        starts      = torch.cat([torch.zeros(1, dtype=torch.long),
                                 sizes.cumsum(0)[:-1]])
        self.register_buffer("_starts", starts, persistent=False)  # for TorchScript
        self._row_slices = [slice(int(s), int(s + k))
                            for s, k in zip(starts, sizes)]
        
        self.irreps_output = self.irreps

    # -----------------------------------------------------------------
    def forward(self, R: torch.Tensor) -> torch.Tensor:
        r"""Encode rotation(s).

        Parameters
        ----------
        R : (..., 3, 3) tensor of proper rotations.

        Returns
        -------
        feat : (..., \sum_{\ell=0}^{L_max}(2\ell+1)) tensor - stacked first columns.
        """
        D = self.irreps.D_from_matrix(R)              # type: ignore # (..., N, N) with N = \sum(2\ell+1)
        cols = [D[..., sl, self._starts[i]]           # first column of each block
                for i, sl in enumerate(self._row_slices)]
        return torch.cat(cols, dim=-1)