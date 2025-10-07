import torch
from e3ti.model.embedding.time import TimeEmbed
from torch import nn
from e3nn.o3 import Irreps
from torch_geometric.data import Data
from e3ti.utils import parse_activation

class EquilibriumEmbedder(nn.Module):
    def __init__(self, use_ff, interp_time, force_field,atom_type):
        super().__init__()
        self.use_ff = use_ff

        self.interpolant_time_embedder = TimeEmbed(interp_time)

        if self.use_ff:
            self.ff_embedder = MLPWithBN(
                in_dim=force_field.in_dim,
                hidden_dims=force_field.hidden_dims,
                out_dim=force_field.out_dim,
                activation=force_field.activation,
                use_input_bn=force_field.use_input_bn,
                affine=force_field.affine,
                track_running_stats=force_field.track_running_stats,
            )

        self.atom_type_embed = nn.Embedding(
            num_embeddings=atom_type.num_types,
            embedding_dim=atom_type.embedding_dim,
        )

    def forward(self, batch : Data) -> Data:
        # Expect: atom_type (B,N), interp_time (B,), and if use_ff: charge/mass/sigma/epsilon (B,N)
        atom_ty  = batch["atom_type"].long()
        t        = batch["t_interpolant"].float()
        
        atom_emb = self.atom_type_embed(atom_ty)               # (B*N, D_atom)

        t_emb = self.interpolant_time_embedder(t)[batch.batch] # (B*N, D_t)

        ff_emb = None
        parts = [atom_emb, t_emb]

        if self.use_ff:
            charge  = batch["charge"].float()
            mass    = batch["mass"].float()
            sigma   = batch["sigma"].float()
            epsilon = batch["epsilon"].float()
            ff_in = torch.stack([charge, mass, sigma, epsilon], dim=-1)  # (B*N, 4)
            ff_emb = self.ff_embedder(ff_in)                             # (B*N, D_ff)
            parts.append(ff_emb)

        f = torch.cat(parts, dim=-1)  # (B, N, D_atom + D_t [+ D_ff])

        batch['f'] = f
        batch['f_irrep'] =Irreps(str(len(f[-1]))+"x0e") 

        return batch


class MLPWithBN(nn.Module):
    def __init__(self, in_dim, hidden_dims=(128, 128), out_dim=1, activation='relu',
                 use_input_bn=True, affine=True, track_running_stats=True):
        super().__init__()
        layers = []
        if use_input_bn:
            layers += [nn.BatchNorm1d(in_dim, affine=affine, track_running_stats=track_running_stats)]

        last = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(last, h, bias=not affine),           # bias not needed if BN affine=True
                nn.BatchNorm1d(h, affine=affine, track_running_stats=track_running_stats),
                parse_activation(activation)
            ]
            last = h

        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)