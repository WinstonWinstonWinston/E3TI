import torch
from torch_cluster import radius_graph
from torch_scatter import scatter,scatter_max
from e3nn import o3
import e3nn.nn as enn
from e3nn.math import soft_unit_step, soft_one_hot_linspace
from e3ti.utils import periodic_radius_graph
 
class SE3Transformer(torch.nn.Module):
    """ 
    SE3Transformer implemenation from https://docs.e3nn.org/en/stable/guide/transformer.html
    Follows details of https://arxiv.org/pdf/2006.10503
    """

    def __init__(self, _max_radius, _number_of_basis, _hidden_size, act, max_neighbors,
                 _irreps_sh,             # max rank to embed edges via spherical tensors
                 _irreps_input,          # e3nn irrep corresponding to input feature
                 _irreps_output,         # desired irrep corresponding to output feature
                 _irreps_key,            # desired irrep corresponding to keys
                 _irreps_values,         # desired irrep corresponding to values
                 _irreps_query,          # desired irrep corresponding to query
                 _edge_basis,            # basis functions to use on edge emebeddings https://docs.e3nn.org/en/latest/api/math/math.html#e3nn.math.soft_one_hot_linspace
                 _periodic = False,      # boolean flag on whether or not to apply pbc, default no
                 ): 

        super().__init__()
        
        self.max_radius = _max_radius
        self.max_neighbors = max_neighbors
        self.number_of_basis = _number_of_basis
        self.irreps_sh = _irreps_sh
        self.irreps_input = _irreps_input
        self.irreps_output = _irreps_output
        self.irreps_key = _irreps_key
        self.irreps_values = _irreps_values
        self.irreps_query = _irreps_query
        self.edge_basis = _edge_basis
        self.act = act
        self.periodic = _periodic

        self.tp_k = o3.FullyConnectedTensorProduct(self.irreps_input, self.irreps_sh, self.irreps_key, shared_weights=False)
        self.fc_k = enn.FullyConnectedNet([self.number_of_basis, _hidden_size, self.tp_k.weight_numel], act=self.act)
        
        self.h_q = o3.Linear(self.irreps_input, self.irreps_query)

        self.tp_v = o3.FullyConnectedTensorProduct(self.irreps_input, self.irreps_sh, self.irreps_output, shared_weights=False)
        self.fc_v = enn.FullyConnectedNet([self.number_of_basis, _hidden_size, self.tp_v.weight_numel], act=self.act)

        self.dot = o3.FullyConnectedTensorProduct(self.irreps_query, self.irreps_key, "0e")

        # Hard coded to stop the sqrt from being zero.
        self.eps = 1e-12

    def forward(self, f, pos, batch, cell_lengths=None):
        if self.periodic:
            edge_src, edge_dst, edge_vec = periodic_radius_graph(pos, batch, self.max_radius, cell_lengths)

        else:
            edge_src, edge_dst = radius_graph(x=pos, 
                                          r=self.max_radius, 
                                          batch=batch, 
                                          loop=False,
                                          max_num_neighbors=self.max_neighbors)
        
            edge_vec = pos[edge_src] - pos[edge_dst]
        
        edge_length = edge_vec.norm(dim=1)

        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis=self.edge_basis,
            cutoff=True
        )

        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))

        edge_sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, True, normalization='component')

        q = self.h_q(f)
        k = self.tp_k(f[edge_src], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(f[edge_src], edge_sh, self.fc_v(edge_length_embedded))

        logits = self.dot(q[edge_dst], k)
        max_dst, _ = scatter_max(logits, edge_dst, dim=0, dim_size=len(f))
        
        shifted = logits - max_dst[edge_dst] # Shift by max to avoid overflow inf/inf
        exp = edge_weight_cutoff[:, None] * shifted.exp() # type: ignore
        z = scatter(exp, edge_dst, dim=0, dim_size=len(f))
        z[z == 0] = 1

        alpha = exp / z[edge_dst]

        to_chk = {
            "f": f,
            "pos": pos,
            "edge_vec": edge_vec,
            "edge_length": edge_length,
            "edge_length_embedded": edge_length_embedded,
            "edge_weight_cutoff": edge_weight_cutoff,
            "edge_sh": edge_sh,
            "q": q,
            "k": k,
            "v": v,
            "exp": exp,
            "z": z,
            "alpha": alpha,
        }

        nan_report = {
            name: {
                "shape": tuple(t.shape),
                "nan_count": torch.isnan(t).sum().item(),
            }
            for name, t in to_chk.items()
            if torch.isnan(t).any()
        }

        dot_vals = self.dot(q[edge_dst], k)                 # logits before exp
        overflow_mask = dot_vals > 80.0                     # 80 ≈ exp(80) ~ 5e34 (float32 max~3e38)

        if overflow_mask.any():                             # only print when we really overflow
            idx = overflow_mask.nonzero(as_tuple=False)[:10]      # first few offenders
            print(
                ">>> exp overflow at", idx.shape[0], "edges.",
                "Sample logits:", dot_vals[idx.flatten()].tolist()[:5]
            )

        for name, tensor in {"exp": exp, "z": z, "z_edge": z[edge_dst]}.items():
            if torch.isinf(tensor).any():
                cnt = torch.isinf(tensor).sum().item()
                print(f">>> {name} has {cnt:,} inf values (shape={tuple(tensor.shape)})")


        if nan_report:                            # print only when something is wrong
            print(">>> NaNs detected:", nan_report)

        return scatter((alpha.relu() + self.eps).sqrt() * v, edge_dst, dim=0, dim_size=len(f))
