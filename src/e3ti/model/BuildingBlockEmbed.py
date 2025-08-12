import torch
from e3nn import o3
import numpy as np
from torch_geometric.nn.pool import global_mean_pool
from e3ti.utils import channels_arr_to_string, parse_activation, repeat_interleave
from e3ti.model.SE3Transformer import SE3Transformer

class BuildingBlockEmbed(torch.nn.Module):

    def __init__(self, bb_cfg):
        super().__init__()
        self._bb_cfg = bb_cfg
        self.atom_type_embedder = torch.nn.Embedding(bb_cfg.max_atoms, bb_cfg.type_f_dim)
        
        edge_basis = bb_cfg.edge_basis
        
        irreps_input = o3.Irreps(channels_arr_to_string(bb_cfg.input_channels))
        irreps_output = o3.Irreps(channels_arr_to_string(bb_cfg.output_channels))
        self.irreps_output = irreps_output
        self.out_scalar_feats = np.where(np.array(irreps_output.ls) == 0)[0] # type: ignore # grabs all the scalar feature idxs

        irreps_key = o3.Irreps(channels_arr_to_string(bb_cfg.key_channels))
        irreps_query = o3.Irreps(channels_arr_to_string(bb_cfg.query_channels))
        irreps_values = o3.Irreps(channels_arr_to_string(bb_cfg.output_channels)) # output channels = value channels

        irreps_sh = o3.Irreps.spherical_harmonics(bb_cfg.edge_l_max)

        max_radius = bb_cfg.max_radius
        number_of_basis = bb_cfg.number_of_basis
        hidden_size = bb_cfg.hidden_size
        max_neighbors = bb_cfg.max_neighbors

        act = parse_activation(bb_cfg.act)

        self.eg3nn_layers = torch.nn.ModuleList()
        # Loop over the first attention module which maps from irreps_input -> irreps_output
        self.eg3nn_layers.append(SE3Transformer(max_radius, number_of_basis, hidden_size, act, max_neighbors,
            irreps_sh,           # max rank to embed edges via spherical tensors
            irreps_input,        # e3nn irrep corresponding to input feature
            irreps_output,       # desired irrep corresponding to output feature
            irreps_key,          # desired irrep corresponding to keys
            irreps_values,       # desired irrep corresponding to values
            irreps_query,        # desired irrep corresponding to query
            edge_basis,          # basis functions to use on edge emebeddings
            ))

        # Loop over the rest of the layers which map from irreps_output -> irreps_output
        for _ in range(bb_cfg.num_layers-1):
            self.eg3nn_layers.append(SE3Transformer(max_radius, number_of_basis, hidden_size, act, max_neighbors,
                 irreps_sh,           # max rank to embed edges via spherical tensors
                 irreps_output,        # e3nn irrep corresponding to input feature
                 irreps_output,       # desired irrep corresponding to output feature
                 irreps_key,          # desired irrep corresponding to keys
                 irreps_values,       # desired irrep corresponding to values
                 irreps_query,        # desired irrep corresponding to query
                 edge_basis           # basis functions to use on edge emebeddings
                )
            )
    
    def forward(self, batch):
        local_coords = batch['local_coords']    # [N, 3]

        # Compute node features
        node_attr = self.atom_type_embedder(batch['atom_types'] - 1)     # [N, D]
        
        # Compute edge index
        batch_bb = repeat_interleave(batch['bb_num_vec'])

        # Forward through each equivariant attention mechanism
        for i in range(len(self.eg3nn_layers)):
           
            node_attr = self.eg3nn_layers[i](
                f=node_attr,
                pos=local_coords,
                batch=batch_bb,
            )

        # mean pooling over all features of the building block
        bb_attr = global_mean_pool(node_attr, batch_bb)

        return bb_attr