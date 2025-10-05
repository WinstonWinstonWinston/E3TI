
import torch
from e3nn import o3
from e3nn.o3 import Irreps,Linear
from e3nn.nn import BatchNorm

from e3ti.utils import channels_arr_to_string, parse_activation
from e3ti.model.equivariant.blocks.SE3Transformer import SE3Transformer
from typing import Dict

class MultiSE3Transformer(torch.nn.Module):

    def __init__(self, model_conf) -> None:
        super().__init__()
        self._model_conf = model_conf
    
        self.irreps_input = o3.Irreps(channels_arr_to_string(model_conf.input_channels))
        self.irreps_readout = o3.Irreps(channels_arr_to_string(model_conf.readout_channels))

        edge_basis = model_conf.edge_basis
        irreps_hidden = o3.Irreps(channels_arr_to_string(model_conf.hidden_channels))

        irreps_key = o3.Irreps(channels_arr_to_string(model_conf.key_channels))
        irreps_query = o3.Irreps(channels_arr_to_string(model_conf.query_channels))
        irreps_values = o3.Irreps(channels_arr_to_string(model_conf.hidden_channels)) # hidden channels = value channels

        irreps_sh = o3.Irreps.spherical_harmonics(model_conf.edge_l_max)

        max_radius = model_conf.max_radius
        number_of_basis = model_conf.number_of_basis
        hidden_size = model_conf.hidden_size
        max_neighbors = model_conf.max_neighbors

        act = parse_activation(model_conf.act)

        self.lin_in = Linear(self.irreps_input, irreps_hidden)

        self.eg3nn_layers = torch.nn.ModuleList()
        # Loop over the first attention module which maps from irreps_input -> irreps_output
        self.eg3nn_layers.append(SE3Transformer(max_radius, number_of_basis, hidden_size, act, max_neighbors,
            irreps_sh,           # max rank to embed edges via spherical tensors
            self.irreps_input,        # e3nn irrep corresponding to input feature
            irreps_hidden,       # desired irrep corresponding to output feature
            irreps_key,          # desired irrep corresponding to keys
            irreps_values,       # desired irrep corresponding to values
            irreps_query,        # desired irrep corresponding to query
            edge_basis,          # basis functions to use on edge emebeddings
            ))

        # Loop over the rest of the layers which map from irreps_output -> irreps_output
        for _ in range(model_conf.num_layers-1):
            self.eg3nn_layers.append(SE3Transformer(max_radius, number_of_basis, hidden_size, act, max_neighbors,
                 irreps_sh,           # max rank to embed edges via spherical tensors
                 irreps_hidden,       # e3nn irrep corresponding to input feature
                 irreps_hidden,       # desired irrep corresponding to output feature
                 irreps_key,          # desired irrep corresponding to keys
                 irreps_values,       # desired irrep corresponding to values
                 irreps_query,        # desired irrep corresponding to query
                 edge_basis,          # basis functions to use on edge emebeddings
                )
            )
       
        self.batch_norm  = BatchNorm(irreps_hidden) if model_conf.bn else lambda x:x 
        self.readout = o3.FullyConnectedTensorProduct(irreps_hidden, irreps_hidden, self.irreps_readout, shared_weights=True, internal_weights=True)

    def forward(self, batch):
        batch_idx = batch['batch']      # for radius_graph
        node_feats = batch['f']
        pos = batch['x']

        # convert shape with linear layer
        node_feats = self.lin_in('f')

        # --------- PAY ATTENTION!!! --------- 
        for i in range(len(self.eg3nn_layers)):
            # Do message passing
            node_feat_update = self.eg3nn_layers[i](
                f=node_feats,
                pos=pos,
                batch=batch_idx
            )        
            node_feats = node_feats + node_feat_update # Give skip connection here to help with gradient flow
            node_feats = self.batch_norm(node_feats)

        batch['v'] = self.readout(node_feats)

        return batch
    