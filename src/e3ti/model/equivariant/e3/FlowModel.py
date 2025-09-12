
import torch
from e3nn import o3
from e3nn.nn import BatchNorm

from e3ti.utils import channels_arr_to_string, parse_activation, combine_features, rodrigues_vec
from e3ti.model.e3.SE3Transformer import SE3Transformer
from e3ti.model.e3.BuildingBlockEmbed import BuildingBlockEmbed
from e3ti.model.e3.TimeEmbed import TimeEmbed
from e3ti.model.e3.LatticeEmbed import LatticeEmbed
from e3ti.model.e3.RotationEmbed import RotationEmbed

class FlowModel(torch.nn.Module):

    def __init__(self, model_conf):
        super().__init__()
        self._model_conf = model_conf

        # Set up embeddings
        self.bb_embedder = BuildingBlockEmbed(model_conf.BuildingBlockEmbed)
        self.time_embedder = TimeEmbed(model_conf.TimeEmbed)
        self.lattice_embedder = LatticeEmbed(model_conf.LatticeEmbed)
        self.rotation_embedder = RotationEmbed(model_conf.RotationEmbed)

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
        self.periodic =  model_conf.periodic

        act = parse_activation(model_conf.act)

        self.eg3nn_layers = torch.nn.ModuleList()
        # Loop over the first attention module which maps from irreps_input -> irreps_output
        self.eg3nn_layers.append(SE3Transformer(max_radius, number_of_basis, hidden_size, act, max_neighbors,
            irreps_sh,           # max rank to embed edges via spherical tensors
            irreps_hidden,        # e3nn irrep corresponding to input feature
            irreps_hidden,       # desired irrep corresponding to output feature
            irreps_key,          # desired irrep corresponding to keys
            irreps_values,       # desired irrep corresponding to values
            irreps_query,        # desired irrep corresponding to query
            edge_basis,          # basis functions to use on edge emebeddings
            self.periodic,
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
                 self.periodic,
                )
            )
        combined_in_node = sum([self.bb_embedder.irreps_output,self.rotation_embedder.irreps_output], o3.Irreps())
        combined_in_node = combined_in_node.regroup() # type: ignore
        self.combine_embed_node = o3.FullyConnectedTensorProduct(combined_in_node, combined_in_node, irreps_hidden, shared_weights=True, internal_weights=True)
       
        combined_in_lattice = sum([o3.Irreps(f"{self.time_embedder.embedding_dim}x0e"),self.lattice_embedder.irreps_output], o3.Irreps())
        combined_in_lattice = combined_in_lattice.regroup() # type: ignore
        self.combine_embed_lattice= o3.FullyConnectedTensorProduct(combined_in_lattice, combined_in_lattice, self.lattice_embedder.irreps_output, shared_weights=True, internal_weights=True)

        self.lattice_interaction = o3.FullyConnectedTensorProduct(self.lattice_embedder.irreps_output, irreps_hidden, self.lattice_embedder.irreps_output, shared_weights=True, internal_weights=True)
        self.node_interaction = o3.FullyConnectedTensorProduct(self.lattice_embedder.irreps_output, irreps_hidden, irreps_hidden, shared_weights=True, internal_weights=True)
        
        self.bn_lattice  = BatchNorm(self.lattice_embedder.irreps_output) 
        self.bn_nodes  = BatchNorm(irreps_hidden) 

        self.trans_readout = o3.FullyConnectedTensorProduct(irreps_hidden, irreps_hidden, o3.Irreps('1x1e'), shared_weights=True, internal_weights=True)
        self.rot_readout = o3.FullyConnectedTensorProduct(irreps_hidden, irreps_hidden, o3.Irreps('1x1e'), shared_weights=True, internal_weights=True)
        self.lattice_readout = o3.FullyConnectedTensorProduct(self.lattice_embedder.irreps_output, self.lattice_embedder.irreps_output, o3.Irreps('3x1e'), shared_weights=True, internal_weights=True)

    def forward(self, batch):

        t         = batch['t']          # interpolant time for so3??
        trans_t   = batch['trans_t']    # noisey bb translations 
        rotmats_t = batch['rotmats_t']  # noisey bb rotation matrices
        lattice_t = batch['lattice_t']  # noisey lattice vectors
        batch_idx = batch['batch']      # for radius_graph
       
        #TODO: Consider https://pymatgen.org/pymatgen.core.html#pymatgen.core.lattice.Lattice.get_points_in_sphere for radial graphs
        #TODO: Similarly https://docs.e3nn.org/en/stable/guide/periodic_boundary_conditions.html

        # --------- EMBED ALL THE THINGS 🫨🤯😵‍💫 ---------
        # Emebed building blocks, takes global scatter mean of bb of e3nn irreps
        bb_emb = self.bb_embedder(batch)
        # Sinusoidal time embed
        time_emb = self.time_embedder(t)
        # Convert rotation matrices to rank l spherical tensors
        rotmats_emb = self.rotation_embedder(rotmats_t)
        # Convert lattice to rank l spherical tensors
        lattice_emb = self.lattice_embedder(lattice_t)
        # This is essentially the node positions. Need to embed these. This is done implicitly as "edge features"
        pos = trans_t

        for name, tensor in {
            "bb_emb": bb_emb,
            "time_emb": time_emb,
            "rotmats_emb": rotmats_emb,
            "lattice_emb": lattice_emb,
            "pos": pos,
        }.items():
            print(f"{name} NaN present: {torch.isnan(tensor).any().item()}")
        print()

        node_pairs= [
            (bb_emb       , self.bb_embedder.irreps_output),
            (rotmats_emb  , self.rotation_embedder.irreps_output),
            ]
        
        system_pairs = [
            (time_emb     , f"{self.time_embedder.embedding_dim}x0e"),
            (lattice_emb  , self.lattice_embedder.irreps_output),
        ]
        
        # combine all the node based irreps into one
        node_feats, node_irreps = combine_features(node_pairs) # type: ignore
        node_feats = self.combine_embed_node(node_feats,node_feats)

        # combine all the time and lattice based irreps into one
        lattice_feats, lattice_irreps = combine_features(system_pairs) # type: ignore
        lattice_feats = self.combine_embed_lattice(lattice_feats,lattice_feats)

        # --------- PAY ATTENTION!!! 📝🧠📌 --------- 
        for i in range(len(self.eg3nn_layers)):
            # Do message passing
            node_feat_update = self.eg3nn_layers[i](
                f=node_feats,
                pos=pos,
                batch=batch_idx,
                cell_lengths = torch.ones(len(t), 3).to('cuda') if self.periodic else None 
            )
            node_feats = node_feats + node_feat_update # Give skip connection here to help with gradient flow
            
            # repeat index once and reuse it everywhere
            idx = torch.repeat_interleave(                        # [total_nodes]
                    torch.arange(lattice_feats.size(0),           # 0 … B‑1
                                device=lattice_feats.device),
                    batch.num_bbs)

            # 1) lattice + node interaction -> new lattice feats
            tmp           = self.lattice_interaction(lattice_feats[idx], node_feats)   # [N, F_lat]
            lattice_feats = tmp[torch.cumsum(batch.num_bbs, 0) - 1]                    # [B, F_lat]

            for name, tensor in {
            "lattice_feats": lattice_feats,
            "node_feat_update": node_feat_update,
            "node_feats": node_feats,
            }.items():
                print(f"{name} NaN present: {torch.isnan(tensor).any().item()}")

            # 2) updated lattice + node interaction  -> new node feats
            node_feats = self.node_interaction(lattice_feats[idx], node_feats)

            for name, tensor in {
            "node_feats_after": node_feats,
            }.items():
                print(f"{name} NaN present: {torch.isnan(tensor).any().item()}")

            print()

            # 3) apply batch norm to data to prevent bread in code (nan)
            node_feats = self.bn_nodes(node_feats)
            lattice_feats = self.bn_lattice(lattice_feats)

        # --------- GRAPH READOUT TIME 📊🧐🗒️ ---------  
        trans_1 = trans_t + self.trans_readout(node_feats,node_feats)
        rot_1 = rodrigues_vec(self.rot_readout(node_feats,node_feats)) # use axis angle to convert irreps to rot matrix
        lat_1 = self.lattice_readout(lattice_feats,lattice_feats)

        return {
            'pred_trans': trans_1,
            'pred_rotmats': rot_1,
            'pred_lattice': lat_1,
        }
    