import torch.nn as nn
import torch
from torch import Tensor
from itertools import combinations
import numpy as np

class VectorMatrixLayer(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(A,B))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        # x: [b, N, A, d]
        # weight: [A, B]
        # output: [B, N, B, d]
        return torch.einsum('AB,bnAd->bnBd', self.weight, x)
        
    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.weight.shape[0]}, out_channels={self.weight.shape[1]})"

class EquivariantGraphNeuralNetwork(nn.Module):

    def __init__(self, 
                f_0_layers, f_1_layers, f_2_layers, f_3_layers, f_4_layers, f_5_layers,
                W_1_layers, W_2_layers, W_3_layers, W_4_layers,
                message_passing_steps, p, activation_function,
                t_diff_embedding, s_phys_embedding,
                device
                ):

        super(EquivariantGraphNeuralNetwork, self).__init__()
        
        self.message_passing_steps = message_passing_steps
        self.activation_function = activation_function
        self.device = device
        self.p = p
         
        # assume a fully connected graph (undirected, no self-loops)

        # --- channel sizes --------------------------------------------------------------------------------------

        # scalar channel count
        self.C_z = f_3_layers[-1]

        # vector channel count
        self.C_v = f_2_layers[-1]

        # data channel count
        self.C_x = f_1_layers[-1]

        # physical time channel count
        self.C_s = s_phys_embedding.dim

        # diffusion time channel count
        self.C_t = t_diff_embedding.dim

        # --- embedding networks --------------------------------------------------------------------------------------

        self.t_diff_embed = t_diff_embedding
        self.s_phys_embed = s_phys_embedding
        
        f_0_modules = []
        for in_features, out_features in zip(f_0_layers[:-1], f_0_layers[1:]):
            f_0_modules.append(nn.Linear(in_features, out_features))
            f_0_modules.append(activation_function)
            f_0_modules.append(torch.nn.Dropout(p=self.p))
        self.f_0 = nn.Sequential(*f_0_modules[:-2])

        self.W_1 = VectorMatrixLayer(W_1_layers[0][0],W_1_layers[1][0])

        # --- message networks --------------------------------------------------------------------------------------

        # --- scalar features --------------------------------------------------------

        f_3_modules = []
        for in_features, out_features in zip(f_3_layers[:-1], f_3_layers[1:]):
            f_3_modules.append(nn.Linear(in_features, out_features))
            f_3_modules.append(activation_function)
            f_3_modules.append(torch.nn.Dropout(p=self.p))
        self.f_3 = nn.Sequential(*f_3_modules[:-2]) # exclude dropout and activation on the last pass to put in (-infty,infty)

        # --- vector features --------------------------------------------------------

        f_2_modules = []
        for in_features, out_features in zip(f_2_layers[:-1], f_2_layers[1:]):
            f_2_modules.append(nn.Linear(in_features, out_features))
            f_2_modules.append(activation_function)
            f_2_modules.append(torch.nn.Dropout(p=self.p))
        self.f_2 = nn.Sequential(*f_2_modules[:-2]) # exclude dropout and activation on the last pass to put in (-infty,infty)

        f_1_modules = []
        for in_features, out_features in zip(f_1_layers[:-1], f_1_layers[1:]):
            f_1_modules.append(nn.Linear(in_features, out_features))
            f_1_modules.append(activation_function)
            f_1_modules.append(torch.nn.Dropout(p=self.p))
        self.f_1 = nn.Sequential(*f_1_modules[:-2]) # exclude dropout and activation on the last pass to put in (-infty,infty)

        self.W_2 = VectorMatrixLayer(W_2_layers[0][0],W_2_layers[1][0])
        
        # --- update networks --------------------------------------------------------------------------------------

        # --- vector features --------------------------------------------------------
        
        f_4_modules = []
        for in_features, out_features in zip(f_4_layers[:-1], f_4_layers[1:]):
            f_4_modules.append(nn.Linear(in_features, out_features))
            f_4_modules.append(activation_function)
            f_4_modules.append(torch.nn.Dropout(p=self.p))
        self.f_4 = nn.Sequential(*f_4_modules[:-2]) # exclude dropout and activation on the last pass to put in (-infty,infty)

        self.W_3 = VectorMatrixLayer(W_3_layers[0][0],W_3_layers[1][0])

        # --- scalar features --------------------------------------------------------
        
        f_5_modules = []
        for in_features, out_features in zip(f_5_layers[:-1], f_5_layers[1:]):
            f_5_modules.append(nn.Linear(in_features, out_features))
            f_5_modules.append(activation_function)
            f_5_modules.append(torch.nn.Dropout(p=self.p))
        self.f_5 = nn.Sequential(*f_5_modules[:-2]) # exclude dropout and activation on the last pass to put in (-infty,infty)


        # --- readout --------------------------------------------------------------------------------------
        
        self.W_4 = VectorMatrixLayer(W_4_layers[0][0],W_4_layers[1][0])

    def embed(self,  x_s_t, x_0_0, v_0_0, z_0_0, t_diff, s_phys):
        r"""
        Args:
            x_s_t   (Tensor): Target input vectors, shape [B, N, C_x, 3].
            x_0_0   (Tensor): Auxiliary input vectors, shape [B, N, C_x, 3].
            v_0_0   (Tensor): Auxiliary velocity vectors, shape [B, N, C_v, 3].
            z_0_0   (Tensor): Auxiliary scalar features, shape [B, N, C_z].
            t_diff  (Tensor): Diffusion time indices, shape [B,].
            s_phys  (Tensor): Physical time indices, shape [B,].

        Returns:
            vectorial_feat (torch.Tensor):
                Shape [B, N, C_v, 3]
            scalar_feat (torch.Tensor):
                Shape [B, N, C_z]
        """

        B = len(t_diff)
        N = len(x_s_t[0])
        
        # embed the vector features and momentum
        # [B, N, 2C_x + C_v_i, 3] -> [B, N, C_v, 3]
        vectorial_feat = self.W_1(torch.cat([x_s_t,x_0_0,v_0_0], dim=2))
        scalarize_vectorial_feat = (vectorial_feat**2).sum(dim=-1)**0.5 # [B, N, C_v]

        # embed the times
        t_diff = self.t_diff_embed(t_diff).unsqueeze(dim=1).expand(B, N, self.C_t) #  [B, N, C_t]
        s_phys = self.s_phys_embed(s_phys).unsqueeze(dim=1).expand(B, N, self.C_s) #  [B, N, C_s]

        # [B, N, C_s + C_t + C_v + C_z_i] -> [B, N, C_z]
        scalar_feat = self.f_0(torch.cat([scalarize_vectorial_feat, t_diff, s_phys, z_0_0], dim=-1))

        return vectorial_feat, scalar_feat

    def message(self, vectorial_feat: Tensor, scalar_feat: Tensor, x_s_t: Tensor):
        r"""
        Parameters:
            vectorial_feat (torch.Tensor):
                Vectorial representations. Shape [B, N, C_v, 3]
            scalar_feat (torch.Tensor):
                Scalar representations. Shape [B, N, C_z]
            x_s_t (torch.Tensor):
                Atom's Node positions across C_x. Shape [B, N, C_x, 3]

        Returns:
            vectorial_message (torch.Tensor):
                Shape [B, N, C_v, 3]
            scalar_message (torch.Tensor):
                Shape [B, N, C_z]
        """
        B, N = x_s_t.shape[0], x_s_t.shape[1]
        
        # compute all pairwise differences
        x_ij_full = x_s_t[:, :, None, :, :] - x_s_t[:, None, :, :, :]  #  broadcasted over batch [B, N, N, C_x, 3]

        # create a mask to exclude diagonal of an N,N matrix
        mask = ~torch.eye(N, dtype=bool, device=self.device) #  [N, N]
        
        # apply mask across batch, reshape
        x_ij_matrix = x_ij_full[:, mask].view(B, N, N - 1, self.C_x, 3)  # [B, N, N-1, C_x, 3]

        # compute per-channel magnitudes
        abs_x_ij = torch.norm(x_ij_matrix, dim=-1)  # [B, N, N-1, C_x]

        # expand scalars to all (i, j) pairs
        z_j = scalar_feat.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, C_z]
        z_j_no_diag = z_j[:, mask].view(B, N, N - 1, self.C_z) # [B, N, N-1, C_z]

        # expand vectors to all (i, j) pairs
        v_j = vectorial_feat.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [B, N, N, C_v, 3]
        v_j = v_j[:, mask].view(B, N, N - 1, self.C_v, 3)  # [B, N, N-1, C_v, 3]

        # cat features for f_something
        cat_feats = torch.cat([z_j_no_diag, abs_x_ij], dim=-1)  # [B, N, N-1, C_z + C_x]
        
        # apply non-linearity
        # f_1: scalar message embedding from (z_j, |x_ij|)
        # f_2: vector weights from (z_j, |x_ij|) for combining vectorial features
        # f_3: scalar weights from (z_j, |x_ij|) for combining edge features
        transformed_scalar_feat = self.f_3(cat_feats) # [B, N, N-1, C_z]
        transformed_vector_feat = self.f_2(cat_feats) # [B, N, N-1, C_v]
        transformed_edge_feat   = self.f_1(cat_feats) # [B, N, N-1, C_x]

        # gates from f_2: [B, N, N-1, C_v]
        v_gate = transformed_vector_feat.unsqueeze(-1)  # [B, N, N-1, C_v, 1]
        
        # channel-wise gating of vector features
        gated_vectors = v_gate * v_j  # [B, N, N-1, C_v, 3]
        
        # aggregate over neighbors
        vectorial_message = gated_vectors.mean(dim=2)  # [B, N, C_v, 3]

        # expand gate for broadcasting
        edge_gate = transformed_edge_feat.unsqueeze(-1)  # [B, N, N-1, C_x, 1]
        
        # channel-wise gating
        gated_x_ij = edge_gate * x_ij_matrix  # [B, N, N-1, C_x, 3]
        
        # aggregate over neighbors
        x_message = gated_x_ij.mean(dim=2)  # [B, N, C_x, 3]

        # aggregate over neighbors
        scalar_message = transformed_scalar_feat.mean(dim=2)  # [B, N, C_z]

        # compute scalar message
        scalar_feat = scalar_feat + scalar_message  # [B, N, C_z]
        vectorial_feat = vectorial_feat + self.W_2(x_message) + vectorial_message  # [B, N, C_v, 3]
        
        return vectorial_feat, scalar_feat
        
    def update(self, vectorial_feat: Tensor, scalar_feat: Tensor):
        r"""
        Parameters:
            vectorial_feat (torch.Tensor):
                Vectorial representations. Shape [B, N, embedding_dim, 3]
            scalar_feat (torch.Tensor):
                Scalar representations. Shape [B, N, embedding_dim]

        Returns:
            vectorial_update (torch.Tensor):
                Shape [B, N, embedding_dim, 3]
            scalar_update (torch.Tensor):
                Shape [B, N, embedding_dim]
        """
        scalarize_vectorial_feat = (vectorial_feat**2).sum(dim=-1)**0.5 # [B, N, C_v]

        # [B, N, C_v + C_z] ->  [B, N, C]
        scalar_delta = self.f_5(torch.cat([scalar_feat, scalarize_vectorial_feat],dim=-1)) # [B, N, C_z]

        vector_gate = self.f_4(torch.cat([scalar_feat, scalarize_vectorial_feat],dim=-1)) # [B, N, C_v]
        vector_delta = vector_gate.unsqueeze(dim=-1)*vectorial_feat # [B, N, C, 3]

        scalar_update = scalar_feat + scalar_delta

        # apply channel mixing
        mixed_vector_delta = self.W_3(vector_delta)  # [B, N, C_v, 3]

        # Update vectorial feature
        vectorial_update = vectorial_feat + mixed_vector_delta  # [B, N, C_v, 3]

        return vectorial_update, scalar_update

    def forward(self, x_s_t, x_0_0, v_0_0, s_0_0, t_diff, s_phys):
        r"""
        Args:
            x_s_t   (Tensor): Target input vectors, shape [B, N, C_x, 3].
            x_0_0   (Tensor): Auxiliary input vectors, shape [B, N, C_x, 3].
            v_0_0   (Tensor): Auxiliary velocity vectors, shape [B, N, C_v, 3].
            z_0_0   (Tensor): Auxiliary scalar features, shape [B, N, C_z].
            t_diff  (Tensor): Diffusion time indices, shape [B,].
            s_phys  (Tensor): Physical time indices, shape [B,].

        Returns:
            epsilon_s_t   (Tensor): Output expected noise in, shape [B, N, C_x, 3]
        """

        vectorial_feat, scalar_feat = self.embed(x_s_t, x_0_0, v_0_0, s_0_0, t_diff, s_phys) # [B,N,C_v,3], [B,N,C_z,3]

        # start a loop over the number of message + update steps
        for t in range(self.message_passing_steps):
            # apply message
            # [B,N,C_v,3], [B,N,C_z,3]
            vectorial_feat_t, scalar_feat_t = self.message(vectorial_feat, scalar_feat, x_s_t)
            # apply update
            # [B,N,C_v,3], [B,N,C_z,3]
            vectorial_feat, scalar_feat = self.update(vectorial_feat_t, scalar_feat_t)
            
        # readout the new position and momentum from a linear combination of the chanels
        # [B, N, C_v, 3] -> [B, N, C_x, 3]
        return self.W_4(vectorial_feat)