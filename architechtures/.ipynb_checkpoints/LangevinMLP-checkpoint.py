import torch
import torch.nn as nn

# Author: Winston Sullivan, Sarupria Group, University of Minnesota
# Date: 2025-03-28
# Description: This is a very basic MLP with the intent that it is for approximating Langevin style equations in an ITO
#              framework. In all honesty, I suspect this form should technically work for any SDE with a single time variable. 
#              If you find an error that is dire my contact info can be found winstonsullivan.netlify.app


class LangevinMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) for modeling Langevin dynamics in DDPM.

    This MLP is designed to approximate functions related to the Langevin equation,
    using an input state `x_s_t`, an initial condition `x_0_0`, and embeddings
    for diffusion time (`t_diff`) and physical time (`s_phys`).

    The architecture supports customizable activation functions and allows the use
    of any compatible embedding module.

    Attributes:
        data_dim (int): Dimension of the input data.
        device (str): Device where the model is located.
        embedding_module: An instance of an embedding module that provides `t_diff_embedding` and `t_phys_embedding`.
        module (nn.Sequential): The MLP model constructed based on the given layer configuration.
    """

    def __init__(self, layers, data_dim, t_diff_embedding, s_phys_embedding, activation=nn.ReLU, device='cpu'):
        """
        Initializes the LangevinMLP model.

        Args:
            layers (list of int): List of hidden layer sizes.
            data_dim (int): Dimensionality of the input data.
            embedding_module: An instance of an embedding class providing `t_diff_embedding` and `t_phys_embedding`.
            activation (nn.Module, optional): Activation function class (default: nn.ReLU).
            device (str, optional): Device to use for computations (default: 'cpu').
        """
        super(LangevinMLP, self).__init__()
        self.data_dim = data_dim
        self.device = device
        # Pre-instantiated embedding modules
        self.s_phys_embedding = s_phys_embedding  
        self.t_diff_embedding = t_diff_embedding  

        # Define MLP layers with customizable activation functions
        modules = [nn.Linear(2 * data_dim +  self.t_diff_embedding.dim+self.s_phys_embedding.dim, layers[0]), activation]
        for in_features, out_features in zip(layers[:-1], layers[1:]):
            modules.append(nn.Linear(in_features, out_features))
            modules.append(activation)

        modules.append(nn.Linear(layers[-1], data_dim))  # No activation after the final layer to allow predictions in (-\infty, \infty)

        self.module = nn.Sequential(*modules)
        self.to(device)

    def forward(self, x_s_t, x_0_0, t_diff, s_phys):
        """
        Forward pass of the LangevinMLP.

        Args:
            x_s_t (torch.Tensor): Current state tensor.
            x_0_0 (torch.Tensor): Initial condition tensor.
            t_diff (torch.Tensor): Diffusion time tensor.
            s_phys (torch.Tensor): Physical time tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the MLP.
        """
        # Compute embeddings
        t_diff_embed = self.t_diff_embedding(t_diff)
        s_phys_embed = self.s_phys_embedding(s_phys)

        # Concatenate inputs
        x = torch.cat((x_s_t, x_0_0, t_diff_embed, s_phys_embed), dim=1)

        # Perform forward pass through module
        return self.module(x)
