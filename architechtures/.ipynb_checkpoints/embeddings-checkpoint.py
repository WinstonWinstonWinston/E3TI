import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Author: Winston Sullivan, Sarupria Group, University of Minnesota
# Date: 2025-03-28
# Description: This is an embedding module that implements a fourier like series with variable scale. It is good for
#              time-like variables. If you find an error that is dire my contact info can be found winstonsullivan.netlify.app

class SinCosTimeEmbedding(nn.Module):
    """
    Sine-Cosine Time Embedding module for encoding time steps into a fixed-size representation.
    
    This module uses sinusoidal and cosine functions to generate embeddings for time steps,
    following the positional encoding technique often used in transformers.
    
    Attributes:
        dim (int): The dimensionality of the time embedding.
        max_t (int): The maximum possible time step value.
        scale (torch.Tensor or nn.Parameter): Scaling factor applied to frequencies.
        freqs (torch.Tensor): Precomputed frequency values for sine-cosine embedding.
    """
    def __init__(self, dim, max_t=1000, init_scale=1.0, learnable_scale=False, device = 'cuda'):
        """
        Initializes the SinCosTimeEmbedding module.
        
        Parameters:
        - dim (int): The dimensionality of the output embedding.
        - max_t (int, optional): The maximum time step value (default: 1000).
        - init_scale (float, optional): Initial scaling factor for frequencies (default: 1.0).
        - learnable_scale (bool, optional): If True, the scale is a learnable parameter (default: False).
        """
        super().__init__()
        self.dim = dim
        self.max_t = max_t
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32,device = device))
        else:
            self.scale = torch.tensor(init_scale, dtype=torch.float32,device = args.device)

        exponents = torch.linspace(0, 1, dim // 2,device = device) * torch.log(torch.tensor(max_t,device = device))
        self.register_buffer("freqs", torch.exp(-exponents))  # Fixed frequencies
        self.device = device

    def forward(self, t):
        """
        Computes the sine-cosine embedding for the given time steps.
        
        Parameters:
        - t (torch.Tensor): Input tensor of shape (batch,) representing time steps.
          Can be float values rather than just integers.
        
        Returns:
        - torch.Tensor: Time embeddings of shape (batch, dim).
        """
        t = t[:, None] * (self.freqs[None, :] * self.scale)  # Apply scale
        return torch.cat([torch.sin(t), torch.cos(t)], dim=-1)  # Shape: (batch, dim)
    
    def plot_embedding(self, num_steps=100):
        """
        Generates a contour plot of the time embeddings.
        
        Parameters:
        - num_steps (int): Number of time steps to sample for visualization.
        
        Returns:
        - matplotlib.figure.Figure: The generated figure.
        """
        t = torch.linspace(0, self.max_t, num_steps, dtype=torch.float32,device=self.device)
        embeddings = self.forward(t).detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        t_grid, y_grid = np.meshgrid(t.detach().cpu().numpy(), np.arange(self.dim))
        c = ax.contourf(t_grid, y_grid, embeddings.T, levels=100, cmap='bone')
        plt.colorbar(c, ax=ax)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Embedding Component")
        ax.set_title("Sine-Cosine Time Embedding Contour Plot")
        
        return fig
