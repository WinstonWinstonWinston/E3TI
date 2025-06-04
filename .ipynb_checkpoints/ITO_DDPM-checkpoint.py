import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# Author: Winston Sullivan, Sarupria Group, University of Minnesota
# Lat Updated: 2025-05-27
# Description: This is a base class that all implicit transfer operator DDPMs should follow. Ideally it is NN agnostic.
#              It is meant to implement a generic diffusion loss and the key ITO sampling methods. If you find an error
#              that is dire my contact info can be found winstonsullivan.netlify.app
#
# Notes: Currently there is no ancestral sampling routine. If I want to inlcude forces as a part of this then there needs
# to be some force field to be called at that point in the algorithm. Also the way this is build I think should allow for 
# any scalar-vector net based on the description of them in "A Hitchhiker’s Guide to Geometric GNNs for 3D Atomic Systems".
# Also I think some of the doc strings are a bit fuckt rn, in that they are just wrong. This portion needs work before any 
# sort of realistic sharing to the community.

class ImplicitTransferOperatorDDPM(nn.Module):
    def __init__(self, architecture: nn.Module, noise_schedule, t_diff_max: int = 1000, s_phys_max: int = 1000, device: str = "cpu"):
        """
        Denoising Diffusion Probabilistic Model (DDPM) backbone.

        Args:
            architechture (nn.Module): Neural network architechture (e.g., E(3) PaiNN, SE(3) ChiroPaiNN, MLP, etc.).
            noise_schedule: Noise schedule class that provides alphas, betas, and other parameters.
            t_diff_max (int): Maximum diffusion steps (default: 1000).
            s_phys_max (int): Maximum physical time (default: 1000).
            device (str): Device for computation ('cuda', 'mps', or 'cpu').
        """
        super().__init__()
        self.architecture = architecture  # NN structure agnostic
        self.noise_schedule = noise_schedule
        self.t_diff_max = t_diff_max
        self.s_phys_max = s_phys_max
        self.device = device  # Store device setting

        # Extract useful parameters from the noise schedule and move to device
        self.alphas = noise_schedule.alphas.to(device)
        self.betas = noise_schedule.betas.to(device)
        self.alpha_bars = noise_schedule.alpha_bars.to(device)  # Cumulative product of alphas

    def forward(self, x_s_t, x_0_0, v_0_0, s_0_0, t_diff, s_phys):
        """
        Forward pass of the diffusion architechture.
    
        Args:
            x_s_t (List[Tensor]): List of input samples at time `s`, which may be clean or noisy depending on `t_diff`, each (batch_size, N, 3).
            x_0_0 (List[Tensor]): List of corresponding initial conditions of x_s_t, each (batch_size, N, 3).
            v_0_0 (List[Tensor]): List of corresponding vector features of x_0_t, each (batch_size, N, 3).
            s_0_0 (List[Tensor]): List of corresponding scalar features of x_0_t, each (batch_size, N).
            t_diff (Tensor): Diffusion time indices, shape (batch_size,).
            s_phys (Tensor): Physical time indices, shape (batch_size,).
    
        Returns:
            Tensor: Model's predicted noise.
        """
        # stack across channel dimension
        x_s_t   = torch.stack(x_s_t, dim=2)   # (B, N, C_x, 3)
        x_0_0   = torch.stack(x_0_0, dim=2)   # (B, N, C_x, 3)
        v_0_0   = torch.stack(v_0_0, dim=2)   # (B, N, C_v, 3)
        s_0_0   = torch.stack(s_0_0, dim=2)   # (B, N, C_z)

        # call architecture with stacked tensors
        out = self.architecture(x_s_t, x_0_0, v_0_0, s_0_0, t_diff, s_phys) # (B, N, C_x, 3)
        return list(out.unbind(dim=2))  # List of length C_x, each (B, N, 3)


    def loss(self, x_s_0, x_0_0, v_0_0, s_0_0, t_diff, s_phys, flatten=True):
        """
        Compute DDPM loss as the MSE between predicted and true noise.
    
        Args:
            x_s_0 (List[Tensor]): List of input samples at time `s`, each (batch_size, N, 3).
            x_0_0 (List[Tensor]): List of corresponding initial conditions of x_s_t, each (batch_size, N, 3).
            v_0_0 (List[Tensor]): List of corresponding vector features of x_0_t, each (batch_size, N, 3).
            s_0_0 (List[Tensor]): List of corresponding scalar features of x_0_t, each (batch_size, N).
            t_diff (Tensor): Diffusion time indices, shape (batch_size,).
            s_phys (Tensor): Physical time indices, shape (batch_size,).
            flatten (bool): Whether to return the mean squared error (MSE) or element-wise squared error.

        Returns:
            Tensor: Loss value.
        """
        noise_list = []
        x_s_t_list = []
        for x_s_0_i in x_s_0:
            # Sample Gaussian noise (same shape as x_s_0)
            noise_i = torch.randn_like(x_s_0_i,device=self.device)
            # Forward process: Generate noisy x_s_t
            x_s_t_i = torch.sqrt(self.alpha_bars[t_diff].view(-1,1,1)) * x_s_0_i + torch.sqrt(1 - self.alpha_bars[t_diff].view(-1,1,1)) * noise_i
            x_s_t_list.append(x_s_t_i)
            noise_list.append(noise_i)
    
        # Predict noise using the model
        noise_pred_list = self.forward(x_s_t_list, x_0_0, v_0_0, s_0_0, t_diff, s_phys)

        losses = []
        # Loop over all denoised vector features 
        for noise_pred_i,noise_i in zip(noise_pred_list,noise_list):
            # Compute loss (MSE between predicted and true noise)
            losses.append(nn.functional.mse_loss(noise_pred_i, noise_i) if flatten else (noise_pred_i - noise_i) ** 2)
        
        return losses

    @torch.no_grad()
    def sample_reverse_trajectory(self, x_0_0, v_0_0, s_0_0, s_phys):
        """
        Sample from the reverse diffusion process (denoising from noise to data).

        Args:
            x_0_0 (List[Tensor]): List of corresponding initial conditions of x_s_t, each (sample_count, N, 3).
            v_0_0 (List[Tensor]): List of corresponding vector features of x_0_t, each (sample_count, N, 3).
            s_0_0 (List[Tensor]): List of corresponding scalar features of x_0_t, each (sample_count, N).
            s_phys (sample_count): Physical time parameter.

        Returns:
            List[Tensor]: The full reverse trajectory list with elements of shape (t_diff_max, sample_count, N, 3).
        """

        # Start from pure noise in each vector dimension
        x_s_t_list = [torch.randn(x_0_0[0].size(), device=self.device) for i in range(len(x_0_0))]

        # Store full trajectory 
        trajectories = [torch.empty((self.t_diff_max, x_0_0[0].shape[0], x_0_0[0].shape[1],  x_0_0[0].shape[2]), device=self.device) for i in range(len(x_0_0))]
        
        # Reverse diffusion process: from time t_diff_max to 0
        for t_idx in reversed(range(self.t_diff_max)):
            
            # Time tensor for current step, copy this for each sample in sample_count 
            t_diff = torch.full((x_s_t_list[0].shape[0],), t_idx, device=self.device, dtype=torch.long)
            
            # Predict noise for the current step
            noise_pred_list  = self(x_s_t_list, x_0_0, v_0_0, s_0_0, t_diff, s_phys)
            means = [(1 / torch.sqrt(self.alphas[t_idx])) * (x_s_t - (1 - self.alphas[t_idx]) * noise_pred / torch.sqrt(1 - self.alpha_bars[t_idx])) for noise_pred,x_s_t in zip(noise_pred_list,x_s_t_list)]

            # Add noise for t > 0
            if t_idx > 0:
                x_s_t_list = [mean + self.betas[t_idx]**0.5 * torch.randn_like(mean,device=self.device) for mean in means]
            else:
                x_s_t_list = means  # No noise at t = 0
                
            for trajectory,x_s_t_i in zip(trajectories,x_s_t_list):
                trajectory[t_idx] = x_s_t_i  # Store step in trajectory

        return trajectories

    @torch.no_grad()
    def sample_forward_trajectory(self, x_s_0):
        """
        Sample a forward diffusion process (noising from data to noise).

        Args:
            x_s_0 (Tensor): Clean data sample, shape (sample_count, N).
        Returns:
            Tensor: The full forward trajectory of shape (t_diff_max, sample_count, N).
        """
        trajectories = [torch.empty((self.t_diff_max, x_s_0[0].shape[0], x_s_0[0].shape[1],  x_s_0[0].shape[2]), device=self.device) for i in range(len(x_s_0))]
        
        x_s_t = x_s_0

        for t_idx in range(self.t_diff_max):
            mean = [torch.sqrt(1.0 - self.betas[t_idx]) * x_s_t_i for x_s_t_i in x_s_t]
            std = torch.sqrt(self.betas[t_idx])  

            # Sample noise
            x_s_t = [mean_i + std * torch.randn_like(mean_i,device=self.device) for mean_i in mean]  # Apply diffusion update
            for trajectory,x_s_t_i in zip(trajectories,x_s_t):
                trajectory[t_idx] = x_s_t_i  # Store step in trajectory

        return trajectories

    
    @torch.no_grad()
    def sample_reverse_marginal(self, x_0_0, v_0_0, s_0_0, t_diff, s_phys):
        """
        Sample from the reverse diffusion process (denoising) at a given time t_idx.

        Args:
            x_0_0 (List[Tensor]): List of corresponding initial conditions of x_s_t, each (sample_count, N, 3).
            v_0_0 (List[Tensor]): List of corresponding vector features of x_0_t, each (sample_count, N, 3).
            s_0_0 (List[Tensor]): List of corresponding scalar features of x_0_t, each (sample_count, N).
            s_phys (sample_count): Physical time parameter.
            t_diff (int): Diffusion step index.

        Returns:
            Tensor: Sample at time t_diff after partial denoising, shape (sample_count, N).
        """
        # Start from pure noise in each vector dimension
        x_s_t_list = [torch.randn(x_0_0[0].size(), device=self.device) for i in range(len(x_0_0))]
        
        # Reverse diffusion process: from time t_diff_max to t_diff
        for t_idx in reversed(range(t_diff, self.t_diff_max)):
            
            # Time tensor for current step, copy this for each sample in sample_count 
            t_diff = torch.full((x_s_t_list[0].shape[0],), t_idx, device=self.device, dtype=torch.long)
            
            # Predict noise for the current step
            noise_pred_list  = self(x_s_t_list, x_0_0, v_0_0, s_0_0, t_diff, s_phys)
            means = [(1 / torch.sqrt(self.alphas[t_idx])) * (x_s_t - (1 - self.alphas[t_idx]) * noise_pred / torch.sqrt(1 - self.alpha_bars[t_idx])) for noise_pred,x_s_t in zip(noise_pred_list,x_s_t_list)]

            # Add noise for t > 0
            if t_idx > 0:
                x_s_t_list = [mean + self.betas[t_idx]**0.5 * torch.randn_like(mean,device=self.device) for mean in means]
            else:
                x_s_t_list = means  # No noise at t = 0

        return x_s_t_list

    @torch.no_grad()
    def sample_forward_marginal(self, x_s_0, t_diff):
        """
        Sample from the forward diffusion process at a given diffusion step t_idx.

        Args:
            x_s_0 (Tensor): Clean data sample, shape (sample_count, N).
            t_diff (Tensor): Diffusion step index, shape (sample_count).

        Returns:
            Tensor: Noisy sample at time t_idx, shape (sample_count, N).
        """
        # Noise scaling factor
        alpha_bar_t = self.alpha_bars[t_diff].view(-1, 1, 1)

        # Sample noise
        noise = [torch.randn_like(x_s_0_i,device=self.device) for x_s_0_i in x_s_0]

        # Compute the noisy sample at time t_idx
        x_s_t = [torch.sqrt(alpha_bar_t) * x_s_0_i + torch.sqrt(1 - alpha_bar_t) * noise_i for noise_i,x_s_0_i in zip(noise,x_s_0)]

        return x_s_t

    # Depreciated for now. 
    # @torch.no_grad()
    # def ancestral_sampling(self, x_0_0, s_phys, s_phys_tau, save_trajectory=False):
    #     """
    #     Perform ancestral sampling, applying multiple full compositions and a final partial composition.

    #     Args:
    #         x_0_0 (Tensor): Initial condition, shape (sample_count, N).
    #         s_phys (int): Indexed physical time.
    #         s_phys_tau (int): Coarse stepping integer, must be > 1 and < self.s_phys_max.
    #         save_trajectory (bool): If True, saves and returns the full trajectory.

    #     Returns:
    #         Tensor: Final sampled output matching `self.s_phys_max`, or the full trajectory if `save_trajectory=True`.
    #     """
    #     # Check if s_phys_tau is valid
    #     if s_phys_tau > self.s_phys_max or s_phys_tau <= 1:
    #         raise ValueError("s_phys_tau must be greater than 1 and less than or equal to self.s_phys_max.")

    #     # If s_phys is smaller than the stepping interval, return forward marginal sampling
    #     if s_phys < s_phys_tau:
    #         return self.sample_forward_marginal(x_0_0, s_phys)

    #     # Determine number of full compositions and remainder
    #     num_full_compositions = s_phys // s_phys_tau
    #     remainder_steps = s_phys % s_phys_tau  # Leftover steps for the final partial composition

    #     # Convert s_phys_tau and remainder_steps into tensors with batch dimensions
    #     batch_size = x_0_0.size(0)
    #     s_phys_tau_tensor = torch.full((batch_size,), s_phys_tau, device=self.device, dtype=torch.long)
    #     remainder_steps_tensor = torch.full((batch_size,), remainder_steps, device=self.device, dtype=torch.long)
    #     t_diff_tensor = torch.full((batch_size,), self.t_diff_max, device=self.device, dtype=torch.long)

    #     # If saving trajectory, pre-allocate storage
    #     if save_trajectory:
    #         total_steps = num_full_compositions + (1 if remainder_steps > 0 else 0)
    #         trajectory = torch.empty((total_steps, *x_0_0.shape), device=self.device)

    #     # Perform full compositions
    #     x_s_0 = x_0_0  # Start with the initial condition
    #     for i in range(num_full_compositions):
    #         x_s_0 = self.sample_reverse_marginal(x_s_0, t_diff_tensor, s_phys_tau_tensor)
    #         if save_trajectory:
    #             trajectory[i] = x_s_0

    #     # Perform the final partial composition to match remainder, if needed
    #     if remainder_steps > 0:
    #         x_s_0 = self.sample_reverse_marginal(x_s_0, t_diff_tensor, remainder_steps_tensor)
    #         if save_trajectory:
    #             trajectory[num_full_compositions] = x_s_0

    #     return trajectory if save_trajectory else x_s_0

# Noise Scheduler Classes
class BaseNoiseSchedule:
    """
    Base class for noise schedules used in diffusion models.
    Computes beta, alpha, and cumulative alpha_bar values based on a given time schedule.
    """
    def __init__(self, low: float, high: float, diffusion_steps: int, device: torch.device):
        """
        Initializes the noise schedule.
        
        Parameters:
        - low (float): The lower bound of the time schedule.
        - high (float): The upper bound of the time schedule.
        - diffusion_steps (int): Number of diffusion steps.
        - device (torch.device): Device to store tensors.
        """
        self.diffusion_steps = diffusion_steps
        self.ts = torch.linspace(low, high, diffusion_steps, device=device)
        self.betas = self.compute_betas()
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def compute_betas(self):
        """
        Method to compute the beta values. Must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement compute_betas()")
    
    def plot(self, title: str):
        """
        Plots the beta, alpha, alpha_bar, and signal-to-noise ratio (SNR) values.
         
        Returns:
            matplotlib.figure.Figure: The generated figure containing the plot.
        """
        snr = self.alpha_bars / (1 - self.alpha_bars + 1e-8)  # Avoid division by zero
        ts_cpu = self.ts.detach().cpu()

        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        
        axs[0].plot(ts_cpu, self.betas.detach().cpu(), label="Beta", linestyle="--")
        axs[0].set_ylabel("Beta")
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(ts_cpu, self.alphas.detach().cpu(), label="Alpha", linestyle="-." )
        axs[1].set_ylabel("Alpha")
        axs[1].legend()
        axs[1].grid(True)
        
        axs[2].plot(ts_cpu, self.alpha_bars.detach().cpu(), label="Alpha Bar", linestyle=":")
        axs[2].set_ylabel("Alpha Bar")
        axs[2].legend()
        axs[2].grid(True)
        
        axs[3].plot(ts_cpu, snr.detach().cpu(), label="SNR", linestyle="-")
        axs[3].set_xlabel("Time Step")
        axs[3].set_ylabel("SNR")
        axs[3].legend()
        axs[3].grid(True)
        
        fig.suptitle(title)
        return fig

class SigmoidNoiseSchedule(BaseNoiseSchedule):
    """Noise schedule with a sigmoid-based beta schedule."""
    def compute_betas(self):
        return torch.sigmoid(self.ts)

class LinearNoiseSchedule(BaseNoiseSchedule):
    """Noise schedule with a linear beta schedule."""
    def compute_betas(self):
        return (self.ts - self.ts.min()) / (self.ts.max() - self.ts.min())

class CosineNoiseSchedule(BaseNoiseSchedule):
    """Noise schedule with a cosine beta schedule."""
    def compute_betas(self):
        return 0.5 * (1 - torch.cos(torch.pi * (self.ts - self.ts.min()) / (self.ts.max() - self.ts.min())))

class QuadraticNoiseSchedule(BaseNoiseSchedule):
    """Noise schedule with a quadratic beta schedule."""
    def compute_betas(self):
        return ((self.ts - self.ts.min()) / (self.ts.max() - self.ts.min())) ** 2