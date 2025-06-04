import torch
import numpy as np
import os
import matplotlib.pyplot as plt

# Author: Winston Sullivan, Sarupria Group, University of Minnesota
# Date: 2025-04-04
# Description: This maybe the jankiest class in the whole fileset. I barely know what it does, it gives me an aneurysm 
#              to look too closely at my scaling. I had to do a bunch of shenanigans to make it have the magic of "vectorization".
#              Good luck to all who read this. If you find an error that is dire my contact info can be found winstonsullivan.netlify.app

class TrajectoryDataset():
    """
    Dataset class for trajectories with randomly selected initial conditions (IC).

    Args:
        trajectory (Tensor): Tensor of trajectories with shape [trajectory index, time, features].
        t_diff_max (int): Maximum diffusion time step.
        s_phys_max (int): Maximum physical time for sampling.
        device (str): Device to place tensors on (default: 'cpu').
        seed (int, optional): Seed for random number generation.
    """
    def __init__(self, trajectory, t_diff_max, s_phys_max, device, scaling_sample_size=None, scale_data = True, seed=None):
        super().__init__()
        self.device = device
        self.trajectory = trajectory.to(self.device)
        self.s_phys_max = s_phys_max
        self.t_diff_max = t_diff_max
        self.num_trajs = len(self.trajectory[0])
        self.data_dim = len(self.trajectory[0,0])
        self.scale_data = scale_data
    
        if scale_data:
            
            assert s_phys_max < len(self.trajectory)/2
            assert not(scaling_sample_size is None)

            s_phys_max_here = int(1.5*s_phys_max)
            
            traj_idxs = torch.randint(0, self.num_trajs, (scaling_sample_size,),device=self.device)
            N_vals = torch.rand(scaling_sample_size,device=self.device)  * np.log(s_phys_max_here)
            ic_idx = torch.randint(1, len(trajectory)- (s_phys_max_here), (scaling_sample_size,),device=self.device)
            
            s_phys = torch.floor(torch.exp(N_vals)).long()
            t_diff = torch.randint(0, t_diff_max, (scaling_sample_size
                                                   ,),device=self.device)
            x_0_0 = trajectory[ic_idx, traj_idxs]
            x_s_0 = trajectory[ic_idx + s_phys, traj_idxs]
            
            # Repeat each row i exactly s_phys[i] times            
            R = torch.arange(scaling_sample_size,device=self.device).repeat_interleave(s_phys) # length of R == sum of all snippet-lengths
            
            range_tensor = torch.arange(s_phys_max_here, device=self.device)
            
            # (N, max_len) matrix where True means the index is valid
            mask_1 = range_tensor.unsqueeze(0) < s_phys.unsqueeze(1)
            
            # Use the mask to extract valid aranges
            O = range_tensor.expand(s_phys.shape[0], -1)[mask_1]
            
            # T and S:
            T = traj_idxs[R]   # shape == (sum_of_lengths,)
            S = ic_idx[R]      # shape == (sum_of_lengths,)
            
            sub_intervals = torch.zeros((scaling_sample_size, s_phys_max_here, len(self.trajectory[0,0])), dtype=self.trajectory.dtype,device=self.device)
            
            sub_intervals[R, O, :] = (
                self.trajectory[S + O,T, :]    # the raw slice
                - self.trajectory[S,T, : ]     # subtract the first row
            )
            
            mask_2 = torch.arange(s_phys_max_here,device=self.device) < s_phys[:, None]    
            
            # sum of values at each offset
            sums = sub_intervals.sum(dim=0)  # (max_len, D)
            # sum of squares
            sums_sq = (sub_intervals ** 2).sum(dim=0)  # (max_len, D)
            # how many sub-intervals contribute at each offset
            counts = mask_2.sum(dim=0).unsqueeze(-1)  # (max_len, 1)
            
            counts_clamped = counts.clamp(min=1)
            mean_by_offset = sums / counts_clamped
            mean_sq = sums_sq / counts_clamped
            var = mean_sq - mean_by_offset ** 2
            std_by_offset = var.clamp_min(0.0).sqrt()

            # If offset has no valid sub-intervals, set those to NaN
            invalid_offsets = (counts.squeeze(-1) == 0)
            mean_by_offset[invalid_offsets] = float('nan')
            std_by_offset[invalid_offsets] = float('nan')

            self.mean = mean_by_offset[:s_phys_max].to(self.device)
            self.std = std_by_offset[:s_phys_max].to(self.device) + 1e-5

    def getitems(self, batch_size):
        traj_idxs = torch.randint(0, self.num_trajs, (batch_size,),device=self.device)
        
        N_vals = torch.rand(batch_size,device=self.device)  * np.log(self.s_phys_max)
        ic_idx = torch.randint(1, len(self.trajectory)- (self.s_phys_max), (batch_size,),device=self.device)
        
        s_phys = torch.floor(torch.exp(N_vals)).long()
        t_diff = torch.randint(0, self.t_diff_max, (batch_size,),device=self.device)
        x_0_0 = self.trajectory[ic_idx, traj_idxs]
        x_s_0 = self.trajectory[ic_idx + s_phys, traj_idxs]
        
        if self.scale_data:
            x_s_0 = self.scale(x_s_0 - x_0_0, s_phys)
        
        return {"x_0_0":x_0_0,
                "x_s_0":x_s_0,
                "t_diff":t_diff,
                "s_phys":s_phys}

    def getitem(self):
        return self.getitems(1)

    def scale(self,x_s_0, s_phys):
        return (x_s_0 - self.mean[s_phys])/self.std[s_phys]

    def unscale(self,x_s_0, s_phys):
        return x_s_0*self.std[s_phys] + self.mean[s_phys]

    def plot_mean_std(self, variable: int):
        """
        Plots the mean and standard deviation over time for the given variable index.
    
        Args:
            variable (int): Index of the variable (column) to plot from self.mean and self.std.
    
        Returns:
            matplotlib.figure.Figure: The generated plot figure.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
    
        mean_slice = self.mean[:, variable].detach().cpu()
        std_slice = self.std[:, variable].detach().cpu()
        t_range = torch.arange(len(mean_slice))
    
        ax.plot(t_range, mean_slice, label="Mean")
        ax.fill_between(t_range, mean_slice - std_slice, mean_slice + std_slice,
                        alpha=0.3, label="±1 Std Dev")
    
        ax.set_xlabel("Time step")
        ax.set_ylabel(f"Variable {variable} value")
        ax.set_title(f"Mean and Std Dev of Variable {variable}")
        ax.legend()
        ax.grid(True)

        return fig 

class TrajectoryPositionMomentum(TrajectoryDataset):
    """
    Dataset class wrapper for positional and momenta trajectories with randomly selected initial conditions (IC).

    Args:
        data_path (str): Filepath to position data generated by MD2D.
        t_diff_max (int): Maximum diffusion time step.
        s_phys_max (int): Maximum physical time for sampling.
        device (str): Device to place tensors on (default: 'cpu').
        seed (int, optional): Seed for random number generation.
    """
    def __init__(self, data_path, t_diff_max, s_phys_max, device, **kwargs):
        self.device = device
        position_data = torch.load(data_path+"_position", map_location=device)
        momentum_data = torch.load(data_path+"_momentum", map_location=device)
        trajectory = torch.cat([position_data, momentum_data], dim=-1)
        super().__init__(trajectory, t_diff_max, s_phys_max, device, **kwargs)


class TrajectoryADP():
    """
    Dataset class wrapper for positional and momenta trajectories with randomly selected initial conditions (IC).

    Args:
        data_path (str): Filepath to position data generated by MD2D.
        t_diff_max (int): Maximum diffusion time step.
        s_phys_max (int): Maximum physical time for sampling.
        device (str): Device to place tensors on (default: 'cpu').
        seed (int, optional): Seed for random number generation.
    """
    def __init__(self, data_path, t_diff_max, s_phys_max, device, **kwargs):
        self.device = device
        position_data = torch.load(data_path+"_position", map_location=device)
        momentum_data = torch.load(data_path+"_momentum", map_location=device)
        trajectory = torch.cat([position_data, momentum_data], dim=-1)