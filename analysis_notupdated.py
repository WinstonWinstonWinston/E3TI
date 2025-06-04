import torch
import matplotlib.pyplot as plt
import numpy as np

# Author: Winston Sullivan, Sarupria Group, University of Minnesota
# Date: 2025-03-28
# Description: This is meant to implement some plotting utility functions that are good for analyzing an ITO DDPM. Currently
#              the compute_expectation and the visualize_expectation_ functions dont work how I want them to. In fact they 
#              may not even work at all. The idea is there, but right now there is some jank with the scaling and the requirement
#              for a potential class from MD2D in order to compute the Hamiltonian. I think it is a cool idea but it needs some more
#              time to become a very crisp idea. I suspect it could be used to decipher when certain features manifest along a diffusion
#              trajectory at some physical time. I also suspect there will be useful info in the level sets of H(s,t).
#              If you find an error that is dire my contact info can be found winstonsullivan.netlify.app

class DDPMAnalysisTool:
    def __init__(self, model):
        """
        A tool for computing expectations over sampled reverse marginals in a DDPM model.
        
        Args:
            model: The DDPM model instance from ITO_DDPM.py.
        """
        self.model = model
        
    def compute_expectation(self, x_grid, s_grid, t_grid, function=lambda x: x):
        """
        Computes the expectation of a function applied to the sampled reverse marginal of the DDPM model.
        
        Args:
            x_grid (Tensor): Grid over the state space x_0_0, shape (X, N).
            s_grid (Tensor): Grid over the physics time s_phys, shape (S,).
            t_grid (Tensor): Grid over the diffusion time t_diff, shape (T,).
            function (callable): Function to apply to sampled states, output shape (X * S * T, M).

        Returns:
            Tensor: A 4D tensor of expected values, shape (X, S, T, M).
        """
        X, N = x_grid.shape
        S = s_grid.shape[0]
        T = t_grid.shape[0]
        
        # Expand grids to match the required shapes for batch processing
        x_0_0 = x_grid[:, None, None, :].expand(X, S, T, N).reshape(-1, N)
        s_phys = s_grid[None, :, None].expand(X, S, T).reshape(-1)
        t_diff = t_grid[None, None, :].expand(X, S, T).reshape(-1)
        
        # Sample from model
        samples = self.model.sample_reverse_marginal(x_0_0, t_diff, s_phys)  # Shape (X * S * T, N)
        
        # Apply function and reshape
        function_output = function(samples)  # Expected shape (X * S * T, M)
        M = function_output.shape[-1]  # Determine the output feature size
        expectation_values = function_output.view(X, S, T, M)
        
        return expectation_values
        

    @torch.no_grad()
    def visualize_loss_contour(self, x_0_0_grid, x_s_0_grid, step_s, step_t, levels=50, cmap='viridis', name='',repeats = 64):
        """
        Returns a contour plot figure of the loss function over s_phys and t_diff, averaged over x_0_0_grid.
        
        Args:
            x_0_0_grid (Tensor): Grid over the initial state x_0_0, shape (X, N).
            x_s_0_grid (Tensor): Grid over the sampled state x_s_0, shape (X, N).
            step_s (float): Step size for s_phys.
            step_t (float): Step size for t_diff.
            
        Returns:
            matplotlib.figure.Figure: The generated figure containing the plot.
        """
        s_values = torch.arange(0, self.model.s_phys_max, step_s, device=self.model.device)
        t_values = torch.arange(0, self.model.t_diff_max, step_t, device=self.model.device)
        
        S, T = len(s_values), len(t_values)
        s_grid, t_grid = torch.meshgrid(s_values, t_values, indexing='ij')
        
        # Expand grids for batch processing
        s_phys = s_grid.reshape(-1).repeat(int(x_0_0_grid.shape[0]/repeats), 1).flatten()
        t_diff = t_grid.reshape(-1).repeat(int(x_0_0_grid.shape[0]/repeats), 1).flatten()
        
        for r in range(repeats):
            x_0_0 = x_0_0_grid[int(r*(x_0_0_grid.shape[0]/repeats)):int((r+1)*(x_0_0_grid.shape[0]/repeats))].repeat(S * T, 1)
            x_s_0 = x_s_0_grid[int(r*(x_0_0_grid.shape[0]/repeats)):int((r+1)*(x_0_0_grid.shape[0]/repeats))].repeat(S * T, 1)
            if r != 0:
                # Compute loss over batch
                loss_values = self.model.loss(x_s_0, x_0_0, t_diff, s_phys, flatten=False)
                # Reshape into 4D tensor and average over initial configuration and data dimension
                loss_grid += loss_values.view(int(x_0_0_grid.shape[0]/repeats), S, T, loss_values.shape[-1]).mean(dim=0).mean(dim=-1)
            else:
                loss_values = self.model.loss(x_s_0, x_0_0, t_diff, s_phys, flatten=False)
                loss_grid = loss_values.view(int(x_0_0_grid.shape[0]/repeats), S, T, loss_values.shape[-1]).mean(dim=0).mean(dim=-1)
        loss_grid /= repeats
        
        # Detach tensors and move to CPU for plotting
        s_grid_np = s_grid.detach().cpu().numpy()
        t_grid_np = t_grid.detach().cpu().numpy()
        loss_grid_np = loss_grid.detach().cpu().numpy()
        
        # Create figure and contour plot
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(t_grid_np,s_grid_np, loss_grid_np, levels=levels, cmap=cmap, extend='both')
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('$|\epsilon -\epsilon_{Model}|^2(t_{Diff},s_{Phys})$')
        
        ax.set_ylabel("Physical Time $s_{Phys}$")
        ax.set_xlabel("Diffusion Time $t_{Diff}$")
        ax.set_title(f"Contour Plot of {name} Loss")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return fig

    @torch.no_grad()
    def visualize_loss_along_curve(self, x_0_0_grid, x_s_0_grid, curve_fn, z_values, name='', repeats = 64, **curve_kwargs):
        """
        Returns a figure of the loss as a function of a parameterized curve mapping z ∈ [0,1] to (t_diff, s_phys).
        
        Args:
            x_0_0_grid (Tensor): Grid over the initial state x_0_0, shape (X, N).
            x_s_0_grid (Tensor): Grid over the sampled state x_s_0, shape (X, N).
            curve_fn (callable): Function mapping z, **curve_kwargs -> (t_diff, s_phys).
            z_values (Tensor): 1D tensor of parameter values in [0,1].
            name (str): Optional name for labeling.
            **curve_kwargs: Arbitrary additional arguments passed to curve_fn.
            
        Returns:
            matplotlib.figure.Figure: The generated figure containing the plot.
        """
        t_diff, s_phys = curve_fn(z_values, **curve_kwargs)
        
        # Repeat grids to match z_values dimension
        x_0_0 = x_0_0_grid.repeat(len(z_values), 1)
        x_s_0 = x_s_0_grid.repeat(len(z_values), 1)

        # Ensure t_diff and s_phys broadcast correctly
        t_diff = t_diff.repeat(int(x_0_0_grid.shape[0]/repeats))
        s_phys = s_phys.repeat(int(x_0_0_grid.shape[0]/repeats))
        
        for r in range(repeats):
            x_0_0 = x_0_0_grid[int(r*(x_0_0_grid.shape[0]/repeats)):int((r+1)*(x_0_0_grid.shape[0]/repeats))].repeat(len(z_values), 1)
            x_s_0 = x_s_0_grid[int(r*(x_0_0_grid.shape[0]/repeats)):int((r+1)*(x_0_0_grid.shape[0]/repeats))].repeat(len(z_values), 1)
            if r != 0:
                loss_values = self.model.loss(x_s_0, x_0_0, t_diff, s_phys, flatten=False)
                loss_mean += loss_values.view(int(x_0_0_grid.shape[0]/repeats), len(z_values),loss_values.shape[-1]).mean(dim=0).mean(dim=-1)
            else:
                loss_values = self.model.loss(x_s_0, x_0_0, t_diff, s_phys, flatten=False)
                loss_mean = loss_values.view(int(x_0_0_grid.shape[0]/repeats), len(z_values),loss_values.shape[-1]).mean(dim=0).mean(dim=-1)

        loss_mean /= repeats
    
        # Detach and move to CPU for plotting
        z_values_np = z_values.detach().cpu().numpy()
        loss_mean_np = loss_mean.detach().cpu().numpy()
    
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(z_values_np, loss_mean_np, label=f"{name} Loss", color='b')
        ax.set_xlabel("z (Curve Parameter)")
        ax.set_ylabel("Loss")
        ax.set_title(f"{name} Loss Along Parameterized Curve")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    
        return fig

    def visualize_expectation_contour(self, x_0_0_grid, step_s, step_t, levels=50, cmap='viridis', function=lambda x: x, name='Expectation'):
        """
        Visualizes a contour plot of the function of the sampled states over s_phys and t_diff.
        
        Args:
            x_0_0_grid (Tensor): Grid over the initial state x_0_0, shape (X, N).
            step_s (float): Step size for s_phys.
            step_t (float): Step size for t_diff.
            levels (int): Number of contour levels.
            cmap (str): Colormap for visualization.
            name (str): Name for the plot title.
            function (callable): Function to apply to sampled states must return scalar, output shape (X * S * T, 1).
            
        Returns:
            matplotlib.figure.Figure: The generated figure containing the plot.
        """
        s_values = torch.arange(0, self.model.s_phys_max, step_s)
        t_values = torch.arange(0, self.model.t_diff_max, step_t)
        
        S, T = len(s_values), len(t_values)
        mean_values = self.compute_expectation(x_0_0_grid, s_values, t_values, function)
        mean_grid = mean_values.squeeze(-1).mean(dim=0)  # Reduce M dimension and average over X
        
        # Detach tensors and move to CPU for plotting
        s_grid, t_grid = torch.meshgrid(s_values, t_values, indexing='ij')
        s_grid_np = s_grid.detach().cpu().numpy()
        t_grid_np = t_grid.detach().cpu().numpy()
        mean_grid_np = mean_grid.detach().cpu().numpy()
        
        # Plot contour
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = plt.contourf(s_grid_np, t_grid_np, mean_grid_np, levels=levels, cmap=cmap, extend='both')
        ax.colorbar(contour, label=f'Mean of Samples')
        ax.set_ylabel("Physical Time $s_{Phys}$")
        ax.set_xlabel("Diffusion Time $t_{Diff}$")
        ax.set_title(f"{name} Contour Plot of {str(function)}")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return fig
        
    def visualize_expectation_along_curve(self, x_0_0_grid, curve_fn, z_values, function=lambda x: x, name='Expectation', **curve_kwargs):
        """
        Visualizes the function as a parameterized curve mapping z ∈ [0,1] to (t_diff, s_phys).
        
        Args:
            x_0_0_grid (Tensor): Grid over the initial state x_0_0, shape (X, N).
            curve_fn (callable): Function mapping z -> (t_diff, s_phys).
            z_values (Tensor): 1D tensor of parameter values in [0,1].
            function (callable): Function to apply to sampled states, must return scalar, output shape (X * S * T, 1).
            name (str): Name for the plot title.
            **curve_kwargs: Arbitrary additional arguments passed to curve_fn.
            
        Returns:
            matplotlib.figure.Figure: The generated figure containing the plot.
        """
        t_diff, s_phys = curve_fn(z_values,**curve_kwargs)
        x_0_0 = x_0_0_grid.repeat(len(z_values), 1)
        
        # Compute expectation over batch
        expectation_values = self.compute_expectation(x_0_0, s_phys, t_diff, function)
        expectation_mean = expectation_values.squeeze(-1).mean(dim=0)  # Reduce M dimension and average over X
        
        # Detach and move to CPU for plotting
        z_values_np = z_values.detach().cpu().numpy()
        expectation_mean_np = expectation_mean.detach().cpu().numpy()
        
        # Plot expectation along curve
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(z_values_np, expectation_mean_np, color='b')
        ax.set_xlabel("z (Curve Parameter)")
        ax.set_ylabel(f"{str(function)}")
        ax.set_title(f"{name} {str(function)} Along Parameterized Curve")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        return fig

class Observables:
    """
    A toolbox of standalone methods for computing observables from sampled data.
    These methods do not interact with instance attributes and are purely functional.
    """
    
    @staticmethod
    def kinetic_energy(coordinates: torch.Tensor, mass: float):
        """
        Computes the kinetic energy of a batch of momenta.
        
        Args:
            coordinates (Tensor): Input coordinates, shape (X, N), where N is even.
            mass (float): Mass of the particles.
        
        Returns:
            Tensor: Kinetic energy values for each sample, shape (X,).
        """
        X, N = coordinates.shape
        assert N % 2 == 0, "The number of coordinates (N) must be even to separate position and momentum."
        momentum = coordinates[:, N//2:]
        return (0.5 / mass) * (momentum ** 2).sum(dim=-1)
    
    @staticmethod
    def potential_energy(coordinates: torch.Tensor, potential: callable):
        """
        Computes the potential energy of a batch of positions using the provided potential function.
        
        Args:
            coordinates (Tensor): Input coordinates, shape (X, N), where N is even.
            potential (callable): A function that maps positions to their potential energy batchwise.
        
        Returns:
            Tensor: Potential energy values for each sample, shape (X,).
        """
        X, N = coordinates.shape
        assert N % 2 == 0, "The number of coordinates (N) must be even to separate position and momentum."
        position = coordinates[:, :N//2]
        return potential(position)
    
    @staticmethod
    def energy(coordinates: torch.Tensor, potential: callable, mass: float = 1.0):
        """
        Computes the total energy of a batch of coordinates using the provided potential function.
        The first half of the coordinates represent position, and the last half represent momentum.
        
        Args:
            coordinates (Tensor): Input coordinates, shape (X, N), where N is even.
            potential (callable): A function that maps positions to their potential energy batchwise.
            mass (float): Mass of the particles.
        
        Returns:
            Tensor: Energy values for each sample, shape (X,).
        """
        X, N = coordinates.shape
        assert N % 2 == 0, "The number of coordinates (N) must be even to separate position and momentum."
        
        position, momentum = coordinates[:, :N//2], coordinates[:, N//2:]
        kinetic = Observables.kinetic_energy(momentum, mass)
        potential = Observables.potential_energy(position, potential)
        
        return kinetic + potential

class ParametricCurves:
    def __init__(self, t_diff_max, s_phys_max):
        """
        Args:
            t_diff_max (int): Maximum integer value for t_diff.
            s_phys_max (int): Maximum integer value for s_phys.
        """
        self.t_diff_max = t_diff_max-1
        self.s_phys_max = s_phys_max-1

    def _clamp_round(self, t_diff, s_phys):
        """Helper function to round and clamp values."""
        t_diff = torch.clamp(t_diff.round().int(), 0, self.t_diff_max)
        s_phys = torch.clamp(s_phys.round().int(), 0, self.s_phys_max)
        return t_diff, s_phys

    def diagonal_line(self, z_values):
        """Diagonal line from (0,0) to (t_diff_max, s_phys_max)."""
        t_diff = z_values * self.t_diff_max
        s_phys = z_values * self.s_phys_max
        return self._clamp_round(t_diff, s_phys)

    def edges(self, z_values):
        """Traces the edges of the (t_diff_max, s_phys_max) space."""
        t_diff = torch.cat((z_values * self.t_diff_max, torch.ones_like(z_values) * self.t_diff_max))
        s_phys = torch.cat((torch.zeros_like(z_values), z_values * self.s_phys_max))
        return self._clamp_round(t_diff, s_phys)

    def sinusoid_diagonal(self, z_values):
        """Sinusoid oscillating along the diagonal."""
        t_diff = z_values * self.t_diff_max
        s_phys = (torch.sin(z_values * torch.pi * 2) * 0.5 + 0.5) * self.s_phys_max
        return self._clamp_round(t_diff, s_phys)

    def fixed_t_diff(self, z_values, t_value = 0):
        """Horizontal line at a fixed t_diff value."""
        t_diff = torch.full_like(z_values, t_value)
        s_phys = z_values * self.s_phys_max
        return self._clamp_round(t_diff, s_phys)

    def fixed_s_phys(self, z_values, s_value = 0):
        """Vertical line at a fixed s_phys value."""
        t_diff = z_values * self.t_diff_max
        s_phys = torch.full_like(z_values, s_value)
        return self._clamp_round(t_diff, s_phys)


def plot_diffusion_paths(model, dataset, args, var, s_phys):
    """
    A really jank plotting function that shows the distribution of var wrt to the data and then wrt to the DDPM.
    It will also plot a forward and backward path in diffusion time of var. This method assumes you have at least 
    25000 trajectories. 
    """
    with torch.no_grad():
        sample_count=50

        forward_time_trajectories = model.sample_forward_trajectory(dataset.scale(dataset.trajectory[s_phys,:sample_count,:] -dataset.trajectory[0,:sample_count,:] ,s_phys), s_phys*torch.ones(sample_count,device=args.device,dtype=torch.long)).detach().cpu()
        reverse_time_trajectories = model.sample_reverse_trajectory(dataset.trajectory[0,:sample_count,:], s_phys*torch.ones(sample_count,device=args.device,dtype=torch.long)).detach().cpu()
        
        
        forward_path_fig = plt.figure(figsize=(8, 5))
        for i in range(sample_count):
            plt.plot(forward_time_trajectories[:, i, var],
                     alpha=0.5, linewidth=0.2,
                     color='k')
        plt.title("Forward Trajectories, a.k.a. the Noising Process $s_{Phys} = $ "+ str(s_phys))
        plt.ylabel("$x_t$")
        plt.xlabel("Diffusion Time")
        plt.ylim(-6,6)
        
        reverse_path_fig =plt.figure(figsize=(8, 5))
        for i in range(sample_count):
            plt.plot(reverse_time_trajectories[:, i, var],
                     alpha=0.5, linewidth=0.2,
                     color='k')
        plt.title("Reverse Trajectories, a.k.a. the Denoising Process $s_{Phys} = $ "+ str(s_phys))
        plt.ylabel("$x_t$")
        plt.xlabel("Diffusion Time")
        plt.gca().invert_xaxis()
        plt.ylim(-6,6)

        sample_count = 6400

        forward_marginal = model.sample_forward_marginal(dataset.trajectory[0,:sample_count,:], (args.t_diff_max-1)*torch.ones(sample_count,device=args.device,dtype=torch.long),s_phys*torch.ones(sample_count,device=args.device,dtype=torch.long)).detach().cpu()
        
        forward_marginal_fig = plt.figure(figsize=(8,5))
        counts, bins, _ = plt.hist(forward_marginal[:,var], bins=100, color='blue', alpha=0.7,density=True,label='Data Distribution')
        x = torch.linspace(bins[0], bins[-1], 100)
        pdf = (1 / torch.sqrt(torch.tensor(2 * np.pi))) * torch.exp(-0.5 * x ** 2)
        plt.plot(x.numpy(), pdf.numpy(), 'r-', lw=2, label='Standard Normal PDF')
        plt.title("Histogram of final time $x_t$ for $s_{Phys} = $ "+ str(s_phys))
        plt.xlabel("Value")
        plt.ylabel(f"p({str(var)})")
        plt.legend()
    
        sample_count=6400
    
        reverse_marginal =  model.sample_reverse_marginal(dataset.trajectory[0,:sample_count,:], (args.t_diff_max-1)*torch.ones(sample_count,device=args.device,dtype=torch.long),s_phys*torch.ones(sample_count,device=args.device,dtype=torch.long)).detach().cpu()

        bins = np.linspace(-4,4,100)
        
        reverse_marginal_fig =  plt.figure(figsize=(8,5))
        counts, bins, _ = plt.hist(reverse_marginal[:,var].detach().cpu(), bins=bins, alpha=0.7,density=True,label='DDPM Distribution')
        counts, bins, _ = plt.hist(dataset.scale(dataset.trajectory[s_phys,:sample_count,:] - dataset.trajectory[0,:sample_count,:] ,s_phys)[:,var].detach().cpu(), bins=bins, alpha=0.7,density=True,label='Data Distribution')
    
        plt.title("Histogram of final time $x_t$ for $s_{Phys} = $ "+ str(s_phys))
        plt.xlabel("Value")
        plt.ylabel(f"p({str(var)})")
        plt.legend()

        return forward_path_fig, reverse_path_fig, forward_marginal_fig, reverse_marginal_fig