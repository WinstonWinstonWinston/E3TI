# Torch Imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Metric tracking
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Argument reading
import argparse
import os

# Custom functions
import ITO_DDPM
from analysis import * 
from analysis import plot_diffusion_paths
from utils import * 

import gc

# Author: Winston Sullivan, Sarupria Group, University of Minnesota
# Last Updated: 2025-05-27
# Description: This implements a train method for an ITO DDPM. If utils is configured correctly then it should be fairly generic.
#              If you find an error that is dire my contact info can be found winstonsullivan.netlify.app
# Notes: Man I really dislike this implementation. Thinking about coding is way more fun than actually doing it. Im gonna wing it in a notebook first to see if im on a fools errand. 

def train(args):
    """
    Trains the ITO_DDPM model for a specified args.
    """
    # Initialize Wandb
    wandb.init(project=args.name, config=vars(args))

    device = args.device

    print("Loading dataset...")
    dataset_args = {
        k: smart_cast(v)
        for arg in args.dataset_args if '=' in arg 
        for k, v in [arg.split('=', 1)]  
    }

    dataset = get_dataset(args.dataset, args.data_path, args,**dataset_args)

    print("Initializing model...")
    
    model = get_architecture(args, dataset.trajectory.to(device)).to(device) # Use trajectories for scaling 
    analysis_tool = DDPMAnalysisTool(model)

    print(model)

    print("Initializing optimizer...")
    optimizer_args = {
        k: smart_cast(v)
        for arg in args.optimizer_args if '=' in arg 
        for k, v in [arg.split('=', 1)]  
    }
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, **optimizer_args)

    print("Initializing scheduler...")
    scheduler_args = {
        k: smart_cast(v)
        for arg in args.scheduler_args if '=' in arg 
        for k, v in [arg.split('=', 1)]  
    }
    scheduler = get_scheduler(args.scheduler, optimizer, **scheduler_args)

    # See if we already trained a bit
    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path)
        print(f"Resuming from epoch {start_epoch}")

    # Plotting Parameters
    STEP_S = 10
    STEP_T = 10
    curves = ParametricCurves(model.t_diff_max, model.s_phys_max)
    z_values = torch.linspace(0,1,100,device = args.device)
    sample_count = 512
    repeats = 64

    print("Starting training loop...")
    wandb_step = 0
    for epoch in range(start_epoch, args.epochs):
        loader = tqdm(range(0, args.epoch_size, args.batch_size), desc=f"Epoch {epoch+1}/{args.epochs}", leave=True, ncols=120, unit=' batch')
        for step in loader:

            batch = dataset.getitems(args.batch_size)
            x_0_0 = batch['x_0_0'] # Initial condition inferred vector features
            x_s_0 = batch['x_s_0'] # Final condition inferred vector features
            v_0_0 = batch['v_0_0'] # Initial auxillary vector features
            s_0_0 = batch['s_0_0'] # Initial auxillary scalar features
            t_diff = batch['t_diff'] # Diffusion time to train on (determines noise)
            s_phys = batch['s_phys'] # Physical time
            
            optimizer.zero_grad()
            
            loss = model.loss(x_s_t, x_0_0, v_0_0, s_0_0, t_diff, s_phys, flatten=True)
            loss.backward()


            if args.grad_clipping != 0:
                # Needs to be inspected. Does not work as intended.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clipping)

            optimizer.step()
            
            loader.set_postfix(batch=f"{int((step+1)/(args.batch_size))}/{int(args.epoch_size/args.batch_size)} ", loss=f"{loss.item():.6f}")
            
            if wandb_step % args.log_interval == 0:
                log_metrics(wandb_step, loss.item(), optimizer, model)
            
            if wandb_step % args.plot_interval == 0:
                
                # Plot Loss
                loss_contour = analysis_tool.visualize_loss_contour(x_0_0[:int(sample_count*repeats)], x_s_0[:int(sample_count*repeats)], step_s=STEP_S, step_t=STEP_T,repeats=repeats)
                loss_1_t = analysis_tool.visualize_loss_along_curve(x_0_0[:int(sample_count*repeats)], x_s_0[:int(sample_count*repeats)], curves.fixed_s_phys, z_values, name='s = 1',s_value = 1,repeats=repeats)
                loss_s_0 = analysis_tool.visualize_loss_along_curve(x_0_0[:int(sample_count*repeats)], x_s_0[:int(sample_count*repeats)], curves.fixed_t_diff, z_values, name='t = 0',repeats=repeats)

                # Plot paths
                forward_path_fig_x_10, reverse_path_fig_x_10, forward_marginal_fig_x_10, reverse_marginal_fig_x_10 =  plot_diffusion_paths(model,dataset, args ,0, 10)
                forward_path_fig_y_10, reverse_path_fig_y_10, forward_marginal_fig_y_10, reverse_marginal_fig_y_10 =  plot_diffusion_paths(model,dataset, args ,1, 10)
                forward_path_fig_x_499, reverse_path_fig_x_499, forward_marginal_fig_x_499, reverse_marginal_fig_x_499 =  plot_diffusion_paths(model,dataset, args ,0, 499)
                forward_path_fig_y_499, reverse_path_fig_y_499, forward_marginal_fig_y_499, reverse_marginal_fig_y_499 =  plot_diffusion_paths(model,dataset, args ,1, 499)
               
                # Save Plots to wandb
                wandb.log({"loss_contour": wandb.Image(loss_contour)}, step=wandb_step)
                wandb.log({"loss_1_t": wandb.Image(loss_1_t)}, step=wandb_step)
                wandb.log({"loss_s_0": wandb.Image(loss_s_0)}, step=wandb_step)

                wandb.log({"reverse_marginal_fig_x_10": wandb.Image(reverse_marginal_fig_x_10)}, step=wandb_step)
                wandb.log({"reverse_marginal_fig_x_499": wandb.Image(reverse_marginal_fig_x_499)}, step=wandb_step)
                wandb.log({"reverse_marginal_fig_y_10": wandb.Image(reverse_marginal_fig_y_10)}, step=wandb_step)
                wandb.log({"reverse_marginal_fig_y_499": wandb.Image(reverse_marginal_fig_y_499)}, step=wandb_step)

                # Close Plots 
                plt.close('all')

                # save model everytime a summary set of plots in made.
                save_checkpoint(args.name, model, optimizer, epoch+1, args.checkpoint_path)

            wandb_step += 1
            scheduler.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

    print("Training completed.")
    return model,dataset,args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required Args
    parser.add_argument('--name', type=str, required=True, help='Name of this run')
    parser.add_argument('--device', type=str, required=True, help='Device to run on')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='TrajectoryPositionMomentum', help='Dataset class name')
    parser.add_argument('--dataset_args', nargs='*', default=[], help='Extra dataset args in key=value format')
    parser.add_argument('--data_path', type=str, default='datasets/MullerBrown/FixedIC', help='Path to dataset')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=2**10, help='Number of training epochs')
    parser.add_argument('--epoch_size', type=int, default=2**22, help='Number of samples in each epoch')
    parser.add_argument('--batch_size', type=int, default=2**20, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.04, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer choice')
    parser.add_argument('--optimizer_args', nargs='*', default=[], help='Optimizer arguments in key=value format')
    parser.add_argument('--scheduler', type=str, default='ExponentialLR', help='Scheduler choice')
    parser.add_argument('--scheduler_args', nargs='*', default=["gamma=0.9999"], help='Scheduler arguments in key=value format')
    parser.add_argument('--grad_clipping', type=float, default=0.005, help='max_norm of the grad clipping. If 0 clipping is disabled.')
    
    # DDPM parameters
    parser.add_argument('--architecture', type=str, default='painn', help='Architecture choice')
    parser.add_argument('--architecture_args', nargs='*', default=[], help='Architecture arguments in key=value format')
    parser.add_argument('--noise_schedule', type=str, default='linear', help='Noise schedule choice')
    parser.add_argument('--noise_schedule_args', nargs='*', default=[], help='Noise schedule arguments in key=value format')
    parser.add_argument('--s_phys_max', type=int, default=500, help='Maximum physical time step')
    parser.add_argument('--t_diff_max', type=int, default=1000, help='Maximum diffusion steps')
    parser.add_argument('--scaled', type=bool, default=True, help='To scale or not to scale')
    
    # Saving parameters
    parser.add_argument('--checkpoint_path', type=str, default='models/MullerBrown/FixedIC/checkpoint.pth', help='Checkpoint path')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--log_interval', type=int, default=8, help='Number of steps between logging')
    parser.add_argument('--plot_interval', type=int, default=512, help='Number of steps between loss contour plots')
    
    args = parser.parse_args()
    train(args)