import torch
import torch.nn as nn
import torch.optim as optim
import os
import wandb
from utils import *
from ITO_DDPM import ImplicitTransferOperatorDDPM,BaseNoiseSchedule, SigmoidNoiseSchedule, LinearNoiseSchedule, CosineNoiseSchedule, QuadraticNoiseSchedule
from datasets.dataset import TrajectoryPositionMomentum
from datasets.dataset import TrajectoryPositionMomentum
from architechtures.LangevinMLP import LangevinMLP 
from architechtures import embeddings

# Author: Winston Sullivan, Sarupria Group, University of Minnesota
# Date: 2025-03-28
# Description: This is a file that stores many helper functions for training ITO DDPM models.
#              If you find an error that is dire my contact info can be found winstonsullivan.netlify.app


def get_optimizer(optimizer_name, model_params, lr, **optimizer_args):
    """
    Returns an optimizer based on the given name and parameters.
    """
    optimizers = {
        "adam": optim.Adam,
        "adamw": optim.AdamW,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop
    }
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    optimizer_args = {"betas": (0.9, 0.999),  # Coefficients for computing running averages of gradient and its square
    "eps": 1e-8,  # Term added to the denominator to improve numerical stability
    "weight_decay": 0.0,  # L2 penalty (regularization term)
    "amsgrad": False,  # Whether to use the AMSGrad variant of Adam
   }
    return optimizers[optimizer_name.lower()](model_params, lr=lr, **optimizer_args)

def get_scheduler(scheduler_name, optimizer, **scheduler_args):
    """
    Returns a learning rate scheduler based on the given name and parameters.
    """
    schedulers = {
        "step": optim.lr_scheduler.StepLR,
        "exponential": optim.lr_scheduler.ExponentialLR,
        "cosine": optim.lr_scheduler.CosineAnnealingLR
    }
    if scheduler_name.lower() not in schedulers:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return schedulers[scheduler_name.lower()](optimizer, **scheduler_args)

def get_architecture(args, x_reference=None):
    """
    Returns an architechture
    """
    architectures = {
        "mlp": LangevinMLP,
    }

    noise_schedules = {
        "sigmoid": SigmoidNoiseSchedule,
        "linear": LinearNoiseSchedule,
        "cosine": CosineNoiseSchedule,
        "quadratic": QuadraticNoiseSchedule,
    }

    if args.architecture.lower() not in architectures:
        raise ValueError(f"Unsupported architecture: {args.architecture}")

    if args.noise_schedule.lower() not in noise_schedules:
        raise ValueError(f"Unsupported noise schedule: {args.noise_schedule}")

    # Read args and create noise schedulerparser = argparse.ArgumentParser()
    noise_schedule_args = {
        k: smart_cast(v)
        for arg in args.noise_schedule_args if '=' in arg 
        for k, v in [arg.split('=', 1)]  
    }

    selected_noise_schedule = noise_schedules[args.noise_schedule.lower()](**noise_schedule_args)

    # Read args and create architechture
    # ToDo: Make this toggleable, currently only MLP with fixed things.
    architecture_args = {
        k: smart_cast(v)
        for arg in args.architecture_args if '=' in arg 
        for k, v in [arg.split('=', 1)]  
    }

    num_layers = 4
    layer_width= 64
    layers = [layer_width] * num_layers
    
    s_phys_embedding = embeddings.SinCosTimeEmbedding(max_t=args.s_phys_max, init_scale=1.0, dim=architecture_args['embed_dim'],learnable_scale=architecture_args['learnable_scale'],device = args.device)
    t_diff_embedding = embeddings.SinCosTimeEmbedding(max_t=args.t_diff_max, init_scale=1.0, dim=architecture_args['embed_dim'],learnable_scale=architecture_args['learnable_scale'],device = args.device)
    selected_architecture = architectures[args.architecture.lower()](layers, architecture_args['data_dim'], t_diff_embedding, s_phys_embedding, activation=nn.ReLU(), device=args.device)
    
    if smart_cast(args.scaled):
        if x_reference is None:
            raise ValueError("x_reference tensor must be provided if scaled=True")
        return DDPMWithScaler(
        architechture=selected_architecture,
        noise_schedule=selected_noise_schedule,
        t_diff_max=args.t_diff_max,
        s_phys_max=args.s_phys_max,
        device=args.device, x_reference=x_reference)
        

    return ImplicitTransferOperatorDDPM(
        architechture=selected_architecture,
        noise_schedule=selected_noise_schedule,
        t_diff_max=args.t_diff_max,
        s_phys_max=args.s_phys_max,
        device=args.device)

def smart_cast(value):
    """Convert value to int, float, bool, or keep as string."""
    if value.lower() in {'true', 'false'}:  # Handle booleans
        return value.lower() == 'true'  # Converts 'true' → True, 'false' → False
    elif value.replace('-', '', 1).isdigit():  # Convert integers
        return int(value)
    elif value.replace('.', '', 1).replace('-', '', 1).isdigit():  # Convert floats
        return float(value)
    return value  # Keep as string if none of the above

def get_dataset(dataset_name, data_path, args,**kwargs):
    """
    Returns a dataset instance based on the given name and parameters.
    """
    datasets = {
        "trajectorypositionmomentum": TrajectoryPositionMomentum,
        # add more datasets as needed
    }
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower not in datasets:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(datasets.keys())}")
    
    return datasets[dataset_name_lower](data_path, args.t_diff_max, args.s_phys_max, **kwargs)

def log_metrics(step, loss, optimizer, model):
    """
    Logs training metrics to Weights & Biases (wandb) during training steps.
    """
    wandb.log({
        "train/loss": loss,
        "train/lr": optimizer.param_groups[0]["lr"],
        "train/grad_norm": torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None])),
    }, step=step)
    
    all_weights = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
    all_grads = torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None])
    
    wandb.log({
        "weights/all": wandb.Histogram(all_weights.cpu().numpy()),
        "gradients/all": wandb.Histogram(all_grads.cpu().numpy()),
    }, step=step)

    # Uncomment me to get specific weights of the network
    # for name, param in model.named_parameters():
    #     wandb.log({
    #         f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy()),
    #         f"gradients/{name}": wandb.Histogram(param.grad.detach().cpu().numpy()) if param.grad is not None else None
    #     }, step=step)

def save_checkpoint(name, model, optimizer, epoch, checkpoint_path):
    """Save model and optimizer state to a checkpoint file."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    os.makedirs(checkpoint_path, exist_ok=True)  # Ensure the checkpoint directory exists
    checkpoint_file = os.path.join(checkpoint_path, f"{name}_{epoch}_checkpoint.pth")
    torch.save(checkpoint, checkpoint_file)
    print(f"Checkpoint saved: {checkpoint_file}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model and optimizer state from a checkpoint file."""
    if not os.path.isfile(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")

    return epoch  # Return the last saved epoch
