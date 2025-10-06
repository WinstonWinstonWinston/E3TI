import math
import pytorch_lightning as pl
import torch
from torch import nn
from torch_scatter import scatter
from tqdm import tqdm
from ito.model import beta_schedule, ema, dpm_solve
import wandb
from scipy.spatial.transform import Rotation as R
import os
import numpy as np
import copy
from hydra.utils import get_class


class DDPMBase(pl.LightningModule):
    def __init__(
        self,
        score_model_class,
        score_model_kwargs,
        diffusion_steps=1000,
        lr=1e-3,
        beta_scheduler=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if beta_scheduler is None:
            beta_min = 1e-4
            beta_max = 0.02
            beta_scheduler = beta_schedule.SigmoidalBetaScheduler(
                diffusion_steps, beta_min, beta_max
            )

        self.beta_scheduler = beta_scheduler
#         self.score_model = score_model_class

        # 3) Resolve class if a string path was given
        if isinstance(score_model_class, str):
            score_model_class = get_class(score_model_class)

        # 4) Instantiate the score model here
        self.score_model = score_model_class(**(score_model_kwargs or {}))


        self.register_buffer("betas", self.beta_scheduler.get_betas())
        self.register_buffer("alphas", self.beta_scheduler.get_alphas())
        self.register_buffer("alpha_bars", self.beta_scheduler.get_alpha_bars())

        self.diffusion_steps = diffusion_steps
        self.lr = lr

        self.ema = ema.ExponentialMovingAverage(
            self.score_model.parameters(), decay=0.99
        )


    def forward(self, *args):
        score = self.score_model(*args)
        return score

    def training_step(self, batch, _):
        loss, noise_batch, epsilon, epsilon_hat, loss_supp, batch_predy = self.get_loss(batch)
        self.last_batch = batch["batch_t"]
        self.noise_batch = noise_batch
        self.epsilon = epsilon
        self.epsilon_hat = epsilon_hat
        self.loss_supp = loss_supp
        self.batch_predy = batch_predy
        
#         batch_size = batch['batch_0'].batch[-1] + 1  # Number of graphs in the batch (specific to PyG)
        self.batch_size = batch['batch_0'].num_graphs  # Number of graphs in the batch (specific to PyG)
        self.log("train/loss", loss, batch_size = self.batch_size)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size = self.batch_size)
        wandb.log({"train_loss": loss})
        wandb.log({"loss_supp": loss_supp})

        # Log parameters and gradients; Added by Praveen on 12/17/2024
        for name, param in self.named_parameters():
#             self.log(f'param/{name}', param.mean(), prog_bar=True)
            wandb.log({f"param/{name}": param.mean()})

            if param.grad is not None:
#                 self.log(f'grad/{name}', param.grad.norm(), prog_bar=True)
                wandb.log({f"grad/{name}": param.grad.norm()})

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        return loss
    
    def validation_step(self, batch, _):
        loss, noise_batch, epsilon, epsilon_hat, loss_supp, batch_predy = self.get_loss(batch)
#         print(f"Validation Step - Batch Index: {batch.batch}, Loss: {loss.item()}")
        self.last_batch = batch["batch_t"]
        self.noise_batch = noise_batch
        self.epsilon = epsilon
        self.epsilon_hat = epsilon_hat
        self.loss_supp = loss_supp
        self.batch_predy = batch_predy
        self.batch_size = batch["batch_t"].num_graphs
        self.log("val/loss", loss, batch_size = self.batch_size)

        # Store loss for later averaging
        if not hasattr(self, "val_losses"):
            self.val_losses = []
        self.val_losses.append(loss.detach())  # Detach to avoid memory leaks

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, batch_size = self.batch_size)
        self.log("val/loss", loss, batch_size = self.batch_size)
        wandb.log({"val_loss": loss})
#         wandb.log({"val_loss_supp": loss_supp})

        if torch.isnan(loss):
            raise ValueError("Val Loss is NaN")

        return loss

    def on_train_epoch_end(self):
        # Retrieve the logged training loss for the epoch
        epoch_loss = self.trainer.callback_metrics.get("train_loss")
    #         print(f"Epoch {self.current_epoch} Loss: {epoch_loss:.6f}")
        if self.last_batch is not None:
            self.last_batch = self.last_batch.to(self.device)
            self.epsilon_hat = self.epsilon_hat.to(self.device)

            # Print epoch loss
            print(f"\nEpoch {self.current_epoch} Loss - ITO: {epoch_loss:.6f}")

            # Print last batch outputs (only first 5 values for readability)
            print("Ground Truth (batch.y):", self.epsilon.x[:5].detach().cpu().numpy())
            print("Predicted Output (epsilon_hat.y):", self.epsilon_hat.x[:5].detach().cpu().numpy())
        
        ## Store predicted coordinates at the end of fifth epoch
#         if self.current_epoch % 10 == 0:
        if self.current_epoch == 10:
            output_dir = os.path.join("outputs/train/vmd", "train_predictions")
            print('Epoch 10 predictions stored in: ', output_dir, ' with shape:', self.batch_predy.cpu().numpy().shape)
            os.makedirs(output_dir, exist_ok=True)  # Create directory if not exists
            np.save(os.path.join(output_dir, f"ito_train_predy_epoch{self.current_epoch}.npy"), self.batch_predy.cpu().numpy())

    def on_validation_epoch_end(self):
        
        print(f"Epoch {self.current_epoch} - Validation Mode: {self.training}")  
        if hasattr(self, "val_losses") and len(self.val_losses) > 0:
            avg_val_loss = torch.stack(self.val_losses).mean()
            self.log("val_loss", avg_val_loss, prog_bar=True, logger=True, batch_size = self.batch_size)
            wandb.log({"val_loss": avg_val_loss})

            # Print validation loss at the end of each epoch
            print(f"\nEpoch {self.current_epoch} Validation Loss - ITO: {avg_val_loss:.6f}")

            # Clear stored losses for the next epoch
            self.val_losses.clear()
#         if self.current_epoch % 10 == 0:
        if self.current_epoch == 10:
            output_dir = os.path.join("outputs/train/vmd", "val_predictions")
            print('Epoch 10 predictions stored in: ', output_dir, ' with shape:', self.batch_predy.cpu().numpy().shape)
            os.makedirs(output_dir, exist_ok=True)  # Create directory if not exists
            np.save(os.path.join(output_dir, f"ito_val_predy_epoch{self.current_epoch}.npy"), self.batch_predy.cpu().numpy())

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.score_model.parameters())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "train/loss",
            },
        }

    def _ode_sample(self, batch, forward_callback, ode_steps=100):
        ns = dpm_solve.NoiseScheduleVP(
            "discrete",
            betas=self.betas,
        )

        def t_diff_and_forward(x, t):
            t = t[0]
            batch.t_diff = torch.ones_like(batch.batch) * t
            batch.x = x
            epsilon_hat = forward_callback(batch)
            return epsilon_hat.x

        wrapped_model = dpm_solve.model_wrapper(t_diff_and_forward, ns)
        dpm_solver = dpm_solve.DPM_Solver(wrapped_model, ns)
        batch.x = dpm_solver.sample(batch.x, ode_steps)
        return batch

    def _sample(self, batch, forward_callback, save_intermediate_steps):
        batch.x = torch.randn_like(batch.x, device=self.device)
        intermediate_trajectories = {}
        with torch.no_grad():
            for t in tqdm(torch.arange(self.diffusion_steps - 1, 0, -1)):
                batch.t_diff = torch.ones_like(batch.batch) * t
                epsilon_hat = forward_callback(batch)
                batch.x = self.denoise_sample(t, batch.x, epsilon_hat)
                # Save intermediate trajectory if t is in save_intermediate_steps
                if save_intermediate_steps and t.item() in save_intermediate_steps:
                    # Deepcopy to avoid mutation in next steps
                    intermediate_trajectories[t.item()] = copy.deepcopy(batch)

        return batch, intermediate_trajectories

    def denoise_sample(self, t, x, epsilon_hat):
        epsilon = torch.randn(x.shape).to(device=x.device)
        preepsilon_scale = 1 / math.sqrt(self.alphas[t])
        epsilon_scale = (1 - self.alphas[t]) / math.sqrt(1 - self.alpha_bars[t])
        post_sigma = math.sqrt(self.betas[t]) * epsilon
        x = preepsilon_scale * (x - epsilon_scale * epsilon_hat.x) + post_sigma

        return x

    def get_noise_img_and_epsilon(self, batch):
        ts = torch.randint(1, self.diffusion_steps, [len(batch)], device=self.device)

        epsilon = self.get_epsilon(batch)

        alpha_bars = self.alpha_bars[ts]
        noise_batch = batch.clone()

        alpha_bars = alpha_bars[batch.batch]

        noise_batch.x = (
            torch.sqrt(alpha_bars) * batch.x.T
            + torch.sqrt(1 - alpha_bars) * epsilon.x.T
        ).T

        noise_batch.t_diff = ts[batch.batch]

        return noise_batch, epsilon, alpha_bars

    def get_epsilon(self, batch):
        epsilon = batch.clone()
        epsilon.x = torch.randn(batch.x.shape, device=self.device)
        return epsilon


class TLDDPM(DDPMBase):
#     def sample(self, cond_batch, save_intermediate_steps, ode_steps=0):
    def sample(self, cond_batch, ode_steps=0):
        cond_batch.to(self.device)

        batch = cond_batch.clone()
        batch.x = torch.randn_like(batch.x, device=self.device)

        def forward_callback(batch):
            return self.forward(batch, cond_batch)

        if ode_steps:
            return self._ode_sample(batch, forward_callback, ode_steps=ode_steps)
        return self._sample(batch, forward_callback=forward_callback, save_intermediate_steps=save_intermediate_steps)

    def get_loss(self, batch):
        batch_0 = batch["batch_0"]
        batch_t = batch["batch_t"]

        noise_batch, epsilon, alpha_bars = self.get_noise_img_and_epsilon(batch_t)
        epsilon_hat = self.forward(noise_batch, batch_0)
        batch_predy = ((1/torch.sqrt(alpha_bars))*(noise_batch.x.T - torch.sqrt(1 - alpha_bars) * epsilon_hat.x.T)).T
        loss_supp = nn.functional.mse_loss(batch_predy, batch_t.x, reduction="none").sum(-1)
        loss = nn.functional.mse_loss(epsilon_hat.x, epsilon.x, reduction="none").sum(
            -1
        )
#         loss = nn.functional.mse_loss(epsilon_hat.x, batch_t.x, reduction="none").sum(
#             -1
#         )
        loss_supp = scatter(loss_supp, noise_batch.batch, reduce="mean").mean()
        loss = scatter(loss, noise_batch.batch, reduce="mean").mean()
        return loss, noise_batch, epsilon, epsilon_hat, loss_supp, batch_predy