import torch
import math
from torch.nn import functional as F

class TimeEmbed(torch.nn.Module):
    def __init__(self, timecfg):
        super().__init__()
        self.embedding_dim = timecfg.embedding_dim
        self.max_positions = timecfg.max_positions

    def forward(self, timesteps):
        """
        From Fairseq.
        Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
        """
        assert len(timesteps.shape) == 1
        timesteps = timesteps * self.max_positions
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], self.embedding_dim)
        return emb