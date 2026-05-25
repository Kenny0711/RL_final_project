import math
import torch
import numpy as np


def cosine_beta_schedule(T: int = 200, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = f / f[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return betas.clamp(0.0001, 0.9999).float()


class NoiseSchedule:
    def __init__(self, T: int = 200, schedule: str = "cosine", device: str = "cpu"):
        self.T = T
        self.device = device

        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.ones(1), alphas_bar[:-1]])

        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_bar = alphas_bar.to(device)
        self.alphas_bar_prev = alphas_bar_prev.to(device)

        self.sqrt_alphas_bar = alphas_bar.sqrt().to(device)
        self.sqrt_one_minus_alphas_bar = (1.0 - alphas_bar).sqrt().to(device)

        # posterior variance for p(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        ).to(device)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape):
        vals = arr[t]
        while vals.ndim < len(shape):
            vals = vals.unsqueeze(-1)
        return vals.expand(shape)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: corrupt x0 to x_t at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_bar, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_bar, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def p_sample_step(self, model, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """One reverse step: sample x_{t-1} from x_t using the model."""
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_bar, t, x_t.shape)
        sqrt_recip_a = (1.0 / self.alphas[t].sqrt())
        sqrt_recip_a = self._extract(
            (1.0 / self.alphas.sqrt()), t, x_t.shape
        )

        # predicted noise from model
        model_out = model(x_t, t)
        mean = sqrt_recip_a * (x_t - betas_t / sqrt_1mab * model_out)

        # no noise at t=0
        noise = torch.randn_like(x_t)
        mask = (t > 0).float()
        while mask.ndim < x_t.ndim:
            mask = mask.unsqueeze(-1)

        var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean + mask * var.sqrt() * noise

    def p_sample_loop(self, model, shape,
                      start_from_x: torch.Tensor = None,
                      start_t: int = None) -> torch.Tensor:
        """
        Full reverse diffusion.
        If start_from_x and start_t provided: SDEdit-style repair (starts from noisy x).
        Otherwise: unconditional generation from pure noise.
        """
        device = self.device
        if start_from_x is not None and start_t is not None:
            x = start_from_x.to(device)
            t_range = range(start_t - 1, -1, -1)
        else:
            x = torch.randn(shape, device=device)
            t_range = range(self.T - 1, -1, -1)

        model.eval()
        with torch.no_grad():
            for t_val in t_range:
                t_batch = torch.full((shape[0],), t_val, dtype=torch.long, device=device)
                x = self.p_sample_step(model, x, t_batch)
        return x
