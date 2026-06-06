import os

import numpy as np
import torch
import torch.nn as nn

from .noise_schedule import NoiseSchedule
from .unet1d import TemporalUNet


class DDPM:
    def __init__(self, model: TemporalUNet, schedule: NoiseSchedule, device: str = "cpu"):
        self.model = model.to(device)
        self.schedule = schedule
        self.device = device

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Standard DDPM eps-prediction loss.
        Predicts the noise added at timestep t, not the clean x0.
        """
        batch_size = x0.shape[0]
        t = torch.randint(0, self.schedule.T, (batch_size,), device=self.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        eps_pred = self.model(x_t, t)
        return nn.functional.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def repair(self, x_bad: np.ndarray, start_t: int = 500,
               norm_stats: dict = None) -> np.ndarray:
        """
        SDEdit-style trajectory repair.
        Adds noise to x_bad up to start_t, then denoises back to t=0.
        """
        start_t = min(start_t, self.schedule.T - 1)
        x_tensor = torch.tensor(x_bad, dtype=torch.float32, device=self.device)
        squeeze = False
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)
            squeeze = True

        batch_size = x_tensor.shape[0]
        t_batch = torch.full((batch_size,), start_t, dtype=torch.long, device=self.device)
        x_noisy = self.schedule.q_sample(x_tensor, t_batch)

        x_repaired = self.schedule.p_sample_loop(
            self.model,
            x_noisy.shape,
            start_from_x=x_noisy,
            start_t=start_t,
        )

        result = x_repaired.cpu().numpy()
        if squeeze:
            result = result[0]
        return result

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int = 50,
                 n_features: int = 2, norm_stats: dict = None) -> np.ndarray:
        """Unconditional generation from pure noise."""
        shape = (n_samples, seq_len, n_features)
        x = self.schedule.p_sample_loop(self.model, shape)
        return x.cpu().numpy()

    def guided_repair(self, x_bad: np.ndarray, safety_critic,
                      beta: float = 1.0, start_t: int = 500,
                      norm_stats: dict = None, guidance_every: int = 1,
                      guidance_clip: float = 1.0) -> np.ndarray:
        """
        SDEdit repair with Safety Critic guidance.

        The diffusion model first proposes a denoising step. Then the Safety
        Critic gradient nudges the normalized trajectory toward lower predicted
        cost:

            x_prev = denoise(x_t) - beta * grad_x mean(critic(x))

        safety_critic must accept a normalized joint trajectory tensor with
        shape (batch, seq_len, obs_dim + act_dim) and return per-step costs.
        """
        start_t = min(start_t, self.schedule.T - 1)
        guidance_every = max(1, int(guidance_every))

        x_tensor = torch.tensor(x_bad, dtype=torch.float32, device=self.device)
        squeeze = False
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)
            squeeze = True

        batch_size = x_tensor.shape[0]
        t_batch = torch.full((batch_size,), start_t, dtype=torch.long, device=self.device)
        x = self.schedule.q_sample(x_tensor, t_batch)

        self.model.eval()
        safety_critic.eval()

        for step_idx, t_val in enumerate(range(start_t - 1, -1, -1)):
            t = torch.full((batch_size,), t_val, dtype=torch.long, device=self.device)

            with torch.no_grad():
                x = self.schedule.p_sample_step(self.model, x, t)

            if beta > 0 and step_idx % guidance_every == 0:
                x_guided = x.detach().requires_grad_(True)
                costs = safety_critic(x_guided)
                cost_loss = costs.mean()
                grad = torch.autograd.grad(cost_loss, x_guided)[0]
                if guidance_clip is not None and guidance_clip > 0:
                    grad = grad.clamp(-guidance_clip, guidance_clip)
                x = (x_guided - beta * grad).detach().clamp(-1.0, 1.0)

        result = x.cpu().numpy()
        if squeeze:
            result = result[0]
        return result

    def save_checkpoint(self, path: str, norm_stats: dict = None, extra: dict = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "norm_stats": norm_stats or {},
            "T": self.schedule.T,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        return payload
