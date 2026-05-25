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
        This is more stable at high noise levels.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.schedule.T, (B,), device=self.device)
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
        start_t=500 (default): medium repair — preserves coarse shape, fixes local noise/drift.
        Higher start_t → more aggressive (closer to unconditional generation).
        """
        start_t = min(start_t, self.schedule.T - 1)
        x_tensor = torch.tensor(x_bad, dtype=torch.float32, device=self.device)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        B = x_tensor.shape[0]
        t_batch = torch.full((B,), start_t, dtype=torch.long, device=self.device)
        x_noisy = self.schedule.q_sample(x_tensor, t_batch)

        x_repaired = self.schedule.p_sample_loop(
            self.model, x_noisy.shape,
            start_from_x=x_noisy, start_t=start_t
        )

        result = x_repaired.cpu().numpy()
        if x_bad.ndim == 2:
            result = result[0]
        return result

    @torch.no_grad()
    def generate(self, n_samples: int, seq_len: int = 50,
                 n_features: int = 2, norm_stats: dict = None) -> np.ndarray:
        """Unconditional generation from pure noise. Used for Baseline 2."""
        shape = (n_samples, seq_len, n_features)
        x = self.schedule.p_sample_loop(self.model, shape)
        return x.cpu().numpy()

    def guided_repair(self, x_bad: np.ndarray, safety_critic,
                      beta: float = 1.0, start_t: int = 500,
                      norm_stats: dict = None) -> np.ndarray:
        """
        Placeholder for Safe-GTA full pipeline (Week 2+).
        Modifies each reverse step with safety gradient guidance:
            x_{t-1} = p_sample_step(model, x_t, t)
                    - beta * grad_{x_t}(safety_critic(x_t))

        safety_critic must implement: critic(obs_tensor) -> cost_tensor in [0, 1]
        """
        raise NotImplementedError(
            "guided_repair not yet implemented. "
            "SafetyCritic interface: critic(obs: Tensor) -> cost: Tensor in [0, 1]"
        )

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
