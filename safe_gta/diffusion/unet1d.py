import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim * 2),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        groups = min(8, out_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.act = nn.SiLU()
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        h = self.act(self.norm1(self.conv1(x)))
        # inject time embedding channel-wise
        t = self.act(self.time_proj(t_emb)).unsqueeze(-1)
        h = h + t
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class TemporalUNet(nn.Module):
    """
    1D Temporal U-Net for trajectory denoising.
    Input/output shape: (batch, seq_len, n_features)
    Internally operates on (batch, n_features, seq_len) for Conv1d.
    """

    def __init__(self, seq_len: int = 50, n_features: int = 2,
                 base_ch: int = 32, time_emb_dim: int = 128):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        t_dim = time_emb_dim * 2  # after MLP

        ch = [base_ch, base_ch * 2, base_ch * 4]  # [32, 64, 128]

        # Encoder
        self.enc1 = ResidualBlock1D(n_features, ch[0], t_dim)
        self.enc2 = ResidualBlock1D(ch[0], ch[1], t_dim)
        self.down2 = nn.MaxPool1d(2)
        self.enc3 = ResidualBlock1D(ch[1], ch[2], t_dim)
        self.down3 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock1D(ch[2], ch[2], t_dim)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = ResidualBlock1D(ch[2] + ch[1], ch[1], t_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = ResidualBlock1D(ch[1] + ch[0], ch[0], t_dim)

        # Output head
        self.out = nn.Conv1d(ch[0], n_features, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) → (B, F, L)
        x = x.permute(0, 2, 1)
        t_emb = self.time_emb(t)

        # Encoder
        s1 = self.enc1(x, t_emb)                      # (B, 32, L)
        s2 = self.enc2(self.down2(s1), t_emb)         # (B, 64, L/2)
        s3 = self.enc3(self.down3(s2), t_emb)         # (B, 128, L/4)

        # Bottleneck
        h = self.bottleneck(s3, t_emb)                # (B, 128, L/4)

        # Decoder with skip connections + size alignment
        h = self.up3(h)
        h = _pad_to(h, s2)
        h = self.dec3(torch.cat([h, s2], dim=1), t_emb)   # (B, 64, L/2)

        h = self.up2(h)
        h = _pad_to(h, s1)
        h = self.dec2(torch.cat([h, s1], dim=1), t_emb)   # (B, 32, L)

        out = self.out(h)                              # (B, F, L)
        return out.permute(0, 2, 1)                   # (B, L, F)


def _pad_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Pad or trim x along the last dim to match ref's length."""
    diff = ref.shape[-1] - x.shape[-1]
    if diff > 0:
        x = F.pad(x, (0, diff))
    elif diff < 0:
        x = x[..., :ref.shape[-1]]
    return x
