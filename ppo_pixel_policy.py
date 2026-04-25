from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal
import torch.nn.functional as F


def _init_layer(layer: nn.Module, std: float = math.sqrt(2.0), bias: float = 0.0) -> nn.Module:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias)
    return layer


def _atanh(x: Tensor) -> Tensor:
    x = x.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _logit(x: Tensor) -> Tensor:
    x = x.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(x) - torch.log1p(-x)


def _tanh_log_abs_det_jacobian(x: Tensor) -> Tensor:
    # Stable log(1 - tanh(x)^2)
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


def _sigmoid_log_abs_det_jacobian(x: Tensor) -> Tensor:
    # log(sigmoid(x) * (1 - sigmoid(x)))
    return -F.softplus(-x) - F.softplus(x)


@dataclass
class PolicyStep:
    action: Tensor
    log_prob: Tensor
    value: Tensor
    mean_action: Tensor
    raw_action: Tensor


class PixelEncoder(nn.Module):
    """CNN encoder for stacked CarRacing frames."""

    def __init__(
        self,
        in_channels: int = 12,
        image_size: int = 96,
        feature_dim: int = 512,
        pooled_size: int = 6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.pooled_size = pooled_size

        self.conv = nn.Sequential(
            _init_layer(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _init_layer(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _init_layer(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            _init_layer(nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            conv_out_dim = self.pool(self.conv(dummy)).flatten(1).shape[1]

        self.fc = nn.Sequential(
            _init_layer(nn.Linear(conv_out_dim, feature_dim)),
            nn.ReLU(),
        )

    def preprocess(self, obs: Tensor) -> Tensor:
        """
        Supported layouts:
        - [B, T, H, W, C]  where C=3
        - [B, H, W, C]
        - [B, C, H, W]
        """
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)

        if obs.ndim == 5:
            b, t, h, w, c = obs.shape
            if c != 3:
                raise ValueError(f"Expected RGB frames in last dim, got shape {tuple(obs.shape)}")
            obs = obs.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
        elif obs.ndim == 4:
            if obs.shape[1] == self.in_channels:
                pass
            elif obs.shape[-1] in (1, 3, self.in_channels):
                obs = obs.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported observation shape {tuple(obs.shape)}")
        else:
            raise ValueError(f"Unsupported observation rank {obs.ndim}")

        if obs.shape[1] != self.in_channels:
            raise ValueError(
                f"Encoder expected {self.in_channels} channels, got shape {tuple(obs.shape)}"
            )

        obs = obs.float()
        if obs.max() > 1.0:
            obs = obs / 255.0
        return obs

    def forward(self, obs: Tensor) -> Tensor:
        x = self.preprocess(obs)
        x = self.conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class CarRacingPPOPolicy(nn.Module):
    """
    PPO actor-critic for pixel observations and continuous CarRacing actions.

    Action mapping:
    - steer: tanh(raw)    -> [-1, 1]
    - gas: sigmoid(raw)   -> [0, 1]
    - brake: sigmoid(raw) -> [0, 1]
    """

    def __init__(
        self,
        num_frames: int = 4,
        image_size: int = 96,
        encoder_dim: int = 512,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.action_dim = 3
        self.in_channels = num_frames * 3

        self.encoder = PixelEncoder(
            in_channels=self.in_channels,
            image_size=image_size,
            feature_dim=encoder_dim,
        )

        self.trunk = nn.Sequential(
            _init_layer(nn.Linear(encoder_dim, hidden_dim)),
            nn.ReLU(),
        )

        self.actor_mean = _init_layer(nn.Linear(hidden_dim, self.action_dim), std=0.01)
        self.critic_head = _init_layer(nn.Linear(hidden_dim, 1), std=1.0)
        self.actor_logstd = nn.Parameter(torch.tensor([0.0, -0.5, -0.5], dtype=torch.float32))

    def _features(self, obs: Tensor) -> Tensor:
        z = self.encoder(obs)
        return self.trunk(z)

    def policy_params(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        h = self._features(obs)
        mean = self.actor_mean(h)
        log_std = self.actor_logstd.clamp(-2.0, 2.0).expand_as(mean)
        std = log_std.exp()
        return mean, std

    def value(self, obs: Tensor) -> Tensor:
        h = self._features(obs)
        return self.critic_head(h).squeeze(-1)

    @staticmethod
    def squash_action(raw_action: Tensor) -> Tensor:
        steer = torch.tanh(raw_action[..., 0:1])
        gas = torch.sigmoid(raw_action[..., 1:2])
        brake = torch.sigmoid(raw_action[..., 2:3])
        return torch.cat([steer, gas, brake], dim=-1)

    @staticmethod
    def unsquash_action(action: Tensor) -> Tensor:
        steer = _atanh(action[..., 0:1])
        gas = _logit(action[..., 1:2])
        brake = _logit(action[..., 2:3])
        return torch.cat([steer, gas, brake], dim=-1)

    @staticmethod
    def _squash_log_prob_correction(raw_action: Tensor) -> Tensor:
        steer_correction = _tanh_log_abs_det_jacobian(raw_action[..., 0:1])
        gas_correction = _sigmoid_log_abs_det_jacobian(raw_action[..., 1:2])
        brake_correction = _sigmoid_log_abs_det_jacobian(raw_action[..., 2:3])
        return torch.cat([steer_correction, gas_correction, brake_correction], dim=-1).sum(-1)

    def _distribution(self, obs: Tensor) -> tuple[Normal, Tensor]:
        mean, std = self.policy_params(obs)
        return Normal(mean, std), mean

    def act(self, obs: Tensor, deterministic: bool = False) -> PolicyStep:
        dist, mean = self._distribution(obs)
        value = self.value(obs)

        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()

        action = self.squash_action(raw_action)
        raw_log_prob = dist.log_prob(raw_action).sum(-1)
        correction = self._squash_log_prob_correction(raw_action)
        log_prob = raw_log_prob - correction
        mean_action = self.squash_action(mean)

        return PolicyStep(
            action=action,
            log_prob=log_prob,
            value=value,
            mean_action=mean_action,
            raw_action=raw_action,
        )

    def evaluate_actions(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
        - log_prob: exact log-prob under the squashed policy
        - entropy: pre-squash Gaussian entropy (common PPO approximation)
        - value: state-value estimate
        """
        dist, _ = self._distribution(obs)
        value = self.value(obs)

        raw_action = self.unsquash_action(action)
        raw_log_prob = dist.log_prob(raw_action).sum(-1)
        correction = self._squash_log_prob_correction(raw_action)
        log_prob = raw_log_prob - correction
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        mean, _ = self.policy_params(obs)
        value = self.value(obs)
        return self.squash_action(mean), value


__all__ = ["CarRacingPPOPolicy", "PixelEncoder", "PolicyStep"]
