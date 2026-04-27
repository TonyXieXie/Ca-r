from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def resize_observations(obs: np.ndarray, image_size: int) -> np.ndarray:
    if obs.ndim == 3:
        batch = obs[None, ...]
        squeeze = True
    elif obs.ndim == 4:
        batch = obs
        squeeze = False
    else:
        raise ValueError(f"Unsupported observation rank {obs.ndim}")

    if batch.shape[-1] != 3:
        raise ValueError(f"Expected RGB observations, got shape {tuple(batch.shape)}")

    if batch.shape[1] == image_size and batch.shape[2] == image_size:
        return obs

    tensor = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).to(torch.float32)
    resized = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    resized = (
        resized.round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
    return resized[0] if squeeze else resized


def init_frame_stack(obs: np.ndarray, num_frames: int) -> np.ndarray:
    return np.repeat(obs[:, None, ...], repeats=num_frames, axis=1)


def update_frame_stack(stacked_obs: np.ndarray, next_obs: np.ndarray, done: np.ndarray) -> np.ndarray:
    stacked_obs = np.roll(stacked_obs, shift=-1, axis=1)
    stacked_obs[:, -1] = next_obs
    if np.any(done):
        stacked_obs[done] = np.repeat(next_obs[done, None, ...], repeats=stacked_obs.shape[1], axis=1)
    return stacked_obs
