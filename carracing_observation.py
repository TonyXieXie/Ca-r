from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.box2d.car_racing import CarRacing, FPS, InvalidAction, PLAYFIELD
from PIL import Image


DEFAULT_IMAGE_SIZE = 256
DEFAULT_OBS_SOURCE = "rgb_array"
DEFAULT_MAX_EPISODE_STEPS = 1000


def _resample_bilinear() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def resize_frame_to_square(frame: np.ndarray, image_size: int) -> np.ndarray:
    if frame.shape == (image_size, image_size, 3):
        return frame.astype(np.uint8, copy=False)

    image = Image.fromarray(frame)
    scale = min(image_size / image.width, image_size / image.height)
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))
    resized = image.resize((resized_width, resized_height), _resample_bilinear())

    canvas = Image.new("RGB", (image_size, image_size))
    offset_x = (image_size - resized_width) // 2
    offset_y = (image_size - resized_height) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return np.asarray(canvas, dtype=np.uint8)


class ConfigurableCarRacing(CarRacing):
    """CarRacing variant that emits either state_pixels or rgb_array as observations."""

    def __init__(self, *args, observation_mode: str = "state_pixels", **kwargs) -> None:
        if observation_mode not in {"state_pixels", "rgb_array"}:
            raise ValueError(f"Unsupported observation_mode={observation_mode!r}")
        self.observation_mode = observation_mode
        super().__init__(*args, **kwargs)

    def step(self, action: np.ndarray | int):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                action = action.astype(np.float64)
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render(self.observation_mode)

        step_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, bool] = {}
        if action is not None:
            self.reward -= 0.1
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track) or self.new_lap:
                terminated = True
                info["lap_finished"] = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                step_reward = -100.0

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, info


class CarRacingObservationWrapper(gym.Wrapper):
    """Resize observations from CarRacing to a square RGB tensor."""

    def __init__(self, env: gym.Env, image_size: int) -> None:
        super().__init__(env)
        self.image_size = int(image_size)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )

    def _transform_observation(self, obs: np.ndarray) -> np.ndarray:
        return resize_frame_to_square(obs, self.image_size)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform_observation(obs), reward, terminated, truncated, info


def make_carracing_env(
    *,
    domain_randomize: bool,
    render_mode: str | None,
    obs_source: str,
    image_size: int,
    continuous: bool = True,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
) -> gym.Env:
    env: gym.Env = ConfigurableCarRacing(
        render_mode=render_mode,
        domain_randomize=domain_randomize,
        continuous=continuous,
        observation_mode=obs_source,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = CarRacingObservationWrapper(env, image_size=image_size)
    return env
