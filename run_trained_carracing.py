from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from carracing_observation import DEFAULT_OBS_SOURCE, make_carracing_env
from carracing_obs import init_frame_stack, resize_observations, update_frame_stack
from ppo_pixel_policy import CarRacingPPOPolicy


DEFAULT_RUN_DIR = Path("runs") / "carracing_ppo_20260423_103309"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a trained PPO checkpoint on Gymnasium CarRacing-v3."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Run directory containing best_model.pt and config artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Explicit checkpoint path. Overrides --run-dir/best_model.pt when provided.",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for evaluation. Defaults to the checkpoint training seed.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array", "none"],
        help="Use 'human' for a live window, 'none' for headless runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Render FPS when using a visible window.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional manual episode cap. 0 means no extra cap.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of using the deterministic mean action.",
    )
    parser.add_argument(
        "--domain-randomize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint domain_randomize setting.",
    )
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        resolved = checkpoint
    else:
        resolved = run_dir / "best_model.pt"

    if resolved.exists():
        return resolved.resolve()

    if checkpoint is None:
        candidates = sorted(run_dir.glob("checkpoint_*.pt"))
        if candidates:
            return candidates[-1].resolve()

    raise FileNotFoundError(f"Checkpoint not found: {resolved}")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    # This checkpoint is generated locally by train_ppo_carracing.py and stores argparse values
    # including pathlib.Path objects, so full payload loading is required on newer PyTorch builds.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint_path = resolve_checkpoint(args.run_dir, args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    checkpoint_args = checkpoint.get("args", {})

    num_frames = int(checkpoint_args.get("num_frames", 4))
    image_size = int(checkpoint_args.get("image_size", 96))
    seed = int(checkpoint_args.get("seed", 7)) if args.seed is None else int(args.seed)
    domain_randomize = bool(checkpoint_args.get("domain_randomize", False))
    if args.domain_randomize is not None:
        domain_randomize = args.domain_randomize

    policy = CarRacingPPOPolicy(num_frames=num_frames, image_size=image_size).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    render_mode = None if args.render_mode == "none" else args.render_mode
    env = make_carracing_env(
        domain_randomize=domain_randomize,
        render_mode=render_mode,
        obs_source=DEFAULT_OBS_SOURCE,
        image_size=image_size,
        continuous=True,
    )

    if render_mode == "human" and args.fps > 0 and hasattr(env, "metadata"):
        env.metadata["render_fps"] = args.fps

    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"num_frames={num_frames}")
    print(f"image_size={image_size}")
    print(f"seed={seed}")
    print(f"domain_randomize={domain_randomize}")
    print(f"deterministic={not args.stochastic}")
    print(f"episodes={args.episodes}")
    if args.max_steps > 0:
        print(f"max_steps={args.max_steps}")

    returns: list[float] = []
    lengths: list[int] = []

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=seed + episode)
            obs = resize_observations(obs, image_size)
            stacked_obs = init_frame_stack(obs[None, ...], num_frames)
            ep_return = 0.0
            ep_length = 0
            done = False

            while not done:
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(stacked_obs).to(device)
                    policy_step = policy.act(obs_tensor, deterministic=not args.stochastic)

                action = policy_step.action[0].detach().cpu().numpy().astype(np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = resize_observations(next_obs, image_size)
                done = bool(terminated or truncated)
                ep_return += float(reward)
                ep_length += 1

                if args.max_steps > 0 and ep_length >= args.max_steps:
                    done = True

                stacked_obs = update_frame_stack(
                    stacked_obs,
                    next_obs[None, ...],
                    np.array([done], dtype=np.bool_),
                )

            returns.append(ep_return)
            lengths.append(ep_length)
            lap_finished = bool(info.get("lap_finished", False))
            print(
                f"episode={episode + 1} return={ep_return:.3f} length={ep_length} "
                f"lap_finished={lap_finished}"
            )
    finally:
        env.close()

    print(
        "summary "
        f"mean_return={float(np.mean(returns)):.3f} "
        f"std_return={float(np.std(returns)):.3f} "
        f"mean_length={float(np.mean(lengths)):.1f}"
    )


if __name__ == "__main__":
    main()
