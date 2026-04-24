import argparse

import gymnasium as gym
import numpy as np
from gymnasium.utils.play import play
from pygame import K_DOWN, K_LEFT, K_RIGHT, K_UP


def continuous_mapping() -> tuple[dict[tuple[int, ...], np.ndarray], np.ndarray]:
    noop = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    keys_to_action = {
        (K_UP,): np.array([0.0, 1.0, 0.0], dtype=np.float32),
        (K_DOWN,): np.array([0.0, 0.0, 0.8], dtype=np.float32),
        (K_LEFT,): np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        (K_RIGHT,): np.array([1.0, 0.0, 0.0], dtype=np.float32),
        (K_UP, K_LEFT): np.array([-1.0, 1.0, 0.0], dtype=np.float32),
        (K_UP, K_RIGHT): np.array([1.0, 1.0, 0.0], dtype=np.float32),
        (K_DOWN, K_LEFT): np.array([-1.0, 0.0, 0.8], dtype=np.float32),
        (K_DOWN, K_RIGHT): np.array([1.0, 0.0, 0.8], dtype=np.float32),
    }
    return keys_to_action, noop


def discrete_mapping() -> tuple[dict[tuple[int, ...], int], int]:
    # 0: noop, 1: left, 2: right, 3: gas, 4: brake
    noop = 0
    keys_to_action = {
        (K_LEFT,): 1,
        (K_RIGHT,): 2,
        (K_UP,): 3,
        (K_DOWN,): 4,
    }
    return keys_to_action, noop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Gymnasium CarRacing-v3 with keyboard controls."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed used for the initial environment reset.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Playback FPS for the interactive window.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.0,
        help="Window scale factor applied by gymnasium.utils.play.",
    )
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="Use the environment's discrete action space instead of continuous controls.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start stepping immediately instead of waiting for the first key press.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    continuous = not args.discrete

    if continuous:
        keys_to_action, noop = continuous_mapping()
    else:
        keys_to_action, noop = discrete_mapping()

    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        continuous=continuous,
    )

    print("Controls: Up=gas, Down=brake, Left/Right=steer, Esc=quit")
    if args.discrete:
        print("Mode: discrete")
    else:
        print("Mode: continuous")

    try:
        play(
            env,
            keys_to_action=keys_to_action,
            noop=noop,
            fps=args.fps,
            zoom=args.zoom,
            seed=args.seed,
            wait_on_player=not args.no_wait,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
