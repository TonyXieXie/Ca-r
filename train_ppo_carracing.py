from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode
from torch import Tensor
from torch.optim import Adam

from carracing_observation import DEFAULT_OBS_SOURCE, make_carracing_env
from carracing_obs import init_frame_stack, resize_observations, update_frame_stack
from ppo_pixel_policy import CarRacingPPOPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Gymnasium CarRacing-v3 from pixels.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--vector-env", type=str, default="async", choices=["async", "sync"])
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--target-kl", type=float, default=0.015, help="Approx KL threshold for early stopping (0 = disabled).")
    parser.add_argument("--lr-anneal", action="store_true", help="Linearly anneal learning rate to 0 over training.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-deterministic", action="store_true")
    parser.add_argument("--save-dir", type=Path, default=Path("runs") / "carracing_ppo")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--async-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pending-evals", type=int, default=4)
    parser.add_argument("--eval-vector-env", type=str, default="async", choices=["async", "sync"])
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available.")
    parser.add_argument("--fail-on-nonfinite", action="store_true", help="Abort training if NaN/Inf appears.")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Initialize policy weights from a saved checkpoint without resuming optimizer state.",
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(
    idx: int,
    seed: int,
    capture_video: bool,
    run_dir: Path,
    domain_randomize: bool,
    image_size: int,
):
    def thunk():
        render_mode = "rgb_array" if capture_video and idx == 0 else None
        env = make_carracing_env(
            domain_randomize=domain_randomize,
            render_mode=render_mode,
            obs_source=DEFAULT_OBS_SOURCE,
            image_size=image_size,
            continuous=True,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(run_dir / "videos"),
                episode_trigger=lambda episode_id: episode_id % 20 == 0,
            )
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        return env

    return thunk

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    if var_y < 1e-8:
        return float("nan")
    return 1.0 - np.var(y_true - y_pred) / var_y


def save_checkpoint(
    save_dir: Path,
    policy: CarRacingPPOPolicy,
    optimizer: Adam,
    args: argparse.Namespace,
    update: int,
    global_step: int,
    filename: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / (filename or f"checkpoint_{update:05d}.pt")
    policy_to_save = getattr(policy, "_orig_mod", policy)
    payload = {
        "model": policy_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "update": update,
        "global_step": global_step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def save_eval_snapshot(
    save_dir: Path,
    policy: CarRacingPPOPolicy,
    args: argparse.Namespace,
    update: int,
    global_step: int,
    eval_seed: int,
) -> Path:
    snapshot_dir = save_dir / "eval_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"eval_update_{update:05d}.pt"
    policy_to_save = getattr(policy, "_orig_mod", policy)
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in policy_to_save.state_dict().items()},
            "args": vars(args),
            "update": update,
            "global_step": global_step,
            "eval_seed": eval_seed,
        },
        snapshot_path,
    )
    return snapshot_path


def save_best_eval_checkpoint(
    save_dir: Path,
    snapshot: dict[str, Any],
    best_eval_return: float,
) -> Path:
    checkpoint_path = save_dir / "best_model.pt"
    payload = {
        "model": snapshot["model"],
        "optimizer": None,
        "args": snapshot["args"],
        "update": int(snapshot["update"]),
        "global_step": int(snapshot["global_step"]),
        "best_eval_return": float(best_eval_return),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_finite(name: str, value: Tensor, fail_on_nonfinite: bool) -> bool:
    is_finite = bool(torch.isfinite(value).all().item())
    if fail_on_nonfinite and not is_finite:
        raise RuntimeError(f"Non-finite tensor encountered: {name}")
    return is_finite


def evaluate_policy(
    policy: CarRacingPPOPolicy,
    device: torch.device,
    num_frames: int,
    eval_episodes: int,
    seed: int,
    domain_randomize: bool,
    eval_vector_env: str = "async",
) -> dict[str, float]:
    if eval_episodes <= 0:
        return {
            "eval_return_mean": float("nan"),
            "eval_return_std": float("nan"),
            "eval_length_mean": float("nan"),
            "eval_steps_total": 0,
        }

    env_fns = [
        make_env(
            idx=i,
            seed=seed,
            capture_video=False,
            run_dir=Path("."),
            domain_randomize=domain_randomize,
            image_size=policy.image_size,
        )
        for i in range(eval_episodes)
    ]
    vector_env_cls = AsyncVectorEnv if eval_vector_env == "async" else SyncVectorEnv
    envs = vector_env_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    try:
        obs, _ = envs.reset(seed=[seed + episode for episode in range(eval_episodes)])
        obs = resize_observations(obs, policy.image_size)
        stacked_obs = init_frame_stack(obs, num_frames)
        returns = np.zeros(eval_episodes, dtype=np.float64)
        lengths = np.zeros(eval_episodes, dtype=np.int64)
        done = np.zeros(eval_episodes, dtype=np.bool_)

        while not bool(done.all()):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(stacked_obs).to(device)
                step = policy.act(obs_tensor, deterministic=True)
            actions = step.action.cpu().numpy()
            actions[done] = 0.0
            next_obs, rewards, terminated, truncated, _ = envs.step(actions)
            next_obs = resize_observations(next_obs, policy.image_size)
            step_done = np.logical_or(terminated, truncated)
            active = np.logical_not(done)
            returns[active] += rewards[active]
            lengths[active] += 1
            done = np.logical_or(done, step_done)
            stacked_obs = update_frame_stack(stacked_obs, next_obs, done)
    finally:
        envs.close()

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_steps_total": int(sum(lengths)),
    }


def eval_snapshot_worker(snapshot_path: str, eval_episodes: int, eval_vector_env: str) -> dict[str, Any]:
    eval_start_time = time.perf_counter()
    device = torch.device("cpu")
    checkpoint = torch.load(snapshot_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    num_frames = int(checkpoint_args.get("num_frames", 1))
    image_size = int(checkpoint_args.get("image_size", 96))
    domain_randomize = bool(checkpoint_args.get("domain_randomize", False))
    policy = CarRacingPPOPolicy(num_frames=num_frames, image_size=image_size).to(device)
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    metrics = evaluate_policy(
        policy=policy,
        device=device,
        num_frames=num_frames,
        eval_episodes=eval_episodes,
        seed=int(checkpoint["eval_seed"]),
        domain_randomize=domain_randomize,
        eval_vector_env=eval_vector_env,
    )
    eval_time_sec = time.perf_counter() - eval_start_time
    eval_steps_total = int(metrics["eval_steps_total"])
    metrics.update(
        {
            "record_type": "eval_result",
            "update": int(checkpoint["update"]),
            "global_step": int(checkpoint["global_step"]),
            "eval_seed": int(checkpoint["eval_seed"]),
            "eval_time_sec": float(eval_time_sec),
            "eval_sps": float(eval_steps_total / max(eval_time_sec, 1e-6)),
            "snapshot_path": snapshot_path,
        }
    )
    return metrics


def process_eval_results(
    pending_evals: dict[Future[dict[str, Any]], Path],
    metrics_path: Path,
    save_dir: Path,
    best_eval_return: float,
    wait: bool = False,
) -> float:
    while pending_evals:
        completed = [future for future in pending_evals if wait or future.done()]
        if not completed:
            break
        for future in completed:
            snapshot_path = pending_evals.pop(future)
            try:
                eval_data = future.result()
            except Exception as exc:
                eval_data = {
                    "record_type": "eval_result",
                    "update": None,
                    "global_step": None,
                    "snapshot_path": str(snapshot_path),
                    "eval_error": repr(exc),
                }
            save_time_sec = 0.0
            if "eval_return_mean" in eval_data and eval_data["eval_return_mean"] > best_eval_return:
                best_eval_return = float(eval_data["eval_return_mean"])
                best_save_start_time = time.perf_counter()
                snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
                best_path = save_best_eval_checkpoint(save_dir, snapshot, best_eval_return)
                best_checkpoint_time_sec = time.perf_counter() - best_save_start_time
                save_time_sec += best_checkpoint_time_sec
                eval_data["best_model"] = str(best_path)
                eval_data["best_eval_return"] = float(best_eval_return)
                eval_data["best_checkpoint_time_sec"] = float(best_checkpoint_time_sec)
            else:
                eval_data.setdefault("best_checkpoint_time_sec", None)
            eval_data["save_time_sec"] = float(save_time_sec) if save_time_sec > 0 else None
            print(json.dumps(eval_data, ensure_ascii=False))
            append_jsonl(metrics_path, eval_data)
            try:
                snapshot_path.unlink(missing_ok=True)
            except OSError:
                pass
        if not wait:
            break
    return best_eval_return


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.torch_deterministic)
    device = choose_device(args.device)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.save_dir / "metrics.jsonl"
    config_path = args.save_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    env_fns = [
        make_env(
            idx=i,
            seed=args.seed,
            capture_video=args.capture_video,
            run_dir=args.save_dir,
            domain_randomize=args.domain_randomize,
            image_size=args.image_size,
        )
        for i in range(args.num_envs)
    ]
    vector_env_cls = AsyncVectorEnv if args.vector_env == "async" else SyncVectorEnv
    envs = vector_env_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)

    obs, _ = envs.reset(seed=args.seed)
    obs = resize_observations(obs, args.image_size)
    stacked_obs = init_frame_stack(obs, args.num_frames)

    policy = CarRacingPPOPolicy(num_frames=args.num_frames, image_size=args.image_size).to(device)
    if args.init_from is not None:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["model"])
    if args.compile and hasattr(torch, "compile"):
        policy = torch.compile(policy)  # type: ignore[assignment]
    optimizer = Adam(policy.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = args.num_envs * args.num_steps
    if batch_size % args.num_minibatches != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by num_minibatches={args.num_minibatches}"
        )
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_timesteps // batch_size
    initial_lr = args.learning_rate

    obs_buffer = np.zeros(
        (args.num_steps, args.num_envs, args.num_frames, args.image_size, args.image_size, 3),
        dtype=np.uint8,
    )
    action_buffer = np.zeros((args.num_steps, args.num_envs, 3), dtype=np.float32)
    logprob_buffer = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    reward_buffer = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    done_buffer = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    value_buffer = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    episode_returns = np.zeros(args.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(args.num_envs, dtype=np.int32)
    recent_returns: deque[float] = deque(maxlen=20)
    recent_lengths: deque[int] = deque(maxlen=20)

    global_step = 0
    start_time = time.perf_counter()
    best_eval_return = float("-inf")
    pending_evals: dict[Future[dict[str, Any]], Path] = {}
    eval_executor: ProcessPoolExecutor | None = None
    if args.async_eval and args.eval_every > 0:
        eval_executor = ProcessPoolExecutor(
            max_workers=max(args.max_pending_evals, 1),
            mp_context=mp.get_context("spawn"),
        )

    for update in range(1, num_updates + 1):
        update_start_time = time.perf_counter()
        actor_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []
        total_losses: list[float] = []
        approx_kls: list[float] = []
        grad_norms: list[float] = []
        finite_ok = True
        eval_time_sec = 0.0
        eval_steps_total = 0
        best_checkpoint_time_sec = 0.0
        periodic_checkpoint_time_sec = 0.0
        save_time_sec = 0.0
        periodic_checkpoint_path: Path | None = None

        # Linear learning rate annealing
        if args.lr_anneal:
            frac = 1.0 - (update - 1) / num_updates
            lr_now = frac * initial_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

        rollout_start_time = time.perf_counter()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buffer[step] = stacked_obs

            with torch.no_grad():
                obs_tensor = torch.from_numpy(stacked_obs).to(device)
                policy_step = policy.act(obs_tensor)

            actions = policy_step.action.cpu().numpy()
            values = policy_step.value.cpu().numpy()
            log_probs = policy_step.log_prob.cpu().numpy()

            next_obs, rewards, terminated, truncated, infos = envs.step(actions)
            next_obs = resize_observations(next_obs, args.image_size)
            done = np.logical_or(terminated, truncated)

            action_buffer[step] = actions
            value_buffer[step] = values
            logprob_buffer[step] = log_probs
            reward_buffer[step] = rewards
            done_buffer[step] = done.astype(np.float32)

            episode_returns += rewards
            episode_lengths += 1
            done_indices = np.flatnonzero(done)
            for idx in done_indices:
                recent_returns.append(float(episode_returns[idx]))
                recent_lengths.append(int(episode_lengths[idx]))
                episode_returns[idx] = 0.0
                episode_lengths[idx] = 0

            stacked_obs = update_frame_stack(stacked_obs, next_obs, done)
        rollout_time_sec = time.perf_counter() - rollout_start_time

        postprocess_start_time = time.perf_counter()
        with torch.no_grad():
            next_obs_tensor = torch.from_numpy(stacked_obs).to(device)
            next_value = policy.value(next_obs_tensor).cpu().numpy()

        advantages = np.zeros_like(reward_buffer, dtype=np.float32)
        lastgaelam = np.zeros(args.num_envs, dtype=np.float32)
        for step in reversed(range(args.num_steps)):
            if step == args.num_steps - 1:
                next_non_terminal = 1.0 - done_buffer[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - done_buffer[step]
                next_values = value_buffer[step + 1]

            delta = reward_buffer[step] + args.gamma * next_values * next_non_terminal - value_buffer[step]
            lastgaelam = delta + args.gamma * args.gae_lambda * next_non_terminal * lastgaelam
            advantages[step] = lastgaelam

        returns = advantages + value_buffer

        b_obs = torch.from_numpy(
            obs_buffer.reshape(batch_size, args.num_frames, args.image_size, args.image_size, 3)
        ).to(device)
        b_actions = torch.from_numpy(action_buffer.reshape(batch_size, 3)).to(device)
        b_logprobs = torch.from_numpy(logprob_buffer.reshape(batch_size)).to(device)
        b_advantages = torch.from_numpy(advantages.reshape(batch_size)).to(device)
        b_returns = torch.from_numpy(returns.reshape(batch_size)).to(device)
        b_values = torch.from_numpy(value_buffer.reshape(batch_size)).to(device)
        postprocess_time_sec = time.perf_counter() - postprocess_start_time

        batch_indices = np.arange(batch_size)
        clipfracs = []

        optimize_start_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(batch_indices)
            epoch_kls = []
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = batch_indices[start:end]

                new_logprob, entropy, new_values = policy.evaluate_actions(
                    b_obs[mb_idx],
                    b_actions[mb_idx],
                )
                logratio = new_logprob - b_logprobs[mb_idx]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                    approx_kl = float(((-logratio).mean()).item())
                    approx_kls.append(approx_kl)
                    epoch_kls.append(approx_kl)

                # KL early stopping: if average KL this epoch is too large, stop updating
                if args.target_kl > 0 and len(epoch_kls) >= 2 and np.mean(epoch_kls) > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_idx]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std(unbiased=False) + 1e-8
                )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * ((new_values - b_returns[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

                # Abort early or surface warnings if the optimization state blows up.
                finite_ok &= ensure_finite("loss", loss, args.fail_on_nonfinite)
                finite_ok &= ensure_finite("new_logprob", new_logprob, args.fail_on_nonfinite)
                finite_ok &= ensure_finite("new_values", new_values, args.fail_on_nonfinite)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                actor_losses.append(float(pg_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_values.append(float(entropy_loss.item()))
                total_losses.append(float(loss.item()))
                grad_norms.append(float(grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm))
        optimize_time_sec = time.perf_counter() - optimize_start_time

        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        ev = explained_variance(y_pred, y_true)
        log_data = {
            "update": int(update),
            "global_step": int(global_step),
            "avg_return_20": float(np.mean(recent_returns)) if recent_returns else None,
            "avg_ep_len_20": float(np.mean(recent_lengths)) if recent_lengths else None,
            "explained_variance": None if np.isnan(ev) else float(ev),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "policy_loss": float(np.mean(actor_losses)) if actor_losses else None,
            "value_loss": float(np.mean(value_losses)) if value_losses else None,
            "entropy": float(np.mean(entropy_values)) if entropy_values else None,
            "total_loss": float(np.mean(total_losses)) if total_losses else None,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else None,
            "grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
            "rollout_reward_mean": float(reward_buffer.mean()),
            "rollout_reward_std": float(reward_buffer.std()),
            "value_mean": float(value_buffer.mean()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "finite_ok": bool(finite_ok),
            "eval_time_sec": None,
            "eval_steps_total": None,
            "eval_sps": None,
            "pending_evals": len(pending_evals),
            "eval_submitted": False,
            "eval_skipped_pending": False,
            "best_checkpoint_time_sec": None,
            "checkpoint_time_sec": None,
            "save_time_sec": None,
        }
        if args.eval_every > 0 and update % args.eval_every == 0:
            eval_seed = args.seed + 10_000 + update * 100
            if args.async_eval:
                if eval_executor is not None and len(pending_evals) < max(args.max_pending_evals, 1):
                    snapshot_path = save_eval_snapshot(args.save_dir, policy, args, update, global_step, eval_seed)
                    future = eval_executor.submit(
                        eval_snapshot_worker,
                        str(snapshot_path),
                        args.eval_episodes,
                        args.eval_vector_env,
                    )
                    pending_evals[future] = snapshot_path
                    log_data["eval_submitted"] = True
                    log_data["pending_evals"] = len(pending_evals)
                else:
                    log_data["eval_skipped_pending"] = True
            else:
                eval_start_time = time.perf_counter()
                eval_metrics = evaluate_policy(
                    policy=policy,
                    device=device,
                    num_frames=args.num_frames,
                    eval_episodes=args.eval_episodes,
                    seed=eval_seed,
                    domain_randomize=args.domain_randomize,
                    eval_vector_env=args.eval_vector_env,
                )
                eval_time_sec = time.perf_counter() - eval_start_time
                eval_steps_total = int(eval_metrics["eval_steps_total"])
                log_data.update(eval_metrics)
                log_data["eval_time_sec"] = float(eval_time_sec)
                log_data["eval_steps_total"] = int(eval_steps_total)
                log_data["eval_sps"] = float(eval_steps_total / max(eval_time_sec, 1e-6))
                if eval_metrics["eval_return_mean"] > best_eval_return:
                    best_eval_return = eval_metrics["eval_return_mean"]
                    best_save_start_time = time.perf_counter()
                    best_path = save_checkpoint(
                        args.save_dir,
                        policy,
                        optimizer,
                        args,
                        update,
                        global_step,
                        filename="best_model.pt",
                        extra={"best_eval_return": best_eval_return},
                    )
                    best_checkpoint_time_sec = time.perf_counter() - best_save_start_time
                    save_time_sec += best_checkpoint_time_sec
                    log_data["best_model"] = str(best_path)
                    log_data["best_eval_return"] = float(best_eval_return)
                    log_data["best_checkpoint_time_sec"] = float(best_checkpoint_time_sec)

        if update % args.save_every == 0 or update == num_updates:
            checkpoint_save_start_time = time.perf_counter()
            periodic_checkpoint_path = save_checkpoint(args.save_dir, policy, optimizer, args, update, global_step)
            periodic_checkpoint_time_sec = time.perf_counter() - checkpoint_save_start_time
            save_time_sec += periodic_checkpoint_time_sec

        elapsed_time_sec = time.perf_counter() - start_time
        update_time_sec = time.perf_counter() - update_start_time
        steps_remaining = max(args.total_timesteps - global_step, 0)
        overall_sps = global_step / max(elapsed_time_sec, 1e-6)
        sps = int(overall_sps)
        log_data.update(
            {
                "sps": int(sps),
                "sps_float": float(overall_sps),
                "elapsed_time_sec": float(elapsed_time_sec),
                "remaining_time_sec": float(steps_remaining / max(overall_sps, 1e-6)),
                "update_time_sec": float(update_time_sec),
                "rollout_time_sec": float(rollout_time_sec),
                "postprocess_time_sec": float(postprocess_time_sec),
                "optimize_time_sec": float(optimize_time_sec),
                "rollout_sps": float(batch_size / max(rollout_time_sec, 1e-6)),
                "update_sps": float(batch_size / max(update_time_sec, 1e-6)),
                "checkpoint_time_sec": float(periodic_checkpoint_time_sec) if periodic_checkpoint_path is not None else None,
                "save_time_sec": float(save_time_sec) if save_time_sec > 0 else None,
            }
        )

        if update % args.log_every == 0:
            print(json.dumps(log_data, ensure_ascii=False))
            append_jsonl(metrics_path, log_data)

        if periodic_checkpoint_path is not None:
            print(json.dumps({"checkpoint": str(periodic_checkpoint_path)}, ensure_ascii=False))

        if args.async_eval:
            best_eval_return = process_eval_results(
                pending_evals,
                metrics_path,
                args.save_dir,
                best_eval_return,
                wait=False,
            )

    if args.async_eval:
        best_eval_return = process_eval_results(
            pending_evals,
            metrics_path,
            args.save_dir,
            best_eval_return,
            wait=True,
        )
    if eval_executor is not None:
        eval_executor.shutdown(wait=True)
    envs.close()


if __name__ == "__main__":
    main()





