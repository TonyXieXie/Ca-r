from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import random
import shutil
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.vector.vector_env import AutoresetMode
from torch import Tensor
from torch.optim import Adam

from carracing_observation import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OBS_SOURCE,
    make_carracing_env,
)
from ppo_pixel_policy import CarRacingPPOPolicy

BEST_MODEL_FILENAME = "best_model.pt"
EVAL_RESULTS_FILENAME = "eval_results.jsonl"
EVAL_STATUS_FILENAME = "eval_status.json"
EVAL_SNAPSHOT_GLOB = "eval_snapshot_*.pt"
EVAL_SEED_OFFSET = 10_000
EVAL_SEED_STRIDE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Gymnasium CarRacing-v3 from pixels.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--vector-backend", type=str, default="sync", choices=["sync", "async"])
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--obs-source",
        type=str,
        default=DEFAULT_OBS_SOURCE,
        choices=["state_pixels", "rgb_array"],
    )
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
    parser.add_argument("--eval-mode", type=str, default="async", choices=["sync", "async"])
    parser.add_argument("--eval-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available.")
    parser.add_argument("--fail-on-nonfinite", action="store_true", help="Abort training if NaN/Inf appears.")
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_unlink(path: Path | str) -> None:
    target = Path(path)
    try:
        if target.exists():
            target.unlink()
    except FileNotFoundError:
        pass


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def minimal_checkpoint_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_frames": int(args.num_frames),
        "image_size": int(args.image_size),
        "obs_source": str(args.obs_source),
        "seed": int(args.seed),
        "domain_randomize": bool(args.domain_randomize),
    }


def freeze_eval_spec(args: argparse.Namespace, eval_device: torch.device) -> dict[str, Any]:
    eval_num_envs = max(1, int(args.eval_episodes))
    return {
        "device": str(eval_device),
        "num_frames": int(args.num_frames),
        "image_size": int(args.image_size),
        "obs_source": str(args.obs_source),
        "eval_episodes": int(args.eval_episodes),
        "eval_num_envs": eval_num_envs,
        "vector_backend": "async" if eval_num_envs > 1 else "sync",
        "domain_randomize": bool(args.domain_randomize),
        "deterministic": True,
        "seed_base": int(args.seed),
        "seed_offset": int(EVAL_SEED_OFFSET),
        "seed_stride": int(EVAL_SEED_STRIDE),
        "episode_stride": 1,
    }


def compute_eval_seed(eval_spec: dict[str, Any], update: int) -> int:
    return int(eval_spec["seed_base"]) + int(eval_spec["seed_offset"]) + update * int(eval_spec["seed_stride"])


def cleanup_eval_temp_files(save_dir: Path) -> None:
    for snapshot in save_dir.glob(EVAL_SNAPSHOT_GLOB):
        safe_unlink(snapshot)


def cleanup_cuda_resources(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_payload(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)

    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return payload


def make_env(
    env_id: str,
    idx: int,
    seed: int,
    capture_video: bool,
    run_dir: Path,
    domain_randomize: bool,
    image_size: int,
    obs_source: str,
):
    def thunk():
        render_mode = "rgb_array" if capture_video or obs_source == "rgb_array" else None
        env = make_carracing_env(
            domain_randomize=domain_randomize,
            render_mode=render_mode,
            obs_source=obs_source,
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


def init_frame_stack(obs: np.ndarray, num_frames: int) -> np.ndarray:
    return np.repeat(obs[:, None, ...], repeats=num_frames, axis=1)


def update_frame_stack(stacked_obs: np.ndarray, next_obs: np.ndarray, done: np.ndarray) -> np.ndarray:
    stacked_obs = np.roll(stacked_obs, shift=-1, axis=1)
    stacked_obs[:, -1] = next_obs
    if np.any(done):
        stacked_obs[done] = np.repeat(next_obs[done, None, ...], repeats=stacked_obs.shape[1], axis=1)
    return stacked_obs


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
    eval_spec: dict[str, Any],
    update: int,
    global_step: int,
    eval_seed: int,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = save_dir / f"eval_snapshot_{update:05d}.pt"
    policy_to_save = getattr(policy, "_orig_mod", policy)
    payload = {
        "snapshot_type": "async_eval",
        "model": policy_to_save.state_dict(),
        "args": minimal_checkpoint_args(args),
        "eval_spec": dict(eval_spec),
        "update": int(update),
        "global_step": int(global_step),
        "eval_seed": int(eval_seed),
    }
    torch.save(payload, snapshot_path)
    return snapshot_path


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
    image_size: int,
    obs_source: str,
    eval_episodes: int,
    seed: int,
    domain_randomize: bool,
    deterministic: bool = True,
    vector_backend: str = "sync",
    eval_num_envs: int | None = None,
) -> dict[str, float]:
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")

    returns = np.zeros(eval_episodes, dtype=np.float64)
    lengths = np.zeros(eval_episodes, dtype=np.int32)
    finished = np.zeros(eval_episodes, dtype=np.bool_)

    render_mode = "rgb_array" if obs_source == "rgb_array" else None
    eval_num_envs = max(1, min(eval_episodes, eval_num_envs or eval_episodes))

    def make_eval_env(env_seed: int):
        def thunk():
            env = make_carracing_env(
                domain_randomize=domain_randomize,
                render_mode=render_mode,
                obs_source=obs_source,
                image_size=image_size,
                continuous=True,
            )
            env.reset(seed=env_seed)
            env.action_space.seed(env_seed)
            return env

        return thunk

    env_fns = [make_eval_env(seed + idx) for idx in range(eval_num_envs)]
    vector_env_cls = AsyncVectorEnv if vector_backend == "async" else SyncVectorEnv
    envs = vector_env_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    try:
        obs, _ = envs.reset(seed=[seed + idx for idx in range(eval_num_envs)])
        stacked_obs = init_frame_stack(obs, num_frames)
        episode_slots = np.arange(eval_num_envs, dtype=np.int32)
        next_episode_id = eval_num_envs

        while not np.all(finished):
            with torch.no_grad():
                obs_tensor = torch.from_numpy(stacked_obs).to(device)
                step = policy.act(obs_tensor, deterministic=deterministic)
            actions = step.action.cpu().numpy()
            next_obs, reward, terminated, truncated, infos = envs.step(actions)
            done = np.logical_or(terminated, truncated)

            active_mask = episode_slots >= 0
            active_slots = episode_slots[active_mask]
            returns[active_slots] += reward[active_mask]
            lengths[active_slots] += 1

            done_indices = np.flatnonzero(done)
            for env_idx in done_indices:
                episode_id = int(episode_slots[env_idx])
                if episode_id >= 0:
                    finished[episode_id] = True
                if next_episode_id < eval_episodes:
                    episode_slots[env_idx] = next_episode_id
                    next_episode_id += 1
                else:
                    episode_slots[env_idx] = -1

            stacked_obs = update_frame_stack(stacked_obs, next_obs, done)
            if np.any(episode_slots < 0):
                inactive = episode_slots < 0
                actions[inactive] = 0.0
                stacked_obs[inactive] = 0
    finally:
        envs.close()

    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_length_mean": float(np.mean(lengths)),
    }


def evaluate_snapshot(snapshot_path: Path, eval_spec: dict[str, Any], eval_seed: int) -> dict[str, float]:
    device = torch.device(str(eval_spec["device"]))
    payload = load_model_payload(snapshot_path, device)
    policy = CarRacingPPOPolicy(
        num_frames=int(eval_spec["num_frames"]),
        image_size=int(eval_spec["image_size"]),
    ).to(device)
    policy.load_state_dict(payload["model"])
    policy.eval()

    try:
        return evaluate_policy(
            policy=policy,
            device=device,
            num_frames=int(eval_spec["num_frames"]),
            image_size=int(eval_spec["image_size"]),
            obs_source=str(eval_spec["obs_source"]),
            eval_episodes=int(eval_spec["eval_episodes"]),
            seed=int(eval_seed),
            domain_randomize=bool(eval_spec["domain_randomize"]),
            deterministic=bool(eval_spec["deterministic"]),
            vector_backend=str(eval_spec["vector_backend"]),
            eval_num_envs=int(eval_spec["eval_num_envs"]),
        )
    finally:
        del policy
        del payload
        cleanup_cuda_resources(device)


def async_eval_worker(request_queue: Any, result_queue: Any) -> None:
    while True:
        request = request_queue.get()
        if request is None:
            break

        snapshot_path = Path(request["snapshot_path"])
        response = {
            "update": int(request["update"]),
            "global_step": int(request["global_step"]),
            "checkpoint": str(snapshot_path),
            "eval_seed": int(request["eval_seed"]),
        }

        try:
            metrics = evaluate_snapshot(
                snapshot_path=snapshot_path,
                eval_spec=dict(request["eval_spec"]),
                eval_seed=int(request["eval_seed"]),
            )
            response.update(metrics)
            response["status"] = "ok"
        except Exception as exc:
            response["status"] = "error"
            response["error"] = repr(exc)
        finally:
            response["completed_at"] = utc_now_iso()
            result_queue.put(response)

    cleanup_cuda_resources(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class AsyncEvalManager:
    def __init__(
        self,
        save_dir: Path,
        eval_results_path: Path,
        eval_status_path: Path,
        eval_spec: dict[str, Any],
    ) -> None:
        self.save_dir = save_dir
        self.eval_results_path = eval_results_path
        self.eval_status_path = eval_status_path
        self.eval_spec = dict(eval_spec)
        self._ctx = mp.get_context("spawn")
        self._request_queue = self._ctx.Queue(maxsize=1)
        self._result_queue = self._ctx.Queue()
        self._pending: deque[dict[str, Any]] = deque()
        self._in_flight: dict[str, Any] | None = None
        self._worker: mp.Process | None = None
        self._closed = False
        self.last_completed_update: int | None = None
        self._start_worker()
        self._write_status()

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = self._ctx.Process(
            target=async_eval_worker,
            args=(self._request_queue, self._result_queue),
            daemon=False,
        )
        self._worker.start()

    def _worker_alive(self) -> bool:
        return bool(self._worker is not None and self._worker.is_alive())

    def _write_status(self) -> None:
        status = {
            "mode": "async",
            "worker_alive": self._worker_alive(),
            "queue_depth": len(self._pending) + (1 if self._in_flight is not None else 0),
            "pending_queue_depth": len(self._pending),
            "in_flight_update": None if self._in_flight is None else int(self._in_flight["update"]),
            "last_completed_eval_update": self.last_completed_update,
            "updated_at": utc_now_iso(),
        }
        write_json(self.eval_status_path, status)

    def _dispatch_next(self) -> None:
        if self._in_flight is not None or not self._pending:
            return
        self._start_worker()
        request = self._pending.popleft()
        self._request_queue.put(request)
        self._in_flight = request
        self._write_status()

    def enqueue(self, request: dict[str, Any]) -> None:
        self.poll_results()
        self._pending.append(dict(request))
        self._dispatch_next()
        self._write_status()

    def _handle_worker_failure(self, reason: str) -> dict[str, Any] | None:
        if self._in_flight is None:
            self._start_worker()
            self._write_status()
            return None

        failed_request = self._in_flight
        self._in_flight = None
        safe_unlink(failed_request["snapshot_path"])
        if self._worker is not None:
            self._worker.join(timeout=0.1)
        self._start_worker()
        self._dispatch_next()
        self._write_status()
        return {
            "update": int(failed_request["update"]),
            "global_step": int(failed_request["global_step"]),
            "checkpoint": str(failed_request["snapshot_path"]),
            "eval_seed": int(failed_request["eval_seed"]),
            "status": "error",
            "error": reason,
            "completed_at": utc_now_iso(),
        }

    def poll_results(self) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []

        while True:
            try:
                result = self._result_queue.get_nowait()
            except Empty:
                break
            completed.append(result)
            self.last_completed_update = int(result["update"])
            self._in_flight = None
            self._dispatch_next()

        if not self._worker_alive():
            failure = self._handle_worker_failure("async eval worker exited unexpectedly")
            if failure is not None:
                completed.append(failure)

        if completed:
            self._write_status()
        return completed

    def close(self, drain: bool = True) -> list[dict[str, Any]]:
        if self._closed:
            return []

        completed: list[dict[str, Any]] = []
        if drain:
            while self._pending or self._in_flight is not None:
                polled = self.poll_results()
                if polled:
                    completed.extend(polled)
                    continue
                time.sleep(0.05)
        else:
            while True:
                polled = self.poll_results()
                if not polled:
                    break
                completed.extend(polled)

        if self._worker_alive():
            self._request_queue.put(None)
            if self._worker is not None:
                self._worker.join(timeout=5)
                if self._worker.is_alive():
                    self._worker.terminate()
                    self._worker.join(timeout=5)

        self._request_queue.close()
        self._result_queue.close()
        self._closed = True
        self._write_status()
        return completed


def promote_best_model(snapshot_path: Path, best_model_path: Path) -> Path:
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot_path, best_model_path)
    return best_model_path


def process_eval_result(
    result: dict[str, Any],
    eval_results_path: Path,
    best_model_path: Path,
    best_eval_return: float,
) -> float:
    is_best_model = False
    checkpoint_path = Path(result["checkpoint"])
    eval_return_mean = result.get("eval_return_mean")

    if result.get("status") == "ok" and isinstance(eval_return_mean, (int, float)):
        eval_return_value = float(eval_return_mean)
        if eval_return_value > best_eval_return:
            promote_best_model(checkpoint_path, best_model_path)
            best_eval_return = eval_return_value
            is_best_model = True

    record = dict(result)
    record["is_best_model"] = is_best_model
    if is_best_model:
        record["best_model"] = str(best_model_path)
        record["best_eval_return"] = best_eval_return
    append_jsonl(eval_results_path, record)
    safe_unlink(checkpoint_path)
    return best_eval_return


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.torch_deterministic)
    device = choose_device(args.device)
    eval_device = choose_device(args.eval_device)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.save_dir / "metrics.jsonl"
    eval_results_path = args.save_dir / EVAL_RESULTS_FILENAME
    eval_status_path = args.save_dir / EVAL_STATUS_FILENAME
    best_model_path = args.save_dir / BEST_MODEL_FILENAME
    config_path = args.save_dir / "config.json"
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    cleanup_eval_temp_files(args.save_dir)
    safe_unlink(eval_results_path)
    write_json(
        eval_status_path,
        {
            "mode": args.eval_mode,
            "worker_alive": False,
            "queue_depth": 0,
            "pending_queue_depth": 0,
            "in_flight_update": None,
            "last_completed_eval_update": None,
            "updated_at": utc_now_iso(),
        },
    )

    env_fns = [
        make_env(
            "CarRacing-v3",
            idx=i,
            seed=args.seed,
            capture_video=args.capture_video,
            run_dir=args.save_dir,
            domain_randomize=args.domain_randomize,
            image_size=args.image_size,
            obs_source=args.obs_source,
        )
        for i in range(args.num_envs)
    ]
    vector_env_cls = AsyncVectorEnv if args.vector_backend == "async" else SyncVectorEnv
    envs = vector_env_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)

    obs, _ = envs.reset(seed=args.seed)
    obs_shape = tuple(int(dim) for dim in obs.shape[1:])
    stacked_obs = init_frame_stack(obs, args.num_frames)

    policy = CarRacingPPOPolicy(num_frames=args.num_frames, image_size=obs_shape[0]).to(device)
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
        (args.num_steps, args.num_envs, args.num_frames, *obs_shape),
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
    start_time = time.time()
    best_eval_return = float("-inf")
    eval_spec = freeze_eval_spec(args, eval_device)
    async_eval_manager: AsyncEvalManager | None = None
    if args.eval_mode == "async" and args.eval_every > 0:
        async_eval_manager = AsyncEvalManager(
            save_dir=args.save_dir,
            eval_results_path=eval_results_path,
            eval_status_path=eval_status_path,
            eval_spec=eval_spec,
        )

    try:
        for update in range(1, num_updates + 1):
            if async_eval_manager is not None:
                for result in async_eval_manager.poll_results():
                    best_eval_return = process_eval_result(
                        result=result,
                        eval_results_path=eval_results_path,
                        best_model_path=best_model_path,
                        best_eval_return=best_eval_return,
                    )

            actor_losses: list[float] = []
            value_losses: list[float] = []
            entropy_values: list[float] = []
            total_losses: list[float] = []
            approx_kls: list[float] = []
            grad_norms: list[float] = []
            finite_ok = True

            # Linear learning rate annealing
            if args.lr_anneal:
                frac = 1.0 - (update - 1) / num_updates
                lr_now = frac * initial_lr
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now

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

            b_obs = torch.from_numpy(obs_buffer.reshape(batch_size, args.num_frames, *obs_shape)).to(device)
            b_actions = torch.from_numpy(action_buffer.reshape(batch_size, 3)).to(device)
            b_logprobs = torch.from_numpy(logprob_buffer.reshape(batch_size)).to(device)
            b_advantages = torch.from_numpy(advantages.reshape(batch_size)).to(device)
            b_returns = torch.from_numpy(returns.reshape(batch_size)).to(device)
            b_values = torch.from_numpy(value_buffer.reshape(batch_size)).to(device)

            batch_indices = np.arange(batch_size)
            clipfracs = []

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
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

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

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            ev = explained_variance(y_pred, y_true)
            sps = int(global_step / max(time.time() - start_time, 1e-6))

            log_data = {
                "update": int(update),
                "global_step": int(global_step),
                "sps": int(sps),
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
            }

            if args.eval_every > 0 and update % args.eval_every == 0:
                if args.eval_mode == "sync":
                    eval_seed = compute_eval_seed(eval_spec, update)
                    eval_metrics = evaluate_policy(
                        policy=policy,
                        device=eval_device,
                        num_frames=int(eval_spec["num_frames"]),
                        image_size=int(eval_spec["image_size"]),
                        obs_source=str(eval_spec["obs_source"]),
                        eval_episodes=int(eval_spec["eval_episodes"]),
                        seed=eval_seed,
                        domain_randomize=bool(eval_spec["domain_randomize"]),
                        deterministic=bool(eval_spec["deterministic"]),
                    )
                    log_data.update(eval_metrics)
                    sync_best_model = False
                    if eval_metrics["eval_return_mean"] > best_eval_return:
                        best_eval_return = eval_metrics["eval_return_mean"]
                        best_path = save_checkpoint(
                            args.save_dir,
                            policy,
                            optimizer,
                            args,
                            update,
                            global_step,
                            filename=BEST_MODEL_FILENAME,
                            extra={"best_eval_return": best_eval_return},
                        )
                        sync_best_model = True
                        log_data["best_model"] = str(best_path)
                        log_data["best_eval_return"] = float(best_eval_return)

                    eval_record = {
                        "update": int(update),
                        "global_step": int(global_step),
                        "checkpoint": str(best_model_path) if sync_best_model else None,
                        "eval_seed": int(eval_seed),
                        "status": "ok",
                        "completed_at": utc_now_iso(),
                        "is_best_model": sync_best_model,
                        **eval_metrics,
                    }
                    if sync_best_model:
                        eval_record["best_model"] = str(best_model_path)
                        eval_record["best_eval_return"] = float(best_eval_return)
                    append_jsonl(eval_results_path, eval_record)
                elif async_eval_manager is not None:
                    eval_seed = compute_eval_seed(eval_spec, update)
                    snapshot_path = save_eval_snapshot(
                        args.save_dir,
                        policy,
                        args,
                        eval_spec,
                        update,
                        global_step,
                        eval_seed,
                    )
                    async_eval_manager.enqueue(
                        {
                            "snapshot_path": str(snapshot_path),
                            "update": int(update),
                            "global_step": int(global_step),
                            "eval_seed": int(eval_seed),
                            "eval_spec": dict(eval_spec),
                        }
                    )

            if update % args.log_every == 0:
                print(json.dumps(log_data, ensure_ascii=False))
                append_jsonl(metrics_path, log_data)

            if update % args.save_every == 0 or update == num_updates:
                checkpoint_path = save_checkpoint(args.save_dir, policy, optimizer, args, update, global_step)
                print(json.dumps({"checkpoint": str(checkpoint_path)}, ensure_ascii=False))
    finally:
        if async_eval_manager is not None:
            for result in async_eval_manager.close(drain=True):
                best_eval_return = process_eval_result(
                    result=result,
                    eval_results_path=eval_results_path,
                    best_model_path=best_model_path,
                    best_eval_return=best_eval_return,
                )
        cleanup_eval_temp_files(args.save_dir)
        envs.close()


if __name__ == "__main__":
    main()
