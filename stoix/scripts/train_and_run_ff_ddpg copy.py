"""Utility script to train a feed-forward DDPG agent and run evaluations.

This script composes a Hydra configuration, launches the standard Stoix DDPG
training loop, and then runs a configurable number of evaluation episodes with
the freshly trained policy. It is intended as a lightweight convenience wrapper
around the existing training entrypoint that ships with Stoix.
"""



from __future__ import annotations

import argparse
import copy
from pathlib import Path
from pprint import pprint
from typing import Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from flax import jax_utils
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf

from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.systems.ddpg.ff_ddpg import run_experiment
from stoix.utils import make_env as environments
from stoix.utils.jax_utils import unreplicate_batch_dim

import plotly.express as px
import plotly.io as pio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_CONFIG = "default/anakin/default_ff_ddpg"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Stoix feed-forward DDPG agent and run post-training evaluations. "
            "Configuration overrides use standard Hydra syntax (e.g. \"arch.seed=7\")."
        )
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG,
        help=(
            "Relative Hydra config path to load (default: %(default)s). "
            "Paths are resolved relative to `stoix/configs`."
        ),
    )
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        default=[],
        help=(
            "Hydra-style config override. May be supplied multiple times, e.g. "
            "-o arch.seed=123 -o env.scenario.name=cartpole."
        ),
    )
    parser.add_argument(
        "--rollout-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to run after training (default: %(default)s).",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1,
        help=(
            "Additional offset added to the training seed for post-training evaluations. "
            "Useful when running multiple evaluations without retraining."
        ),
    )
    return parser.parse_args()


def _resolve_config_location(config_name: str) -> Tuple[Path, str]:
    """Split a config reference into directory and base filename.

    Hydra expects `config_name` to be the filename (without extension) scoped to the
    directory provided during initialization. For convenience, we allow callers to
    specify paths like ``default/anakin/default_ff_ddpg`` and map them to the
    corresponding directory and config name automatically.
    """

    config_path = Path(config_name)
    if config_path.suffix:
        # Strip optional extension so users can pass `foo/bar.yaml` if desired.
        config_path = config_path.with_suffix("")

    if config_path.parent == Path("."):
        return CONFIG_ROOT, config_path.name

    resolved_dir = CONFIG_ROOT / config_path.parent
    return resolved_dir, config_path.name


def _load_config(config_name: str, overrides: Iterable[str]) -> DictConfig:
    """Compose a Hydra configuration with optional overrides."""
    override_list = list(overrides)
    config_dir, resolved_name = _resolve_config_location(config_name)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.2", config_dir=str(config_dir)):
        cfg = compose(config_name=resolved_name, overrides=override_list)
    OmegaConf.set_struct(cfg, False)
    return cfg


def _summarize_metrics(metrics: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    """Return mean and standard deviation for each logged metric."""
    summary: Dict[str, Tuple[float, float]] = {}
    for key, values in metrics.items():
        arr = np.asarray(values)
        summary[key] = (float(np.mean(arr)), float(np.std(arr)))
    return summary


def _evaluate_policy(
    config: DictConfig,
    actor_apply_fn,
    actor_params,
    num_episodes: int,
    seed_offset: int,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, object]]]:
    """Roll out the trained policy for a handful of evaluation episodes."""
    eval_config = copy.deepcopy(config)
    eval_config.arch.num_eval_episodes = max(1, num_episodes)

    _, eval_env = environments.make(config=eval_config)
    eval_key = jax.random.PRNGKey(eval_config.arch.seed + seed_offset)

    replicated_params = _replicate_params_for_evaluator(actor_params, eval_config)
    evaluator, _, (broadcast_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=eval_key,
        eval_act_fn=get_distribution_act_fn(eval_config, actor_apply_fn),
        params=replicated_params,
        config=eval_config,
    )

    eval_output = evaluator(broadcast_params, eval_keys)
    eval_metrics = jax.device_get(eval_output.episode_metrics)
    detailed_rollout = _collect_rollout_details(
        config=config,
        actor_apply_fn=actor_apply_fn,
        actor_params=actor_params,
        num_episodes=num_episodes,
        seed_offset=seed_offset,
    )
    return {name: np.asarray(value) for name, value in eval_metrics.items()}, detailed_rollout


def _ensure_buffer_capacity(config: DictConfig) -> None:
    """Expand the replay buffer if warmup would overflow it on this hardware."""

    n_devices = max(1, jax.local_device_count())
    shard_factor = n_devices * max(1, int(config.arch.update_batch_size))

    total_num_envs = int(config.arch.total_num_envs)
    if total_num_envs % shard_factor != 0:
        # Defer to downstream assertions (check_total_timesteps) if the config is invalid.
        return

    envs_per_shard = total_num_envs // shard_factor
    warmup_per_shard = int(config.system.warmup_steps) * envs_per_shard
    minimum_total = warmup_per_shard * shard_factor

    if config.system.total_buffer_size >= minimum_total:
        return

    new_total = minimum_total
    remainder = new_total % shard_factor
    if remainder:
        new_total += shard_factor - remainder

    print(
        "Auto-adjusting system.total_buffer_size from "
        f"{config.system.total_buffer_size} to {new_total} to accommodate warmup "
    f"({config.system.warmup_steps} steps × {envs_per_shard} envs per shard)."
    )
    config.system.total_buffer_size = new_total


def _replicate_params_for_evaluator(params: FrozenDict, config: DictConfig) -> FrozenDict:
    """Match the shape expected by evaluator_setup (device × batch × ...)."""

    expected_prefix = (jax.local_device_count(), max(1, int(config.arch.update_batch_size)))
    leaves = jax.tree_util.tree_leaves(params)
    if leaves:
        lead_ndim = len(expected_prefix)
        first = leaves[0]
        if first.ndim >= lead_ndim and tuple(first.shape[:lead_ndim]) == expected_prefix:
            return params

    base_params = jax.tree_util.tree_map(jnp.asarray, params)
    update_batch = max(1, int(config.arch.update_batch_size))

    if update_batch == 1:
        batched_params = jax.tree_util.tree_map(lambda x: x[None, ...], base_params)
    else:
        batched_params = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None, ...], (update_batch,) + x.shape),
            base_params,
        )

    return jax_utils.replicate(batched_params)


def _extract_single_actor_params(params: FrozenDict) -> FrozenDict:
    """Strip device and batch axes from replicated actor parameters."""

    leaves = jax.tree_util.tree_leaves(params)
    if not leaves:
        return params

    candidate = params
    if leaves[0].ndim >= 2:
        candidate = unreplicate_batch_dim(candidate)

    leaves = jax.tree_util.tree_leaves(candidate)
    if leaves and leaves[0].ndim >= 1:
        candidate = jax.tree_util.tree_map(lambda x: x[0], candidate)

    return jax.tree_util.tree_map(jnp.asarray, candidate)


def _tree_to_serializable(tree: object) -> object:
    """Convert a pytree of JAX/NumPy arrays into Python-native containers."""

    device_fetched = jax.device_get(tree)

    def _convert(x: object) -> object:
        if isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)):
            return np.asarray(x).tolist()
        return x

    return jax.tree_util.tree_map(_convert, device_fetched)


def _collect_rollout_details(
    config: DictConfig,
    actor_apply_fn,
    actor_params: FrozenDict,
    num_episodes: int,
    seed_offset: int,
) -> List[Dict[str, object]]:
    """Generate a detailed rollout trace with full environment state and extras."""

    if num_episodes <= 0:
        return []

    detail_cfg = copy.deepcopy(config)
    detail_cfg.arch.total_num_envs = 1
    detail_cfg.arch.num_envs = 1
    detail_cfg.arch.update_batch_size = 1
    detail_cfg.arch.num_eval_episodes = 1
    detail_cfg.num_devices = 1

    _, eval_env = environments.make(config=detail_cfg)

    params_single = _extract_single_actor_params(actor_params)
    act_fn = get_distribution_act_fn(detail_cfg, actor_apply_fn)

    rollout: List[Dict[str, object]] = []
    key = jax.random.PRNGKey(detail_cfg.arch.seed + seed_offset)
    key, reset_key = jax.random.split(key)
    env_state, timestep = eval_env.reset(reset_key)

    max_steps = 200  # safeguard against infinite loops
    episodes = 0
    step_counter = 0

    current_state = env_state
    current_timestep = timestep

    while episodes < num_episodes and step_counter < max_steps:
        obs = jax.tree_util.tree_map(lambda x: jnp.asarray(x), current_timestep.observation)
        batched_obs = jax.tree_util.tree_map(lambda x: x[None, ...], obs)

        key, actor_key = jax.random.split(key)
        action = act_fn(params_single, batched_obs, actor_key)
        action_array = jnp.asarray(action)

        next_state, next_timestep = eval_env.step(current_state, action_array.squeeze(0))

        step_record: Dict[str, object] = {
            "step": step_counter,
            "env_state": _tree_to_serializable(current_state),
            "observation": _tree_to_serializable(obs),
            "action": np.asarray(action_array.squeeze(0)).tolist(),
            "reward": float(np.asarray(next_timestep.reward).squeeze()),
            "discount": float(np.asarray(next_timestep.discount).squeeze()),
            "extras": _tree_to_serializable(next_timestep.extras),
            "terminal": bool(np.asarray(next_timestep.last()).squeeze()),
            "next_env_state": _tree_to_serializable(next_state),
        }
        rollout.append(step_record)

        step_counter += 1
        if step_record["terminal"]:
            episodes += 1
            if episodes >= num_episodes:
                break

        current_state = next_state
        current_timestep = next_timestep

    if step_counter >= max_steps:
        print(
            "Warning: rollout collection reached max_steps limit before completing the requested"
            " number of episodes."
        )

    return rollout


def _print_rollout_details(rollout: List[Dict[str, object]]) -> None:
    if not rollout:
        print("\nNo detailed rollout data captured.")
        return

    print("\nDetailed rollout trace (per step):")
    for record in rollout:
        print(f"\nStep {record['step']}:")
        pprint({k: v for k, v in record.items() if k != "step"})


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config_name, args.overrides)

    _ensure_buffer_capacity(cfg)

    print("Starting DDPG training with configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    eval_score, actor_apply_fn, actor_params = run_experiment(cfg, return_actor=True)
    print(f"Training complete. Final evaluation metric ({cfg.env.eval_metric}): {eval_score:.4f}")

    print(
        f"Running {args.rollout_episodes} post-training evaluation episode(s) using seed offset "
        f"{args.seed_offset}..."
    )
    metrics, rollout_details = _evaluate_policy(
        config=cfg,
        actor_apply_fn=actor_apply_fn,
        actor_params=actor_params,
        num_episodes=args.rollout_episodes,
        seed_offset=args.seed_offset,
    )

    summary = _summarize_metrics(metrics)
    print("\nAggregated evaluation metrics (mean ± std):")
    for name, (mean, std) in summary.items():
        print(f"  {name:20s} : {mean:.4f} ± {std:.4f}")

    if "episode_return" in metrics:
        flat_returns = np.asarray(metrics["episode_return"]).reshape(-1)
        preview = ", ".join(f"{ret:.2f}" for ret in flat_returns[: min(10, len(flat_returns))])
        print(f"\nPer-episode returns (first {min(10, len(flat_returns))} shown): {preview}")
    x = []
    y = []
    z = []
    # _print_rollout_details(rollout_details)

    x = [record['env_state'].position[0] for record in rollout_details]
    y = [record['env_state'].position[1] for record in rollout_details]
    z = [record['env_state'].position[2] for record in rollout_details]

    # speed is now a scalar (forward speed), calculate vertical_speed from speed and glide angle
    speed = [record['env_state'].speed for record in rollout_details]
    glide_angle = [record['env_state'].attitude[0] for record in rollout_details]
    side_angle = [record['env_state'].attitude[1] for record in rollout_details]
    # Calculate vertical_speed = speed * sin(glide_angle)
    vertical_speed = [s * np.sin(g) for s, g in zip(speed, glide_angle)]
    bank_control = [record['env_state'].controls[0] for record in rollout_details]
    attack_control = [record['env_state'].controls[1] for record in rollout_details]
    sideslip_control = [record['env_state'].controls[2] for record in rollout_details]
    # thermal_centers_x = [record['extras']['thermal_centers_cord'][0] for record in rollout_details]
    # thermal_centers_y = [record['extras']['thermal_centers_cord'][1] for record in rollout_details]
    rewards = [record['reward'] for record in rollout_details]
    times = [record['step'] for record in rollout_details]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    print(bank_control)
    print(attack_control)
    labels = [
		(x, "Position X (m)"),
		(y, "Position Y (m)"),
		(z, "Altitude Z (m)"),
		(speed, "Speed (m/s)"),
		(vertical_speed, "Vertical Speed (m/s)"),
		(glide_angle, "Glide Angle (rad)"),
		(side_angle, "Side Angle (rad)"),
		(bank_control, "Bank Control (rad)"),
		(attack_control, "Attack Control (rad)"),
		(sideslip_control, "Sideslip Control (rad)"),
        (rewards, "Reward (m)"),
	]

    for ax, (data, title) in zip(axes, labels):
        ax.plot(times, data, linewidth=1.0)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Glider State Trajectories")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_dir = Path("outputs") / "glider_rollout"
    timeseries_path = output_dir / "state_timeseries.png"
    fig.savefig(timeseries_path, dpi=150)
    plt.close(fig)

    figg = px.line_3d(x=x, y=y, z=z)
    figg.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X Position (m)'),
            yaxis=dict(title='Y Position (m)'),
            zaxis=dict(title='Altitude Z (m)')
        )
    )
    pio.write_html(figg, file="optimized_simulation.html", auto_open=True)

    print("\nDone. Logs, checkpoints, and evaluation artifacts are available under \"results/\".")


if __name__ == "__main__":
    main()
