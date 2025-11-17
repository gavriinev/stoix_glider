from __future__ import annotations

from enum import IntEnum
from math import ceil
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
from flax import struct


class ThermalMode(IntEnum):
    """Supported thermal center dynamics."""

    MOVING = 0
    FIXED = 1
    MOUNTAIN = 2


class DecayType(IntEnum):
    """Functional form for the thermal decay factor."""

    CONSTANT = 0
    EXPONENTIAL = 1
    LINEAR = 2
    QUADRATIC = 3


_MODE_LOOKUP = {
    "moving": ThermalMode.MOVING,
    "fixed": ThermalMode.FIXED,
    "mountain": ThermalMode.MOUNTAIN,
}

_DECAY_LOOKUP = {
    "constant": DecayType.CONSTANT,
    "exp": DecayType.EXPONENTIAL,
    "exponential": DecayType.EXPONENTIAL,
    "linear": DecayType.LINEAR,
    "quadratic": DecayType.QUADRATIC,
}


def _parse_mode(raw_mode: str | ThermalMode) -> ThermalMode:
    if isinstance(raw_mode, ThermalMode):
        return raw_mode
    lowered = raw_mode.lower()
    if lowered not in _MODE_LOOKUP:
        raise ValueError(f"Unsupported thermal mode '{raw_mode}'")
    return _MODE_LOOKUP[lowered]


def _parse_decay(decay_cfg: float | Sequence[float] | dict | None) -> tuple[DecayType, jnp.ndarray]:
    if decay_cfg is None:
        return DecayType.CONSTANT, jnp.zeros((2,), dtype=jnp.float32)

    if isinstance(decay_cfg, (float, int)):
        scale = float(decay_cfg)
        return DecayType.CONSTANT, jnp.array([scale, 0.0], dtype=jnp.float32)

    if isinstance(decay_cfg, dict):
        decay_type = decay_cfg.get("type", "constant")
        if decay_type not in _DECAY_LOOKUP:
            raise ValueError(f"Unsupported decay type '{decay_type}'")
        parsed = _DECAY_LOOKUP[decay_type]
        if parsed == DecayType.EXPONENTIAL:
            rate = float(decay_cfg.get("rate", 0.0))
            return parsed, jnp.array([rate, 0.0], dtype=jnp.float32)
        if parsed == DecayType.LINEAR:
            slope = float(decay_cfg.get("slope", 0.0))
            return parsed, jnp.array([slope, 0.0], dtype=jnp.float32)
        if parsed == DecayType.QUADRATIC:
            a = float(decay_cfg.get("a", 0.0))
            b = float(decay_cfg.get("b", 0.0))
            return parsed, jnp.array([a, b], dtype=jnp.float32)
        return parsed, jnp.zeros((2,), dtype=jnp.float32)

    if isinstance(decay_cfg, Sequence):
        coeffs = list(decay_cfg)
        padded = coeffs + [0.0] * (2 - len(coeffs))
        return DecayType.QUADRATIC, jnp.array(padded[:2], dtype=jnp.float32)

    raise TypeError("Unsupported decay configuration type.")


@struct.dataclass
class WindModel:
    """JAX-friendly representation of a thermal area."""

    centers: jax.Array
    w_stars: jax.Array
    thermal_heights: jax.Array
    modes: jax.Array
    decay_types: jax.Array
    decay_params: jax.Array
    horizontal_wind: jax.Array
    noise_schedule: jax.Array
    time_for_noise_change: float
    total_time: float

    @staticmethod
    def default() -> "WindModel":
        center = jnp.array([[0.0, 0.0, 500.0]], dtype=jnp.float32)
        key = jax.random.PRNGKey(0)
        # horizontal_wind = jnp.array([jax.random.uniform(key, minval=0.0, maxval=4.0), 0.0, 0.0], dtype=jnp.float32)
        horizontal_wind = jnp.array([0.0, 4.0, 0.0], dtype=jnp.float32)
        return WindModel(
            centers=center,
            w_stars=jnp.array([5.0], dtype=jnp.float32),
            thermal_heights=jnp.array([2000.0], dtype=jnp.float32),
            modes=jnp.array([ThermalMode.MOVING], dtype=jnp.int32),
            decay_types=jnp.array([DecayType.CONSTANT], dtype=jnp.int32),
            decay_params=jnp.zeros((1, 2), dtype=jnp.float32),
            horizontal_wind = horizontal_wind,
            noise_schedule=jnp.zeros((1, 3), dtype=jnp.float32),
            time_for_noise_change=100.0,
            total_time=200.0,
        )

    @property
    def num_thermals(self) -> int:
        return int(self.centers.shape[0])


def build_wind_model(
    x_bounds: Sequence[float] = (-500.0, 500.0),
    y_bounds: Sequence[float] = (-500.0, 500.0),
    z_bounds: Sequence[float] = (0.0, 1000.0),
    thermal_height: float = 2000.0,
    w_star: float = 5.0,
    horizontal_wind: Sequence[float] | None = None,
    noise_wind: Sequence[float] | None = None,
    mode: str | ThermalMode = "moving",
    total_time: float = 200.0,
    time_for_noise_change: float = 5.0,
    thermals_settings_list: Iterable[dict] | None = None,
    key: jax.Array | int | None = None,
) -> WindModel:
    horizontal = (
        jnp.array(horizontal_wind, dtype=jnp.float32)
        if horizontal_wind is not None
        else jnp.zeros((3,), dtype=jnp.float32)
    )

    default_center = jnp.array(
        [
            0.5 * (x_bounds[0] + x_bounds[1]),
            0.5 * (y_bounds[0] + y_bounds[1]),
            0.5 * (z_bounds[0] + z_bounds[1]),
        ],
        dtype=jnp.float32,
    )

    thermals = list(thermals_settings_list or [])
    if not thermals:
        thermals = [
            {
                "mode": mode,
                "center": default_center,
                "w_star": w_star,
                "thermal_height": thermal_height,
                "decay_by_time_factor": None,
            }
        ]

    centers = []
    w_stars = []
    heights = []
    modes = []
    decay_types = []
    decay_params = []

    for thermal in thermals:
        centers.append(jnp.array(thermal.get("center", default_center), dtype=jnp.float32))
        w_stars.append(float(thermal.get("w_star", w_star)))
        heights.append(float(thermal.get("thermal_height", thermal_height)))
        parsed_mode = _parse_mode(thermal.get("mode", mode))
        modes.append(int(parsed_mode))
        decay_type, decay_param = _parse_decay(thermal.get("decay_by_time_factor", None))
        decay_types.append(int(decay_type))
        decay_params.append(decay_param)

    centers_arr = jnp.stack(centers, axis=0)
    w_stars_arr = jnp.asarray(w_stars, dtype=jnp.float32)
    heights_arr = jnp.asarray(heights, dtype=jnp.float32)
    modes_arr = jnp.asarray(modes, dtype=jnp.int32)
    decay_types_arr = jnp.asarray(decay_types, dtype=jnp.int32)
    decay_params_arr = jnp.stack(decay_params, axis=0)

    if noise_wind is not None and any(float(v) != 0.0 for v in noise_wind):
        if key is None:
            raise ValueError("A PRNG key or seed is required when noise_wind is specified.")
        rng = jax.random.PRNGKey(key) if isinstance(key, int) else key
        steps = max(1, int(ceil(total_time / time_for_noise_change)))
        noise_scale = jnp.asarray(noise_wind, dtype=jnp.float32)
        noise_schedule = (
            jax.random.normal(rng, shape=(steps, 3), dtype=jnp.float32) * noise_scale
        )
    else:
        noise_schedule = jnp.zeros((1, 3), dtype=jnp.float32)

    return WindModel(
        centers=centers_arr,
        w_stars=w_stars_arr,
        thermal_heights=heights_arr,
        modes=modes_arr,
        decay_types=decay_types_arr,
        decay_params=decay_params_arr,
        horizontal_wind=horizontal,
        noise_schedule=noise_schedule,
        time_for_noise_change=time_for_noise_change,
        total_time=total_time,
    )


def _decay_factor(decay_types: jax.Array, decay_params: jax.Array, t: jax.Array) -> jax.Array:
    base = jnp.ones_like(decay_types, dtype=jnp.float32)
    base = jnp.where(
        decay_types == int(DecayType.CONSTANT),
        jnp.where(decay_params[:, 0] == 0.0, 1.0, decay_params[:, 0]),
        base,
    )
    exp_decay = jnp.exp(-jnp.clip(decay_params[:, 0], a_min=0.0) * t)
    linear_decay = jnp.maximum(0.0, 1.0 - decay_params[:, 0] * t)
    quad_decay = jnp.maximum(
        0.0, 1.0 + decay_params[:, 0] * t + decay_params[:, 1] * t * t
    )

    decay = base
    decay = jnp.where(decay_types == int(DecayType.EXPONENTIAL), exp_decay, decay)
    decay = jnp.where(decay_types == int(DecayType.LINEAR), linear_decay, decay)
    decay = jnp.where(decay_types == int(DecayType.QUADRATIC), quad_decay, decay)
    return decay


def _centers_xy(model: WindModel, z: jax.Array, t: jax.Array) -> jax.Array:
    base_xy = model.centers[:, :2]
    base_z = model.centers[:, 2]

    moving_xy = base_xy + model.horizontal_wind[:2] * t
    fixed_xy = base_xy

    denom = model.w_stars * 0.4576
    safe_denom = jnp.where(denom == 0.0, jnp.inf, denom)
    incline = model.horizontal_wind / safe_denom[:, None]
    mountain_xy = base_xy + (z - base_z)[:, None] * incline[:, :2]

    centers_xy = jnp.where(
        (model.modes == int(ThermalMode.MOVING))[:, None], moving_xy, fixed_xy
    )
    centers_xy = jnp.where(
        (model.modes == int(ThermalMode.MOUNTAIN))[:, None], mountain_xy, centers_xy
    )
    return centers_xy


def _lenschow_model(
    r: jax.Array, z: jax.Array, zi: jax.Array, w_star: jax.Array
) -> jax.Array:
    cond = (z > 0.0) & (z <= zi)
    z_ratio = jnp.where(cond, z / zi, 0.0)
    z_third = jnp.power(jnp.clip(z_ratio, a_min=0.0), 1.0 / 3.0)
    d = 0.16 * z_third * (1.0 - 0.25 * z_ratio) * zi
    d_sq = jnp.square(jnp.maximum(d * 0.5, 1e-3))
    w_peak = w_star * z_third * (1.0 - 1.1 * z_ratio)

    r_sq = jnp.square(r)
    exponent = jnp.exp(-r_sq / d_sq)
    profile = 1.0 - r_sq / d_sq
    w_total = w_peak * exponent * profile
    return jnp.where(cond, w_total, 0.0)


def _sample_noise(model: WindModel, t: jax.Array) -> jax.Array:
    if model.noise_schedule.size == 0:
        return jnp.zeros((3,), dtype=jnp.float32)

    steps = model.noise_schedule.shape[0]
    if steps == 1:
        return model.noise_schedule[0]

    idx = jnp.floor(t / model.time_for_noise_change).astype(jnp.int32)
    idx = jnp.mod(idx, steps)
    return model.noise_schedule[idx]


def _wind_single(model: WindModel, position: jax.Array, t: jax.Array) -> jax.Array:
    z = position[2]
    centers_xy = _centers_xy(model, z, t)
    r = jnp.linalg.norm(position[:2] - centers_xy, axis=1)
    zi = model.thermal_heights
    w_star = model.w_stars
    decay = _decay_factor(model.decay_types, model.decay_params, t)
    w_total = _lenschow_model(r, z, zi, w_star) * decay
    thermal_wind = jnp.stack(
        [jnp.zeros_like(w_total), jnp.zeros_like(w_total), w_total], axis=1
    )
    return (
        jnp.sum(thermal_wind, axis=0)
        + model.horizontal_wind
        + _sample_noise(model, t)
    )


def wind_at(model: WindModel, position: jax.Array, t: jax.Array) -> jax.Array:
    """Compute wind velocity at ``position`` and time ``t``.

    Parameters
    ----------
    model:
        Wind model description returned by :func:`build_wind_model`.
    position:
        Either a single world position ``(x, y, z)`` or a batch of positions with
        shape ``(batch, 3)``.
    t:
        Scalar simulation time or a batch of times matching ``position``.
    """

    position = jnp.asarray(position, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)

    if position.ndim == 1:
        return _wind_single(model, position, t)

    if t.ndim == 0:
        t = jnp.broadcast_to(t, (position.shape[0],))

    return jax.vmap(lambda p, ti: _wind_single(model, p, ti))(position, t)


def thermal_centers(model: WindModel, z: jax.Array, t: jax.Array) -> jax.Array:
    """Return thermal core coordinates at altitude ``z`` and time ``t``.

    This utility mirrors the internal center evolution logic used by :func:`wind_at`
    and handles broadcasting for batched ``z``/``t`` inputs. When ``z`` and ``t`` are
    scalars, the result has shape ``(num_thermals, 3)``. Otherwise the shape is the
    broadcasted shape of ``z`` and ``t`` followed by ``(num_thermals, 3)``.
    """

    z = jnp.asarray(z, dtype=jnp.float32)
    t = jnp.asarray(t, dtype=jnp.float32)

    def _centers_for(z_val: jax.Array, t_val: jax.Array) -> jax.Array:
        xy = _centers_xy(model, z_val, t_val)
        z_col = jnp.broadcast_to(z_val, (model.num_thermals, 1))
        return jnp.concatenate([xy, z_col], axis=1)
   

    broadcast_shape = jnp.broadcast_shapes(z.shape, t.shape)
    z_b = jnp.broadcast_to(z, broadcast_shape).reshape(-1)
    t_b = jnp.broadcast_to(t, broadcast_shape).reshape(-1)

    centers = jax.vmap(_centers_for)(z_b, t_b)
    return centers.reshape(broadcast_shape + (model.num_thermals, 3))
