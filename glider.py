"""JAX-friendly glider environment following the :mod:`Simulation.swimmer` template.

This module provides a lightweight aerodynamic model inspired by
``Simulation.glider_simulation.GliderFlight`` and the Gym wrapper in
``gym_glider.envs.glider_env_learning``.  The goal is to expose the same
JAX-first API used by the ``Swimmer`` environment while keeping the state and
controls relevant for soaring RL research.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from stoix.utils.wind import WindModel, thermal_centers, wind_at

from gymnax.environments import environment, spaces


DEG2RAD = jnp.pi / 180.0
RAD2DEG = 180.0 / jnp.pi


@struct.dataclass
class EnvState(environment.EnvState):
    """State container for the glider."""

    position: jax.Array  # (x, y, z)
    speed: jax.Array  # (forward_speed, vertical_speed)
    attitude: jax.Array  # (glide_angle, side_angle)
    controls: jax.Array  # (bank_angle, attack_angle, sideslip_angle)
    time: jax.Array  # integer step counter
    
    # History buffers for observations
    speed_history: jax.Array  # (history_seconds, 2) - forward_speed and vertical_speed
    controls_history: jax.Array  # (history_seconds, 2) - bank_angle and attack_angle
    angle_from_wind_history: jax.Array  # (history_seconds,) - angle_from_wind
    wind_velocity_history: jax.Array  # (1,) - wind velocity for last second


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Environment parameters controlling physics and limits."""

    dt: float = 0.01
    max_steps_in_episode: int = 200
    history_seconds: int = struct.field(pytree_node=False, default=8)  # Number of seconds of history to store

    # physical constants
    g: float = 9.81
    rho: float = 1.225
    mass: float = 5.0
    wingspan: float = 2.5
    aspect_ratio: float = 16.0

    # aerodynamic coefficients (mirroring ``GliderFlight`` defaults)
    e: float = 0.95
    a0: float = 0.1 * RAD2DEG
    alpha0: float = -2.5 * DEG2RAD
    V_H: float = 0.4
    V_V: float = 0.02
    C_d_0: float = 0.01
    C_d_L: float = 0.05
    C_L_min: float = 0.4
    C_D_F: float = 0.008
    C_D_T: float = 0.01
    C_D_E: float = 0.002

    # operational bounds
    horizontal_bound: float = 5_000.0
    vertical_bounds: tuple[float, float] = (0.0, 1_000.0)
    speed_bounds: tuple[float, float] = (3.0, 30.0)
    glide_limits: tuple[float, float] = (-25.0 * DEG2RAD, 45.0 * DEG2RAD)
    side_limits: tuple[float, float] = (-jnp.pi, jnp.pi)
    bank_limits: tuple[float, float] = (-50.0 * DEG2RAD, 50.0 * DEG2RAD)
    attack_limits: tuple[float, float] = (-30.0 * DEG2RAD, 30.0 * DEG2RAD)
    sideslip_limits: tuple[float, float] = (-50.0 * DEG2RAD, 50.0 * DEG2RAD)

    # control increments per unit action
    action_deltas: tuple[float, float, float] = (
        15.0 * DEG2RAD,
        10.0 * DEG2RAD,
        3.0 * DEG2RAD,
    )

    # observation scaling
    observation_clip: float = 5.0

    # initial conditions
    initial_altitude: float = 500.0
    initial_speed: float = 10.0

    # reward shaping
    vertical_speed_weight: float = 1.0
    speed_penalty_weight: float = 0.01

    wind_model: WindModel = struct.field(default_factory=WindModel.default)


def calculate_angle_from_wind(bank_angle: jax.Array, side_angle: jax.Array, params: EnvParams) -> jax.Array:
    """Calculate angle from wind based on bank angle, side angle and wind direction.
    
    This function replicates the logic from glider_simulation.py:
    angle_from_wind = (2 * (bank_angle >= 0) - 1) * (((side_angle - wind_angle) % 360) - 180)
    
    Args:
        bank_angle: Current bank angle in radians
        side_angle: Current side angle in radians  
        params: Environment parameters containing wind model
        
    Returns:
        Angle from wind in radians
    """
    # Get horizontal wind components
    horizontal_wind = params.wind_model.horizontal_wind
    
    # Calculate wind angle from horizontal wind components
    wind_angle = jnp.arctan2(horizontal_wind[1], horizontal_wind[0])
    
    # Convert angles to degrees for calculation (matching original implementation)
    bank_angle_deg = bank_angle * RAD2DEG
    side_angle_deg = side_angle * RAD2DEG  
    wind_angle_deg = wind_angle * RAD2DEG
    
    # Calculate angle from wind using original formula
    sign_factor = 2.0 * (bank_angle_deg >= 0) - 1.0
    angle_diff = ((side_angle_deg - wind_angle_deg) % 360.0) - 180.0
    angle_from_wind_deg = sign_factor * angle_diff
    
    # Convert back to radians
    angle_from_wind = angle_from_wind_deg * DEG2RAD
    
    return angle_from_wind


def calculate_wind_velocity(params: EnvParams) -> jax.Array:
    """Calculate wind velocity magnitude from horizontal wind components.
    
    Args:
        params: Environment parameters containing wind model
        
    Returns:
        Wind velocity magnitude
    """
    horizontal_wind = params.wind_model.horizontal_wind
    return jnp.linalg.norm(horizontal_wind)


def _wing_surface_area(params: EnvParams) -> jax.Array:
    return (params.wingspan**2) / params.aspect_ratio


def _tail_moment_arm(params: EnvParams) -> jax.Array:
    return 0.28 * params.wingspan


def _mean_aerodynamic_chord(params: EnvParams) -> jax.Array:
    return 1.03 * params.wingspan / params.aspect_ratio


def _fuselage_area(params: EnvParams) -> jax.Array:
    span = params.wingspan
    return 0.01553571429 * span**2 + 0.01950357142 * span - 0.01030412685


def _horizontal_tail_surface(params: EnvParams) -> jax.Array:
    S = _wing_surface_area(params)
    lt = _tail_moment_arm(params)
    c_bar = _mean_aerodynamic_chord(params)
    return params.V_H * c_bar * S / lt


def _vertical_tail_surface(params: EnvParams) -> jax.Array:
    S = _wing_surface_area(params)
    lt = _tail_moment_arm(params)
    return params.V_V * params.wingspan * S / lt


def _lift_curve_slope(params: EnvParams) -> jax.Array:
    return params.a0 / (1 + params.a0 / (jnp.pi * params.e * params.aspect_ratio))


def _yaw_curve_slope(params: EnvParams) -> jax.Array:
    S = _wing_surface_area(params)
    S_V = jnp.maximum(_vertical_tail_surface(params), 1e-6)
    AR_V = 0.5 * params.aspect_ratio
    return (params.a0 / (1 + params.a0 / (jnp.pi * params.e * AR_V))) * (S_V / S)


def calculate_forces(
    params: EnvParams, speed: jax.Array, attack_angle: jax.Array, sideslip_angle: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (drag, lift, side) force magnitudes."""

    S = _wing_surface_area(params)
    S_V = jnp.maximum(_vertical_tail_surface(params), 1e-6)
    S_F = _fuselage_area(params)
    S_T = _horizontal_tail_surface(params)

    C_L_alpha = _lift_curve_slope(params)
    C_C_beta = _yaw_curve_slope(params)

    attack_offset = attack_angle - params.alpha0
    C_L = C_L_alpha * attack_offset
    C_C = C_C_beta * sideslip_angle

    base_drag = (
        params.C_D_F * S_F / S
        + params.C_D_T * (S_T + S_V) / S
        + params.C_D_E
        + params.C_d_0
    )
    induced_drag = (C_L**2 + (C_C**2) * (S / S_V)) / (jnp.pi * params.e * params.aspect_ratio)
    C_D = base_drag + params.C_d_L * (C_L - params.C_L_min) ** 2 + induced_drag

    q = 0.5 * params.rho * speed**2

    drag = q * S * C_D
    lift = q * S * C_L
    side = q * S * C_C

    return drag, lift, side


def get_wind(position: jax.Array, t: jax.Array, params: EnvParams) -> jax.Array:
    """Return the ambient wind vector at ``position`` and simulation time ``t``."""

    return wind_at(params.wind_model, position, t)


def _rotation_matrix_x(angle: jax.Array) -> jax.Array:
        """Матрица вращения вокруг оси X"""
        cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
        return jnp.array([[1, 0, 0],
                        [0, cos_a, -sin_a],
                        [0, sin_a, cos_a]])

def _rotation_matrix_y(angle: jax.Array) -> jax.Array:
    """Матрица вращения вокруг оси Y"""
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[cos_a, 0, sin_a],
                    [0, 1, 0],
                    [-sin_a, 0, cos_a]])

def _rotation_matrix_z(angle: jax.Array) -> jax.Array:
    """Матрица вращения вокруг оси Z"""
    cos_a, sin_a = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]])


def _step_per_sec(state: EnvState, controls: jax.Array, params: EnvParams) -> tuple[jax.Array, jax.Array, jax.Array]:
    bank, attack, sideslip = controls

    unit_x = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    unit_y = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    unit_z = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)

    R_x_bank = _rotation_matrix_x(bank)
    R_z_v_to_b = _rotation_matrix_z(-sideslip)
    R_y_v_to_b = _rotation_matrix_y(attack)
    R_v_to_b = R_y_v_to_b @ R_z_v_to_b

    dt = jnp.asarray(params.dt, dtype=jnp.float32)
    mass = jnp.asarray(params.mass, dtype=jnp.float32)
    g = jnp.asarray(params.g, dtype=jnp.float32)
    speed_min = jnp.asarray(params.speed_bounds[0], dtype=jnp.float32)
    speed_max = jnp.asarray(params.speed_bounds[1], dtype=jnp.float32)
    glide_min = jnp.asarray(params.glide_limits[0], dtype=jnp.float32)
    glide_max = jnp.asarray(params.glide_limits[1], dtype=jnp.float32)

    def _safe_cos(angle: jax.Array) -> jax.Array:
        cos_val = jnp.cos(angle)
        return jnp.where(
            jnp.abs(cos_val) < 1e-6,
            jnp.where(cos_val >= 0, 1e-6, -1e-6),
            cos_val,
        )

    def body_fun(step_idx: int, carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        position, speed, attitude, current_time = carry

        speed = jnp.maximum(speed, 1e-6)
        glide_angle, side_angle = attitude

        R_y_glide = _rotation_matrix_y(-glide_angle)
        R_z_side = _rotation_matrix_z(side_angle)
        R_i_to_v_positive_glide = R_z_side @ R_y_glide @ R_x_bank

        v_vector_in_i = R_i_to_v_positive_glide @ (speed * unit_x)

        w_vector_in_i = get_wind(position, current_time, params)

        relative_velocity = v_vector_in_i - w_vector_in_i
        relative_velocity_norm = jnp.linalg.norm(relative_velocity)
        safe_relative_norm = jnp.maximum(relative_velocity_norm, 1e-6)
        relative_velocity_direction = relative_velocity / safe_relative_norm

        glide_angle_in_w = jnp.arcsin(jnp.clip(relative_velocity_direction[2], -1.0, 1.0))
        cos_glide_in_w = _safe_cos(glide_angle_in_w)
        side_angle_in_w = jnp.sign(relative_velocity_direction[1] / cos_glide_in_w) * jnp.arccos(
            jnp.clip(relative_velocity_direction[0] / cos_glide_in_w, -1.0, 1.0)
        )

        R_y_att = _rotation_matrix_y(glide_angle)
        R_z_att = _rotation_matrix_z(side_angle)
        R_i_to_v = R_z_att @ R_y_att @ R_x_bank

        R_z_b_to_m = _rotation_matrix_z(-side_angle_in_w)
        R_y_b_to_m = _rotation_matrix_y(-glide_angle_in_w)
        R_b_to_m = R_y_b_to_m @ R_z_b_to_m

        third_col_of_m = R_b_to_m @ (R_i_to_v @ (R_v_to_b @ unit_z))
        second_col_of_m = R_b_to_m @ (R_i_to_v @ (R_v_to_b @ unit_y))
        first_col_of_m = R_b_to_m @ (R_i_to_v @ (R_v_to_b @ unit_x))

        attack_angle_in_w = jnp.arcsin(jnp.clip(third_col_of_m[0], -1.0, 1.0))
        cos_attack_in_w = _safe_cos(attack_angle_in_w)
        bank_angle_in_w = jnp.sign(-third_col_of_m[1] / cos_attack_in_w) * jnp.arccos(
            jnp.clip(third_col_of_m[2] / cos_attack_in_w, -1.0, 1.0)
        )
        sideslip_angle_in_w = jnp.sign(second_col_of_m[0] / cos_attack_in_w) * jnp.arccos(
            jnp.clip(first_col_of_m[0] / cos_attack_in_w, -1.0, 1.0)
        )

        d_w, l_w, c_w = calculate_forces(
            params, safe_relative_norm, attack_angle_in_w, sideslip_angle_in_w
        )

        R_v_to_i = R_i_to_v.T
        R_x_w = _rotation_matrix_x(bank_angle_in_w)
        R_y_w = _rotation_matrix_y(glide_angle_in_w)
        R_z_w = _rotation_matrix_z(side_angle_in_w)
        R_i_to_w = R_z_w @ R_y_w @ R_x_w

        forces_temp = R_i_to_w @ jnp.array([-d_w, -c_w, -l_w], dtype=jnp.float32)
        forces_in_v = R_v_to_i @ forces_temp
        d_v, c_v, l_v = -forces_in_v

        sin_glide = jnp.sin(glide_angle)
        cos_glide = jnp.cos(glide_angle)
        sin_side = jnp.sin(side_angle)
        cos_side = jnp.cos(side_angle)

        dz = speed * sin_glide
        dx = speed * cos_side * cos_glide
        dy = speed * sin_side * cos_glide

        dv = -d_v / mass - g * sin_glide
        inv_speed = 1.0 / jnp.maximum(speed, 1e-6)
        d_glide = (c_v * jnp.sin(bank) + l_v * jnp.cos(bank)) * inv_speed / mass - g * cos_glide * inv_speed
        d_side = (l_v * jnp.sin(bank) - c_v * jnp.cos(bank)) * inv_speed / mass

        position = position + dt * jnp.array([dx, dy, dz], dtype=jnp.float32)
        speed = jnp.clip(speed + dt * dv, 1e-6, 1000)
        glide_angle = glide_angle + dt * d_glide
        side_angle = side_angle + dt * d_side
        attitude = jnp.array([glide_angle, side_angle], dtype=jnp.float32)
        current_time = current_time + dt

        return position, speed, attitude, current_time

    init_time = jnp.asarray(state.time, dtype=jnp.float32)
    init_carry = (state.position, state.speed, state.attitude, init_time)
    final_position, final_speed, final_attitude, _ = jax.lax.fori_loop(0, 100, body_fun, init_carry)

    return final_position, final_speed, final_attitude
        


# тут происходит нормализация наблюдений
def _concat_obs(state: EnvState, params: EnvParams) -> jax.Array:
    # Observation now contains:
    # - speed_history: (history_seconds, 2) = forward_speed, vertical_speed for last N seconds
    # - controls_history: (history_seconds, 2) = bank_angle, attack_angle for last N seconds  
    # - angle_from_wind_history: (history_seconds,) = angle_from_wind for last N seconds
    # - wind_velocity_history: (1,) = wind velocity for last second
    
    components = (
        jnp.reshape(state.speed_history, (-1,)),  # flatten to (history_seconds * 2,)
        jnp.reshape(state.controls_history, (-1,)),  # flatten to (history_seconds * 2,)
        jnp.reshape(state.angle_from_wind_history, (-1,)),  # flatten to (history_seconds,)
        jnp.reshape(state.wind_velocity_history, (-1,)),  # flatten to (1,)
    )
    obs = jnp.concatenate(components, axis=0)
    
    return obs

# непонятная херня
def _wrap_angle(angle: jax.Array) -> jax.Array:
    return ((angle + jnp.pi) % (2 * jnp.pi)) - jnp.pi


class Glider(environment.Environment[EnvState, EnvParams]):
    """Continuous-control environment for a soaring glider."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        #
        action = jnp.asarray(action, dtype=jnp.float32)
        action = jnp.clip(action, -1.0, 1.0)
        delta = action * jnp.array(params.action_deltas, dtype=jnp.float32)

        bank = jnp.clip(state.controls[0] + delta[0], *params.bank_limits)
        attack = jnp.clip(state.controls[1] + delta[1], *params.attack_limits)
        sideslip = jnp.clip(state.controls[2] + delta[2], *params.sideslip_limits)
        controls = jnp.array([bank, attack, sideslip], dtype=jnp.float32)

        position, speed, attitude = _step_per_sec(state, controls, params)

        step_number = state.time + jnp.int32(1)


        out_of_bounds_xy = (jnp.abs(position[0]) > params.horizontal_bound) | (
            jnp.abs(position[1]) > params.horizontal_bound) | (
            jnp.abs(position[0]) < -params.horizontal_bound) | (
            jnp.abs(position[1]) < -params.horizontal_bound
        )
        out_of_bounds_z = (position[2] < params.vertical_bounds[0]) | (
            position[2] > params.vertical_bounds[1]
        )
        low_speed = (speed <= jnp.linalg.norm(params.wind_model.horizontal_wind))
        high_attack = jnp.where(attack > 30.0 * DEG2RAD, (attack*RAD2DEG - 30), 0.0)
        truncated = out_of_bounds_xy | out_of_bounds_z | low_speed
        terminated =  (step_number >= params.max_steps_in_episode)
        done = truncated | terminated

        vertical_speed = speed * jnp.sin(attitude[0])
        #reward = params.vertical_speed_weight * vertical_speed - params.speed_penalty_weight * (speed**2)
        max_steps_f = jnp.asarray(params.max_steps_in_episode, dtype=jnp.float32)
        step_f = step_number.astype(jnp.float32)
        low_speed_reward = -(max_steps_f - step_f)
        thermal_centers_cord = thermal_centers(params.wind_model, position[2], step_f)

        reward_action = vertical_speed + 15/(jnp.linalg.norm(jnp.array([position[0] - thermal_centers_cord[0], position[1] - thermal_centers_cord[1]])) )
        # reward_action = vertical_speed

        reward = jnp.where(
            out_of_bounds_xy | out_of_bounds_z,
            jnp.float32(-1000.0),
            jnp.where(low_speed, low_speed_reward, reward_action),
        )

        # Update history buffers - shift old values and add new ones
        # Speed history: (forward_speed, vertical_speed)
        new_speed_entry = jnp.array([speed, vertical_speed], dtype=jnp.float32)
        speed_history = jnp.roll(state.speed_history, shift=-1, axis=0)
        speed_history = speed_history.at[-1].set(new_speed_entry)
        
        # Controls history: (bank_angle, attack_angle)
        new_controls_entry = jnp.array([bank, attack, sideslip], dtype=jnp.float32)
        controls_history = jnp.roll(state.controls_history, shift=-1, axis=0)
        controls_history = controls_history.at[-1].set(new_controls_entry)
        
        # Angle from wind history
        current_angle_from_wind = calculate_angle_from_wind(bank, attitude[1], params)
        angle_from_wind_history = jnp.roll(state.angle_from_wind_history, shift=-1)
        angle_from_wind_history = angle_from_wind_history.at[-1].set(current_angle_from_wind)
        
        # Wind velocity history (only last second)
        current_wind_velocity = calculate_wind_velocity(params)
        wind_velocity_history = jnp.array([current_wind_velocity], dtype=jnp.float32)

        next_state = EnvState(
            position=position, 
            speed=speed, 
            attitude=attitude, 
            controls=controls, 
            time=step_number,
            speed_history=speed_history,
            controls_history=controls_history,
            angle_from_wind_history=angle_from_wind_history,
            wind_velocity_history=wind_velocity_history,
        )
        # тут все непонятно
        obs = jax.lax.stop_gradient(self.get_obs(next_state, params))
        horizontal_wind = params.wind_model.horizontal_wind
    
        # Calculate wind angle from horizontal wind components
        wind_angle = jnp.arctan2(horizontal_wind[1], horizontal_wind[0])
        info = {
            "current_angle_from_wind": current_angle_from_wind,
            "current_wind_velocity": current_wind_velocity,
            "wind_angle": wind_angle,
            "position": position,
            "speed": speed,
            "attitude": attitude,
            "step_number": step_number,
            "discount": self.discount(next_state, params),
            "raw_state": next_state,
            "vertical_speed": vertical_speed,
            "out_of_bounds": truncated,
            "low_speed": low_speed,
            "terminated": terminated,
            "truncated": truncated,
            "thermal_centers_cord": thermal_centers_cord,
            "reward": reward,
        }

        return obs, jax.lax.stop_gradient(next_state), reward, done, info

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        key_pos, key_side = jax.random.split(key)
        xy = jax.random.uniform(key_pos, shape=(2,), minval=-50.0, maxval=50.0)
        position = jnp.array([xy[0], xy[1], params.initial_altitude], dtype=jnp.float32)        

        side_angle = jax.random.uniform(key_side, (), minval=-100.0 * DEG2RAD, maxval=100.0 * DEG2RAD)
        attitude = jnp.array([5.0 * DEG2RAD, side_angle], dtype=jnp.float32)
        controls = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

        # Initialize history buffers
        initial_forward_speed = jnp.float32(params.initial_speed)
        initial_vertical_speed = initial_forward_speed * jnp.sin(attitude[0])
        speed_history = jnp.zeros((params.history_seconds, 2), dtype=jnp.float32)
        speed_history = speed_history.at[:, 0].set(initial_forward_speed)
        speed_history = speed_history.at[:, 1].set(initial_vertical_speed)
        
        controls_history = jnp.zeros((params.history_seconds, 3), dtype=jnp.float32)
        controls_history = controls_history.at[:, 0].set(controls[0])  # attack_angle
        controls_history = controls_history.at[:, 1].set(controls[1])  # attack_angle
        controls_history = controls_history.at[:, 2].set(controls[2])  # attack_angle
        
        # Calculate initial angle_from_wind
        initial_angle_from_wind = calculate_angle_from_wind(controls[0], side_angle, params)
        angle_from_wind_history = jnp.full((params.history_seconds,), initial_angle_from_wind, dtype=jnp.float32)
        
        # Calculate initial wind velocity
        initial_wind_velocity = calculate_wind_velocity(params)
        wind_velocity_history = jnp.array([initial_wind_velocity], dtype=jnp.float32)

        state = EnvState(
            position=position,
            speed=jnp.float32(params.initial_speed),
            attitude=attitude,
            controls=controls,
            time=params.history_seconds,
            speed_history=speed_history,
            controls_history=controls_history,
            angle_from_wind_history=angle_from_wind_history,
            wind_velocity_history=wind_velocity_history,
        )

        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams, key: Any = None) -> jax.Array:
        return _concat_obs(state, params)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        done = state.time >= params.max_steps_in_episode
        return jnp.array(done)

    @property
    def name(self) -> str:
        return "Glider"

    @property
    def num_actions(self) -> int:
        return 3

    def action_space(self, params: EnvParams | None = None) -> spaces.Box:
        # For 3D action space, low and high must also be arrays
    
        return spaces.Box(-1, 1, (3,), jnp.float32)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        # Observation size: history_seconds * 5 + 1
        # - speed_history: history_seconds * 2 (forward_speed, vertical_speed)
        # - controls_history: history_seconds * 2 (bank_angle, attack_angle)
        # - angle_from_wind_history: history_seconds
        # - wind_velocity_history: 1
        obs_size = params.history_seconds * 6 + 1
        
        # Set realistic bounds for unnormalized observations
        # Speed: can range from speed_bounds
        # Controls: bank_angle and attack_angle within their limits
        # Angle from wind: typically [-pi, pi]
        # Wind velocity: [0, speed_bounds[1]]
        low = jnp.full((obs_size,), -1000.0, dtype=jnp.float32)
        high = jnp.full((obs_size,), 1000.0, dtype=jnp.float32)
        
        return spaces.Box(low, high, (obs_size,), jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": spaces.Box(
                    -params.horizontal_bound,
                    params.horizontal_bound,
                    (3,),
                    jnp.float32,
                ),
                "speed": spaces.Box(
                    params.speed_bounds[0],
                    params.speed_bounds[1],
                    (),
                    jnp.float32,
                ),
                "attitude": spaces.Box(
                    jnp.array([params.glide_limits[0], params.side_limits[0]], dtype=jnp.float32),
                    jnp.array([params.glide_limits[1], params.side_limits[1]], dtype=jnp.float32),
                    (2,),
                    jnp.float32,
                ),
                "controls": spaces.Box(
                    jnp.array([params.bank_limits[0], params.attack_limits[0], params.sideslip_limits[0]], dtype=jnp.float32),
                    jnp.array([params.bank_limits[1], params.attack_limits[1], params.sideslip_limits[1]], dtype=jnp.float32),
                    (3,),
                    jnp.float32,
                ),
                "time": spaces.Discrete(params.max_steps_in_episode + 1),
            }
        )

    def discount(self, state: EnvState, params: EnvParams) -> jax.Array:
        return jnp.where(self.is_terminal(state, params), 0.0, 1.0)


  