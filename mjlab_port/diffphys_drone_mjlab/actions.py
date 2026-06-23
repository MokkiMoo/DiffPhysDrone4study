"""Action terms for the mjlab DiffPhysDrone visualization port."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _gravity_vector(
    reference: torch.Tensor,
    gravity: torch.Tensor | None = None,
) -> torch.Tensor:
    if gravity is not None:
        return gravity
    return torch.tensor([0.0, 0.0, -9.80665], dtype=reference.dtype, device=reference.device)


def _quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert a rotation matrix to a wxyz quaternion."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrix with shape [..., 3, 3], got {matrix.shape}")

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    q_abs = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ).clamp_min(0.0)
    max_idx = q_abs.argmax(dim=-1)
    q_abs_max = torch.gather(q_abs, -1, max_idx[..., None]).clamp_min(1e-12).sqrt()
    quat = torch.zeros(*matrix.shape[:-2], 4, dtype=matrix.dtype, device=matrix.device)

    def _assign(mask: torch.Tensor, values: torch.Tensor) -> None:
        if mask.any():
            quat[mask] = values[mask]

    _assign(
        max_idx == 0,
        torch.stack(
            [
                q_abs[..., 0],
                m21 - m12,
                m02 - m20,
                m10 - m01,
            ],
            dim=-1,
        ),
    )
    _assign(
        max_idx == 1,
        torch.stack(
            [
                m21 - m12,
                q_abs[..., 1],
                m10 + m01,
                m02 + m20,
            ],
            dim=-1,
        ),
    )
    _assign(
        max_idx == 2,
        torch.stack(
            [
                m02 - m20,
                m10 + m01,
                q_abs[..., 2],
                m12 + m21,
            ],
            dim=-1,
        ),
    )
    _assign(
        max_idx == 3,
        torch.stack(
            [
                m10 - m01,
                m02 + m20,
                m12 + m21,
                q_abs[..., 3],
            ],
            dim=-1,
        ),
    )
    quat = quat / (2.0 * q_abs_max)
    return torch.nn.functional.normalize(quat, dim=-1)


def _rotation_from_forward_and_up(forward: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    forward = torch.nn.functional.normalize(forward, dim=-1)
    left = torch.cross(up, forward, dim=-1)
    left = torch.nn.functional.normalize(left, dim=-1)
    up = torch.cross(forward, left, dim=-1)
    return torch.stack([forward, left, up], dim=-1)


def _update_visual_rotation(
    rotation: torch.Tensor,
    accel: torch.Tensor,
    v_pred: torch.Tensor,
    dt: float,
    yaw_delay_rate: float = 6.0,
    yaw_inertia: float = 5.0,
    gravity: torch.Tensor | None = None,
) -> torch.Tensor:
    gravity = _gravity_vector(accel, gravity=gravity)
    self_forward_vec = rotation[..., 0]
    self_up_vec = accel - gravity
    thrust = torch.linalg.norm(self_up_vec, dim=-1, keepdim=True).clamp_min(1e-6)
    self_up_vec = self_up_vec / thrust

    alpha = math.exp(-yaw_delay_rate * dt)
    forward_vec = self_forward_vec * alpha + torch.nn.functional.normalize(
        self_forward_vec * yaw_inertia + v_pred,
        dim=-1,
    ) * (1.0 - alpha)
    denom = torch.where(
        self_up_vec[..., 2].abs() > 1e-4,
        -self_up_vec[..., 2],
        torch.full_like(self_up_vec[..., 2], -1e-4),
    )
    forward_vec = forward_vec.clone()
    forward_vec[..., 2] = (
        forward_vec[..., 0] * self_up_vec[..., 0]
        + forward_vec[..., 1] * self_up_vec[..., 1]
    ) / denom
    return _rotation_from_forward_and_up(forward_vec, self_up_vec)


def _orientation_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """World-frame small-angle error that rotates current axes toward target axes."""
    return 0.5 * (
        torch.cross(current[..., 0], target[..., 0], dim=-1)
        + torch.cross(current[..., 1], target[..., 1], dim=-1)
        + torch.cross(current[..., 2], target[..., 2], dim=-1)
    )


def model_action_to_accel(
    action_6d: torch.Tensor,
    rotation: torch.Tensor,
    gravity: torch.Tensor | None = None,
    thrust_error: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Convert the original 6D model output into a world-frame acceleration.

    The root training loop interprets the model output as two 3D body-frame
    vectors, `a_pred` and `v_pred`, then computes:

        act = (a_pred - v_pred - g) * thrust_error + g

    Args:
        action_6d: Tensor of shape ``[B, 6]``.
        rotation: Body-to-world rotation matrix, shape ``[B, 3, 3]``.
        gravity: World-frame gravity vector. Defaults to Earth gravity.
        thrust_error: Scalar or per-env multiplier.

    Returns:
        World-frame acceleration command, shape ``[B, 3]``.
    """
    if action_6d.shape[-1] != 6:
        raise ValueError(f"Expected action_6d last dim 6, got {action_6d.shape[-1]}")
    if gravity is None:
        gravity = torch.tensor(
            [0.0, 0.0, -9.80665], dtype=action_6d.dtype, device=action_6d.device
        )
    a_pred, v_pred = (rotation @ action_6d.reshape(action_6d.shape[0], 3, 2)).unbind(
        -1
    )
    return (a_pred - v_pred - gravity) * thrust_error + gravity


@dataclass(kw_only=True)
class AccelActionCfg(ActionTermCfg):
    """Acceleration-command action mapped to a body external wrench.

    The processed action is interpreted as world-frame desired acceleration.
    This v1 visual port intentionally applies the corresponding force directly
    to the drone body instead of solving rotor mixing.
    """

    body_name: str = "base"
    action_dim: int = 3
    mass: float = 1.0
    max_accel: float = 25.0
    gravity_compensation: bool = True
    control_delay_s: float = 0.08
    linear_drag: float = 0.05
    quadratic_drag: float = 0.02
    wind_std: float = 0.15
    wind_time_constant_s: float = 4.0
    thrust_est_error_std: float = 0.01
    visual_attitude_tracking: bool = True
    visual_yaw_delay_rate: float = 6.0
    yaw_inertia: float = 5.0
    attitude_torque_kp: float = 0.20
    attitude_torque_kd: float = 0.04
    max_attitude_torque: float = 0.20

    def build(self, env: ManagerBasedRlEnv) -> "AccelAction":
        return AccelAction(self, env)


class AccelAction(ActionTerm):
    """Apply acceleration commands as external force on the drone base body."""

    cfg: AccelActionCfg

    def __init__(self, cfg: AccelActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._raw_actions = torch.zeros(
            self.num_envs, self.cfg.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._delayed_accel = torch.zeros(self.num_envs, 3, device=self.device)
        self._wind = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_torque = torch.zeros_like(self._last_force)
        self._thrust_error = torch.ones(self.num_envs, 1, device=self.device)
        self._visual_rotation = matrix_from_quat(self._entity.data.root_link_quat_w)
        self._body_id = self._resolve_body_id()

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_action(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def last_force(self) -> torch.Tensor:
        return self._last_force

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        if self.cfg.action_dim == 3:
            accel = actions
        elif self.cfg.action_dim == 6:
            rotation = matrix_from_quat(self._entity.data.root_link_quat_w)
            accel = model_action_to_accel(
                actions,
                rotation,
                thrust_error=self._thrust_error,
            )
        else:
            raise ValueError("AccelActionCfg.action_dim must be 3 or 6")
        self._processed_actions[:] = accel.clamp(
            min=-self.cfg.max_accel, max=self.cfg.max_accel
        )

    def apply_actions(self) -> None:
        dt = float(self._env.physics_dt)
        delay_alpha = math.exp(-dt / max(self.cfg.control_delay_s, 1e-6))
        self._delayed_accel.mul_(delay_alpha).add_(
            self._processed_actions, alpha=1.0 - delay_alpha
        )

        if self.cfg.wind_std > 0.0:
            wind_alpha = math.sqrt(1.0 - dt / max(self.cfg.wind_time_constant_s, dt))
            noise_alpha = math.sqrt(max(dt / self.cfg.wind_time_constant_s, 0.0))
            self._wind.mul_(wind_alpha).add_(
                torch.randn_like(self._wind), alpha=self.cfg.wind_std * noise_alpha
            )

        velocity = self._entity.data.root_link_vel_w[:, :3]
        speed = velocity.norm(dim=-1, keepdim=True)
        drag = (
            self.cfg.linear_drag * velocity
            + self.cfg.quadratic_drag * velocity * speed
        )
        accel = self._delayed_accel + self._wind - drag
        if self.cfg.gravity_compensation:
            accel = accel + torch.tensor(
                [0.0, 0.0, 9.80665], device=self.device, dtype=accel.dtype
            )

        self._last_force[:] = accel * self.cfg.mass
        self._last_torque.zero_()

        if self.cfg.visual_attitude_tracking:
            v_pred = getattr(self._env, "diffphys_visual_v_pred", None)
            if v_pred is None:
                target = getattr(self._env, "diffphys_target_pos", None)
                if target is not None:
                    drone_pos = self._entity.data.root_link_pos_w
                    v_pred = target - drone_pos
                else:
                    v_pred = self._entity.data.root_link_vel_w[:, :3]
            if v_pred.shape[-1] != 3:
                raise ValueError(
                    f"Expected visual v_pred with last dim 3, got {v_pred.shape}"
                )
            self._visual_rotation = _update_visual_rotation(
                self._visual_rotation,
                self._delayed_accel,
                v_pred,
                dt=dt,
                yaw_delay_rate=self.cfg.visual_yaw_delay_rate,
                yaw_inertia=self.cfg.yaw_inertia,
            )
            current_rotation = matrix_from_quat(self._entity.data.root_link_quat_w)
            attitude_error = _orientation_error(current_rotation, self._visual_rotation)
            angular_velocity = self._entity.data.root_link_vel_w[:, 3:]
            self._last_torque[:] = (
                self.cfg.attitude_torque_kp * attitude_error
                - self.cfg.attitude_torque_kd * angular_velocity
            ).clamp(
                min=-self.cfg.max_attitude_torque,
                max=self.cfg.max_attitude_torque,
            )

        self._entity.write_external_wrench_to_sim(
            forces=self._last_force[:, None, :],
            torques=self._last_torque[:, None, :],
            body_ids=[self._body_id],
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._delayed_accel[env_ids] = 0.0
        self._wind[env_ids] = 0.0
        self._last_force[env_ids] = 0.0
        self._last_torque[env_ids] = 0.0
        self._thrust_error[env_ids] = 1.0 + (
            torch.randn_like(self._thrust_error[env_ids])
            * self.cfg.thrust_est_error_std
        )
        self._visual_rotation[env_ids] = matrix_from_quat(
            self._entity.data.root_link_quat_w[env_ids]
        )
        self._entity.write_external_wrench_to_sim(
            forces=self._last_force[env_ids, None, :],
            torques=self._last_torque[env_ids, None, :],
            env_ids=env_ids,
            body_ids=[self._body_id],
        )

    def _resolve_body_id(self) -> int:
        body_ids, _ = self._entity.find_bodies(
            (self.cfg.body_name,),
            preserve_order=True,
        )
        if not body_ids:
            raise ValueError(
                f"Could not resolve body '{self.cfg.body_name}' for entity "
                f"'{self.cfg.entity_name}'."
            )
        return body_ids[0]
