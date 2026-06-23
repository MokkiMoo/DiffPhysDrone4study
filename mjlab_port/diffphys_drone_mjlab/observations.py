"""Observation helpers for the mjlab visual port."""

from __future__ import annotations

import torch


def drone_root_state(env, entity_name: str = "drone") -> torch.Tensor:
    """Return root pose and velocity as a compact vector."""
    drone = env.scene[entity_name]
    return torch.cat(
        [drone.data.root_link_pose_w, drone.data.root_link_vel_w],
        dim=-1,
    )


def target_vector(env, entity_name: str = "drone") -> torch.Tensor:
    """Vector from drone root position to the sampled target."""
    drone = env.scene[entity_name]
    target = getattr(env, "diffphys_target_pos", None)
    if target is None:
        target = torch.zeros(env.num_envs, 3, device=env.device)
    return target - drone.data.root_link_pose_w[:, :3]


def depth_rays(env, sensor_name: str = "depth") -> torch.Tensor:
    """Return raycast distances with misses replaced by the max distance."""
    sensor = env.scene[sensor_name]
    distances = sensor.data.distances
    max_distance = float(sensor.cfg.max_distance)
    return torch.where(distances < 0.0, torch.full_like(distances, max_distance), distances)


def last_external_force(env, action_name: str = "accel") -> torch.Tensor:
    """Expose the force produced by AccelAction for debugging."""
    term = env.action_manager.get_term(action_name)
    return term.last_force

