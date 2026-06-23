"""Reset and randomization events for the mjlab visual port."""

from __future__ import annotations

import torch


_P_INIT = torch.tensor(
    [
        [-1.5, -3.0, 1.0],
        [9.5, -3.0, 1.0],
        [-0.5, 1.0, 1.0],
        [8.5, 1.0, 1.0],
        [0.0, 3.0, 1.0],
        [8.0, 3.0, 1.0],
        [-1.0, -1.0, 1.0],
        [9.0, -1.0, 1.0],
    ],
    dtype=torch.float32,
)

_P_END = torch.tensor(
    [
        [8.0, 3.0, 1.0],
        [0.0, 3.0, 1.0],
        [8.0, -1.0, 1.0],
        [0.0, -1.0, 1.0],
        [8.0, -3.0, 1.0],
        [0.0, -3.0, 1.0],
        [8.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)


def reset_drone_and_target(
    env,
    env_ids: torch.Tensor,
    entity_name: str = "drone",
    position_noise: float = 0.1,
    z_offset: float = 1.0,
) -> None:
    """Reset drone root state and sample a target from the original layout."""
    drone = env.scene[entity_name]
    device = env.device
    env_ids_t = _as_env_id_tensor(env, env_ids)
    count = len(env_ids_t)
    base_indices = env_ids_t % _P_INIT.shape[0]
    p_init = _P_INIT.to(device=device)[base_indices]
    p_end = _P_END.to(device=device)[base_indices]
    max_speed_sample = torch.rand(count, 1, device=device) * 2.5 + 0.75
    scale_x = (max_speed_sample - 0.5).clamp_min(1.0)
    scale = torch.cat(
        [
            scale_x,
            torch.rand(count, 1, device=device) + 0.5,
            torch.rand(count, 1, device=device) - 0.5,
        ],
        dim=-1,
    )
    pos = p_init * scale + torch.randn(count, 3, device=device) * position_noise
    target = p_end * scale + torch.randn(count, 3, device=device) * position_noise
    pos[:, 2] += z_offset
    target[:, 2] += z_offset
    target_delta = target - pos
    target_yaw = torch.atan2(target_delta[:, 1], target_delta[:, 0])
    root_state = torch.zeros(count, 13, device=device)
    root_state[:, :3] = pos
    root_state[:, 3] = torch.cos(0.5 * target_yaw)
    root_state[:, 6] = torch.sin(0.5 * target_yaw)
    drone.write_root_state_to_sim(root_state, env_ids=env_ids_t)

    if not hasattr(env, "diffphys_target_pos"):
        env.diffphys_target_pos = torch.zeros(env.num_envs, 3, device=device)
    env.diffphys_target_pos[env_ids_t] = target

    if not hasattr(env, "diffphys_reset_scale"):
        env.diffphys_reset_scale = torch.ones(env.num_envs, 3, device=device)
    env.diffphys_reset_scale[env_ids_t] = scale
    if not hasattr(env, "diffphys_reset_max_speed"):
        env.diffphys_reset_max_speed = torch.ones(env.num_envs, 1, device=device)
    env.diffphys_reset_max_speed[env_ids_t] = max_speed_sample
    _write_target_marker(env, env_ids_t, target)


def reset_obstacle_geoms(
    env,
    env_ids: torch.Tensor,
    geom_prefix: str = "obstacle_",
    num_obstacles: int = 12,
    obstacle_layout: str = "diffphys",
    ball_count: int = 30,
    voxel_count: int = 30,
    cylinder_count: int = 30,
    horizontal_cylinder_count: int = 2,
    z_offset: float = 1.0,
) -> None:
    """Randomize fixed obstacle geoms if per-world geom fields are available."""
    model = env.sim.model
    if not hasattr(model, "geom_pos") or not hasattr(model, "geom_size"):
        return

    env_ids_t = _as_env_id_tensor(env, env_ids)
    if obstacle_layout == "boxes":
        _reset_box_geoms(env, env_ids_t, geom_prefix, num_obstacles, z_offset)
        return
    if obstacle_layout == "diffphys":
        _reset_diffphys_geoms(
            env,
            env_ids_t,
            ball_count=ball_count,
            voxel_count=voxel_count,
            cylinder_count=cylinder_count,
            horizontal_cylinder_count=horizontal_cylinder_count,
            z_offset=z_offset,
        )
        return
    raise ValueError(
        "obstacle_layout must be 'diffphys' or 'boxes', got "
        f"{obstacle_layout!r}"
    )


def _reset_box_geoms(
    env,
    env_ids: torch.Tensor,
    geom_prefix: str,
    num_obstacles: int,
    z_offset: float,
) -> None:
    geom_ids = _find_geom_ids(env, geom_prefix, num_obstacles)
    if not geom_ids:
        return
    device = env.device
    geom_ids_t = torch.tensor(geom_ids, dtype=torch.long, device=device)
    count = len(env_ids)
    positions = torch.empty(count, len(geom_ids), 3, device=device)
    positions[..., 0] = torch.rand(count, len(geom_ids), device=device) * 8.0
    positions[..., 1] = torch.rand(count, len(geom_ids), device=device) * 8.0 - 4.0
    positions[..., 2] = (
        torch.rand(count, len(geom_ids), device=device) * 2.2 + 0.2 + z_offset
    )

    sizes = torch.empty(count, len(geom_ids), 3, device=device)
    sizes[..., 0] = torch.rand(count, len(geom_ids), device=device) * 0.25 + 0.12
    sizes[..., 1] = torch.rand(count, len(geom_ids), device=device) * 0.45 + 0.18
    sizes[..., 2] = torch.rand(count, len(geom_ids), device=device) * 0.55 + 0.18

    _write_geom_field(env.sim.model.geom_pos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.data.geom_xpos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.model.geom_size, env_ids, geom_ids_t, sizes)
    _update_obstacle_bounds(env, env_ids, geom_ids_t, sizes)


def _reset_diffphys_geoms(
    env,
    env_ids: torch.Tensor,
    ball_count: int,
    voxel_count: int,
    cylinder_count: int,
    horizontal_cylinder_count: int,
    z_offset: float,
) -> None:
    _reset_ball_geoms(env, env_ids, ball_count, z_offset)
    _reset_voxel_geoms(env, env_ids, voxel_count, z_offset)
    _reset_vertical_cylinder_geoms(env, env_ids, cylinder_count, z_offset)
    _reset_horizontal_cylinder_geoms(env, env_ids, horizontal_cylinder_count, z_offset)


def _reset_ball_geoms(env, env_ids: torch.Tensor, count_per_env: int, z_offset: float) -> None:
    geom_ids = _find_geom_ids(env, "ball_", count_per_env)
    if not geom_ids:
        return
    device = env.device
    count = len(env_ids)
    n = len(geom_ids)
    scale_x, scale_y = _diffphys_obstacle_scales(env, env_ids)
    geom_ids_t = torch.tensor(geom_ids, dtype=torch.long, device=device)
    positions = torch.empty(count, n, 3, device=device)
    positions[..., 0] = torch.rand(count, n, device=device) * 8.0 * scale_x
    positions[..., 1] = (torch.rand(count, n, device=device) * 18.0 - 9.0) * scale_y
    positions[..., 2] = torch.rand(count, n, device=device) * 6.0 - 1.0 + z_offset
    radii = torch.rand(count, n, 1, device=device) * 0.2 + 0.4
    sizes = torch.cat([radii, torch.zeros(count, n, 2, device=device)], dim=-1)
    half_extents = radii.expand(-1, -1, 3)
    _write_geom_field(env.sim.model.geom_pos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.data.geom_xpos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.model.geom_size, env_ids, geom_ids_t, sizes)
    _update_obstacle_bounds(env, env_ids, geom_ids_t, half_extents)


def _reset_voxel_geoms(env, env_ids: torch.Tensor, count_per_env: int, z_offset: float) -> None:
    geom_ids = _find_geom_ids(env, "voxel_", count_per_env)
    if not geom_ids:
        return
    device = env.device
    count = len(env_ids)
    n = len(geom_ids)
    scale_x, scale_y = _diffphys_obstacle_scales(env, env_ids)
    geom_ids_t = torch.tensor(geom_ids, dtype=torch.long, device=device)
    positions = torch.empty(count, n, 3, device=device)
    positions[..., 0] = torch.rand(count, n, device=device) * 8.0 * scale_x
    positions[..., 1] = (torch.rand(count, n, device=device) * 18.0 - 9.0) * scale_y
    positions[..., 2] = torch.rand(count, n, device=device) * 6.0 - 1.0 + z_offset
    sizes = torch.rand(count, n, 3, device=device) * 0.1 + 0.2
    _write_geom_field(env.sim.model.geom_pos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.data.geom_xpos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.model.geom_size, env_ids, geom_ids_t, sizes)
    _update_obstacle_bounds(env, env_ids, geom_ids_t, sizes)


def _reset_vertical_cylinder_geoms(
    env,
    env_ids: torch.Tensor,
    count_per_env: int,
    z_offset: float,
) -> None:
    geom_ids = _find_geom_ids(env, "cylinder_", count_per_env)
    if not geom_ids:
        return
    device = env.device
    count = len(env_ids)
    n = len(geom_ids)
    scale_x, scale_y = _diffphys_obstacle_scales(env, env_ids)
    geom_ids_t = torch.tensor(geom_ids, dtype=torch.long, device=device)
    positions = torch.empty(count, n, 3, device=device)
    positions[..., 0] = torch.rand(count, n, device=device) * 8.0 * scale_x
    positions[..., 1] = (torch.rand(count, n, device=device) * 18.0 - 9.0) * scale_y
    positions[..., 2] = z_offset + 1.5
    radii = torch.rand(count, n, 1, device=device) * 0.35 + 0.05
    half_heights = torch.full((count, n, 1), 3.5, device=device)
    sizes = torch.cat([radii, half_heights, torch.zeros_like(radii)], dim=-1)
    half_extents = torch.cat([radii, radii, half_heights], dim=-1)
    _write_geom_field(env.sim.model.geom_pos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.data.geom_xpos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.model.geom_size, env_ids, geom_ids_t, sizes)
    _update_obstacle_bounds(env, env_ids, geom_ids_t, half_extents)


def _reset_horizontal_cylinder_geoms(
    env,
    env_ids: torch.Tensor,
    count_per_env: int,
    z_offset: float,
) -> None:
    geom_ids = _find_geom_ids(env, "cylinder_h_", count_per_env)
    if not geom_ids:
        return
    device = env.device
    count = len(env_ids)
    n = len(geom_ids)
    scale_x, _ = _diffphys_obstacle_scales(env, env_ids)
    geom_ids_t = torch.tensor(geom_ids, dtype=torch.long, device=device)
    positions = torch.empty(count, n, 3, device=device)
    positions[..., 0] = torch.rand(count, n, device=device) * 8.0 * scale_x
    positions[..., 1] = 0.0
    positions[..., 2] = torch.rand(count, n, device=device) * 6.0 + z_offset
    radii = torch.rand(count, n, 1, device=device) * 0.1 + 0.05
    half_lengths = torch.full((count, n, 1), 9.0, device=device)
    sizes = torch.cat([radii, half_lengths, torch.zeros_like(radii)], dim=-1)
    half_extents = torch.cat([radii, half_lengths, radii], dim=-1)
    _write_geom_field(env.sim.model.geom_pos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.data.geom_xpos, env_ids, geom_ids_t, positions)
    _write_geom_field(env.sim.model.geom_size, env_ids, geom_ids_t, sizes)
    _update_obstacle_bounds(env, env_ids, geom_ids_t, half_extents)


def _as_env_id_tensor(env, env_ids: torch.Tensor | slice) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[env_ids]
    return env_ids


def _diffphys_obstacle_scales(
    env,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reset_scale = getattr(env, "diffphys_reset_scale", None)
    if reset_scale is None:
        ones = torch.ones(len(env_ids), 1, device=env.device)
        return ones, ones

    scale_x = reset_scale[env_ids, 0:1].clamp_min(1e-6)
    reset_max_speed = getattr(env, "diffphys_reset_max_speed", None)
    if reset_max_speed is None:
        max_speed_sample = scale_x + 0.5
    else:
        max_speed_sample = reset_max_speed[env_ids]
    scale_y = (max_speed_sample + 4.0) / scale_x
    return scale_x, scale_y


def _find_geom_ids(env, prefix: str, count: int) -> list[int]:
    geom_ids = []
    for i in range(count):
        name = f"{prefix}{i:02d}"
        try:
            geom_ids.append(env.sim.mj_model.geom(name).id)
        except (KeyError, ValueError):
            continue
    return geom_ids


def _write_target_marker(env, env_ids: torch.Tensor, target: torch.Tensor) -> None:
    model = env.sim.model
    if not hasattr(model, "site_pos"):
        return
    try:
        site_id = env.sim.mj_model.site("target_marker").id
    except (KeyError, ValueError, AttributeError):
        return
    if model.site_pos.ndim == 2:
        model.site_pos[site_id, :] = target[0]
    elif model.site_pos.shape[0] == 1:
        model.site_pos[0, site_id, :] = target[0]
    else:
        model.site_pos[env_ids, site_id, :] = target


def _write_geom_field(
    field: torch.Tensor,
    env_ids: torch.Tensor,
    geom_ids: torch.Tensor,
    values: torch.Tensor,
) -> None:
    if field.ndim == 2:
        field[geom_ids, :] = values[0]
    elif field.shape[0] == 1:
        field[0, geom_ids, :] = values[0]
    else:
        field[env_ids[:, None], geom_ids[None, :], :] = values


def _update_obstacle_bounds(
    env,
    env_ids: torch.Tensor,
    geom_ids: torch.Tensor,
    sizes: torch.Tensor,
) -> None:
    model = env.sim.model
    if not hasattr(model, "geom_rbound") or not hasattr(model, "geom_aabb"):
        return

    rbound = torch.linalg.norm(sizes, dim=-1)
    aabb_half = sizes
    if model.geom_rbound.ndim == 1:
        model.geom_rbound[geom_ids] = rbound[0]
    elif model.geom_rbound.shape[0] == 1:
        model.geom_rbound[0, geom_ids] = rbound[0]
    else:
        model.geom_rbound[env_ids[:, None], geom_ids[None, :]] = rbound

    if model.geom_aabb.ndim == 3:
        model.geom_aabb[geom_ids, 1, :] = aabb_half[0]
    elif model.geom_aabb.shape[0] == 1:
        model.geom_aabb[0, geom_ids, 1, :] = aabb_half[0]
    else:
        model.geom_aabb[env_ids[:, None], geom_ids[None, :], 1, :] = aabb_half
