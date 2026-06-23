"""mjlab environment configuration for the DiffPhysDrone visual port."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ObjRef, PinholeCameraPatternCfg, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from diffphys_drone_mjlab.actions import AccelActionCfg
from diffphys_drone_mjlab.events import reset_drone_and_target, reset_obstacle_geoms
from diffphys_drone_mjlab.observations import (
    depth_rays,
    drone_root_state,
    last_external_force,
    target_vector,
)

TASK_ID = "Mjlab-DiffPhysDrone-Visual"
_ASSET_DIR = Path(__file__).parent / "assets"
_DRONE_XML = _ASSET_DIR / "diffphys_drone.xml"
CUDA_TO_MJLAB_Z_OFFSET = 1.0
DIFFPHYS_FOV_X_HALF_TAN = 0.53
DEFAULT_BALL_COUNT = 30
DEFAULT_VOXEL_COUNT = 30
DEFAULT_CYLINDER_COUNT = 30
DEFAULT_HORIZONTAL_CYLINDER_COUNT = 2


def _get_drone_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_DRONE_XML))


def _diffphys_fovy_deg(
    width: int,
    height: int,
    fov_x_half_tan: float = DIFFPHYS_FOV_X_HALF_TAN,
) -> float:
    fov_y_half_tan = fov_x_half_tan / width * height
    return math.degrees(2.0 * math.atan(fov_y_half_tan))


def _add_box_obstacles_to_spec(spec: mujoco.MjSpec, num_obstacles: int) -> None:
    for i in range(num_obstacles):
        x = 1.0 + (i % 4) * 1.7
        y = -2.5 + (i // 4) * 1.7
        z = CUDA_TO_MJLAB_Z_OFFSET + 0.7 + (i % 3) * 0.35
        spec.worldbody.add_geom(
            name=f"obstacle_{i:02d}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, z],
            size=[0.16, 0.25, 0.6],
            rgba=[0.68, 0.64, 0.58, 1.0],
            contype=1,
            conaffinity=1,
            group=0,
        )


def _add_diffphys_obstacles_to_spec(
    spec: mujoco.MjSpec,
    ball_count: int,
    voxel_count: int,
    cylinder_count: int,
    horizontal_cylinder_count: int,
) -> None:
    for i in range(ball_count):
        spec.worldbody.add_geom(
            name=f"ball_{i:02d}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            pos=[1.0 + (i % 6), -4.0 + (i // 6) * 1.6, CUDA_TO_MJLAB_Z_OFFSET + 1.0],
            size=[0.45],
            rgba=[0.42, 0.58, 0.72, 1.0],
            contype=1,
            conaffinity=1,
            group=0,
        )
    for i in range(voxel_count):
        spec.worldbody.add_geom(
            name=f"voxel_{i:02d}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.7 + (i % 6) * 1.2, -4.5 + (i // 6) * 1.8, CUDA_TO_MJLAB_Z_OFFSET + 1.2],
            size=[0.22, 0.22, 0.22],
            rgba=[0.68, 0.64, 0.58, 1.0],
            contype=1,
            conaffinity=1,
            group=0,
        )
    for i in range(cylinder_count):
        spec.worldbody.add_geom(
            name=f"cylinder_{i:02d}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[0.8 + (i % 6) * 1.3, -4.5 + (i // 6) * 1.8, CUDA_TO_MJLAB_Z_OFFSET + 1.5],
            size=[0.12, 2.5],
            rgba=[0.48, 0.48, 0.46, 1.0],
            contype=1,
            conaffinity=1,
            group=0,
        )
    for i in range(horizontal_cylinder_count):
        spec.worldbody.add_geom(
            name=f"cylinder_h_{i:02d}",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[2.0 + i * 3.0, 0.0, CUDA_TO_MJLAB_Z_OFFSET + 1.5],
            quat=[0.70710678, -0.70710678, 0.0, 0.0],
            size=[0.08, 5.5],
            rgba=[0.48, 0.48, 0.46, 1.0],
            contype=1,
            conaffinity=1,
            group=0,
        )


def _add_obstacles_to_spec(
    spec: mujoco.MjSpec,
    *,
    obstacle_layout: str,
    num_obstacles: int,
    ball_count: int,
    voxel_count: int,
    cylinder_count: int,
    horizontal_cylinder_count: int,
) -> None:
    if obstacle_layout == "boxes":
        _add_box_obstacles_to_spec(spec, num_obstacles)
    elif obstacle_layout == "diffphys":
        _add_diffphys_obstacles_to_spec(
            spec,
            ball_count=ball_count,
            voxel_count=voxel_count,
            cylinder_count=cylinder_count,
            horizontal_cylinder_count=horizontal_cylinder_count,
        )
    else:
        raise ValueError(
            "obstacle_layout must be 'diffphys' or 'boxes', got "
            f"{obstacle_layout!r}"
        )
    spec.worldbody.add_site(
        name="target_marker",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        pos=[8.0, 2.0, CUDA_TO_MJLAB_Z_OFFSET + 1.0],
        size=[0.08],
        rgba=[0.05, 0.75, 0.32, 1.0],
    )


def _alive_reward(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


@requires_model_fields("geom_pos", "geom_size", "geom_rbound", "geom_aabb")
def _reset_obstacles_with_fields(
    env,
    env_ids: torch.Tensor,
    obstacle_layout: str,
    num_obstacles: int,
    ball_count: int,
    voxel_count: int,
    cylinder_count: int,
    horizontal_cylinder_count: int,
    z_offset: float,
) -> None:
    reset_obstacle_geoms(
        env,
        env_ids,
        obstacle_layout=obstacle_layout,
        num_obstacles=num_obstacles,
        ball_count=ball_count,
        voxel_count=voxel_count,
        cylinder_count=cylinder_count,
        horizontal_cylinder_count=horizontal_cylinder_count,
        z_offset=z_offset,
    )


def make_diffphys_drone_env_cfg(
    *,
    play: bool = True,
    num_envs: int = 1,
    num_obstacles: int = 12,
    obstacle_layout: str = "diffphys",
    ball_count: int = DEFAULT_BALL_COUNT,
    voxel_count: int = DEFAULT_VOXEL_COUNT,
    cylinder_count: int = DEFAULT_CYLINDER_COUNT,
    horizontal_cylinder_count: int = DEFAULT_HORIZONTAL_CYLINDER_COUNT,
    action_dim: int = 3,
    depth_width: int = 64,
    depth_height: int = 48,
    fov_x_half_tan: float = DIFFPHYS_FOV_X_HALF_TAN,
) -> ManagerBasedRlEnvCfg:
    """Build the mjlab visual environment config."""
    drone_cfg = EntityCfg(
        spec_fn=_get_drone_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, CUDA_TO_MJLAB_Z_OFFSET + 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )

    observations = {
        "actor": ObservationGroupCfg(
            {
                "root_state": ObservationTermCfg(func=drone_root_state),
                "target_vector": ObservationTermCfg(func=target_vector),
                "depth": ObservationTermCfg(
                    func=depth_rays,
                    params={"sensor_name": "depth"},
                    clip=(0.0, 24.0),
                ),
                "external_force": ObservationTermCfg(func=last_external_force),
            },
            concatenate_terms=True,
            nan_policy="sanitize",
        )
    }

    events = {
        "reset_drone": EventTermCfg(
            func=reset_drone_and_target,
            mode="reset",
            params={
                "entity_name": "drone",
                "z_offset": CUDA_TO_MJLAB_Z_OFFSET,
            },
        ),
        "reset_obstacles": EventTermCfg(
            func=_reset_obstacles_with_fields,
            mode="reset",
            params={
                "obstacle_layout": obstacle_layout,
                "num_obstacles": num_obstacles,
                "ball_count": ball_count,
                "voxel_count": voxel_count,
                "cylinder_count": cylinder_count,
                "horizontal_cylinder_count": horizontal_cylinder_count,
                "z_offset": CUDA_TO_MJLAB_Z_OFFSET,
            },
        ),
    }

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"drone": drone_cfg},
        sensors=(
            RayCastSensorCfg(
                name="depth",
                frame=ObjRef(type="site", name="camera_site", entity="drone"),
                pattern=PinholeCameraPatternCfg(
                    width=depth_width,
                    height=depth_height,
                    fovy=_diffphys_fovy_deg(depth_width, depth_height, fov_x_half_tan),
                ),
                ray_alignment="base",
                max_distance=24.0,
                include_geom_groups=(0,),
                debug_vis=play,
            ),
        ),
        num_envs=num_envs,
        env_spacing=12.0,
        extent=18.0,
        spec_fn=lambda spec: _add_obstacles_to_spec(
            spec,
            obstacle_layout=obstacle_layout,
            num_obstacles=num_obstacles,
            ball_count=ball_count,
            voxel_count=voxel_count,
            cylinder_count=cylinder_count,
            horizontal_cylinder_count=horizontal_cylinder_count,
        ),
    )

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="drone",
        body_name="base",
        distance=6.0,
        elevation=-25.0,
        azimuth=130.0,
        width=960,
        height=720,
    )

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        actions={
            "accel": AccelActionCfg(
                entity_name="drone",
                body_name="base",
                action_dim=action_dim,
                mass=1.0,
                max_accel=25.0,
            )
        },
        events=events,
        rewards={"alive": RewardTermCfg(func=_alive_reward, weight=1.0)},
        terminations={"time_out": TerminationTermCfg(func=time_out, time_out=True)},
        sim=SimulationCfg(
            njmax=256,
            mujoco=MujocoCfg(timestep=0.002, integrator="implicitfast"),
        ),
        decimation=32,
        episode_length_s=12.0,
        viewer=viewer,
        scale_rewards_by_dt=False,
    )


def register_diffphys_drone_task() -> None:
    """Register the visual task with mjlab if the registry is available."""
    from mjlab.rl import (
        RslRlModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    if TASK_ID in list_tasks():
        return
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_diffphys_drone_env_cfg(play=False, num_envs=64),
        play_env_cfg=make_diffphys_drone_env_cfg(play=True, num_envs=1),
        rl_cfg=RslRlOnPolicyRunnerCfg(
            max_iterations=1,
            actor=RslRlModelCfg(
                hidden_dims=(128, 128),
                distribution_cfg={
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                    "std_type": "scalar",
                },
            ),
            critic=RslRlModelCfg(hidden_dims=(128, 128)),
            algorithm=RslRlPpoAlgorithmCfg(),
        ),
    )


try:
    register_diffphys_drone_task()
except Exception:
    # Registration is best-effort for direct imports. The demo script imports the
    # config directly, so a registry-version mismatch should not break that path.
    pass
