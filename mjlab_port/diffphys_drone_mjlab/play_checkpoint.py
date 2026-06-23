"""Run an original DiffPhysDrone checkpoint in the mjlab visual environment."""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from diffphys_drone_mjlab.actions import model_action_to_accel
from diffphys_drone_mjlab.env_cfg import make_diffphys_drone_env_cfg


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _native_viewer_available() -> bool:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        import glfw

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok = glfw.init()
        if ok:
            glfw.terminate()
        return bool(ok)
    except Exception:
        return False


def _load_original_model(resume: Path, device: str, no_odom: bool):
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from model import Model

    model = Model(7 if no_odom else 10, 6).to(device)
    state_dict = torch.load(resume, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("missing_keys:", missing)
    if unexpected:
        print("unexpected_keys:", unexpected)
    model.eval()
    return model


def _yaw_level_rotation(quat_w: torch.Tensor) -> torch.Tensor:
    rotation = matrix_from_quat(quat_w)
    fwd = rotation[:, :, 0].clone()
    up = torch.zeros_like(fwd)
    fwd[:, 2] = 0.0
    up[:, 2] = 1.0
    fwd = F.normalize(fwd, dim=-1)
    left = torch.cross(up, fwd, dim=-1)
    return torch.stack([fwd, left, up], dim=-1)


def _target_yaw_rotation(env: ManagerBasedRlEnv) -> torch.Tensor:
    drone = env.scene["drone"]
    pos = drone.data.root_link_pos_w
    target = getattr(env, "diffphys_target_pos", torch.zeros_like(pos))
    fwd = target - pos
    fwd[:, 2] = 0.0
    fwd_norm = torch.norm(fwd, dim=-1, keepdim=True)
    fallback = _yaw_level_rotation(drone.data.root_link_quat_w)[:, :, 0]
    fwd = torch.where(fwd_norm > 1e-5, fwd / fwd_norm.clamp_min(1e-5), fallback)
    up = torch.zeros_like(fwd)
    up[:, 2] = 1.0
    left = torch.cross(up, fwd, dim=-1)
    return torch.stack([fwd, left, up], dim=-1)


def _depth_image(
    env: ManagerBasedRlEnv,
    width: int,
    height: int,
    flip_x: bool,
    flip_y: bool,
) -> torch.Tensor:
    depth = env.scene["depth"].data.distances
    depth = torch.where(depth < 0.0, torch.full_like(depth, 24.0), depth)
    depth = depth.reshape(env.num_envs, height, width)
    if flip_x:
        depth = torch.flip(depth, dims=(-1,))
    if flip_y:
        depth = torch.flip(depth, dims=(-2,))
    return depth


def _state_tensor(
    env: ManagerBasedRlEnv,
    rotation_yaw: torch.Tensor,
    no_odom: bool,
    max_speed: float,
    margin: float,
    attitude_up: torch.Tensor | None = None,
) -> torch.Tensor:
    drone = env.scene["drone"]
    pos = drone.data.root_link_pos_w
    vel = drone.data.root_link_lin_vel_w
    target = getattr(env, "diffphys_target_pos", torch.zeros_like(pos))
    target_v_raw = target - pos
    target_v_norm = torch.norm(target_v_raw, dim=-1, keepdim=True).clamp_min(1e-6)
    target_v = target_v_raw / target_v_norm * torch.minimum(
        target_v_norm,
        torch.full_like(target_v_norm, max_speed),
    )
    if attitude_up is None:
        attitude_up = matrix_from_quat(drone.data.root_link_quat_w)[:, 2]

    terms = [
        torch.squeeze(target_v[:, None] @ rotation_yaw, 1),
        attitude_up,
        torch.full((env.num_envs, 1), margin, device=env.device),
    ]
    if not no_odom:
        terms.insert(0, torch.squeeze(vel[:, None] @ rotation_yaw, 1))
    return torch.cat(terms, dim=-1)


class OriginalCheckpointPolicy:
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        resume: Path,
        width: int,
        height: int,
        no_odom: bool,
        max_speed: float,
        margin: float,
        action_scale: float,
        noise_std: float,
        policy_frame: str,
        flip_depth_x: bool,
        flip_depth_y: bool,
    ) -> None:
        self.env = env
        self.width = width
        self.height = height
        self.no_odom = no_odom
        self.max_speed = max_speed
        self.margin = margin
        self.action_scale = action_scale
        self.noise_std = noise_std
        self.policy_frame = policy_frame
        self.flip_depth_x = flip_depth_x
        self.flip_depth_y = flip_depth_y
        self.model = _load_original_model(resume, env.device, no_odom)
        self.hx: torch.Tensor | None = None
        self.thrust_error = torch.ones(env.num_envs, 1, device=env.device)

    @torch.no_grad()
    def __call__(self, obs=None) -> torch.Tensor:
        del obs
        depth = _depth_image(
            self.env,
            self.width,
            self.height,
            flip_x=self.flip_depth_x,
            flip_y=self.flip_depth_y,
        )
        if self.policy_frame == "target":
            rotation_yaw = _target_yaw_rotation(self.env)
            attitude_up = matrix_from_quat(
                self.env.scene["drone"].data.root_link_quat_w
            )[:, 2]
        else:
            rotation_yaw = _yaw_level_rotation(self.env.scene["drone"].data.root_link_quat_w)
            attitude_up = None
        state = _state_tensor(
            self.env,
            rotation_yaw,
            no_odom=self.no_odom,
            max_speed=self.max_speed,
            margin=self.margin,
            attitude_up=attitude_up,
        )
        x = 3 / depth.clamp(0.3, 24.0) - 0.6
        if self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std
        x = F.max_pool2d(x[:, None], 4, 4)
        model_action, _, self.hx = self.model(x, state, self.hx)
        _, v_pred = (
            rotation_yaw @ model_action.reshape(self.env.num_envs, 3, 2)
        ).unbind(-1)
        self.env.diffphys_visual_v_pred = v_pred
        accel = model_action_to_accel(
            model_action,
            rotation_yaw,
            thrust_error=self.thrust_error,
        )
        return accel * self.action_scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="viser")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--no-odom", action="store_true")
    parser.add_argument("--max-speed", type=float, default=3.0)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--policy-frame", choices=("target", "body"), default="target")
    parser.set_defaults(flip_depth_x=True)
    parser.add_argument("--flip-depth-x", dest="flip_depth_x", action="store_true")
    parser.add_argument("--no-flip-depth-x", dest="flip_depth_x", action="store_false")
    parser.add_argument("--flip-depth-y", action="store_true")
    parser.add_argument("--depth-width", type=int, default=64)
    parser.add_argument("--depth-height", type=int, default=48)
    parser.add_argument(
        "--obstacle-layout",
        choices=("diffphys", "boxes"),
        default="diffphys",
    )
    parser.add_argument("--num-obstacles", type=int, default=12)
    args = parser.parse_args()

    cfg = make_diffphys_drone_env_cfg(
        play=True,
        num_envs=args.num_envs,
        action_dim=3,
        depth_width=args.depth_width,
        depth_height=args.depth_height,
        obstacle_layout=args.obstacle_layout,
        num_obstacles=args.num_obstacles,
    )
    env = ManagerBasedRlEnv(cfg, device=args.device)
    env.reset()
    policy = OriginalCheckpointPolicy(
        env=env,
        resume=args.resume,
        width=args.depth_width,
        height=args.depth_height,
        no_odom=args.no_odom,
        max_speed=args.max_speed,
        margin=args.margin,
        action_scale=args.action_scale,
        noise_std=args.noise_std,
        policy_frame=args.policy_frame,
        flip_depth_x=args.flip_depth_x,
        flip_depth_y=args.flip_depth_y,
    )

    try:
        if args.headless:
            for _ in range(args.steps):
                action = policy()
                obs, reward, terminated, truncated, extras = env.step(action)
                del obs, reward, terminated, truncated, extras
                if args.sleep > 0.0:
                    time.sleep(args.sleep)
            return

        viewer = args.viewer
        if viewer == "auto":
            viewer = "native" if _native_viewer_available() else "viser"
        if viewer == "native":
            NativeMujocoViewer(env, policy).run()
        elif viewer == "viser":
            ViserPlayViewer(env, policy).run()
        else:
            raise ValueError(f"Unsupported viewer: {viewer}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
