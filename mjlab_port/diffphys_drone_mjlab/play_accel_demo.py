"""Run the mjlab DiffPhysDrone visual port with simple demo policies."""

from __future__ import annotations

import argparse
import os
import time
import warnings

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from diffphys_drone_mjlab.env_cfg import make_diffphys_drone_env_cfg


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


def _target_policy(env: ManagerBasedRlEnv) -> torch.Tensor:
    drone = env.scene["drone"]
    pos = drone.data.root_link_pose_w[:, :3]
    vel = drone.data.root_link_vel_w[:, :3]
    target = getattr(env, "diffphys_target_pos", torch.zeros_like(pos))
    command = 2.0 * (target - pos) - 1.2 * vel
    return command.clamp(-8.0, 8.0)


def _make_action(env: ManagerBasedRlEnv, policy: str) -> torch.Tensor:
    dim = env.action_manager.total_action_dim
    if policy == "zero":
        return torch.zeros(env.num_envs, dim, device=env.device)
    if policy == "random":
        return torch.randn(env.num_envs, dim, device=env.device) * 2.0
    if policy == "target":
        action = _target_policy(env)
        if dim == 3:
            return action
        padded = torch.zeros(env.num_envs, dim, device=env.device)
        padded[:, :3] = action
        return padded
    raise ValueError(f"Unknown policy: {policy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy", choices=("zero", "random", "target"), default="target")
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="viser")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--render-mode", choices=("rgb_array",), default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
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
        obstacle_layout=args.obstacle_layout,
        num_obstacles=args.num_obstacles,
    )
    env = ManagerBasedRlEnv(cfg, device=args.device, render_mode=args.render_mode)
    env.reset()
    try:
        if args.headless:
            for _ in range(args.steps):
                action = _make_action(env, args.policy)
                obs, reward, terminated, truncated, extras = env.step(action)
                del obs, reward, terminated, truncated, extras
                if args.render_mode == "rgb_array":
                    env.render()
                if args.sleep > 0.0:
                    time.sleep(args.sleep)
            return

        viewer = args.viewer
        if viewer == "auto":
            viewer = "native" if _native_viewer_available() else "viser"

        policy = lambda obs: _make_action(env, args.policy)
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
