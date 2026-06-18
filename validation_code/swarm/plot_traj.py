import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation
from matplotlib import patches
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_history")
    parser.add_argument("--output", default=None)
    parser.add_argument("--scene_objects", default=None)
    parser.add_argument("--no_obstacles", default=False, action="store_true")
    parser.add_argument("--side_view", default=False, action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if args.output is None:
        output_name = "traj_topdown_side.mp4" if args.side_view else "traj_topdown.mp4"
        args.output = os.path.join(os.path.dirname(os.path.abspath(args.traj_history)), output_name)

    with open(args.traj_history) as f:
        traj = json.load(f)

    scene_objects = []
    scene_objects_path = args.scene_objects
    if scene_objects_path is None:
        candidate = os.path.join(os.path.dirname(args.traj_history), "scene_objects.json")
        if os.path.exists(candidate):
            scene_objects_path = candidate
    if scene_objects_path and not args.no_obstacles:
        with open(scene_objects_path) as f:
            scene_objects = json.load(f)

    names = sorted(traj)
    max_len = max(len(traj[name]) for name in names)
    xs = [p[0] for name in names for p in traj[name]]
    ys = [p[1] for name in names for p in traj[name]]
    pad = 1.0

    if args.side_view:
        fig, (ax, ax_side) = plt.subplots(2, 1, figsize=(8, 8), dpi=120, constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        ax_side = None
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    obstacle_count = 0
    for obj in scene_objects:
        x, y, _ = obj["position"]
        sx, sy, _ = obj["scale"]
        width = max(abs(sx), 0.05)
        height = max(abs(sy), 0.05)
        if x + width / 2 < x_min or x - width / 2 > x_max:
            continue
        if y + height / 2 < y_min or y - height / 2 > y_max:
            continue
        rect = patches.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            facecolor="#4a4a4a",
            edgecolor="#111111",
            alpha=0.28,
            linewidth=0.8,
        )
        ax.add_patch(rect)
        obstacle_count += 1
    if obstacle_count:
        obstacle_patch = patches.Patch(facecolor="#4a4a4a", edgecolor="#111111", alpha=0.28, label="obstacle footprint")
    else:
        obstacle_patch = None

    side_lines = {}
    side_heads = {}
    if ax_side is not None:
        zs = [p[2] for name in names for p in traj[name]]
        ax_side.set_xlim(min(xs) - pad, max(xs) + pad)
        ax_side.set_ylim(min(zs) - pad, max(zs) + pad)
        ax_side.set_xlabel("x")
        ax_side.set_ylabel("z")
        ax_side.grid(True, alpha=0.25)
        ax_side.set_title("Side view (x-z)")
        for obj in scene_objects:
            x, _, z = obj["position"]
            sx, _, sz = obj["scale"]
            width = max(abs(sx), 0.05)
            height = max(abs(sz), 0.05)
            if x + width / 2 < min(xs) - pad or x - width / 2 > max(xs) + pad:
                continue
            if z + height / 2 < min(zs) - pad or z - height / 2 > max(zs) + pad:
                continue
            ax_side.add_patch(patches.Rectangle(
                (x - width / 2, z - height / 2),
                width,
                height,
                facecolor="#4a4a4a",
                edgecolor="#111111",
                alpha=0.28,
                linewidth=0.8,
            ))

    lines = {}
    heads = {}
    for name in names:
        line, = ax.plot([], [], linewidth=1.8, label=name)
        head, = ax.plot([], [], "o", markersize=5, color=line.get_color())
        lines[name] = line
        heads[name] = head
        if ax_side is not None:
            side_line, = ax_side.plot([], [], linewidth=1.5, color=line.get_color())
            side_head, = ax_side.plot([], [], "o", markersize=4, color=line.get_color())
            side_lines[name] = side_line
            side_heads[name] = side_head
    handles, labels = ax.get_legend_handles_labels()
    if obstacle_patch is not None:
        handles.append(obstacle_patch)
        labels.append("obstacle footprint")
    ax.legend(handles, labels, loc="upper right", fontsize=8)

    def update(frame):
        ax.set_title(f"Swarm trajectory frame {frame + 1}/{max_len}")
        artists = []
        for name in names:
            points = traj[name][:frame + 1]
            if not points:
                continue
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z = [p[2] for p in points]
            lines[name].set_data(x, y)
            heads[name].set_data([x[-1]], [y[-1]])
            artists.extend([lines[name], heads[name]])
            if ax_side is not None:
                side_lines[name].set_data(x, z)
                side_heads[name].set_data([x[-1]], [z[-1]])
                artists.extend([side_lines[name], side_heads[name]])
        return artists

    anim = animation.FuncAnimation(fig, update, frames=max_len, interval=1000 / args.fps, blit=False)
    if os.path.exists("/usr/bin/ffmpeg"):
        plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
    writer = animation.FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=1800,
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "main", "-movflags", "+faststart"],
    )
    anim.save(args.output, writer=writer)


if __name__ == "__main__":
    main()
