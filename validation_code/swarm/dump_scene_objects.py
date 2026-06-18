import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import airsim


def default_output_root():
    env_output_root = os.environ.get("DIFFPHYSDRONE_OUTPUT_ROOT")
    if env_output_root:
        return Path(env_output_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "Outputs" / "DiffPhysDrone"


def resolve_output_path(path, output_root):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return output_root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--pattern", default="1M_Cube_Chamfer.*")
    args = parser.parse_args()
    output_root = resolve_output_path(args.output_root, Path.cwd()) if args.output_root else default_output_root()
    output_path = resolve_output_path(args.output, output_root) if args.output else output_root / "eval" / "scene_objects" / f"{datetime.now():%Y%m%d_%H%M%S}.json"
    args.output_root = str(output_root)
    args.output = str(output_path)

    client = airsim.MultirotorClient()
    client.confirmConnection()

    objects = []
    for object_name in client.simListSceneObjects(args.pattern):
        pose = client.simGetObjectPose(object_name)
        scale = client.simGetObjectScale(object_name)
        objects.append({
            "name": object_name,
            "position": [pose.position.x_val, pose.position.y_val, pose.position.z_val],
            "orientation": [
                pose.orientation.w_val,
                pose.orientation.x_val,
                pose.orientation.y_val,
                pose.orientation.z_val,
            ],
            "scale": [scale.x_val, scale.y_val, scale.z_val],
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(objects, f)

    print(f"saved {len(objects)} objects to {args.output}")


if __name__ == "__main__":
    main()
