# DiffPhysDrone mjlab visual port

This directory contains an mjlab/MuJoCo-Warp visualization port of the
DiffPhysDrone environment. It is intentionally separate from the CUDA training
stack in the repository root.

The v1 port keeps the original acceleration-style control interface and maps it
to an external wrench on a MuJoCo free body. It is meant for simulation
inspection and viewer debugging, not as a drop-in replacement for the
differentiable CUDA training loop.

The port keeps the original DiffPhysDrone camera/depth conventions as closely as
possible:

- mjlab's default plane stays at `z=0`, so CUDA-world positions are shifted up by
  1 m to preserve the original `z=-1` ground-relative geometry.
- The raycast camera derives its vertical FOV from the original
  `fov_x_half_tan=0.53` setting.
- The visible drone attitude follows the original `update_state_vec`-style
  acceleration and `v_pred` alignment while body translation is integrated by
  MuJoCo.
- The default obstacle layout instantiates the original sphere, box, vertical
  cylinder, and horizontal cylinder families. Use `--obstacle-layout boxes` for
  the older compact debug scene.

The visible drone body is rendered separately from its collision geometry. The
physical collision model is a 0.15 m spherical envelope, matching the original
DiffPhysDrone training environment's randomized `drone_radius` scale.

## Requirements

Install mjlab separately, then add this package to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/mjlab_port:$PYTHONPATH"
python -c "import mjlab, mujoco; import diffphys_drone_mjlab"
```

## Smoke test

```bash
python -m diffphys_drone_mjlab.play_accel_demo --num-envs 1 --steps 200 --policy target
```

Use `--render-mode rgb_array` for a headless offscreen smoke test if your mjlab
installation supports rendering in the current environment.

For a lighter debug scene:

```bash
python -m diffphys_drone_mjlab.play_accel_demo \
  --num-envs 1 \
  --steps 200 \
  --policy target \
  --obstacle-layout boxes
```

## Play an original DiffPhysDrone checkpoint

```bash
python -m diffphys_drone_mjlab.play_checkpoint \
  --resume /path/to/checkpoint.pth \
  --num-envs 1
```

The checkpoint runner imports the repository root `model.py`, reconstructs the
original image/state preprocessing, and converts the model's 6D output into the
3D acceleration command used by the mjlab visual port. It is a visualization
compatibility path, not a guarantee that the original CUDA-trained policy will be
dynamically stable under MuJoCo-Warp without tuning.

Useful tuning flags:

```bash
python -m diffphys_drone_mjlab.play_checkpoint \
  --resume /path/to/checkpoint.pth \
  --max-speed 3.0 \
  --margin 0.2 \
  --action-scale 1.0 \
  --policy-frame target
```

`--policy-frame target` is the default. It uses the target direction as the
level heading frame for the original checkpoint. Use `--policy-frame body` only
when you want to debug the raw MuJoCo body orientation. The checkpoint path flips
depth x by default to match the original CUDA renderer. Use `--no-flip-depth-x`
or `--flip-depth-y` only when debugging a suspected image-axis mismatch.

For non-interactive validation:

```bash
python -m diffphys_drone_mjlab.play_checkpoint \
  --resume /path/to/checkpoint.pth \
  --headless \
  --steps 100
```

## Public entry points

- `diffphys_drone_mjlab.make_diffphys_drone_env_cfg(...)`
- `diffphys_drone_mjlab.AccelActionCfg`
- `diffphys_drone_mjlab.model_action_to_accel(...)`
- Registered task id: `Mjlab-DiffPhysDrone-Visual`
