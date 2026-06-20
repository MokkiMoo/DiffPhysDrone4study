# Repository Guidelines

## Project Structure & Module Organization

This repository contains vision-based agile flight training code with a compact Python/CUDA layout. Root-level Python files are the main training stack: `main_cuda.py` starts training, `env_cuda.py` defines the differentiable environment, and `model.py` contains the network. CUDA extension sources live in `src/`: `quadsim.cpp`, `quadsim_kernel.cu`, `dynamics_kernel.cu`, `setup.py`, and `test.py`. Training presets are stored in `configs/`. Demo media is under `gifs/`. Swarm simulator validation code and helper scripts are isolated in `validation_code/swarm/`.

## Build, Test, and Development Commands

- `pip install -e src`: builds and installs the `quadsim_cuda` PyTorch CUDA extension in editable mode.
- `python src/test.py`: compares CUDA forward/backward results against a PyTorch reference; requires a CUDA-capable GPU and the extension build above.
- `python main_cuda.py $(cat configs/single_agent.args)`: starts single-agent training.
- `python main_cuda.py $(cat configs/multi_agent.args)`: starts multi-agent training.
- `cd validation_code/swarm && ./run_sim_x11.sh`: launches the AirSim validation simulator when the downloaded simulator package is present.
- `cd validation_code/swarm && python eval.py --resume swarm.pth --target_speed 2.5`: evaluates a trained swarm checkpoint.

Training and evaluation outputs are written outside the repository by default under `../Outputs/DiffPhysDrone/`. Use `DIFFPHYSDRONE_OUTPUT_ROOT` or `--output_root` to redirect generated checkpoints, logs, and videos.

## Coding Style & Naming Conventions

Use Python 3.11-compatible code for the main training path. Follow the existing style: 4-space indentation, `snake_case` functions and variables, `CapWords` classes, and lowercase argparse flags such as `--output_root`. Keep CUDA extension names aligned with `src/setup.py` and the `quadsim_cuda` import. Prefer explicit tensor shapes and device placement in new training or simulation code.

## Testing Guidelines

There is no general test runner configured. Treat `python src/test.py` as the required CUDA extension regression check after changes to `src/*.cu`, `src/*.cpp`, or physics math in `env_cuda.py`. For training changes, run a short smoke test with a small `--num_iters` value and confirm checkpoint/TensorBoard output paths are created as expected. For swarm changes, use `validation_code/swarm/batch_test.sh` when simulator access is available.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Add reproducible training and evaluation helpers` or `update readme`. Prefer concise, descriptive messages that name the changed area. Pull requests should include the purpose, key commands run, GPU/CUDA environment used, affected configs, and any checkpoint or simulator artifacts needed to reproduce results. Do not commit generated checkpoints, TensorBoard logs, AirSim outputs, or large videos.
