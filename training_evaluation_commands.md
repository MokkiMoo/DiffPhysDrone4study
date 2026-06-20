# DiffPhysDrone 训练与评估命令说明

## 一、环境与 CUDA 扩展准备

推荐环境：

- Python 3.11
- PyTorch 2.2.2
- CUDA 11.8

构建 CUDA 扩展：

```bash
pip install -e src
```

作用：编译并安装 `src/` 下的 `quadsim_cuda` PyTorch CUDA 扩展。训练和 `src/test.py` 都依赖该扩展。

CUDA 扩展回归测试：

```bash
python src/test.py
```

作用：将 `quadsim_cuda` 的 forward/backward 结果与 PyTorch 实现进行对比。需要 NVIDIA GPU 和可用 CUDA。

## 二、训练命令

### 1. 单机/单智能体训练

```bash
python main_cuda.py $(cat configs/single_agent.args)
```

等价展开为：

```bash
python main_cuda.py --single --speed_mtp 4 --coef_d_acc 0.01 --coef_d_jerk 0.001 --ground_voxels --random_rotation --yaw_drift --coef_collide 7.5 --coef_obj_avoidance 3.0 --cam_angle 20 --fov_x_half_tan 0.82
```

### 2. 多智能体训练

```bash
python main_cuda.py $(cat configs/multi_agent.args)
```

等价展开为：

```bash
python main_cuda.py --gate --coef_collide 5.0 --coef_obj_avoidance 2.0 --batch 256 --fov_x_half_tan 0.82 --timesteps 180
```

说明：`main_cuda.py` 中正式参数名是 `--batch_size`。`configs/multi_agent.args` 中的 `--batch` 是 argparse 对 `--batch_size` 的唯一前缀缩写，可以解析；为了清晰，建议新命令写成 `--batch_size 256`。

### 3. 断点继续训练

```bash
python main_cuda.py $(cat configs/single_agent.args) --resume <checkpoint.pth>
```

作用：从已有模型权重继续训练。

### 4. 指定输出目录

```bash
python main_cuda.py $(cat configs/single_agent.args) --output_root /home/renzihao/CodeHJM/output/DiffphysDrone
```

或：

```bash
DIFFPHYSDRONE_OUTPUT_ROOT=/home/renzihao/CodeHJM/output/DiffphysDrone python main_cuda.py $(cat configs/single_agent.args)
```

默认输出位置：

```text
../output/DiffphysDrone/train/<timestamp>_train/
```

### 5. 快速冒烟测试

```bash
python main_cuda.py $(cat configs/single_agent.args) --num_iters 10 --checkpoint_interval 5 --save_final
```

作用：只跑少量迭代，用于检查 CUDA、训练循环、checkpoint 和 TensorBoard 日志是否正常。

## 三、训练参数含义：`main_cuda.py`

- `--resume <path>`：载入已有 checkpoint 的模型权重继续训练。默认 `None`。
- `--batch_size <int>`：每个训练迭代并行仿真的样本数量。默认 `64`。
- `--num_iters <int>`：总训练迭代次数。默认 `50000`。
- `--coef_v <float>`：目标速度与真实速度差异的 smooth L1 loss 权重。默认 `1.0`。
- `--coef_speed <float>`：速度相关 legacy loss 权重。默认 `0.0`。
- `--coef_v_pred <float>`：网络速度估计 `v_pred` 与真实速度的 MSE loss 权重。默认 `2.0`。
- `--coef_collide <float>`：碰撞惩罚权重。距离障碍物越近，softplus 碰撞损失越大。默认 `2.0`。
- `--coef_obj_avoidance <float>`：障碍物避让/安全距离 loss 权重。默认 `1.5`。
- `--coef_d_acc <float>`：控制加速度正则项权重。默认 `0.01`。
- `--coef_d_jerk <float>`：控制 jerk 正则项权重，用于抑制控制变化过快。默认 `0.001`。
- `--coef_d_snap <float>`：snap 正则项权重，当前作为 legacy 项保留。默认 `0.0`。
- `--coef_ground_affinity <float>`：地面相关 legacy loss 权重。默认 `0.0`。
- `--coef_bias <float>`：速度方向偏置 loss 权重。默认 `0.0`。
- `--lr <float>`：AdamW 初始学习率。默认 `1e-3`。
- `--grad_decay <float>`：可微物理状态中的梯度衰减系数，用于位置/速度梯度传播。默认 `0.4`。
- `--speed_mtp <float>`：环境最大速度倍率。`Env` 中 `max_speed = 随机基础速度 * speed_mtp`。默认 `1.0`。
- `--fov_x_half_tan <float>`：深度相机水平半视场角正切值。值越大，水平视野越宽。默认 `0.53`。
- `--timesteps <int>`：每个训练样本 rollout 的时间步数。默认 `150`。
- `--cam_angle <int>`：相机俯仰角基础值，单位为度；`Env` 会叠加随机扰动。默认 `10`。
- `--single`：启用单智能体训练；`Env` 中每组无人机数量设为 `1`。默认关闭。
- `--gate`：在环境中加入门形障碍/通道结构。默认关闭。
- `--ground_voxels`：增加地面障碍物体素和地面球体结构。默认关闭。
- `--scaffold`：随机加入脚手架状障碍结构。默认关闭。
- `--random_rotation`：对初始点、目标点和障碍物整体施加随机 yaw 旋转，增强泛化。默认关闭。
- `--yaw_drift`：训练时给目标方向加入 yaw 漂移扰动。默认关闭。
- `--no_odom`：不向模型输入本体速度里程计信息；模型输入维度从 `10` 降为 `7`。默认关闭。
- `--output_root <path>`：输出根目录。相对路径会基于当前工作目录解析。默认 `/home/renzihao/CodeHJM/output/DiffphysDrone`，且脚本会拒绝写入当前代码仓库内部。
- `--run_name <name>`：本次训练 run 名称。默认使用当前时间戳，例如 `20260620_120000_train`。
- `--checkpoint_dir <path>`：checkpoint 保存目录。默认 `<output_root>/train/<run_name>/`。
- `--tensorboard_dir <path>`：TensorBoard 日志目录。默认 `<checkpoint_dir>/tensorboard/`。
- `--checkpoint_interval <int>`：每隔多少次迭代保存一次 checkpoint。默认 `10000`；设为 `0` 或负数可关闭周期保存。
- `--save_final`：训练结束后额外保存 `checkpoint_final_<num_iters>.pth`。默认关闭。

## 四、评估命令

评估代码位于 `validation_code/swarm/`，需要先准备/下载 AirSim Blocks 模拟器，并确保 `settings.json` 可被模拟器读取。

### 1. 启动 AirSim 模拟器

```bash
cd validation_code/swarm
./run_sim_x11.sh
```

`run_sim_x11.sh` 会调用：

```bash
./LinuxNoEditor/Blocks.sh -ResX=896 -ResY=504 -windowed -WinX=512 -WinY=304 -settings=$PWD/settings.json
```

作用：以窗口模式启动 Blocks 模拟器，并加载本项目的 AirSim 配置。

### 2. 单次 swarm 评估

```bash
cd validation_code/swarm
python eval.py --resume <checkpoint.pth> --target_speed 2.5
```

### 3. 使用默认示例 checkpoint 评估

```bash
cd validation_code/swarm
python eval.py --resume swarm.pth --target_speed 2.5
```

### 4. 连续运行 10 次评估

```bash
cd validation_code/swarm
./batch_test.sh
```

`batch_test.sh` 内部重复执行：

```bash
python eval.py --resume swarm.pth --target_speed 2.5
```

默认评估输出位置：

```text
../../../output/DiffphysDrone/eval/exps_<target_speed>/<timestamp>/
```

## 五、评估参数含义：`validation_code/swarm/eval.py`

- `--resume <path>`：要加载的模型 checkpoint。默认 `exps/run1/checkpoint0004.pth`。
- `--env <name>`：评估任务名称。当前代码只定义了 `swap`，即 6 架无人机位置交换任务。默认 `swap`。
- `--target_speed <float>`：目标速度上限，单位 m/s。代码注释说明真实速度可能约低 2 m/s。默认 `2`。
- `--margin <float>`：无人机体半径/安全半径，作为模型输入的一部分。默认 `0.15 m`。
- `--smoothness <float>`：平滑性参数，当前在 `eval.py` 中定义但未被后续逻辑实际使用。默认 `0.5`。
- `--clockspeed <float>`：期望模拟时钟倍率相关参数；`settings.json` 中 AirSim `ClockSpeed` 为 `0.25`。`eval.py` 会按 `15 * clockspeed` 初始化控制频率，并动态更新 rate。默认 `0.25`。
- `--sr <int>`：深度图下采样倍率。AirSim 深度图经 `max_pool2d` 按 `sr` 下采样后输入模型；默认 `3`，对应 `48x36` 深度图下采样到 `16x12`。
- `--no_odom`：评估时不输入本体速度里程计信息，需与训练时的 `--no_odom` 模型匹配。默认关闭。
- `--no_screen_record`：关闭 x11grab 屏幕录制，只保留深度视频和日志。默认关闭。
- `--screen_display <display>`：ffmpeg x11grab 使用的 `DISPLAY`。默认读取环境变量 `DISPLAY`，若不存在则为 `:0`。
- `--screen_offset <x,y>`：屏幕录制区域偏移。默认 `512,340`。
- `--screen_encoder <encoder>`：屏幕录制视频编码器。默认 `h264_nvenc`；没有 NVIDIA 编码器时可尝试 `libx264`。
- `--scene_object_pattern <regex>`：AirSim 场景对象匹配模式，用于记录障碍物/场景对象信息。默认 `1M_Cube_Chamfer.*`。
- `--output_root <path>`：评估输出根目录。默认 `/home/renzihao/CodeHJM/output/DiffphysDrone`，且脚本会拒绝写入当前代码仓库内部。
- `--log_dir <path>`：本次评估日志目录。默认 `<output_root>/eval/exps_<target_speed>/<timestamp>/`。

## 六、评估输出文件

每次 `eval.py` 运行会在 `log_dir` 中生成：

- `log`：记录参数、每架无人机飞行距离、耗时、是否到达、碰撞对象。
- `depth.mp4`：模型输入深度图的拼接视频。
- `<timestamp>.mp4`：屏幕录制视频；使用 `--no_screen_record` 时不生成。
- `traj_history.json`：各无人机轨迹历史。
- `scene_objects.json`：匹配到的场景对象信息。
- `eval.py`：当前评估脚本副本，便于复现实验。

## 七、常用查看命令

查看 TensorBoard：

```bash
tensorboard --logdir ../output/DiffphysDrone/train
```

查看所有评估日志：

```bash
tail ../../../output/DiffphysDrone/eval/exps_*/*/log
```

在仓库根目录查看默认训练输出：

```bash
ls ../output/DiffphysDrone/train
```

在 `validation_code/swarm` 目录查看默认评估输出：

```bash
ls ../../../output/DiffphysDrone/eval
```
