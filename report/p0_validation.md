# P0 流程验证记录

验证日期：2026-06-04

## 目标

验证第一阶段要求的完整链路：环境检查、数据准备、训练、网络快照权重热启动、生成和
指标结果落盘。P0 使用合成数据，只验证工程流程，不用于课程报告结论。

## 固定基线

- 后端：NVIDIA StyleGAN2-ADA PyTorch；
- 后端提交：`d72cc7d041b42ec8e806021a205ed9349f87c6a4`；
- 目标镜像：Ubuntu 22.04、Python 3.12、PyTorch 2.8、CUDA 12.8 devel；
- 目标 GPU：RTX PRO 6000 96GB；
- P0 配置：`configs/baseline/p0_smoke.json`；
- 数据：64 张 32x32 合成图像；
- 训练：单卡、batch 32、每次 1 kimg、ADA `blit`。

## 本地验证结果

本地验证环境不是目标训练机：Windows、RTX 4060 Laptop、PyTorch 2.11、CUDA 13.0，
且没有 `nvcc`。官方融合算子因此使用慢速回退实现，但完整流程仍已通过。

| 检查项 | 结果 |
|---|---|
| 固定版本后端重建与补丁检查 | 通过 |
| 官方数据 ZIP 转换 | 通过，64 张图像 |
| 官方训练配置 `--dry-run` | 通过 |
| 现代 PyTorch `grid_sample` 二阶梯度 | 通过 |
| 首次短训练与快照保存 | 通过，1 kimg |
| 从快照权重启动新的短训练 | 通过，1 kimg |
| 生成样本 | 通过，64 张 |
| P0 轻量指标落盘 | 通过，值为 `6.9031` |
| 不依赖 GPU 的单元测试 | 5 项通过 |

P0 轻量指标仅用于流程验收，不是 FID/KID，禁止写入最终报告作为模型质量结论。

## 目标机验收结果

目标 RTX PRO 6000 机器已完成严格 P0 流程。首次训练和权重热启动训练均正常输出 `Exiting...`，
日志中没有 `Traceback` 或训练中断。

| 检查项 | 结果 |
|---|---|
| 首次短训练 | 通过，结束于 1.024 kimg，约 4.21 秒/kimg |
| 从快照权重启动新的短训练 | 通过，结束于 1.024 kimg，约 4.58 秒/kimg |
| 峰值 GPU 显存 | 约 4.41 GiB |
| 网络快照 | 通过，共输出 4 个快照 |
| 权重加载连续性 | 通过，新训练初始图与首次训练最终图哈希一致，新训练最终图与其不同 |
| 生成样本 | 通过，64 张 |
| P0 轻量指标 | 通过，值为 `6.9031` |

当前生成图仍以模糊彩色斑点为主。P0 总训练量仅为 2 kimg，其目标是验证工程链路，
因此不能据此判断正式模型质量。

这里验证的是“从旧快照加载 G/D/G_ema 权重后启动新训练”，不是可恢复优化器、ADA 状态和
已完成 kimg 的真正断点续训。项目命令使用 `warm-start`；旧 `resume` 仅作为兼容别名。

目标机训练日志与产物已同步，`results/logs/environment.json` 也已替换为目标机记录
（Linux、CUDA 12.8）。

## 兼容性修复

固定版本官方代码针对 PyTorch 1.7-1.9。项目 bootstrap 会依次应用 `patches/` 下的五个补丁
（`modern-pytorch`、`modern-warnings`、`python312-distutils`、`ddp-noise-buffer`、`numpy-scalar`）：

1. 适配 PyTorch 2.x 的 `InfiniteSampler` 基类初始化方式；
2. 恢复 ADA/R1 所需的 `grid_sample` 二阶梯度路径；
3. 适配现代 `grid_sampler_2d_backward` 算子签名；
4. PyTorch 1.11+ 使用原生 `conv2d` 回退路径，不再重复输出不支持警告；
5. 移除 Pillow 可自动推断的 `Image.fromarray(..., mode)` 参数；
6. 适配 Python 3.12（distutils 移除）与多卡 DDP 下的 noise buffer；
7. 将 `FullyConnectedLayer` 的 `torch.full(..., np.float32(bias_init))` 改为 `float(...)`，
   避免现代 torch 拒收 numpy 标量。

目标机日志中每段训练曾重复输出 5896 次 `conv2d_gradfix not supported`。该信息是原版代码在
PyTorch 2.8 上选择原生 `torch.nn.functional.conv2d` 的警告，不是训练错误，也未影响本次 P0。
Pillow 的 `mode` 弃用信息同样不影响结果；上述补丁已清理这两类兼容性警告。

## 目标机验收命令

```bash
python -m pip install -r requirements-gpu.txt
python scripts/bootstrap_stylegan2_ada.py
python scripts/preflight.py --strict
python scripts/run_p0.py --config configs/baseline/p0_smoke.json
```

严格预检必须确认 `nvcc`、`g++`、`ninja` 可用，并成功编译、执行官方融合 CUDA 算子。
目标机 P0 通过后，以目标机生成的 `results/logs/environment.json` 和训练日志替换本地记录。
环境记录包含依赖清单、主仓库状态、后端提交、后端补丁状态、GPU 和 CUDA 工具链信息。

注意：即使训练配置使用 `metrics=none`，官方训练入口在导入指标模块时仍依赖 `scipy`。

注意：`torch==2.8.0+cu128` 按 NumPy 2.x 编译，运行环境必须 `numpy>=2.0,<2.3`。装成 NumPy 1.x
会出现 `torch.from_numpy(): expected np.ndarray (got numpy.ndarray)` 等 ABI 报错（曾导致 P2
训练崩溃）。换机器后务必重新 `pip install -r requirements-gpu.txt` 并用训练所用的解释器验证：
`python -c "import numpy,torch;print(numpy.__version__);torch.from_numpy(numpy.zeros(3))"`。

注意：评估指标依赖从 NVIDIA CDN 下载特征网络——`inception-2015-12-05.pkl`（FID/KID/最近邻）和
`vgg16.pt`（pr50k3 的 Precision/Recall）。该 CDN 在国内极慢甚至超时。可先开学术加速再评估，或手动下载后
按 `<url-md5>_<文件名>` 命名放入 `~/.cache/dnnlib/downloads/`。下载一次即缓存，后续各组评估不再联网。

注意：离线评估用单卡运行（`calc_metrics.py --gpus 1`）。双卡评估在 `pr50k3_full` 收尾会触发
PyTorch 2.8 的 NCCL/TCPStore 拆进程组报错，导致 Precision/Recall 不落盘；指标与卡数无关，单卡稳定，
多卡机可一卡一组并行。
