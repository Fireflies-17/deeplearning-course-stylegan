# P0 流程验证记录

验证日期：2026-06-04

## 目标

验证第一阶段要求的完整链路：环境检查、数据准备、训练、网络快照加载、继续训练、生成和
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
| 加载快照并继续短训练 | 通过，1 kimg |
| 生成样本 | 通过，64 张 |
| P0 轻量指标落盘 | 通过，值为 `6.9031` |
| 不依赖 GPU 的单元测试 | 4 项通过 |

P0 轻量指标仅用于流程验收，不是 FID/KID，禁止写入最终报告作为模型质量结论。

## 兼容性修复

固定版本官方代码针对 PyTorch 1.7-1.9。项目 bootstrap 会应用
`patches/stylegan2-ada-pytorch-modern-pytorch.patch`：

1. 适配 PyTorch 2.x 的 `InfiniteSampler` 基类初始化方式；
2. 恢复 ADA/R1 所需的 `grid_sample` 二阶梯度路径；
3. 适配现代 `grid_sampler_2d_backward` 算子签名。

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
