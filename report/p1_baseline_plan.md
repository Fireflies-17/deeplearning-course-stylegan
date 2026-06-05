# P1 基础复现计划

启动日期：2026-06-05

## 当前决策

- 正式数据集：AFHQ Cat 训练集，512x512；
- 许可证口径：Creative Commons BY-NC 4.0 by NAVER Corporation，仅用于非商业课程研究；
- 基线后端：固定提交的 NVIDIA StyleGAN2-ADA PyTorch；
- 基线目标：先完成 100 kimg 短跑，确认数据、速度、快照、生成与指标链路，再启动 5000 kimg 正式基线候选。

## 基线配置

| 配置 | 作用 | 训练预算 | 训练指标 |
|---|---|---:|---|
| `configs/baseline/p1_afhqcat512_short.json` | P1 短跑和耗时估算 | 100 kimg | none |
| `configs/baseline/p1_afhqcat512_baseline.json` | P1 正式基线候选 | 5000 kimg | fid50k_full |

两份配置共同固定：

- `cfg=paper512`；
- `aug=ada`；
- `augpipe=bgc`；
- `target=0.6`；
- `mirror=true`；
- `seed=42`；
- 单卡训练。

`paper512` 对应官方 AFHQ 512 训练配置；`mirror=true` 参考官方复现实验中对 AFHQ/MetFaces 等小数据集的水平翻转设置。

## 目标机执行顺序

```bash
python scripts/bootstrap_stylegan2_ada.py
python scripts/preflight.py --strict
python scripts/prepare_data.py convert \
  --source data/raw/afhq/train/cat \
  --dest data/processed/afhqcat-512.zip
python scripts/run_experiment.py train \
  --config configs/baseline/p1_afhqcat512_short.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p1_afhqcat512_short.json
python scripts/run_experiment.py generate \
  --config configs/baseline/p1_afhqcat512_short.json \
  --network latest
python scripts/run_experiment.py evaluate \
  --config configs/baseline/p1_afhqcat512_short.json \
  --network latest \
  --metrics fid50k_full
```

短跑通过后启动正式基线：

```bash
python scripts/run_experiment.py train \
  --config configs/baseline/p1_afhqcat512_baseline.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p1_afhqcat512_baseline.json
```

## 判断规则

短跑结束后，从官方 `log.txt` 记录以下信息：

- 数据集名称、分辨率和训练集实际大小；
- 每 kimg 秒数；
- 峰值显存；
- 是否出现 `Traceback`、CUDA 扩展回退或数据加载错误；
- 100 kimg 样本图是否能反映猫脸结构雏形；
- 单次 `fid50k_full` 能否成功落盘。

若 5000 kimg 预计无法在 2026-06-11 前完成，则把完整基线预算降到 1000 或 2000 kimg，并在本文件中记录原因。预算调整必须先于 P2 对比实验，避免事后根据结果选择预算。

## P1 完成标志

- [ ] `data/afhqcat512.md` 中补齐实际数量、大小和校验信息；
- [ ] 100 kimg 短跑完成，并记录训练速度；
- [ ] 至少一次基线训练完成；
- [ ] 生成 64 张固定 seed 样本；
- [ ] 产出 FID，最好补充 KID 与 Precision/Recall；
- [ ] 汇总基线失败模式，为 P2 诊断提供依据。
