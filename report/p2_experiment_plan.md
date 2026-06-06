# P2/P4 因素矩阵实验计划

更新时间：2026-06-05

## 实验定位

本课程设计不做全因子组合实验，而采用围绕中心基线展开的分层因素矩阵。中心基线为：

```text
LSUN Church Outdoor 100k / StyleGAN2-ADA / augpipe=bgc / target=0.6 / 1000 kimg
```

当前正在运行的 2000 kimg 基线仍作为最终展示模型；其中 1000 kimg 快照作为因素矩阵的中心对照点。
论文中不把训练时长作为一个独立研究因素，只将 2000 kimg 结果用于展示更充分训练后的最终生成效果。

## 因素矩阵

| 维度 | 水平 | 比较对象 | 研究问题 |
|---|---|---|---|
| 增强策略 | noaug / fixed p=0.2 / ADA | 100k、1000 kimg | 自适应增强是否优于无增强和固定增强 |
| 数据规模 | 50k / 100k | ADA target=0.6、1000 kimg | 训练样本减少是否导致质量下降或过拟合加重 |
| ADA 目标值 | target=0.4 / 0.6 / 0.8 | 100k、1000 kimg | 判别器目标置信度如何影响增强强度和生成质量 |
| 生成阶段控制 | trunc=0.5 / 0.7 / 1.0 / 1.2 | 同一个最终模型 | 质量、多样性和结构稳定性的权衡 |

这不是完整笛卡尔积，而是以中心基线为锚点的一因素变化设计。这样可以保证每个结论都有明确控制变量，
同时避免把实验扩展成难以完成的组合爆炸。

## 训练实验

### E1：中心基线与最终模型

配置：

```bash
configs/baseline/p1_lsun_church256_baseline.json
```

当前 2000 kimg 训练完成后，先记录：

- `training_options.json`；
- `log.txt`；
- `stats.jsonl`；
- 1000 kimg 附近快照；
- 2000 kimg 最终快照；
- `fakes*.png` 和 `reals.png`。

生成最终样本：

```bash
python scripts/run_experiment.py generate \
  --config configs/baseline/p1_lsun_church256_baseline.json \
  --network latest \
  --seeds 0-63 \
  --trunc 1.0
```

计算最终模型完整指标：

```bash
python scripts/run_experiment.py evaluate \
  --config configs/baseline/p1_lsun_church256_baseline.json \
  --network latest \
  --metrics fid50k_full,kid50k_full,pr50k3_full
```

论文用途：

- 作为最终生成效果展示；
- 作为方法复现的主结果；
- 用曲线和样本说明训练是否稳定。

### E2：无增强对照

配置：

```bash
configs/baseline/p2_lsun_church256_noada_1000.json
```

执行：

```bash
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_noada_1000.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_noada_1000.json
python scripts/run_experiment.py generate \
  --config configs/baseline/p2_lsun_church256_noada_1000.json \
  --network latest
python scripts/run_experiment.py evaluate \
  --config configs/baseline/p2_lsun_church256_noada_1000.json \
  --network latest \
  --metrics fid50k_full
```

比较对象：E1 的 1000 kimg 快照。

论文用途：说明没有数据增强时，模型是否更容易出现判别器过拟合、结构不稳定或多样性下降。

### E3：固定增强对照

配置：

```bash
configs/baseline/p2_lsun_church256_fixedp02_1000.json
```

执行：

```bash
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_fixedp02_1000.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_fixedp02_1000.json
python scripts/run_experiment.py generate \
  --config configs/baseline/p2_lsun_church256_fixedp02_1000.json \
  --network latest
python scripts/run_experiment.py evaluate \
  --config configs/baseline/p2_lsun_church256_fixedp02_1000.json \
  --network latest \
  --metrics fid50k_full
```

比较对象：E1 的 1000 kimg 快照和 E2。

论文用途：构成“无增强 - 固定增强 - 自适应增强”的完整增强策略比较。

### E4：50k 数据规模对照

配置：

```bash
configs/baseline/p2_lsun_church256_subset50k_ada_1000.json
```

执行：

```bash
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_subset50k_ada_1000.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_subset50k_ada_1000.json
python scripts/run_experiment.py generate \
  --config configs/baseline/p2_lsun_church256_subset50k_ada_1000.json \
  --network latest
python scripts/run_experiment.py evaluate \
  --config configs/baseline/p2_lsun_church256_subset50k_ada_1000.json \
  --network latest \
  --metrics fid50k_full
```

比较对象：E1 的 1000 kimg 快照。

论文用途：分析数据量减少对 FID、ADA 概率、建筑结构完整性和样本多样性的影响。

### E5/E6：ADA target 对照

配置：

```bash
configs/baseline/p2_lsun_church256_target04_1000.json
configs/baseline/p2_lsun_church256_target08_1000.json
```

执行：

```bash
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_target04_1000.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_target04_1000.json
python scripts/run_experiment.py generate \
  --config configs/baseline/p2_lsun_church256_target04_1000.json \
  --network latest
python scripts/run_experiment.py evaluate \
  --config configs/baseline/p2_lsun_church256_target04_1000.json \
  --network latest \
  --metrics fid50k_full

python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_target08_1000.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_target08_1000.json
python scripts/run_experiment.py generate \
  --config configs/baseline/p2_lsun_church256_target08_1000.json \
  --network latest
python scripts/run_experiment.py evaluate \
  --config configs/baseline/p2_lsun_church256_target08_1000.json \
  --network latest \
  --metrics fid50k_full
```

比较对象：E1 的 1000 kimg 快照。

论文用途：作为本文的轻量改进工作。根据不同 target 下的增强概率、FID 和样本失败模式，
讨论当前数据集上是否存在比默认 target=0.6 更合适的选择。

## 生成阶段分析

对最终 2000 kimg 模型执行不同 truncation，不额外训练：

```bash
python scripts/run_experiment.py generate \
  --config configs/baseline/p1_lsun_church256_baseline.json \
  --network latest \
  --seeds 0-63 \
  --trunc 0.5 \
  --outdir results/samples/trunc-0.5

python scripts/run_experiment.py generate \
  --config configs/baseline/p1_lsun_church256_baseline.json \
  --network latest \
  --seeds 0-63 \
  --trunc 0.7 \
  --outdir results/samples/trunc-0.7

python scripts/run_experiment.py generate \
  --config configs/baseline/p1_lsun_church256_baseline.json \
  --network latest \
  --seeds 0-63 \
  --trunc 1.0 \
  --outdir results/samples/trunc-1.0

python scripts/run_experiment.py generate \
  --config configs/baseline/p1_lsun_church256_baseline.json \
  --network latest \
  --seeds 0-63 \
  --trunc 1.2 \
  --outdir results/samples/trunc-1.2
```

论文用途：展示生成阶段参数对样本锐度、结构稳定性和多样性的影响。

## 结果表格

主表建议：

| 实验 | 数据规模 | 增强模式 | ADA target / p | kimg | FID | KID | Precision | Recall | 主要现象 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| E1-center | 100k | ADA | target=0.6 | 1000 | | | | | |
| E1-final | 100k | ADA | target=0.6 | 2000 | | | | | |
| E2 | 100k | noaug | - | 1000 | | | | | |
| E3 | 100k | fixed | p=0.2 | 1000 | | | | | |
| E4 | 50k | ADA | target=0.6 | 1000 | | | | | |
| E5 | 100k | ADA | target=0.4 | 1000 | | | | | |
| E6 | 100k | ADA | target=0.8 | 1000 | | | | | |

如果指标费用或时间不足，所有实验至少计算 FID；KID、Precision、Recall 优先补给 E1-final、E2、E3、E4。

## 失败案例标注

固定每组 64 张样本，按以下类型人工标注和挑图：

| 类型 | 观察点 |
|---|---|
| 建筑结构失败 | 屋顶、墙体、窗户、塔尖是否断裂或混合 |
| 场景布局失败 | 天空、地面、树木、建筑边界是否混乱 |
| 纹理重复 | 是否出现局部重复、棋盘纹或过锐纹理 |
| 多样性不足 | 是否大量生成相似角度、相似色调、相似构图 |
| 训练集记忆风险 | 与训练图像最近邻是否过近 |

## 内部执行约束

当前已启动的 E1 约 96 元。后续 E2-E6 每组约 48 元，总训练费用约 336 元；预留约 64 元用于
指标计算、生成图和必要重跑，总费用控制在 400 元以内。

如果中途必须删实验，删除优先级为：

1. 先删 E6 `target=0.8`；
2. 再删 E3 固定增强；
3. E2 无增强、E4 数据规模和 E5 `target=0.4` 尽量保留。
