# P2 因素矩阵实验计划

更新时间：2026-06-06

## 实验定位

本课程设计不做全因子组合实验，而采用围绕中心基线展开的分层因素矩阵。中心基线为：

```text
LSUN Church Outdoor 100k / StyleGAN2-ADA / augpipe=bgc / target=0.6
```

公平比较点统一取 **1500 kimg**。所有 P2 对照组（E2-E6）训练到 1500 kimg，E1 取其 1512 kimg 快照作为
同预算锚点。已完成的 2000 kimg 基线不参与 1500 kimg 公平排名，仅用于最终模型展示、训练曲线和
生成阶段分析。

### 为什么把比较点定在 1500 kimg

来自已完成 2000 kimg 基线 `stats.jsonl` 的实测 ADA 概率：

| kimg | ADA p |
|---:|---:|
| 1000 | ≈ 0.013 |
| 1500 | ≈ 0.110 |
| 2000 | ≈ 0.153 |

在 1000 kimg 时 ADA 几乎未启动（p≈0.01），此时 ADA / noaug / fixed 三组近乎等价，增强策略对照没有
区分度。1500 kimg 时 ADA p≈0.11，三种策略才真正分叉，因此选定 1500 kimg 作为公平预算点。

## 因素矩阵

| 维度 | 水平 | 比较对象 | 研究问题 |
|---|---|---|---|
| 增强策略 | noaug / fixed p=0.2 / ADA | 100k、1500 kimg | 自适应增强是否优于无增强和固定增强 |
| 数据规模 | 50k / 100k | ADA target=0.6、1500 kimg | 训练样本减少是否导致质量下降或过拟合加重 |
| ADA 目标值 | target=0.4 / 0.6 /（可选 0.8） | 100k、1500 kimg | 判别器目标置信度如何影响增强强度和生成质量 |
| 生成阶段控制 | trunc=0.5 / 0.7 / 1.0 / 1.2 | 2000 kimg 最终模型 | 质量、多样性和结构稳定性的权衡 |

这不是完整笛卡尔积，而是以中心基线为锚点的一因素变化设计，保证每个结论都有明确控制变量，
同时避免组合爆炸。

## 口径约定（适用于所有组）

- **训练预算**：以 `kimg` 计，统一 1500 kimg，全局 batch、seed=42、`paper256`、双卡保持一致。
- **数据增强标签**：所有组 `mirror=true`，即数据集水平翻转始终开启；"noaug" 仅表示关闭 ADA 管线，
  报告中需明确这一点，避免歧义。
- **训练期指标**：对照组 `metrics=none`，训练时不算 FID 以节约费用；学习曲线由训练后离线对快照计算。
- **正式评估**：统一对完整 100k 原始分布、`mirror=false` 评估；50k 模型（E4）同样对完整 100k 分布评估，
  保证可比性。对应配置 `evaluate` 块已写入 `"mirror": false, "gpus": 1`，并由测试断言保证。

## 训练实验

配置文件（均已对齐 1500 kimg + `mirror=false` 评估）：

| 实验 | 配置 | 数据规模 | 增强模式 | target/p | 作用 |
|---|---|---:|---|---|---|
| E1 | `configs/baseline/p1_lsun_church256_baseline.json` | 100k | ADA | target=0.6 | 中心基线，取 1512 kimg 快照；2000 kimg 用于展示 |
| E2 | `configs/baseline/p2_lsun_church256_noada_1500.json` | 100k | noaug | - | 无增强对照 |
| E3 | `configs/baseline/p2_lsun_church256_fixedp02_1500.json` | 100k | fixed | p=0.2 | 固定增强对照 |
| E4 | `configs/baseline/p2_lsun_church256_subset50k_ada_1500.json` | 50k | ADA | target=0.6 | 数据规模与过拟合分析 |
| E5 | `configs/baseline/p2_lsun_church256_target04_1500.json` | 100k | ADA | target=0.4 | ADA 参数敏感性分析 |
| E6（可选） | `configs/baseline/p2_lsun_church256_target08_1500.json` | 100k | ADA | target=0.8 | 预算允许时补充，非核心 |

每组统一执行流程（以 E2 为例，其余替换配置路径）：

```bash
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_noada_1500.json --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p2_lsun_church256_noada_1500.json
python scripts/run_experiment.py generate \
  --config configs/baseline/p2_lsun_church256_noada_1500.json --network latest --seeds 0-63
python scripts/run_experiment.py evaluate \
  --config configs/baseline/p2_lsun_church256_noada_1500.json --network latest \
  --metrics fid50k_full,kid50k_full,pr50k3_full
```

各组论文用途：

- **E2 / E3**：与 E1 1512 kimg 快照构成"无增强 - 固定增强 - 自适应增强"的完整增强策略比较，
  说明 ADA 在 100k 规模下是否仍有价值（注意：100k 下 ADA p 本就偏低，预期差异可能不大，
  这一观察本身是结论的一部分）。
- **E4**：分析数据量减半对 FID、ADA 概率、判别器过拟合、建筑结构完整性和多样性的影响。预期 50k 下
  ADA 概率显著高于 100k，是矩阵中信号最强、最可解释的一轴。
- **E5 /（E6）**：分析不同 ADA target 下的增强概率、FID 和失败模式，讨论本数据集上是否存在比默认
  target=0.6 更合适的取值，作为本文的轻量改进探讨。

## 学习曲线与汇总分析

训练后离线计算，不在训练期占用费用：

- 对现有快照计算 FID50k_full。E1 有多检查点曲线；E2-E5 当前仅保留最终评估点，图中以独立散点显示，
  不把它们误称为完整学习曲线。
- 对各组 1500 kimg 模型及 E1 2000 kimg 模型计算 FID、KID、Precision、Recall。
- 从 `stats.jsonl` 提取 ADA 概率、生成器/判别器损失、sec/kimg、峰值显存、总耗时。

新增分析脚本（见 `scripts/`）：

- `scripts/analyze_results.py`：接受重复的 `--run E1=<run-dir>` 参数，汇总各组学习曲线、最终指标、
  ADA/损失/资源记录，输出 CSV 与曲线图。
- `scripts/nearest_neighbors.py`：接受网络、数据、种子范围、Top-K，用官方 Inception 特征做流式最近邻
  检索，输出每个生成样本及其 Top-K 真实近邻表格与拼图。
- `scripts/interpolate.py`：对指定锚点种子做 W 空间（默认）或 Z 空间插值，支持截断，输出逐帧 PNG、
  拼图 `contact_sheet.png` 及可选 mp4，用于展示隐空间连续性。

离线评估实测注意：双卡 `calc_metrics` 在 `pr50k3_full` 收尾时会触发 PyTorch 2.8 的
NCCL/TCPStore 拆进程组报错导致该指标不落盘。因此所有正式配置已统一为**单卡评估**（`gpus=1`、
`mirror=false`）；多卡机可一卡一组并行。E1 同预算锚点取最接近 1500 的
`network-snapshot-001512.pkl`（+12 kimg）。
评估所需的 `inception-2015-12-05.pt` / `vgg16.pt` 从 NVIDIA CDN 下载在国内很慢，需提前用学术加速或手动
放入 `~/.cache/dnnlib/downloads/`（命名 `<url-md5>_<文件名>`）。

## 生成阶段分析

对 2000 kimg 最终模型执行不同 truncation，不额外训练：

```bash
for t in 0.5 0.7 1.0 1.2; do
  python scripts/run_experiment.py generate \
    --config configs/baseline/p1_lsun_church256_baseline.json \
    --network latest --seeds 0-63 --trunc $t \
    --outdir results/samples/trunc-$t
done
```

论文用途：展示 truncation 对样本锐度、结构稳定性与多样性的权衡（同种子配对对照）。

## 结果表格

| 实验 | 数据规模 | 增强模式 | target/p | kimg | FID | KID | Precision | Recall | 主要现象 |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| E1-center | 100k | ADA | target=0.6 | 1512 | 16.63 | 0.0111 | 0.589 | 0.060 | seed=42 下 recall 较高 |
| E1-final | 100k | ADA | target=0.6 | 2000 | | | | | 2000 kimg 展示模型，PR 待补 |
| E2 | 100k | noaug | - | 1500 | 18.59 | 0.0113 | 0.557 | 0.028 | FID/KID 最差，无增强基准 |
| E3 | 100k | fixed | p=0.2 | 1500 | 16.47 | 0.0091 | 0.624 | 0.026 | precision 最高、recall 最低，最锐但最不多样 |
| E4 | 50k | ADA | target=0.6 | 1500 | 15.82 | 0.0082 | 0.547 | 0.037 | 固定 kimg 下 FID 较低，不能推断少数据更优 |
| E5 | 100k | ADA | target=0.4 | 1500 | 15.48 | 0.0075 | 0.604 | 0.030 | 单种子 FID/KID 最优，属探索性结果 |
| E6 | 100k | ADA | target=0.8 | 1500 | | | | | 未运行 |

保守结论：① ADA（E1）相对 noaug（E2）的 FID 改善在两个 seed 下排序一致，是当前最稳健的结果；
② E4 只能说明固定 1500 kimg 图像曝光预算下，50k 条件没有表现出更差 FID。它经历更多等效 epoch，
且 E4/E4b 由不同训练 seed 选出不同 50k 子集，因此不能推出“数据越少越好”；
③ E3、E5 没有第二种子，只能作为 Precision/Recall 和 ADA target 行为的探索性案例。FID 不覆盖
多样性，仍需同时报告 Precision/Recall、样本和最近邻证据。

### 第二随机种子（seed=1）复核

对最关键两轴各重跑一次（E1/E2/E4 → E1b/E2b/E4b，seed=1，1500 kimg，评估口径不变），结果已核验可用
（训练跑满 1500 kimg、增强行为正确、评估对全量 100k；详见 `report/p2_seed_repeat_runbook.md`）。跨种子
FID（均值 [min, max]，由 `analyze_results.py --seed-group` 生成 `results/analysis/seed_aggregate.csv`）：

| 组 | 配置 | seed=42 | seed=1 | 均值 [范围] |
|---|---|---:|---:|---|
| E1 | 100k ADA t=0.6 | 16.63 | 15.93 | 16.28 [15.93, 16.63] |
| E2 | 100k noaug | 18.59 | 18.87 | 18.73 [18.59, 18.87] |
| E4 | 50k ADA t=0.6 | 15.82 | 14.01 | 14.91 [14.01, 15.82] |

- **ADA 开/关稳健**：E1 与 E2 的种子区间不相交（16.63 < 18.59），两种子下增强都明显降 FID（结论①稳健）。
- **固定预算现象可重复，但不是纯种子检验**：固定 1500 kimg 下，E4 两次 FID 都低于对应 E1。
  但 E4/E4b 的训练 seed 同时决定 50k 子集抽样，且 E4 范围约 1.80，明显大于 E1/E2。因此只能表述为
  “本实验的 50k 条件在固定 kimg 下未劣于 100k”，不能据此推断数据量减少改善泛化。

## 数据质量与记忆风险

- LSUN Church 存在明显图库水印，生成模型会学习伪水印，必须纳入数据质量分析。
- 单独统计真实样本水印比例与生成伪水印比例。
- 用 `nearest_neighbors.py` 做生成图与训练集最近邻对比；没有最近邻证据时只称"数据污染复现"，
  不称"记忆"。

## 失败案例标注

固定每组 64 张（seeds 0-63）样本，按以下类型人工标注挑图：

| 类型 | 观察点 |
|---|---|
| 建筑结构失败 | 屋顶、墙体、窗户、塔尖是否断裂或混合 |
| 场景布局失败 | 天空、地面、树木、建筑边界是否混乱 |
| 纹理重复 | 是否出现局部重复、棋盘纹或过锐纹理 |
| 多样性不足 | 是否大量相似角度、色调、构图 |
| 训练集记忆风险 | 与训练图像最近邻是否过近 |

## 验收与假设

- 所有配置先通过 `--dry-run`，`tests/` 中全部不依赖 GPU 的测试继续通过，且测试断言评估命令含 `--mirror=false`、
  P2 配置 `kimg=1500`。
- 1500 kimg 结果解释为固定预算比较，不宣称完全收敛。
- 单训练种子结果作为系统性探索，不作统计显著性声明。ADA 开/关轴补 seed=1 后以 n=2 的
  “均值 + 范围”报告。E4 的第二次运行还改变了确定性抽取的 50k 子集，应称为“不同训练随机性与
  不同子集下的重复条件”，不能当作只改变初始化的纯种子重复。
- 每组必须具有 1500 kimg 快照、64 张固定种子样本、完整指标和资源记录。

## 预算

- E1 中心基线已含在 2000 kimg 训练内。
- E2-E5 每组 1500 kimg，双卡约 11-12 sec/kimg，约 5 小时/组。
- 删实验优先级（若必须删）：先删 E6 target=0.8，再删 E3 固定增强；E2 无增强、E4 数据规模、
  E5 target=0.4 尽量保留。
- 训练加指标/生成总费用控制在约 400 元以内。
