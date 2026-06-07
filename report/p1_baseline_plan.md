# P1 基础复现计划

重启日期：2026-06-05

## 当前决策

- 正式数据集：LSUN Church Outdoor，256x256，100,000 张训练图像子集；
- 许可证口径：使用 LSUN 项目页、LSUN 论文和 OpenDataLab 页面说明数据来源，仅用于课程研究；
- 基线后端：固定提交的 NVIDIA StyleGAN2-ADA PyTorch；
- 基线目标：先完成单卡 100 kimg 短跑，确认数据、速度、快照、生成与指标链路，再启动双卡 2000 kimg 基线候选；
- 网络策略：大陆目标机优先使用 OpenDataLab 获取 LSUN，无法使用 CLI 时采用浏览器下载后手动传输；不依赖 Dropbox/Hugging Face 单一路径。

选择 LSUN Church Outdoor 的原因：它比 CIFAR-10 更有视觉冲击力，比 FFHQ/CelebA 更少人脸隐私风险，
比 LSUN Bedroom 全量更容易在课程时间内完成。固定 100,000 张子集后，报告仍可说明实验建立在
经典大规模 LSUN 数据集之上，同时避免百万级数据搬运成为主风险。

## 基线配置

| 配置 | 作用 | 训练预算 | 训练指标 |
|---|---|---:|---|
| `configs/baseline/p1_lsun_church256_short.json` | P1 单卡短跑和耗时估算 | 100 kimg | none |
| `configs/baseline/p1_lsun_church256_baseline.json` | P1 双卡正式基线候选 | 2000 kimg | fid50k_full |

两份配置共同固定：

- `cfg=paper256`；
- `cond=false`；
- `aug=ada`；
- `augpipe=bgc`；
- `target=0.6`；
- `mirror=true`；
- `seed=42`；
- 短跑使用单卡，正式 2000 kimg 基线使用双卡训练与双卡评估。

`paper256` 是 StyleGAN2-ADA 后端用于 256x256 复现实验的官方基配置。`mirror=true` 用于扩增
无标签场景数据，适合 Church Outdoor 这类左右翻转后仍合理的场景类别。

## 目标机执行顺序

> 数据的下载策略（OpenDataLab 单类别包、`openxlab` 临时环境注意事项）、目录整理与
> `dataset_tool.py` 转换命令统一记录在 **`data/lsun_church256.md`**，此处不再重复，只列 P1
> 训练前后的步骤。

环境与后端就绪、并按 `data/lsun_church256.md` 备好 `data/processed/lsun-church-256-100k.zip`：

```bash
python -m pip install -r requirements-gpu.txt
python scripts/bootstrap_stylegan2_ada.py
python scripts/preflight.py --strict
```

短跑校验整条链路（数据 → 训练 → 生成 → 指标）：

```bash
python scripts/run_experiment.py train \
  --config configs/baseline/p1_lsun_church256_short.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p1_lsun_church256_short.json
python scripts/run_experiment.py generate \
  --config configs/baseline/p1_lsun_church256_short.json \
  --network latest
python scripts/run_experiment.py evaluate \
  --config configs/baseline/p1_lsun_church256_short.json \
  --network latest \
  --metrics fid50k_full
```

短跑通过后启动正式基线：

```bash
python scripts/run_experiment.py train \
  --config configs/baseline/p1_lsun_church256_baseline.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p1_lsun_church256_baseline.json
```

## 判断规则

短跑结束后，从官方 `log.txt` 记录以下信息：

- 数据集名称、分辨率和训练集实际大小；
- 每 kimg 秒数；
- 峰值显存；
- 是否出现 `Traceback`、CUDA 扩展回退、LMDB 读取错误或数据加载错误；
- 100 kimg 样本图是否出现建筑轮廓、天空/地面分离、整体透视等粗粒度结构；
- 单次 `fid50k_full` 能否成功落盘。

100 kimg 单卡短跑实际耗时约 40 分钟，稳定速度约 23.3 sec/kimg。按该速度估算，5000 kimg
单卡约需 32 小时以上；为给 P2 对比实验和报告整理留出时间，P1 基线预算固定为 2000 kimg。
正式基线改用双卡，后续 P2/P4 对比实验若延续该基线，GPU 数也应保持为 2。训练过程中保留
`fid50k_full`，用于记录随训练推进的基线指标；训练完成后再统一补算
`kid50k_full,pr50k3_full`。
预算调整发生在 P2 对比实验之前，避免事后根据结果选择预算。

## P1 完成标志

- [ ] `data/lsun_church256.md` 中补齐下载日期、大小和校验信息；（仍为占位，需在目标机回填）
- [x] 100 kimg 短跑完成，并记录训练速度（约 23.3 sec/kimg 单卡）；
- [x] 至少一次基线训练完成（双卡 2000 kimg，FID50k_full≈13，曲线单调下降）；
- [x] 生成固定 seed 样本（各组 `results/samples/p2-*`，另有插值/风格混合/截断/最近邻展示）；
- [x] 产出 FID/KID/Precision/Recall（E1-E5 已离线补算，汇总于 `results/analysis/`）；
- [ ] 汇总建筑结构、纹理重复、天空/边界伪影等失败模式，为报告分析提供依据（待人工挑图 + 撰写）。
