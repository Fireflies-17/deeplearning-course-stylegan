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

先安装 LSUN LMDB 转换依赖：

```bash
python -m pip install -r requirements-gpu.txt
```

不要下载 OpenDataLab 上的全量 LSUN 包；全量包约 TB 级，不适合本项目。P1 只需要
`church_outdoor_train_lmdb.zip` 这个单类别训练包，公开镜像记录的文件大小约 2.45GB。

优先直接下载并解压单类别包：

```bash
python scripts/download_lsun_church.py --extract
```

如果默认官方地址在中国大陆不可达，进入 OpenDataLab 的 LSUN 数据集页面，只选择
Church Outdoor / `church_outdoor_train_lmdb.zip` 单文件下载，不要选择 full/all/complete
全量下载包。OpenDataLab 的文件名和 CLI 数据集名可能调整，以页面显示的命令为准：

```text
https://opendatalab.org.cn/OpenDataLab/lsun
```

`openxlab` 可以用于下载这个单文件，但不要长期留在 StyleGAN 训练环境中。它会把 `requests`、
`rich` 和 `tqdm` 约束到旧版本，和目标机已有工具以及本项目依赖冲突。需要 CLI 下载时，
优先单独创建临时下载环境；训练环境只保留 `requirements-gpu.txt` 中的依赖。

如果只能在训练环境中临时安装 `openxlab` 下载数据，下载完成后先修复环境：

```bash
python -m pip uninstall -y openxlab
python -m pip install -U -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip check
```

将下载并解压后的 Church Outdoor 训练集整理为：

```text
data/raw/lsun/church_outdoor_train_lmdb
```

数据转换与短跑：

```bash
python scripts/bootstrap_stylegan2_ada.py
python scripts/preflight.py --strict
python scripts/prepare_data.py convert \
  --source data/raw/lsun/church_outdoor_train_lmdb \
  --dest data/processed/lsun-church-256-100k.zip \
  --resolution 256x256 \
  --transform center-crop \
  --max-images 100000
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
正式基线改用双卡，后续 P2/P4 对比实验若延续该基线，GPU 数也应保持为 2。
预算调整发生在 P2 对比实验之前，避免事后根据结果选择预算。

## P1 完成标志

- [ ] `data/lsun_church256.md` 中补齐下载日期、大小和校验信息；
- [x] 100 kimg 短跑完成，并记录训练速度；
- [ ] 至少一次基线训练完成；
- [ ] 生成 64 张固定 seed 样本；
- [ ] 产出 FID，最好补充 KID 与 Precision/Recall；
- [ ] 汇总建筑结构、纹理重复、天空/边界伪影等失败模式，为 P2 诊断提供依据。
