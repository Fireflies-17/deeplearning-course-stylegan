# LSUN Church Outdoor 256 数据记录

决策日期：2026-06-05

## 数据集选择

P1 正式数据集改为 LSUN Church Outdoor，转换为 256x256 RGB，固定使用 100,000 张训练图像。

选择理由：

- LSUN 是生成模型和场景理解领域长期使用的经典大规模数据集；
- Church Outdoor 比 CIFAR-10 视觉冲击更强，能展示建筑结构、天空、树木和透视关系；
- 相比 Bedroom 全量百万级数据，Church Outdoor 的获取、转换和训练成本更可控；
- 不涉及真人身份图像，伦理与隐私风险低于 FFHQ/CelebA；
- 100,000 张子集足够支撑 P1 基础复现和 P2 的预算、数据规模、过拟合诊断。

本项目不采用 LSUN Bedroom 全量作为 P1 默认方案。Bedroom 更大、更常见，但完整下载和
I/O 成本更重，会把课程周期里的主要风险从模型训练转移到数据搬运。

## 来源与许可证口径

- 数据来源：Large-scale Scene Understanding (LSUN) 数据集，类别为 outdoor church。
- 数据规模口径：LSUN 分类数据集包含多个场景类别，每类训练图像从约 120,000 到 3,000,000 张不等。
- 本项目使用范围：仅使用 Church Outdoor 训练图像中的前 100,000 张，按 StyleGAN2-ADA
  `dataset_tool.py` 遍历顺序固定。
- 分辨率：统一中心裁剪并缩放到 256x256。
- 标签：无类别条件，P1 配置设置 `cond=false`。
- 报告引用：LSUN 论文、LSUN 项目页、StyleGAN2-ADA 官方数据准备说明。
- 主要来源链接：
  - https://arxiv.org/abs/1506.03365
  - https://www.yf.io/p/lsun/
  - https://opendatalab.org.cn/OpenDataLab/lsun
  - https://github.com/NVlabs/stylegan2-ada-pytorch#preparing-datasets

## 本项目路径

- 原始数据目录：`data/raw/lsun/church_outdoor_train_lmdb`
- StyleGAN2-ADA ZIP：`data/processed/lsun-church-256-100k.zip`
- 短跑配置：`configs/baseline/p1_lsun_church256_short.json`
- 基线配置：`configs/baseline/p1_lsun_church256_baseline.json`

`data/raw/` 和 `data/processed/` 不纳入版本控制。

## 下载策略

不要下载 OpenDataLab 上的全量 LSUN 包；全量包约 TB 级，不适合本项目。P1 只需要
`church_outdoor_train_lmdb.zip` 这个单类别训练包，公开镜像记录的文件大小约 2.45GB。

优先直接下载单类别包：

```bash
python scripts/download_lsun_church.py --extract
```

如果默认官方地址在中国大陆不可达，进入 OpenDataLab 的 LSUN 数据集页面，只选择
Church Outdoor / `church_outdoor_train_lmdb.zip` 单文件下载，不要选择 full/all/complete
全量下载包。OpenDataLab 的具体文件名和 CLI 数据集名可能调整，因此以页面显示为准：

```text
https://opendatalab.org.cn/OpenDataLab/lsun
```

`openxlab` 可以用于下载这个单文件，但不要长期留在 StyleGAN 训练环境中。`openxlab 0.1.3`
会要求旧版 `requests`、`rich` 和 `tqdm`，容易破坏目标机已有的 `datasets`、
`jupyterlab-server` 和训练依赖。推荐做法是单独创建临时下载环境安装 OpenDataLab CLI，
下载完成后只把数据目录拷贝回本项目。

如果只能在训练环境中临时安装 `openxlab` 下载数据，下载完成后必须移除并恢复项目依赖：

```bash
python -m pip uninstall -y openxlab
python -m pip install -U -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip check
```

如果目标机无法直接下载，就在本地浏览器或其他网络环境下载
`church_outdoor_train_lmdb.zip`，再传到目标机并解压整理为：

```text
data/raw/lsun/church_outdoor_train_lmdb
```

目录名必须以 `_lmdb` 结尾，StyleGAN2-ADA 的 `dataset_tool.py` 才会按 LMDB 读取。

## 转换

LSUN LMDB 转换依赖 `lmdb` 和 `opencv-python-headless`。目标机安装依赖后执行：

```bash
python scripts/bootstrap_stylegan2_ada.py
python scripts/prepare_data.py convert \
  --source data/raw/lsun/church_outdoor_train_lmdb \
  --dest data/processed/lsun-church-256-100k.zip \
  --resolution 256x256 \
  --transform center-crop \
  --max-images 100000
```

如果下载得到的不是 LMDB，而是已经解压好的图片目录，也可以把 `--source` 指向图片目录；
其余参数保持不变。

## P1 训练

```bash
python scripts/run_experiment.py train \
  --config configs/baseline/p1_lsun_church256_short.json \
  --dry-run
python scripts/run_experiment.py train \
  --config configs/baseline/p1_lsun_church256_short.json
```

短跑通过后启动双卡 2000 kimg 基线候选。100 kimg 单卡短跑实际耗时约 40 分钟，按该速度估算
5000 kimg 单卡需要 32 小时以上；为给 P2 对比实验和报告整理留出时间，本项目把 P1 基线预算
固定为 2000 kimg，并从正式基线开始固定使用 2 张 GPU。

## 待补充验收

- [ ] 记录下载日期和来源；（**仍缺，需在目标机回填，报告复现性需要**）
- [ ] 记录原始 LMDB 或图片目录大小；（**仍缺**）
- [ ] 记录 `data/processed/lsun-church-256-100k.zip` 文件大小和校验值；（**仍缺**，
  目标机执行 `ls -l` 与 `md5sum data/processed/lsun-church-256-100k.zip` 回填）
- [x] 确认转换日志显示图像数量为 100,000、分辨率为 256x256（基线 `log.txt` 已记录
  `Number of images: 100000` / `Image resolution: 256`）；
- [x] 运行 `configs/baseline/p1_lsun_church256_short.json` 完成 100 kimg 短跑；
- [x] 根据短跑日志中的 sec/kimg 估算基线训练时间，并固定双卡 2000 kimg 预算。
