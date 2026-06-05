# AFHQ Cat 512 数据记录

决策日期：2026-06-05

## 数据集选择

P1 正式数据集暂定为 AFHQ Cat 的训练集子集，保持 512x512 分辨率。选择理由：

- 与 StyleGAN2-ADA 官方说明中的 AFHQ 512 训练配置直接匹配；
- 主题单一，样本对齐质量高，适合观察生成质量、ADA 概率和判别器过拟合；
- 数据规模约为几千张，符合当前“有限数据训练与迁移学习”方向；
- 不涉及真人身份图像，课程报告中的伦理与隐私负担低于人脸数据集。

## 来源与许可证

- 数据来源：ClovaAI StarGAN v2 发布的 Animal Faces-HQ (AFHQ) 数据集。
- 数据规模：原始 AFHQ 包含 3 个域（cat、dog、wildlife），总计 15,000 张 512x512 图像，每个域约 5,000 张；每个域 500 张作为测试集。
- 本项目使用范围：仅使用 `train/cat`，不使用测试集训练。
- 许可证：Creative Commons BY-NC 4.0 by NAVER Corporation，仅用于非商业课程研究；报告中需引用 StarGAN v2 论文并说明数据来源。
- 主要来源链接：
  - https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq
  - https://github.com/NVlabs/stylegan2-ada-pytorch#preparing-datasets

下载完成后必须记录实际文件数量、下载日期和文件校验信息；在未核对前，不把“约 5,000 张”写成最终精确数。

## 本项目路径

- 原始图像目录：`data/raw/afhq/train/cat`
- StyleGAN2-ADA ZIP：`data/processed/afhqcat-512.zip`
- 短跑配置：`configs/baseline/p1_afhqcat512_short.json`
- 基线配置：`configs/baseline/p1_afhqcat512_baseline.json`

`data/raw/` 和 `data/processed/` 不纳入版本控制。

## 下载与转换

在目标 Linux 机器上获取数据：

```bash
git clone https://github.com/clovaai/stargan-v2.git /tmp/stargan-v2
cd /tmp/stargan-v2
bash download.sh afhq-dataset
mkdir -p /path/to/deeplearning-course-stylegan/data/raw/afhq
cp -r data/afhq/train /path/to/deeplearning-course-stylegan/data/raw/afhq/
```

若目标机器无法连接 Dropbox，可使用 Hugging Face 的 AFHQv2 镜像下载 `train/cat`。该脚本会
逐张保存图片，网络中断后重复执行即可跳过已保存图片：

```bash
python -m pip install -U datasets pillow tqdm
python scripts/download_afhqcat_hf.py --hf-endpoint https://hf-mirror.com
```

回到本项目根目录，转换为 StyleGAN2-ADA ZIP：

```bash
python scripts/bootstrap_stylegan2_ada.py
python scripts/prepare_data.py convert \
  --source data/raw/afhq/train/cat \
  --dest data/processed/afhqcat-512.zip
```

如果使用更新后的 AFHQv2，必须另建记录和配置文件，不能复用本记录的实验名。

## 待补充验收

- [ ] 记录下载日期；
- [ ] 记录 `data/raw/afhq/train/cat` 实际 PNG/JPG 数量；
- [ ] 记录 `data/processed/afhqcat-512.zip` 文件大小和校验值；
- [ ] 运行 `configs/baseline/p1_afhqcat512_short.json` 完成 100 kimg 短跑；
- [ ] 根据短跑日志中的 sec/kimg 估算 5000 kimg 基线训练时间。
