# 目标机数据溯源状态

截至 2026-06-09，本地仓库不包含正式 LSUN 原始目录和转换 ZIP，但目标机采集结果已经回传为
`lsun_target_machine.json`，并通过 `scripts/build_evidence.py` 验证。

在仍保留数据的目标机执行：

```bash
cd /root/autodl-tmp/deeplearning-course-stylegan

python scripts/collect_dataset_provenance.py \
  --zip data/processed/lsun-church-256-100k.zip \
  --raw data/raw/lsun/church_outdoor_train_lmdb \
  --history ~/.bash_history \
  --output evidence/provenance/lsun_target_machine.json
```

当前已封闭事实：

- 转换 ZIP：19,712,169,438 字节；
- 100,000 张 256x256 RGB 图像；
- SHA-256：`0bef00af2e08661b1b76b17576863287a5403f71f6fc5ba58a7a3bbd225bc25d`；
- MD5：`e3e655220014f8d52cb9858df4d77d36`；
- 原始 LMDB：2 个文件，共 2,788,702,848 字节。

`/root/.bash_history` 在采集时不存在，因此实际下载来源和下载日期无法恢复。计划下载链接和文件
修改时间均不得替代这两个缺失事实。结构化摘要见 `provenance_summary.md`。
