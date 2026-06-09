# 目标机数据溯源状态

截至 2026-06-09，本地仓库不包含正式 LSUN 原始目录和
`data/processed/lsun-church-256-100k.zip`，因此不能在本机生成真实大小、SHA-256 或 MD5。

在仍保留数据的目标机执行：

```bash
cd /root/autodl-tmp/deeplearning-course-stylegan

python scripts/collect_dataset_provenance.py \
  --zip data/processed/lsun-church-256-100k.zip \
  --raw data/raw/lsun/church_outdoor_train_lmdb \
  --history ~/.bash_history \
  --output evidence/provenance/lsun_target_machine.json
```

将生成的 `lsun_target_machine.json` 回传到本目录后，才可以把下载来源、日期、原始目录大小、
转换 ZIP 大小和哈希标为已回填。在此之前，这些字段的正式状态是“未保留或尚未恢复”，不得使用
计划下载链接或文件修改时间替代。
