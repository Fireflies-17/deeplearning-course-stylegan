# 项目证据索引

更新时间：2026-06-09

本目录只负责封闭实验、视觉和数据溯源证据，不包含正式课程报告或最终提交包。

## 证据优先级

1. 原始指标 JSONL、训练 `stats.jsonl` 和 `training_options.json`；
2. 实验配置与自动生成的汇总表；
3. 生成图像、人工视觉标注和近邻拼图；
4. Claude 配置或会话记录只用于寻找线索，不作为正式证据。

## 文件

- `experiment_manifest.csv`：E1-E5、E1b、E2b、E4b 的协议、资源、指标值和原始来源。
- `claim_evidence_matrix.csv`：允许使用的结论、证据强度和禁止越过的边界。
- `visual/failure_cases.csv`：四组 P2 固定 seed 样例，共 64 张的完整失败模式标注。
- `visual/watermark_annotations.csv`：88 张独立生成样本的水印审计。
- `visual/interpolation_watermark_audit.csv`：151 帧插值轨迹的独立水印审计。
- `nearest_neighbor_audit.md`：8 个生成样本、24 条近邻记录的完整性和结论边界。
- `figure_manifest.csv`：证据图、来源范围和使用限制。
- `provenance/`：目标机数据元数据。运行 `scripts/collect_dataset_provenance.py` 后生成
  `lsun_target_machine.json`；在文件回传前，实际来源、下载日期、文件大小和哈希仍不得推测。

## 固定结论

- ADA 相比 noaug 的 FID 改善在两个保留随机条件下排序一致。
- 50k 结果只描述固定 1500 kimg 图像曝光预算，不解释为“数据越少越好”。
- E3 和 E5 均为单 seed 探索性结果。
- 单个正式训练 run 使用双卡；四卡和六卡机器用于并行调度多个双卡 run。
- 水印是训练数据污染特征被学习和复现的证据，不等价于特定样本记忆。
- 最近邻结论只覆盖当前 8 个生成样本，不证明模型总体不存在记忆。

## 重建与验证

```bash
python scripts/build_evidence.py
python -m unittest discover -s tests -v
```
