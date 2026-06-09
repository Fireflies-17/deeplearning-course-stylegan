# 实验结果

- `runs/`：官方 StyleGAN2-ADA 训练输出目录（含 seed=1 重复实验 `*-seed1-1500/`）。
- `logs/`：环境记录和辅助日志。
- `samples/`：生成图像网格或单张样本。
- `tables/`：机器可读的指标汇总表。
- `figures/`：可直接用于报告的图表。
- `analysis/`：`scripts/analyze_results.py` 的汇总产物——`fair_comparison.csv`（1500 kimg 公平对齐）、
  `seed_aggregate.csv` + `seed_fid_comparison.png`（E1/E2/E4 重复条件的均值与范围）、各曲线 PNG。
  `summary.csv` 为每个指标单独记录 `*_final_kimg` 和 `*_final_eval_count`，避免把不同快照的指标误认为
  来自同一个模型。FID 图中只有一个评估点的实验以散点显示，不代表完整学习曲线。

正式证据口径不直接以汇总表为唯一来源。`scripts/build_evidence.py` 会重新读取各 run 的
`metric-*.jsonl`、`stats.jsonl` 和配置，生成 `evidence/experiment_manifest.csv`，并记录每项指标的
原始文件、快照 kimg、重复评估次数和解释范围。

视觉证据分为三类：

- `evidence/visual/failure_cases.csv`：四组 P2 固定 seed 样例，共 64 张逐张标注；
- `evidence/visual/watermark_annotations.csv`：88 张独立生成样本，允许计算独立样本比例；
- `evidence/visual/interpolation_watermark_audit.csv`：151 帧相关轨迹，只分析连续出现区间，不并入比例。

`nn/e1-final/` 只支持当前 8 个生成样本的局部结论。完整核验和禁止扩大的表述见
`evidence/nearest_neighbor_audit.md`。

生成产物默认不纳入版本控制。
