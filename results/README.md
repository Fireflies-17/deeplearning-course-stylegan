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

生成产物默认不纳入版本控制。
