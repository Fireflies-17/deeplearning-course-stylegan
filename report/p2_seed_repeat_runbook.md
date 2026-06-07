# P2 第二种子重复实验执行清单（目标机）

> **状态（2026-06-07）：已完成并核验可用。** seed=1 三个 run（E1b/E2b/E4b）均训练满 1500 kimg、增强行为正确、
> 评估口径与原 P2 一致（`mirror=false`、对全量 100k）。已解包到 `results/`，跨种子均值/范围见
> `results/analysis/seed_aggregate.csv` 与 `seed_fid_comparison.png`。
> 复核要点：seed 的具体值未随 bundle 回传（`random_seed` 只嵌在未打包的 `.pkl` 里），但 kimg≈0 的首拍
> 训练轨迹与 seed=42 明显不同，可反证确为不同种子；若需硬证据，下次回传时附任一 `.pkl` 或
> `training_options.json` 即可。
> 跨种子 FID（均值[范围]）：E1 16.28[15.93,16.63]、E2 18.73[18.59,18.87]、E4 14.91[14.01,15.82]——
> "增强有用"与"固定预算下 50k 不劣于 100k"两结论在两种子下均稳健。

目的：给两条主线结论加误差范围（n=2 种子），堵住"单次结果是否随机波动"的质疑。

- **ADA 开/关**：E1 (100k, ADA t=0.6) vs E2 (100k, 无增强)
- **数据规模**：E1 (100k) vs E4 (50k, ADA t=0.6)

做法：在 1500 kimg 公平预算下，用 **seed=1**（原实验为 seed=42）各重跑一次 E1、E2、E4，记为 E1b / E2b / E4b。E3、E5 不重复（边际价值低）。

**计划：租 6 卡机，3 个 run 全并行**（6 卡 = 3 × 2，零空闲）。每组 2 卡约 5 h → 一波跑完墙钟约 5 h，共 30 卡·时。单卡 6 元/时 → 训练约 180 元，加评估/建环境约 10–20 元，合计 **约 180–200 元**。
（卡·时与费用跟摆法无关：6 卡全并行 5h、2 卡顺序 15h，都是 30 卡·时 ≈ 180 元；6 卡只是更快。）

---

## 0. 前置检查

```bash
cd <repo>                       # 仓库根目录
git log --oneline -1            # 确认在正确 commit
nvidia-smi                      # 确认 4 张卡可见
ls data/processed/lsun-church-256-100k.zip    # 数据在位
python scripts/preflight.py --strict          # 严格预检（环境/后端/数据）
```

如果实例是新开的，先按 `README.md` / `environment.yml` 建好环境（注意 `numpy>=2.0,<2.3` 的 ABI 约束）。

## 1. 干跑校验（不真正训练）

```bash
GPU_PAIRS="0,1 2,3 4,5" bash scripts/run_p2_seed1.sh --dry-run
```

三个配置都应通过后端 dry-run。先确认 `nvidia-smi` 看到 6 张卡。

## 2. 训练（6 卡全并行，墙钟约 5 h）

```bash
GPU_PAIRS="0,1 2,3 4,5" bash scripts/run_p2_seed1.sh
```

- 三槽并行一波跑完：E1b (GPU 0,1) + E2b (GPU 2,3) + E4b (GPU 4,5)
- 日志在 `results/logs/p2-seed1/<label>.log`
- 产物落在 `results/runs/p2-lsun-church256-*-seed1-1500/00000-.../`
- （备用：若只抢到 2 卡，改 `GPU_PAIRS="0,1"` 即顺序跑，墙钟约 15 h，费用相同）

## 3. 离线评估（每组最终 1500 快照算 FID/KID/PR）

```bash
for c in baseline_seed1_1500 noada_seed1_1500 subset50k_ada_seed1_1500; do
  python scripts/run_experiment.py evaluate \
    --config configs/baseline/p2_lsun_church256_${c}.json --network latest
done
```

评估口径与原 P2 一致：`mirror=false`、对全量 100k 分布、`fid50k_full,kid50k_full,pr50k3_full`。

## 4. 打包回传

需要带回本地的（小文件即可，**不必**带 `.pkl` 快照，除非要重新生成样本）：

```bash
tar -czf seed1_bundle.tar.gz \
  results/runs/p2-lsun-church256-100k-ada-seed1-1500/*/stats.jsonl \
  results/runs/p2-lsun-church256-100k-ada-seed1-1500/*/metric-*.jsonl \
  results/runs/p2-lsun-church256-100k-ada-seed1-1500/*/log.txt \
  results/runs/p2-lsun-church256-100k-noada-seed1-1500/*/stats.jsonl \
  results/runs/p2-lsun-church256-100k-noada-seed1-1500/*/metric-*.jsonl \
  results/runs/p2-lsun-church256-100k-noada-seed1-1500/*/log.txt \
  results/runs/p2-lsun-church256-50k-ada-seed1-1500/*/stats.jsonl \
  results/runs/p2-lsun-church256-50k-ada-seed1-1500/*/metric-*.jsonl \
  results/runs/p2-lsun-church256-50k-ada-seed1-1500/*/log.txt
```

（如果还想要每组的展示样本，再各跑一次 `run_experiment.py generate` 并一并打包 `results/samples/*-seed1-1500/`。）

## 5. 回到本地后（我来做）

- 把 `seed1_bundle.tar.gz` 解到 `results/`；
- 用 `scripts/analyze_results.py` 把 E1b/E2b/E4b 纳入汇总，计算 E1/E2/E4 两种子的均值与范围（误差棒）；
- 更新对比图与报告里"结论是否稳健"的论述。

---

记账：实测训练 ~12.4 sec/kimg；成本 = 卡数 × 小时 × 单卡时价。30 卡·时 × 单卡价 r → r=6/8/10 时约 180 / 240 / 300 元。
