# 项目进度清单

更新时间：2026-06-07

当前阶段：**实验、评估与报告图均已基本齐备，主要剩报告撰写**（E1-E5 因素矩阵 + seed=1 重复，曲线/样本/插值/风格混合/最近邻图都已产出）。后续不再需要新的训练开销。

> 注：项目实际走"基线 + E1–E5 因素矩阵"路线，已把原 P2 诊断 / P3 方向选择 / P4 正式实验三段合并落地；
> 下方按原阶段保留的 checklist 仅作进度对照，"框未勾"多指撰写或可选补充，**不代表实验未做**。

当前阻塞项：无硬阻塞。E1-E5 已在 1500 kimg 公平预算下补齐 FID/KID/PR（评估 `mirror=false`，对全量 100k 真实分布，单卡 `calc_metrics`——双卡评估在 `pr50k3` 收尾会触发 NCCL/TCPStore 报错，故改单卡）。关键两轴（ADA 开/关、数据规模）已用 seed=1 各重跑 E1/E2/E4 并核验可用（见 `report/p2_seed_repeat_runbook.md`），跨种子均值/范围见 `results/analysis/seed_aggregate.csv`。报告图已存于 `results/{analysis,samples,nn}/`。**下一步为撰写 20+ 页中文报告**（剩余非写作项：数据溯源回填、失败案例标注、水印分析，均为轻量补充）。

## 近期任务

- [x] 选择 Ubuntu 22.04、Python 3.12、PyTorch 2.8、CUDA 12.8 **devel** 镜像；
- [x] 将项目迁移到 RTX PRO 6000 96GB 目标机器；
- [x] 安装 `requirements-gpu.txt` 中的附加依赖；
- [x] 运行 `python scripts/bootstrap_stylegan2_ada.py`；
- [x] 运行 `python scripts/preflight.py --strict`，确认严格预检通过；
- [x] 运行 `python scripts/run_p0.py --config configs/baseline/p0_smoke.json`；
- [x] 检查目标机训练日志、快照、生成样本和 P0 指标文件；
- [x] 根据目标机结果更新 `report/p0_validation.md`；
- [x] 将目标机生成的 `results/logs/environment.json` 同步回本地；
- [x] 推翻 AFHQ/CIFAR-10 P1 路线，改为 LSUN Church Outdoor 256x256 100k 子集；
- [x] 建立 `data/lsun_church256.md` 数据下载、校验、转换和验收记录；
- [x] 新增 P1 LSUN Church 256 短跑与基线训练配置；
- [x] 在目标机下载并转换 LSUN Church Outdoor 数据；
- [x] 完成 P1 100 kimg 短跑并估算基线时间；
- [x] 运行 P1 2000 kimg 双卡基线训练（最终 FID50k_full ≈ 13.0）；
- [x] 将 E2-E5 对照配置统一对齐到 1500 kimg 公平预算、评估 `mirror=false`；
- [ ] **回填数据溯源**：`data/lsun_church256.md` 补下载日期/来源、原始包大小、zip 的大小与 md5（目标机 `ls -l` + `md5sum`，报告复现性需要）；
- [x] 运行 E2-E5 训练到 1500 kimg；
- [x] E1 单卡补算 FID/KID/PR（@1512 kimg）；
- [x] E2-E5 单卡补算 FID/KID/PR，并用 `scripts/analyze_results.py` 汇总学习曲线与总表；
- [x] 关键两轴补第二随机种子（seed=1，重跑 E1/E2/E4 至 1500 kimg）并核验可用、计算跨种子均值/范围；
- [x] 生成报告用图：学习曲线（`results/analysis/`）、每组样本网格（`results/samples/p2-*`）、截断展示（`show_trunc07`）、风格混合（`stylemix`）、插值（`interp_e1`）、最近邻（`results/nn/e1-final`）均已产出；
- [ ] 撰写 20+ 页中文报告；并按需补：失败案例标注、水印/伪水印统计、（可选）seed1 样本与各组最近邻。

## P0：流程验证

- [x] 建立项目目录与实验代码骨架；
- [x] 固定 NVIDIA StyleGAN2-ADA PyTorch 后端提交；
- [x] 建立第三方后端自动下载与补丁流程；
- [x] 适配 PyTorch 2.x 的 `InfiniteSampler`；
- [x] 修复 ADA/R1 所需的 `grid_sample` 二阶梯度兼容性；
- [x] 建立合成数据生成与官方 ZIP 转换流程；
- [x] 建立统一训练、快照加载、生成和评估入口；
- [x] 建立软硬件环境记录脚本；
- [x] 建立 GPU、CUDA 工具链与融合算子严格预检；
- [x] 在本地完成训练、快照加载、继续训练、生成和轻量指标端到端验证；
- [x] 通过 12 项不依赖 GPU 的单元测试；
- [x] 在目标 RTX PRO 6000 机器上通过严格 P0 流程；
- [x] 在目标机器上验证默认 `bgc` ADA 增强短跑；

详细记录：`report/p0_validation.md`

## P1：基础复现

- [x] 确定正式数据集、规模、分辨率与许可证口径；
- [x] 建立正式数据下载、校验和转换记录；
- [x] 锁定 StyleGAN2-ADA 基线训练配置；
- [x] 完成基线短跑并估算完整训练时间（双卡 paper256 约 11-12 sec/kimg）；
- [x] 完成至少一次完整基线训练（2000 kimg）；
- [x] 生成可信的基线样本、训练曲线和官方 FID/KID/Precision/Recall（已离线补齐，见 `results/analysis/`）；
- [ ] 总结基线主要失败模式（待撰写）。

> **说明**：P2 诊断、P3 方向选择、P4 正式实验三段已合并落地为一套 **E1–E5 因素矩阵 + seed=1 重复**
> （详见 `report/p2_experiment_plan.md`）。下面按原阶段勾选实际完成情况——**训练与评估实验已基本齐备**，
> 余下主要是「报告出图」与「撰写」，不再需要新的训练开销。

## P2：问题诊断（由 E1–E5 因素矩阵实现）

- [x] 分析数据规模对结果的影响（E4 50k vs E1 100k，固定 1500 kimg）；
- [x] 分析训练预算对结果的影响（1500 kimg 公平点 vs 2000 kimg 基线，FID–kimg 学习曲线）；
- [x] 分析关键超参数与训练稳定性（E3 固定 p=0.2、E5 ADA target=0.4）；
- [x] 检查判别器过拟合现象（ADA p 曲线 + recall 退化，数据已就绪）；
- [x] 完成生成样本与训练集最近邻对比（`results/nn/e1-final/`，E1 最终模型已出；其余组可选补）；
- [x] 确定稳定、值得研究的实验现象（增强有用、固定预算下 50k 不劣于 100k，已 n=2 种子佐证）。

## P3：研究方向选择（已合入因素矩阵，无独立阶段）

- [x] 确定最终研究问题（有限数据下 ADA 行为、数据规模与 ADA target 诊断）；
- [x] 锁定实验组、控制变量、训练预算和主要指标（E1–E5，1500 kimg 公平预算，主指标 FID）；
- [x] 明确对照方法范围（ADA 开/关 + 固定 p + target 扫描；本轮不引入迁移学习/FreezeD）；
- [x] 记录方向选择依据（见 `report/p2_experiment_plan.md`）。

## P4：正式实验（E1–E5 + seed=1 重复）

- [x] 完成主实验所有对照组（E1–E5）；
- [x] 完成必要的消融实验（E2 无增强消融）；
- [x] 完成关键参数分析（E3 固定 p、E5 ADA target）；
- [x] 使用多个随机种子验证关键结论（ADA 开/关、数据规模两轴已 n=2 种子）；
- [x] 汇总 FID、KID、Precision、Recall 和资源效率指标（`results/analysis/`）；
- [x] 生成报告图：学习曲线、样本网格、插值、风格混合、最近邻均已产出（`results/{analysis,samples,nn}/`）；
- [ ] 整理失败案例及原因分析（待人工挑图 + 撰写）；水印/伪水印统计待补。

## 报告与提交

- [ ] 补充 StyleGAN2、StyleGAN2-ADA 和评价指标相关文献；
- [ ] 完成摘要；
- [ ] 完成引言；
- [ ] 完成研究方法；
- [ ] 完成实验与结果分析；
- [ ] 完成结论；
- [ ] 完成参考文献；
- [ ] 检查报告图表与代码输出一致；
- [ ] 检查最终代码可在干净环境复现；
- [ ] 清理不应提交的数据、缓存和大体积快照；
- [ ] 在 2026-06-30 24:00 前完成最终提交。
