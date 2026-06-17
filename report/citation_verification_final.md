# 全文引用核查报告

核查日期：2026-06-10

## 1. 核查范围

本次核查覆盖以下成稿文件：

1. `report/introduction.md`
2. `report/literature_review.md`
3. `report/methods.md`
4. `report/findings.md`
5. `report/discussion.md`
6. `report/title_abstract_keywords.md`
7. `report/references.bib`

真实性判定复用检查点4已经完成的逐条核查结果。该轮核查已通过会议或论文官方页面、
Crossref、doi.org 与期刊元数据确认全部19条文献，并完成3处元数据规范化。本检查点重点检查
全文作者—年份引用与题录库之间的双向映射、重复项、遗漏项和跨章节格式一致性。

## 2. 全文映射结果

自动扫描得到30个“章节—引用”链接，共涉及19条唯一文献；`references.bib` 同样包含19条题录。

| 检查项目 | 结果 |
|---|---:|
| 唯一文内引用 | 19 |
| BibTeX题录 | 19 |
| 文内引用缺少题录 | 0 |
| 题录未被正文引用 | 0 |
| 未知作者—年份引用 | 0 |
| 重复BibTeX键 | 0 |
| DOI条目 | 8 |
| 重复DOI | 0 |

各文件的引用覆盖如下：

- 引言：7条，包括GAN、StyleGAN、ADA、DiffAugment、FID和Precision/Recall的核心来源。
- 相关工作与技术背景：19条，覆盖全部英文基础文献和8条中文应用文献。
- 研究方法与实验设计：无新增作者—年份引用。
- 实验结果与分析：无新增作者—年份引用。
- 讨论与结论：5条，均为前文已核实来源。
- 标题、摘要与关键词：无文内引用。

## 3. 真实性与元数据复核

| 类别 | 数量 | 判定 |
|---|---:|---|
| 英文文献 | 11 | 11/11 Confirmed |
| 中文文献 | 8 | 8/8 Confirmed |
| 合计 | 19 | 19/19 Confirmed |

英文文献均可映射到NeurIPS、ICLR/OpenReview、CVF、PMLR或arXiv/DataCite等官方来源。
8条中文文献均具有可解析DOI，题名、作者、年份和期刊信息已与Crossref及期刊元数据交叉核对。
本轮未发现虚构、无法确认或需要删除的文献。

## 4. 一致性处理

本轮统一了1处英文叙述式引用：

- `Karras 等（2018）` → `Karras et al. (2018)`

同时确认：

- `Karras et al. (2020a)` 对应StyleGAN2论文；
- `Karras et al. (2020b)` 对应StyleGAN2-ADA论文；
- 两条同年文献的a/b后缀在文内引用、参考文献和BibTeX中一致；
- `Bińkowski` 与 `Kynkäänniemi` 的姓名重音符号已保留；
- 中文文献作者、年份与DOI均一致。

## 5. 结论与装配要求

全文引用核查通过。当前引用库不存在缺失、悬空、重复或真实性存疑条目，可以进入完稿合成。

最终合并Word时应将 `literature_review.md` 中的“参考文献”部分移至全文末尾，避免参考文献
出现在第二章之后；只保留一份完整的19条参考文献列表。
