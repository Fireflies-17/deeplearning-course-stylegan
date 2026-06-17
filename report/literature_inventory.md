# 核心文献检索清单

更新时间：2026-06-10

## 检索范围

本轮只建立报告所需的核心文献版图，不撰写文献综述正文。检索分为三组：

1. GAN 与 StyleGAN 架构演进；
2. 有限数据条件下的 GAN 训练与数据增强；
3. 生成模型评价指标与 LSUN 数据集。

优先采用论文原文页面、会议官方论文页和作者提供的预印本。当前课程并未要求中文文献，报告可
主要引用英文原始论文。

## 核心文献

| 编号 | 文献 | 年份/出处 | 报告用途 | 核验与本地状态 |
|---|---|---|---|---|
| R1 | Goodfellow et al., *Generative Adversarial Networks* | NeurIPS 2014 | GAN 基本博弈目标与生成器/判别器结构 | [arXiv](https://arxiv.org/abs/1406.2661)，未保存本地 PDF |
| R2 | Arjovsky et al., *Wasserstein Generative Adversarial Networks* | ICML 2017 | GAN 训练稳定性、模式崩溃和距离度量背景 | [PMLR](https://proceedings.mlr.press/v70/arjovsky17a.html)，已有本地 PDF |
| R3 | Karras et al., *Progressive Growing of GANs for Improved Quality, Stability, and Variation* | ICLR 2018 | StyleGAN 的渐进式生成背景 | [arXiv](https://arxiv.org/abs/1710.10196)，未保存本地 PDF |
| R4 | Karras et al., *A Style-Based Generator Architecture for Generative Adversarial Networks* | CVPR 2019 | 映射网络、风格调制、噪声注入、风格混合与截断 | [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html)，已有本地 PDF |
| R5 | Karras et al., *Analyzing and Improving the Image Quality of StyleGAN* | CVPR 2020 | StyleGAN2 的权重调制/解调、路径长度正则和伪影改进 | [CVF](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html)，未保存本地 PDF |
| R6 | Karras et al., *Training Generative Adversarial Networks with Limited Data* | NeurIPS 2020 | ADA 原理、判别器过拟合、增强概率与 target | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/8d30aa96e72440759f74bd2306c1fa3d-Abstract.html)，未保存本地 PDF |
| R7 | Zhao et al., *Differentiable Augmentation for Data-Efficient GAN Training* | NeurIPS 2020 | 有限数据增强的替代路线，用于界定 ADA 的方法背景 | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/55479c55ebd1efd3ff125f1337100388-Abstract.html)，未保存本地 PDF |
| R8 | Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* | NeurIPS 2017 | FID 定义、用途及其不能单独表示多样性的边界 | [NeurIPS](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html)，未保存本地 PDF |
| R9 | Bińkowski et al., *Demystifying MMD GANs* | ICLR 2018 | KID 的来源与无偏估计背景 | [OpenReview](https://openreview.net/forum?id=r1lUOzWCW)，未保存本地 PDF |
| R10 | Kynkäänniemi et al., *Improved Precision and Recall Metric for Assessing Generative Models* | NeurIPS 2019 | 将样本保真度与分布覆盖度分开解释 | [NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html)，未保存本地 PDF |
| R11 | Yu et al., *LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop* | arXiv 2015 | LSUN 数据集来源、规模和场景类别背景 | [arXiv](https://arxiv.org/abs/1506.03365)，未保存本地 PDF |

以上文献已覆盖当前报告中的模型、增强、数据和四类正式指标。StyleGAN3、GAN 综述以及更多
小样本 GAN 方法可作为扩展文献，但不是完成本课程报告的必要条件。

## 中文数据库检索式

知网自动化需要可交互浏览器、登录和可能出现的验证码。2026-06-10 实际启动自动检索后，
浏览器停留在 `chrome-error://chromewebdata/`，因此本轮记录为“CNKI 访问失败”，不解释为
零篇文献。随后使用 Crossref 公开题录进行替代检索，并逐条通过 DOI 解析到期刊官网。

后续在可访问知网的环境中，可在主题字段使用以下检索式复核和扩展：

```text
("生成对抗网络" OR "GAN") AND ("图像生成" OR "图像合成")
("StyleGAN" OR "风格生成对抗网络") AND ("高分辨率图像" OR "图像生成")
("生成对抗网络" OR "GAN") AND ("有限数据" OR "小样本" OR "数据增强")
("生成模型" OR "生成对抗网络") AND ("FID" OR "生成质量评价" OR "精确率召回率")
```

筛选建议：学术期刊和学位论文分开检索；优先近五年综述或技术分析；中文来源只用于辅助解释，
模型原理和指标定义仍引用原始英文论文。

## 中文文献补充

完整分类、检索记录和 DOI 链接见
[中文文献补充工作簿](cnki/中文文献补充_StyleGAN_20260610.xlsx)。

| 编号 | 文献 | 年份/期刊 | 分类 | 核验 |
|---|---|---|---|---|
| C1 | 李海军等：《基于ISE-StyleGAN的红外舰船图像生成算法》 | 2022，《光子学报》 | 直接相关：StyleGAN图像生成 | [DOI](https://doi.org/10.3788/gzxb20225112.1210004) |
| C2 | 秦魁等：《fire-GAN：基于生成对抗网络的火焰图像生成算法》 | 2023，《激光与光电子学进展》 | 直接相关：GAN图像生成 | [DOI](https://doi.org/10.3788/lop220989) |
| C3 | 李冰等：《基于条件生成对抗网络的红外图像生成算法》 | 2021，《光子学报》 | 直接相关：GAN图像生成 | [DOI](https://doi.org/10.3788/gzxb20215011.1110004) |
| C4 | 张印等：《基于条件对抗生成网络数据增强的相敏光时域反射仪模式识别》 | 2024，《光学学报》 | 间接相关：数据增强/有限样本 | [DOI](https://doi.org/10.3788/aos231392) |
| C5 | 孟奇等：《基于双通道生成对抗网络的镜片缺陷数据增强》 | 2021，《激光与光电子学进展》 | 间接相关：数据增强/有限样本 | [DOI](https://doi.org/10.3788/lop202158.2015001) |
| C6 | 林森、刘世本：《基于多尺度生成对抗网络的水下图像增强》 | 2021，《激光与光电子学进展》 | 间接相关：图像增强/质量重建 | [DOI](https://doi.org/10.3788/lop202158.1610017) |
| C7 | 宋巍等：《基于预处理图像惩罚的生成对抗网络水下图像增强》 | 2021，《激光与光电子学进展》 | 间接相关：图像增强/质量重建 | [DOI](https://doi.org/10.3788/lop202158.1210024) |
| C8 | 柯舒婷等：《生成对抗网络对OCT视网膜图像的超分辨率重建》 | 2022，《中国激光》 | 间接相关：图像增强/质量重建 | [DOI](https://doi.org/10.3788/cjl202249.1507203) |

8 条 DOI 均于 2026-06-10 成功解析至期刊官网。C1-C3 可进入国内研究现状；C4-C8
用于有限样本、数据增强和图像质量的应用背景，不替代 StyleGAN2-ADA、FID、KID 和
Precision/Recall 的原始英文来源。

## 使用边界

- R6 是本项目方法核心，必须引用。
- R8-R10 分别对应 FID、KID、Precision/Recall，指标章节不可只引用实现仓库。
- 不能用 FID 单独推出“多样性更高”或“没有过拟合”。
- E3、E5 只有单随机条件，相关文献不能替代重复实验。
- 水印和最近邻结果属于本项目经验分析，不应被包装成已有论文已经证明的结论。
