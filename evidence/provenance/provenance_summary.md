# LSUN 目标机溯源摘要

采集时间（UTC）：2026-06-09T12:11:25.540217+00:00

## 已直接观察的文件事实

- 转换 ZIP：`/root/autodl-tmp/deeplearning-course-stylegan/data/processed/lsun-church-256-100k.zip`
- ZIP 大小：19712169438 字节
- ZIP 文件时间（UTC）：2026-06-05T08:51:40.005719+00:00
- SHA-256：`0bef00af2e08661b1b76b17576863287a5403f71f6fc5ba58a7a3bbd225bc25d`
- MD5：`e3e655220014f8d52cb9858df4d77d36`
- ZIP 条目：100001，其中图像 100000 张、`dataset.json` 1 个
- 抽样检查：32 张，均为 256x256 RGB
- 原始 LMDB：`/root/autodl-tmp/deeplearning-course-stylegan/data/raw/lsun/church_outdoor_train_lmdb`
- 原始目录：2 个文件，共 2788702848 字节
- 原始文件时间范围（UTC）：2026-06-05T08:03:33.937429+00:00 至 2026-06-05T08:04:07.229490+00:00

文件时间只表示目标机文件系统中的修改时间，不认定为下载日期。

## 下载来源判定

Shell 历史不可用，实际下载来源与下载日期无法恢复。

项目文档中的 LSUN 官方页和 OpenDataLab 只保留为计划来源，不能冒充实际下载事实。

## 采集环境

- 主机：`autodl-container-3hy7n2qhcg-4855777a`
- 平台：`Linux-5.15.0-78-generic-x86_64-with-glibc2.35`
- Python：`3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0]`
- PyTorch：`2.8.0+cu128`
- PyTorch CUDA：`12.8`
- NVCC：见目标机 JSON 原文

本次采集进程未取得可用 GPU，`nvidia-smi` 也因权限不可用。该状态只描述元数据采集时的
进程环境，不否定训练日志与配置中已经封闭的单卡、双卡、四卡和六卡调度事实。
