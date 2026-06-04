# 第三方仓库

## StyleGAN2-ADA PyTorch

当前实验后端使用 NVIDIA 官方 StyleGAN2-ADA PyTorch 仓库：

- 源代码：<https://github.com/NVlabs/stylegan2-ada-pytorch>
- 固定提交：`d72cc7d041b42ec8e806021a205ed9349f87c6a4`
- 许可证：NVIDIA Source Code License-NC

使用以下命令重新创建被 Git 忽略的后端代码目录：

```bash
python scripts/bootstrap_stylegan2_ada.py
```

bootstrap 脚本还会应用 `patches/stylegan2-ada-pytorch-modern-pytorch.patch`。该补丁恢复
PyTorch 2.x 中 ADA/R1 所需的自定义 `grid_sample` 二阶梯度路径，适配现代算子签名，
并根据当前 PyTorch Sampler API 更新 `InfiniteSampler`。上游提交保持固定，本地差异可以
通过补丁稳定复现。

不要手动修改 bootstrap 创建的后端代码目录。项目专用的包装逻辑和实验代码应放在
`src/` 与 `scripts/` 中。

## StyleGAN

`stylegan/` 是 NVIDIA 官方旧版 TensorFlow 1.x 实现，仅作为参考，不被当前实验流程使用：

- 源代码：<https://github.com/NVlabs/stylegan>
- 当前固定提交：`1e0d5c781384ef12b50ef20a62fee5d78b38e88f`
- 许可证：CC BY-NC 4.0

该目录被 Git 忽略，且不由 StyleGAN2-ADA bootstrap 脚本管理。需要重新创建时运行：

```bash
git clone https://github.com/NVlabs/stylegan.git third_party/stylegan
git -C third_party/stylegan checkout --detach 1e0d5c781384ef12b50ef20a62fee5d78b38e88f
```
