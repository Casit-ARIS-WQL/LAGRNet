# LAGRNet — revised manuscript implementation

本目录从 <https://github.com/Casit-ARIS-WQL/LAGRNet> 的
`1560157222b1ba3958def21e73adbc2e9167577a` 下载，在该版本上修改。
依据本地最新 `jssc-module.tex` 的方法与训练协议实现；保留 Git 历史便于对比。

**这是一份新实现，不是已报告实验的原始训练记录。未执行四张 H20 上的 480K 次训练，
未声明复现论文中的精度、消融或速度。** 网络结构、损失计算和训练评估设置见
[实现说明](docs/method_alignment.md)。测试结果见 [验证记录](docs/verification.md)。

## 主要实现

- 真正的 Swin 前三个 stage，stride 4/8/16；动态分辨率，无静默 CNN 替代。
- GFM：默认 K=4，MLP 生成、Frobenius 归一化、逆 homography 双线性采样、attention 和 1×1 投影。
- RCL：每个模块只接收一个输入特征，由 Avg Pool 在模块内部构造 P₀–P₆（D=6）。
  这七个分次特征不等于七个 RCL 模块。图 2 的四个 RCL（两个编码器分支、两个自顶向下融合）均已实现；
  各模块内部共享 3×3 核，按不重叠输出通道 mask 构造分次算子，先卷积再 resize，最后以 softmax 归一化的可学习权重融合各次数输出。
- SM：重叠窗口分别池化，按实际交叠构建 Čech nerve 的 1-skeleton（包括有交叠的对角邻居）；
  Cayley 正交限制映射、两步 ReLU 传播，最终 cochain 重算映射并计算 learned energy。
  默认 518×518 输入形成 49 个 patch、156 条无向边；图随输入尺寸重建。
- 五项实际激活的损失：`depth + pho + 0.5 grp + 0.1 sheaf + 0.01 sm`。
  光度误差经 GFM 几何预测分支回传至 backbone；原图与变换图都会预测，群一致性在有效对应区域计算。
- 六个有标注源按样本拼接；KITTI / NYUv2 / ETH3D 仅供评估。
- raw relative inverse-depth 输出；评估用 GT 做 scale–shift 对齐，禁止混用旧的 metric-depth 输出逻辑。

## 环境

Python 3.10+。请在独立环境安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

当前机器是 Intel macOS，测试使用 PyTorch 2.2.2 / torchvision 0.17.2；完整本机版本在
`requirements-tested-macos.txt`。CUDA 主机按 PyTorch 官方方式安装匹配的 torch/torchvision，
再安装其余依赖。训练默认使用 pretrained Swin-Base；首次运行会下载 timm 对应预训练权重。
权重下载失败会报错，不会悄悄改成随机 CNN。

## 数据准备

使用明确的 JSONL 清单，不假定不同数据集使用相同文件结构或深度编码。
每行至少包含：

```json
{"dataset":"BlendedMVS","split":"train","image":"rgb/0001.png","depth":"depth/0001.npy","depth_scale":1.0}
```

`image` 为 RGB，`depth` 为与 RGB 原始像素对齐的二维深度 `.npy` 或单通道图像。
`depth_scale` 是将存储值转换为深度的**乘数**，由实际标注格式决定；0、NaN、Inf 为无效标签。
路径相对清单文件解析。若原始标注是 disparity、inverse depth 或带相机模型的其他格式，
先按数据集真实标定转换为本文接口需要的深度，不能直接重命名为 depth。

训练清单须包含 `BlendedMVS, DIML, HRWSI, IRS, MegaDepth, TartanAir` 六个名字，
`split` 均为 `train`；可用一份拼接清单或重复传入 `--manifest`。不做数据集级重平衡。
系统检查名字、文件、形状和重复图像，但实际划分成员身份需要由正确的官方清单保证。

评估清单的合法配对为：

| dataset | split |
|---|---|
| KITTI | eigen_test |
| NYUv2 | official_test |
| ETH3D | all_available_gt |

可增加 `mask`（原始分辨率二值有效区域）、`min_depth`、`max_depth`；它们明确决定打分像素。
**没有暗中假设 benchmark crop、深度上限或 GT 投影方法。** 图像推理不裁剪；打分时可应用清单指定的 mask。
清单不是完整数据下载器，项目不包含六个大型训练数据集或 benchmark GT。

## 训练

```bash
torchrun --standalone --nproc_per_node=4 train.py \
  --manifest /data/manifests/six_labeled_sources.jsonl \
  --output runs/paper
```

默认全局 batch=128（四卡时每卡 32），AdamW，初始学习率 `1e-4`，poly power=0.9，
总步数 480000。短边 resize 到 518 后随机裁剪 518×518，RGB 用 ImageNet mean/std。
监督标签转为 image-wise [0,1] inverse depth，预测与标签分别做 median/MAD 归一化。
训练日志记录每一项 loss；任何启用的辅助项缺输入都会报错。

可用 `--config experiment.json` 覆盖 `configs/paper.py` 中的字段（未知字段报错）；
覆盖后的完整参数会保存进 checkpoint 和 `config.json`。显式将某项权重设为 0 才表示消融。
`--stop-after 100` 可作短程检查而不改变 480K 的 LR 衰减周期。

```bash
python train.py --manifest /data/manifests/six_labeled_sources.jsonl \
  --output runs/paper --resume runs/paper/latest.pt
```

保存模型、optimizer、步数、配置、训练清单 SHA256、源码 SHA256 和基线 commit。
恢复时校验配置和清单。恢复 sample 顺序，但不承诺多 worker 数据增强流逐位一致。
不根据测试集指标选 `best`；`latest.pt` 是最近训练步，最终模型为 `step_480000.pt`。

## 推理与评估

```bash
python inference.py --checkpoint runs/paper/latest.pt \
  --image /path/image.png --output-dir outputs/predictions

python evaluate.py --checkpoint runs/paper/latest.pt \
  --manifest /data/manifests/kitti_eigen_test.jsonl \
  --manifest /data/manifests/nyuv2_official_test.jsonl \
  --manifest /data/manifests/eth3d_all_gt.jsonl \
  --diagnostics --output outputs/evaluation.json
```

`.inverse_depth.npy` 保留浮点原始预测，无 min–max、无“乘 80 米”。`.preview.png` 只用于可视化。
评估在原始 GT 分辨率上逐图做最小二乘 scale–shift 拟合，计算 inverse-depth 指标；
三套数据齐全时才输出等权 dataset mean。没有 GT 的输入只能做相对深度推理，不能得到标定米制深度。

非正的对齐预测默认报错。若实验协议明确要求，只能显式指定 `--nonpositive exclude` 或 `clamp`；
输出会记录选择及每张图的正值覆盖率。
`--diagnostics` 在同一最终 cochain 上记录 learned sheaf energy 与 ordinary graph energy，
二者都除以节点数，均含 1/2 因子，不把两者当成同一个统计量。

## 文件

| 文件 | 作用 |
|---|---|
| model.py | Swin、GFM、单输入 Avg Pool 分次 RCL、SM、decoder |
| geometry.py | 可微 homography、有效区、合成与重建 |
| losses.py / engine.py | 五项损失及两路预测 |
| datasets/relative_depth.py | 六源训练与独立评估接口 |
| train.py / configs/paper.py | 分布式迭代训练、配置、记录与恢复 |
| evaluation.py / evaluate.py | 逆深度对齐和评估 |
| inference.py | 原始相对逆深度导出 |
| tests/ | 数学、梯度、数据协议与真实 Swin 检查 |

`configs/kitti_config.py` 和 `datasets/kitti_dataset.py` 仅保留作旧版本参考，
新入口不再导入它们。当前 checkpoint 使用 format_version=3。旧版本的固定平均 Fuse、固定四邻域 SM、两输入 RCL 或 metric-depth checkpoint 不能直接用于当前架构，加载时会拒绝。
