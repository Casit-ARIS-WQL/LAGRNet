# LAGRNet
## 目录

- [特性](#特性)
- [架构概览](#架构概览)
- [环境要求](#环境要求)
- [安装](#安装)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
  - [训练](#训练)
  - [推理](#推理)
- [配置参数](#配置参数)
- [评估指标](#评估指标)
- [项目结构](#项目结构)

---

## 特性

- 🔥 **Swin Transformer 骨干**：提取多尺度（stride 4/8/16）层次特征，支持 `timm` 预训练权重，可自动降级为轻量卷积金字塔。
- 🌀 **GFM**：通过 PGL(3) 单应性变换群对特征图进行轨道聚合，增强视角不变性。
- 🔁 **RCL**：基于分级 Cauchy 乘积的环卷积，实现多尺度特征的跨度融合。
- 🧩 **SM**：将补丁图像建模为细胞层（cellular sheaf），通过 Cayley 正交限制映射和显式 Euler 扩散步骤平滑深度预测。
- 📊 **Loss**：光度损失（photometric）、群一致性损失（group consistency）、层状能量损失（sheaf energy）与有监督深度 L1 损失联合优化。
- 🖥️ **TensorBoard**：自动记录训练/验证损失与指标，支持断点续训。

---

## 架构概览

```
Input Image (3, H, W)
        │
        ▼
┌─────────────────────┐
│   SwinBackbone      │  ─── 输出三阶段特征 F0/F1/F2 (stride 4/8/16)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   GFM               │  ─── PGL(3) 轨道聚合，注意力加权融合
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   RCL               │  ─── 分级环卷积，多尺度特征融合
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   SM                │  ─── 细胞层扩散，平滑深度场
└─────────────────────┘
        │
        ▼
   Depth Map (1, H, W)  ─── 输出归一化到 [0,1]，乘以 max_depth 得到米制深度
```

---

## 环境要求

| 依赖包 | 最低版本 |
|---|---|
| Python | 3.8+ |
| PyTorch | 1.12.0 |
| torchvision | 0.13.0 |
| timm | 0.6.0 |
| numpy | 1.21.0 |
| Pillow | 9.0.0 |
| tensorboard | 2.10.0 |
| matplotlib | 3.5.0 |
| tqdm | 4.64.0 |

---

## 安装

```bash
# 克隆仓库
git clone https://github.com/Casit-ARIS-WQL/depth_estimation.git
cd depth_estimation

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# 安装依赖
pip install -r requirements.txt
```

---

## 数据集准备

本项目使用 **KITTI 深度估计数据集**，支持 Eigen split。请按如下目录结构组织数据：

```
/data/kitti/
├── raw_data/                          # KITTI 原始同步+校正数据
│   ├── 2011_09_26/
│   │   ├── 2011_09_26_drive_0001_sync/
│   │   │   ├── image_02/data/        # 左彩色相机
│   │   │   ├── image_03/data/        # 右彩色相机
│   │   │   └── ...
│   │   └── calib_cam_to_cam.txt
│   └── ...
├── depth_annotated/                   # KITTI 深度补全标注
│   ├── train/
│   │   └── <drive_sync>/proj_depth/groundtruth/image_02/
│   └── val/
└── splits/                            # Eigen split 文件
    ├── eigen_train_files.txt
    ├── eigen_val_files.txt
    └── eigen_test_files.txt
```

**Split 文件格式**（每行一个样本）：
```
2011_09_26/2011_09_26_drive_0001_sync 0000000000 l
```
字段含义：`日期/驾驶序列`  `帧编号`  `相机侧（l=左, r=右）`

---

## 快速开始

### 训练

**基础训练**（使用默认配置）：
```bash
python train.py --data_root /data/kitti
```

**自定义超参数**：
```bash
python train.py \
    --data_root /data/kitti \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --gpu 0
```

**从断点续训**：
```bash
python train.py \
    --data_root /data/kitti \
    --resume checkpoints/latest.pth
```

**不使用预训练骨干**：
```bash
python train.py --data_root /data/kitti --no_pretrained
```

训练日志与 TensorBoard 文件保存在 `./runs/`，模型权重保存在 `./checkpoints/`：
- `best.pth`：验证集 abs_rel 最优模型
- `latest.pth`：最新 epoch 模型
- `epoch_XXX.pth`：按周期保存的检查点

**启动 TensorBoard**：
```bash
tensorboard --logdir ./runs
```

---

### 推理

**单张图像推理**：
```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --image path/to/image.png \
    --colormap
```

**批量推理（文件夹）**：
```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --image_dir path/to/images/ \
    --output_dir ./output \
    --colormap \
    --save_numpy
```

**在 KITTI 验证/测试集上推理并评估**：
```bash
python inference.py \
    --checkpoint checkpoints/best.pth \
    --data_root /data/kitti \
    --split val \
    --output_dir ./output \
    --colormap
```

**推理输出**：
- `<name>_depth.png`：16-bit PNG 深度图（KITTI 格式，值 = 深度(m) × 256）
- `<name>_depth_colored.png`：plasma 色图可视化（`--colormap` 开启）
- `<name>_depth.npy`：原始浮点深度数组（`--save_numpy` 开启）

---

## 配置参数

所有默认配置位于 `configs/kitti_config.py`，可通过命令行参数覆盖：

### 训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_root` | *(必填)* | KITTI 数据集根目录 |
| `--epochs` | 50 | 训练总轮数 |
| `--batch_size` | 8 | 每批样本数 |
| `--lr` | 1e-4 | 初始学习率（AdamW） |
| `--num_workers` | 4 | 数据加载线程数 |
| `--height` | 352 | 输入图像高度 |
| `--width` | 1216 | 输入图像宽度 |
| `--max_depth` | 80.0 | 最大深度（米） |
| `--resume` | None | 断点续训路径 |
| `--log_dir` | `./runs` | TensorBoard 日志目录 |
| `--checkpoint_dir` | `./checkpoints` | 模型保存目录 |
| `--no_pretrained` | False | 禁用预训练骨干 |
| `--gpu` | 0 | 使用的 GPU 编号 |

### 推理参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--checkpoint` | *(必填)* | 训练好的模型权重路径 |
| `--image` | None | 单张输入图像路径 |
| `--image_dir` | None | 批量输入图像目录 |
| `--data_root` | None | KITTI 数据根目录（用于 split 推理） |
| `--split` | `test` | KITTI split（`val` 或 `test`） |
| `--output_dir` | `./output` | 输出保存目录 |
| `--colormap` | False | 保存彩色深度可视化 |
| `--save_numpy` | False | 保存 `.npy` 原始深度数组 |
| `--gpu` | 0 | 使用的 GPU 编号 |

### 模型核心配置（`configs/kitti_config.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `unified_channels` | 256 | 统一特征通道数 |
| `K_orbit` | 8 | GFM 轨道大小（K 个单应变换） |
| `D_grade` | 1 | RCL 分级次数 |
| `sheaf_dim` | 128 | SheafModule 潜在维度 d |
| `patch_grid` | (7, 7) | Sheaf 补丁网格大小 |

### 损失权重

| 参数 | 默认值 | 说明 |
|---|---|---|
| `lam_pho` | 1.0 | 光度损失权重 |
| `lam_grp` | 0.1 | 群一致性损失权重 |
| `lam_sheaf` | 0.01 | Sheaf 能量损失权重 |
| `lam_sm` | 0.1 | 平滑度损失权重 |

---

## 评估指标

在 KITTI 深度评估中使用标准指标（有效深度范围：1e-3 m ~ max_depth m）：

| 指标 | 说明 | 越小越好 |
|---|---|---|
| `abs_rel` | 平均绝对相对误差 | ✅ |
| `sq_rel` | 平均平方相对误差 | ✅ |
| `rmse` | 均方根误差（米） | ✅ |
| `rmse_log` | 对数空间均方根误差 | ✅ |
| `δ < 1.25` (a1) | 阈值精度（1.25） | 越大越好 |
| `δ < 1.25²` (a2) | 阈值精度（1.5625） | 越大越好 |
| `δ < 1.25³` (a3) | 阈值精度（1.953） | 越大越好 |

---

## 项目结构

```
depth_estimation/
├── model.py                    # LAGRNet 完整模型定义
│   ├── SwinBackbone            # Swin Transformer 骨干（支持 timm / 轻量卷积回退）
│   ├── GFM                     # 群特征流形模块（PGL(3) 单应变换轨道聚合）
│   ├── RCL                     # 分级环卷积层
│   ├── SheafModule             # 细胞层扩散模块（Cayley 正交限制映射）
│   ├── LAGRNet                 # 完整网络（骨干 + GFM + RCL + Sheaf + 解码头）
│   └── LAGRLoss                # 多目标联合损失函数
├── train.py                    # 训练脚本（支持断点续训、TensorBoard、验证）
├── inference.py                # 推理脚本（单张/批量/KITTI split）
├── configs/
│   ├── __init__.py
│   └── kitti_config.py         # 所有超参数默认配置
├── datasets/
│   ├── __init__.py
│   └── kitti_dataset.py        # KITTI 数据集加载（Eigen split，数据增强）
└── requirements.txt            # Python 依赖列表
```

---
