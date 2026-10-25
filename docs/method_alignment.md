# 方法与实现说明

基线源码：`1560157222b1ba3958def21e73adbc2e9167577a`。
2026-09-29 按最新正文和现有图 2、图 4、图 5 校正实现；图文件保持原样，正文与回复信同步定义。

## 模块与计算流程

| 论文内容 | 对应实现 | 检查内容 |
|---|---|---|
| Backbone：前三个 Swin stage，4/8/16 | `SwinBackbone` | 真实 timm Swin，非正方形与非整除尺寸，不允许静默 fallback |
| GFM：MLP、K=4、Frobenius、逆变换、attention、1×1 | `GFM`, `geometry.warp` | 正负 projective representative 等价、有效像素与反向传播 |
| RCL：D=6，C×C×3×3 共享核，disjoint output mask，先卷积后 resize | `RCL` | 单输入 Avg Pool 七个分次特征、逐项 Cauchy 求和、线性部分、可学习 Fuse、每组核及融合权重梯度 |
| 分次环只作用于固定权重/网格的线性算子 | `RCL(..., linear_only=True)` | 不把 BN/ReLU 误称为环同态，不把 channel masks 误称为频带 |
| SM：重叠 patch 的 Čech nerve、不同 learned restrictions、Cayley、两步 ReLU 传播 | `SheafModule` | patch 实际交集、逐 patch 池化、正交性、Laplacian 与能量对应 |
| 最终 cochain 重新生成限制映射 | `SheafModule.forward` | loss 依赖最终 H，也经过 restriction generator 回传 |
| `L_sheaf = sum_b E(h_b)/(BN)` | `LAGRLoss` | 边只计一次，E 含 1/2，按 batch 和节点数归一化 |
| Label-based affine-invariant inverse depth | `RelativeDepthDataset`, `affine_invariant_depth_loss` | 同一有效像素分别 median/MAD，标签 image-wise min–max |
| 五项 loss，权重 1,1,.5,.1,.01 | `engine.objective`, `LAGRLoss` | 每项实际计算，缺少必需输入报错，不静默置零 |
| 合成 source 来自原图，光度项训练 backbone | `synthetic_view_reconstructions` | 经 GFM 预测变换回到 backbone；无 detach、无外部相机帧 |
| 原图和 homography 增广图的群一致性 | `engine.objective` | 两次网络预测、逆变换和有效区域 |
| edge-aware smoothness 的像素求和 | `smoothness_loss` | RGB 原始 [0,1]，空间求和、batch 平均 |
| 六个 labeled sources；三个 benchmark 只测试 | `RelativeDepthDataset`, `train.py` | 训练/评估名字和 split 检查，不使用 unlabeled teacher 分支 |
| 518 短边、训练 crop，未裁剪推理，RGB 标准化 | dataset/inference | 标签与图像相同几何变换、原尺寸打分 |
| AdamW, 1e-4, poly .9, global B128, 480K | `PaperConfig`, `train.py` | 按 optimizer iteration 调度，四卡全局 batch128 |
| 逆深度 scale–shift 对齐再打分 | `evaluation.py` | 不先 min–max、不还原 metric depth，三个数据集等权 mean |

## 每个 RCL 的单输入和七个内部特征

RCL 只接收一个 B×C×H×W 特征张量 F。D=6 表示最高次数，共 P₀(F)…P₆(F) 七个内部特征分量，**不是七个 RCL，也不是七个外部输入**。

P_j(F)=AvgPool(F, kernel_size=2^j, stride=2^j, ceil_mode=True, count_include_pad=False)。
所有分量直接从同一 F 池化；边界不足一个窗口时只对实际像素平均。P₀ 为原特征。
固定形状的 Avg Pool 是线性映射；分次环证明对应后面的共享核卷积和重采样算子。

- 65×65 输入的七个网格为 65、33、17、9、5、3、2。
- 33×33 输入的七个网格为 33、17、9、5、3、2、1。
- 小输入可能重复出现 1×1 网格，仍以不同次数标记。
- 各次数结果先按 Cauchy 规则求和，再 BN/ReLU；Figure 4 的 Fuse 将所有次数结果 resize 到输出网格，以 beta=softmax(a) 融合。每个模块独立学习 D+1 个 logits；零初始化仅使初始权重相等。
- 不再采用两个 Swin 特征之间的插值混合，也不再仅选择最高次数输出。

## 图 2 的四个 RCL 模块

| 模块 | 单个输入（必要时先对拼接通道投影） | 输出网格（518输入） | 通道 |
|---|---|---|---|
| rcl1 | F₁ | 65→33 | 256 |
| rcl2 | F₂ | 33→17 | 512 |
| rcl_fuse1 | concat(rcl1, resize(SM(rcl2))) 后 1×1 投影 | 33→33 | 256 |
| rcl_fuse0 | concat(GFM(F₀), resize(rcl_fuse1)) 后 1×1 投影 | 130→130 | 128 |

每个模块均内部生成 D+1 个分次特征，并拥有自己的 W_RCL；共享核只在该模块内部跨次数使用。
两个编码器分支输出按 ceil(H/2)、ceil(W/2) 取网格，两个融合模块保持当前输入网格。
SM 使用 coarsest 分支；fine 融合结果进入两次 2× 上采样的 decoder。
默认保持 Swin-Base 的 128/256/512 原生通道，unified_channels=None。显式小通道覆盖仅用于测试或消融。

## 重叠 patch 与 Čech nerve

SM 先以 1×1 卷积将通道投影至 d，再分别在每个实际窗口内 adaptive-average-pool 到一个向量。
每个轴长 L 使用 g=min(7,L) 个分区（测试/消融可通过 patch_grid 指定上限），分界为 floor(iL/g)。
窗口将每个分区向两侧各扩展一个特征网格像素，裁剪到有效区域；小尺寸产生的重复区间合并。
两个轴的区间作笛卡尔积，按行优先排列 patch。该窗口集合覆盖全部输入网格。

A_ij=1 当且仅当 i≠j 且两个窗口至少共享一个实际网格像素；仅接触半开边界不算交叠。
不固定为四邻域，不凭网格索引省略交叠的对角 patch，也不预设只连接最近位置。
默认 coarsest 输入 17×17 产生 7×7=49 个 patch 和 156 条无向边。
图和窗口在输入尺寸变化时重建，不写入 checkpoint；极小输入可以只有一个 patch，此时边集为空、能量为零。

每条边只保留 u<v 的一个方向，两端限制映射分别由有序特征对生成。
两次传播均为 ReLU(H − tau W_k L_S(H) H)，每次重新生成限制映射；最终 cochain 再次生成映射供能量使用。
最终 patch 特征按窗口网格排列后用于 top-down 融合。普通图能量与 learned sheaf energy 是分别计算的诊断量。

## 光度项及 backbone 梯度

参考图为原始输入，source 为合成视图。光度项使用 GFM 预测的 homography，
通过以下路径训练 backbone：

`I -> Swin F0 -> GAP/MLP -> theta_s -> synthetic source -> inverse reconstruction -> L_pho`。

令 S 为特征像素到图像像素的中心对齐缩放矩阵，image-space homography 为 S theta_s S^-1。
合成 I_s=rho(theta_s)I，重建 rho(theta_s^-1)I_s。两条采样路径都保留梯度，
不 detach source、theta 或 backbone 特征。逐视图使用有效往返区域计算 RGB L1，再对视图和 batch 平均。

该计算实现 synthetic-view consistency，并把梯度连到 backbone；
**它是离散重采样一致性正则项，不是来自独立真实视图的几何观测。** 理想连续可逆变换的往返为恒等，
该项的数值来自离散插值。它不能单独监督 metric depth，也不保证 learned geometry 的唯一性，
恒等变换可能使该项很小。整体模型仍由深度监督和其他目标共同训练。

## 参数与数值设置

配置和代码采用以下设置：

- Backbone 为 `swin_base_patch4_window7_224`。
- 四个 RCL 通道为 256、512、256、128；patch grid 上限=7×7，单像素交叠扩展，tau=.5，SM ReLU；stalk 默认等于 Swin 的 C0=128。
- Decoder 通过两层2×上采样从 stride4 回到原图。
- AdamW weight_decay=1e-4，smoothness gamma=1；颜色 jitter=.1，flip 概率.5。
- 训练 homography 在归一化坐标内 I+Uniform(-.05,.05)，h33=1。
- GFM 最后一层权重以 std=1e-5 初始化，使初始变换接近恒等并保留有效区域；
  Frobenius 归一化本身不保证可逆，奇异矩阵报错，不用 identity 悄悄替换。
- MAD 归一化的数值 eps=1e-6；全无标注或常数 target 无法提供相对结构，明确报错。
- 评估 scale–shift 为逐图最小二乘；RMSE_log 用自然对数。
- 评估 mask、深度上下限、GT 文件/单位由 manifest 明确给出，不根据数据集名字猜测。
- 对齐后非正值默认报错；显式 exclude/clamp 策略、覆盖率和有效像素数写入评估记录。
- 逐图指标在数据集内等权平均，再对三个数据集等权平均；此聚合细节也记录在 JSON。
- checkpoint 按迭代保存，不以测试集选择最佳模型；预训练初始化由 timm 的模型权重配置决定。

## 不作为无条件实现要求的理论条件

GFM 的 conjugation / attention invariance 是条件命题的前提，不是普通 MLP 的恒等性质。
当前代码没有假装它们自动成立。SE(2) 的单次采样误差证明也没有被用来替代一般 PGL(3) 验证。
RCL 的投影矩条件在共享掩码核族中可构造实现；训练没有把这些矩条件作为硬约束。
learned sheaf energy 不被 ordinary graph energy 替换，后者仅作为另一个诊断量。
两步非线性 feature-adaptive diffusion 不宣称保证能量单调下降或达到 global section。

## 历史与后续实验

改后的代码需要重新训练才能给出它自己的实验结果。已有真实消融结果保持原样。
本项目只核对实现计算、数据接口、梯度和局部运行，不凭数值单元测试声称完成论文实验复现。
完整训练数据、官方 split 成员、具体评分 mask 和训练好的 checkpoint 不在本地项目内。

## 分布式实现细节

原图/变换图的两次 forward 汇入同一个 loss。DDP 采用 `broadcast_buffers=False`，
防止第二次 forward 广播 BN/Cayley buffers 时修改第一次 forward 为反向传播保存的张量。
BatchNorm 采用每个进程的 batch 统计，参数梯度仍全局平均；不是 SyncBatchNorm。
checkpoint 保存 rank0 的 running statistics。默认全局 B128 / 四卡得到每卡 B32。

## 参数计数

图 2 对应的默认架构为 72,148,029 参数（72.1M），其中 backbone 为 59,436,664。
此为参数张量计数，未重测原有 A100 延迟和 FLOPs。

checkpoint 格式为 3；旧图及旧 Fuse 架构的文件显式拒绝加载，避免带着旧定义继续训练或评估。
