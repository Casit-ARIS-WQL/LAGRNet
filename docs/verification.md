# 本地验证记录 2026-09-29

本轮以最新正文的可学习 RCL Fuse 与 Čech nerve SM 为准重新验证。详细结果及源码校验值见 verification.json。

- **22 项自动测试全部通过**：单输入 Avg Pool 七分量、Cauchy 作用、非均匀可学习 Fuse 与梯度、窗口覆盖与实际交集、对角交叠、逐 patch 池化、动态尺寸、ReLU、单 patch 空边集、sheaf 能量、五项损失、真实 Swin 光度梯度及旧 checkpoint 拒绝。
- 默认 Swin-Base、K=4、D=6、stalk128 的 518×518 前向通过，输出为 1×1×518×518，预测与 learned energy 有限。随机初始化，无预训练下载。
- 默认 17×17 SM 输入生成 49 个重叠 patch、156 条无向交叠边。ReLU 与最终 cochain 的限制映射重算已验证。
- 参数量为 **72,148,029（72.1M）**，backbone 为 59,436,664。四个 RCL 的七个可学习融合 logits 共增加 28 个参数。
- 命令行完整检查通过：真实 Swin-Tiny、小通道配置与六条合成数据，训练一步、恢复至第二步、保存 format_version=3、推理导出及三种评估入口。输入/导出保持 96×128；这些只是接口样本。
- 双进程 Gloo 检查通过：两次优化均包含五项损失及两路 forward，可学习 Fuse 梯度非零，跨进程梯度同步。
- 图结构按当前输入重新生成，不从 checkpoint 恢复旧固定图。旧格式 1/2 显式拒绝。

测试命令：

```bash
python -m pytest -q
GLOO_SOCKET_IFNAME=lo0 python tests/check_distributed.py --output /tmp/lagrnet-ddp.json
```

lo0 适用于本机 macOS；Linux 可选择对应接口或省略此环境变量。Gloo 需要本地进程通信权限。

本轮没有完成 480K 次训练、四 H20/NCCL 验证、论文精度/消融复现或 A100/FLOPs 重测。CPU 前向耗时只是运行检查，不能代替论文硬件测量。现有实验数字与图文件未改。
