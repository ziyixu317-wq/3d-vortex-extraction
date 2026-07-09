# 3D Vortex Transformer 项目开发上下文 (Context)

> [!IMPORTANT]
> 此文档保存了关于“3D非定常流场涡特征智能化提取”代码复现项目的核心上下文信息。在新对话框中加载或向 AI 提供本文档，即可无缝继续接下来的开发工作。

## 1. 背景与目标 (Background & Objective)
- **课题**：基于拉格朗日演化轨迹的泛化结构提取框架。
- **目标**：将现有的 2D Vortex Transformer 架构扩展至 3D 涡结构提取，并使用 PyTorch 落地实现。
- **项目路径**：`C:\Users\徐子屹\Desktop\antigravity\3dvortex transformer`
- **参考资料**：
  - 本地 2D 参考代码仓库：`C:\Users\徐子屹\Desktop\antigravity\PyflowVis-main`
  - 3D 涡结构提取方案："C:\Users\徐子屹\Desktop\研究方案.pdf"
  - 必须**强制参考**本地 2D 代码的参数及架构连贯性（包括网络层数、隐藏层维度、物理计算逻辑等）。

## 2. 已完成工作 (Completed Tasks)
**[x] 阶段一 (Phase 1)：3D 十字采样与拉格朗日迹线计算模块 (`phase1_sampling_trace.py`)**
- 构建了基于真实物理坐标系的拉格朗日积分网络（RK4 / Euler）。
- 成功修复了早期因物理尺度冲突导致流线变平行的核心 Bug。
- 开发了 `load_vti_series_uvw` 实现高强健性的 VTI 解析。

**[x] 阶段二 (Phase 2)：3D 空间图嵌入模块 (`phase2_sge.py`)**
- 实现了 `SpatialGraphEmbedding3D` 核心模块。
- 确认了 `dmodel = 252` 隐层维度，与原 2D 架构完全对齐。

**[x] 阶段三/四 (Phase 3 & 4)：3D Transformer 时空编码与解码模块 (`phase3_4_transformer_head.py`)**
- 严格对齐 2D 的 Transformer 结构，将注意力约束在单条迹线的 $L$ 个时间步。
- 实现了针对 3D 大规模数据的 Chunking 分块 `propagate_features`。
- 将特征重构解码器严格对齐为单层 `Linear(dmodel, 1)`，并成功适配 BCE 损失。

**[x] 阶段五 (Phase 5)：整体流水线联调与数据集适配 (`train_kaggle.py`, `phase5_kaggle_pipeline.py`)**
- `train_kaggle.py`：构建了内存安全且收敛正常的端到端训练脚本，实时利用 `IVD > 0` 作为 Ground Truth，同时做到了高效的显存释放策略。
- `phase5_kaggle_pipeline.py`：升级为专用推理+导出脚本，能够自动读取 `.pth` 权重，使用 `@torch.no_grad()` 和 VRAM 释放技术，并在最后利用 `vtkImageData` 输出包含真实掩码与预测掩码的 `prediction_result_t0.vti` 以供 ParaView 渲染。

## 3. 当前遇到的瓶颈与 OOM 问题剖析 (Current Issue: Inference OOM)

> [!WARNING]
> 我们在 Kaggle 真实 $640 \times 240 \times 80$ 流场进行端到端推理时，再次遭遇 OOM 显存溢出：`Tried to allocate 80.00 MiB ... of which 40.81 MiB is free`。

### 3.1 内存爆炸根源分析
即便在极小分块 (`chunk_size=20000`) 的情况下也发生了 OOM，这是由于目前特征插值的**逻辑顺序**引发了恐怖的显存堆积：
1. `predictor` 中的 Transformer 输出稀疏点的特征 `x_pooled` (形状为 `B, N, 252`)。
2. 当前代码中，我们把 `252` 维的深度特征，通过 `propagate_features` 直接插值上采样到了整个 $12.28$ 百万网格中 (`B, M=12288000, 252`)。
3. 一个包含 **一千两百多万** 个体素，每个体素拥有 **252** 个 Float32 特征的张量，其体积计算如下：
   $$ 12,288,000 \times 252 \times 4 \text{ 字节} = \mathbf{12.38 \text{ GB}} $$
4. 这个骇人的 12.38GB 巨型张量，加上底层速度场与真值等，瞬间撑爆了 Kaggle 的 15GB 显存，导致无法进行后续的 `Linear(dmodel, 1)`。

### 3.2 解决策略 (Solution for the next Agent)
要彻底解决这个问题并保持物理意义不变，我们需要在 `phase3_4_transformer_head.py` 中**颠倒特征解码与插值上采样的顺序**：

- **第一步 (FC Decoding)**：直接让全连接层 `self.fc` 将 Transformer 输出的稀疏特征 `x_pooled` (维度 `(B, N, 252)`) 映射为标量 Logits。
  - 即执行：`logits_sparse = self.fc(x_pooled)`。此时形状降为 `(B, N, 1)`。
- **第二步 (Scalar Propagation)**：使用 `propagate_features` 将这 `(B, N, 1)` 的标量掩码插值上采样回全流场的 12.28 百万个网格点！
  - 上采样后的结果形状为 `(B, M=12288000, 1)`，该张量的内存占用将骤减为：
  $$ 12,288,000 \times 1 \times 4 \text{ 字节} = \mathbf{49 \text{ MB}} $$
- **第三步**：如果 `fc` 后原本还有 `self.feature_propagation` 操作，请将其移到上采样之前（对 N 个点操作）以保持数学等效性，且运算极度轻量化。

**通过这个小小的次序颠倒，我们可以将 12.3 GB 的显存需求直接抹除至不足 50 MB，这将成为 3D 巨大流场预测的终极破局之道！**

## 4. 关键开发规范 (Development Guidelines)
> [!WARNING]
> 为了防止 3D 拓展架构出现崩坏，新 AI 助手需注意：
> 1. 所有超参数必须对标原 2D 库对应位置。
> 2. 永远遵循**“物理坐标下完成轨迹演化积分，在送入深度网络前再使用边界进行归一化处理”**的数据隔离原则。
> 3. **防 OOM 规范**：严禁沿用 2D 的全网格生成再采样机制；必须确保使用 Chunking 分块矩阵乘法计算距离；严禁将高维 `dmodel` 特征直接插值到全流场，应先解码为单通道标量 Logits 再上采样。
> 4. **空间局部簇拓扑**：3D 迹线局部簇必须严格遵循 7 点十字拓展原则。
> 5. 每次输出完，检查一下文件夹下所有代码的一致性。
