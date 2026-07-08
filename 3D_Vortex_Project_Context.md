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
- 成功修复了早期因“物理空间与 PyTorch `[-1,1]` 归一化空间尺度冲突”导致流线全变平行直线的核心计算 Bug。
- **Kaggle 真实流场跑通与 OOM 优化**：
  - 开发了高强健性的 VTI 解析器 (`load_vti_series_uvw`)，能够自动提取并合并分离存储的 `u, v, w` 数据。
  - 通过引入时间步切片机制 (`start_idx`, `end_idx`)，精准解决了全量加载导致 Kaggle OOM 内存爆满的问题。
  - 实现了基于 `image.GetBounds()` 的物理边界自适应读取。
  - 在真实的 `halfcylinder3d-re640` 数据集上成功可视化并验证了三维涡旋迹线的物理准确性。

## 3. 待完成任务 (Pending Tasks)
接下来的核心任务是顺着数据流向下游，完成 Transformer 架构的核心组件构建：

- **[ ] 阶段二 (Phase 2)：3D 空间图嵌入模块 (Spatial Graph Embedding, SGE)**
  - 根据 `phase1` 计算出的 3D 拉格朗日轨迹特征（位置坐标、标量场、局部形变等），设计序列嵌入映射机制，将时空轨迹编码为高维向量。
- **[ ] 阶段三 (Phase 3)：3D Transformer 时空特征交互编码模块**
  - 使用 Transformer Encoder 对嵌入后的轨迹数据进行 Multi-Head Attention 计算，捕获流体内部的 3D 拓扑演化规律及空间涡旋相关性。
- **[ ] 阶段四 (Phase 4)：涡特征预测与解码模块 (Prediction Head)**
  - 根据高维表征解码出涡旋区域的识别结果（分割或分类）。
- **[ ] 阶段五 (Phase 5)：整体流水线联调与数据集适配**
  - 构建完整的 PyTorch Dataset / DataLoader，将此前针对 VTI 测试用的工具链重构为可应对大规模网络训练的数据流读取格式，并统一全流程。

## 4. 关键开发规范 (Development Guidelines)
> [!WARNING]
> 为了防止 3D 拓展架构出现崩坏，新 AI 助手需注意：
> 1. 所有超参数必须对标原 2D 库对应位置。
> 2. 永远遵循**“物理坐标下完成轨迹演化积分，在送入深度网络前再使用边界进行归一化处理”**的数据隔离原则。
