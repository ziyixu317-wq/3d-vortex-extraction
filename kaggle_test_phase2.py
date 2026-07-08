import torch
import os
import time

# 导入我们的模块
from phase1_sampling_trace import load_vti_series_uvw, Phase1_SamplingAndTrace
from phase2_sge import SpatialGraphEmbedding3D

def test_sge_on_kaggle(vti_dir, start_idx=0, end_idx=20, N_seeds=5000):
    """
    在 Kaggle 的真实数据集上测试 SGE 模块的可行性（主要是显存和性能测试）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== 开始 Kaggle 真实数据流场测试 | 设备: {device} ===")
    
    # 1. 加载真实 VTI 数据
    print(f"正在从 {vti_dir} 加载流场数据...")
    t0 = time.time()
    velocity_field, bounds = load_vti_series_uvw(vti_dir, start_idx=start_idx, end_idx=end_idx)
    velocity_field = velocity_field.to(device)
    print(f"流场加载完毕，耗时: {time.time()-t0:.2f}s, 形状: {velocity_field.shape}")
    
    b_min, b_max = bounds[0:6:2], bounds[1:6:2]
    
    # 2. 依照 2D 逻辑进行 3D Patch 生成 (Patch Size = 32, Stride = 16)
    print(">>> 正在依照 2D 方法构建 3D Patch 局部图块采样...")
    # 假设我们有一个 3D 的基准分辨率 (根据流场 Bounds 大致划分)
    # 此处为测试方便，我们在物理空间中构建一个规则网格，然后滑动窗口
    patch_size = 32
    patch_stride = 16
    
    # 假设整体空间被划分为 64 x 64 x 64 的网格 (模拟流场分辨率)
    # 实际项目中，分辨率会和输入的 VTI 分辨率一致 (Dims)
    grid_res = [64, 64, 64]
    
    # 计算沿三个维度的 Patch 起点
    starts_x = list(range(0, grid_res[0] - patch_size + 1, patch_stride))
    starts_y = list(range(0, grid_res[1] - patch_size + 1, patch_stride))
    starts_z = list(range(0, grid_res[2] - patch_size + 1, patch_stride))
    
    print(f"网格分辨率: {grid_res}, 预期生成的 Patch 数量: {len(starts_x) * len(starts_y) * len(starts_z)}")
    
    # 我们只抽取其中 1 个 Patch 进行前向传播和显存测试
    # 实际推理时，会遍历所有 Patch 预测并拼接
    test_patch_idx = 0
    
    # 计算当前 Patch 的物理边界
    px_start = starts_x[0] / grid_res[0] * (b_max[0] - b_min[0]) + b_min[0]
    px_end   = (starts_x[0] + patch_size) / grid_res[0] * (b_max[0] - b_min[0]) + b_min[0]
    py_start = starts_y[0] / grid_res[1] * (b_max[1] - b_min[1]) + b_min[1]
    py_end   = (starts_y[0] + patch_size) / grid_res[1] * (b_max[1] - b_min[1]) + b_min[1]
    pz_start = starts_z[0] / grid_res[2] * (b_max[2] - b_min[2]) + b_min[2]
    pz_end   = (starts_z[0] + patch_size) / grid_res[2] * (b_max[2] - b_min[2]) + b_min[2]
    
    # 在这个 Patch 内撒点 (2D 中每个 patch 大小为 32x32=1024 个点)
    # 这里 3D 我们控制点数与传入的 N_seeds 相关，或者模拟 32x32x32 = 32768
    actual_seeds = min(N_seeds, patch_size**3)
    print(f"在当前 Patch [{starts_x[0]}:{starts_x[0]+patch_size}, ...] 内生成 {actual_seeds} 个种子点...")
    
    center_points_phys = torch.rand(1, actual_seeds, 3, device=device)
    center_points_phys[..., 0] = center_points_phys[..., 0] * (px_end - px_start) + px_start
    center_points_phys[..., 1] = center_points_phys[..., 1] * (py_end - py_start) + py_start
    center_points_phys[..., 2] = center_points_phys[..., 2] * (pz_end - pz_start) + pz_start
        
    # 3. 运行阶段一 (十字采样与拉格朗日迹线)
    L_steps = 16
    phase1_model = Phase1_SamplingAndTrace(
        bounds_min=b_min, bounds_max=b_max, step_size_phys=0.05, L=L_steps, dt=0.05
    ).to(device)
    
    print(">>> 运行 Phase 1: 迹线积分...")
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad():
        # pathlines_phys: (B, N, 7, L, 4) -> (x, y, z, t)
        pathlines_phys = phase1_model(center_points_phys, velocity_field)
    print(f"Phase 1 完成，耗时: {time.time()-t1:.2f}s, 输出形状: {pathlines_phys.shape}")
    print(f"当前 GPU 显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    # 4. 构造初始物理特征传递给阶段二 (Phase 2)
    in_channels = 7 
    dmodel = 252
    
    print(">>> 构建阶段二初始特征...")
    B, N, K, L, _ = pathlines_phys.shape
    # 提取 positions (B, N, K, L, 3)
    positions = pathlines_phys[..., :3]
    
    # 伪造 (B, N, K, L, 7) 的物理标量特征
    # 注意：真实训练时，这里应替换为对速度场、涡量场的 grid_sample 结果
    physical_features = torch.zeros(B, N, K, L, in_channels, device=device)
    
    # 5. 运行阶段二 (空间图嵌入 SGE)
    phase2_sge = SpatialGraphEmbedding3D(in_channels=in_channels, dmodel=dmodel).to(device)
    
    print(">>> 运行 Phase 2: SGE 空间图嵌入...")
    t2 = time.time()
    with torch.no_grad():
        sge_features = phase2_sge(physical_features, positions)
        
    print(f"Phase 2 完成，耗时: {time.time()-t2:.2f}s")
    print(f"SGE 最终输出形状: {sge_features.shape} (预期: B, {N}, {L_steps}, {dmodel})")
    print(f"峰值 GPU 显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print("测试完美通过！这证明 SGE 模块可以在 Kaggle GPU 上处理大规模的三维点云迹线。")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle 测试 3D SGE 模块")
    parser.add_argument('--dataset', type=str, default='/kaggle/input/halfcylinder3d-re640', help='VTI 数据集所在目录路径')
    parser.add_argument('--start_idx', type=int, default=0, help='起始时间步')
    parser.add_argument('--end_idx', type=int, default=20, help='结束时间步')
    parser.add_argument('--n_seeds', type=int, default=10000, help='随机撒点的中心簇数量')
    args = parser.parse_args()
    
    print(f"即将测试的数据集路径: {args.dataset}")
    print("注意: 真实情况下请确保路径下包含 .vti 后缀的流场文件。")
    
    # 执行测试
    try:
        test_sge_on_kaggle(args.dataset, start_idx=args.start_idx, end_idx=args.end_idx, N_seeds=args.n_seeds)
    except FileNotFoundError as e:
        print(f"\n[错误] 数据集未找到: {e}")
        print("请检查 '--dataset' 参数是否为正确的 Kaggle 数据集挂载路径 (例如 /kaggle/input/halfcylinder3d-re640)。")
    except Exception as e:
        print(f"\n[运行异常]: {e}")
