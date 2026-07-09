import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 导入前四个阶段的模块
from phase1_sampling_trace import Phase1_SamplingAndTrace, load_vti_series_uvw
from phase2_sge import SpatialGraphEmbedding3D
from phase3_4_transformer_head import VortexPredictor3D

def compute_vorticity_and_ivd(velocity_field):
    """
    计算速度场的涡量和瞬时涡量偏差 (IVD)
    velocity_field: (B, T, C, D, H, W)，其中 C=3 为 (u, v, w)
    
    返回:
    vorticity: (B, T, 3, D, H, W)
    ivd: (B, T, 1, D, H, W)
    """
    B, T, C, D, H, W = velocity_field.shape
    assert C == 3, "Velocity field must have 3 channels (u, v, w)"
    
    # 获取各个分量
    u = velocity_field[:, :, 0, ...]
    v = velocity_field[:, :, 1, ...]
    w = velocity_field[:, :, 2, ...]
    
    # dim: 4->W(x), 3->H(y), 2->D(z)
    du_dx, du_dy, du_dz = torch.gradient(u, dim=(4, 3, 2))
    dv_dx, dv_dy, dv_dz = torch.gradient(v, dim=(4, 3, 2))
    dw_dx, dw_dy, dw_dz = torch.gradient(w, dim=(4, 3, 2))
    
    # 计算涡量 omega = curl(V)
    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy
    vorticity = torch.stack([omega_x, omega_y, omega_z], dim=2)  # (B, T, 3, D, H, W)
    
    # 计算 IVD: 瞬时涡量偏差
    # 此处使用涡量模长减去其空间均值作为 IVD 的一种近似表示
    omega_mag = torch.norm(vorticity, dim=2, keepdim=True)  # (B, T, 1, D, H, W)
    spatial_mean = omega_mag.mean(dim=(3, 4, 5), keepdim=True)
    ivd = omega_mag - spatial_mean  # (B, T, 1, D, H, W)
    
    return vorticity, ivd

def sample_features(physical_features, phys_positions, b_min, b_max):
    """
    基于坐标从网格特征张量中采样特征
    physical_features: (B, T, C, D, H, W)
    phys_positions: (B, N, K, L, 3) 此处的 L 相当于时间步
    """
    B, N, K, L, _ = phys_positions.shape
    _, T, C, _, _, _ = physical_features.shape
    
    # 将 phys_positions 映射到 [-1, 1] 供 grid_sample 使用
    # b_min, b_max 需要匹配张量所在设备
    norm_positions = 2.0 * (phys_positions - b_min) / (b_max - b_min) - 1.0
    
    # phys_positions (B, N, K, L, 3) 按照时间拆分
    sampled_features = []
    for l in range(L):
        pos_t = norm_positions[:, :, :, l, :]  # (B, N, K, 3)
        grid = pos_t.view(B, 1, 1, N * K, 3)
        
        # 对应时间步的特征
        feat_t = physical_features[:, l, ...]  # (B, C, D, H, W)
        
        feat_sampled = torch.nn.functional.grid_sample(
            feat_t, grid, mode='bilinear', padding_mode='border', align_corners=True
        ) # (B, C, 1, 1, N*K)
        
        feat_sampled = feat_sampled.squeeze(2).squeeze(2).transpose(1, 2).view(B, N, K, C)
        sampled_features.append(feat_sampled)
        
    return torch.stack(sampled_features, dim=3)  # (B, N, K, L, C)

def visualize_2d_like(positions, predictions, save_path="vortex_predictions.png"):
    """
    提供类似 2D Vortex Transformer 的降维/切片散点可视化
    """
    pos_np = positions.detach().cpu().numpy()[0]   # (N, 3)
    pred_np = predictions.detach().cpu().numpy()[0] # (N,)
    
    fig = plt.figure(figsize=(10, 8))
    
    # 3D 散点图
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], c=pred_np, cmap='jet', alpha=0.8, s=20)
    plt.colorbar(sc, label='Vortex Probability')
    ax.set_title("3D Vortex Prediction Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存至 {save_path}")

def compute_ivd_ground_truth(ivd_field):
    """
    根据瞬时涡量偏差 (IVD) 准则计算真实涡流标签 (Ground Truth)
    ivd_field: (B, T, 1, D, H, W)
    返回:
    gt_mask: (B, T, D, H, W) 二值化掩码，1 表示涡流，0 表示非涡流
    """
    # 采用 IVD > 0 (即涡量大于空间平均值) 作为涡旋结构的二分类真实标签
    gt_mask = (ivd_field > 0).float().squeeze(2)
    return gt_mask

def visualize_3d_mask(prob_grid, gt_mask, save_path="kaggle_pipeline_visualization.png"):
    """
    可视化 3D 预测掩码与 Ground Truth
    """
    prob_np = prob_grid.detach().cpu().numpy()
    gt_np = gt_mask.detach().cpu().numpy()
    
    # 为了散点可视化，只提取概率较高的点或真实的涡点
    z_pred, y_pred, x_pred = np.where(prob_np > 0.5)
    z_gt, y_gt, x_gt = np.where(gt_np == 1.0)
    
    fig = plt.figure(figsize=(16, 8))
    
    ax1 = fig.add_subplot(121, projection='3d')
    if len(x_pred) > 0:
        ax1.scatter(x_pred, y_pred, z_pred, c='r', alpha=0.5, s=5, label='Predicted Vortex')
    ax1.set_title("Predicted 3D Vortex Mask (>0.5)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    ax2 = fig.add_subplot(122, projection='3d')
    if len(x_gt) > 0:
        ax2.scatter(x_gt, y_gt, z_gt, c='b', alpha=0.5, s=5, label='Ground Truth (IVD>0)')
    ax2.set_title("Ground Truth Vortex Mask (IVD Criterion)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"密集流场 3D 掩码可视化结果已保存至 {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Kaggle 真实数据集联调验证脚本")
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/ziyixu317/halfcylinder3d-re640',
                        help='VTI 文件所在的目录路径')
    parser.add_argument('--start_idx', type=int, default=0, help='起始时间步')
    parser.add_argument('--end_idx', type=int, default=16, help='结束时间步 (Kaggle 防 OOM，建议小范围)')
    parser.add_argument('--num_seeds', type=int, default=2048, help='Patch 内部采样的种子点数量 N')
    parser.add_argument('--dmodel', type=int, default=252, help='Transformer 隐层维度')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的计算设备: {device}")
    
    # 2. 数据加载与初始特征构造
    if not os.path.exists(args.data_dir):
        print(f"[警告] 找不到路径 {args.data_dir}，正在使用随机数据模拟联调 (确保代码逻辑畅通)...")
        # 实际数据集维度: x=640, y=240, z=80. 在 PyTorch 中对应 D=80, H=240, W=640
        B, T, D, H, W = 1, args.end_idx - args.start_idx, 80, 240, 640
        # 如果内存不足，可以在本地调小此处模拟的维度
        velocity_field = torch.randn(B, T, 3, D, H, W, device=device)
        bounds = (-2.0, 8.0, -2.0, 2.0, -2.0, 2.0)
    else:
        velocity_field, bounds = load_vti_series_uvw(args.data_dir, args.start_idx, args.end_idx)
        velocity_field = velocity_field.to(device)
        B, T, _, D, H, W = velocity_field.shape
        
    b_min = torch.tensor([bounds[0], bounds[2], bounds[4]], device=device)
    b_max = torch.tensor([bounds[1], bounds[3], bounds[5]], device=device)
    
    print("正在计算速度场的涡量和 IVD 特征...")
    vorticity, ivd = compute_vorticity_and_ivd(velocity_field)
    
    # 获取 IVD 真实标签
    gt_mask_seq = compute_ivd_ground_truth(ivd)  # (B, T, D, H, W)
    gt_mask_t0 = gt_mask_seq[0, 0, ...]          # (D, H, W) 仅取当前起始时间步做参考
    
    # 组合底层物理场特征 (B, T, 7, D, H, W)
    physical_features = torch.cat([velocity_field, vorticity, ivd], dim=2)
    print(f"融合后物理场特征维度: {physical_features.shape}")
    
    # 3. 生成全流场密集网格，并从中抽取 N 个采样点作为追踪种子
    # Meshgrid: X, Y, Z (注意 PyTorch meshgrid 对应的维度顺序)
    print(f"生成全局密集网格 ({D}x{H}x{W}) 并采样 {args.num_seeds} 个迹线种子点...")
    z_coords = torch.linspace(bounds[4], bounds[5], D, device=device)
    y_coords = torch.linspace(bounds[2], bounds[3], H, device=device)
    x_coords = torch.linspace(bounds[0], bounds[1], W, device=device)
    
    # 顺序需匹配 D(z), H(y), W(x)
    grid_z, grid_y, grid_x = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    # 组装为 (1, D*H*W, 3) 坐标
    full_pos = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(1, D * H * W, 3)
    
    perm = torch.randperm(D * H * W, device=device)
    sampled_idx = perm[:args.num_seeds]
    center_points_phys = full_pos[:, sampled_idx, :]  # (1, N, 3)
    
    # --- 阶段一：十字采样与演化 ---
    print("-> 运行阶段一 (Phase 1): 十字采样与拉格朗日迹线积分")
    L_steps = physical_features.shape[1]
    phase1 = Phase1_SamplingAndTrace(
        bounds_min=b_min.cpu().numpy(), bounds_max=b_max.cpu().numpy(), 
        L=L_steps, dt=0.05
    ).to(device)
    
    pathlines_phys = phase1(center_points_phys, velocity_field)
    positions = pathlines_phys[..., :3]  # (1, N, K, L, 3)
    
    print("-> 获取演化迹线上的物理特征")
    pathlines_features = sample_features(physical_features, positions, b_min, b_max)
    
    # --- 阶段二：图嵌入 ---
    print("-> 运行阶段二 (Phase 2): 空间图嵌入")
    phase2 = SpatialGraphEmbedding3D(in_channels=7, dmodel=args.dmodel).to(device)
    sge_features = phase2(pathlines_features, positions)
    
    # --- 阶段三/四：时序 Transformer 与预测 ---
    print("-> 运行阶段三/四 (Phase 3 & 4): 时空演化重组、特征上采样与最终二值化掩码预测")
    center_positions = positions[:, :, 0, :, :]  # (1, N, L, 3)
    
    predictor = VortexPredictor3D(dmodel=args.dmodel).to(device)
    
    # 输入 full_pos 让网络自动执行 propagate_features 到整个 D*H*W 网格
    logits_full = predictor(sge_features, center_positions, full_pos=full_pos) # (1, D*H*W)
    
    # 对输出 Logits 施加 Sigmoid 得到概率
    prob_full = torch.sigmoid(logits_full)
    
    # 重塑拼接回 3D 张量
    prob_grid = prob_full.view(D, H, W)
    print(f"预测与空间拼接完成！输出 3D 掩码维度: {prob_grid.shape}")
    
    # 4. 可视化呈现：拼接后的概率分布 vs IVD Ground Truth
    save_path = "kaggle_pipeline_visualization.png"
    visualize_3d_mask(prob_grid, gt_mask_t0, save_path)
    print("整体代码流水线验证成功，特征反投影与完整掩码流场拼接顺利执行完毕！")

if __name__ == '__main__':
    main()
