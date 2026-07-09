import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 导入我们的核心模块
from phase1_sampling_trace import Phase1_SamplingAndTrace, load_vti_series_uvw
from phase2_sge import SpatialGraphEmbedding3D
from phase3_4_transformer_head import VortexPredictor3D
from phase5_kaggle_pipeline import compute_vorticity_and_ivd, sample_features, compute_ivd_ground_truth

def train_kaggle():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/ziyixu317/halfcylinder3d-re640')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_seeds', type=int, default=1024)
    parser.add_argument('--dmodel', type=int, default=252)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Start Training on {device}...")

    # 1. 准备数据
    # 在 Kaggle 上，如果内存允许，可以一次性加载一段连续的帧作为训练集
    start_idx, end_idx = 0, 16 
    
    if not os.path.exists(args.data_dir):
        print(f"[警告] 找不到 {args.data_dir}，使用随机张量模拟训练运行...")
        B, T, D, H, W = 1, end_idx - start_idx, 80, 240, 640
        velocity_field = torch.randn(B, T, 3, D, H, W, device=device)
        bounds = (-2.0, 8.0, -2.0, 2.0, -2.0, 2.0)
    else:
        velocity_field, bounds = load_vti_series_uvw(args.data_dir, start_idx, end_idx)
        velocity_field = velocity_field.to(device)
        
    B, T, _, D, H, W = velocity_field.shape
    
    b_min = torch.tensor([bounds[0], bounds[2], bounds[4]], device=device)
    b_max = torch.tensor([bounds[1], bounds[3], bounds[5]], device=device)

    # 预计算底层特征与真实标签 (Ground Truth)
    vorticity, ivd = compute_vorticity_and_ivd(velocity_field)
    gt_mask_seq = compute_ivd_ground_truth(ivd)
    physical_features = torch.cat([velocity_field, vorticity, ivd], dim=2)
    
    # 极度重要：释放中间张量，直接节省高达 5.5 GB 显存！
    del velocity_field, vorticity, ivd
    torch.cuda.empty_cache()
    
    # 2. 初始化网络模型与优化器
    phase1 = Phase1_SamplingAndTrace(bounds_min=b_min.cpu().numpy(), bounds_max=b_max.cpu().numpy(), L=T, dt=0.05).to(device)
    phase2 = SpatialGraphEmbedding3D(in_channels=7, dmodel=args.dmodel).to(device)
    predictor = VortexPredictor3D(dmodel=args.dmodel).to(device)

    # 将需要训练的参数组合起来
    trainable_params = list(phase2.parameters()) + list(predictor.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    # 使用 BCEWithLogitsLoss (内部自带 Sigmoid，数值更稳定)
    criterion = nn.BCEWithLogitsLoss()

    # 3. 训练循环
    for epoch in range(args.epochs):
        phase2.train()
        predictor.train()
        optimizer.zero_grad()

        # 为了防止显存爆炸，每个 Epoch 我们在整个 3D 空间中随机采样 num_seeds 个点进行训练
        perm = torch.randperm(D * H * W, device=device)
        sampled_idx = perm[:args.num_seeds]
        
        # 构造密集网格的坐标 (供取 GT 和采样使用)
        z_c = torch.linspace(bounds[4], bounds[5], D, device=device)
        y_c = torch.linspace(bounds[2], bounds[3], H, device=device)
        x_c = torch.linspace(bounds[0], bounds[1], W, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(z_c, y_c, x_c, indexing='ij')
        full_pos = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(1, D * H * W, 3)

        center_points_phys = full_pos[:, sampled_idx, :] # (1, N, 3)

        # ---------------- 前向传播 ----------------
        # velocity_field 已经被清理，直接使用 physical_features 的前 3 个通道
        pathlines_phys = phase1(center_points_phys, physical_features[:, :, :3, ...])
        positions = pathlines_phys[..., :3]
        pathlines_features = sample_features(physical_features, positions, b_min, b_max)
        
        sge_features = phase2(pathlines_features, positions)
        center_positions = positions[:, :, 0, :, :]
        
        # 为了训练效率，我们不需要每次都将特征 propagate 到整个空间，直接对采样的 N 个点计算 Loss 即可！
        # logits: (1, N)
        logits = predictor(sge_features, center_positions) 
        
        # ---------------- 计算 Loss ----------------
        # 提取真实标签 (GT)
        # GT 张量拉平为 (D*H*W,)，并取出与采样点对应的标签
        gt_flat = gt_mask_seq[0, 0].view(D * H * W)
        gt_sampled = gt_flat[sampled_idx].unsqueeze(0) # (1, N)

        loss = criterion(logits, gt_sampled)
        
        # ---------------- 反向传播 ----------------
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")

    # 4. 训练结束，保存模型权重
    os.makedirs('/kaggle/working/weights', exist_ok=True)
    torch.save({
        'phase2_state_dict': phase2.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
    }, '/kaggle/working/weights/3d_vortex_model_latest.pth')
    
    print("模型训练完成并已保存至 /kaggle/working/weights/3d_vortex_model_latest.pth")

if __name__ == '__main__':
    train_kaggle()
