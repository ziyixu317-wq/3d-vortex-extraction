import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossSampling3D(nn.Module):
    """
    3D十字采样模块：对连续的3D空间局部交叠图块内的候选中心点进行十字采样。
    参考本地2D代码（如PyflowVis的PathlineSpatialSamplingLayer）中对空间的离散化处理逻辑。
    """
    def __init__(self, step_size=0.1):
        """
        初始化3D十字采样器。
        :param step_size: 采样步长 Delta。由于网格点通常会被归一化到[-1, 1]或[0, 1]，该步长需与坐标系对应。
        """
        super(CrossSampling3D, self).__init__()
        self.step_size = step_size
        # 三维空间中的标准正交基 e_d，额外采样6个对称点，加上本身共7个点
        self.register_buffer('offsets', torch.tensor([
            [0.0, 0.0, 0.0],    # 中心点 p
            [1.0, 0.0, 0.0],    # p + \Delta e_1
            [-1.0, 0.0, 0.0],   # p - \Delta e_1
            [0.0, 1.0, 0.0],    # p + \Delta e_2
            [0.0, -1.0, 0.0],   # p - \Delta e_2
            [0.0, 0.0, 1.0],    # p + \Delta e_3
            [0.0, 0.0, -1.0]    # p - \Delta e_3
        ]) * step_size) # Shape: (7, 3)

    def forward(self, center_points):
        """
        前向传播采样
        :param center_points: (B, N, 3) N个候选中心点 p=(x,y,z)
        :return: sampled_points: (B, N, 7, 3) 包含7个初始种子点的局部簇 S_p
        """
        # (B, N, 1, 3) + (1, 1, 7, 3) 广播机制相加 -> (B, N, 7, 3)
        sampled_points = center_points.unsqueeze(2) + self.offsets.view(1, 1, 7, 3)
        return sampled_points


class PathlineIntegration3D(nn.Module):
    """
    迹线计算模块（拉格朗日积分）：在时变速度场上进行显式数值积分。
    参考本地2D代码参数：集成长度 L 和步长 dt 应当对齐 2D 代码中时序采样的参数设定。
    """
    def __init__(self, L=16, dt=0.05, method='rk4'):
        """
        :param L: 离散时间序列迹线长度
        :param dt: 积分时间步长
        :param method: 数值积分方法，支持 'euler' 和 'rk4'
        """
        super(PathlineIntegration3D, self).__init__()
        self.L = L
        self.dt = dt
        self.method = method

    def get_velocity(self, velocity_field, positions, t_idx):
        """
        基于三维线性插值 (Trilinear Interpolation) 获取空间点在时间步 t_idx 的速度分量。
        :param velocity_field: (B, T, 3, D, H, W) 3D时变速度场
        :param positions: (B, N, 7, 3) 当前的迹线位置，需归一化至 [-1, 1] 才能使用 grid_sample
        :param t_idx: 当前所在的时间索引
        :return: 采样出的速度分量 (B, N, 7, 3)
        """
        B, N, K, _ = positions.shape
        # 防止越界，如果积分步长超过提供的速度场时间T，则取最后一个时间步
        t_idx_clamped = min(t_idx, velocity_field.shape[1] - 1)
        
        # 提取目标时间步的3D速度场: (B, 3, D, H, W)
        v_t = velocity_field[:, t_idx_clamped]
        
        # grid_sample 要求 grid 形状为 (B, D_out, H_out, W_out, 3)
        # 我们将 N*K 作为 1D 空间处理：(B, 1, 1, N*K, 3)
        grid = positions.view(B, 1, 1, N * K, 3)
        
        # 双线性(三线性)插值: (B, 3, 1, 1, N*K) -> 转置回 (B, N*K, 3) -> (B, N, K, 3)
        v_sampled = F.grid_sample(v_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
        v_sampled = v_sampled.squeeze(2).squeeze(2).transpose(1, 2).view(B, N, K, 3)
        return v_sampled

    def forward(self, seeds, velocity_field, start_t_idx=0):
        """
        :param seeds: (B, N, 7, 3) 经过3D十字采样得到的初始种子点
        :param velocity_field: (B, T, 3, D, H, W) 3D速度场张量
        :param start_t_idx: 积分的起始时间索引
        :return: pathlines (B, N, 7, L, 4) 包含四维信息 (x,y,z,t) 的迹线张量
        """
        B, N, K, _ = seeds.shape
        pathlines = []
        
        current_pos = seeds
        for l in range(self.L):
            t_idx = start_t_idx + l
            
            # 将物理时间或离散时间步记录为第4维特征
            current_time = torch.full((B, N, K, 1), float(t_idx * self.dt), device=seeds.device)
            # 组装四维张量 c_i(t_l) = (x_i(t_l), y_i(t_l), z_i(t_l), t_l)
            c_i = torch.cat([current_pos, current_time], dim=-1)
            pathlines.append(c_i)
            
            if l == self.L - 1:
                break
                
            if self.method == 'euler':
                # 显式欧拉法积分: p_{t+1} = p_t + V(p_t) * dt
                v = self.get_velocity(velocity_field, current_pos, t_idx)
                current_pos = current_pos + v * self.dt
            elif self.method == 'rk4':
                # 四阶龙格-库塔法 (RK4)，保证更好的物理拉格朗日客观性
                v1 = self.get_velocity(velocity_field, current_pos, t_idx)
                k1 = v1 * self.dt
                
                v2 = self.get_velocity(velocity_field, current_pos + 0.5 * k1, t_idx)
                k2 = v2 * self.dt
                
                v3 = self.get_velocity(velocity_field, current_pos + 0.5 * k2, t_idx)
                k3 = v3 * self.dt
                
                v4 = self.get_velocity(velocity_field, current_pos + k3, min(t_idx + 1, velocity_field.shape[1] - 1))
                k4 = v4 * self.dt
                
                current_pos = current_pos + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        # 将列表堆叠，得到最终迹线张量形状：(B, N, 7, L, 4)
        pathlines_tensor = torch.stack(pathlines, dim=3) 
        return pathlines_tensor


class Phase1_SamplingAndTrace(nn.Module):
    """
    阶段一：3D十字采样与迹线计算模块总封装
    """
    def __init__(self, step_size=0.1, L=16, dt=0.05, method='rk4'):
        super(Phase1_SamplingAndTrace, self).__init__()
        self.sampler = CrossSampling3D(step_size=step_size)
        self.integrator = PathlineIntegration3D(L=L, dt=dt, method=method)
        
    def forward(self, center_points, velocity_field, start_t_idx=0):
        """
        :param center_points: (B, N, 3) 归一化在[-1, 1]内的中心候选点
        :param velocity_field: (B, T, 3, D, H, W) 3D时变速度场
        :return: (B, N, 7, L, 4) 迹线特征序列
        """
        seeds = self.sampler(center_points)
        pathlines = self.integrator(seeds, velocity_field, start_t_idx)
        return pathlines

# ==========================================
# 测试代码示例 (当使用 python 运行此文件时执行)
# ==========================================
if __name__ == '__main__':
    print("=== 开始测试阶段一模块 (3D十字采样与迹线计算) ===")
    
    # 构造虚假的超参数
    B, T = 2, 20       # Batch size 和 时间步长
    D, H, W = 16, 16, 16 # 空间维度分辨率
    N = 100            # 每个batch测试的候选中心点数量
    L = 16             # 迹线长度
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 创建随机的 3D 速度场 (B, T, 3, D, H, W)
    velocity_field = torch.randn(B, T, 3, D, H, W, device=device)
    
    # 2. 随机生成 N 个中心候选点，坐标归一化在 [-1, 1] 内 (供 grid_sample 正常工作)
    center_points = torch.rand(B, N, 3, device=device) * 2.0 - 1.0
    
    # 3. 实例化网络模块 (参考本地2D默认参数风格)
    phase1_module = Phase1_SamplingAndTrace(step_size=0.05, L=L, dt=0.1, method='rk4').to(device)
    
    # 4. 执行前向传播
    print("正在执行拉格朗日积分 (RK4)...")
    pathlines = phase1_module(center_points, velocity_field, start_t_idx=0)
    
    # 5. 验证输出形状
    expected_shape = (B, N, 7, L, 4)
    print(f"输出形状: {pathlines.shape}")
    if pathlines.shape == expected_shape:
        print("✅ 形状匹配！测试通过。")
        print(f"样例迹线张量值 (第1批第1点第1分支的首个坐标+时间):\n {pathlines[0, 0, 0, 0]}")
    else:
        print(f"❌ 形状不匹配！预期: {expected_shape}, 实际: {pathlines.shape}")
