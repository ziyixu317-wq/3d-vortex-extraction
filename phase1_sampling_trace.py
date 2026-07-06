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
# 测试代码示例：如何使用自己的 .vti 文件进行测试
# ==========================================
import glob
import os
import argparse

def load_vti_series(file_directory, array_name=None):
    """
    读取一系列按照时间步命名的 .vti 文件，并将它们组装成 PyTorch 张量。
    :param file_directory: 存放 .vti 文件的文件夹路径
    :param array_name: 速度场对应的数组名称，如 'velocity', 'U' 等。
                       如果为 None，则默认读取第一个 Point Data 数组。
    :return: velocity_field_tensor (1, T, 3, D, H, W)
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        raise ImportError("未检测到 vtk 库，请使用 'pip install vtk' 进行安装")

    # 获取所有的 vti 文件，并按名称（通常按时间步命名的数字）排序
    vti_files = sorted(glob.glob(os.path.join(file_directory, '*.vti')))
    
    if not vti_files:
        raise FileNotFoundError(f"在 {file_directory} 中没有找到 .vti 文件！请检查路径。")
    
    print(f"找到 {len(vti_files)} 个 .vti 文件，正在加载...")
    tensors = []
    
    for f in vti_files:
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(f)
        reader.Update()
        image = reader.GetOutput()
        
        # VTI 图像维度顺序通常为 (nx, ny, nz) = (W, H, D)
        dims = image.GetDimensions()
        
        point_data = image.GetPointData()
        if array_name:
            array = point_data.GetArray(array_name)
        else:
            # 默认取第一个存在的数组（通常是速度uvw）
            array = point_data.GetArray(0)
            
        if array is None:
            raise ValueError(f"在文件 {f} 中没有找到指定的数组！请检查文件结构或指定正确的 array_name")
            
        # 转换为 numpy 数组
        numpy_array = vtk_to_numpy(array)
        
        # numpy数组展平时VTK通常是以 z, y, x (D, H, W) 为主序存储的
        # reshape为 (D, H, W, 3)，代表 (Z维度分辨率, Y维度分辨率, X维度分辨率, 3个速度分量)
        reshaped_array = numpy_array.reshape((dims[2], dims[1], dims[0], 3))
        
        # 转化为 PyTorch tensor，并将通道维度前置: (3, D, H, W)
        tensor = torch.from_numpy(reshaped_array).float().permute(3, 0, 1, 2)
        tensors.append(tensor)

    # 沿时间步维度 T 堆叠： (T, 3, D, H, W)
    velocity_sequence = torch.stack(tensors, dim=0)
    
    # 增加 Batch 维度： (1, T, 3, D, H, W)
    velocity_field = velocity_sequence.unsqueeze(0)
    print(f"加载完成！速度场张量维度为: {velocity_field.shape}")
    
    return velocity_field

def generate_patch_centers(batch_size, num_patches=100, device='cpu'):
    """
    模拟：将连续3D空间划分为局部交叠图块（Patches）并获取图块中心点坐标 p
    在真实场景中，这里的候选点是通过网格空间等间距划分(P x P x P)或根据流场特征预选得出的。
    由于 grid_sample 要求坐标范围在 [-1, 1]，此处的候选中心点坐标也需处于此范围。
    """
    # 这里我们随机生成均匀分布于 [-1, 1] 之间的点来模拟这 N=100 个图块的中心
    return torch.rand(batch_size, num_patches, 3, device=device) * 2.0 - 1.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D十字采样与迹线计算模块测试")
    parser.add_argument('--vti_dir', type=str, required=True, help='存放VTI文件的文件夹路径')
    parser.add_argument('--array_name', type=str, default=None, help='VTI文件中速度场数组的名称（可选）')
    parser.add_argument('--patches', type=int, default=100, help='划分的局部图块(中心点)数量 N')
    parser.add_argument('--L', type=int, default=16, help='迹线长度')
    args = parser.parse_args()

    print("=== 开始测试阶段一模块 (使用真实 .vti 数据) ===")
    try:
        # 1. 加载真实数据
        velocity_field = load_vti_series(args.vti_dir, array_name=args.array_name)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        velocity_field = velocity_field.to(device)

        # 2. 准备中心候选点 (对应于划分的 P x P x P 的局部图块)
        B, T, C, D, H, W = velocity_field.shape
        N = args.patches # 测试提取N个图块中心
        print(f"模拟将空间划分为 {N} 个交叠图块（Patches）...")
        center_points = generate_patch_centers(B, num_patches=N, device=device)

        # 3. 初始化并调用阶段一网络模块
        # 如果给定数据的时间步 T 小于需要的轨迹长度 L，则以实际 T 为准
        actual_L = min(args.L, T)
        print("初始化阶段一网络模块...")
        phase1_module = Phase1_SamplingAndTrace(step_size=0.05, L=actual_L, dt=0.1, method='rk4').to(device)
        
        print("正在进行真实数据的拉格朗日积分...")
        pathlines = phase1_module(center_points, velocity_field, start_t_idx=0)
        
        print(f"✅ 测试成功！最终的迹线张量维度: {pathlines.shape} (预期为 [1, {N}, 7, {actual_L}, 4])")
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
    except ImportError as e:
        print(f"❌ 错误: {e}")
    except Exception as e:
        print(f"❌ 未知错误: {e}")

