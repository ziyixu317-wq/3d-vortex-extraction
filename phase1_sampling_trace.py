import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PathlineIntegration3D(nn.Module):
    """
    3D拉格朗日迹线积分模块 (基于真实物理坐标)
    """
    def __init__(self, bounds_min, bounds_max, L=16, dt=0.05, method='rk4'):
        super().__init__()
        self.L = L
        self.dt = dt
        self.method = method
        self.register_buffer('b_min', torch.tensor(bounds_min, dtype=torch.float32).view(1, 1, 1, 3))
        self.register_buffer('b_max', torch.tensor(bounds_max, dtype=torch.float32).view(1, 1, 1, 3))

    def get_velocity(self, velocity_field, phys_positions, t_idx):
        B, N, K, _ = phys_positions.shape
        t_idx_clamped = min(int(t_idx), velocity_field.shape[1] - 1)
        v_t = velocity_field[:, t_idx_clamped]
        
        # 将物理坐标映射到 [-1, 1] 供 PyTorch 查表
        norm_positions = 2.0 * (phys_positions - self.b_min) / (self.b_max - self.b_min) - 1.0
        
        grid = norm_positions.view(B, 1, 1, N * K, 3)
        # grid_sample 要求坐标是 (B, D, H, W, 3)
        v_sampled = F.grid_sample(v_t, grid, mode='bilinear', padding_mode='border', align_corners=True)
        v_sampled = v_sampled.squeeze(2).squeeze(2).transpose(1, 2).view(B, N, K, 3)
        return v_sampled

    def forward(self, seeds_phys, velocity_field, start_t_idx=0):
        B, N, K, _ = seeds_phys.shape
        pathlines = []
        current_pos = seeds_phys 
        
        for l in range(self.L):
            t_idx = start_t_idx + l
            current_time = torch.full((B, N, K, 1), float(t_idx * self.dt), device=seeds_phys.device)
            c_i = torch.cat([current_pos, current_time], dim=-1)
            pathlines.append(c_i)
            
            if l == self.L - 1:
                break
                
            if self.method == 'rk4':
                v1 = self.get_velocity(velocity_field, current_pos, t_idx)
                k1 = v1 * self.dt
                v2 = self.get_velocity(velocity_field, current_pos + 0.5 * k1, t_idx)
                k2 = v2 * self.dt
                v3 = self.get_velocity(velocity_field, current_pos + 0.5 * k2, t_idx)
                k3 = v3 * self.dt
                v4 = self.get_velocity(velocity_field, current_pos + k3, min(t_idx + 1, velocity_field.shape[1] - 1))
                k4 = v4 * self.dt
                current_pos = current_pos + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            elif self.method == 'euler':
                v = self.get_velocity(velocity_field, current_pos, t_idx)
                current_pos = current_pos + v * self.dt

        return torch.stack(pathlines, dim=3)


class Phase1_SamplingAndTrace(nn.Module):
    """
    阶段一：十字采样 + 拉格朗日迹线计算 (真实物理坐标框架)
    """
    def __init__(self, bounds_min, bounds_max, step_size_phys=0.05, L=16, dt=0.05, method='rk4'):
        super().__init__()
        self.integrator = PathlineIntegration3D(bounds_min, bounds_max, L, dt, method)
        # 十字采样器：工作在真实的物理坐标系下
        self.register_buffer('offsets', torch.tensor([
            [0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ], dtype=torch.float32) * step_size_phys) 

    def forward(self, center_points_phys, velocity_field, start_t_idx=0):
        seeds_phys = center_points_phys.unsqueeze(2) + self.offsets.view(1, 1, 7, 3)
        pathlines_phys = self.integrator(seeds_phys, velocity_field, start_t_idx)
        return pathlines_phys


# =======================================================
# 辅助函数：针对 u, v, w 分离存储的 VTI 文件的读取函数
# =======================================================
def load_vti_series_uvw(file_directory, start_idx=0, end_idx=20):
    import glob
    import os
    import numpy as np
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        raise ImportError("请安装 vtk 库：pip install vtk")
        
    all_files = sorted(glob.glob(os.path.join(file_directory, '*.vti')))
    vti_files = all_files[start_idx:end_idx]
    if not vti_files:
        raise FileNotFoundError(f"在 {file_directory} 中没有找到 .vti 文件！")
    print(f"找到 {len(vti_files)} 个 .vti 文件，正在合并 u, v, w 分量...")
    
    tensors = []
    bounds = None
    for i, f in enumerate(vti_files):
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(f)
        reader.Update()
        image = reader.GetOutput()
        
        if i == 0:
            bounds = image.GetBounds() # (xmin, xmax, ymin, ymax, zmin, zmax)

        dims = image.GetDimensions()
        point_data = image.GetPointData()
        
        u_arr = vtk_to_numpy(point_data.GetArray('u'))
        v_arr = vtk_to_numpy(point_data.GetArray('v'))
        w_arr = vtk_to_numpy(point_data.GetArray('w'))
        
        vector_arr = np.stack([u_arr, v_arr, w_arr], axis=-1)
        reshaped_array = vector_arr.reshape((dims[2], dims[1], dims[0], 3))
        
        tensor = torch.from_numpy(reshaped_array).float().permute(3, 0, 1, 2)
        tensors.append(tensor)

    velocity_field = torch.stack(tensors, dim=0).unsqueeze(0)
    print(f"加载完成！速度场维度: {velocity_field.shape}")
    return velocity_field, bounds


if __name__ == '__main__':
    # 简单的本地随机数据测试，保证代码能运行
    print("=== 本地测试阶段一模块 (物理坐标框架) ===")
    
    # 模拟真实物理边界 (X: -2~8, Y: -2~2, Z: -2~2)
    b_min = [-2.0, -2.0, -2.0]
    b_max = [8.0, 2.0, 2.0]
    
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟生成随机速度场数据 (Batch=1, T=16, C=3, Z=20, Y=40, X=100)
    B, T, C, D, H, W = 1, 16, 3, 20, 40, 100
    velocity_field = torch.randn(B, T, C, D, H, W, device=device) * 0.5 + 1.0 # 主流速度给 1.0
    
    # 模拟撒点
    N = 10
    center_points_phys = torch.rand(B, N, 3, device=device)
    center_points_phys[..., 0] = center_points_phys[..., 0] * 10.0 - 2.0 # -2 ~ 8
    center_points_phys[..., 1] = center_points_phys[..., 1] * 4.0 - 2.0  # -2 ~ 2
    center_points_phys[..., 2] = center_points_phys[..., 2] * 4.0 - 2.0  # -2 ~ 2
    
    # 初始化网络
    model = Phase1_SamplingAndTrace(
        bounds_min=b_min, 
        bounds_max=b_max, 
        step_size_phys=0.1, 
        L=16, 
        dt=0.1
    ).to(device)
    
    print("运行拉格朗日积分...")
    pathlines = model(center_points_phys, velocity_field)
    print(f"迹线输出形状: {pathlines.shape} (预期: B, N, 7, L, 4)")
    print("测试通过！可以将其上传至 Kaggle 进行真实流场测试。")
