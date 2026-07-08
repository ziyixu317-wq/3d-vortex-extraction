import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGraphEmbedding3D(nn.Module):
    r"""
    3D 空间图嵌入模块 (Spatial Graph Embedding, SGE)
    
    用于将阶段一 (Phase 1) 提取出的局部时空拉格朗日轨迹族特征，
    进行邻域特征聚合（类似于 Dynamic EdgeConv），并编码为高维向量。
    """
    def __init__(self, in_channels, dmodel=252):
        super().__init__()
        self.in_channels = in_channels
        self.dmodel = dmodel
        
        # 依据 PDF 公式(8)：h_ij = MLP(Concat(h_i, h_j - h_i, \Delta p_ij))
        # Concat 的通道数 = h_i (in_channels) + (h_j - h_i) (in_channels) + \Delta p_ij (3)
        concat_dim = in_channels * 2 + 3
        
        # 使用 MLP 映射到高维特征 dmodel
        # 考虑到输入可能是 (B, N, K, L, C) 的 5D 张量，为了方便使用 Conv2d / Conv3d 或者 Linear
        # 这里使用 nn.Sequential 包装的 Linear 层。
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, dmodel // 2),
            nn.BatchNorm1d(dmodel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dmodel // 2, dmodel),
            nn.BatchNorm1d(dmodel),
            nn.ReLU(inplace=True)
        )

    def forward(self, pathlines, positions):
        r"""
        前向传播计算
        
        Args:
            pathlines: (B, N, K, L, in_channels)
                       高维特征张量，K=7 为局部十字采样大小，L为序列长度。
                       特征内容通常包含速度、涡量、瞬时涡量偏差(IVD)等拼接结果。
            positions: (B, N, K, L, 3) 
                       物理坐标，用于计算 \Delta p_ij。
                       
        Returns:
            sge_features: (B, N, L, dmodel)
        """
        B, N, K, L, C = pathlines.shape
        _, _, _, _, pos_dim = positions.shape
        assert pos_dim == 3, "positions 必须包含 3 个空间坐标维度"
        assert C == self.in_channels, f"特征通道数不匹配: 期望 {self.in_channels}, 实际 {C}"
        
        # 取出中心点 (索引 0 为中心，1~6 为十字方向偏移邻居)
        # 形状: (B, N, 1, L, C)
        h_i = pathlines[:, :, 0:1, :, :]
        p_i = positions[:, :, 0:1, :, :]
        
        # 取出邻域点 (为了实现最大池化聚合，直接包含全体 7 个点)
        # 形状: (B, N, K, L, C)
        h_j = pathlines
        p_j = positions
        
        # 计算相对特征
        h_i_expanded = h_i.expand(-1, -1, K, -1, -1)  # (B, N, K, L, C)
        p_i_expanded = p_i.expand(-1, -1, K, -1, -1)  # (B, N, K, L, 3)
        
        h_diff = h_j - h_i_expanded                   # h_j - h_i
        p_diff = p_j - p_i_expanded                   # \Delta p_ij
        
        # 特征拼接: Concat(h_i, h_j - h_i, \Delta p_ij)
        # 形状: (B, N, K, L, 2*C + 3)
        concat_features = torch.cat([h_i_expanded, h_diff, p_diff], dim=-1)
        
        # 使用 MLP 提取特征
        # Linear 要求输入在最后一维，(B, N, K, L, concat_dim) -> (B, N, K, L, dmodel)
        # 由于我们包含 BatchNorm1d，它期望输入是 (N_batch, channels, ...)，所以需要 reshape 或直接变换。
        
        concat_dim = concat_features.shape[-1]
        x = concat_features.view(B * N * K * L, concat_dim)
        
        # 手动过 MLP，因为 BatchNorm1d 处理 2D (Batch, Channels) 的效果等同于 1D (Batch*Len, Channels)
        for layer in self.mlp:
            x = layer(x)
            
        x = x.view(B, N, K, L, self.dmodel)
        
        # 最大池化聚合邻域特征：h_i^{SGE} = \max_{j \in N(i)} (h_ij)
        # 沿 K 的维度进行 max (dim=2)
        sge_features, _ = torch.max(x, dim=2)  # 输出形状: (B, N, L, dmodel)
        
        return sge_features

if __name__ == '__main__':
    print("=== 测试 Phase 2: SGE 模块 ===")
    
    # 模拟输入参数
    B = 2       # Batch size
    N = 100     # 采样的中心迹线族数量
    K = 7       # 十字采样簇大小 (1中心+6邻居)
    L = 16      # 时序长度
    
    # 根据用户设定，特征为 速度(3) + 涡量(3) + 瞬时涡量偏差IVD(1) = 7维
    in_channels = 7 
    dmodel = 252    # 对齐 2D 代码的隐层维度
    
    # 构造假数据
    pathlines_features = torch.randn(B, N, K, L, in_channels)
    positions = torch.randn(B, N, K, L, 3)
    
    # 初始化 SGE 模块
    model = SpatialGraphEmbedding3D(in_channels=in_channels, dmodel=dmodel)
    
    # 前向传播
    out = model(pathlines_features, positions)
    
    print(f"输入特征形状: {pathlines_features.shape}")
    print(f"输入坐标形状: {positions.shape}")
    print(f"SGE 输出形状: {out.shape} (预期: B, N, L, dmodel = {B, N, L, dmodel})")
    
    if out.shape == (B, N, L, dmodel):
        print("测试通过！SGE 模块维度变换完全正确。")
    else:
        print("测试失败：输出维度异常。")
