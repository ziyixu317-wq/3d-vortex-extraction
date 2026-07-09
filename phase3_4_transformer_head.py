import torch
import torch.nn as nn
import torch.nn.functional as F

class PathlineTransformerLayer3D(nn.Module):
    r"""
    3D 迹线 Transformer 编码层 (Pathline Transformer Layer)
    
    参考《研究方案.pdf》中方程 (10)-(12) 的设计：
    q_i = \eta(h_i), k_j = \phi(h_j), v_j = \psi(h_j)
    a_{ij} = \rho( \gamma(q_i - k_j + \delta(\Delta p_{ij})) )
    f_i = \sum_j a_{ij} \odot (v_j + \delta(\Delta p_{ij}))
    
    此处自注意力机制不仅在空间上，而是在单个迹线的时序 (L) 维度上提取演化特征，
    结合了粒子的相对位移 \Delta p_{ij} 作为物理诱导。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # \delta 几何映射网络：将相对坐标偏移转换为特征维度的偏置
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        # \gamma 注意力权重提取网络
        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

        # \eta, \phi, \psi 线性投影
        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)
        
        self.linear_out = nn.Linear(dim, dim)

    def forward(self, x, pos):
        r"""
        Args:
            x:   (BN, L, C) 隐式特征 h^{SGE}
            pos: (BN, L, 3) 轨迹物理坐标
        Returns:
            out: (BN, L, C) 更新后的特征
        """
        BN, L, C = x.shape
        
        # 计算相对位移 \Delta p_{ij} = p_i - p_j
        # pos.unsqueeze(2): (BN, L, 1, 3) 对应 i
        # pos.unsqueeze(1): (BN, 1, L, 3) 对应 j
        delta_p = pos.unsqueeze(2) - pos.unsqueeze(1)  # (BN, L, L, 3)
        
        # \delta(\Delta p_{ij})
        pos_enc = self.pos_mlp(delta_p)  # (BN, L, L, C)
        
        # 提取 q_i, k_j, v_j
        q = self.linear_q(x).unsqueeze(2)  # (BN, L, 1, C)
        k = self.linear_k(x).unsqueeze(1)  # (BN, 1, L, C)
        v = self.linear_v(x).unsqueeze(1)  # (BN, 1, L, C)
        
        # 计算能量项: q_i - k_j + \delta(\Delta p_{ij})
        energy = q - k + pos_enc  # (BN, L, L, C)
        
        # \gamma 映射
        attn = self.attn_mlp(energy)
        
        # \rho (Softmax): 沿 j 维度归一化
        attn = F.softmax(attn, dim=2)  # (BN, L, L, C)
        
        # 逐元素加权 \sum_j a_{ij} \odot (v_j + \delta(\Delta p_{ij}))
        v_updated = v + pos_enc        # (BN, L, L, C)
        out = torch.sum(attn * v_updated, dim=2)  # (BN, L, C)
        
        # 最终线性映射
        out = F.relu(self.linear_out(out), inplace=True)
        return out


class VortexPredictor3D(nn.Module):
    r"""
    阶段三与阶段四：3D Transformer 时空特征交互与预测解码模块
    
    1. 接收 Phase 2 输出的节点图特征和中心点物理轨迹。
    2. 使用堆叠的 PathlineTransformerLayer3D 在时间序列上进行自注意力交互。
    3. 沿时间维度进行混合池化提取演化规律。
    4. 通过多层感知机(或全连接层)输出该采样点是否属于涡旋结构的概率。
    """
    def __init__(self, dmodel=252, num_encoder_layers=3, num_classes=1):
        super().__init__()
        self.dmodel = dmodel
        self.num_classes = num_classes
        
        self.transformer_layers = nn.ModuleList([
            PathlineTransformerLayer3D(dmodel) for _ in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(dmodel)
        
        # 预测头 (Phase 4) - 与 2D 版本参数一致
        self.fc = nn.Sequential(
            nn.Linear(dmodel, dmodel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dmodel // 2, dmodel // 4),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Linear(dmodel // 4, num_classes)

    def forward(self, sge_features, center_pathline, full_pos=None):
        r"""
        Args:
            sge_features:    (B, N, L, dmodel) 阶段二(SGE)聚合后的局部特征
            center_pathline: (B, N, L, 3) 阶段一记录的局部中心点随时间的物理演化轨迹
            full_pos:        (B, M, 3) (可选) 该 Patch 内全部的密集体素坐标。若提供，则会上采样特征至所有 M 点。
        Returns:
            predictions:     (B, N) 或 (B, M) 针对每个迹线或网格点的涡旋 Logits 预测
        """
        B, N, L, C = sge_features.shape
        assert C == self.dmodel, f"输入特征维度 {C} 与 dmodel {self.dmodel} 不匹配"
        
        # 展平 B 和 N 维度，让 Transformer 专注处理单条迹线的时间序列 L
        x = sge_features.view(B * N, L, C)
        pos = center_pathline.view(B * N, L, 3)
        
        # 1. 迹线 Transformer 时序特征重组
        for layer in self.transformer_layers:
            x = x + layer(x, pos)
        x = self.norm(x)
        
        x = x.view(B, N, L, self.dmodel)
        
        # 2. 混合池化 (Hybrid Pooling) - PDF 方程(13)
        # \tilde{f}_i = mean(f_i) + max(f_i)
        x_mean = x.mean(dim=2)
        x_max, _ = x.max(dim=2)
        x_pooled = x_mean + x_max  # (B, N, dmodel)
        
        # 3. 涡流分割特征投影
        if full_pos is not None:
            # 需要上采样到全分辨率网格
            sampled_pos = center_pathline[:, :, 0, :3]  # (B, N, 3) 使用 t=0 的位置
            x_pooled = self.propagate_features(full_pos, sampled_pos, x_pooled)  # (B, M, dmodel)
            
        logits = self.output(self.fc(x_pooled)).squeeze(-1)       # (B, N) 或 (B, M)
        
        return logits
        
    def propagate_features(self, full_pos, sampled_pos, sampled_features, chunk_size=100000):
        r"""
        类似 PointNet++ 的距离反比插值特征传播模块
        加入 Chunking 机制，防止 640x240x80 (千万级) 密集网格导致 OOM 显存爆炸
        """
        B, M, _ = full_pos.shape
        _, N, C = sampled_features.shape

        k = min(3, N)
        interpolated_features = []

        # 分块处理密集点，避免 M x N 的庞大距离矩阵占满显存
        for i in range(0, M, chunk_size):
            chunk_full_pos = full_pos[:, i:i+chunk_size, :]
            chunk_M = chunk_full_pos.shape[1]

            # 计算成对距离矩阵 (B, chunk_M, N)
            inner = -2 * torch.matmul(chunk_full_pos, sampled_pos.transpose(2, 1))
            xx = torch.sum(chunk_full_pos**2, dim=2, keepdim=True)
            yy = torch.sum(sampled_pos**2, dim=2, keepdim=True).transpose(2, 1)
            pairwise_distance = xx + inner + yy

            # 寻找 k=3 最近邻
            dist, idx = pairwise_distance.topk(k=k, dim=-1, largest=False)

            # 基于距离的倒数计算权重
            dist = torch.clamp(dist, min=1e-10)
            norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
            weight = (1.0 / dist) / norm  # (B, chunk_M, k)

            # 特征插值 (Batched Gather)
            batch_indices = torch.arange(B).view(B, 1, 1).expand(B, chunk_M, k)
            gathered_features = sampled_features[batch_indices, idx]  # (B, chunk_M, k, C)
            
            chunk_interpolated = torch.sum(weight.unsqueeze(-1) * gathered_features, dim=2)  # (B, chunk_M, C)
            interpolated_features.append(chunk_interpolated)

        return torch.cat(interpolated_features, dim=1)  # (B, M, C)

if __name__ == '__main__':
    print("=== 测试 Phase 3 & 4: 3D Pathline Transformer 预测模块 ===")
    
    B = 2       # Batch size
    N = 100     # Patch 内部采样迹线族数量
    L = 16      # 时序长度
    dmodel = 252 # 隐层维度
    
    # 模拟 Phase 2 输出的特征和中心迹线物理坐标
    sge_features = torch.randn(B, N, L, dmodel)
    center_pathline = torch.randn(B, N, L, 3)
    
    model = VortexPredictor3D(dmodel=dmodel, num_encoder_layers=3, num_classes=1)
    
    print(f"输入特征形状: {sge_features.shape}")
    print(f"输入轨迹形状: {center_pathline.shape}")
    
    predictions = model(sge_features, center_pathline)
    
    print(f"预测输出形状 (Logits): {predictions.shape} (预期: B, N = {B}, {N})")
    
    # 测试提供 full_pos 的上采样
    M = 1000
    full_pos = torch.randn(B, M, 3)
    predictions_full = model(sge_features, center_pathline, full_pos)
    print(f"密集预测输出形状 (Logits): {predictions_full.shape} (预期: B, M = {B}, {M})")
    
    if predictions.shape == (B, N) and predictions_full.shape == (B, M):
        print("测试通过！Transformer 时空演化提取、传播及池化预测维度完全正确。")
    else:
        print("测试失败：输出维度异常。")
