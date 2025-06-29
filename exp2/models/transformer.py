import torch.nn as nn
from .attention import Attention

class FeedForward(nn.Module):
    """前馈神经网络"""
    
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        
        # 创建depth层编码器
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        # 逐层处理，每层包含注意力和前馈网络，都有残差连接
        for attn, ff in self.layers:
            x = attn(x) + x  # 注意力 + 残差连接
            x = ff(x) + x    # 前馈网络 + 残差连接

        return self.norm(x) 