import torch
import torch.nn as nn
from einops import rearrange

class Attention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, dim, heads=6, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # 输入: (batch_size, seq_len, dim)
        x = self.norm(x)

        # 生成 Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 应用softmax
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 应用注意力权重到值
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out) 