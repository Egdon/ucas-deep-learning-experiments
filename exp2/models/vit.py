import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .transformer import Transformer

def pair(t):
    """确保输入是一对数值"""
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    """Vision Transformer模型 - CIFAR-10版本"""
    
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, 
                 heads, mlp_dim, pool='cls', channels=3, dim_head=64, 
                 dropout=0.1, emb_dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 确保图像尺寸能被patch尺寸整除
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Patch嵌入层：将图像分割成patches并映射到embedding空间
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 位置编码：为每个patch位置学习一个嵌入向量
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # CLS token：用于分类的特殊token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer编码器
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.gradient_checkpointing = gradient_checkpointing

        self.pool = pool
        self.to_latent = nn.Identity()

        # 分类头
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # img shape: (batch_size, channels, height, width)
        x = self.to_patch_embedding(img)  # (batch_size, num_patches, dim)
        b, n, _ = x.shape

        # 添加CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches+1, dim)
        
        # 添加位置编码
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # 通过Transformer编码器
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.transformer, x, use_reentrant=False)
        else:
            x = self.transformer(x)

        # 池化：使用CLS token或平均池化
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

def create_vit_small_patch8_224(num_classes=10):
    """创建ViT-Small/8模型用于CIFAR-10 (需要大显存)"""
    return ViT(
        image_size=224,
        patch_size=8,
        num_classes=num_classes,
        dim=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        dropout=0.1,
        emb_dropout=0.1,
        dim_head=64
    )

def create_vit_tiny_patch16_224(num_classes=10, gradient_checkpointing=False):
    """创建ViT-Tiny/16模型用于CIFAR-10 (显存友好)"""
    return ViT(
        image_size=224,
        patch_size=16,
        num_classes=num_classes,
        dim=192,
        depth=6,
        heads=3,
        mlp_dim=768,
        dropout=0.1,
        emb_dropout=0.1,
        dim_head=64,
        gradient_checkpointing=gradient_checkpointing
    ) 