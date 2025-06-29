#!/usr/bin/env python3
"""
Simplified Poetry Transformer with 50M parameters and 3 core mechanisms.
Implements rhythmic positional encoding, forced length decoding, and lightweight architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from .config import config


class RhythmicPositionalEncoding(nn.Module):
    """韵律感知的位置编码 - 核心机制1"""
    
    def __init__(self, hidden_size: int = 384, max_seq_len: int = 125):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 句内位置编码：1,2,3,4,5,1,2,3,4,5...
        self.char_pos_embed = nn.Embedding(8, hidden_size)  # 1-7字位置
        
        # 标准序列位置编码：保持原有能力
        self.seq_pos_embed = nn.Embedding(max_seq_len, hidden_size)
        
        # 句子边界编码
        self.sentence_boundary_embed = nn.Embedding(3, hidden_size)  # 0:句中, 1:句首, 2:句尾
        
    def forward(self, input_ids: torch.Tensor, char_positions: Optional[torch.Tensor] = None,
                sentence_boundaries: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            char_positions: [batch_size, seq_len] 句内字符位置(1-7)
            sentence_boundaries: [batch_size, seq_len] 句子边界信息(0,1,2)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 标准序列位置
        seq_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_pos_emb = self.seq_pos_embed(seq_positions)
        
        # 句内字符位置（如果提供）
        if char_positions is not None:
            char_pos_emb = self.char_pos_embed(char_positions)
        else:
            # 默认假设5言诗句
            char_pos_emb = torch.zeros_like(seq_pos_emb)
        
        # 句子边界（如果提供）
        if sentence_boundaries is not None:
            boundary_emb = self.sentence_boundary_embed(sentence_boundaries)
        else:
            boundary_emb = torch.zeros_like(seq_pos_emb)
        
        return seq_pos_emb + char_pos_emb + boundary_emb


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 投影到Q、K、V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 因果mask（保证只能看到之前的token）
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # 应用额外的attention mask
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        return self.out_proj(attn_output)


class FeedForwardNetwork(nn.Module):
    """前馈网络"""
    
    def __init__(self, hidden_size: int, feedforward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer解码器块"""
    
    def __init__(self, hidden_size: int, num_heads: int, feedforward_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(hidden_size, feedforward_dim, dropout)
        
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        attn_output = self.self_attention(hidden_states, attention_mask)
        hidden_states = residual + self.dropout(attn_output)
        
        # 前馈网络 + 残差连接
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        hidden_states = residual + self.dropout(ff_output)
        
        return hidden_states


class SimplifiedPoetryTransformer(nn.Module):
    """简化的诗歌Transformer - 50M参数轻量版本"""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        super().__init__()
        
        # 使用提供的配置或默认配置
        if config_dict is None:
            self.config = self._get_default_config()
        else:
            self.config = config_dict
            
        # 核心参数
        self.vocab_size = self.config['vocab_size']
        self.hidden_size = self.config['hidden_size']
        self.num_layers = self.config['num_layers']
        self.num_heads = self.config['num_heads']
        self.feedforward_dim = self.config['feedforward_dim']
        self.max_seq_len = self.config['max_seq_len']
        self.dropout = self.config['dropout']
        
        # 核心组件
        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_encoding = RhythmicPositionalEncoding(self.hidden_size, self.max_seq_len)
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                feedforward_dim=self.feedforward_dim,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # 输出层
        self.ln_final = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # 权重初始化
        self.apply(self._init_weights)
        
        # 计算参数量
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def _get_default_config(self) -> Dict:
        """获取优化的50M参数配置 - 平衡深度与宽度"""
        return {
            'vocab_size': 8293,
            'hidden_size': 576,      # 512 → 576维 (+12.5%) 
            'num_layers': 12,        # 保持12层深度
            'num_heads': 9,          # 8 → 9头 (576维需要被9整除，64维/头)
            'feedforward_dim': 2304, # 576 * 4 = 2304维
            'max_seq_len': 125,
            'dropout': 0.1
        }
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, 
                char_positions: Optional[torch.Tensor] = None,
                sentence_boundaries: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len] 输入token ids
            char_positions: [batch_size, seq_len] 句内字符位置
            sentence_boundaries: [batch_size, seq_len] 句子边界信息
            attention_mask: [batch_size, seq_len] 注意力mask
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] 输出logits
        """
        # 嵌入层
        hidden_states = self.embeddings(input_ids)
        
        # 位置编码（包含韵律信息）
        pos_embeddings = self.pos_encoding(input_ids, char_positions, sentence_boundaries)
        hidden_states = hidden_states + pos_embeddings
        
        # Transformer层
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, attention_mask)
        
        # 最终层归一化
        hidden_states = self.ln_final(hidden_states)
        
        # 语言模型头
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def get_num_params(self) -> int:
        """获取参数量"""
        return self.num_parameters
    
    def print_model_info(self):
        """打印模型信息"""
        print("=" * 60)
        print("SIMPLIFIED POETRY TRANSFORMER MODEL INFO")
        print("=" * 60)
        print(f"模型配置:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print(f"\n总参数量: {self.num_parameters:,} ({self.num_parameters/1e6:.1f}M)")
        print(f"目标参数量: ~50M")
        print("=" * 60)


def create_poetry_transformer(config_dict: Optional[Dict] = None) -> SimplifiedPoetryTransformer:
    """创建诗歌Transformer模型"""
    model = SimplifiedPoetryTransformer(config_dict)
    model.print_model_info()
    return model


if __name__ == "__main__":
    # 测试模型创建
    print("创建50M参数轻量Transformer模型...")
    model = create_poetry_transformer()
    
    # 测试前向传播
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 8293, (batch_size, seq_len))
    char_positions = torch.randint(1, 8, (batch_size, seq_len))
    
    print(f"\n测试输入: {input_ids.shape}")
    print(f"字符位置: {char_positions.shape}")
    
    with torch.no_grad():
        logits = model(input_ids, char_positions)
        print(f"输出logits: {logits.shape}")
        print(f"✓ 模型创建和前向传播测试成功！") 