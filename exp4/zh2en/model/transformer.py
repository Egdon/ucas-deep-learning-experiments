import zh2en.config as config
import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = config.device


class LabelSmoothing(nn.Module):
    """实现标签平滑正则化技术"""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Embeddings(nn.Module):
    """词嵌入层"""
    
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 返回词嵌入，需要乘以sqrt(d_model)进行缩放
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """位置编码层"""
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model, device=DEVICE)
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        
        # 计算位置编码的除数项
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * 
                           -(math.log(10000.0) / d_model))

        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加批次维度
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 添加位置编码到词嵌入
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):
    """缩放点积注意力机制"""
    d_k = query.size(-1)

    # 计算注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 应用softmax
    p_attn = F.softmax(scores, dim=-1)

    # 应用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 应用线性变换并重塑为多头形式
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                           for l, x in zip(self.linears, (query, key, value))]

        # 2) 应用注意力机制
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 拼接多头结果
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    """层归一化"""
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """残差连接和层归一化的组合"""
    
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 先归一化再残差连接
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Encoder(nn.Module):
    """编码器：由N个编码器层组成"""
    
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # 依次通过N个编码器层
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """编码器层：自注意力 + 前馈网络"""
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # 自注意力子层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 前馈网络子层
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """解码器：由N个解码器层组成"""
    
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 依次通过N个解码器层
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """解码器层：自注意力 + 交叉注意力 + 前馈网络"""
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        # 自注意力子层
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 交叉注意力子层
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 前馈网络子层
        return self.sublayer[2](x, self.feed_forward)


class Transformer(nn.Module):
    """完整的Transformer模型"""
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 编码器输出作为解码器的memory输入
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


class Generator(nn.Module):
    """生成器：将模型输出转换为词汇表上的概率分布"""
    
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """构造Transformer模型"""
    c = copy.deepcopy
    
    # 实例化各个组件
    attn = MultiHeadedAttention(h, d_model).to(DEVICE)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)
    position = PositionalEncoding(d_model, dropout).to(DEVICE)
    
    # 构造完整模型
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), c(position)),
        Generator(d_model, tgt_vocab)).to(DEVICE)

    # Xavier初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model.to(DEVICE)


def subsequent_mask(size):
    """生成后续掩码，防止解码器看到未来信息"""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def batch_greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    """批量贪心解码"""
    batch_size, src_seq_len = src.size()
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0

    memory = model.encode(src, src_mask)
    tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data)

    for s in range(max_len):
        tgt_mask = subsequent_mask(tgt.size(1)).expand(batch_size, -1, -1).type_as(src.data)
        out = model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask))

        prob = model.generator(out[:, -1, :])
        pred = torch.argmax(prob, dim=-1)

        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
        pred = pred.cpu().numpy()
        
        for i in range(batch_size):
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break

    return results


def greedy_decode(model, src, src_mask, max_len=64, start_symbol=2, end_symbol=3):
    """单句贪心解码"""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, Variable(ys),
                         Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        
        if next_word == end_symbol:
            break
            
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    
    return ys 