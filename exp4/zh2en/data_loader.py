import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load, chinese_tokenizer_load

import config
DEVICE = config.device


def subsequent_mask(size):
    """生成后续掩码，防止解码器看到未来信息"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """批次数据容器，包含掩码处理"""
    
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        src = src.to(DEVICE)
        self.src = src
        
        # 源序列掩码：标记非填充位置
        self.src_mask = (src != pad).unsqueeze(-2)
        
        if trg is not None:
            trg = trg.to(DEVICE)
            # 目标序列输入（去掉最后一个token）
            self.trg = trg[:, :-1]
            # 目标序列标签（去掉第一个token）
            self.trg_y = trg[:, 1:]
            # 目标序列掩码：结合填充掩码和后续掩码
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 统计实际token数量（用于损失归一化）
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """创建标准掩码：隐藏填充和未来词汇"""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    """机器翻译数据集类 - 中译英版本"""
    
    def __init__(self, data_path):
        self.out_en_sent, self.out_cn_sent = self.get_dataset(data_path, sort=True)
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        
        # 特殊token ID（使用中文分词器作为基准，因为源语言是中文）
        self.PAD = self.sp_chn.pad_id()  # 0
        self.BOS = self.sp_chn.bos_id()  # 2
        self.EOS = self.sp_chn.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """按句子长度排序，返回排序后的索引"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """加载数据集并按中文句子长度排序（因为源语言是中文）"""
        dataset = json.load(open(data_path, 'r'))
        out_en_sent = []
        out_cn_sent = []
        
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx][0])  # 英文句子
            out_cn_sent.append(dataset[idx][1])  # 中文句子
        
        if sort:
            # 按中文句子长度排序（源语言）
            sorted_index = self.len_argsort(out_cn_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        
        return out_en_sent, out_cn_sent

    def __getitem__(self, idx):
        """获取单个样本 - 中译英：中文在前，英文在后"""
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        # 中译英：返回[中文, 英文]
        return [chn_text, eng_text]

    def __len__(self):
        """返回数据集大小"""
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        """批处理函数：中译英处理"""
        src_text = [x[0] for x in batch]  # 中文文本（源语言）
        tgt_text = [x[1] for x in batch]  # 英文文本（目标语言）

        # 分词并添加BOS和EOS token
        # 源语言：中文
        src_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        # 目标语言：英文
        tgt_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # 填充到相同长度
        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                  batch_first=True, padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                   batch_first=True, padding_value=self.PAD)

        return Batch(src_text, tgt_text, batch_input, batch_target, self.PAD) 