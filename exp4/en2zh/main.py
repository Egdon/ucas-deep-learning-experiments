import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

from train import train, test, translate
from data_loader import MTDataset
from utils import english_tokenizer_load
from model import make_model, LabelSmoothing


class NoamOpt:
    """Noam优化器包装器，实现学习率调度"""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """更新参数和学习率"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """实现学习率调度公式"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * 
                            min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """获取标准NoamOpt优化器
    
    Args:
        model: Transformer模型
    
    Returns:
        NoamOpt优化器实例
    """
    return NoamOpt(model.src_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run():
    """主运行函数"""
    # 设置日志
    utils.set_logger(config.log_path)
    
    # 创建数据集
    logging.info("-------- 创建数据集 --------")
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)
    
    logging.info("训练集大小: {}".format(len(train_dataset)))
    logging.info("验证集大小: {}".format(len(dev_dataset)))
    logging.info("测试集大小: {}".format(len(test_dataset)))

    # 创建数据加载器
    logging.info("-------- 创建数据加载器 --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("训练批次数: {}".format(len(train_dataloader)))
    logging.info("验证批次数: {}".format(len(dev_dataloader)))
    logging.info("测试批次数: {}".format(len(test_dataloader)))

    # 创建模型
    logging.info("-------- 创建Transformer模型 --------")
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model_par = torch.nn.DataParallel(model)
    
    logging.info("模型参数量: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 设置损失函数
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.tgt_vocab_size, 
                                 padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
        logging.info("使用标签平滑损失函数")
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        logging.info("使用交叉熵损失函数")

    # 设置优化器
    if config.use_noamopt:
        optimizer = get_std_opt(model)
        logging.info("使用NoamOpt优化器")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        logging.info("使用AdamW优化器，学习率: {}".format(config.lr))

    # 开始训练
    logging.info("-------- 开始训练 --------")
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)
    
    # 测试模型
    logging.info("-------- 开始测试 --------")
    test(test_dataloader, model, criterion)


def one_sentence_translate(sent, beam_search=False):
    """单句翻译示例函数
    
    Args:
        sent: 英文句子
        beam_search: 是否使用束搜索（暂未实现）
    """
    # 创建模型
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    
    # 分词处理
    tokenizer = english_tokenizer_load()
    BOS = tokenizer.bos_id()  # 2
    EOS = tokenizer.eos_id()  # 3
    src_tokens = [[BOS] + tokenizer.EncodeAsIds(sent) + [EOS]]
    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
    
    # 翻译
    result = translate(batch_input, model, use_beam=beam_search)
    return result


def translate_example():
    """翻译示例"""
    sent = ("The near-term policy remedies are clear: raise the minimum wage to a level that will keep a "
           "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit "
           "to childless workers.")
    
    print("原文:", sent)
    print("参考译文: 近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平，扩大对无子女劳动者的工资所得税减免。")
    
    result = one_sentence_translate(sent, beam_search=False)
    print("模型译文:", result[0] if result else "翻译失败")


if __name__ == "__main__":
    import os
    import warnings
    
    # 设置GPU环境
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 根据project_definition使用V100S单卡
    warnings.filterwarnings('ignore')
    
    # 运行主程序
    run()
    
    # 运行翻译示例（可选）
    # translate_example() 