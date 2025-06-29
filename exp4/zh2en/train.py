import torch
import torch.nn as nn
from torch.autograd import Variable
import logging
import sacrebleu
from tqdm import tqdm
import time

import config
import sys
sys.path.append('../model')
from model.transformer import batch_greedy_decode
from utils import english_tokenizer_load  # 中译英使用英文分词器做BLEU评估
from train_logger import TrainingLogger


def run_epoch(data, model, loss_compute):
    """运行一个训练或验证epoch"""
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data, desc="Processing batches"):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
    
    return total_loss / total_tokens


def train(train_data, dev_data, model, model_par, criterion, optimizer):
    """训练主循环"""
    best_bleu_score = 0.0
    early_stop = config.early_stop
    
    # 初始化训练日志记录器
    logger = TrainingLogger()
    
    # 记录训练配置
    training_config = {
        "model_layers": config.n_layers,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
        "batch_size": config.batch_size,
        "max_epochs": config.epoch_num,
        "early_stop": config.early_stop,
        "use_noamopt": config.use_noamopt,
        "use_smoothing": config.use_smoothing,
        "dropout": config.dropout,
        "src_vocab_size": config.src_vocab_size,
        "tgt_vocab_size": config.tgt_vocab_size
    }
    logger.log_config(training_config)
    
    logging.info("开始训练...")
    training_start_time = time.time()
    
    for epoch in range(1, config.epoch_num + 1):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = run_epoch(train_data, model_par,
                              MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))
        logging.info("Epoch: {}, 训练损失: {:.4f}".format(epoch, train_loss))
        
        # 验证阶段
        model.eval()
        dev_loss = run_epoch(dev_data, model_par,
                            MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(dev_data, model)
        logging.info('Epoch: {}, 验证损失: {:.4f}, BLEU分数: {:.2f}'.format(epoch, dev_loss, bleu_score))

        # 获取当前学习率
        if hasattr(optimizer, '_rate'):
            current_lr = optimizer._rate  # NoamOpt
        elif hasattr(optimizer, 'param_groups'):
            current_lr = optimizer.param_groups[0]['lr']  # 标准优化器
        else:
            current_lr = 0.0
        
        epoch_time = time.time() - epoch_start_time

        # 保存最佳模型
        if bleu_score > best_bleu_score:
            torch.save(model.state_dict(), config.model_path)
            best_bleu_score = bleu_score
            early_stop = config.early_stop
            logging.info("-------- 保存最佳模型! BLEU: {:.2f} --------".format(bleu_score))
        else:
            early_stop -= 1
            logging.info("早停剩余轮数: {}".format(early_stop))
        
        # 记录epoch数据到日志
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=dev_loss, 
            bleu_score=bleu_score,
            learning_rate=current_lr,
            epoch_time=epoch_time,
            best_bleu=best_bleu_score,
            early_stop_count=early_stop
        )
        
        # 早停检查
        if early_stop == 0:
            logging.info("-------- 早停触发! --------")
            break

    # 训练完成，记录总结信息
    total_training_time = time.time() - training_start_time
    logger.log_training_complete(total_training_time, best_bleu_score, config.model_path)
    
    logging.info("训练完成! 最佳BLEU分数: {:.2f}".format(best_bleu_score))
    logging.info("训练数据已保存到: {}".format(logger.get_metrics_file()))
    logging.info("详细日志已保存到: {}".format(logger.get_detailed_log_file()))


class LossCompute:
    """损失计算和反向传播"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                             y.contiguous().view(-1)) / norm
        loss.backward()
        
        if self.opt is not None:
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        
        return loss.data.item() * norm.float()


class MultiGPULossCompute:
    """多GPU损失计算（适配单GPU使用）"""

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # 分块处理生成
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # 预测分布
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                  requires_grad=self.opt is not None)]
                         for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # 计算损失
            y = [(g.contiguous().view(-1, g.size(-1)),
                 t[:, i:i + chunk_size].contiguous().view(-1))
                for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # 汇总并归一化损失
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # 反向传播
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # 通过transformer反向传播
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        
        return total * normalize


def evaluate(data, model, mode='dev', use_beam=False):
    """在数据集上评估模型并计算BLEU分数 - 中译英版本"""
    sp_eng = english_tokenizer_load()  # 中译英：目标语言是英文
    trg = []
    res = []
    
    with torch.no_grad():
        for batch in tqdm(data, desc="Evaluating"):
            en_sent = batch.trg_text  # 英文目标句子
            src = batch.src
            src_mask = (src != 0).unsqueeze(-2)
            
            # 使用贪心解码（暂不使用束搜索以提高速度）
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
            
            # 解码结果转换为文本（使用英文分词器）
            translation = [sp_eng.decode_ids(_s) for _s in decode_result]
            trg.extend(en_sent)
            res.extend(translation)
    
    # 保存测试结果
    if mode == 'test':
        import os
        os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
        with open(config.output_path, "w", encoding='utf-8') as fp:
            for i in range(len(trg)):
                line = "idx:" + str(i) + " ||| " + trg[i] + ' ||| ' + res[i] + '\n'
                fp.write(line)
    
    # 计算BLEU分数（英文评估）
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='intl')  # 英文使用intl tokenizer
    return float(bleu.score)


def test(data, model, criterion):
    """在测试集上评估模型"""
    with torch.no_grad():
        # 加载最佳模型
        model.load_state_dict(torch.load(config.model_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        
        # 计算测试损失和BLEU分数
        test_loss = run_epoch(data, model_par,
                             MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(data, model, 'test')
        
        logging.info('测试损失: {:.4f}, 测试BLEU分数: {:.2f}'.format(test_loss, bleu_score))


def translate(src, model, use_beam=False):
    """单句翻译 - 中译英版本"""
    sp_eng = english_tokenizer_load()  # 中译英：目标语言是英文
    
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        
        # 贪心解码
        decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        
        # 解码为文本（使用英文分词器）
        translation = [sp_eng.decode_ids(_s) for _s in decode_result]
        return translation 