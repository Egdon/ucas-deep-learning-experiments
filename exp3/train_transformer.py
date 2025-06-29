#!/usr/bin/env python3
"""
Transformer训练脚本 - 支持双环境和三大核心机制
开发环境：RTX 5070Ti
训练环境：V100s 32GB
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import numpy as np
from tqdm import tqdm
import logging

# 项目导入
from models.model import create_poetry_transformer, SimplifiedPoetryTransformer
from models.config import Config, config
from models.dataset import create_dataloaders
from utils.train_utils import (
    create_optimizer, create_scheduler, 
    save_checkpoint, load_checkpoint,
    calculate_metrics, setup_logging
)
from utils.generate_utils import ForcedLengthDecoding, SimplePoetryGenerator
from utils.visualization import create_training_visualizer

class PoetryTransformerTrainer:
    """诗歌Transformer训练器 - 集成三大核心机制"""
    
    def __init__(self, user_config: Dict, resume_from: Optional[str] = None):
        # 使用全局config作为基础，用户配置覆盖
        self.config = config.get_training_config()
        self.config.update(user_config)  # 用户设置覆盖默认值
        
        self.device = self._setup_device()
        self.logger = self._setup_logging()
        
        # 使用config.py的环境检测
        self.environment = config.ENVIRONMENT
        self._apply_environment_optimizations()
        
        # 模型初始化
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 训练组件
        self.optimizer = create_optimizer(self.model, self.config)
        self.scheduler = create_scheduler(self.optimizer, self.config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config['pad_token_id'])
        
        # 数据加载器
        self.train_loader, self.val_loader, self.vocab_info = self._create_dataloaders()
        
        # 生成器（用于验证）
        self.generator = SimplePoetryGenerator(
            self.model, self.vocab_info['ix2word'], self.vocab_info['word2ix']
        )
        
        # 强制句长解码器
        self.forced_decoder = ForcedLengthDecoding(self.vocab_info['word2ix'])
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_stats = []
        
        # 可视化器
        self.visualizer = create_training_visualizer("plots")
        
        # 训练指标记录
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epochs': []
        }
        
        # 恢复训练
        if resume_from:
            self._resume_training(resume_from)
            
        self.logger.info(f"训练器初始化完成 - 环境: {self.environment}")
        self.logger.info(f"模型参数量: {self.model.get_num_params():,}")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 使用GPU: {gpu_name}")
            print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return device
        else:
            print("⚠️  使用CPU训练")
            return torch.device('cpu')
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        return setup_logging(
            log_file=f"logs/transformer_training_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
    
    def _apply_environment_optimizations(self):
        """应用环境优化配置"""
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        print(f"🚀 使用GPU: {gpu_name}")
        print(f"💾 GPU内存: {gpu_memory:.1f}GB")
        print(f"🖥️  检测到{self.environment}环境")
        
        # 根据环境优化配置
        if self.environment == 'server':
            # V100s服务器优化
            optimizations = {
                'batch_size': 256,  # 大幅提升批次大小
                'accumulation_steps': 1,
                'max_seq_len': config.MAX_SEQ_LEN,
                'num_workers': 4,  # 优化工作进程数，避免过度上下文切换
                'enable_amp': True,  # 混合精度训练
                'pin_memory': True,
            }
            print(f"⚡ 服务器优化: 批次{optimizations['batch_size']}, 混合精度训练, {optimizations['num_workers']}工作进程")
            print("🔥 性能优化: 移除重复韵律计算，向量化数据处理")
            
        elif self.environment == 'development':
            optimizations = {
                'batch_size': 64,
                'accumulation_steps': 1,
                'max_seq_len': config.MAX_SEQ_LEN,
                'num_workers': 4,
                'enable_amp': True,
                'pin_memory': True,
            }
            print(f"💻 开发环境优化: 批次{optimizations['batch_size']}, 混合精度训练")
            
        else:
            optimizations = {
                'batch_size': 32,
                'accumulation_steps': 2,
                'max_seq_len': 100,
                'num_workers': 2,
                'enable_amp': torch.cuda.is_available(),
                'pin_memory': False,
            }
            print(f"🔧 通用环境配置: 批次{optimizations['batch_size']}")
        
        # 更新配置
        self.config.update(optimizations)
        
        # 初始化混合精度训练
        if self.config.get('enable_amp', False):
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ 混合精度训练已启用")
        else:
            self.scaler = None
    
    def _create_model(self) -> SimplifiedPoetryTransformer:
        """创建模型（使用config.py配置）"""
        # 使用config.py中的模型配置
        model_config = config.get_model_config()
        
        # 根据实际数据调整vocab_size
        model_config['vocab_size'] = 8293
        
        print("=" * 60)
        print("SIMPLIFIED POETRY TRANSFORMER MODEL INFO")
        print("=" * 60)
        print("模型配置:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        model = create_poetry_transformer(model_config)
        param_count = model.get_num_params()
        print(f"\n总参数量: {param_count:,} ({param_count/1e6:.1f}M)")
        print("目标参数量: ~50M")
        print("=" * 60)
        
        return model
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader, Dict]:
        """创建数据加载器"""
        train_loader, val_loader, vocab_info = create_dataloaders(
            data_dir=self.config.get('data_dir', 'data'),
            batch_size=self.config['batch_size'],
            max_seq_len=self.config['max_seq_len'],
            num_workers=self.config.get('num_workers', 0),
            test_size=0.1,
            add_rhythmic_info=True  # 启用韵律信息
        )
        
        self.logger.info(f"数据加载完成:")
        self.logger.info(f"  训练集: {len(train_loader)} 批次")
        self.logger.info(f"  验证集: {len(val_loader)} 批次")
        self.logger.info(f"  词汇表大小: {len(vocab_info['word2ix'])}")
        
        return train_loader, val_loader, vocab_info
    
    def _compute_rhythmic_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """计算韵律位置信息（优化版本）"""
        # 如果batch中已经包含char_positions，直接返回
        return None  # 这个函数将被移除，由collate_fn直接提供
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 准备输入数据
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # 使用DataLoader预计算的韵律位置（核心机制1）
            char_positions = batch['char_positions'].to(self.device)
            
            # 前向传播（使用轻量Transformer - 核心机制3）
            if self.scaler is not None:
                # 混合精度训练
                with torch.cuda.amp.autocast():
                    logits = self.model(input_ids, char_positions=char_positions)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                    loss = loss / self.config['accumulation_steps']
                
                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                # 标准训练
                logits = self.model(input_ids, char_positions=char_positions)
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss = loss / self.config['accumulation_steps']
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                if self.scaler is not None:
                    # 混合精度训练的优化步骤
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # 标准训练的优化步骤
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item() * self.config['accumulation_steps']
            total_samples += input_ids.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item() * self.config['accumulation_steps']:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="验证中"):
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                
                # 使用DataLoader预计算的韵律位置
                char_positions = batch['char_positions'].to(self.device)
                
                # 前向传播
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = self.model(input_ids, char_positions=char_positions)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                else:
                    logits = self.model(input_ids, char_positions=char_positions)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                total_samples += input_ids.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'val_loss': avg_loss,
            'perplexity': perplexity
        }
    
    def generate_samples(self, num_samples: int = 3) -> List[str]:
        """生成样本诗歌（集成约束解码算法 - 核心机制2）"""
        samples = []
        prompts = ["春", "月", "山"]
        poem_types = ["七言绝句", "五言绝句", "七言律诗"]
        
        # 创建约束解码生成器
        from utils.generate_utils import ConstrainedPoetryGenerator
        constrained_generator = ConstrainedPoetryGenerator(
            self.model, self.vocab_info['ix2word'], self.vocab_info['word2ix']
        )
        
        for i, prompt in enumerate(prompts[:num_samples]):
            poem_type = poem_types[i % len(poem_types)]
            try:
                # 使用约束解码生成严格格律诗歌
                poem = constrained_generator.generate_constrained_poem(
                    prompt=prompt,
                    poem_type=poem_type,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9
                )
                samples.append(f"【{poem_type}】提示'{prompt}':\n{poem}")
            except Exception as e:
                samples.append(f"【{poem_type}】提示'{prompt}': [生成失败: {e}]")
        
        return samples
    
    def train(self, num_epochs: int):
        """主训练循环"""
        self.logger.info(f"开始训练 - 目标epochs: {num_epochs}")
        self.logger.info(f"三大核心机制: ✓韵律位置编码 ✓强制句长解码 ✓轻量Transformer")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 记录学习率
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 合并指标
            metrics = {**train_metrics, **val_metrics, 'learning_rate': current_lr}
            self.training_stats.append(metrics)
            
            # 更新指标历史
            self.metrics_history['epochs'].append(epoch + 1)
            self.metrics_history['train_loss'].append(metrics['train_loss'])
            self.metrics_history['val_loss'].append(metrics['val_loss'])
            self.metrics_history['learning_rate'].append(current_lr)
            
            # 日志记录
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {metrics['train_loss']:.4f} - "
                f"Val Loss: {metrics['val_loss']:.4f} - "
                f"Perplexity: {metrics['perplexity']:.2f} - "
                f"LR: {current_lr:.2e}"
            )
            
            # 生成可视化（每5个epoch）
            if (epoch + 1) % 5 == 0:
                try:
                    self.visualizer.plot_training_curves(
                        self.metrics_history,
                        self.metrics_history['epochs'],
                        title=f"Transformer Poetry Training (Epoch {epoch+1})",
                        save_name=f"training_curves_epoch_{epoch+1:03d}.png"
                    )
                except Exception as e:
                    self.logger.warning(f"可视化生成失败: {e}")
            
            # 生成样本
            if (epoch + 1) % 5 == 0:
                samples = self.generate_samples()
                self.logger.info("生成样本:")
                for sample in samples:
                    self.logger.info(f"  {sample}")
            
            # 保存检查点
            is_best = metrics['val_loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['val_loss']
            
            self._save_checkpoint(epoch, metrics, is_best)
            
            # 早停检查
            if self._should_early_stop():
                self.logger.info("触发早停，训练结束")
                break
        
        # 训练结束后生成最终图表
        try:
            self.logger.info("生成最终训练曲线...")
            final_plot_path = self.visualizer.plot_training_curves(
                self.metrics_history,
                self.metrics_history['epochs'],
                title="Final Transformer Poetry Training Results",
                save_name="final_training_curves.png"
            )
            
            # 生成单独的损失对比图
            self.visualizer.plot_loss_comparison(
                self.metrics_history['train_loss'],
                self.metrics_history['val_loss'],
                self.metrics_history['epochs'],
                save_name="final_loss_comparison.png"
            )
            
            # 生成学习率图
            if len(self.metrics_history['learning_rate']) > 1:
                self.visualizer.plot_learning_rate_schedule(
                    self.metrics_history['learning_rate'],
                    self.metrics_history['epochs'],
                    save_name="final_learning_rate_schedule.png"
                )
            
            # 生成困惑度图
            train_perplexity = [np.exp(loss) for loss in self.metrics_history['train_loss']]
            val_perplexity = [np.exp(loss) for loss in self.metrics_history['val_loss']]
            self.visualizer.plot_perplexity_trend(
                train_perplexity,
                val_perplexity,
                self.metrics_history['epochs'],
                save_name="final_perplexity_trend.png"
            )
            
            self.logger.info(f"📊 所有训练图表已保存到 plots/ 目录")
            
        except Exception as e:
            self.logger.error(f"最终图表生成失败: {e}")
        
        self.logger.info("训练完成！")
    
    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'vocab_info': self.vocab_info,
            'training_stats': self.training_stats
        }
        
        # 保存最新检查点
        save_checkpoint(checkpoint, is_best, 
                       checkpoint_dir=f"checkpoints/{self.environment}")
        
        if is_best:
            self.logger.info(f"🎉 新的最佳模型! Val Loss: {metrics['val_loss']:.4f}")
    
    def _resume_training(self, checkpoint_path: str):
        """恢复训练"""
        checkpoint = load_checkpoint(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.training_stats = checkpoint.get('training_stats', [])
        self.best_loss = min([stat['val_loss'] for stat in self.training_stats], 
                           default=float('inf'))
        
        # 恢复指标历史
        if self.training_stats:
            self.metrics_history = {
                'epochs': list(range(1, len(self.training_stats) + 1)),
                'train_loss': [stat['train_loss'] for stat in self.training_stats],
                'val_loss': [stat['val_loss'] for stat in self.training_stats],
                'learning_rate': [stat.get('learning_rate', 0) for stat in self.training_stats]
            }
        
        self.logger.info(f"从epoch {self.current_epoch} 恢复训练，已恢复 {len(self.training_stats)} 个epoch的指标")
    
    def _should_early_stop(self, patience: int = None) -> bool:
        """早停检查 - 修复逻辑错误"""
        if patience is None:
            patience = config.EARLY_STOPPING_PATIENCE
            
        if len(self.training_stats) < patience:
            return False
        
        # 检查最近patience个epoch中验证损失是否没有改善
        recent_losses = [stat['val_loss'] for stat in self.training_stats[-patience:]]
        min_recent_loss = min(recent_losses)
        
        # 如果最近patience个epoch中最小损失都没有低于历史最佳，则早停
        # 但如果刚好等于最佳（即最近刚达到最佳），不早停
        return min_recent_loss > self.best_loss


def main():
    """主函数"""
    # 用户配置（覆盖默认设置）
    user_config = {
        'data_dir': 'data',
        'pad_token_id': 0,
        'num_epochs': config.NUM_EPOCHS,  # 使用config.py中的epoch配置
        'save_every': 5,
        'validate_every': 1,
        'warmup_steps': 1000,
        'max_grad_norm': 1.0,
        'weight_decay': 0.01,
    }
    
    # 创建必要目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 训练器
    trainer = PoetryTransformerTrainer(user_config)
    
    # 开始训练
    trainer.train(user_config['num_epochs'])


if __name__ == "__main__":
    main() 