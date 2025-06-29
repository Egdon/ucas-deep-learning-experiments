#!/usr/bin/env python3
"""
Training utilities for poetry generation.
Includes basic loss functions, model management, monitoring and early stopping.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
import logging
from collections import defaultdict

from models import config

class SimplifiedPoetryLoss(nn.Module):
    """简化的诗歌生成损失函数"""
    
    def __init__(self, vocab_size: int, padding_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=padding_idx)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算基础语言模型损失
        
        Args:
            logits: 模型输出 (batch_size, seq_len, vocab_size)
            targets: 目标序列 (batch_size, seq_len)
            
        Returns:
            损失值
        """
        return self.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )

class ModelManager:
    """模型管理器，处理模型保存、加载和状态管理"""
    
    def __init__(self, checkpoint_dir: str = None):
        self.checkpoint_dir = Path(checkpoint_dir or config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, metrics: Dict = None,
                       is_best: bool = False) -> str:
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics or {},
            'config': config.get_model_config(),
            'timestamp': time.time()
        }
        
        # 保存常规检查点
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved to {best_path}")
        
        return str(checkpoint_path)
    
    @staticmethod
    def fix_state_dict_keys(state_dict: Dict[str, torch.Tensor], target_is_compiled: bool = False) -> Dict[str, torch.Tensor]:
        """
        修复状态字典键名，处理编译/未编译模型的差异
        
        Args:
            state_dict: 原始状态字典
            target_is_compiled: 目标模型是否为编译模型
            
        Returns:
            修复后的状态字典
        """
        # 检查是否包含_orig_mod前缀
        has_orig_mod = any(key.startswith('_orig_mod.') for key in state_dict.keys())
        
        if has_orig_mod and not target_is_compiled:
            # 移除_orig_mod前缀
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                    fixed_state_dict[new_key] = value
                else:
                    fixed_state_dict[key] = value
            return fixed_state_dict
            
        elif not has_orig_mod and target_is_compiled:
            # 添加_orig_mod前缀
            fixed_state_dict = {}
            for key, value in state_dict.items():
                new_key = f'_orig_mod.{key}'
                fixed_state_dict[new_key] = value
            return fixed_state_dict
            
        else:
            # 无需修改
            return state_dict
    
    @staticmethod
    def smart_load_checkpoint(checkpoint_path: str, model: nn.Module, 
                            device: torch.device = None, strict: bool = False) -> Dict:
        """
        智能加载检查点，自动处理编译状态差异
        
        Args:
            checkpoint_path: 检查点文件路径
            model: 目标模型
            device: 目标设备
            strict: 是否严格匹配键名
            
        Returns:
            检查点信息字典
        """
        if device is None:
            device = next(model.parameters()).device
            
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 检测目标模型是否为编译模型
        target_is_compiled = hasattr(model, '_orig_mod')
        
        # 修复状态字典键名
        fixed_state_dict = ModelManager.fix_state_dict_keys(state_dict, target_is_compiled)
        
        try:
            # 尝试加载
            model.load_state_dict(fixed_state_dict, strict=strict)
            print(f"✓ 模型加载成功: {checkpoint_path}")
            
        except RuntimeError as e:
            if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                print(f"⚠ 键名不匹配，尝试修复...")
                
                # 如果还是失败，尝试反向修复
                reverse_fixed = ModelManager.fix_state_dict_keys(state_dict, not target_is_compiled)
                try:
                    model.load_state_dict(reverse_fixed, strict=strict)
                    print(f"✓ 模型加载成功 (使用反向修复): {checkpoint_path}")
                except RuntimeError as e2:
                    if not strict:
                        # 尝试非严格加载
                        missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
                        print(f"⚠ 非严格模式加载成功")
                        if missing_keys:
                            print(f"  缺失键: {missing_keys[:5]}...")
                        if unexpected_keys:
                            print(f"  多余键: {unexpected_keys[:5]}...")
                    else:
                        raise e2
            else:
                raise e
        
        return checkpoint
    
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: torch.optim.Optimizer = None) -> Dict:
        """加载模型检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
        
        return checkpoint
    
    def load_best_model(self, model: nn.Module) -> Dict:
        """加载最佳模型"""
        best_path = self.checkpoint_dir / 'best_model.pth'
        return self.load_checkpoint(str(best_path), model)
    
    def list_checkpoints(self) -> List[str]:
        """列出所有检查点"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        return sorted([str(cp) for cp in checkpoints])

class TrainingMonitor:
    """训练过程监控和记录"""
    
    def __init__(self, log_dir: str = None):
        self.log_dir = Path(log_dir or config.CHECKPOINT_DIR / 'logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.start_time = None
        
    def start_training(self):
        """开始训练计时"""
        self.start_time = time.time()
        print("Training started...")
        
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float = None,
                  metrics: Dict = None):
        """记录每个epoch的指标"""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        
        if metrics:
            for key, value in metrics.items():
                self.metrics[key].append(value)
        
        # 打印进度
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}", end="")
        if val_loss is not None:
            print(f" | Val Loss: {val_loss:.4f}", end="")
        print(f" | Time: {elapsed/60:.1f}m")
        
    def save_metrics(self, filename: str = 'training_metrics.json'):
        """保存训练指标到文件"""
        metrics_path = self.log_dir / filename
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(dict(self.metrics), f, indent=2, ensure_ascii=False)
        
    def plot_training_curves(self, save_path: str = None) -> str:
        """绘制训练曲线"""
        if not save_path:
            save_path = str(self.log_dir / 'training_curves.png')
        
        plt.style.use(config.PLOT_STYLE)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hierarchical LSTM Poetry Generation Training', fontsize=16, fontweight='bold')
        
        epochs = self.metrics['epoch']
        
        # 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in self.metrics:
            ax1.plot(epochs, self.metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 学习率曲线
        ax2 = axes[0, 1]
        if 'learning_rate' in self.metrics:
            ax2.plot(epochs, self.metrics['learning_rate'], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Learning Rate\nNot Recorded', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 困惑度曲线
        ax3 = axes[1, 0]
        if 'perplexity' in self.metrics:
            ax3.plot(epochs, self.metrics['perplexity'], 'purple', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Perplexity')
            ax3.set_title('Model Perplexity')
            ax3.grid(True, alpha=0.3)
        else:
            # 从损失计算困惑度
            train_perplexity = [np.exp(loss) for loss in self.metrics['train_loss']]
            ax3.plot(epochs, train_perplexity, 'purple', linewidth=2, label='Train')
            if 'val_loss' in self.metrics:
                val_perplexity = [np.exp(loss) for loss in self.metrics['val_loss']]
                ax3.plot(epochs, val_perplexity, 'orange', linewidth=2, label='Validation')
                ax3.legend()
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Perplexity')
            ax3.set_title('Model Perplexity (exp(loss))')
            ax3.grid(True, alpha=0.3)
        
        # 梯度范数
        ax4 = axes[1, 1]
        if 'grad_norm' in self.metrics:
            ax4.plot(epochs, self.metrics['grad_norm'], 'brown', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Gradient Norm')
            ax4.set_title('Gradient Norm')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Gradient Norm\nNot Recorded', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=config.PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_path}")
        return save_path

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.min_delta *= 1 if mode == 'min' else -1
        
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.monitor_op(score, self.best_score - self.min_delta):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print("Restored best weights")
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """保存最佳权重"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

def create_optimizer(model, config_dict: Dict) -> torch.optim.Optimizer:
    """
    创建Transformer优化器（支持AdamW和学习率配置）
    
    Args:
        model: 模型
        config_dict: 配置字典
        
    Returns:
        优化器
    """
    
    optimizer_name = config_dict.get('optimizer', 'adamw')
    learning_rate = config_dict.get('learning_rate', 1e-4)
    weight_decay = config_dict.get('weight_decay', 0.01)
    
    # 区分权重衰减参数
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    if optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.95),  # Transformer常用设置
            eps=1e-8
        )
    elif optimizer_name.lower() == 'adam':
        return torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def create_scheduler(optimizer: torch.optim.Optimizer, config_dict: Dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    创建Transformer学习率调度器（支持预热和余弦退火）
    
    Args:
        optimizer: 优化器
        config_dict: 配置字典
        
    Returns:
        学习率调度器
    """
    
    scheduler_name = config_dict.get('scheduler', 'cosine_with_warmup')
    warmup_steps = config_dict.get('warmup_steps', 1000)
    total_steps = config_dict.get('total_steps', 50000)
    
    if scheduler_name == 'cosine_with_warmup':
        # 带预热的余弦退火调度器
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(
                (step + 1) / warmup_steps,  # 线性预热
                0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
                if step > warmup_steps else 1.0
            ),
            last_epoch=-1  # 避免scheduler警告
        )
    elif scheduler_name == 'linear_warmup':
        # 线性预热 + 线性衰减
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(
                (step + 1) / warmup_steps,
                max(0.0, (total_steps - step) / (total_steps - warmup_steps))
                if step > warmup_steps else 1.0
            ),
            last_epoch=-1  # 避免scheduler警告
        )
    elif scheduler_name == 'step':
        step_size = config_dict.get('step_size', 10)
        gamma = config_dict.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif scheduler_name == 'cosine':
        T_max = config_dict.get('T_max', 50)
        eta_min = config_dict.get('eta_min', 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
    elif scheduler_name == 'plateau':
        mode = config_dict.get('mode', 'min')
        factor = config_dict.get('factor', 0.5)
        patience = config_dict.get('patience', 5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def compute_gradient_norm(model: nn.Module) -> float:
    """计算模型梯度范数"""
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
    
    return total_norm

def clip_gradients(model: nn.Module, max_norm: float = None) -> float:
    """梯度裁剪"""
    max_norm = max_norm or config.GRADIENT_CLIP
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return grad_norm.item()

def save_checkpoint(checkpoint: Dict, is_best: bool = False, checkpoint_dir: str = "checkpoints"):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 保存最新检查点
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{checkpoint['epoch']}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, best_path)

def load_checkpoint(checkpoint_path: str) -> Dict:
    """加载检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """计算训练指标"""
    with torch.no_grad():
        # 准确率
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean().item()
        
        # 困惑度
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')
        perplexity = torch.exp(loss).item()
        
        return {
            'accuracy': accuracy,
            'perplexity': perplexity,
            'loss': loss.item()
        }

def setup_logging(log_file: str = None) -> logging.Logger:
    """设置日志"""
    # 创建日志目录
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.NullHandler(),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

if __name__ == "__main__":
    # 测试训练工具
    print("Testing training utilities...")
    
    # 测试损失函数
    vocab_size = 1000
    batch_size = 4
    seq_len = 20
    
    loss_fn = SimplifiedPoetryLoss(vocab_size)
    
    # 模拟模型输出
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = loss_fn(logits, targets)
    print(f"Loss computation successful: {loss.item():.4f}")
    
    # 测试训练监控
    monitor = TrainingMonitor()
    monitor.start_training()
    
    # 模拟训练数据
    for epoch in range(5):
        train_loss = 3.0 - epoch * 0.5 + np.random.normal(0, 0.1)
        val_loss = 3.2 - epoch * 0.4 + np.random.normal(0, 0.1)
        monitor.log_epoch(epoch + 1, train_loss, val_loss)
    
    # 生成图表
    plot_path = monitor.plot_training_curves()
    print(f"Training monitoring test completed!")
    
    print("Training utilities testing completed!") 