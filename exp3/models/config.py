#!/usr/bin/env python3
"""
Configuration management for poetry generation model.
Global parameter configuration with environment adaptation support.
"""

import torch
import os
from pathlib import Path

# 在类外部定义环境检测函数
def _detect_env():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_name = torch.cuda.get_device_name(0)
        
        # V100s服务器环境检测
        if gpu_memory > 30e9 or 'V100' in gpu_name:
            return 'server'
        # RTX开发环境
        elif 'RTX' in gpu_name:
            return 'development'
        else:
            return 'general'
    else:
        return 'cpu'

class Config:
    """Global configuration class for the poetry generation project."""
    
    # 项目路径配置
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    VISUALIZATION_DIR = PROJECT_ROOT / "visualization" / "export"
    
    # 数据集配置
    DATA_PATH = str(DATA_DIR / "tang.npz")
    
    # 设备和环境自动检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # PyTorch版本检测
    TORCH_VERSION = tuple(map(int, torch.__version__.split('.')[:2]))
    PYTORCH_2_AVAILABLE = TORCH_VERSION >= (2, 0)
    
    ENVIRONMENT = _detect_env()
    
    # 基础模型超参数
    VOCAB_SIZE = 8293  # 从数据分析得出
    SEQUENCE_LENGTH = 125
    
    # Transformer架构配置 - 50M参数轻量方案
    if ENVIRONMENT == 'server':
        # 服务器环境：更大规模
        HIDDEN_SIZE = 576
        NUM_LAYERS = 12
        NUM_HEADS = 9
        FEEDFORWARD_DIM = 2304
    elif ENVIRONMENT == 'development':
        # 开发环境：50M参数轻量配置
        HIDDEN_SIZE = 384
        NUM_LAYERS = 8
        NUM_HEADS = 6
        FEEDFORWARD_DIM = 1536
    else:
        # 通用环境：最小配置
        HIDDEN_SIZE = 256
        NUM_LAYERS = 6
        NUM_HEADS = 4
        FEEDFORWARD_DIM = 1024
    
    # 通用Transformer参数
    MAX_SEQ_LEN = 125
    DROPOUT = 0.1
    
    # 通用训练配置
    if ENVIRONMENT == 'server':
        BATCH_SIZE = 128
        LEARNING_RATE = 0.0008  # 为57.5M模型降低学习率
        NUM_EPOCHS = 100  # 增加训练轮数以充分利用服务器性能
        NUM_WORKERS = 4  # 减少工作进程，避免过度上下文切换
    elif ENVIRONMENT == 'development':
        BATCH_SIZE = 32
        LEARNING_RATE = 0.002
        NUM_EPOCHS = 20
        NUM_WORKERS = 2
    else:
        BATCH_SIZE = 16
        LEARNING_RATE = 0.001
        NUM_EPOCHS = 10
        NUM_WORKERS = 0
    
    # 优化器配置
    OPTIMIZER = 'adam'
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP = 5.0
    
    # 学习率调度
    LR_SCHEDULER = 'cosine'
    LR_WARMUP_EPOCHS = 3
    LR_MIN = 1e-6
    
    # 模型保存和验证
    SAVE_EVERY = 5  # epochs
    VALIDATE_EVERY = 1  # epochs
    EARLY_STOPPING_PATIENCE = 20  # 增加早停patience
    
    # 生成配置
    MAX_GENERATE_LENGTH = 100
    TEMPERATURE = 0.8
    TOP_K = 50
    TOP_P = 0.9
    
    # 特殊标记
    START_TOKEN = '<START>'
    END_TOKEN = '<EOP>'
    PADDING_TOKEN = '</s>'
    
    # 模式标记（为兼容性保留）
    ACROSTIC_MODE_TOKEN = '<ACROSTIC>'
    CONTINUE_MODE_TOKEN = '<CONTINUE>'
    
    # PyTorch 2.0.1特性配置 - 默认关闭编译，需要明确启用
    ENABLE_COMPILE = False  # 默认关闭，避免兼容性问题
    COMPILE_MODE = 'default' if ENABLE_COMPILE else None
    
    # 混合精度训练
    ENABLE_AMP = torch.cuda.is_available()
    
    # 可视化配置
    PLOT_DPI = 300
    PLOT_STYLE = 'seaborn-v0_8-whitegrid'
    PLOT_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.CHECKPOINT_DIR,
            cls.VISUALIZATION_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_config(cls):
        """获取模型配置字典"""
        return {
            'vocab_size': cls.VOCAB_SIZE,
            'hidden_size': cls.HIDDEN_SIZE,
            'num_layers': cls.NUM_LAYERS,
            'num_heads': cls.NUM_HEADS,
            'feedforward_dim': cls.FEEDFORWARD_DIM,
            'max_seq_len': cls.MAX_SEQ_LEN,
            'dropout': cls.DROPOUT,
        }
    
    @classmethod
    def get_training_config(cls):
        """获取训练配置字典"""
        return {
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'num_epochs': cls.NUM_EPOCHS,
            'num_workers': cls.NUM_WORKERS,
            'weight_decay': cls.WEIGHT_DECAY,
            'gradient_clip': cls.GRADIENT_CLIP,
        }
    
    @classmethod
    def print_config(cls):
        """打印当前配置信息"""
        print("=" * 60)
        print("POETRY GENERATION MODEL CONFIG")
        print("=" * 60)
        print(f"Environment: {cls.ENVIRONMENT}")
        print(f"Device: {cls.device}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"PyTorch 2.0+ Features: {cls.PYTORCH_2_AVAILABLE}")
        print(f"Model Compile: {cls.ENABLE_COMPILE}")
        print(f"Mixed Precision: {cls.ENABLE_AMP}")
        print()
        print("Model Configuration:")
        for key, value in cls.get_model_config().items():
            print(f"  {key}: {value}")
        print()
        print("Training Configuration:")
        for key, value in cls.get_training_config().items():
            print(f"  {key}: {value}")
        print("=" * 60)

# 全局配置实例
config = Config()

if __name__ == "__main__":
    # 配置测试
    config.create_directories()
    config.print_config() 