import torch

class Config:
    """训练配置参数"""
    
    # 模型配置 - ViT-Small/8 (V100S 32GB优化版本)
    MODEL = {
        'image_size': 224,
        'patch_size': 8,   # 回到8，获得最佳细节捕获 (784 patches)
        'num_classes': 10,
        'dim': 384,        # 恢复到384维
        'depth': 12,       # 恢复到12层
        'heads': 6,        # 恢复到6个head
        'mlp_dim': 1536,   # 恢复到1536
        'dropout': 0.1,
        'emb_dropout': 0.1,
        'dim_head': 64
    }
    
    # 训练配置
    TRAINING = {
        'batch_size': 64,  # V100S 32GB可以支持更大batch size
        'epochs': 20,      # 减少训练轮数，专注精细调优
        'warmup_epochs': 0, # 不需要预热
        'learning_rate': 1e-6,  # 大幅降低学习率
        'weight_decay': 2e-4,   # 增加正则化
        'label_smoothing': 0.2, # 增加标签平滑
        'grad_clip_norm': 0.5,  # 更小的梯度裁剪
        'early_stopping_patience': 8,  # 减少耐心值
        'use_mixed_precision': True,  # 保留混合精度以提升速度
        'gradient_checkpointing': False  # 关闭梯度检查点，换取训练速度
    }
    
    # 数据配置
    DATA = {
        'data_root': 'data/CIFAR-10',
        'num_workers': 12,  # V100S服务器增加数据加载线程
        'pin_memory': True
    }
    
    # 日志和保存配置
    LOGGING = {
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'log_frequency': 200,  # 减少日志频率，避免干扰进度条
        'checkpoint_frequency': 10,  # 每10个epoch保存一次
        'save_best_only': True
    }
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def get_model_config(cls):
        """获取模型配置"""
        return cls.MODEL.copy()
    
    @classmethod
    def get_training_config(cls):
        """获取训练配置"""
        return cls.TRAINING.copy()
    
    @classmethod
    def get_data_config(cls):
        """获取数据配置"""
        return cls.DATA.copy()
    
    @classmethod
    def print_config(cls):
        """打印所有配置"""
        print("=" * 50)
        print("Configuration:")
        print("=" * 50)
        
        print("\nModel Config:")
        for key, value in cls.MODEL.items():
            print(f"  {key}: {value}")
        
        print("\nTraining Config:")
        for key, value in cls.TRAINING.items():
            print(f"  {key}: {value}")
        
        print("\nData Config:")
        for key, value in cls.DATA.items():
            print(f"  {key}: {value}")
        
        print(f"\nDevice: {cls.DEVICE}")
        print("=" * 50) 