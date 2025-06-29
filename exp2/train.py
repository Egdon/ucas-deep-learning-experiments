#!/usr/bin/env python3
"""
ViT-CIFAR10 训练脚本
基于ViT-Small/8配置，用于CIFAR-10图像分类（V100S 32GB优化版）
"""

import os
import sys
import torch
import random
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import create_vit_small_patch8_224
from data.dataset import create_data_loaders
from utils import Config, Logger, Trainer

def set_seed(seed=42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_gpu():
    """检查GPU可用性"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU可用: {gpu_name} (共{gpu_count}块)")
        print(f"CUDA版本: {torch.version.cuda}")
        
        # 显示显存信息
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"显存使用: {memory_allocated:.2f}GB / {memory_cached:.2f}GB")
        return True
    else:
        print("GPU不可用，将使用CPU训练")
        return False

def count_parameters(model):
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024**2:.2f}MB")
    
    return total_params, trainable_params

def main():
    """主训练函数"""
    print("=" * 80)
    print("ViT-CIFAR10 训练开始")
    print("=" * 80)
    
    # 设置随机种子
    set_seed(42)
    
    # 检查GPU
    gpu_available = check_gpu()
    device = torch.device('cuda' if gpu_available else 'cpu')
    
    # 打印配置信息
    Config.print_config()
    
    # 获取配置
    model_config = Config.get_model_config()
    training_config = Config.get_training_config()
    data_config = Config.get_data_config()
    
    # 更新配置中的设备信息
    training_config['checkpoint_dir'] = Config.LOGGING['checkpoint_dir']
    training_config['log_frequency'] = Config.LOGGING['log_frequency']
    training_config['checkpoint_frequency'] = Config.LOGGING['checkpoint_frequency']
    
    print("\n" + "=" * 50)
    print("加载数据集...")
    print("=" * 50)
    
    # 创建数据加载器
    (train_loader, val_loader), dataset_info = create_data_loaders(
        data_root=data_config['data_root'],
        batch_size=training_config['batch_size'],
        num_workers=data_config['num_workers']
    )
    
    print(f"数据集信息:")
    print(f"  训练样本: {dataset_info['train_samples']:,}")
    print(f"  验证样本: {dataset_info['val_samples']:,}")
    print(f"  类别数量: {dataset_info['num_classes']}")
    print(f"  批大小: {training_config['batch_size']}")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    
    print("\n" + "=" * 50)
    print("创建模型...")
    print("=" * 50)
    
    # 创建模型
    model = create_vit_small_patch8_224(
        num_classes=model_config['num_classes']
    )
    
    # 统计参数
    total_params, trainable_params = count_parameters(model)
    
    # 模型配置信息
    print(f"模型配置:")
    print(f"  图像尺寸: {model_config['image_size']}×{model_config['image_size']}")
    print(f"  Patch尺寸: {model_config['patch_size']}×{model_config['patch_size']}")
    print(f"  Patch数量: {(model_config['image_size']//model_config['patch_size'])**2}")
    print(f"  嵌入维度: {model_config['dim']}")
    print(f"  编码器层数: {model_config['depth']}")
    print(f"  注意力头数: {model_config['heads']}")
    
    print("\n" + "=" * 50)
    print("初始化训练器...")
    print("=" * 50)
    
    # 创建日志记录器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"train_log_{timestamp}.txt"
    metrics_file = f"metrics_{timestamp}.json"
    
    logger = Logger(
        log_dir=Config.LOGGING['log_dir'],
        log_file=log_file,
        metrics_file=metrics_file
    )
    
    # 记录配置到日志
    logger.log_message("训练配置:")
    for key, value in training_config.items():
        logger.log_message(f"  {key}: {value}")
    
    logger.log_message(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
    logger.log_message(f"设备: {device}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        logger=logger,
        device=device
    )
    
    print("训练器初始化完成")
    print(f"日志文件: {os.path.join(Config.LOGGING['log_dir'], log_file)}")
    print(f"指标文件: {os.path.join(Config.LOGGING['log_dir'], metrics_file)}")
    
    try:
        print("\n" + "=" * 50)
        print("开始训练...")
        print("=" * 50)
        
        # 开始训练
        best_val_acc = trainer.train()
        
        print("\n" + "=" * 50)
        print("训练完成")
        print("=" * 50)
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        
        if best_val_acc >= 80.0:
            print("恭喜！达到了80%的目标准确率！")
        else:
            print("未达到80%的目标准确率，可能需要调整超参数")
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        logger.log_message("训练被用户中断")
    
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        logger.log_message(f"训练错误: {e}")
        raise
    
    finally:
        print("\n清理资源...")
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 