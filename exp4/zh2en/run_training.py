#!/usr/bin/env python3
"""
中译英模型训练运行脚本
包含环境检查、GPU信息显示和模型训练启动
"""

import torch
import os
import sys
import warnings

# 添加模型路径
sys.path.append('../model')

import config
import main as main_module
from utils import chinese_tokenizer_load, english_tokenizer_load


def check_environment():
    """检查运行环境"""
    print("🔍 环境检查")
    print("=" * 50)
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # PyTorch版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA信息
    if torch.cuda.is_available():
        print(f"CUDA可用: ✅")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("CUDA可用: ❌")
    
    print(f"使用设备: {config.device}")
    print()


def check_data():
    """检查数据文件"""
    print("📊 数据检查")
    print("=" * 50)
    
    data_files = [
        ("训练集", config.train_data_path),
        ("验证集", config.dev_data_path), 
        ("测试集", config.test_data_path)
    ]
    
    for name, path in data_files:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024  # MB
            print(f"{name}: ✅ ({size:.1f}MB) - {path}")
        else:
            print(f"{name}: ❌ 文件不存在 - {path}")
            return False
    
    print()
    return True


def check_tokenizers():
    """检查分词器"""
    print("🔤 分词器检查")
    print("=" * 50)
    
    try:
        # 中文分词器（源语言）
        cn_tokenizer = chinese_tokenizer_load()
        cn_vocab_size = len(cn_tokenizer)
        print(f"中文分词器: ✅ (词汇表大小: {cn_vocab_size})")
        
        # 英文分词器（目标语言）
        en_tokenizer = english_tokenizer_load()
        en_vocab_size = len(en_tokenizer)
        print(f"英文分词器: ✅ (词汇表大小: {en_vocab_size})")
        
        # 测试分词
        test_cn = "你好，世界！"
        test_en = "Hello, world!"
        
        cn_tokens = cn_tokenizer.EncodeAsIds(test_cn)
        en_tokens = en_tokenizer.EncodeAsIds(test_en)
        
        print(f"测试分词 - 中文: '{test_cn}' -> {cn_tokens}")
        print(f"测试分词 - 英文: '{test_en}' -> {en_tokens}")
        
        print()
        return True
        
    except Exception as e:
        print(f"分词器加载失败: {e}")
        return False


def check_model_config():
    """检查模型配置"""
    print("⚙️  模型配置")
    print("=" * 50)
    
    print(f"模型层数: {config.n_layers}")
    print(f"模型维度: {config.d_model}")
    print(f"注意力头数: {config.n_heads}")
    print(f"前馈网络维度: {config.d_ff}")
    print(f"Dropout: {config.dropout}")
    print(f"源词汇表大小: {config.src_vocab_size}")
    print(f"目标词汇表大小: {config.tgt_vocab_size}")
    print(f"批大小: {config.batch_size}")
    print(f"最大轮数: {config.epoch_num}")
    print(f"早停轮数: {config.early_stop}")
    print(f"使用NoamOpt: {config.use_noamopt}")
    print(f"使用标签平滑: {config.use_smoothing}")
    print()


def load_dataset_info():
    """加载数据集信息"""
    print("📈 数据集信息")
    print("=" * 50)
    
    try:
        from data_loader import MTDataset
        
        # 创建数据集
        train_dataset = MTDataset(config.train_data_path)
        dev_dataset = MTDataset(config.dev_data_path)
        test_dataset = MTDataset(config.test_data_path)
        
        print(f"训练样本数: {len(train_dataset):,}")
        print(f"验证样本数: {len(dev_dataset):,}")
        print(f"测试样本数: {len(test_dataset):,}")
        
        # 显示几个样本
        print("\n样本示例:")
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            print(f"  样本{i+1}:")
            print(f"    中文: {sample[0][:50]}{'...' if len(sample[0]) > 50 else ''}")
            print(f"    英文: {sample[1][:50]}{'...' if len(sample[1]) > 50 else ''}")
        
        print()
        return True
        
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return False


def estimate_model_size():
    """估算模型大小"""
    print("🧮 模型大小估算")
    print("=" * 50)
    
    try:
        from model.transformer import make_model
        
        # 创建模型
        model = make_model(
            config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
            config.d_model, config.d_ff, config.n_heads, config.dropout
        )
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
        print(f"总参数量: {total_params:,} ({total_params/1000000:.2f}M)")
        print(f"可训练参数量: {trainable_params:,} ({trainable_params/1000000:.2f}M)")
    
        # 估算显存占用（粗略估算）
        # 参数 + 梯度 + 优化器状态 + 激活值
        param_memory = total_params * 4 / 1024 / 1024  # MB (float32)
        grad_memory = param_memory  # 梯度
        optimizer_memory = param_memory * 2  # Adam状态
        activation_memory = config.batch_size * 512 * config.d_model * 4 / 1024 / 1024  # 粗略估算
        
        total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
        
        print(f"估计显存占用: {total_memory:.1f}MB ({total_memory/1024:.2f}GB)")
        print()
        
        return True
        
    except Exception as e:
        print(f"模型创建失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 中译英模型训练启动")
    print("=" * 70)
    print()
    
    # 抑制警告
    warnings.filterwarnings('ignore')
    
    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 环境检查
    check_environment()
    
    # 数据检查
    if not check_data():
        print("❌ 数据检查失败，请检查数据文件")
        return
    
    # 分词器检查
    if not check_tokenizers():
        print("❌ 分词器检查失败")
        return
    
    # 模型配置
    check_model_config()
    
    # 数据集信息
    if not load_dataset_info():
        print("❌ 数据集加载失败")
        return
    
    # 模型大小估算
    if not estimate_model_size():
        print("❌ 模型创建失败")
        return
    
    print("✅ 所有检查通过，开始训练...")
    print("=" * 70)
    print()
    
    try:
        # 启动训练
        main_module.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 