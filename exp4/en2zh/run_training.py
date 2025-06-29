#!/usr/bin/env python3
"""
英译中Transformer模型训练启动脚本
第二阶段：英译中模型完整实现
"""

import os
import time
import warnings
import torch

# 环境设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用V100S单卡
warnings.filterwarnings('ignore')

def check_environment():
    """检查训练环境"""
    print("========== 环境检查 ==========")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print("环境检查完成！\n")

def check_data():
    """检查数据准备情况"""
    print("========== 数据检查 ==========")
    import config
    
    datasets = [
        ("训练集", config.train_data_path),
        ("验证集", config.dev_data_path), 
        ("测试集", config.test_data_path)
    ]
    
    for name, path in datasets:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)
            print(f"✅ {name}: {path} ({size:.1f}MB)")
        else:
            print(f"❌ {name}: {path} 文件不存在")
    
    print("数据检查完成！\n")

def check_tokenizers():
    """检查分词器"""
    print("========== 分词器检查 ==========")
    try:
        from utils import english_tokenizer_load, chinese_tokenizer_load
        
        en_tokenizer = english_tokenizer_load()
        cn_tokenizer = chinese_tokenizer_load()
        
        print(f"✅ 英文分词器: {en_tokenizer.vocab_size()} 词汇")
        print(f"✅ 中文分词器: {cn_tokenizer.vocab_size()} 词汇")
        print("分词器检查完成！\n")
        
        return True
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        return False

def run_quick_test():
    """运行快速功能测试"""
    print("========== 快速功能测试 ==========")
    try:
        from data_loader import MTDataset
        from model import make_model
        import config
        
        # 测试数据加载
        print("测试数据加载...")
        train_dataset = MTDataset(config.train_data_path)
        print(f"✅ 训练集加载成功: {len(train_dataset)} 样本")
        
        # 测试模型创建（使用完整配置）
        print("测试模型创建...")
        model = make_model(
            config.src_vocab_size, config.tgt_vocab_size, 
            config.n_layers, config.d_model, config.d_ff, 
            config.n_heads, config.dropout
        )
        param_count = sum(p.numel() for p in model.parameters()) / 1000000.0
        print(f"✅ 模型创建成功: {param_count:.1f}M 参数")
        
        print("快速测试通过！\n")
        return True
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        return False

def start_training():
    """启动训练"""
    print("========== 开始正式训练 ==========")
    print("训练配置:")
    import config
    
    config_items = [
        ("批大小", config.batch_size),
        ("学习率调度", "NoamOpt" if config.use_noamopt else f"固定{config.lr}"),
        ("最大轮数", config.epoch_num),
        ("早停轮数", config.early_stop),
        ("模型层数", config.n_layers),
        ("隐藏维度", config.d_model),
        ("注意力头数", config.n_heads),
    ]
    
    for name, value in config_items:
        print(f"  {name}: {value}")
    
    print(f"\n模型将保存到: {config.model_path}")
    print(f"日志将保存到: {config.log_path}")
    print(f"测试结果将保存到: {config.output_path}\n")
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 导入并运行主程序
        from main import run
        run()
        
        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time
        hours = training_time // 3600
        minutes = (training_time % 3600) // 60
        
        print(f"\n========== 训练完成 ==========")
        print(f"总训练时间: {hours:.0f}小时 {minutes:.0f}分钟")
        print(f"模型保存位置: {config.model_path}")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("=" * 50)
    print("基于Transformer的英译中神经机器翻译")
    print("第二阶段：英译中模型完整实现")
    print("=" * 50)
    
    # 环境检查
    check_environment()
    
    # 数据检查
    check_data()
    
    # 分词器检查
    if not check_tokenizers():
        print("❌ 分词器检查失败，退出训练")
        return
    
    # 快速功能测试
    if not run_quick_test():
        print("❌ 快速测试失败，退出训练")
        return
    
    # 确认开始训练
    print("所有检查通过！准备开始训练...")
    print("注意：训练可能需要几个小时，请确保:")
    print("  1. GPU内存充足")
    print("  2. 网络连接稳定")
    print("  3. 有足够的磁盘空间保存模型")
    
    # 开始训练
    start_training()

if __name__ == "__main__":
    main() 