#!/usr/bin/env python3
"""
模型翻译效果测试脚本 - 中译英版本
用于快速检验中译英模型的翻译质量
"""

import torch
import numpy as np
import json
import random
import argparse
import os
import sys
from datetime import datetime

# 添加模型路径
sys.path.append('../model')

import config
from model.transformer import make_model, batch_greedy_decode
from utils import chinese_tokenizer_load, english_tokenizer_load
from data_loader import MTDataset


class TranslationTester:
    """翻译测试器 - 中译英版本"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or config.model_path
        self.device = config.device
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        print(f"🔄 正在加载模型: {self.model_path}")
        
        # 加载分词器（中译英：源语言中文，目标语言英文）
        self.cn_tokenizer = chinese_tokenizer_load()
        self.en_tokenizer = english_tokenizer_load()
        
        # 特殊token（使用中文分词器，因为源语言是中文）
        self.BOS = self.cn_tokenizer.bos_id()  # 2
        self.EOS = self.cn_tokenizer.eos_id()  # 3
        self.PAD = self.cn_tokenizer.pad_id()  # 0
        
        # 创建并加载模型
        self.model = make_model(
            config.src_vocab_size, config.tgt_vocab_size,
            config.n_layers, config.d_model, config.d_ff,
            config.n_heads, config.dropout
        )
        
        # 加载训练好的参数
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✅ 模型加载成功! 参数量: {sum(p.numel() for p in self.model.parameters()) / 1000000.0:.1f}M")
    
    def translate_sentence(self, chinese_text, max_len=60):
        """翻译单个中文句子"""
        with torch.no_grad():
            # 分词并添加特殊token（使用中文分词器）
            src_tokens = [self.BOS] + self.cn_tokenizer.EncodeAsIds(chinese_text) + [self.EOS]
            src_tensor = torch.LongTensor([src_tokens]).to(self.device)
            src_mask = (src_tensor != self.PAD).unsqueeze(-2)
            
            # 贪心解码
            result = batch_greedy_decode(self.model, src_tensor, src_mask, max_len=max_len)
            
            # 解码为英文文本
            if result and len(result) > 0:
                english_text = self.en_tokenizer.decode_ids(result[0])
                return english_text
            else:
                return "翻译失败"
    
    def translate_batch(self, chinese_texts, max_len=60):
        """批量翻译中文句子"""
        results = []
        
        with torch.no_grad():
            # 准备批量数据
            batch_tokens = []
            for text in chinese_texts:
                src_tokens = [self.BOS] + self.cn_tokenizer.EncodeAsIds(text) + [self.EOS]
                batch_tokens.append(src_tokens)
            
            # 填充到相同长度
            max_src_len = max(len(tokens) for tokens in batch_tokens)
            padded_batch = []
            for tokens in batch_tokens:
                padded_tokens = tokens + [self.PAD] * (max_src_len - len(tokens))
                padded_batch.append(padded_tokens)
            
            # 转换为tensor
            src_tensor = torch.LongTensor(padded_batch).to(self.device)
            src_mask = (src_tensor != self.PAD).unsqueeze(-2)
            
            # 批量解码
            decode_results = batch_greedy_decode(self.model, src_tensor, src_mask, max_len=max_len)
            
            # 转换为英文文本
            for result in decode_results:
                if result:
                    english_text = self.en_tokenizer.decode_ids(result)
                    results.append(english_text)
                else:
                    results.append("翻译失败")
        
        return results
    
    def test_sample_sentences(self):
        """测试一些样例句子"""
        test_samples = [
            {
                "chinese": "你好，你好吗？",
                "reference": "Hello, how are you?"
            },
            {
                "chinese": "我喜欢人工智能和机器学习。",
                "reference": "I love artificial intelligence and machine learning."
            },
            {
                "chinese": "今天天气很好。",
                "reference": "The weather is very nice today."
            },
            {
                "chinese": "你能帮我解决这个问题吗？",
                "reference": "Can you help me with this problem?"
            },
            {
                "chinese": "技术正在迅速改变我们的世界。",
                "reference": "Technology is changing our world rapidly."
            },
            {
                "chinese": "近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平。",
                "reference": "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a fully employed worker and his or her family out of poverty."
            }
        ]
        
        print("=" * 80)
        print("🔍 样例句子翻译测试")
        print("=" * 80)
        
        for i, sample in enumerate(test_samples, 1):
            chinese = sample["chinese"]
            reference = sample["reference"]
            
            print(f"\n【样例 {i}】")
            print(f"原文: {chinese}")
            print(f"参考: {reference}")
            
            # 翻译
            translation = self.translate_sentence(chinese)
            print(f"译文: {translation}")
            
            # 简单质量评估
            if translation and translation != "翻译失败":
                print("✅ 翻译成功")
            else:
                print("❌ 翻译失败")
            
            print("-" * 60)
    
    def test_dataset_samples(self, num_samples=10):
        """从测试数据集中随机选择样本进行测试"""
        print("=" * 80)
        print(f"📊 数据集随机样本测试 (共{num_samples}个样本)")
        print("=" * 80)
        
        try:
            # 加载测试数据集
            test_dataset = MTDataset(config.test_data_path)
            
            # 随机选择样本
            indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
            
            success_count = 0
            
            for i, idx in enumerate(indices, 1):
                sample = test_dataset[idx]
                chinese = sample[0]  # 中译英：中文是源语言
                reference = sample[1]  # 英文是目标语言
                
                print(f"\n【测试样本 {i}】(数据集索引: {idx})")
                print(f"原文: {chinese}")
                print(f"参考: {reference}")
                
                # 翻译
                translation = self.translate_sentence(chinese)
                print(f"译文: {translation}")
                
                if translation and translation != "翻译失败":
                    success_count += 1
                    print("✅ 翻译成功")
                else:
                    print("❌ 翻译失败")
                
                print("-" * 60)
            
            print(f"\n📈 成功率: {success_count}/{num_samples} ({success_count/num_samples*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 数据集测试失败: {e}")
    
    def interactive_translation(self):
        """交互式翻译模式"""
        print("=" * 80)
        print("🗣️  交互式翻译模式")
        print("输入中文句子进行翻译，输入 'quit' 或 'exit' 退出")
        print("=" * 80)
        
        while True:
            try:
                chinese_input = input("\n请输入中文句子: ").strip()
                
                if chinese_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not chinese_input:
                    print("⚠️  请输入有效的中文句子")
                    continue
                
                # 翻译
                translation = self.translate_sentence(chinese_input)
                print(f"翻译结果: {translation}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 翻译出错: {e}")
    
    def save_test_results(self, results, filename=None):
        """保存测试结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"translation_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"📝 测试结果已保存到: {filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中译英模型翻译测试")
    parser.add_argument('--samples', action='store_true', help='测试预设样例')
    parser.add_argument('--dataset', type=int, default=10, help='数据集随机样本数量')
    parser.add_argument('--interactive', action='store_true', help='交互式翻译模式')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--sentence', type=str, help='直接翻译单个句子')
    
    args = parser.parse_args()
    
    try:
        # 创建测试器
        tester = TranslationTester(model_path=args.model)
        
        # 单句翻译
        if args.sentence:
            print(f"原文: {args.sentence}")
            translation = tester.translate_sentence(args.sentence)
            print(f"译文: {translation}")
            return
        
        # 交互式模式
        if args.interactive:
            tester.interactive_translation()
            return
        
        # 默认运行所有测试
        if not any([args.samples, args.dataset]):
            args.samples = True
            args.dataset = 10
        
        # 样例测试
        if args.samples:
            tester.test_sample_sentences()
        
        # 数据集测试
        if args.dataset > 0:
            tester.test_dataset_samples(args.dataset)
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main() 