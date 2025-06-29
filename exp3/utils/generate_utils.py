#!/usr/bin/env python3
"""
Generation utilities for poetry generation.
Includes sampling strategies, text processing and quality assessment.
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Union
import random
from collections import Counter

from models import config

class TextSampler:
    """文本采样策略集合"""
    
    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        """贪婪采样 - 选择概率最高的token"""
        return torch.argmax(logits, dim=-1)
    
    @staticmethod
    def random_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """随机采样"""
        if temperature <= 0:
            return TextSampler.greedy_sample(logits)
        
        # 应用温度
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    @staticmethod
    def top_k_sample(logits: torch.Tensor, k: int, temperature: float = 1.0) -> torch.Tensor:
        """Top-K采样"""
        if k <= 0:
            return TextSampler.random_sample(logits, temperature)
        
        # 获取top-k logits
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # 应用温度
        scaled_logits = top_k_logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # 采样
        sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # 映射回原始词汇表
        return top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    
    @staticmethod
    def nucleus_sample(logits: torch.Tensor, p: float, temperature: float = 1.0) -> torch.Tensor:
        """Nucleus (Top-P) 采样"""
        if p <= 0 or p >= 1:
            return TextSampler.random_sample(logits, temperature)
        
        # 应用温度
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        
        # 排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过p的位置
        sorted_indices_to_remove = cumulative_probs > p
        
        # 保留第一个超过阈值的token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 将要移除的token概率设为0
        sorted_probs[sorted_indices_to_remove] = 0
        
        # 重新归一化
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # 采样
        sampled_indices = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
        
        # 映射回原始索引
        return sorted_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
    
    @staticmethod
    def combined_sample(logits: torch.Tensor, top_k: int = 50, top_p: float = 0.9,
                       temperature: float = 0.8) -> torch.Tensor:
        """组合采样策略：Top-K + Top-P"""
        # 先应用Top-K
        if top_k > 0 and top_k < logits.size(-1):
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits.scatter_(-1, top_k_indices, top_k_logits)
        else:
            filtered_logits = logits
        
        # 再应用Top-P
        return TextSampler.nucleus_sample(filtered_logits, top_p, temperature)

class TextProcessor:
    """文本处理和格式化工具"""
    
    def __init__(self, ix2word: Dict[int, str], word2ix: Dict[str, int]):
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.special_tokens = {
            config.START_TOKEN, config.END_TOKEN, config.PADDING_TOKEN,
            config.ACROSTIC_MODE_TOKEN, config.CONTINUE_MODE_TOKEN
        }
    
    def indices_to_text(self, indices: Union[torch.Tensor, List[int]], 
                       remove_special: bool = True) -> str:
        """将索引序列转换为文本"""
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy().tolist()
        
        text = ""
        for idx in indices:
            if idx in self.ix2word:
                char = self.ix2word[idx]
                if char == config.PADDING_TOKEN:
                    break  # 遇到填充符停止
                elif remove_special and char in self.special_tokens:
                    continue  # 跳过特殊标记
                else:
                    text += char
        
        return text
    
    def text_to_indices(self, text: str, add_special: bool = False) -> List[int]:
        """将文本转换为索引序列"""
        indices = []
        
        if add_special:
            start_idx = self.word2ix.get(config.START_TOKEN)
            if start_idx is not None:
                indices.append(start_idx)
        
        for char in text:
            if char in self.word2ix:
                indices.append(self.word2ix[char])
            # 跳过未知字符
        
        if add_special:
            end_idx = self.word2ix.get(config.END_TOKEN)
            if end_idx is not None:
                indices.append(end_idx)
        
        return indices
    
    def format_poem(self, text: str, poem_type: str = "续写") -> str:
        """格式化诗歌输出"""
        # 基础清理
        text = self.clean_text(text)
        
        # 按句号分割
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        
        if not sentences:
            return text
        
        # 格式化输出
        formatted = f"【{poem_type}诗】\n"
        formatted += "=" * 30 + "\n"
        
        for i, sentence in enumerate(sentences, 1):
            if sentence:
                formatted += f"{sentence}。\n"
        
        formatted += "=" * 30
        return formatted
    
    def clean_text(self, text: str) -> str:
        """清理生成的文本"""
        # 移除连续的重复字符
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 移除异常的标点组合
        text = re.sub(r'[。，、；：！？]{2,}', '。', text)
        
        # 确保句子以句号结尾
        if text and not text.endswith(('。', '！', '？')):
            text += '。'
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """提取句子列表"""
        # 按句号、感叹号、问号分割
        sentences = re.split(r'[。！？]', text)
        return [s.strip() for s in sentences if s.strip()]

class SimplePoetryGenerator:
    """简化的诗歌生成器"""
    
    def __init__(self, model, ix2word: Dict[int, str], word2ix: Dict[str, int]):
        self.model = model
        self.text_processor = TextProcessor(ix2word, word2ix)
        self.sampler = TextSampler()
        self.device = next(model.parameters()).device
        self.period_id = word2ix.get('。', -1)
    
    def generate(self, prompt: str = "", max_length: int = 100,
                temperature: float = 0.8, top_k: int = 50, 
                top_p: float = 0.9, num_samples: int = 1) -> List[str]:
        """基础文本生成"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # 转换提示文本为索引
                if prompt:
                    input_ids = self.text_processor.text_to_indices(prompt, add_special=True)
                else:
                    # 使用START token开始
                    start_token = self.text_processor.word2ix.get(config.START_TOKEN, 0)
                    input_ids = [start_token]
                
                input_tensor = torch.tensor([input_ids], device=self.device)
                
                # 生成
                for _ in range(max_length):
                    # 计算韵律位置（简化版本，用于生成）
                    char_positions = self._compute_char_positions(input_tensor)
                    
                    outputs = self.model(input_tensor, char_positions=char_positions)
                    
                    # 获取最后一个位置的logits
                    if isinstance(outputs, dict):
                        logits = outputs['logits'][:, -1, :]
                    else:
                        logits = outputs[:, -1, :]
                    
                    # 采样下一个token
                    next_token = self.sampler.combined_sample(
                        logits, top_k=top_k, top_p=top_p, temperature=temperature
                    )
                    
                    # 添加到序列，确保维度匹配
                    if next_token.dim() == 0:  # 标量
                        next_token = next_token.unsqueeze(0).unsqueeze(0)  # [1, 1]
                    elif next_token.dim() == 1:  # [batch_size]
                        next_token = next_token.unsqueeze(1)  # [batch_size, 1]
                    
                    input_tensor = torch.cat([input_tensor, next_token], dim=1)
                    
                    # 检查是否结束
                    if next_token.item() == self.text_processor.word2ix.get(config.END_TOKEN, -1):
                        break
                
                # 转换为文本
                generated_ids = input_tensor[0].cpu().numpy().tolist()
                text = self.text_processor.indices_to_text(generated_ids)
                text = self.text_processor.clean_text(text)
                results.append(text)
        
        return results
    
    def _compute_char_positions(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """计算韵律位置（用于生成，简化版本）"""
        batch_size, seq_len = input_tensor.shape
        char_positions = torch.ones_like(input_tensor, dtype=torch.long)
        
        for i in range(batch_size):
            pos = 1
            for j in range(seq_len):
                char_positions[i, j] = pos
                if input_tensor[i, j] == self.period_id:
                    pos = 1  # 重置句内位置
                else:
                    pos = min(pos + 1, 7)  # 限制在1-7范围内
        
        return char_positions.to(self.device)


class ForcedLengthDecoding:
    """强制句长解码 - 核心机制2"""
    
    def __init__(self, word2ix: Dict[str, int]):
        self.word2ix = word2ix
        self.period_token = word2ix.get('。', word2ix.get('<EOP>', 0))
        
    def generate_with_constraint(self, model, start_tokens: torch.Tensor, 
                               target_length: int = 5, max_steps: int = 100,
                               temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        """
        带强制句长约束的生成
        
        Args:
            model: Transformer模型
            start_tokens: [1, seq_len] 起始tokens
            target_length: 目标句长（5=五言，7=七言）
            max_steps: 最大生成步数
            temperature: 采样温度
            top_k: top-k采样
            top_p: nucleus采样
            
        Returns:
            generated: [1, new_seq_len] 生成的序列
        """
        model.eval()
        device = start_tokens.device
        generated = start_tokens.clone()
        
        char_count = 0  # 当前句子中的汉字数量
        
        with torch.no_grad():
            for step in range(max_steps):
                # 计算韵律位置
                char_positions = self._compute_char_positions_for_decoding(generated)
                
                # 前向传播
                logits = model(generated, char_positions=char_positions)
                
                # 获取最后一个位置的logits
                if isinstance(logits, dict):
                    next_logits = logits['logits'][0, -1, :]
                else:
                    next_logits = logits[0, -1, :]
                
                # 强制约束逻辑
                if char_count >= target_length:
                    # 强制下一个token为句号
                    next_token = self.period_token
                    char_count = 0
                    print(f"📏 强制断句：已生成{target_length}字")
                else:
                    # 正常采样
                    next_token = self._sample_token(next_logits, temperature, top_k, top_p)
                    
                    # 检查是否为汉字（简单判断：token id在合理范围内）
                    if self._is_chinese_char(next_token):
                        char_count += 1
                    elif next_token == self.period_token:
                        char_count = 0
                
                # 添加到序列
                next_token_tensor = torch.tensor([[next_token]], device=device)
                generated = torch.cat([generated, next_token_tensor], dim=1)
                
                # 检查结束条件
                if next_token == self.period_token and char_count == 0:
                    # 刚完成一句
                    continue
                elif generated.size(1) >= start_tokens.size(1) + max_steps:
                    break
        
        return generated
    
    def _sample_token(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
        """采样下一个token"""
        return TextSampler.combined_sample(
            logits.unsqueeze(0), top_k=top_k, top_p=top_p, temperature=temperature
        ).item()
    
    def _is_chinese_char(self, token_id: int) -> bool:
        """判断是否为汉字token（排除特殊符号和标点）"""
        # 排除特殊token和标点符号
        special_tokens = {
            self.word2ix.get('</s>', -1),
            self.word2ix.get('<START>', -1), 
            self.word2ix.get('<EOP>', -1),
            self.word2ix.get('<PAD>', -1),
            self.word2ix.get('<UNK>', -1),
            self.period_token,  # 句号
            self.word2ix.get('，', -1),  # 逗号
            self.word2ix.get('！', -1),  # 感叹号
            self.word2ix.get('？', -1),  # 问号
            self.word2ix.get('：', -1),  # 冒号
            self.word2ix.get('；', -1),  # 分号
            self.word2ix.get('"', -1),   # 引号
            self.word2ix.get('"', -1),   # 引号
            -1  # 排除-1这个无效ID
        }
        
        # 如果是特殊token，不是汉字
        if token_id in special_tokens:
            return False
            
        # 简单策略：对于测试环境，ID >= 4 且不在特殊符号集合中的都认为是汉字
        # 对于实际环境，可以根据词汇表规模调整
        vocab_size = len(self.word2ix)
        if vocab_size < 100:  # 测试环境的小词汇表
            return token_id >= 4 and token_id not in special_tokens
        else:  # 实际环境的大词汇表
            return token_id >= 50 and token_id not in special_tokens
    
    def _compute_char_positions_for_decoding(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """为强制句长解码计算韵律位置"""
        batch_size, seq_len = input_tensor.shape
        char_positions = torch.ones_like(input_tensor, dtype=torch.long)
        
        for i in range(batch_size):
            pos = 1
            for j in range(seq_len):
                char_positions[i, j] = pos
                if input_tensor[i, j] == self.period_token:
                    pos = 1  # 重置句内位置
                else:
                    pos = min(pos + 1, 7)  # 限制在1-7范围内
        
        return char_positions.to(input_tensor.device)


class QualityAssessment:
    """生成质量评估工具"""
    
    def __init__(self, ix2word: Dict[int, str], word2ix: Dict[str, int]):
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.processor = TextProcessor(ix2word, word2ix)
    
    def assess_poem_quality(self, poem_text: str, poem_type: str = "续写") -> Dict[str, float]:
        """评估诗歌质量"""
        sentences = self.processor.extract_sentences(poem_text)
        
        metrics = {
            'length_score': self._assess_length(sentences),
            'structure_score': self._assess_structure(sentences),
            'repetition_score': self._assess_repetition(poem_text),
            'coherence_score': self._assess_coherence(sentences),
            'overall_score': 0.0
        }
        
        # 计算总分
        weights = [0.2, 0.3, 0.3, 0.2]
        score_values = [metrics['length_score'], metrics['structure_score'], 
                       metrics['repetition_score'], metrics['coherence_score']]
        metrics['overall_score'] = sum(w * s for w, s in zip(weights, score_values))
        
        return metrics
    
    def _assess_length(self, sentences: List[str]) -> float:
        """评估句子长度合理性"""
        if not sentences:
            return 0.0
        
        lengths = [len(s) for s in sentences]
        
        # 理想长度范围：5-7字
        ideal_range = (5, 7)
        score = 0.0
        
        for length in lengths:
            if ideal_range[0] <= length <= ideal_range[1]:
                score += 1.0
            elif abs(length - ideal_range[0]) <= 2 or abs(length - ideal_range[1]) <= 2:
                score += 0.5
        
        return score / len(sentences) if sentences else 0.0
    
    def _assess_structure(self, sentences: List[str]) -> float:
        """评估诗歌结构"""
        if len(sentences) < 2:
            return 0.3
        elif 2 <= len(sentences) <= 4:
            return 1.0
        elif 5 <= len(sentences) <= 8:
            return 0.8
        else:
            return 0.5
    
    def _assess_repetition(self, text: str) -> float:
        """评估重复度（越低越好）"""
        if not text:
            return 0.0
        
        char_counts = Counter(text)
        text_length = len(text)
        
        # 计算字符重复率
        repetition_rate = sum(count - 1 for count in char_counts.values() if count > 1) / text_length
        
        # 转换为分数（重复率越低分数越高）
        return max(0.0, 1.0 - repetition_rate * 2)
    
    def _assess_coherence(self, sentences: List[str]) -> float:
        """评估语义连贯性（简单启发式）"""
        if len(sentences) < 2:
            return 0.7
        
        # 检查句子长度的一致性
        lengths = [len(s) for s in sentences]
        length_variance = np.var(lengths) if len(lengths) > 1 else 0
        
        # 长度一致性分数
        length_score = max(0.0, 1.0 - length_variance / 10)
        
        return length_score

def create_quality_assessor(ix2word: Dict[int, str], word2ix: Dict[str, int]) -> QualityAssessment:
    """创建质量评估器实例"""
    return QualityAssessment(ix2word, word2ix)

if __name__ == "__main__":
    # 测试生成工具
    print("Testing generation utilities...")
    
    # 测试采样策略
    vocab_size = 1000
    logits = torch.randn(1, vocab_size)
    
    print("Testing sampling strategies:")
    
    # 贪婪采样
    greedy_result = TextSampler.greedy_sample(logits)
    print(f"Greedy sample: {greedy_result.item()}")
    
    # Top-K采样
    topk_result = TextSampler.top_k_sample(logits, k=50)
    print(f"Top-K sample: {topk_result.item()}")
    
    # Nucleus采样
    nucleus_result = TextSampler.nucleus_sample(logits, p=0.9)
    print(f"Nucleus sample: {nucleus_result.item()}")
    
    # 组合采样
    combined_result = TextSampler.combined_sample(logits)
    print(f"Combined sample: {combined_result.item()}")
    
    # 测试文本处理
    print("\nTesting text processing:")
    
    # 创建示例词汇表
    test_ix2word = {0: '</s>', 1: '<START>', 2: '<EOP>', 3: '春', 4: '花', 5: '秋', 6: '月', 7: '。'}
    test_word2ix = {v: k for k, v in test_ix2word.items()}
    
    processor = TextProcessor(test_ix2word, test_word2ix)
    
    # 测试索引到文本
    test_indices = [1, 3, 4, 7, 5, 6, 7, 2]
    text = processor.indices_to_text(test_indices)
    print(f"Indices to text: {text}")
    
    # 测试文本到索引
    test_text = "春花。秋月。"
    indices = processor.text_to_indices(test_text)
    print(f"Text to indices: {indices}")
    
    # 测试格式化
    formatted = processor.format_poem(text, "测试")
    print(f"Formatted poem:\n{formatted}")
    
    # 测试质量评估
    print("\nTesting quality assessment:")
    assessor = QualityAssessment(test_ix2word, test_word2ix)
    quality = assessor.assess_poem_quality("春花秋月夜。明月照人归。")
    print(f"Quality metrics: {quality}")
    
    print("\nGeneration utilities testing completed!")

class ConstrainedPoetryGenerator:
    """约束解码诗歌生成器 - 严格遵循唐诗格律"""
    
    def __init__(self, model, ix2word: Dict[int, str], word2ix: Dict[str, int]):
        self.model = model
        self.text_processor = TextProcessor(ix2word, word2ix)
        self.sampler = TextSampler()
        self.device = next(model.parameters()).device
        self.word2ix = word2ix
        self.ix2word = ix2word
        
        # 重要的token ID
        self.period_id = word2ix.get('。', -1)
        self.comma_id = word2ix.get('，', -1)
        self.start_id = word2ix.get(config.START_TOKEN, 0)
        self.end_id = word2ix.get(config.END_TOKEN, -1)
        
        # 标点符号集合
        self.punctuation_ids = {
            self.period_id, self.comma_id,
            word2ix.get('！', -1), word2ix.get('？', -1),
            word2ix.get('：', -1), word2ix.get('；', -1),
            word2ix.get('"', -1), word2ix.get('"', -1),
            word2ix.get('</s>', -1), word2ix.get('<START>', -1),
            word2ix.get('<EOP>', -1), -1
        }
    
    def generate_constrained_poem(self, prompt: str = "", poem_type: str = "七言绝句",
                                temperature: float = 0.8, top_k: int = 50, 
                                top_p: float = 0.9) -> str:
        """
        生成严格符合格律的诗歌
        
        Args:
            prompt: 提示词
            poem_type: 诗歌类型 ("五言绝句", "七言绝句", "五言律诗", "七言律诗")
            temperature: 采样温度
            top_k: top-k采样
            top_p: nucleus采样
            
        Returns:
            符合格律的诗歌文本
        """
        # 解析诗歌类型
        char_per_line, total_lines = self._parse_poem_type(poem_type)
        
        self.model.eval()
        
        with torch.no_grad():
            # 初始化输入
            if prompt:
                input_ids = self.text_processor.text_to_indices(prompt, add_special=True)
            else:
                input_ids = [self.start_id]
            
            input_tensor = torch.tensor([input_ids], device=self.device)
            
            # 生成计数器
            current_line_chars = 0  # 当前句中的汉字数
            completed_lines = 0     # 已完成的句数
            
            # 如果有提示词，计算其字符数
            if prompt:
                current_line_chars = self._count_chinese_chars(prompt)
            
            # 主生成循环
            while completed_lines < total_lines:
                # 计算韵律位置
                char_positions = self._compute_char_positions(input_tensor)
                
                # 前向传播
                outputs = self.model(input_tensor, char_positions=char_positions)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits'][:, -1, :]
                else:
                    logits = outputs[:, -1, :]
                
                # 约束解码逻辑
                next_token = self._constrained_sampling(
                    logits, current_line_chars, char_per_line, 
                    completed_lines, total_lines, temperature, top_k, top_p
                )
                
                # 更新计数器
                if self._is_chinese_char(next_token):
                    current_line_chars += 1
                elif next_token in [self.period_id, self.comma_id]:
                    completed_lines += 1
                    current_line_chars = 0
                
                # 添加token到序列
                next_token_tensor = torch.tensor([[next_token]], device=self.device)
                input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
                
                # 安全检查：防止无限循环
                if input_tensor.size(1) > 200:
                    break
            
            # 转换为文本
            generated_ids = input_tensor[0].cpu().numpy().tolist()
            text = self.text_processor.indices_to_text(generated_ids)
            return self._format_poem(text, char_per_line)
    
    def _parse_poem_type(self, poem_type: str) -> tuple:
        """解析诗歌类型，返回(每句字数, 总句数)"""
        type_map = {
            "五言绝句": (5, 4),
            "七言绝句": (7, 4), 
            "五言律诗": (5, 8),
            "七言律诗": (7, 8)
        }
        return type_map.get(poem_type, (7, 4))  # 默认七言绝句
    
    def _constrained_sampling(self, logits: torch.Tensor, current_chars: int, 
                            target_chars: int, completed_lines: int, total_lines: int,
                            temperature: float, top_k: int, top_p: float) -> int:
        """约束采样：根据当前状态决定下一个token"""
        
        # 规则1：如果当前句已达到目标字数，强制生成正确的标点符号
        if current_chars >= target_chars:
            # 根据唐诗格律确定标点符号
            return self._get_correct_punctuation(completed_lines, total_lines)
        
        # 规则2：如果已完成所有句子，强制结束
        if completed_lines >= total_lines:
            return self.end_id
        
        # 规则3：正常采样，但排除句号（除非满足条件）
        # 创建掩码，禁止在不合适的位置生成句号
        masked_logits = logits.clone()
        
        # 在字数不足时，大幅降低所有标点符号的概率
        if current_chars < target_chars:
            masked_logits[0, self.period_id] = float('-inf')
            masked_logits[0, self.comma_id] = float('-inf')
        
        # 在最后一句完成后，禁止继续生成汉字
        if completed_lines >= total_lines - 1 and current_chars >= target_chars:
            # 只允许句号
            for token_id in range(len(self.word2ix)):
                if token_id != self.period_id:
                    masked_logits[0, token_id] = float('-inf')
        
        # 使用约束后的logits进行采样
        next_token = self.sampler.combined_sample(
            masked_logits, top_k=top_k, top_p=top_p, temperature=temperature
        )
        
        return next_token.item()
    
    def _get_correct_punctuation(self, completed_lines: int, total_lines: int) -> int:
        """根据唐诗格律获取正确的标点符号"""
        # completed_lines 是当前正在完成的句子索引（0-based）
        
        if total_lines == 4:  # 绝句：1,3句逗号，2,4句句号
            if completed_lines in [0, 2]:  # 第1,3句（0,2索引）
                return self.comma_id
            else:  # 第2,4句（1,3索引）
                return self.period_id
        
        elif total_lines == 8:  # 律诗：前6句逗号，第7句逗号，第8句句号
            if completed_lines < 7:  # 前7句（索引0-6）
                return self.comma_id
            else:  # 第8句（索引7）
                return self.period_id
        
        else:  # 其他情况，默认最后一句句号，其余逗号
            if completed_lines == total_lines - 1:
                return self.period_id
            else:
                return self.comma_id
    
    def _is_chinese_char(self, token_id: int) -> bool:
        """判断是否为汉字token"""
        # 排除标点符号和特殊token
        if token_id in self.punctuation_ids:
            return False
        
        # 简单策略：认为大部分token都是汉字
        vocab_size = len(self.word2ix)
        if vocab_size > 1000:  # 大词汇表
            return token_id >= 100 and token_id not in self.punctuation_ids
        else:  # 小词汇表
            return token_id >= 10 and token_id not in self.punctuation_ids
    
    def _count_chinese_chars(self, text: str) -> int:
        """统计文本中的汉字数量"""
        count = 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # Unicode汉字范围
                count += 1
        return count
    
    def _compute_char_positions(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """计算韵律位置（用于约束生成）"""
        batch_size, seq_len = input_tensor.shape
        char_positions = torch.ones_like(input_tensor, dtype=torch.long)
        
        for i in range(batch_size):
            pos = 1
            for j in range(seq_len):
                char_positions[i, j] = pos
                if input_tensor[i, j] == self.period_id:
                    pos = 1  # 重置句内位置
                else:
                    pos = min(pos + 1, 7)  # 限制在1-7范围内
        
        return char_positions.to(self.device)
    
    def _format_poem(self, text: str, chars_per_line: int) -> str:
        """格式化诗歌输出"""
        # 移除特殊标记
        text = text.replace('<START>', '').replace('<EOP>', '').replace('</s>', '')
        
        # 按句号和逗号分割，保持原有标点
        import re
        lines = re.split(r'[。，]', text)
        formatted_lines = []
        
        # 提取原始标点符号
        punctuation_marks = re.findall(r'[。，]', text)
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line:  # 非空行
                # 确保每行字数正确（后处理保护）
                chinese_chars = [c for c in line if '\u4e00' <= c <= '\u9fff']
                if len(chinese_chars) > chars_per_line:
                    # 截取到目标字数
                    line = ''.join(chinese_chars[:chars_per_line])
                elif len(chinese_chars) < chars_per_line:
                    # 字数不足的情况，保持原样（在生成时应该已经避免）
                    line = ''.join(chinese_chars)
                else:
                    line = ''.join(chinese_chars)
                
                if line:
                    # 添加对应的标点符号
                    if i < len(punctuation_marks):
                        formatted_lines.append(line + punctuation_marks[i])
                    else:
                        formatted_lines.append(line + '。')  # 默认句号
        
        return '\n'.join(formatted_lines)
    
    def generate_acrostic_poem(self, acrostic_chars: str, poem_type: str = "五言绝句",
                             temperature: float = 0.8, top_k: int = 50, 
                             top_p: float = 0.9) -> str:
        """
        生成藏头诗
        
        Args:
            acrostic_chars: 藏头字符串，每个字符对应一句的首字
            poem_type: 诗歌类型 ("五言绝句", "七言绝句", "五言律诗", "七言律诗")
            temperature: 采样温度
            top_k: top-k采样
            top_p: nucleus采样
            
        Returns:
            藏头诗文本
        """
        # 解析诗歌类型
        char_per_line, total_lines = self._parse_poem_type(poem_type)
        
        # 检查藏头字符数量与诗体是否匹配
        if len(acrostic_chars) != total_lines:
            # 如果字符数不匹配，调整字符串
            if len(acrostic_chars) < total_lines:
                # 字符数不足，重复最后一个字符
                acrostic_chars = acrostic_chars + acrostic_chars[-1] * (total_lines - len(acrostic_chars))
            else:
                # 字符数过多，截取前面的字符
                acrostic_chars = acrostic_chars[:total_lines]
        
        self.model.eval()
        
        all_lines = []
        
        with torch.no_grad():
            # 逐句生成藏头诗
            for line_idx, first_char in enumerate(acrostic_chars):
                # 每句都以藏头字符开始
                first_char_id = self.word2ix.get(first_char)
                if first_char_id is None:
                    # 如果藏头字符不在词汇表中，跳过这句
                    continue
                
                # 初始化输入：START + 藏头字符
                input_ids = [self.start_id, first_char_id]
                input_tensor = torch.tensor([input_ids], device=self.device)
                
                # 生成这一句的剩余部分（需要 char_per_line - 1 个汉字 + 标点）
                current_chars = 1  # 已有藏头字符
                
                while current_chars < char_per_line:
                    # 计算韵律位置
                    char_positions = self._compute_char_positions(input_tensor)
                    
                    # 前向传播
                    outputs = self.model(input_tensor, char_positions=char_positions)
                    
                    if isinstance(outputs, dict):
                        logits = outputs['logits'][:, -1, :]
                    else:
                        logits = outputs[:, -1, :]
                    
                    # 采样下一个token（避免标点符号）
                    masked_logits = logits.clone()
                    
                    # 在句子未完成时，禁止生成标点符号
                    if current_chars < char_per_line:
                        masked_logits[0, self.period_id] = float('-inf')
                        masked_logits[0, self.comma_id] = float('-inf')
                    
                    next_token = self.sampler.combined_sample(
                        masked_logits, top_k=top_k, top_p=top_p, temperature=temperature
                    )
                    
                    next_token_id = next_token.item()
                    
                    # 只统计汉字
                    if self._is_chinese_char(next_token_id):
                        current_chars += 1
                    
                    # 添加token到序列
                    next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
                    input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)
                    
                    # 安全检查：防止无限循环
                    if input_tensor.size(1) > 50:
                        break
                
                # 添加正确的标点符号
                correct_punctuation = self._get_correct_punctuation(line_idx, total_lines)
                punctuation_tensor = torch.tensor([[correct_punctuation]], device=self.device)
                input_tensor = torch.cat([input_tensor, punctuation_tensor], dim=1)
                
                # 转换这一句为文本
                line_ids = input_tensor[0].cpu().numpy().tolist()
                line_text = self.text_processor.indices_to_text(line_ids)
                
                # 清理并格式化这一句
                line_text = line_text.replace('<START>', '').replace('<EOP>', '').replace('</s>', '')
                all_lines.append(line_text.strip())
        
        # 组合所有句子
        result = '\n'.join(all_lines)
        return result