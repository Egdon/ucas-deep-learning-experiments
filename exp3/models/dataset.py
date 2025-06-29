#!/usr/bin/env python3
"""
Dataset classes and data processing utilities for poetry generation.
Includes standard poetry dataset and acrostic poetry data augmentation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
from typing import List, Tuple, Dict, Optional

from .config import config

class PoetryDataset(Dataset):
    """Standard poetry dataset for sequence-to-sequence training."""
    
    def __init__(self, data_path: str = None, mode: str = 'train'):
        """
        Initialize poetry dataset.
        
        Args:
            data_path: Path to NPZ data file or directory containing tang.npz
            mode: 'train', 'val', or 'test'
        """
        if data_path is None:
            self.data_path = config.DATA_PATH
        else:
            # 如果传入的是目录，则查找tang.npz文件
            if os.path.isdir(data_path):
                self.data_path = os.path.join(data_path, 'tang.npz')
            else:
                self.data_path = data_path
        
        self.mode = mode
        
        self.data, self.ix2word, self.word2ix = self.load_data()
        self.vocab_size = len(self.word2ix)
        
        # 预处理数据
        self.processed_data = self.preprocess_data()
        
        # 划分数据集
        self.train_data, self.val_data, self.test_data = self.split_data()
        
        # 根据模式选择数据
        if mode == 'train':
            self.current_data = self.train_data
        elif mode == 'val':
            self.current_data = self.val_data
        else:
            self.current_data = self.test_data
    
    def load_data(self) -> Tuple[np.ndarray, Dict, Dict]:
        """加载NPZ数据文件"""
        try:
            # 检查文件是否存在
            if not os.path.exists(self.data_path):
                # 如果文件不存在，尝试查找可能的文件
                possible_paths = [
                    'data/tang.npz',
                    'tang.npz',
                    os.path.join(os.path.dirname(self.data_path), 'tang.npz')
                ]
                
                found_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        found_path = path
                        break
                
                if found_path:
                    print(f"🔍 数据文件未在 {self.data_path} 找到，使用: {found_path}")
                    self.data_path = found_path
                else:
                    raise FileNotFoundError(
                        f"数据文件未找到: {self.data_path}\n"
                        f"请确保以下任一文件存在:\n"
                        f"  - data/tang.npz\n"
                        f"  - tang.npz\n"
                        f"当前工作目录: {os.getcwd()}"
                    )
            
            print(f"📁 加载数据文件: {self.data_path}")
            dataset = np.load(self.data_path, allow_pickle=True)
            data = dataset['data']
            ix2word = dataset['ix2word'].item()
            word2ix = dataset['word2ix'].item()
            print(f"✅ 数据加载成功: {len(data)} 首诗，词汇量 {len(word2ix)}")
            return data, ix2word, word2ix
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {self.data_path}: {e}")
    
    def preprocess_data(self) -> List[np.ndarray]:
        """预处理数据，提取有效序列"""
        processed = []
        padding_token_id = self.word2ix.get(config.PADDING_TOKEN, None)
        
        for poem in self.data:
            if padding_token_id is not None:
                # 找到非填充部分
                non_padding_positions = np.where(poem != padding_token_id)[0]
                if len(non_padding_positions) > 10:  # 最少10个字符
                    valid_sequence = poem[non_padding_positions]
                    processed.append(valid_sequence)
            else:
                processed.append(poem)
        
        return processed
    
    def split_data(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List, List, List]:
        """划分训练、验证和测试集"""
        total_size = len(self.processed_data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        # 随机打乱数据
        indices = list(range(total_size))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_data = [self.processed_data[i] for i in train_indices]
        val_data = [self.processed_data[i] for i in val_indices]
        test_data = [self.processed_data[i] for i in test_indices]
        
        return train_data, val_data, test_data
    
    def __len__(self) -> int:
        return len(self.current_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            input_seq: 输入序列
            target_seq: 目标序列  
            mode_flag: 模式标志（0=续写，1=藏头诗）
        """
        poem = self.current_data[idx]
        
        # 确保序列长度
        max_len = min(len(poem), config.SEQUENCE_LENGTH)
        if max_len < 2:
            # 处理过短序列
            poem = np.pad(poem, (0, 2 - len(poem)), constant_values=self.word2ix[config.PADDING_TOKEN])
            max_len = 2
        
        # 创建输入和目标序列
        input_seq = poem[:max_len-1]
        target_seq = poem[1:max_len]
        
        # 确保长度一致
        if len(input_seq) != len(target_seq):
            min_len = min(len(input_seq), len(target_seq))
            input_seq = input_seq[:min_len]
            target_seq = target_seq[:min_len]
        
        # 转换为tensor
        input_tensor = torch.LongTensor(input_seq)
        target_tensor = torch.LongTensor(target_seq)
        mode_tensor = torch.LongTensor([0])  # 0表示标准续写模式
        
        return input_tensor, target_tensor, mode_tensor

class AcrosticDataset(Dataset):
    """藏头诗数据集，用于训练藏头诗生成功能"""
    
    def __init__(self, base_dataset: PoetryDataset, num_acrostic_samples: int = 5000):
        """
        基于基础数据集创建藏头诗训练数据
        
        Args:
            base_dataset: 基础诗词数据集
            num_acrostic_samples: 生成的藏头诗样本数量
        """
        self.base_dataset = base_dataset
        self.ix2word = base_dataset.ix2word
        self.word2ix = base_dataset.word2ix
        
        # 生成藏头诗训练数据
        self.acrostic_data = self.generate_acrostic_data(num_acrostic_samples)
    
    def generate_acrostic_data(self, num_samples: int) -> List[Tuple[List[int], List[int]]]:
        """
        从基础数据集生成藏头诗训练样本
        
        Returns:
            List of (acrostic_chars, poem_sequence) pairs
        """
        acrostic_samples = []
        
        for _ in range(num_samples):
            # 随机选择一首诗
            poem_idx = random.randint(0, len(self.base_dataset.current_data) - 1)
            poem = self.base_dataset.current_data[poem_idx]
            
            # 转换为文本
            poem_text = self.indices_to_text(poem)
            sentences = poem_text.split('。')
            
            # 提取句首字符作为藏头
            first_chars = []
            valid_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip().replace('<START>', '').replace('<EOP>', '')
                if len(sentence) > 0:
                    first_chars.append(sentence[0])
                    valid_sentences.append(sentence)
            
            # 确保至少有2个句子
            if len(first_chars) >= 2:
                # 构造训练样本
                acrostic_indices = [self.word2ix.get(char, self.word2ix[config.PADDING_TOKEN]) 
                                  for char in first_chars]
                
                acrostic_samples.append((acrostic_indices, poem))
        
        return acrostic_samples
    
    def indices_to_text(self, indices: np.ndarray) -> str:
        """将索引序列转换为文本"""
        text = ""
        for idx in indices:
            if idx in self.ix2word:
                char = self.ix2word[idx]
                if char == config.PADDING_TOKEN:
                    break
                text += char
        return text
    
    def __len__(self) -> int:
        return len(self.acrostic_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取藏头诗训练样本
        
        Returns:
            input_seq: 输入序列（包含藏头约束）
            target_seq: 目标序列
            mode_flag: 模式标志（1=藏头诗）
        """
        acrostic_chars, poem = self.acrostic_data[idx]
        
        # 创建带约束的输入序列
        max_len = min(len(poem), config.SEQUENCE_LENGTH)
        if max_len < 2:
            poem = np.pad(poem, (0, 2 - len(poem)), constant_values=self.word2ix[config.PADDING_TOKEN])
            max_len = 2
        
        input_seq = poem[:max_len-1]
        target_seq = poem[1:max_len]
        
        # 确保长度一致
        if len(input_seq) != len(target_seq):
            min_len = min(len(input_seq), len(target_seq))
            input_seq = input_seq[:min_len]
            target_seq = target_seq[:min_len]
        
        input_tensor = torch.LongTensor(input_seq)
        target_tensor = torch.LongTensor(target_seq)
        mode_tensor = torch.LongTensor([1])  # 1表示藏头诗模式
        
        return input_tensor, target_tensor, mode_tensor

class MixedDataset(Dataset):
    """混合数据集，结合标准诗词和藏头诗训练数据"""
    
    def __init__(self, poetry_dataset: PoetryDataset, acrostic_dataset: AcrosticDataset, 
                 mix_ratio: float = 0.7):
        """
        创建混合数据集
        
        Args:
            poetry_dataset: 标准诗词数据集
            acrostic_dataset: 藏头诗数据集
            mix_ratio: 标准诗词数据的比例
        """
        self.poetry_dataset = poetry_dataset
        self.acrostic_dataset = acrostic_dataset
        self.mix_ratio = mix_ratio
        
        # 计算采样数量
        total_poetry = len(poetry_dataset)
        total_acrostic = len(acrostic_dataset)
        
        self.poetry_samples = int(total_poetry * mix_ratio)
        self.acrostic_samples = int(total_poetry * (1 - mix_ratio))
        
        # 确保不超过可用数据
        self.poetry_samples = min(self.poetry_samples, total_poetry)
        self.acrostic_samples = min(self.acrostic_samples, total_acrostic)
        
        self.total_samples = self.poetry_samples + self.acrostic_samples
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取混合样本"""
        if idx < self.poetry_samples:
            # 标准诗词样本
            poetry_idx = idx % len(self.poetry_dataset)
            return self.poetry_dataset[poetry_idx]
        else:
            # 藏头诗样本
            acrostic_idx = (idx - self.poetry_samples) % len(self.acrostic_dataset)
            return self.acrostic_dataset[acrostic_idx]

def create_dataloaders(data_dir: str = None, 
                      batch_size: int = None,
                      max_seq_len: int = None,
                      test_size: float = 0.1,
                      num_workers: int = None,
                      include_acrostic: bool = True,
                      add_rhythmic_info: bool = False) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    创建训练、验证数据加载器，支持韵律信息
    
    Args:
        data_dir: 数据目录路径
        batch_size: 批次大小
        max_seq_len: 最大序列长度
        test_size: 验证集比例
        num_workers: 工作进程数
        include_acrostic: 是否包含藏头诗数据
        add_rhythmic_info: 是否添加韵律信息
        
    Returns:
        train_loader, val_loader, vocab_info
    """
    batch_size = batch_size or config.BATCH_SIZE
    max_seq_len = max_seq_len or config.SEQUENCE_LENGTH
    num_workers = num_workers or config.NUM_WORKERS
    
    # 创建基础数据集
    train_dataset = PoetryDataset(data_dir, mode='train')
    val_dataset = PoetryDataset(data_dir, mode='val')
    
    if include_acrostic:
        # 创建藏头诗数据集
        acrostic_train = AcrosticDataset(train_dataset, num_acrostic_samples=3000)
        acrostic_val = AcrosticDataset(val_dataset, num_acrostic_samples=500)
        
        # 创建混合数据集
        train_dataset = MixedDataset(train_dataset, acrostic_train)
        val_dataset = MixedDataset(val_dataset, acrostic_val)
    
    # 获取词汇信息
    if include_acrostic:
        word2ix = train_dataset.poetry_dataset.word2ix
        ix2word = train_dataset.poetry_dataset.ix2word
    else:
        word2ix = train_dataset.word2ix
        ix2word = train_dataset.ix2word
    
    vocab_info = {
        'word2ix': word2ix,
        'ix2word': ix2word,
        'vocab_size': len(word2ix)
    }
    
    # 自定义collate函数处理变长序列和韵律信息
    def collate_fn(batch):
        input_seqs, target_seqs, mode_flags = zip(*batch)
        
        # 填充到指定长度
        actual_max_len = min(max_seq_len, max(len(seq) for seq in input_seqs))
        padding_id = word2ix[config.PADDING_TOKEN]
        
        padded_inputs = []
        padded_targets = []
        
        for input_seq, target_seq in zip(input_seqs, target_seqs):
            # 截断或填充到目标长度
            if len(input_seq) > actual_max_len:
                input_seq = input_seq[:actual_max_len]
                target_seq = target_seq[:actual_max_len]
            
            # 填充输入序列
            padded_input = torch.cat([
                input_seq,
                torch.full((actual_max_len - len(input_seq),), padding_id, dtype=torch.long)
            ])
            # 填充目标序列
            padded_target = torch.cat([
                target_seq,
                torch.full((actual_max_len - len(target_seq),), padding_id, dtype=torch.long)
            ])
            
            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)
        
        batch_dict = {
            'input_ids': torch.stack(padded_inputs),
            'target_ids': torch.stack(padded_targets),
            'mode_flags': torch.stack(mode_flags)
        }
        
        # 如果需要韵律信息，计算字符位置（向量化版本）
        if add_rhythmic_info:
            char_positions = []
            period_id = word2ix.get('。', -1)
            
            for input_seq in batch_dict['input_ids']:
                # 向量化计算韵律位置
                pos_seq = torch.ones_like(input_seq, dtype=torch.long)
                
                # 找到句号位置
                period_mask = (input_seq == period_id)
                period_indices = period_mask.nonzero(as_tuple=True)[0]
                
                # 计算每个位置在句子中的位置
                if len(period_indices) > 0:
                    # 为每个句子段分别计算位置
                    start_idx = 0
                    for period_idx in period_indices:
                        segment_len = period_idx - start_idx + 1
                        pos_seq[start_idx:period_idx+1] = torch.arange(1, segment_len + 1, dtype=torch.long)
                        start_idx = period_idx + 1
                    
                    # 处理最后一个句子段
                    if start_idx < len(input_seq):
                        remaining_len = len(input_seq) - start_idx
                        pos_seq[start_idx:] = torch.arange(1, remaining_len + 1, dtype=torch.long)
                else:
                    # 没有句号，整个序列是一个句子
                    pos_seq = torch.arange(1, len(input_seq) + 1, dtype=torch.long)
                
                # 限制位置在1-7范围内
                pos_seq = torch.clamp(pos_seq, min=1, max=7)
                char_positions.append(pos_seq)
            
            batch_dict['char_positions'] = torch.stack(char_positions)
        
        return batch_dict
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, vocab_info

if __name__ == "__main__":
    # 测试数据集
    print("Testing poetry dataset...")
    
    # 创建数据加载器
    dataloaders = create_dataloaders()
    
    # 测试训练数据加载器
    train_loader = dataloaders['train']
    print(f"Train dataset size: {len(train_loader.dataset)}")
    
    # 获取一个批次
    for batch_idx, (inputs, targets, modes) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Target shape: {targets.shape}")
        print(f"  Mode shape: {modes.shape}")
        print(f"  Mode values: {modes.flatten()[:10].tolist()}")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    print("Dataset testing completed!") 