#!/usr/bin/env python3
"""
Poetry Dataset Parser
Analyzes the structure and content of tang.npz dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import sys

class PoetryDatasetAnalyzer:
    """Analyzer for Chinese poetry dataset in NPZ format."""
    
    def __init__(self, data_path="data/tang.npz"):
        """
        Initialize the analyzer with dataset path.
        
        Args:
            data_path (str): Path to the NPZ dataset file
        """
        self.data_path = data_path
        self.data = None
        self.ix2word = None
        self.word2ix = None
        self.loaded = False
        
    def load_dataset(self):
        """Load and validate the dataset."""
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Dataset not found: {self.data_path}")
                
            dataset = np.load(self.data_path, allow_pickle=True)
            
            # 验证数据集完整性
            required_keys = ['data', 'ix2word', 'word2ix']
            for key in required_keys:
                if key not in dataset:
                    raise ValueError(f"Missing required key: {key}")
            
            self.data = dataset['data']
            self.ix2word = dataset['ix2word'].item()
            self.word2ix = dataset['word2ix'].item()
            self.loaded = True
            
            print(f"Dataset loaded successfully from {self.data_path}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)
    
    def analyze_basic_info(self):
        """Analyze basic dataset statistics."""
        if not self.loaded:
            self.load_dataset()
            
        print("\n=== Basic Dataset Information ===")
        print(f"Total poems: {len(self.data)}")
        print(f"Vocabulary size: {len(self.word2ix)}")
        print(f"Max poem length: {self.data.shape[1] if len(self.data.shape) > 1 else 'Variable'}")
        print(f"Data shape: {self.data.shape}")
        print(f"Data type: {self.data.dtype}")
        
        # 分析特殊标记
        special_tokens = []
        for word, idx in self.word2ix.items():
            if word.startswith('<') and word.endswith('>'):
                special_tokens.append((word, idx))
        
        print(f"\nSpecial tokens ({len(special_tokens)}):")
        for token, idx in sorted(special_tokens, key=lambda x: x[1]):
            print(f"  {token}: {idx}")
            
        # 深入分析数据结构
        self._analyze_data_structure()
    
    def _analyze_data_structure(self):
        """深入分析数据集的内部结构"""
        print("\n=== Data Structure Analysis ===")
        
        # 分析各种索引值的分布
        padding_token = self.word2ix.get('</s>', -1)
        start_token = self.word2ix.get('<START>', -1)
        eop_token = self.word2ix.get('<EOP>', -1)
        
        # 统计非填充数据的分布
        non_padding_count = 0
        start_positions = []
        actual_poem_lengths = []
        
        for i, poem in enumerate(self.data):
            non_padding_positions = np.where(poem != padding_token)[0]
            if len(non_padding_positions) > 0:
                non_padding_count += 1
                start_pos = non_padding_positions[0]
                start_positions.append(start_pos)
                actual_poem_lengths.append(len(non_padding_positions))
                
                # 检查前几首诗的具体结构
                if i < 5:
                    print(f"\nPoem {i} structure analysis:")
                    valid_indices = poem[non_padding_positions]
                    print(f"  Non-padding positions: {non_padding_positions[:10].tolist()}...")
                    print(f"  Valid indices: {valid_indices[:10].tolist()}...")
                    
                    # 转换为文本
                    text_chars = []
                    for idx in valid_indices[:20]:
                        if idx in self.ix2word:
                            text_chars.append(self.ix2word[idx])
                    print(f"  Text preview: {''.join(text_chars)}")
        
        print(f"\nNon-empty poems: {non_padding_count}/{len(self.data)} ({non_padding_count/len(self.data)*100:.1f}%)")
        
        if actual_poem_lengths:
            actual_lengths = np.array(actual_poem_lengths)
            print(f"Actual poem length statistics:")
            print(f"  Mean: {actual_lengths.mean():.2f}")
            print(f"  Median: {np.median(actual_lengths):.2f}")
            print(f"  Min: {actual_lengths.min()}")
            print(f"  Max: {actual_lengths.max()}")
            
        if start_positions:
            start_pos_array = np.array(start_positions)
            print(f"Start position statistics:")
            print(f"  Most common start position: {np.bincount(start_pos_array).argmax()}")
            print(f"  Start position range: {start_pos_array.min()} - {start_pos_array.max()}")
    
    def analyze_poem_lengths(self):
        """Analyze the distribution of poem lengths."""
        if not self.loaded:
            self.load_dataset()
            
        # 计算每首诗的实际长度（排除填充标记）
        poem_lengths = []
        padding_token = self.word2ix.get('</s>', None)
        
        for poem in self.data:
            if padding_token is not None:
                # 找到第一个填充标记的位置
                padding_positions = np.where(poem == padding_token)[0]
                if len(padding_positions) > 0:
                    length = padding_positions[0]
                else:
                    length = len(poem)
            else:
                length = len(poem)
            poem_lengths.append(length)
        
        poem_lengths = np.array(poem_lengths)
        
        print("\n=== Poem Length Analysis ===")
        print(f"Mean length: {poem_lengths.mean():.2f}")
        print(f"Median length: {np.median(poem_lengths):.2f}")
        print(f"Min length: {poem_lengths.min()}")
        print(f"Max length: {poem_lengths.max()}")
        print(f"Std deviation: {poem_lengths.std():.2f}")
        
        return poem_lengths
    
    def analyze_vocabulary(self):
        """Analyze vocabulary distribution and frequency."""
        if not self.loaded:
            self.load_dataset()
            
        # 统计字符频率
        char_counts = Counter()
        for poem in self.data:
            for char_idx in poem:
                if char_idx in self.ix2word:
                    char = self.ix2word[char_idx]
                    char_counts[char] += 1
        
        print("\n=== Vocabulary Analysis ===")
        print(f"Total character occurrences: {sum(char_counts.values())}")
        print(f"Unique characters: {len(char_counts)}")
        
        # 最常见的字符
        most_common = char_counts.most_common(20)
        print("\nTop 20 most frequent characters:")
        for i, (char, count) in enumerate(most_common, 1):
            percentage = (count / sum(char_counts.values())) * 100
            print(f"{i:2d}. '{char}': {count:6d} ({percentage:5.2f}%)")
        
        return char_counts
    
    def sample_poems(self, num_samples=5):
        """Display sample poems in readable format."""
        if not self.loaded:
            self.load_dataset()
            
        print(f"\n=== Sample Poems ({num_samples}) ===")
        
        # 寻找真正有内容的诗句
        valid_poems = []
        padding_token = self.word2ix.get('</s>', None)
        start_token = self.word2ix.get('<START>', None)
        
        for idx in range(len(self.data)):
            poem_indices = self.data[idx]
            
            # 更严格的有效性检查
            if padding_token is not None:
                non_padding = poem_indices[poem_indices != padding_token]
                # 确保有足够的非填充内容（至少10个字符）
                if len(non_padding) >= 10:
                    # 检查是否包含START标记
                    if start_token in non_padding:
                        valid_poems.append(idx)
            
            if len(valid_poems) >= num_samples * 20:  # 收集更多候选
                break
        
        if len(valid_poems) == 0:
            print("No valid poems found in dataset")
            # 尝试直接显示前几首诗作为fallback
            print("Showing first few poems as fallback:")
            for i in range(min(5, len(self.data))):
                self._display_poem(i, i)
            return
            
        # 从有效诗句中随机选择
        selected_indices = np.random.choice(valid_poems, min(num_samples, len(valid_poems)), replace=False)
        
        for i, idx in enumerate(selected_indices, 1):
            self._display_poem(idx, i)
    
    def _display_poem(self, idx, display_num):
        """显示单首诗的详细信息"""
        poem_indices = self.data[idx]
        padding_token = self.word2ix.get('</s>', None)
        
        # 分析诗句结构
        if padding_token is not None:
            non_padding_positions = np.where(poem_indices != padding_token)[0]
            if len(non_padding_positions) > 0:
                valid_indices = poem_indices[non_padding_positions]
                start_pos = non_padding_positions[0]
                end_pos = non_padding_positions[-1]
            else:
                valid_indices = poem_indices
                start_pos = 0
                end_pos = len(poem_indices) - 1
        else:
            valid_indices = poem_indices
            start_pos = 0
            end_pos = len(poem_indices) - 1
        
        # 转换为文本
        poem_text = ""
        clean_text = ""
        
        for char_idx in valid_indices:
            if char_idx in self.ix2word:
                char = self.ix2word[char_idx]
                if char == '</s>':  # 填充标记，停止
                    break
                elif char in ['<START>']:
                    poem_text += f"[{char}]"
                elif char in ['<EOP>']:
                    poem_text += f"[{char}]"
                    break
                else:
                    poem_text += char
                    clean_text += char
        
        print(f"\nPoem {display_num} (Index {idx}):")
        print(f"Position in array: {start_pos}-{end_pos}")
        print(f"Raw indices (first 20): {poem_indices[:20].tolist()}")
        print(f"Valid indices (first 20): {valid_indices[:20].tolist()}")
        print(f"Full text: {poem_text}")
        print(f"Clean text: {clean_text}")
        print(f"Clean length: {len(clean_text)} characters")
        
        # 分析诗句结构（句子数量等）
        sentences = clean_text.split('。')
        comma_count = clean_text.count('，')
        period_count = clean_text.count('。')
        print(f"Structure: {period_count} sentences, {comma_count} commas")
    
    def create_visualizations(self, save_dir="visualization/export"):
        """Create and save analysis visualizations."""
        if not self.loaded:
            self.load_dataset()
            
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置学术风格
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # 1. 诗歌长度分布
        poem_lengths = self.analyze_poem_lengths()
        
        plt.figure(figsize=(10, 6))
        plt.hist(poem_lengths, bins=50, alpha=0.7, color=colors[0], edgecolor='black')
        plt.title('Distribution of Poem Lengths', fontsize=14, fontweight='bold')
        plt.xlabel('Poem Length (characters)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/poem_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 词汇频率分析
        char_counts = self.analyze_vocabulary()
        
        # 排除特殊标记，获取最常见的汉字
        regular_chars = {char: count for char, count in char_counts.items() 
                        if not (char.startswith('<') and char.endswith('>'))}
        
        top_chars = dict(Counter(regular_chars).most_common(30))
        
        plt.figure(figsize=(12, 8))
        chars = list(top_chars.keys())
        counts = list(top_chars.values())
        
        bars = plt.barh(range(len(chars)), counts, color=colors[1], alpha=0.8)
        plt.yticks(range(len(chars)), chars)
        plt.xlabel('Frequency', fontsize=12)
        plt.title('Top 30 Most Frequent Characters', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/character_frequency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 数据集概览
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 数据集基本信息
        ax1.text(0.1, 0.8, f"Total Poems: {len(self.data):,}", fontsize=14, transform=ax1.transAxes)
        ax1.text(0.1, 0.6, f"Vocabulary Size: {len(self.word2ix):,}", fontsize=14, transform=ax1.transAxes)
        ax1.text(0.1, 0.4, f"Avg Poem Length: {poem_lengths.mean():.1f}", fontsize=14, transform=ax1.transAxes)
        ax1.text(0.1, 0.2, f"Max Poem Length: {poem_lengths.max()}", fontsize=14, transform=ax1.transAxes)
        ax1.set_title('Dataset Overview', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 长度分布箱线图
        ax2.boxplot(poem_lengths, patch_artist=True, 
                   boxprops=dict(facecolor=colors[2], alpha=0.7))
        ax2.set_title('Poem Length Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Length (characters)')
        ax2.grid(True, alpha=0.3)
        
        # 词汇类型分布
        special_count = len([char for char in self.word2ix if char.startswith('<') and char.endswith('>')])
        regular_count = len(self.word2ix) - special_count
        
        labels = ['Regular Characters', 'Special Tokens']
        sizes = [regular_count, special_count]
        ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:2], startangle=90)
        ax3.set_title('Vocabulary Composition', fontsize=14, fontweight='bold')
        
        # 数据形状可视化
        ax4.bar(['Poems', 'Max Length'], [self.data.shape[0], self.data.shape[1]], 
               color=colors[3], alpha=0.8)
        ax4.set_title('Data Dimensions', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Count')
        for i, v in enumerate([self.data.shape[0], self.data.shape[1]]):
            ax4.text(i, v + max(self.data.shape)*0.02, f'{v:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/dataset_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to {save_dir}/")
    
    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("=" * 60)
        print("TANG POETRY DATASET ANALYSIS REPORT")
        print("=" * 60)
        
        self.analyze_basic_info()
        poem_lengths = self.analyze_poem_lengths()
        char_counts = self.analyze_vocabulary()
        self.sample_poems()
        
        print("\n=== Summary ===")
        print(f"Dataset is ready for training with {len(self.data)} poems")
        print(f"Vocabulary size of {len(self.word2ix)} is suitable for LSTM training")
        print(f"Fixed length of {self.data.shape[1]} characters per poem")
        
        # 创建可视化
        self.create_visualizations()

def main():
    """Main execution function."""
    analyzer = PoetryDatasetAnalyzer()
    analyzer.generate_report()

if __name__ == "__main__":
    main() 