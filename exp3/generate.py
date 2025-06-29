import os
import sys
import argparse
import torch
from pathlib import Path
from typing import List, Optional, Dict
import json
import time

# 项目导入
sys.path.append('.')
from models.config import config
from models.model import create_poetry_transformer
from models.dataset import create_dataloaders
from utils.generate_utils import ConstrainedPoetryGenerator, QualityAssessment

class PoetryGenerationCLI:
    """命令行诗歌生成器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = config.device
        self.model_path = model_path or self._find_best_model()
        
        print(f"🚀 初始化诗歌生成器...")
        print(f"📱 设备: {self.device}")
        print(f"📁 模型路径: {self.model_path}")
        
        # 加载模型和词汇表
        self.model, self.vocab_info = self._load_model()
        
        # 创建生成器
        self.generator = ConstrainedPoetryGenerator(
            self.model, 
            self.vocab_info['ix2word'], 
            self.vocab_info['word2ix']
        )
        
        # 创建质量评估器
        self.assessor = QualityAssessment(
            self.vocab_info['ix2word'],
            self.vocab_info['word2ix']
        )
        
        print(f"✅ 初始化完成!")
    
    def _find_best_model(self) -> str:
        """寻找最佳训练模型"""
        possible_paths = [
            'checkpoints/server/best_model.pth',
            'checkpoints/development/best_model.pth', 
            'checkpoints/best_model.pth',
            'models/poetry_transformer_best.pth'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("未找到训练好的模型文件")
    
    def _load_model(self):
        """加载模型和词汇表"""
        print(f"📦 加载模型: {self.model_path}")
        
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 获取词汇表
        vocab_info = checkpoint.get('vocab_info')
        if vocab_info is None:
            print("📚 从数据集重新加载词汇表...")
            _, _, vocab_info = create_dataloaders(
                data_dir='data', batch_size=32, max_seq_len=config.MAX_SEQ_LEN,
                test_size=0.1, add_rhythmic_info=True
            )
        
        # 创建模型
        saved_config = checkpoint.get('config', {})
        model_config = {
            'vocab_size': saved_config.get('vocab_size', len(vocab_info['word2ix'])),
            'hidden_size': saved_config.get('hidden_size', saved_config.get('d_model', 576)),
            'num_layers': saved_config.get('num_layers', saved_config.get('n_layer', 12)),
            'num_heads': saved_config.get('num_heads', saved_config.get('n_head', 9)),
            'feedforward_dim': saved_config.get('feedforward_dim', saved_config.get('d_ff', 2304)),
            'max_seq_len': saved_config.get('max_seq_len', 125),
            'dropout': saved_config.get('dropout', 0.1)
        }
        
        model = create_poetry_transformer(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # 显示模型信息
        epoch = checkpoint.get('epoch', 'Unknown')
        metrics = checkpoint.get('metrics', {})
        perplexity = metrics.get('perplexity', 'Unknown')
        
        print(f"📊 模型信息:")
        print(f"   训练轮次: {epoch}")
        print(f"   困惑度: {perplexity}")
        print(f"   参数量: {model.get_num_params():,}")
        print(f"   词汇量: {len(vocab_info['word2ix'])}")
        
        return model, vocab_info
    
    def continue_poem(self, first_line: str, poem_type: str = "绝句", 
                     temperature: float = 0.8, top_k: int = 50, 
                     top_p: float = 0.9, num_attempts: int = 3) -> Dict:
        """续写诗歌"""
        print(f"\n🖋️  续写诗歌: {first_line}")
        print(f"📝 诗体类型: {poem_type}")
        
        # 分析格律
        chinese_chars = [c for c in first_line if '\u4e00' <= c <= '\u9fff']
        chars_per_line = len(chinese_chars)
        line_type = "五言" if chars_per_line == 5 else "七言"
        full_poem_type = f"{line_type}{poem_type}"
        
        print(f"🎯 检测格律: {chars_per_line}字/{line_type}")
        
        best_result = None
        best_score = 0
        attempts_results = []
        
        for attempt in range(num_attempts):
            print(f"🔄 尝试 {attempt + 1}/{num_attempts}")
            
            try:
                start_time = time.time()
                
                # 生成诗歌
                poem = self.generator.generate_constrained_poem(
                    prompt=first_line,
                    poem_type=full_poem_type,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                generation_time = time.time() - start_time
                
                # 评估质量
                quality_metrics = self.assessor.assess_poem_quality(poem)
                overall_score = quality_metrics['overall_score']
                
                attempts_results.append({
                    'poem': poem,
                    'score': overall_score,
                    'metrics': quality_metrics,
                    'time': generation_time
                })
                
                print(f"   评分: {overall_score:.2f} ({generation_time:.2f}s)")
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_result = {
                        'poem': poem,
                        'score': overall_score,
                        'metrics': quality_metrics,
                        'time': generation_time,
                        'attempt': attempt + 1
                    }
                    
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
                attempts_results.append({
                    'poem': None,
                    'error': str(e),
                    'score': 0
                })
        
        return {
            'first_line': first_line,
            'poem_type': full_poem_type,
            'chars_per_line': chars_per_line,
            'best_result': best_result,
            'all_attempts': attempts_results,
            'parameters': {
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'num_attempts': num_attempts
            }
        }
    
    def generate_acrostic(self, acrostic_chars: str, poem_type: str = "绝句",
                         temperature: float = 0.8, top_k: int = 50,
                         top_p: float = 0.9, num_attempts: int = 3) -> Dict:
        """生成藏头诗"""
        print(f"\n🎭 生成藏头诗: {acrostic_chars}")
        print(f"📝 诗体类型: {poem_type}")
        
        # 确定诗体格式
        char_count = len(acrostic_chars)
        if poem_type == "绝句" and char_count != 4:
            print(f"⚠️  警告: 绝句需要4个字符，实际{char_count}个，自动调整为律诗")
            poem_type = "律诗"
        elif poem_type == "律诗" and char_count != 8:
            print(f"⚠️  警告: 律诗需要8个字符，实际{char_count}个")
        
        # 默认使用五言格式
        line_type = "五言"
        full_poem_type = f"{line_type}{poem_type}"
        
        print(f"🎯 目标格式: {full_poem_type}")
        
        best_result = None
        best_score = 0
        attempts_results = []
        
        for attempt in range(num_attempts):
            print(f"🔄 尝试 {attempt + 1}/{num_attempts}")
            
            try:
                start_time = time.time()
                
                # 生成藏头诗
                poem = self.generator.generate_acrostic_poem(
                    acrostic_chars=acrostic_chars,
                    poem_type=full_poem_type,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                generation_time = time.time() - start_time
                
                # 验证藏头格式
                acrostic_valid = self._validate_acrostic(poem, acrostic_chars)
                
                # 评估质量
                quality_metrics = self.assessor.assess_poem_quality(poem)
                overall_score = quality_metrics['overall_score']
                
                # 藏头诗额外加分
                if acrostic_valid:
                    overall_score += 0.2  # 藏头正确额外加分
                
                attempts_results.append({
                    'poem': poem,
                    'score': overall_score,
                    'metrics': quality_metrics,
                    'acrostic_valid': acrostic_valid,
                    'time': generation_time
                })
                
                status = "✅" if acrostic_valid else "❌"
                print(f"   藏头验证: {status} 评分: {overall_score:.2f} ({generation_time:.2f}s)")
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_result = {
                        'poem': poem,
                        'score': overall_score,
                        'metrics': quality_metrics,
                        'acrostic_valid': acrostic_valid,
                        'time': generation_time,
                        'attempt': attempt + 1
                    }
                    
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
                attempts_results.append({
                    'poem': None,
                    'error': str(e),
                    'score': 0,
                    'acrostic_valid': False
                })
        
        return {
            'acrostic_chars': acrostic_chars,
            'poem_type': full_poem_type,
            'best_result': best_result,
            'all_attempts': attempts_results,
            'parameters': {
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'num_attempts': num_attempts
            }
        }
    
    def _validate_acrostic(self, poem: str, acrostic_chars: str) -> bool:
        """验证藏头诗格式"""
        lines = poem.strip().split('\n')
        if len(lines) != len(acrostic_chars):
            return False
        
        for i, line in enumerate(lines):
            if not line or line[0] != acrostic_chars[i]:
                return False
        return True
    
    def batch_generate(self, inputs: List[str], mode: str = "continue",
                      **kwargs) -> List[Dict]:
        """批量生成"""
        print(f"\n📦 批量生成 ({mode} 模式)")
        print(f"📊 数量: {len(inputs)}")
        
        results = []
        for i, input_text in enumerate(inputs, 1):
            print(f"\n--- 第 {i}/{len(inputs)} 项 ---")
            
            if mode == "continue":
                result = self.continue_poem(input_text, **kwargs)
            elif mode == "acrostic":
                result = self.generate_acrostic(input_text, **kwargs)
            else:
                raise ValueError(f"未知模式: {mode}")
            
            results.append(result)
        
        return results
    
    def save_results(self, results, output_file: str):
        """保存生成结果"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备保存的数据
        save_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_path': self.model_path,
            'results': results
        }
        
        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 结果已保存到: {output_path}")
    
    def print_result(self, result: Dict, mode: str = "continue"):
        """格式化打印结果"""
        print("\n" + "="*60)
        
        if mode == "continue":
            print(f"📝 续写结果")
            print(f"首句: {result['first_line']}")
            print(f"诗体: {result['poem_type']}")
        else:
            print(f"🎭 藏头诗结果")
            print(f"藏头: {result['acrostic_chars']}")
            print(f"诗体: {result['poem_type']}")
        
        if result['best_result']:
            best = result['best_result']
            print(f"最佳评分: {best['score']:.2f} (第{best['attempt']}次尝试)")
            print(f"生成时间: {best['time']:.2f}秒")
            
            if mode == "acrostic" and 'acrostic_valid' in best:
                status = "✅ 正确" if best['acrostic_valid'] else "❌ 错误"
                print(f"藏头验证: {status}")
            
            print(f"\n完整诗歌:")
            print(best['poem'])
            
            # 显示质量指标
            metrics = best['metrics']
            print(f"\n质量评估:")
            print(f"  句长规范: {metrics['length_score']:.2f}")
            print(f"  结构合理: {metrics['structure_score']:.2f}")
            print(f"  重复控制: {metrics['repetition_score']:.2f}")
            print(f"  语义连贯: {metrics['coherence_score']:.2f}")
            print(f"  整体质量: {metrics['overall_score']:.2f}")
        else:
            print("❌ 生成失败")
        
        print("="*60)

def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="诗歌生成器 - 支持续写和藏头诗生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 续写诗歌
  python generate.py continue "湖光秋月两相和"
  
  # 生成藏头诗
  python generate.py acrostic "春夏秋冬"
  
  # 批量续写
  python generate.py continue --batch inputs.txt --output results.json
  
  # 调整生成参数
  python generate.py continue "床前明月光" --temperature 0.9 --attempts 5
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest='mode', help='生成模式')
    
    # 续写模式
    continue_parser = subparsers.add_parser('continue', help='续写诗歌')
    continue_parser.add_argument('input', nargs='?', help='输入首句')
    continue_parser.add_argument('--poem-type', default='绝句',
                               choices=['绝句', '律诗'], help='诗体类型')
    
    # 藏头诗模式  
    acrostic_parser = subparsers.add_parser('acrostic', help='生成藏头诗')
    acrostic_parser.add_argument('input', nargs='?', help='藏头字符')
    acrostic_parser.add_argument('--poem-type', default='绝句',
                               choices=['绝句', '律诗'], help='诗体类型')
    
    # 通用参数
    for sub_parser in [continue_parser, acrostic_parser]:
        sub_parser.add_argument('--temperature', type=float, default=0.8,
                              help='生成温度 (0.1-2.0)')
        sub_parser.add_argument('--top-k', type=int, default=50,
                              help='Top-K采样参数')
        sub_parser.add_argument('--top-p', type=float, default=0.9,
                              help='Nucleus采样参数')
        sub_parser.add_argument('--attempts', type=int, default=3,
                              help='生成尝试次数')
        sub_parser.add_argument('--model', help='指定模型路径')
        sub_parser.add_argument('--batch', help='批量输入文件')
        sub_parser.add_argument('--output', help='结果输出文件')
    
    return parser

def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # 创建生成器
    try:
        generator = PoetryGenerationCLI(args.model)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 处理输入
    if args.batch:
        # 批量处理
        if not os.path.exists(args.batch):
            print(f"❌ 批量输入文件不存在: {args.batch}")
            return
        
        with open(args.batch, 'r', encoding='utf-8') as f:
            inputs = [line.strip() for line in f if line.strip()]
        
        if not inputs:
            print("❌ 批量输入文件为空")
            return
        
        # 批量生成
        results = generator.batch_generate(
            inputs, 
            mode=args.mode,
            poem_type=args.poem_type,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_attempts=args.attempts
        )
        
        # 打印简要结果
        print(f"\n📊 批量生成完成:")
        for i, result in enumerate(results, 1):
            best = result.get('best_result')
            if best:
                score = best['score']
                input_text = result.get('first_line') or result.get('acrostic_chars')
                print(f"  {i}. {input_text}: {score:.2f}")
            else:
                print(f"  {i}. 生成失败")
        
        # 保存结果
        if args.output:
            generator.save_results(results, args.output)
    
    else:
        # 单个生成
        if not args.input:
            input_text = input(f"请输入{'首句' if args.mode == 'continue' else '藏头字符'}: ").strip()
            if not input_text:
                print("❌ 输入不能为空")
                return
        else:
            input_text = args.input
        
        # 生成
        if args.mode == 'continue':
            result = generator.continue_poem(
                input_text,
                poem_type=args.poem_type,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_attempts=args.attempts
            )
        else:  # acrostic
            result = generator.generate_acrostic(
                input_text,
                poem_type=args.poem_type,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_attempts=args.attempts
            )
        
        # 打印结果
        generator.print_result(result, args.mode)
        
        # 保存结果
        if args.output:
            generator.save_results([result], args.output)

if __name__ == "__main__":
    main() 