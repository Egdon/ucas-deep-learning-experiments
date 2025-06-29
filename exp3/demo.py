import os
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional
import json

# 项目导入
sys.path.append('.')
from models.config import config
from models.model import create_poetry_transformer
from models.dataset import create_dataloaders  
from utils.generate_utils import ConstrainedPoetryGenerator, QualityAssessment

class PoetryDemoInterface:
    """诗歌生成交互演示界面"""
    
    def __init__(self):
        self.generator_cli = None
        self.session_results = []
        self.demo_start_time = time.time()
        
        # 预设测试用例
        self.preset_continue_cases = [
            "湖光秋月两相和",
            "床前明月光", 
            "白日依山尽",
            "春眠不觉晓",
            "独在异乡为异客",
            "红豆生南国"
        ]
        
        self.preset_acrostic_cases = [
            "春夏秋冬",
            "梅兰竹菊", 
            "琴棋书画",
            "诗酒花茶"
        ]
        
        # 参数预设
        self.parameter_presets = {
            "保守": {"temperature": 0.6, "top_k": 30, "top_p": 0.8},
            "标准": {"temperature": 0.8, "top_k": 50, "top_p": 0.9},
            "创新": {"temperature": 1.0, "top_k": 80, "top_p": 0.95}
        }
        
        print(self._get_welcome_banner())
    
    def _get_welcome_banner(self) -> str:
        """获取欢迎横幅"""
        return """
╔══════════════════════════════════════════════════════════════╗
║                    🎭 唐诗生成演示系统 🎭                    ║
║                                                              ║
║  基于 Transformer 架构的智能诗歌生成与约束解码算法           ║
║  支持诗歌续写和藏头诗生成，格律严格控制                      ║
║                                                              ║
║  Model: 57.5M Parameters Poetry Transformer                 ║
║  Features: ✓ 续写诗歌  ✓ 藏头诗  ✓ 格律控制  ✓ 质量评估     ║
╚══════════════════════════════════════════════════════════════╝
        """
    
    def initialize_system(self):
        """初始化系统"""
        print("\n🔧 正在初始化演示系统...")
        
        try:
            # 导入生成器类
            from generate import PoetryGenerationCLI
            
            print("📦 加载训练模型...")
            self.generator_cli = PoetryGenerationCLI()
            
            print("✅ 系统初始化完成!\n")
            return True
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            print("请确保:")
            print("  1. 训练模型文件存在")
            print("  2. 所有依赖包已安装")
            print("  3. 数据文件完整")
            return False
    
    def show_main_menu(self):
        """显示主菜单"""
        print("\n" + "="*70)
        print("🎯 请选择功能:")
        print("  1. 诗歌续写 - 根据首句续写完整诗歌")
        print("  2. 藏头诗生成 - 根据指定字符生成藏头诗") 
        print("  3. 预设测试 - 运行经典测试用例")
        print("  4. 系统信息 - 显示模型和环境信息")
        print("  0. 退出系统")
        print("="*70)
    
    def handle_continue_poem(self):
        """处理诗歌续写"""
        print("\n📝 诗歌续写模式")
        print("-" * 50)
        
        # 输入方式选择
        print("请选择输入方式:")
        print("  1. 手动输入首句")
        print("  2. 从预设用例选择")
        
        choice = input("选择 (1-2): ").strip()
        
        if choice == '1':
            first_line = input("请输入首句: ").strip()
            if not first_line:
                print("❌ 输入不能为空")
                return
        elif choice == '2':
            first_line = self._select_preset_case(self.preset_continue_cases, "续写用例")
            if not first_line:
                return
        else:
            print("❌ 无效选择")
            return
        
        # 诗体类型选择
        poem_type = self._select_poem_type()
        
        # 参数选择
        params = self._select_parameters()
        
        # 生成诗歌
        print(f"\n🚀 开始生成...")
        start_time = time.time()
        
        result = self.generator_cli.continue_poem(
            first_line=first_line,
            poem_type=poem_type,
            **params
        )
        
        total_time = time.time() - start_time
        
        # 显示结果
        self._display_result(result, "continue", total_time)
        
        # 记录到会话历史
        self.session_results.append({
            'type': 'continue',
            'input': first_line,
            'result': result,
            'timestamp': time.strftime('%H:%M:%S')
        })
    
    def handle_acrostic_poem(self):
        """处理藏头诗生成"""
        print("\n🎭 藏头诗生成模式")
        print("-" * 50)
        
        # 输入方式选择
        print("请选择输入方式:")
        print("  1. 手动输入藏头字符")
        print("  2. 从预设用例选择")
        
        choice = input("选择 (1-2): ").strip()
        
        if choice == '1':
            acrostic_chars = input("请输入藏头字符: ").strip()
            if not acrostic_chars:
                print("❌ 输入不能为空")
                return
        elif choice == '2':
            acrostic_chars = self._select_preset_case(self.preset_acrostic_cases, "藏头用例")
            if not acrostic_chars:
                return
        else:
            print("❌ 无效选择")
            return
        
        # 诗体类型选择
        poem_type = self._select_poem_type()
        
        # 参数选择
        params = self._select_parameters()
        
        # 生成藏头诗
        print(f"\n🚀 开始生成...")
        start_time = time.time()
        
        result = self.generator_cli.generate_acrostic(
            acrostic_chars=acrostic_chars,
            poem_type=poem_type,
            **params
        )
        
        total_time = time.time() - start_time
        
        # 显示结果
        self._display_result(result, "acrostic", total_time)
        
        # 记录到会话历史
        self.session_results.append({
            'type': 'acrostic',
            'input': acrostic_chars,
            'result': result,
            'timestamp': time.strftime('%H:%M:%S')
        })
    
    def handle_preset_tests(self):
        """处理预设测试"""
        print("\n🧪 预设测试模式")
        print("-" * 50)
        
        print("请选择测试类型:")
        print("  1. 续写测试 - 运行经典续写用例")
        print("  2. 藏头测试 - 运行藏头诗用例")
        
        choice = input("选择 (1-2): ").strip()
        
        if choice == '1':
            self._run_continue_tests()
        elif choice == '2':
            self._run_acrostic_tests()
        else:
            print("❌ 无效选择")
    
    def _run_continue_tests(self):
        """运行续写测试"""
        print("\n📋 运行续写测试用例...")
        
        test_cases = self.preset_continue_cases[:4]  # 选择前4个
        results = []
        
        for i, first_line in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i}/{len(test_cases)}: {first_line} ---")
            
            result = self.generator_cli.continue_poem(
                first_line=first_line,
                poem_type="绝句",
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                num_attempts=2
            )
            
            results.append(result)
            
            # 显示完整结果
            if result['best_result']:
                score = result['best_result']['score']
                poem = result['best_result']['poem']
                print(f"✅ 成功 - 评分: {score:.2f}")
                print(f"完整诗歌:")
                for line in poem.split('\n'):
                    if line.strip():
                        print(f"    {line}")
            else:
                print("❌ 失败")
        
        # 统计结果
        success_count = sum(1 for r in results if r['best_result'])
        avg_score = sum(r['best_result']['score'] for r in results if r['best_result']) / max(success_count, 1)
        
        print(f"\n📊 测试完成:")
        print(f"  成功率: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
        print(f"  平均评分: {avg_score:.2f}")
    
    def _run_acrostic_tests(self):
        """运行藏头诗测试"""
        print("\n📋 运行藏头诗测试用例...")
        
        test_cases = self.preset_acrostic_cases[:3]  # 选择前3个
        results = []
        
        for i, acrostic_chars in enumerate(test_cases, 1):
            print(f"\n--- 测试 {i}/{len(test_cases)}: {acrostic_chars} ---")
            
            result = self.generator_cli.generate_acrostic(
                acrostic_chars=acrostic_chars,
                poem_type="绝句",
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                num_attempts=2
            )
            
            results.append(result)
            
            # 显示完整结果
            if result['best_result']:
                score = result['best_result']['score']
                valid = result['best_result'].get('acrostic_valid', False)
                poem = result['best_result']['poem']
                status = "✅" if valid else "❌"
                print(f"{status} 评分: {score:.2f} - 藏头{'正确' if valid else '错误'}")
                print(f"\n📜 完整藏头诗:")
                print("    " + "─" * 20)
                for line in poem.split('\n'):
                    if line.strip():
                        print(f"    │ {line}")
                print("    " + "─" * 20)
            else:
                print("❌ 失败")
        
        # 统计结果
        success_count = sum(1 for r in results if r['best_result'])
        valid_count = sum(1 for r in results if r['best_result'] and r['best_result'].get('acrostic_valid', False))
        avg_score = sum(r['best_result']['score'] for r in results if r['best_result']) / max(success_count, 1)
        
        print(f"\n📊 测试完成:")
        print(f"  生成成功: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
        print(f"  藏头正确: {valid_count}/{len(test_cases)} ({valid_count/len(test_cases)*100:.1f}%)")
        print(f"  平均评分: {avg_score:.2f}")
    
    def show_system_info(self):
        """显示系统信息"""
        print("\n🖥️  系统信息")
        print("-" * 50)
        
        if self.generator_cli:
            print("模型信息:")
            print(f"  模型路径: {self.generator_cli.model_path}")
            print(f"  设备: {self.generator_cli.device}")
            print(f"  词汇量: {len(self.generator_cli.vocab_info['word2ix'])}")
        
        print(f"\n环境信息:")
        print(f"  Python版本: {sys.version.split()[0]}")
        print(f"  PyTorch版本: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        print(f"  运行环境: {config.ENVIRONMENT}")
        
        if torch.cuda.is_available():
            print(f"  GPU型号: {torch.cuda.get_device_name()}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        print(f"\n会话信息:")
        print(f"  启动时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.demo_start_time))}")
        print(f"  运行时长: {(time.time() - self.demo_start_time)/60:.1f}分钟")
        print(f"  操作次数: {len(self.session_results)}")
    
    def _select_preset_case(self, cases: List[str], case_type: str) -> Optional[str]:
        """选择预设用例"""
        print(f"\n{case_type}:")
        for i, case in enumerate(cases, 1):
            print(f"  {i}. {case}")
        
        try:
            choice = int(input(f"选择 (1-{len(cases)}): ").strip())
            if 1 <= choice <= len(cases):
                return cases[choice - 1]
            else:
                print("❌ 无效选择")
                return None
        except ValueError:
            print("❌ 请输入数字")
            return None
    
    def _select_poem_type(self) -> str:
        """选择诗体类型"""
        print("\n诗体类型:")
        print("  1. 绝句 (4句)")
        print("  2. 律诗 (8句)")
        
        choice = input("选择 (1-2, 默认绝句): ").strip()
        
        if choice == '2':
            return "律诗"
        else:
            return "绝句"
    
    def _select_parameters(self) -> Dict:
        """选择生成参数"""
        print("\n生成参数:")
        print("  1. 使用预设参数")
        print("  2. 自定义参数")
        
        choice = input("选择 (1-2, 默认预设): ").strip()
        
        if choice == '2':
            return self._custom_parameters()
        else:
            return self._preset_parameters()
    
    def _preset_parameters(self) -> Dict:
        """选择预设参数"""
        print("\n预设参数:")
        for i, (name, params) in enumerate(self.parameter_presets.items(), 1):
            print(f"  {i}. {name}: {params}")
        
        try:
            choice = int(input(f"选择 (1-{len(self.parameter_presets)}, 默认标准): ").strip() or '2')
            preset_names = list(self.parameter_presets.keys())
            if 1 <= choice <= len(preset_names):
                selected = preset_names[choice - 1]
                params = self.parameter_presets[selected].copy()
                params['num_attempts'] = 3
                return params
        except ValueError:
            pass
        
        # 默认返回标准参数
        params = self.parameter_presets["标准"].copy()
        params['num_attempts'] = 3
        return params
    
    def _custom_parameters(self) -> Dict:
        """自定义参数"""
        print("\n自定义参数 (按回车使用默认值):")
        
        try:
            temperature = float(input("温度 (0.1-2.0, 默认0.8): ").strip() or '0.8')
            top_k = int(input("Top-K (1-100, 默认50): ").strip() or '50')
            top_p = float(input("Top-P (0.1-1.0, 默认0.9): ").strip() or '0.9')
            attempts = int(input("尝试次数 (1-10, 默认3): ").strip() or '3')
            
            return {
                'temperature': max(0.1, min(2.0, temperature)),
                'top_k': max(1, min(100, top_k)),
                'top_p': max(0.1, min(1.0, top_p)),
                'num_attempts': max(1, min(10, attempts))
            }
        except ValueError:
            print("⚠️ 参数格式错误，使用默认值")
            return self.parameter_presets["标准"].copy()
    
    def _display_result(self, result: Dict, mode: str, total_time: float):
        """显示生成结果"""
        print("\n" + "🎉 生成完成!" + " "*50)
        print("="*70)
        
        if mode == "continue":
            print(f"📝 续写结果 - {result['first_line']}")
            print(f"🎭 诗体: {result['poem_type']}")
        else:
            print(f"🎭 藏头诗结果 - {result['acrostic_chars']}")
            print(f"📝 诗体: {result['poem_type']}")
        
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        
        if result['best_result']:
            best = result['best_result']
            print(f"🏆 最佳评分: {best['score']:.2f} (第{best['attempt']}次尝试)")
            print(f"⚡ 生成时间: {best['time']:.2f}秒")
            
            if mode == "acrostic" and 'acrostic_valid' in best:
                status = "✅ 正确" if best['acrostic_valid'] else "❌ 错误"
                print(f"🎯 藏头验证: {status}")
            
            print("\n📜 完整诗歌:")
            print("-" * 30)
            for line in best['poem'].split('\n'):
                if line.strip():
                    print(f"    {line}")
            print("-" * 30)
            
            # 显示质量评估
            metrics = best['metrics']
            print(f"\n📊 质量评估:")
            print(f"  📏 句长规范: {metrics['length_score']:.2f}/1.0")
            print(f"  🏗️  结构合理: {metrics['structure_score']:.2f}/1.0")
            print(f"  🔄 重复控制: {metrics['repetition_score']:.2f}/1.0")
            print(f"  🔗 语义连贯: {metrics['coherence_score']:.2f}/1.0")
            print(f"  ⭐ 整体质量: {metrics['overall_score']:.2f}/1.0")
        else:
            print("❌ 生成失败 - 请检查输入或调整参数")
        
        print("="*70)
    
    def run(self):
        """运行演示系统"""
        if not self.initialize_system():
            return
        
        while True:
            try:
                self.show_main_menu()
                choice = input("\n请选择功能 (0-4): ").strip()
                
                if choice == '0':
                    print("\n👋 感谢使用唐诗生成演示系统!")
                    print(f"本次会话共进行了 {len(self.session_results)} 次操作")
                    break
                elif choice == '1':
                    self.handle_continue_poem()
                elif choice == '2':
                    self.handle_acrostic_poem()
                elif choice == '3':
                    self.handle_preset_tests()
                elif choice == '4':
                    self.show_system_info()
                else:
                    print("❌ 无效选择，请重新输入")
                
                # 等待用户确认继续
                if choice in ['1', '2', '3']:
                    input("\n按回车继续...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，退出系统")
                break
            except Exception as e:
                print(f"\n❌ 系统错误: {e}")
                print("请重新选择功能")

def main():
    """主函数"""
    demo = PoetryDemoInterface()
    demo.run()

if __name__ == "__main__":
    main() 