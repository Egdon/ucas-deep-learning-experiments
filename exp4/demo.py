#!/usr/bin/env python3
"""
神经机器翻译模型演示脚本
支持中英文双向翻译的交互式演示
"""

import sys
import os
import re

sys.path.append('en2zh')
sys.path.append('zh2en')

from test_model import TransformerNMT

def print_banner():
    """打印欢迎横幅"""
    print("=" * 60)
    print("神经机器翻译模型演示系统")
    print("基于Transformer架构的中英文翻译模型")
    print("=" * 60)
    print("使用说明:")
    print("- 输入中文，系统自动翻译为英文")
    print("- 输入英文，系统自动翻译为中文")
    print("- 输入 'quit' 或 'exit' 退出程序")
    print("- 输入 'help' 查看帮助信息")
    print("=" * 60)

def print_help():
    """打印帮助信息"""
    print("\n帮助信息:")
    print("1. 直接输入要翻译的文本，系统会自动检测语言")
    print("2. 支持中文到英文的翻译")
    print("3. 支持英文到中文的翻译")
    print("4. 输入 'quit', 'exit', 'q' 可退出程序")
    print("5. 输入 'clear' 可清屏")
    print("6. 输入 'help' 查看此帮助信息")
    print()

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def detect_language_simple(text):
    """简单的语言检测"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    if chinese_pattern.search(text):
        return 'zh', '中文'
    else:
        return 'en', '英文'

def main():
    """主函数"""
    try:
        print("正在初始化翻译模型...")
        model = TransformerNMT()
        print("模型初始化完成!\n")
        
        print_banner()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n请输入要翻译的文本: ").strip()
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("感谢使用神经机器翻译系统，再见!")
                    break
                elif user_input.lower() == 'help':
                    print_help()
                    continue
                elif user_input.lower() == 'clear':
                    clear_screen()
                    print_banner()
                    continue
                elif not user_input:
                    print("请输入有效的文本")
                    continue
                
                # 检测语言
                lang_code, lang_name = detect_language_simple(user_input)
                
                print(f"\n检测到输入语言: {lang_name}")
                print(f"原文: {user_input}")
                
                # 进行翻译
                print("正在翻译...")
                if lang_code == 'zh':
                    # 中译英
                    translation = model.translate_zh2en(user_input)
                    target_lang = "英文"
                else:
                    # 英译中
                    translation = model.translate_en2zh(user_input)
                    target_lang = "中文"
                
                # 显示结果
                print(f"翻译结果({target_lang}): {translation}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断")
                break
            except Exception as e:
                print(f"翻译过程中出现错误: {str(e)}")
                print("请重试或输入 'quit' 退出")
                
    except Exception as e:
        print(f"模型初始化失败: {str(e)}")
        print("请检查模型文件是否存在")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 