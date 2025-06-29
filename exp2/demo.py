#!/usr/bin/env python3
"""
ViT-CIFAR10 单图片测试演示脚本
用于演示模型对单张图片的预测效果
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime

# 导入模型
from models import create_vit_small_patch8_224

class SingleImageTester:
    """单图片测试器"""
    
    def __init__(self, model_path='checkpoints/best_model.pth', device=None):
        """初始化测试器"""
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"模型加载完成，使用设备: {self.device}")
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        model = create_vit_small_patch8_224(num_classes=10)
        
        if os.path.exists(model_path):
            print(f"加载模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型加载成功 (训练轮数: {checkpoint.get('epoch', 'Unknown')})")
        else:
            print(f"警告: 未找到模型文件 {model_path}")
            print("将使用随机初始化的模型")
        
        model.to(self.device)
        model.eval()
        return model
    
    def load_image(self, image_path):
        """加载并预处理图片"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            
            # 预处理
            processed_image = self.transform(image).unsqueeze(0)  # 添加batch维度
            
            return original_image, processed_image
            
        except Exception as e:
            raise ValueError(f"图片加载失败: {e}")
    
    def predict(self, processed_image, top_k=5):
        """对图片进行预测"""
        with torch.no_grad():
            processed_image = processed_image.to(self.device)
            
            # 前向传播
            outputs = self.model(processed_image)
            probabilities = F.softmax(outputs, dim=1)
            
            # 获取Top-K预测
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            results = {
                'predicted_class': top_indices[0][0].item(),
                'predicted_name': self.class_names[top_indices[0][0].item()],
                'confidence': top_probs[0][0].item(),
                'top_k_predictions': []
            }
            
            # 添加Top-K结果
            for i in range(top_k):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                results['top_k_predictions'].append({
                    'class_idx': class_idx,
                    'class_name': self.class_names[class_idx],
                    'probability': prob
                })
            
            return results
    
    def visualize_prediction(self, original_image, results, save_path=None):
        """可视化预测结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 显示原图
        ax1.imshow(original_image)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 显示预测结果
        classes = [pred['class_name'] for pred in results['top_k_predictions']]
        probs = [pred['probability'] for pred in results['top_k_predictions']]
        
        # 创建颜色映射
        colors = ['#2E8B57' if i == 0 else '#4682B4' for i in range(len(classes))]
        
        bars = ax2.barh(range(len(classes)), probs, color=colors, alpha=0.8)
        ax2.set_yticks(range(len(classes)))
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Confidence', fontsize=12)
        ax2.set_title('Top-5 Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        
        # 添加数值标签
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{prob:.3f}', va='center', fontsize=10)
        
        # 高亮最高预测
        ax2.text(0.02, len(classes)-0.5, 
                f'Predicted: {results["predicted_name"]}\nConfidence: {results["confidence"]:.3f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果图已保存: {save_path}")
        
        plt.show()
    
    def test_single_image(self, image_path, save_result=True, show_plot=True):
        """测试单张图片"""
        print(f"\n开始测试图片: {image_path}")
        print("=" * 50)
        
        try:
            # 加载图片
            original_image, processed_image = self.load_image(image_path)
            print(f"图片加载成功，尺寸: {original_image.size}")
            
            # 预测
            results = self.predict(processed_image)
            
            # 显示结果
            print(f"\n预测结果:")
            print(f"  预测类别: {results['predicted_name']}")
            print(f"  置信度: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)")
            
            print(f"\nTop-5 预测:")
            for i, pred in enumerate(results['top_k_predictions'], 1):
                print(f"  {i}. {pred['class_name']:>10}: {pred['probability']:.4f} ({pred['probability']*100:.1f}%)")
            
            # 可视化
            if show_plot:
                save_path = None
                if save_result:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    save_path = f"result/prediction_{image_name}_{timestamp}.png"
                
                self.visualize_prediction(original_image, results, save_path)
            
            return results
            
        except Exception as e:
            print(f"测试失败: {e}")
            return None
    
    def batch_test(self, image_dir, pattern="*.jpg"):
        """批量测试图片"""
        import glob
        
        if not os.path.exists(image_dir):
            print(f"目录不存在: {image_dir}")
            return
        
        # 查找图片文件
        image_files = glob.glob(os.path.join(image_dir, pattern))
        if not image_files:
            print(f"在 {image_dir} 中未找到匹配 {pattern} 的图片文件")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始批量测试...")
        
        results_summary = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 测试: {os.path.basename(image_path)}")
            result = self.test_single_image(image_path, save_result=False, show_plot=False)
            
            if result:
                results_summary.append({
                    'image_path': image_path,
                    'predicted_class': result['predicted_name'],
                    'confidence': result['confidence']
                })
        
        # 显示批量测试总结
        print("\n" + "=" * 60)
        print("批量测试总结")
        print("=" * 60)
        
        for result in results_summary:
            image_name = os.path.basename(result['image_path'])
            print(f"{image_name:>20}: {result['predicted_class']:>10} ({result['confidence']:.3f})")
        
        return results_summary

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ViT-CIFAR10 单图片测试演示')
    parser.add_argument('--image', '-i', type=str, help='要测试的图片路径')
    parser.add_argument('--batch', '-b', type=str, help='批量测试的图片目录')
    parser.add_argument('--model', '-m', type=str, default='checkpoints/best_model.pth',
                       help='模型文件路径')
    parser.add_argument('--no-save', action='store_true', help='不保存预测结果图')
    parser.add_argument('--no-show', action='store_true', help='不显示图片')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.image and not args.batch:
        print("请指定要测试的图片 (--image) 或图片目录 (--batch)")
        print("使用 --help 查看详细帮助")
        return
    
    try:
        # 创建测试器
        tester = SingleImageTester(model_path=args.model)
        
        if args.image:
            # 单图片测试
            tester.test_single_image(
                args.image, 
                save_result=not args.no_save,
                show_plot=not args.no_show
            )
        
        if args.batch:
            # 批量测试
            tester.batch_test(args.batch)
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 