# ViT-CIFAR10 图像分类项目

基于Vision Transformer (ViT)的CIFAR-10图像分类项目，使用ViT-Small/8架构实现。

## 文件结构

```
├── model.py                    # ViT模型定义
├── train.py                    # 训练脚本
├── test_model.py              # 模型测试脚本
├── demo_single_image.py       # 单图片演示脚本
├── checkpoints/               # 模型检查点目录
│   └── best_model.pth        # 最佳模型权重
├── data/                      # 数据集目录
│   └── CIFAR-10/             # CIFAR-10数据集
├── logs/                      # 训练日志和测试结果
│   ├── train_log_improved_*.txt
│   ├── metrics_improved_*.json
│   └── test_results_*.json
└── result/                    # 可视化结果
    ├── loss_curves.png
    ├── accuracy_curves.png
    ├── learning_rate.png
    ├── training_overview.png
    ├── overfitting_analysis.png
    ├── confusion_matrix.png
    ├── class_accuracy.png
    └── prediction_*.png
```

## 训练方法

### 数据集和模型下载
- 数据集：https://www.cs.toronto.edu/~kriz/cifar.html （下载后放在data目录下）
- 模型（下载后放在checkpoints/目录下）：
    - google drive: https://drive.google.com/file/d/1eKW1a4phe4uu5lfV-ZZrshpq3_xOma4K/view?usp=sharing 
    - 百度网盘: 链接: https://pan.baidu.com/s/1vWaXNeh-SMJJf36vtVYsIw 提取码: dtvc 

### 基本训练
```bash
python train.py
```

### 训练参数说明
- 模型架构: ViT-Small/8 (21,661,450参数)
- 输入尺寸: 224x224
- 批次大小: 64
- 学习率: 1e-3 (余弦退火调度)
- 训练轮数: 50
- 优化器: AdamW
- 数据增强: 随机裁剪、水平翻转、颜色抖动

### 训练输出
- 模型检查点保存在 `checkpoints/` 目录
- 训练日志保存在 `logs/` 目录
- 自动保存最佳验证准确率模型

## 测试方法

### 完整测试评估
```bash
python test_model.py
```

### 测试功能
- 加载最佳模型进行测试集评估
- 计算总体准确率和各类别准确率
- 生成混淆矩阵和分类报告
- 绘制准确率柱状图
- 保存详细测试结果到JSON文件

### 测试输出
- 控制台显示测试结果统计
- 混淆矩阵图: `result/confusion_matrix.png`
- 类别准确率图: `result/class_accuracy.png`
- 详细结果: `logs/test_results_*.json`

## 演示方法

### 单张图片预测
```bash
python demo_single_image.py --image path/to/image.jpg
```

### 批量图片测试
```bash
python demo_single_image.py --batch path/to/images/
```

### 演示参数
- `--image, -i`: 指定单张图片路径
- `--batch, -b`: 指定批量测试目录
- `--model, -m`: 指定模型文件路径 (默认: checkpoints/best_model.pth)
- `--no-save`: 不保存预测结果图
- `--no-show`: 不显示图片 (适合服务器环境)

### 演示功能
- 显示Top-5预测结果和置信度
- 可视化原图和预测结果柱状图
- 自动保存预测结果图到 `result/` 目录
- 支持JPEG、PNG等多种图片格式

### CIFAR-10类别
模型可识别10个类别: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### 使用示例
```bash
# 基本演示
python demo_single_image.py --image test_airplane.jpg

# 批量测试不显示图片
python demo_single_image.py --batch test_images/ --no-show

# 指定模型路径
python demo_single_image.py --image test.jpg --model checkpoints/best_model.pth
```
