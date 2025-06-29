# 国科大深度学习实验作业-UCAS Deep Learning Experiments 🧠

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 比较菜，只做了五个，搞不动了

## 实验模块

| 实验 | 技术栈 | 数据集 | 性能指标 | 关键特性 |
|------|--------|--------|----------|----------|
| **手写数字识别** | CNN + PyTorch | MNIST | 准确率 98%+ | 经典卷积神经网络，深度学习入门 |
| **图像分类** | Vision Transformer | CIFAR-10 | 准确率 80%+ | 前沿ViT架构，Attention机制应用 |
| **自动写诗** | LSTM + Embedding | 唐诗数据集 | 流畅度评估 | 序列生成，中文NLP处理 |
| **神经机器翻译** | Transformer | 中英平行语料 | BLEU4 > 25 | Seq2Seq架构，双向注意力机制 |
| **目标检测** | YOLOv5 | 自定义数据集 | mAP 90%+ | 实时检测，端到端训练 |


## 🏗️ 项目结构

```
deep-learning-experiments/
├── exp1/ - 手写数字识别 (CNN)
├── exp2/ - 图像分类 (ViT)
├── exp3/ - 自动写诗 (LSTM)
├── exp4/ - 机器翻译 (Transformer)
├── exp5/ - 目标检测 (YOLOv5)
├── doc/ - 实验指导书markdown版本
├── 实验要求/ - 课程要求和评估标准
└── 报告&PPT/ - 实验报告和演示文稿
```

每个实验文件夹包含：
- `train.py` - 模型训练脚本
- `test.py` / `demo.py` - 模型测试和演示
- `models/` - 网络架构定义
- `data/` - 数据加载和预处理
- `utils/` - 工具函数和辅助代码
- `README.md` - 具体实验说明

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 10.2+ (可选，用于GPU加速)

### 安装依赖
```bash
# 克隆项目 
git clone https://github.com/your-username/deep-learning-experiments.git
cd deep-learning-experiments

# 为每个实验安装对应依赖
cd 代码/exp1
pip install -r requirements.txt
```

### 运行示例
```bash
# 手写数字识别
cd 代码/exp1
python mnist_cnn.py

# 图像分类
cd ../exp2
python train.py

# 自动写诗
cd ../exp3
python demo.py --start_words "湖光秋月两相和"
```

## 📊 性能指标

| 实验 | 指标类型 | 目标值 | 实际达成 | 备注 |
|------|----------|--------|----------|------|
| MNIST识别 | 测试准确率 | ≥98% | 98.2% | 在测试集上的分类准确率 |
| CIFAR-10分类 | 测试准确率 | ≥80% | 82.1% | ViT模型在CIFAR-10上的性能 |
| 自动写诗 | 困惑度 | ≤10 | 3.58 | 诗歌生成质量评估 |
| 机器翻译 | BLEU4 | >14 | 16.3 | 中英文翻译质量评估 |
| 目标检测 | mAP | ≥90% | 92.5% | 自定义数据集上的检测精度 |

