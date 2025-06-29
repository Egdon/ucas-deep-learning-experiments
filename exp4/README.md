# 双向中英神经机器翻译系统

基于Transformer架构的中英文双向神经机器翻译系统，支持英译中和中译英两个翻译方向。

## 项目概述

本项目实现了一个完整的神经机器翻译系统，包含以下特性：

- **双向翻译**：支持英文→中文和中文→英文翻译
- **Transformer架构**：采用标准的6层Encoder-Decoder结构
- **独立模型设计**：每个翻译方向使用独立训练的模型
- **完整工具链**：包含训练、测试、评估和演示功能
- **自动语言检测**：演示系统能自动识别输入语言并选择相应模型

## 文件结构

```
exp4/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖包列表
├── demo.py                      # 交互式翻译演示脚本
├── test_model.py               # 模型测试和评估脚本
│
├── data/                       # 数据目录
│   ├── corpus.en               # 英文语料文件
│   ├── corpus.ch               # 中文语料文件
│   ├── get_corpus.py           # 语料处理脚本
│   └── json/                   # JSON格式数据
│       ├── train.json          # 训练数据
│       ├── dev.json            # 验证数据
│       └── test.json           # 测试数据
│
├── tokenizer/                  # 分词器目录
│   ├── eng.model               # 英文SentencePiece模型
│   ├── eng.vocab               # 英文词汇表
│   ├── chn.model               # 中文SentencePiece模型
│   ├── chn.vocab               # 中文词汇表
│   └── tokenize.py             # 分词器工具脚本
│
├── checkpoint/                 # 训练好的模型
│   ├── en2zh.pth               # 英译中模型权重
│   └── zh2en.pth               # 中译英模型权重
│
├── en2zh/                      # 英译中模型模块
│   ├── config.py               # 英译中模型配置
│   ├── data_loader.py          # 数据加载器
│   ├── train.py                # 训练脚本
│   ├── main.py                 # 主训练程序
│   ├── run_training.py         # 训练启动脚本
│   ├── utils.py                # 工具函数
│   ├── train_logger.py         # 训练日志记录
│   ├── test_translation.py     # 翻译测试脚本
│   ├── model/                  # 模型定义
│   │   ├── __init__.py
│   │   └── transformer.py     # Transformer模型实现
│   ├── tokenizer/              # 模型专用分词器链接
│   └── experiment/             # 训练实验记录
│
├── zh2en/                      # 中译英模型模块
│   ├── config.py               # 中译英模型配置
│   ├── data_loader.py          # 数据加载器
│   ├── train.py                # 训练脚本
│   ├── main.py                 # 主训练程序
│   ├── run_training.py         # 训练启动脚本
│   ├── utils.py                # 工具函数
│   ├── train_logger.py         # 训练日志记录
│   ├── test_translation.py     # 翻译测试脚本
│   ├── model/                  # 模型定义
│   │   ├── __init__.py
│   │   └── transformer.py     # Transformer模型实现
│   ├── tokenizer/              # 模型专用分词器链接
│   └── experiment/             # 训练实验记录
│
└── result/                     # 结果输出目录
    └── (测试结果和日志文件)
```

## 如何训练

### 环境准备

1. **安装依赖包**：
```bash
pip install -r requirements.txt
```

2. **检查CUDA环境**（推荐使用GPU训练）：
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 数据准备

数据集下载：
   - google drive: https://drive.google.com/file/d/1mnSYJiuOvLNmkxQFdt42-sv3xAlIlVKv/view?usp=sharing
   - 百度网盘: https://pan.baidu.com/s/1NykJ6VdSRklAC0g3ru-USA 提取码: x87y

项目已包含预处理好的数据：
   - 训练数据：`data/json/train.json`（176,943个句对）
   - 验证数据：`data/json/dev.json`
   - 测试数据：`data/json/test.json`
   - SentencePiece分词器：`tokenizer/`目录下的模型和词汇文件


### 模型准备

- 英译中模型：`checkpoint/en2zh.pth`
  - google drive: https://drive.google.com/file/d/1s2evo0ZXb78ZlbqtyOcI7hC8THIJ8IxK/view?usp=sharing
  - 百度网盘: https://pan.baidu.com/s/1R4_u5mfnAEbO-OARsDXukA 提取码: y4p6 
- 中译英模型：`checkpoint/zh2en.pth`
  - google drive: https://drive.google.com/file/d/1pC2o8-4ZvGvXHoKNTkIeeNS8-sjWYxg4/view?usp=sharing
  - 百度网盘: https://pan.baidu.com/s/1Jt4NwAexZCbhS42Q1h0sNA 提取码: x32e 

### 训练英译中模型

1. **进入英译中目录**：
```bash
cd en2zh
```

2. **启动训练**：
```bash
python run_training.py
```

3. **监控训练进度**：
   - 训练日志会实时显示在终端
   - 模型会自动保存到 `../checkpoint/en2zh.pth`
   - 训练记录保存在 `experiment/` 目录

### 训练中译英模型

1. **进入中译英目录**：
```bash
cd zh2en
```

2. **启动训练**：
```bash
python run_training.py
```

3. **训练配置**：
   - 自动使用最佳可用GPU
   - 默认训练40个epoch，早停机制为5个epoch
   - 使用NoamOpt优化器，warmup步数为10,000
   - 批量大小：32（可在config.py中调整）

### 训练参数说明

主要超参数（在各自的config.py中配置）：
- `d_model=512`：模型维度
- `n_heads=8`：注意力头数
- `n_layers=6`：编码器/解码器层数
- `d_ff=2048`：前馈网络维度
- `dropout=0.1`：丢弃率
- `max_epochs=40`：最大训练轮数

## 如何使用test_model.py

`test_model.py` 是模型测试和评估脚本，提供完整的模型性能测试功能。

### 主要功能

1. **模型加载和初始化**：自动加载训练好的英译中和中译英模型
2. **BLEU评分计算**：使用标准的BLEU-4指标评估翻译质量
3. **双向翻译测试**：分别测试英译中和中译英性能
4. **样本翻译展示**：显示具体的翻译样例
5. **结果保存**：将评估结果保存到文件

### 使用方法

#### 基本测试
```bash
python test_model.py
```

#### 命令行参数
```bash
python test_model.py [选项]

可选参数：
  --en2zh_model PATH     英译中模型路径 (默认: checkpoint/en2zh.pth)
  --zh2en_model PATH     中译英模型路径 (默认: checkpoint/zh2en.pth)
  --sample_size N        测试样本数量 (默认: 使用全部数据)
  --output_file PATH     结果保存文件名
```

#### 使用示例

1. **完整评估**（使用全部测试数据）：
```bash
python test_model.py
```

2. **快速测试**（使用1000个样本）：
```bash
python test_model.py --sample_size 1000
```

3. **指定模型路径**：
```bash
python test_model.py --en2zh_model my_models/en2zh_best.pth --zh2en_model my_models/zh2en_best.pth
```

4. **保存结果到指定文件**：
```bash
python test_model.py --output_file my_evaluation_results.json
```

### 输出说明

测试脚本会输出以下信息：

1. **模型加载信息**：
   - 设备信息（CPU/GPU）
   - 模型参数数量
   - 加载状态

2. **翻译样例展示**：
   - 随机选择的翻译样例
   - 原文、参考翻译、模型翻译对比

3. **性能评估结果**：
   - 英译中BLEU-4分数
   - 中译英BLEU-4分数
   - 整体平均BLEU分数

4. **详细统计**：
   - 测试样本数量
   - 翻译时间统计
   - 各方向性能分析

### 评估指标

- **BLEU-4分数**：使用标准的BLEU-4指标，范围0-100
- **目标分数**：每个方向的BLEU-4分数应≥14
- **计算方法**：使用sacrebleu库进行标准化计算

## 如何使用demo.py

`demo.py` 是交互式翻译演示脚本，提供用户友好的翻译体验。

### 主要功能

1. **自动语言检测**：根据输入文本自动识别中文或英文
2. **智能翻译选择**：自动选择合适的翻译模型
3. **实时翻译**：输入文本后即时显示翻译结果
4. **交互式界面**：提供清晰的用户界面和操作提示

### 启动方法

```bash
python demo.py
```

### 使用说明

1. **启动程序**：
   - 运行命令后，系统会自动加载两个翻译模型
   - 显示欢迎界面和使用说明

2. **进行翻译**：
   - 在提示符后输入要翻译的文本
   - 系统自动检测语言（中文/英文）
   - 显示检测结果和翻译结果

3. **交互命令**：
   - `help`：显示帮助信息
   - `clear`：清屏
   - `quit`、`exit`、`q`：退出程序

### 使用示例

```
请输入要翻译的文本: Hello, how are you?

检测到输入语言: 英文
原文: Hello, how are you?
正在翻译...
翻译结果(中文): 你好，你好吗？
--------------------------------------------------

请输入要翻译的文本: 今天天气真好

检测到输入语言: 中文  
原文: 今天天气真好
正在翻译...
翻译结果(英文): The weather is really good today
--------------------------------------------------
```

