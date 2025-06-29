# 实验1：MNIST手写数字识别

## 实验目标
使用卷积神经网络(CNN)对MNIST手写数字进行分类，目标准确率≥98%。

## 项目结构
```
exp1/
├── mnist_cnn.py          # 主程序
├── requirements.txt      # 依赖包
├── data/                 # 数据存放目录(自动创建)
├── models/               # 模型保存目录(自动创建)
├── results/              # 结果图表目录(自动创建)
└── README.md            # 说明文档
```

## 运行方法

1. 安装依赖包：
```bash
pip install -r requirements.txt
```

2. 下载数据集：
mnist下载链接：http://yann.lecun.com/exdb/mnist （下载后放在data目录下）

2. 运行训练：
```bash
python mnist_cnn.py
```

## 模型架构
- 输入：28x28灰度图像
- Conv1：1→16通道，5x5卷积核，ReLU激活，2x2最大池化
- Conv2：16→32通道，5x5卷积核，ReLU激活，2x2最大池化  
- FC：32*7*7→10输出(0-9类别)

## 训练参数
- 批大小：64
- 学习率：0.001
- 优化器：Adam
- 损失函数：交叉熵
- 训练轮数：15

## 输出文件
- `models/best_model.pth`：最佳模型
- `models/final_model.pth`：最终模型
- `results/training_history_*.png`：训练曲线图
- `results/test_samples_*.png`：测试样本展示 