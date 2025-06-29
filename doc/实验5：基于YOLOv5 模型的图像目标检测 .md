
基于YOLOv5 模型的图像目标检测 
实验指导书 
一、实验目的 
本实验旨在使用 YOLOv5 目标检测模型进行图像中的目标检测。
YOLOv5 是一种流行的目标检测模型，采用了单阶段（one-stage）检
测方法，具有极高的检测速度和良好的检测性能。 
（1）理解 YOLOv5 模型的基本原理和工作机制。  
（2）掌握 YOLOv5 模型的部署和使用方法。  
（3）进行基于 YOLOv5 模型的目标检测实验，包括模型训练、推理和评
估。  
二、实验要求 
（1）基于 Python 语言和任意一种深度学习框架（实验指导书中使用 
PyTorch 框架进行介绍），从零开始完成数据读取、网络构建、模型训
练和模型测试等过程，最终实现一个可以完成基于 YOLOv5 模型的图像
目标检测的程序。 
（2）在自定义数据集上进行训练和评估，实现测试集检测精度达到
90%以上。 
（3）在规定时间内提交实验报告、代码和 PPT。 
三、YOLOv5 模型原理 
YOLOv5 主要分为输入端，backbone、Neck 和 head(prediction)。
backbone 是 New CSP-Darknet53。Neck 层为 SPFF 和 New CSP-PAN。
Head 层为 YOLOv3 head。 
YOLOv5 6.0 版本的主要架构如下图所示： 
 
四、实验所需工具 
在开始实验之前，请确保已经完成以下准备工作：  
（1）安装 Python 虚拟环境（建议使用 Anaconda）。  
（2）安装 PyTorch 和 torchvision 库。 
（3）下载 YOLOv5 源代码包。  
五、实验步骤  
（以下步骤为 YOLOv5 安装配置及训练推理的详细步骤）。   
注意：在运行 YOLOv5 前，先要安装好 Anaconda 环境，具体操作可
参考下述 Anaconda3 的安装配置及使用教程（详细过程）： 
https://howiexue.blog.csdn.net/article/details/118442904。  
YOLOv5 官方指南：https://docs.ultralytics.com/quick-
start/。 
5.1. 下载 YOLOv5 
5.1.1 下载 YOLOv5 源码 
 Github 地址：https://github.com/ultralytics/YOLOv5  
  
命令行 git clone 到本地工作目录，等待下载完成：  
git clone https://github.com/ultralytics/YOLOv5  
 
 
YOLOv5 代码目录架构：  
  
 
5.1.2 下载 YOLOv5 预训练模型 
可以直接采用已经初始训练好的权重，不需要在本机再做训练。  
下载地址：https://github.com/ultralytics/YOLOv5/releases。  
  
