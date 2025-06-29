import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积+激活+池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,       # 输入通道数(MNIST为灰度图，1通道)
                out_channels=16,     # 输出通道数(卷积核数量)
                kernel_size=5,       # 卷积核大小
                stride=1,            # 步长
                padding=2            # 填充数，使输出尺寸不变
            ),
            nn.ReLU(),               # 激活函数
            nn.MaxPool2d(2)          # 池化核大小 2x2
        )
        
        # 第二层卷积+激活+池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全连接层
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出10个类别(0-9)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平特征图为一维向量
        output = self.out(x)
        return output

class MNISTTrainer:
    def __init__(self, batch_size=64, learning_rate=0.001, epochs=15):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
               
        # 初始化数据加载器
        self.train_loader, self.test_loader = self.prepare_data()
        
        # 初始化模型
        self.model = CNN().to(self.device)
        
        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # 记录训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def prepare_data(self):
        """准备MNIST数据集"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
        ])
        
        # 下载并加载训练集
        train_dataset = torchvision.datasets.MNIST(
            root='./data/',
            train=True,
            transform=transform,
            download=True
        )
        
        # 下载并加载测试集
        test_dataset = torchvision.datasets.MNIST(
            root='./data/',
            train=False,
            transform=transform,
            download=True
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"使用设备: {self.device}")
        
        return train_loader, test_loader

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 每100个batch打印一次进度
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.6f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc

    def evaluate(self):
        """评估模型在测试集上的性能"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / total
        
        return test_loss, accuracy

    def train(self):
        """训练过程"""
        print("开始训练...")
        best_accuracy = 0.0
        
        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch+1}/{self.epochs}')
            print('-' * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 评估
            test_loss, test_acc = self.evaluate()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            print(f'训练损失: {train_loss:.6f}, 训练准确率: {train_acc:.2f}%')
            print(f'测试损失: {test_loss:.6f}, 测试准确率: {test_acc:.2f}%')
            
            # 保存最佳模型
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.save_model('best_model.pth')
                print(f'保存最佳模型，准确率: {best_accuracy:.2f}%')
        
        print(f'\n训练完成！最佳测试准确率: {best_accuracy:.2f}%')
        
        # 保存最终模型和绘制结果
        self.save_model('final_model.pth')
        self.plot_training_history()
        
        return best_accuracy

    def save_model(self, filename):
        """保存模型"""
        filepath = os.path.join('models', filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies
        }, filepath)

    def load_model(self, filename):
        """加载模型"""
        filepath = os.path.join('models', filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.test_accuracies = checkpoint['test_accuracies']

    def plot_training_history(self):
        """绘制训练历史图表"""
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 训练准确率曲线
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accuracies, 'r-', label='Training Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # 测试准确率曲线
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.test_accuracies, 'g-', label='Test Accuracy')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'results/training_history_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def test_samples(self, num_samples=10):
        """测试部分样本并显示结果"""
        self.model.eval()
        data_iter = iter(self.test_loader)
        images, labels = next(data_iter)
        
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
        
        # 显示前num_samples个样本
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i in range(num_samples):
            row = i // 5
            col = i % 5
            
            image = images[i].cpu().numpy().squeeze()
            true_label = labels[i].item()
            pred_label = predicted[i].cpu().item()
            
            axes[row, col].imshow(image, cmap='gray')
            axes[row, col].set_title(f'True: {true_label}, Pred: {pred_label}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'results/test_samples_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    print("MNIST手写数字识别实验")
    print("=" * 50)
    
    # 创建训练器
    trainer = MNISTTrainer(
        batch_size=64,
        learning_rate=0.001,
        epochs=15
    )
    
    # 开始训练
    best_accuracy = trainer.train()
    
    # 测试样本
    print("\n显示测试样本...")
    trainer.test_samples()

    print(f"\n最高准确率: {best_accuracy:.2f}%")
   
if __name__ == "__main__":
    main()