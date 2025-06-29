import os
import json
import time
from datetime import datetime

class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir="logs", log_file="train_log.txt", metrics_file="metrics.json"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        self.metrics_file = os.path.join(log_dir, metrics_file)
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化指标存储
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # 写入开始时间
        self.log_message("=" * 80)
        self.log_message(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message("=" * 80)
    
    def log_message(self, message):
        """记录文本消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # 打印到控制台
        print(log_entry)
        
        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + '\n')
    
    def log_epoch(self, epoch, total_epochs, train_loss, val_loss, train_acc, val_acc, lr, epoch_time):
        """记录epoch训练结果"""
        # 添加到指标存储
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['val_acc'].append(val_acc)
        self.metrics['learning_rate'].append(lr)
        self.metrics['epoch_time'].append(epoch_time)
        
        # 格式化日志消息
        message = (f"Epoch: {epoch:03d}/{total_epochs:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Val Acc: {val_acc:.2f}% | "
                  f"LR: {lr:.2e} | "
                  f"Time: {epoch_time:.1f}min")
        
        self.log_message(message)
        
        # 保存指标到JSON文件
        self.save_metrics()
    
    def log_batch(self, epoch, batch_idx, total_batches, loss, acc, lr):
        """记录batch训练结果"""
        if batch_idx % 50 == 0:  # 每50个batch记录一次
            message = (f"Epoch {epoch} | "
                      f"Batch: {batch_idx:04d}/{total_batches:04d} | "
                      f"Loss: {loss:.4f} | "
                      f"Acc: {acc:.2f}% | "
                      f"LR: {lr:.2e}")
            self.log_message(message)
    
    def save_metrics(self):
        """保存指标到JSON文件"""
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_best_model(self, epoch, val_acc, model_path):
        """记录最佳模型保存"""
        message = f"New best model saved at epoch {epoch} with validation accuracy: {val_acc:.2f}% -> {model_path}"
        self.log_message(message)
    
    def log_early_stopping(self, epoch, patience):
        """记录早停"""
        message = f"Early stopping triggered at epoch {epoch} (patience: {patience})"
        self.log_message(message)
    
    def log_training_complete(self, total_time, best_val_acc):
        """记录训练完成"""
        self.log_message("=" * 80)
        self.log_message(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Total training time: {total_time/3600:.2f} hours")
        self.log_message(f"Best validation accuracy: {best_val_acc:.2f}%")
        self.log_message("=" * 80)
    
    def get_metrics(self):
        """获取所有指标"""
        return self.metrics.copy() 