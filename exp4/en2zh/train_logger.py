import json
import csv
import os
from datetime import datetime


class TrainingLogger:
    """训练过程日志记录器"""
    
    def __init__(self, log_dir="./experiment"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化记录文件
        self.metrics_file = os.path.join(log_dir, "training_metrics.csv")
        self.detailed_log = os.path.join(log_dir, "training_detailed.json")
        
        # 初始化数据存储
        self.training_data = {
            "start_time": datetime.now().isoformat(),
            "epochs": [],
            "config": {}
        }
        
        # 创建CSV文件并写入表头
        self._init_csv()
    
    def _init_csv(self):
        """初始化CSV文件"""
        with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'train_loss', 
                'val_loss',
                'bleu_score',
                'learning_rate',
                'epoch_time',
                'best_bleu',
                'early_stop_count'
            ])
    
    def log_config(self, config_dict):
        """记录训练配置"""
        self.training_data["config"] = config_dict
        self._save_detailed_log()
    
    def log_epoch(self, epoch, train_loss, val_loss, bleu_score, learning_rate, 
                  epoch_time, best_bleu, early_stop_count):
        """记录每个epoch的训练数据"""
        
        # 确保所有数值都是Python原生类型
        def convert_to_python_type(value):
            """将PyTorch tensor或numpy类型转换为Python原生类型"""
            if hasattr(value, 'item'):  # PyTorch tensor
                return value.item()
            elif hasattr(value, 'tolist'):  # numpy array
                return value.tolist()
            else:
                return float(value)
        
        # 转换所有数值类型
        train_loss = convert_to_python_type(train_loss)
        val_loss = convert_to_python_type(val_loss)
        bleu_score = convert_to_python_type(bleu_score)
        learning_rate = convert_to_python_type(learning_rate)
        epoch_time = convert_to_python_type(epoch_time)
        best_bleu = convert_to_python_type(best_bleu)
        early_stop_count = int(early_stop_count)
        
        # 记录到CSV
        with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}", 
                f"{bleu_score:.4f}",
                f"{learning_rate:.8f}",
                f"{epoch_time:.2f}",
                f"{best_bleu:.4f}",
                early_stop_count
            ])
        
        # 记录到详细日志
        epoch_data = {
            "epoch": int(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "bleu_score": bleu_score,
            "learning_rate": learning_rate,
            "epoch_time": epoch_time,
            "best_bleu": best_bleu,
            "early_stop_count": early_stop_count,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_data["epochs"].append(epoch_data)
        self._save_detailed_log()
        
        print(f"📊 Epoch {epoch} 数据已记录到日志文件")
    
    def log_training_complete(self, total_time, final_bleu, model_path):
        """记录训练完成信息"""
        completion_data = {
            "end_time": datetime.now().isoformat(),
            "total_training_time": total_time,
            "final_bleu_score": final_bleu,
            "model_saved_path": model_path,
            "total_epochs": len(self.training_data["epochs"])
        }
        
        self.training_data["training_summary"] = completion_data
        self._save_detailed_log()
        
        # 创建训练总结文件
        summary_file = os.path.join(self.log_dir, "training_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("训练完成总结\n")
            f.write("=" * 50 + "\n")
            f.write(f"开始时间: {self.training_data['start_time']}\n")
            f.write(f"结束时间: {completion_data['end_time']}\n")
            f.write(f"总训练时间: {total_time:.2f} 秒\n")
            f.write(f"总epoch数: {completion_data['total_epochs']}\n")
            f.write(f"最终BLEU分数: {final_bleu:.4f}\n")
            f.write(f"模型保存路径: {model_path}\n")
            f.write(f"指标文件: {self.metrics_file}\n")
            f.write(f"详细日志: {self.detailed_log}\n")
        
        print(f"📋 训练总结已保存到: {summary_file}")
    
    def _save_detailed_log(self):
        """保存详细的JSON日志"""
        with open(self.detailed_log, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
    
    def get_metrics_file(self):
        """获取指标CSV文件路径"""
        return self.metrics_file
    
    def get_detailed_log_file(self):
        """获取详细日志文件路径"""
        return self.detailed_log 