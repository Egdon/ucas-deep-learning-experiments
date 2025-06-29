import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
# PyTorch 2.1.2 兼容性处理
try:
    # PyTorch >= 2.0
    from torch.amp import autocast, GradScaler
    # 检查autocast是否支持device_type参数
    import inspect
    autocast_signature = inspect.signature(autocast.__init__)
    if 'device_type' in autocast_signature.parameters:
        # 新版本API
        def create_autocast():
            return autocast(device_type='cuda', dtype=torch.float16)
    else:
        # 旧版本API  
        def create_autocast():
            return autocast()
except ImportError:
    try:
        # PyTorch 1.6-1.9
        from torch.cuda.amp import autocast, GradScaler
        def create_autocast():
            return autocast()
    except ImportError:
        # 不支持混合精度，创建占位符
        class autocast:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        
        class GradScaler:
            def __init__(self):
                pass
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                return optimizer.step()
            def update(self):
                pass
            def unscale_(self, optimizer):
                pass
        
        def create_autocast():
            return autocast()
from tqdm import tqdm

class Trainer:
    """ViT训练器"""
    
    def __init__(self, model, train_loader, val_loader, config, logger, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = device
        
        # 设置优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # 设置学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
        
        # 设置损失函数（带标签平滑）
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config['label_smoothing']
        )
        
        # 混合精度训练
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # 训练状态
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
        self.start_time = time.time()
        
        # 创建检查点目录
        os.makedirs(config.get('checkpoint_dir', 'checkpoints'), exist_ok=True)
    
    def warmup_scheduler(self, epoch, warmup_epochs=10):
        """学习率预热"""
        if epoch < warmup_epochs:
            lr = self.config['learning_rate'] * (epoch + 1) / warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度条，优化显示
        pbar = tqdm(self.train_loader, 
                   desc=f'Epoch {epoch}', 
                   leave=True,  # 保留进度条避免重叠
                   dynamic_ncols=True,  # 动态调整宽度
                   ascii=True,  # 使用ASCII字符
                   position=0)  # 固定位置
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with create_autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # 反向传播（混合精度）
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip_norm']
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip_norm']
                )
                
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # 更新进度条
            acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录batch日志
            if batch_idx % self.config.get('log_frequency', 50) == 0:
                self.logger.log_batch(
                    epoch, batch_idx, len(self.train_loader),
                    loss.item(), acc, self.optimizer.param_groups[0]['lr']
                )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, 
                       desc='Validating', 
                       leave=True,
                       dynamic_ncols=True,
                       ascii=True,
                       position=1)  # 不同位置避免冲突
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_mixed_precision:
                    with create_autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # 更新进度条
                acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{acc:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """保存检查点"""
        checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'config': self.config
        }
        
        # 保存最新模型
        latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        torch.save(state, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(state, best_path)
            self.logger.log_best_model(epoch, val_acc, best_path)
        
        # 定期保存检查点
        if epoch % self.config.get('checkpoint_frequency', 10) == 0:
            epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pth')
            torch.save(state, epoch_path)
    
    def train(self):
        """完整训练流程"""
        self.logger.log_message("Starting training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start_time = time.time()
            
            # 学习率预热
            if epoch <= self.config.get('warmup_epochs', 10):
                self.warmup_scheduler(epoch, self.config.get('warmup_epochs', 10))
            else:
                self.scheduler.step()
            
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 计算epoch时间
            epoch_time = (time.time() - epoch_start_time) / 60.0
            
            # 记录日志
            self.logger.log_epoch(
                epoch, self.config['epochs'],
                train_loss, val_loss, train_acc, val_acc,
                self.optimizer.param_groups[0]['lr'], epoch_time
            )
            
            # 检查是否为最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, val_acc, is_best)
            
            # 早停检查
            patience = self.config.get('early_stopping_patience', 15)
            if self.early_stopping_counter >= patience:
                self.logger.log_early_stopping(epoch, patience)
                break
        
        # 训练完成
        total_time = time.time() - self.start_time
        self.logger.log_training_complete(total_time, self.best_val_acc)
        
        return self.best_val_acc 