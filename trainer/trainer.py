import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from datetime import datetime
from configs.config import ModelConfig, DataConfig, TrainConfig
from utils.utils import TrainVisualizer


class Trainer:
    
    def __init__(self, model, train_loader, val_loader):
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.train_config = TrainConfig()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.visualizer = TrainVisualizer() 
        
        self.model = model.to(self.train_config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 创建带时间戳的模型保存目录
        self.model_save_dir = os.path.join(
            self.train_config.save_dir, 
            self.timestamp
        )
        os.makedirs(self.model_save_dir, exist_ok=True)


        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay
        )
        
        if self.train_config.use_scheduler:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_config.epochs,
                eta_min=1e-6
            )
        
        self.criterion = nn.CrossEntropyLoss()
        
        os.makedirs(self.train_config.save_dir, exist_ok=True)
        
        self.best_val_acc = 0.0
    
    def train_epoch(self, epoch):
        """训练一个轮次"""
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(self.train_config.device), targets.to(self.train_config.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 打印中间结果
            if (batch_idx + 1) % self.train_config.log_interval == 0:
                avg_loss = train_loss / (batch_idx + 1)
                acc = 100.0 * correct / total
                pbar.set_description(f'Epoch {epoch+1} | Train Loss: {avg_loss:.3f} | Train Acc: {acc:.2f}%')
        
        return train_loss / len(self.train_loader), 100.0 * correct / total
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.train_config.device), targets.to(self.train_config.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return val_loss / len(self.val_loader), 100.0 * correct / total
    
    def train(self):
        print(f"开始训练，使用设备: {self.train_config.device}")
        
        for epoch in range(self.train_config.epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            if self.train_config.use_scheduler:
                self.scheduler.step()
            

            end_time = time.time()
            epoch_time = end_time - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1}/{self.train_config.epochs} | '
                  f'Time: {epoch_time:.1f}s | '
                  f'LR: {current_lr:.6f} | '
                  f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')
            
            self.visualizer.record(train_loss, train_acc, val_loss, val_acc, current_lr)

            # 保存最佳模型
            if val_acc > self.best_val_acc and self.train_config.save_best:
                self.best_val_acc = val_acc
                save_path = os.path.join(
                    self.model_save_dir, 
                    f'best_model_epoch_{epoch+1}_acc_{val_acc:.2f}.pth'
                )
                torch.save(self.model.state_dict(), save_path)
                print(f'模型已保存至: {save_path}')
        # 画图
        self.visualizer.save_all_plots()
        print(f'训练完成！最佳验证精度: {self.best_val_acc:.2f}%')    


def check_trainer():
    from models.vit import VisionTransformer
    from data.dataset import get_dataloaders
    
    model = VisionTransformer()
    train_loader, val_loader = get_dataloaders()
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train()