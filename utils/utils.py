import matplotlib.pyplot as plt
import os
from datetime import datetime
from configs.config import TrainConfig

plt.rcParams["font.family"] = ["Arial", "Helvetica", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class TrainVisualizer:
    def __init__(self):
        self.train_config = TrainConfig()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_save_dir()
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.lrs = []  # 记录学习率变化

    def _create_save_dir(self):
        """创建带时间戳的保存目录"""
        base_dir = self.train_config.save_result_plot
        self.save_dir = os.path.join(base_dir, self.timestamp)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"已创建图像保存目录：{self.save_dir}")


    def record(self, train_loss, train_acc, val_loss, val_acc, lr):
        """记录每个epoch的训练数据"""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.lrs.append(lr)

    def plot_loss_curve(self):
        """loss"""
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, label="Train Loss", marker="o", linestyle="-")
        plt.plot(epochs, self.val_losses, label="Val Loss", marker="s", linestyle="--")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(alpha=0.3)
        plt.legend()
        save_path = os.path.join(self.save_dir, "loss_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"损失曲线已保存至: {save_path}")

    def plot_acc_curve(self):
        """准确率"""
        epochs = range(1, len(self.train_accs) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_accs, label="Train Acc", marker="o", linestyle="-")
        plt.plot(epochs, self.val_accs, label="Val Acc", marker="s", linestyle="--")
        plt.ylim(70, 100)  
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Acc (%)")
        plt.grid(alpha=0.3)
        plt.legend()
        # 保存图像
        save_path = os.path.join(self.save_dir, "acc_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"准确率曲线已保存至: {save_path}")

    def plot_lr_curve(self):
        """绘制学习率变化曲线（如果使用调度器）"""
        if len(self.lrs) == 0:
            return
        epochs = range(1, len(self.lrs) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.lrs, label="Learning Rate", marker="^", color="orange")
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")  # 对数刻度，更清晰显示学习率衰减
        plt.grid(alpha=0.3)
        plt.legend()
        save_path = os.path.join(self.save_dir, "lr_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"学习率曲线已保存至: {save_path}")

    def save_all_plots(self):
        """一次性生成所有曲线"""
        self.plot_loss_curve()
        self.plot_acc_curve()
        self.plot_lr_curve()


