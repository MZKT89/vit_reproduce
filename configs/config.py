import torch

class ModelConfig:
    # Patch
    image_size = 28  # Fashion-MNIST 28 * 28
    patch_size = 4   # 拆成4 * 4  (28/4) ^ 2 = 49 个patch

    # embedding 
    in_channels = 1        # 输入图像通道数 (Fashion-MNIST是灰度图，单通道）
    hidden_size = 128  # 每个patch被embedded到128维

    # transformer
    num_layers = 2     # encoder 两层
    num_heads = 4      # 头数
    mlp_dim = 256      # MLP中间层维度（hidden_size的2倍，原论文中MLP维度是hidden_size的4倍
    # 正则化
    dropout_rate = 0.1 # Dropout概率
    # 任务参数
    num_classes = 10   # 分类数（Fashion-MNIST共10类）
    use_cls_token = True  # 是否使用[CLS] token


class DataConfig:
    data_dir = "./data/datasets"  
    mean = [0.5]       # 归一化均值（Fashion-MNIST是灰度图，单通道）
    std = [0.5]        # 标准差

    batch_size = 64    
    num_workers = 0   # mac好像有冲突


class TrainConfig:
    epochs = 20        # 训练总轮数
    device = "mps" if torch.backends.mps.is_available() else "cpu"  #

    learning_rate = 7e-4 # 3e-4
    weight_decay = 1e-4   

    use_scheduler = True
    scheduler_type = "CosineAnnealingLR"  # 这里硬编码了
    # log
    log_interval = 100  # 100 batch
    save_dir = "./checkpoints" 
    save_result_plot = './results' 
    save_best = True    


model_config = ModelConfig()
data_config = DataConfig()
train_config = TrainConfig()