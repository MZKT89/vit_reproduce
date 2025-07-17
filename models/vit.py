import torch
import torch.nn as nn
import torch.nn.functional as F


from configs.config import ModelConfig

model_config = ModelConfig()

'''
cls token
PatchEmbedding
TransformerBlock (多头注意力 + MLP)
'''

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # image：28*28，patch：4*4
        self.image_size = model_config.image_size
        self.patch_size = model_config.patch_size
        
        # Patch数量：(28/4)^2 = 7×7 = 49个Patch
        self.num_patches = (self.image_size // self.patch_size) ** 2
        

        ''' Google 实现
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=self.patches.size,
            strides=self.patches.size,
            padding='VALID',
            name='embedding')(
                x)
        '''
        # 用卷积层实现 分块+嵌入
        # kernel=patch（4×4），step=patch（确保不重叠）
        # 输出通道数=hidden_size（128），相当于每个Patch被映射为128维向量
        # 维度变化 (in - k + 2*pad) / stride + 1 
        # (28 -4 + 2*0) / 4 + 1 = 7
        # 输出(batch_size, 128, 7, 7)
        self.proj = nn.Conv2d(
            in_channels=model_config.in_channels,  # pytorch需要显式声明一下in_channels, 这里数据集图片单通道，为1
            out_channels=model_config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x):
        # 输入图像 (batch_size, 1, 28, 28)
        batch_size = x.shape[0]
        
        # 分块+嵌入 (batch_size, 1, 28, 28) -> (batch_size, 128, 7, 7) 每个patch 展平成128维 
        x = self.proj(x)
        
        # 转换为序列 (batch_size, 128, 7, 7) -> (batch_size, 128, 49) -> (batch_size, 49, 128)
        x = x.flatten(2)  # 索引为2后面的维展平（7*7 -> 49）
        x = x.transpose(1, 2)  # (batch, 序列长度, 特征维度) 调整成transformer输入格式
        
        return x  #(batch_size, 49, 128)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = model_config.hidden_size
        self.num_heads = model_config.num_heads  # 4个头
        
        # 每个头的维度  128 ÷ 4 = 32
        self.head_dim = self.hidden_size // self.num_heads
        assert self.head_dim * self.num_heads == self.hidden_size, "隐藏维度必须能被头数整除"
        
        # 线性层 生成qkv 
        self.qkv = nn.Linear(self.hidden_size, self.hidden_size * 3)  # 输出维度：128×3=384
        # 输出投影层 将拼接后的特征微调（维度不变）
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        # Dropout 
        self.dropout = nn.Dropout(model_config.dropout_rate)

    def forward(self, x):
        # x 输入序列  (batch_size, 序列长度, 128) 序列长度这里是49个patch + 1(cls) = 50
        batch_size, seq_len, _ = x.shape
        
        # 计算Q、K、V
        # (batch_size, seq_len, 128) -> (batch_size, seq_len, 384) -> 拆分为Q、K、V  拆分384 = 3(q,k,v) * num_heads(4个头) * head_dim(每个头 128 /4 = 32维 )
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv.unbind(0)  # 拆分Q、K、V (batch, num_heads, seq_len, head_dim)
        
        # 注意力权重 Q×K^T / sqrt(head_dim)
        #  q (batch, heads, seq_len, head_dim)
        #  k^T 把后面两个维度转置 (batch, heads, head_dim, seq_len)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (batch, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)  # 归一化权重 attn[b, h, i, j] bth 样本 hth 头中，Q_i 对K_j 的关注程度 把K_j归一化
        attn = self.dropout(attn)  
        
        # attn × V
        x = (attn @ v).transpose(1, 2)  # (batch, seq_len, num_heads, head_dim) 转置方便拼接头
        
        # 拼接多头
        x = x.reshape(batch_size, seq_len, self.hidden_size)  # (batch, seq_len, 128)
        
        x = self.proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(model_config.hidden_size, model_config.mlp_dim)  # 128 -> 256
        self.fc2 = nn.Linear(model_config.mlp_dim, model_config.hidden_size)  # 256 -> 128
        self.act = nn.GELU()  # ViT原论文用的GELU
        self.dropout = nn.Dropout(model_config.dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_config.hidden_size) 
        self.attn = MultiHeadAttention()  
        self.norm2 = nn.LayerNorm(model_config.hidden_size)  
        self.mlp = MLP()  

    def forward(self, x):
        # 注意力 + res
        x = x + self.attn(self.norm1(x))  
        # MLP + res
        x = x + self.mlp(self.norm2(x))  
        return x


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedding()  # patch embedding
        self.num_patches = self.patch_embed.num_patches  # 49个Patch PatchEmbedding里面算过了
        
        # [CLS] Token 可学习向量 用来最终分类
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_config.hidden_size))  # (1, 1, 128)
        
        # position embedding 记录Patch的位置信息 可学习向量
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, model_config.hidden_size)  # +1是因为加入了cls_token (1, 50, 128)
        )
        
        # num_layers model 里面设置两层transformerblock
        self.blocks = nn.ModuleList([
            TransformerBlock() for _ in range(model_config.num_layers)
        ])
        
        # 分类头 输出10类概率
        self.norm = nn.LayerNorm(model_config.hidden_size)  # 最终归一化
        self.head = nn.Linear(model_config.hidden_size, model_config.num_classes)  # 128 -> 10

    def forward(self, x):
        #  x (batch_size, 1, 28, 28)
        batch_size = x.shape[0]
        
        # patch embedding (batch_size, 1, 28, 28) -> (batch_size, 49, 128)
        x = self.patch_embed(x)
        
        # 添加CLS (batch_size, 49, 128) -> (batch_size, 50, 128)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # 扩展到batch_size：(batch_size, 1, 128) (-1：保持该维度不变)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 49+1, 128)
        
        # 添加 position embedding (batch_size, 50, 128) + (1, 50, 128) -> (batch_size, 50, 128) 每个patch和cls位置信息是一样的，与样本无关，这里会直接通过广播机制复制batch_size份
        x = x + self.pos_embed
        
        # 经过2层Transformer
        for block in self.blocks:
            x = block(x)  # 输出 (batch_size, 50, 128)
        
        # 取[CLS]令牌的输出
        x = self.norm(x)  # 归一化
        cls_out = x[:, 0]  # 取第一个位置cls_token (batch_size, 128)
        logits = self.head(cls_out)  # 分类头 (batch_size, 10)
        
        return logits  # 输出10类的预测概率（未经过softmax）


def check_model():
    model = VisionTransformer()
    #测试一下输出的维度对不对
    dummy_img = torch.randn(1, 1, 28, 28)  # (batch_size=1, 通道=1, 28, 28)
    output = model(dummy_img)
    print(f"输入形状：{dummy_img.shape}")
    print(f"输出形状：{output.shape}")  # 应该是 (1, 10)
    print("模型结构正确")