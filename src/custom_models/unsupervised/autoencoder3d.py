# src/custom_models/unsupervised/autoencoder3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 定义 3D 自编码器模型
class Autoencoder3D(nn.Module):
    """
    一个简单的 3D 卷积自编码器，用于从高维数据中提取低维潜在特征。
    """
    def __init__(self, latent_dim=128): # 增加默认的 latent_dim 以匹配你提供的原始代码的输出尺寸
        super(Autoencoder3D, self).__init__()
        # 编码器: 3D 卷积层提取特征
        self.encoder = nn.Sequential(
            # 输入: [B, 2, 1820, 96, 1] (假设原始尺寸是 1820x96，通道2)
            # 你原始代码的尺寸是 1819，但是 Conv3d 会有尺寸计算，用 1820 比较规整，
            # 如果是 1819，可能需要调整 padding 或 output_padding
            # Conv3d output_size = (input_size - kernel_size + 2*padding) / stride + 1
            nn.Conv3d(2, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 910, 48, 1]
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 455, 24, 1]
            nn.ReLU(),
            # 这里是编码器的最终输出层，输出的特征图将是潜在空间表示
            nn.Conv3d(64, latent_dim, kernel_size=3, stride=2, padding=1), # [B, latent_dim, 228, 12, 1]
            nn.ReLU()
        )
        # 解码器: 3D 反卷积层重建数据
        self.decoder = nn.Sequential(
            # ConvTranspose3d output_size = (input_size - 1) * stride - 2*padding + kernel_size + output_padding
            nn.ConvTranspose3d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 64, 455, 24, 1]
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, 910, 48, 1]
            nn.ReLU(),
            nn.ConvTranspose3d(32, 2, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 2, 1820, 96, 1]
            nn.Sigmoid() # 归一化到 [0,1]，适合图像或类似的非负数据
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def load_autoencoder_model(model_path, latent_dim, device):
    """
    加载预训练的 Autoencoder3D 模型权重。

    Args:
        model_path (str): .pth 模型文件的路径。
        latent_dim (int): 自编码器的潜在维度。
        device (torch.device): 模型加载到的设备 (CPU 或 CUDA)。

    Returns:
        Autoencoder3D: 加载了预训练权重的模型实例。
    """
    model = Autoencoder3D(latent_dim=latent_dim).to(device)
    if not torch.cuda.is_available(): # 如果没有 GPU，强制加载到 CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"警告: CUDA 不可用，模型 '{model_path}' 已加载到 CPU。")
    else:
        model.load_state_dict(torch.load(model_path))
    model.eval() # 设置为评估模式，关闭 dropout 等
    print(f"Autoencoder3D 模型已从 '{model_path}' 加载到 {device}。")
    return model

def extract_features_with_autoencoder(model, data_tensor, device):
    """
    使用预训练的自编码器提取数据的潜在特征。

    Args:
        model (Autoencoder3D): 加载了权重的 Autoencoder3D 模型实例。
        data_tensor (torch.Tensor): 输入数据，应为 torch.float32 类型，形状为 [N, C, D, H, W] 或 [N, C, H, W, D]。
                                    根据你提供的代码，你的数据形状是 [N, C, H, W, D] (即 [B, 2, 1819, 96, 1])
        device (torch.device): 模型运行所在的设备。

    Returns:
        np.ndarray: 提取的潜在特征，形状为 [N, flattened_latent_dim]。
    """
    model.eval() # 确保模型处于评估模式
    with torch.no_grad(): # 在推理过程中禁用梯度计算
        # 确保输入数据类型和设备正确
        data_tensor = data_tensor.float().to(device)
        encoded_data, _ = model(data_tensor)
        # 将编码后的特征从 PyTorch tensor 转换为 NumPy 数组
        # 并将其展平为 2D 数组，每行代表一个样本的特征向量
        encoded_features = encoded_data.cpu().numpy().reshape(len(encoded_data), -1)
    print(f"通过 Autoencoder3D 提取了 {encoded_features.shape[0]} 个样本的潜在特征，特征维度为 {encoded_features.shape[1]}。")
    return encoded_features