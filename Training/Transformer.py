import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio

# 自定义数据集：从指定文件夹读取低分辨率和高分辨率影像（假设均为单波段 TIFF 文件）
class RemoteSensingDataset(Dataset):
    def __init__(self, lr_folder, hr_folder, transform=None):
        self.lr_paths = sorted(glob.glob(os.path.join(lr_folder, '*.TIF')))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_folder, '*.TIF')))
        assert len(self.lr_paths) == len(self.hr_paths), "低分和高分图像数量不匹配！"
        self.transform = transform

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr_path = self.lr_paths[idx]
        hr_path = self.hr_paths[idx]
        with rasterio.open(lr_path) as src:
            lr = src.read(1).astype(np.float32)
        with rasterio.open(hr_path) as src:
            hr = src.read(1).astype(np.float32)
        # 简单归一化（可根据数据特点调整）
        lr = (lr - lr.min()) / (lr.max() - lr.min() + 1e-6)
        hr = (hr - hr.min()) / (hr.max() - hr.min() + 1e-6)
        # 扩展通道维度，变为 (1, H, W)
        lr = np.expand_dims(lr, axis=0)
        hr = np.expand_dims(hr, axis=0)
        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)
        return torch.from_numpy(lr), torch.from_numpy(hr)

# 定义基于 Transformer 的超分模型
class SRTransformer(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, num_heads=4, num_layers=4, upscale_factor=33):
        super(SRTransformer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 构造 Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.conv2 = nn.Conv2d(embed_dim, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv1(x)
        x = self.relu(x)
        B, C, H, W = x.shape
        # 将空间维度展平为序列：(H*W, B, C)
        x_flat = x.view(B, C, H * W).permute(2, 0, 1)
        x_trans = self.transformer(x_flat)
        x_trans = x_trans.permute(1, 2, 0).view(B, C, H, W)
        x = self.conv2(x_trans)
        x = self.pixel_shuffle(x)
        return x


def train_transformer_model():
    # 修改低分辨率图像所在的文件夹名称为 'LST_30m_990m'（代表990m分辨率数据）
    lr_folder = 'E:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m_990m'
    # 高分辨率图像所在的文件夹（例如30m分辨率数据）
    hr_folder = 'E:/MyProjects/MachineLearning/Data/Usable_Data/LST_30m'

    # 设置批次大小、训练周期数和学习率
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4

    # 检查 GPU 是否可用，如果不可用则抛出错误并中断程序
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU不可用，请检查CUDA和cuDNN的安装情况。")
    device = torch.device('cuda')  # 此时device必然为GPU

    # 创建数据集实例，RemoteSensingDataset用于加载低分与高分图像数据
    dataset = RemoteSensingDataset(lr_folder, hr_folder)
    # 仅使用数据列表中下标为5（即第6个文件）的数据来验证模型性能
    dataset.lr_paths = [dataset.lr_paths[5]]
    dataset.hr_paths = [dataset.hr_paths[5]]

    # 构建 DataLoader 进行批量数据加载，shuffle=True表示随机打乱数据，num_workers=4加速数据读取
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 上采样倍数说明：
    # upscale_factor 指定模型输出图像相对于输入图像的尺寸放大比例。
    # 例如，upscale_factor=2 表示输出图像的高度和宽度均为输入图像的2倍，即实现2倍分辨率提升。
    model = SRTransformer(upscale_factor=33).to(device)

    # 定义均方误差损失，用于衡量模型输出和真实高分辨率图像之间的误差
    criterion = nn.MSELoss()
    # 使用Adam优化器更新模型参数，学习率设置为 learning_rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化 best_loss 为无穷大，用于保存最佳（最低）损失
    best_loss = float('inf')
    # 指定保存训练中最佳模型的目录
    model_save_dir = 'E:/MyProjects/MachineLearning/Data/Final_Data/Models/transformer'
    # 创建目录（如果不存在的话）
    os.makedirs(model_save_dir, exist_ok=True)

    # 开始训练过程，共进行 num_epochs 个周期
    for epoch in range(num_epochs):
        # 将模型设置为训练模式
        model.train()
        # 初始化当前周期累计损失为 0
        epoch_loss = 0.0
        # 遍历 DataLoader 中的每个批次
        for lr_imgs, hr_imgs in dataloader:
            # 将当前批次数据移动到 GPU 上
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # 清空优化器累积的梯度
            optimizer.zero_grad()
            # 前向传播，得到模型预测的高分辨率图像
            outputs = model(lr_imgs)
            # 计算预测输出与真实高分图像之间的均方误差损失
            loss = criterion(outputs, hr_imgs)
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 累加当前批次损失（乘以批次中样本数，以便后续求平均）
            epoch_loss += loss.item() * lr_imgs.size(0)
        # 计算整个数据集上的平均损失
        epoch_loss /= len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        # 若本周期损失低于之前记录的最佳损失，则保存当前模型参数
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_transformer_model.pth'))
    print("Transformer 模型训练完成。")


if __name__ == '__main__':
    train_transformer_model()
