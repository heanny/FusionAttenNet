import logging
import csv
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import pandas as pd
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import copy
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# BrainMapDataset class
class BrainMapDataset(Dataset):
    def __init__(self, image_data, features_and_targets, mean=None, std=None, training=False):
        self.image_data = image_data
        self.features_and_targets = features_and_targets
        self.mean = mean
        self.std = std
        self.training = training

    def __len__(self):
        return len(self.features_and_targets)

    def __getitem__(self, idx):
        images = self.image_data[idx].astype(np.float32)

        # 应用标准化
        if self.mean is not None and self.std is not None:
            images = (images - self.mean[:, None, None]) / (self.std[:, None, None] + 1e-8)

        # 添加噪声增强
        if self.training:
            # 增加更多数据增强
            if np.random.rand() > 0.5:
                images = np.flip(images, axis=2)  # 水平翻转
            
            # 随机旋转
            angle = np.random.uniform(-10, 10)
            images = scipy.ndimage.rotate(images, angle, axes=(1,2), reshape=False, mode='nearest')
        
            noise = np.random.normal(0, 0.02, images.shape).astype(np.float32)
            images = images + noise
        
        images = torch.from_numpy(images).float()  # convert to [4, 512, 512] 的张量
        features_and_target = self.features_and_targets[idx]
        # 新的组合：
        # targets: attention_scores (index 0) 和 age (index 2)
        targets = np.array([
            features_and_target[0],  # attention_scores
            features_and_target[2]   # age
        ]).astype(np.float32)

        # extra_features(as input): aggressive_behaviour_scores (index 1) 和 sex (index 3) 和 maternal_edu_level (index 4)
        extra_features = np.array([
            features_and_target[1],  # aggressive_behaviour_scores
            features_and_target[3],  # sex
            features_and_target[4]   # maternal_edu_level
        ]).astype(np.float32)

        return images, torch.tensor(extra_features).float(), torch.tensor(targets).float()


# BrainADHDModel class 
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * out
        
    
class BrainADHDModel(nn.Module):
    def __init__(self, num_phenotypes):
        super().__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 修改第一个卷积层以接受4通道输入
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 将原始权重复制到新的卷积层的前3个通道
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = original_conv.weight
            # 对第4个通道进行初始化（可以使用前3个通道的平均值）
            resnet.conv1.weight[:, 3] = original_conv.weight.mean(dim=1)
        
        # 提取除最后的全连接层之外的所有层
        self.frontend = nn.Sequential(*list(resnet.children())[:-2])
        
        # Attention modules
        self.spatial_attention = SpatialAttention()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, 1),
            nn.Sigmoid()
        )
        
        # Feature processing
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.phenotype_encoder = nn.Sequential(
            nn.Linear(num_phenotypes, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final prediction layers
        self.shared_features = nn.Sequential(
            nn.Linear(2048 + 64, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 添加feature注意力机制
        self.feature_attention = nn.Sequential(
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 128),
        nn.Sigmoid()
        )
        
        # 分别预测年龄和注意力
        self.age_predictor = nn.Linear(128, 1)
        self.attention_predictor = nn.Linear(128, 1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, phenotypes):
        # Extract visual features
        x = self.frontend(x)
        
        # Apply attention
        x = self.spatial_attention(x)
        x = x * self.channel_attention(x)
        
        # Global pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Process phenotype data
        phenotype_features = self.phenotype_encoder(phenotypes)
        
        # Combine features
        combined = torch.cat([x, phenotype_features], dim=1)
        shared_features = self.shared_features(combined)
        attention_weights = self.feature_attention(shared_features)
        weighted_features = shared_features * attention_weights
        
        # Final prediction
        # 使用加权特征进行预测
        age_pred = self.age_predictor(weighted_features)
        attention_pred = self.attention_predictor(weighted_features)

        return torch.cat([attention_pred, age_pred], dim=1)

class BrainADHDModel_NoSpatialAttention(nn.Module):
    def __init__(self, num_phenotypes):
        super().__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 修改第一个卷积层以接受4通道输入
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 将原始权重复制到新的卷积层的前3个通道
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = original_conv.weight
            # 对第4个通道进行初始化（可以使用前3个通道的平均值）
            resnet.conv1.weight[:, 3] = original_conv.weight.mean(dim=1)
        
        # 提取除最后的全连接层之外的所有层
        self.frontend = nn.Sequential(*list(resnet.children())[:-2])
        
        # Attention modules
        # self.spatial_attention = SpatialAttention()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, 1),
            nn.Sigmoid()
        )
        
        # Feature processing
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.phenotype_encoder = nn.Sequential(
            nn.Linear(num_phenotypes, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final prediction layers
        self.shared_features = nn.Sequential(
            nn.Linear(2048 + 64, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 添加feature注意力机制
        self.feature_attention = nn.Sequential(
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, 128),
        nn.Sigmoid()
        )
        
        # 分别预测年龄和注意力
        self.age_predictor = nn.Linear(128, 1)
        self.attention_predictor = nn.Linear(128, 1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, phenotypes):
        # Extract visual features
        x = self.frontend(x)
        
        # Apply attention
        # x = self.spatial_attention(x)
        x = x * self.channel_attention(x)
        
        # Global pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Process phenotype data
        phenotype_features = self.phenotype_encoder(phenotypes)
        
        # Combine features
        combined = torch.cat([x, phenotype_features], dim=1)
        shared_features = self.shared_features(combined)
        attention_weights = self.feature_attention(shared_features)
        weighted_features = shared_features * attention_weights
        
        # Final prediction
        # 使用加权特征进行预测
        age_pred = self.age_predictor(weighted_features)
        attention_pred = self.attention_predictor(weighted_features)

        return torch.cat([attention_pred, age_pred], dim=1)


class BrainADHDModel_NoFeatureAttention(nn.Module):
    def __init__(self, num_phenotypes):
        super().__init__()
        
        # Use ResNet50 as backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 修改第一个卷积层以接受4通道输入
        original_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 将原始权重复制到新的卷积层的前3个通道
        with torch.no_grad():
            resnet.conv1.weight[:, :3] = original_conv.weight
            # 对第4个通道进行初始化（可以使用前3个通道的平均值）
            resnet.conv1.weight[:, 3] = original_conv.weight.mean(dim=1)
        
        # 提取除最后的全连接层之外的所有层
        self.frontend = nn.Sequential(*list(resnet.children())[:-2])
        
        # Attention modules
        self.spatial_attention = SpatialAttention()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 2048, 1),
            nn.Sigmoid()
        )
        
        # Feature processing
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.phenotype_encoder = nn.Sequential(
            nn.Linear(num_phenotypes, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final prediction layers
        self.shared_features = nn.Sequential(
            nn.Linear(2048 + 64, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 添加feature注意力机制
        # self.feature_attention = nn.Sequential(
        # nn.Linear(128, 32),
        # nn.ReLU(),
        # nn.Linear(32, 128),
        # nn.Sigmoid()
        # )
        
        # 分别预测年龄和注意力
        self.age_predictor = nn.Linear(128, 1)
        self.attention_predictor = nn.Linear(128, 1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, phenotypes):
        # Extract visual features
        x = self.frontend(x)
        
        # Apply attention
        x = self.spatial_attention(x)
        x = x * self.channel_attention(x)
        
        # Global pooling
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Process phenotype data
        phenotype_features = self.phenotype_encoder(phenotypes)
        
        # Combine features
        combined = torch.cat([x, phenotype_features], dim=1)
        shared_features = self.shared_features(combined)
        # attention_weights = self.feature_attention(shared_features)
        # weighted_features = shared_features * attention_weights
        
        # Final prediction
        # 使用加权特征进行预测
        age_pred = self.age_predictor(shared_features)
        attention_pred = self.attention_predictor(shared_features)

        return torch.cat([attention_pred, age_pred], dim=1)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        l1_loss = self.l1(outputs, targets)
        return 0.5 * mse_loss + 0.5 * l1_loss + 0.1 * torch.abs(outputs - targets).max()
    


# Optimizer and scheduler setup
def get_training_components(model, config, steps_per_epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['initial_lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['initial_lr'] * 15,
        epochs=config['num_epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=15.0,
        final_div_factor=1000.0
    )
    
    criterion = CombinedLoss()
    
    return optimizer, scheduler, criterion


# train and validate loop functions
def train_epoch(model, loader, optimizer, scheduler, criterion, device, gradient_accumulation_steps):
    model.train()
    total_loss = 0
    total_att_loss = 0
    total_age_loss = 0
    step_learning_rates = []
    num_batches = len(loader)
    # optimizer.zero_grad()

    for i, (brain_images, phenotypes, targets) in enumerate(loader):
        if i == 0:  # 只为第一个批次打印形状信息
            print(f"\nBatch {i} shapes:")
            print(f"brain_images: {brain_images.shape}")
            print(f"phenotypes: {phenotypes.shape}")
            print(f"targets: {targets.shape}")
        
        # 只在必要时清理内存
        if (i + 1) % 10 == 0:  # 每10个batch才清理一次
            torch.cuda.empty_cache()

        # Move to device and convert to half precision if needed
        brain_images = brain_images.to(device)
        phenotypes = phenotypes.to(device)
        targets = targets.to(device)
        # optimizer.zero_grad()

        # Forward pass
        outputs = model(brain_images, phenotypes)

        # 分别计算两个任务的损失
        attention_loss = criterion(outputs[:, 0], targets[:, 0])  # attention score
        age_loss = criterion(outputs[:, 1], targets[:, 1])       # age

        # 动态权重
        att_weight = 0.5  # attention_score预测效果差，给更大权重
        age_weight = 0.5
        loss = (att_weight * attention_loss + age_weight * age_loss) / gradient_accumulation_steps

        # loss = criterion(outputs, targets)
        # loss = loss / gradient_accumulation_steps  # Normalize loss due to gradient_accumulation_steps

        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item()

        # Backward pass with gradient accumulation
        loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step_learning_rates.append(optimizer.param_groups[0]['lr'])
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_att_loss += attention_loss.item()
        total_age_loss += age_loss.item()

        # Clear some memory
        del brain_images, phenotypes, targets, outputs, loss, attention_loss, age_loss
        gc.collect()
        torch.cuda.empty_cache()

        # 打印进度
        # if i % 200 == 0:
        #     print(f'Batch [{i}/{num_batches}], Loss: {loss.item():.4f}')

    return (total_loss / len(loader), 
            total_att_loss / len(loader),
            total_age_loss / len(loader)), step_learning_rates

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets_list = []
    
    with torch.no_grad():
        for brain_images, phenotypes, targets in loader:
            brain_images = brain_images.to(device)
            phenotypes = phenotypes.to(device)
            targets = targets.to(device)
            
            outputs = model(brain_images, phenotypes)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())
            
            # Clear memory
            del brain_images, phenotypes, targets, outputs, loss
            gc.collect()
            torch.cuda.empty_cache()
    
    return total_loss / len(loader), np.array(predictions), np.array(targets_list)


def load_data(image_path, phenotype_path, use_mmap=True):
    """
    Load data with mmap mode

    Parameters:
    image_path : str
        brain image data path
    phenotype_path : str
        phenotype data path
    use_mmap : bool
        use mmap mode or not
    
    Returns:
    tuple : (image_data, phenotype_data)

    """
    try:
        # check file size
        image_size = os.path.getsize(image_path) / (1024 ** 3)  # to GB
        phenotype_size = os.path.getsize(phenotype_path) / (1024 ** 3)
        
        # If the file is large and mmap is enabled, use memory mapped mode
        if use_mmap and (image_size > 1 or phenotype_size > 1):  # if it's larger than 1GB
            image_data = np.load(image_path, mmap_mode='r')  # read-only mode
            
            phenotype_data = np.load(phenotype_path, mmap_mode='r')
            print(f"Loaded data using memory mapping. Image data shape: {image_data.shape}")
        else:
            image_data = np.load(image_path)
            phenotype_data = np.load(phenotype_path)
            print(f"Loaded data into memory. Image data shape: {image_data.shape}")
        
        return image_data, phenotype_data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# 数据加载优化
def get_data_loaders(image_data, phenotype_tensor, batch_size, num_workers=2):
    # 计算均值和标准差
    mean = np.mean(image_data, axis=(0, 2, 3)).astype(np.float32)# image shape: (N, C, H, W)
    std = np.std(image_data, axis=(0, 2, 3)).astype(np.float32)
    
    # 数据集分割
    indices = np.arange(len(image_data))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.11111, random_state=42)
    
    # 创建数据集
    train_dataset = BrainMapDataset(
        image_data[train_idx], 
        phenotype_tensor[train_idx],
        mean=mean,
        std=std,
        training=True  # 为训练集启用数据增强
    )
    val_dataset = BrainMapDataset(
        image_data[val_idx], 
        phenotype_tensor[val_idx],
        mean=mean,
        std=std,
        training=False  # 验证集不需要启用数据增强
    )
    test_dataset = BrainMapDataset(
        image_data[test_idx], 
        phenotype_tensor[test_idx],
        mean=mean,
        std=std,
        training=False  # 测试集不使用数据增强
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader

train_history = {
    'train_losses': [],
    'val_losses': [],
    'train_att_losses': [],
    'train_age_losses': [],
    'learning_rates': []
}

class ResultsManager:
    """改进的实验结果管理器"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.create_directories()
        
    def create_directories(self):
        """创建完整的目录结构"""
        self.subdirs = {
            'attention': {
                'raw_data': os.path.join(self.output_dir, 'attention/raw_data'),
                'analysis': os.path.join(self.output_dir, 'attention/analysis'),
            },
            'cross_validation': {
                'fold_results': os.path.join(self.output_dir, 'cross_validation/fold_results'),
                'metrics': os.path.join(self.output_dir, 'cross_validation/metrics'),
                'analysis': os.path.join(self.output_dir, 'cross_validation/analysis'),
            },
            'error_analysis': {
                'raw_data': os.path.join(self.output_dir, 'error_analysis/raw_data'),
                'statistics': os.path.join(self.output_dir, 'error_analysis/statistics'),
                'detailed_analysis': os.path.join(self.output_dir, 'error_analysis/detailed_analysis'),
            },
            'logs': os.path.join(self.output_dir, 'logs'),
            'models': os.path.join(self.output_dir, 'models'),
            'plots': 
                {
                'attention': os.path.join(self.output_dir, 'plots/attention_plots'),
                'training': os.path.join(self.output_dir, 'plots/training_plots'),
                'error': os.path.join(self.output_dir, 'plots/error_plots'),
                }
        }
    
        
        
        for main_dir in self.subdirs.values():
            if isinstance(main_dir, dict):
                for sub_dir in main_dir.values():
                    os.makedirs(sub_dir, exist_ok=True)
            else:
                os.makedirs(main_dir, exist_ok=True)

    # 在保存数据之前添加维度检查和处理
    def _process_attention_data(self, data):
        """处理注意力数据的维度"""
        if isinstance(data, np.ndarray):
            # 如果是通道注意力权重（形状为 [batch, channels, 1, 1]）
            if data.shape[-2:] == (1, 1):
                data = data.squeeze()
                # 如果还有多余的维度，只保留前两个维度
                if data.ndim > 2:
                    data = data[:, :data.shape[1]]
            # 对于其他注意力数据
            elif data.ndim > 2:
                data = data.squeeze()
                if data.ndim > 2:
                    # 展平最后的维度
                    shape = data.shape
                    data = data.reshape(shape[0], -1)
        return data

    def save_attention_analysis(self, model, test_loader, device, name=None):
        """增强的注意力分析与存储，支持不同模型架构"""
        prefix = f'{name}_' if name is not None else ''
        model.eval()
            
        # 首先检查模型类型
        has_spatial = hasattr(model, 'spatial_attention')
        has_feature = hasattr(model, 'feature_attention')

        # 只创建需要的列表        
        attention_data = {
            'raw_data': {
                'channel_attention_weights': [],
                'targets': [],
                'predictions': [],
                'metadata': []
            },
            'analysis': {
                'attention_correlations': {},
                'attention_patterns': {},
                'attention_statistics': {}
            }
        }
            
        if has_spatial:
            attention_data['raw_data']['spatial_attention_maps'] = []
        if has_feature:
            attention_data['raw_data']['feature_attention_weights'] = []

        # 收集所有数据
        with torch.no_grad():
            for brain_maps, extra_features, targets in test_loader:
                try:
                    brain_maps = brain_maps.to(device)
                    extra_features = extra_features.to(device)
                    
                    # 获取中间特征
                    features = model.frontend(brain_maps)
                    
                    # 根据模型类型获取不同的注意力信息
                    if hasattr(model, 'spatial_attention'):
                        spatial_att = model.spatial_attention(features)
                        attention_data['raw_data']['spatial_attention_maps'].append(
                            spatial_att.cpu().numpy()
                        )
                    
                    channel_att = model.channel_attention(features)
                    attention_data['raw_data']['channel_attention_weights'].append(
                        channel_att.cpu().numpy()
                    )
                    
                    # 获取feature attention (如果存在)
                    x = model.avg_pool(features * channel_att)
                    x = x.view(x.size(0), -1)
                    phenotype_features = model.phenotype_encoder(extra_features)
                    combined = torch.cat([x, phenotype_features], dim=1)
                    shared_features = model.shared_features(combined)
                    
                    if hasattr(model, 'feature_attention'):
                        feature_att = model.feature_attention(shared_features)
                        attention_data['raw_data']['feature_attention_weights'].append(
                            feature_att.cpu().numpy()
                        )
                    
                    # 获取预测结果
                    outputs = model(brain_maps, extra_features)
                    
                    # 保存预测和目标
                    attention_data['raw_data']['targets'].append(targets.numpy())
                    attention_data['raw_data']['predictions'].append(outputs.cpu().numpy())
                    attention_data['raw_data']['metadata'].append(extra_features.cpu().numpy())
                                
                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    continue
                    
                # 清理内存
                del brain_maps, extra_features, features, x, outputs
                torch.cuda.empty_cache()
        
        # 合并所有数据
        for key in attention_data['raw_data']:
                if attention_data['raw_data'][key]:  # 如果列表非空
                    try:
                        attention_data['raw_data'][key] = np.concatenate(attention_data['raw_data'][key])
                    except Exception as e:
                        logging.error(f"Error concatenating {key}: {str(e)}")
                        attention_data['raw_data'][key] = np.array([])
    
        
        # 计算统计信息
        attention_stats = {}
        
        if has_spatial and len(attention_data['raw_data']['spatial_attention_maps']) > 0:
            maps = attention_data['raw_data']['spatial_attention_maps']
            attention_stats['spatial_attention'] = {
                'mean': np.mean(maps, axis=0),
                'std': np.std(maps, axis=0)
            }
        
        weights = weights = attention_data['raw_data']['channel_attention_weights']
        if len(weights) > 0:
            attention_stats['channel_attention'] = {
                'mean': np.mean(weights, axis=0),
                'std': np.std(weights, axis=0)
            }
        
        if has_feature and len(attention_data['raw_data']['feature_attention_weights']) > 0:
            weights = attention_data['raw_data']['feature_attention_weights']
            attention_stats['feature_attention'] = {
                'mean': np.mean(weights, axis=0),
                'std': np.std(weights, axis=0)
            }
        
        attention_data['analysis']['attention_statistics'] = attention_stats
        
        # 保存数据
        for data_type, data in attention_data['raw_data'].items():
            if isinstance(data, (np.ndarray, list)) and len(data) > 0:
                processed_data = self._process_attention_data(data)
                np.save(os.path.join(self.subdirs['attention']['raw_data'], 
                                f'{prefix}{data_type}.npy'), processed_data)
        
        for analysis_type, analysis in attention_data['analysis'].items():
            if analysis:  # 只保存非空分析
                np.save(os.path.join(self.subdirs['attention']['analysis'], 
                            f'{prefix}{analysis_type}.npy'), analysis)
        
        return attention_data

    def save_cv_results(self, cv_results, fold=None):
        """增强的交叉验证结果存储"""
        prefix = f'fold_{fold}_' if fold is not None else ''
        
        # 保存每个fold的详细结果
        for fold_idx, fold_data in enumerate(cv_results['fold_train_history']):
            fold_dir = os.path.join(self.subdirs['cross_validation']['fold_results'], f'fold_{fold_idx}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # 保存训练历史
            np.save(os.path.join(fold_dir, f'{prefix}training_history.npy'), {
                'train_losses': fold_data['train_losses'],
                'val_losses': fold_data['val_losses'],
                'train_att_losses': fold_data['train_att_losses'],
                'train_age_losses': fold_data['train_age_losses'],
                'learning_rates': fold_data['learning_rates']
            })
        
        # 保存整体指标
        metrics_data = {
            'fold_metrics': cv_results['fold_metrics'],
            'overall_metrics': {
                'mean_r2': np.mean([fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]),
                'std_r2': np.std([fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]),
                'mean_mse': np.mean([fold['sum_att']['MSE'] for fold in cv_results['fold_metrics']]),
                'mean_mae': np.mean([fold['sum_att']['MAE'] for fold in cv_results['fold_metrics']])
            }
        }
        np.save(os.path.join(self.subdirs['cross_validation']['metrics'], f'{prefix}all_metrics.npy'), metrics_data)
        
        # 保存预测结果
        predictions_data = {
            'predictions': cv_results['predictions'],
            'targets': cv_results['targets']
        }
        np.save(os.path.join(self.subdirs['cross_validation']['analysis'], f'{prefix}predictions.npy'), predictions_data)
        
    def save_config(self, config):
        """保存实验配置"""
        path = os.path.join(self.subdirs['logs'], 'experiment_config.json')
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    def save_model(self, model, name, fold=None):
        """保存模型权重"""
        if fold is not None:
            name = f"{name}_fold_{fold}.pth"
        else:
            name = f"{name}.pth"
        path = os.path.join(self.subdirs['models'], name)
        torch.save(model.state_dict(), path)

    def save_training_history(self, history, prefix=''):
        """保存训练历史"""
        save_path = os.path.join(self.subdirs['plots']['training'])
        np.save(os.path.join(save_path, f'{prefix}training_history.npy'), history)


    def save_error_analysis(self, predictions, targets, metadata):
        """增强的错误分析与存储"""
        error_data = {
            'raw_data': {
                'predictions': predictions,  # shape: (n_samples, 2) [attention_score, age]
                'targets': targets,         # shape: (n_samples, 2) [attention_score, age]
                'metadata': metadata,       # shape: (n_samples, 3) [aggressive_score, sex, edu_level]
                'errors': {
                    'attention': {
                        'values': predictions[:, 0] - targets[:, 0],
                        'absolute': np.abs(predictions[:, 0] - targets[:, 0]),
                        'relative': np.abs((predictions[:, 0] - targets[:, 0]) / (targets[:, 0] + 1e-8))
                    },
                    'age': {
                        'values': predictions[:, 1] - targets[:, 1],
                        'absolute': np.abs(predictions[:, 1] - targets[:, 1]),
                        'relative': np.abs((predictions[:, 1] - targets[:, 1]) / (targets[:, 1] + 1e-8))
                    }
                }
            },
            'statistics': {
                'attention': {
                    'basic_stats': {
                        'mean_error': np.mean(predictions[:, 0] - targets[:, 0]),
                        'std_error': np.std(predictions[:, 0] - targets[:, 0]),
                        'mae': np.mean(np.abs(predictions[:, 0] - targets[:, 0])),
                        'mse': np.mean((predictions[:, 0] - targets[:, 0])**2),
                        'rmse': np.sqrt(np.mean((predictions[:, 0] - targets[:, 0])**2))
                    },
                    'percentiles': np.percentile(np.abs(predictions[:, 0] - targets[:, 0]), 
                                            [25, 50, 75, 90, 95])
                },
                'age': {
                    'basic_stats': {
                        'mean_error': np.mean(predictions[:, 1] - targets[:, 1]),
                        'std_error': np.std(predictions[:, 1] - targets[:, 1]),
                        'mae': np.mean(np.abs(predictions[:, 1] - targets[:, 1])),
                        'mse': np.mean((predictions[:, 1] - targets[:, 1])**2),
                        'rmse': np.sqrt(np.mean((predictions[:, 1] - targets[:, 1])**2))
                    },
                    'percentiles': np.percentile(np.abs(predictions[:, 1] - targets[:, 1]), 
                                            [25, 50, 75, 90, 95])
                }
            }
        }
        
        # 添加高错误分析
        for i, target_type in enumerate(['attention', 'age']):
            errors = error_data['raw_data']['errors'][target_type]['values']
            error_mean = np.mean(errors)
            error_std = np.std(errors)
            high_error_mask = np.abs(errors - error_mean) > 2 * error_std
            
            error_data[f'high_error_{target_type}'] = {
                'indices': np.where(high_error_mask)[0],
                'predictions': predictions[high_error_mask, i],
                'targets': targets[high_error_mask, i],
                'errors': errors[high_error_mask],
                'metadata_stats': {
                    'aggressive_score_mean': np.mean(metadata[high_error_mask, 0]),
                    'sex_distribution': np.bincount(metadata[high_error_mask, 1].astype(int)),
                    'edu_level_distribution': np.bincount(metadata[high_error_mask, 2].astype(int))
                }
            }
            
            # 添加错误相关性分析
            error_data[f'error_correlations_{target_type}'] = {
                'aggressive_score': np.corrcoef(metadata[:, 0], np.abs(errors))[0, 1],
                'sex': np.corrcoef(metadata[:, 1], np.abs(errors))[0, 1],
                'edu_level': np.corrcoef(metadata[:, 2], np.abs(errors))[0, 1]
            }
        
        # 保存所有数据
        np.save(os.path.join(self.subdirs['error_analysis']['raw_data'], 'raw_data.npy'), 
                error_data['raw_data'])
        np.save(os.path.join(self.subdirs['error_analysis']['statistics'], 'statistics.npy'), 
                error_data['statistics'])
        np.save(os.path.join(self.subdirs['error_analysis']['detailed_analysis'], 'high_error_analysis.npy'), 
                {k: v for k, v in error_data.items() if k.startswith('high_error_')})
        np.save(os.path.join(self.subdirs['error_analysis']['detailed_analysis'], 'error_correlations.npy'), 
                {k: v for k, v in error_data.items() if k.startswith('error_correlations_')})
        
        return error_data
    

class AblationStudyManager:
    def __init__(self, config, output_dir, full_model_path):
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, f'ablation_study_{self.timestamp}')
        self.results_manager = ResultsManager(self.output_dir)
        self.full_model_path = full_model_path
        # '/home/jouyang1/training_all_data_rh_mirror_attmap_20250205_041248/models/all_rh_best_model.pth'
        # '/home/jouyang1/training_all_data_lh_mirror_attmap_20250205_031917/models/all_lh_best_model.pth'

        # 设置实验配置
        self.experiment_configs = {
            'no_feature_attention': {'model_class': BrainADHDModel_NoFeatureAttention, 'name': 'no_feature_attention'},
            'no_spatial_attention': {'model_class': BrainADHDModel_NoSpatialAttention, 'name': 'no_spatial_attention'}
        }
        
        self.metrics_names = ['MSE', 'RMSE', 'MAE', 'R2', 'Pearson']
        self.tasks = ['attention', 'age']
        
        # 存储实验结果
        self.results = {name: {} for name in self.experiment_configs.keys()}
        
        self._setup_logging()

    def load_baseline_results(self, test_loader, device):
        """加载基准模型的结果"""
        # 初始化基准模型
        full_model = BrainADHDModel(self.config['num_phenotypes']).to(device)
        full_model.load_state_dict(torch.load(self.full_model_path))
        full_model.eval()
        
        criterion = CombinedLoss()
        
        # 在测试集上评估基准模型
        with torch.no_grad():
            test_loss, predictions, targets = validate_epoch(
                full_model, test_loader, criterion, device
            )
            
        # 构造基准结果字典
        baseline_results = {
            'test_loss': test_loss,
            'metrics': {
                'attention': self.calculate_metrics(predictions[:, 0], targets[:, 0]),
                'age': self.calculate_metrics(predictions[:, 1], targets[:, 1])
            },
            'predictions': predictions,
            'targets': targets
        }
        
        # 保存基准模型的注意力分析
        attention_data = self.results_manager.save_attention_analysis(
            full_model, test_loader, device, 'baseline'
        )
        baseline_results['attention_analysis'] = attention_data
        
        # 保存基准模型的错误分析
        error_data = self.results_manager.save_error_analysis(
            predictions, targets, 
            np.array([test_loader.dataset.features_and_targets[i][1:] 
                    for i in range(len(test_loader.dataset))])
        )
        baseline_results['error_analysis'] = error_data
        
        return baseline_results
    
    def _setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'ablation_study.log')),
                logging.StreamHandler()
            ]
        )
    def _safe_mean(self, data, axis=None):
        """安全计算均值，处理空数组和无效值"""
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            # 确保数据是numpy数组
            data = np.asarray(data)
            # 移除nan和inf
            data = data[~np.isnan(data) & ~np.isinf(data)]
            if len(data) > 0:
                return np.mean(data, axis=axis)
        return 0.0

    def _safe_std(self, data, axis=None):
        """安全计算标准差，处理空数组和无效值"""
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            data = np.asarray(data)
            data = data[~np.isnan(data) & ~np.isinf(data)]
            if len(data) > 0:
                return np.std(data, axis=axis)
        return 0.0

    
    def calculate_metrics(self, predictions, targets):
        """计算所有评估指标"""
        metrics = {
            'MSE': mean_squared_error(targets, predictions),
            'RMSE': np.sqrt(mean_squared_error(targets, predictions)),
            'MAE': mean_absolute_error(targets, predictions),
            'R2': r2_score(targets, predictions),
            'Pearson': np.corrcoef(targets, predictions)[0,1]
        }
        return metrics
    
    def run_single_experiment(self, model_config, train_loader, val_loader, test_loader, device):
        """运行单个实验配置"""
        model_class = model_config['model_class']
        model_name = model_config['name']
        
        # 初始化模型
        model = model_class(self.config['num_phenotypes']).to(device)
        
        # 获取训练组件
        optimizer, scheduler, criterion = get_training_components(
            model, self.config, steps_per_epoch=len(train_loader)
        )
        
        # 训练模型
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_att_losses': [],
            'train_age_losses': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        patience = self.config['patience']
        counter = 0
        start_time = time.time()
        for epoch in range(self.config['num_epochs']):
            # 训练
            (train_loss, train_att_loss, train_age_loss), epoch_lrs = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, device,
                self.config['gradient_accumulation_steps']
            )
            
            # 验证
            val_loss, val_predictions, val_targets = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # 更新历史记录
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_att_losses'].append(train_att_loss)
            history['train_age_losses'].append(train_age_loss)
            history['learning_rates'].extend(epoch_lrs)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存最佳模型
                self.results_manager.save_model(model, f'best_model_{model_name}')
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logging.info(f"Early stopping for {model_name} at epoch {epoch}")
                    break
                    
            # 记录进度
            if epoch % 5 == 0:
                logging.info(f"{model_name} - Epoch {epoch}: "
                           f"Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}")
            
            if epoch % 10 == 0:
                elapsed = time.time() - start_time
                logging.info(f"Experiment {model_config['name']}: "
                            f"Training progress: Epoch {epoch}/{self.config['num_epochs']} "
                            f"({elapsed:.2f}s elapsed)")
        
        # 在测试集上评估
        model.load_state_dict(torch.load(os.path.join(
            self.results_manager.subdirs['models'], 
            f'best_model_{model_name}.pth'
        )))
        
        test_loss, predictions, targets = validate_epoch(model, test_loader, criterion, device)
        
        # 分别计算两个任务的指标
        results = {
            'history': history,
            'test_loss': test_loss,
            'metrics': {
                'attention': self.calculate_metrics(predictions[:, 0], targets[:, 0]),
                'age': self.calculate_metrics(predictions[:, 1], targets[:, 1])
            },
            'predictions': predictions,
            'targets': targets
        }
        
        # 保存注意力分析
        attention_data = self.results_manager.save_attention_analysis(
            model, test_loader, device, model_name)
        results['attention_analysis'] = attention_data
        
        # 保存错误分析
        error_data = self.results_manager.save_error_analysis(
            predictions, targets, np.array([test_loader.dataset.features_and_targets[i][1:] for i in range(len(test_loader.dataset))]))
        
        results['error_analysis'] = error_data
        
        return results
    
    def analyze_training_dynamics(self):
        """分析训练动态"""
        plt.figure(figsize=(15, 10))
        
        # 1. 训练损失对比
        plt.subplot(2, 2, 1)
        for name, result in self.results.items():
            plt.plot(result['history']['train_losses'], label=name)
        plt.title('Training Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 2. 验证损失对比
        plt.subplot(2, 2, 2)
        for name, result in self.results.items():
            plt.plot(result['history']['val_losses'], label=name)
        plt.title('Validation Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 3. 收敛速度对比
        plt.subplot(2, 2, 3)
        convergence_data = []
        for name, result in self.results.items():
            val_losses = result['history']['val_losses']
            min_loss_epoch = np.argmin(val_losses)
            convergence_data.append({
                'name': name,
                'best_epoch': min_loss_epoch,
                'best_loss': val_losses[min_loss_epoch]
            })
        
        names = [d['name'] for d in convergence_data]
        epochs = [d['best_epoch'] for d in convergence_data]
        plt.bar(names, epochs)
        plt.title('Epochs to Best Performance')
        plt.ylabel('Epochs')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_manager.subdirs['plots']['training'], 'training_dynamics.png'))
        plt.close()
        
    def analyze_performance_metrics(self):
        """分析性能指标"""
        # 创建性能对比表格
        comparison_table = {task: pd.DataFrame(index=self.metrics_names) for task in self.tasks}
        # 首先添加baseline的结果
        for task in self.tasks:
            comparison_table[task]['baseline'] = pd.Series(
                self.baseline_results['metrics'][task]
            )

        for name, result in self.results.items():
            for task in self.tasks:
                metrics = result['metrics'][task]
                comparison_table[task][name] = pd.Series(metrics)
        
        # 计算相对于baseline的性能变化
        relative_change = {task: pd.DataFrame(index=self.metrics_names) for task in self.tasks}

        for task in self.tasks:
            baseline_metrics = comparison_table[task]['baseline']
            for model in self.experiment_configs.keys():
                relative_change[task][model] = (
                    (comparison_table[task][model] - baseline_metrics) 
                    / baseline_metrics * 100
                    )
        # 保存结果
        for task in self.tasks:
            comparison_table[task].to_csv(os.path.join(
                self.results_manager.subdirs['logs'], 
                f'{task}_metrics_comparison.csv'
            ))
            if task in relative_change:
                relative_change[task].to_csv(os.path.join(
                    self.results_manager.subdirs['logs'], 
                    f'{task}_relative_changes.csv'
                ))
        
        return comparison_table, relative_change
    
    #     return attention_stats
    def _ensure_numpy_array(self, data):
        """确保数据是numpy数组"""
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def _analyze_spatial_attention(self, attention_data):
        """分析空间注意力模式"""
        if not attention_data or 'spatial_attention_maps' not in attention_data.get('raw_data', {}):
            return None
            
        try:
            maps = attention_data['raw_data']['spatial_attention_maps']
            if not isinstance(maps, (list, np.ndarray)) or len(maps) == 0:
                return None
                
            maps = np.asarray(maps)
            if maps.size == 0:
                return None
                
            # 重塑数据以便分析
            if maps.ndim > 2:
                maps = maps.reshape(maps.shape[0], -1)
                
            # 计算基本统计量
            stats = {
                'mean_attention': float(np.mean(maps)),
                'std_attention': float(np.std(maps)),
                'max_attention': float(np.max(maps)),
                'min_attention': float(np.min(maps)),
                'median_attention': float(np.median(maps)),
                'percentiles': {
                    '25': float(np.percentile(maps, 25)),
                    '75': float(np.percentile(maps, 75))
                }
            }
            
            # 计算注意力分布
            attention_distribution = np.histogram(maps.flatten(), bins=20, range=(0, 1))[0]
            stats['distribution'] = attention_distribution.tolist()
            
            return stats
        except Exception as e:
            logging.error(f"Error analyzing spatial attention: {str(e)}")
            return None

    def _analyze_channel_attention(self, attention_data):
        """分析通道注意力模式"""
        if not attention_data or 'channel_attention_weights' not in attention_data.get('raw_data', {}):
            return None
            
        try:
            weights = attention_data['raw_data']['channel_attention_weights']
            if not isinstance(weights, (list, np.ndarray)) or len(weights) == 0:
                return None
                
            weights = np.asarray(weights)
            if weights.size == 0:
                return None
                
            # 重塑数据以便分析
            if weights.ndim > 2:
                weights = weights.reshape(weights.shape[0], -1)
                
            # 计算基本统计量
            stats = {
                'mean_weights': float(np.mean(weights)),
                'std_weights': float(np.std(weights)),
                'max_weights': float(np.max(weights)),
                'min_weights': float(np.min(weights)),
                'median_weights': float(np.median(weights))
            }
            
            # 分析通道激活模式
            channel_means = np.mean(weights, axis=0)
            channel_stds = np.std(weights, axis=0)
            
            stats['channel_statistics'] = {
                'most_active_channels': np.argsort(channel_means)[-5:].tolist(),  # top 5 active channels
                'least_active_channels': np.argsort(channel_means)[:5].tolist(),  # bottom 5 active channels
                'channel_means': channel_means.tolist(),
                'channel_stds': channel_stds.tolist()
            }
            
            return stats
        except Exception as e:
            logging.error(f"Error analyzing channel attention: {str(e)}")
            return None

    def _analyze_feature_attention(self, attention_data):
        """分析特征注意力模式"""
        if not attention_data or 'feature_attention_weights' not in attention_data.get('raw_data', {}):
            return None
            
        try:
            weights = attention_data['raw_data']['feature_attention_weights']
            if not isinstance(weights, (list, np.ndarray)) or len(weights) == 0:
                return None
                
            weights = np.asarray(weights)
            if weights.size == 0:
                return None
                
            # 重塑数据以便分析
            if weights.ndim > 2:
                weights = weights.reshape(weights.shape[0], -1)
                
            # 计算基本统计量
            stats = {
                'mean_weights': float(np.mean(weights)),
                'std_weights': float(np.std(weights)),
                'max_weights': float(np.max(weights)),
                'min_weights': float(np.min(weights)),
                'median_weights': float(np.median(weights)),
                'percentiles': {
                    '25': float(np.percentile(weights, 25)),
                    '75': float(np.percentile(weights, 75))
                }
            }
            
            # 分析特征权重分布
            feature_means = np.mean(weights, axis=0)
            feature_stds = np.std(weights, axis=0)
            
            stats['feature_importance'] = {
                'top_features': np.argsort(feature_means)[-5:].tolist(),  # top 5 important features
                'bottom_features': np.argsort(feature_means)[:5].tolist(),  # bottom 5 important features
                'feature_means': feature_means.tolist(),
                'feature_stds': feature_stds.tolist()
            }
            
            return stats
        except Exception as e:
            logging.error(f"Error analyzing feature attention: {str(e)}")
            return None

    def compare_attention_patterns(self):
        """比较不同模型的注意力模式"""
        comparisons = {}
        
        try:
            # 获取基准模型的注意力统计
            if hasattr(self, 'baseline_results'):
                baseline_attention = self.baseline_results.get('attention_analysis', {})
                baseline_stats = {
                    'spatial': self._analyze_spatial_attention(baseline_attention),
                    'channel': self._analyze_channel_attention(baseline_attention),
                    'feature': self._analyze_feature_attention(baseline_attention)
                }
                
                # 对每个实验模型进行比较
                for name, result in self.results.items():
                    attention_data = result.get('attention_analysis', {})
                    comparisons[name] = {}
                    
                    # 根据模型类型选择要比较的注意力机制
                    if 'no_spatial_attention' in name:
                        # 比较channel和feature attention
                        comparisons[name]['channel'] = self._compare_attention_stats(
                            baseline_stats['channel'],
                            self._analyze_channel_attention(attention_data)
                        )
                        comparisons[name]['feature'] = self._compare_attention_stats(
                            baseline_stats['feature'],
                            self._analyze_feature_attention(attention_data)
                        )
                    elif 'no_feature_attention' in name:
                        # 比较spatial和channel attention
                        comparisons[name]['spatial'] = self._compare_attention_stats(
                            baseline_stats['spatial'],
                            self._analyze_spatial_attention(attention_data)
                        )
                        comparisons[name]['channel'] = self._compare_attention_stats(
                            baseline_stats['channel'],
                            self._analyze_channel_attention(attention_data)
                        )
                        
            return comparisons
        except Exception as e:
            logging.error(f"Error comparing attention patterns: {str(e)}")
            return {}

    def _compare_attention_stats(self, baseline_stats, ablation_stats):
        """比较基准模型和消融模型的注意力统计信息"""
        if not baseline_stats or not ablation_stats:
            return None
            
        try:
            comparison = {}
            
            # 比较基本统计量
            for key in ['mean_weights', 'std_weights', 'max_weights', 'min_weights']:
                if key in baseline_stats and key in ablation_stats:
                    comparison[f'{key}_diff'] = ablation_stats[key] - baseline_stats[key]
                    comparison[f'{key}_relative_change'] = (
                        (ablation_stats[key] - baseline_stats[key]) / baseline_stats[key] * 100
                        if baseline_stats[key] != 0 else float('inf')
                    )
                    
            return comparison
        except Exception as e:
            logging.error(f"Error comparing attention statistics: {str(e)}")
            return None

            
    # def _process_single_model_attention(self, attention_data):
        """处理单个模型的注意力数据"""
        stats = {}
        try:
            raw_data = attention_data.get('raw_data', {})
            
            # 处理 channel attention weights
            if ('channel_attention_maps' in raw_data and raw_data['channel_attention_maps']) and (len(raw_data['channel_attention_weights']) > 0):
                if raw_data.get('channel_attention_weights') is not None:
                    try:
                        weights = np.asarray(raw_data['channel_attention_weights'])
                        if len(weights) > 0:  # 确保数据不为空
                            if weights.ndim > 2:
                                weights = weights.reshape(weights.shape[0], -1)
                            
                            stats['channel_attention'] = {
                                'mean': float(self._safe_mean(weights)),
                                'std': float(self._safe_std(weights)),
                                'max': float(np.nanmax(weights) if len(weights) > 0 else 0),
                                'min': float(np.nanmin(weights) if len(weights) > 0 else 0)
                            }
                    except Exception as e:
                        logging.warning(f"Error processing channel attention: {str(e)}")
            
            # 处理 spatial attention maps
            if ('spatial_attention_maps' in raw_data and raw_data['spatial_attention_maps']) and (len(raw_data['spatial_attention_maps']) > 0):
                if raw_data.get('spatial_attention_maps') is not None:
                    try:
                        maps = np.asarray(raw_data['spatial_attention_maps'])
                        if len(maps) > 0:  # 确保数据不为空
                            if maps.ndim > 2:
                                maps = maps.reshape(maps.shape[0], -1)
                            
                            stats['spatial_attention'] = {
                                'mean': float(self._safe_mean(weights)),
                                'std': float(self._safe_std(weights)),
                                'max': float(np.nanmax(weights) if len(weights) > 0 else 0),
                                'min': float(np.nanmin(weights) if len(weights) > 0 else 0)
                            }
                    except Exception as e:
                        logging.warning(f"Error processing spatial attention: {str(e)}")
            
            # 处理 feature attention weights
            if ('feature_attention_weights' in raw_data and raw_data['feature_attention_weights']) and (len(raw_data['feature_attention_weights']) > 0):
                if raw_data.get('feature_attention_weights') is not None:
                    try:
                        weights = np.asarray(raw_data['feature_attention_weights'])
                        if len(weights) > 0:  # 确保数据不为空
                            if weights.ndim > 2:
                                weights = weights.reshape(weights.shape[0], -1)
                            
                            stats['feature_attention'] = {
                                'mean': float(self._safe_mean(weights)),
                                'std': float(self._safe_std(weights)),
                                'max': float(np.nanmax(weights) if len(weights) > 0 else 0),
                                'min': float(np.nanmin(weights) if len(weights) > 0 else 0)
                            }
                    except Exception as e:
                        logging.warning(f"Error processing feature attention: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error in processing attention data: {str(e)}")
            return {}
        
        return stats
    

    def analyze_attention_patterns(self):
        """分析注意力模式"""
        attention_stats = {}
        
        # 添加baseline数据
        if hasattr(self, 'baseline_results'):
            try:
                baseline_attention = self.baseline_results.get('attention_analysis', {})
                if baseline_attention:
                    # 基准模型有所有类型的注意力机制
                    attention_stats['baseline'] = {
                        'spatial_attention': self._analyze_spatial_attention(baseline_attention),
                        'channel_attention': self._analyze_channel_attention(baseline_attention),
                        'feature_attention': self._analyze_feature_attention(baseline_attention)
                    }
                    # attention_stats['baseline'] = self._process_single_model_attention(baseline_attention)
            except Exception as e:
                logging.error(f"Error processing baseline attention data: {str(e)}")
        
        # 处理其他模型
        for name, result in self.results.items():
            attention_stats[name] = {}
            # attention_data = result.get('attention_analysis', {})
            try:
                attention_data = result.get('attention_analysis', {})
                if 'no_spatial_attention' in name:
                    # 只分析channel和feature attention
                    attention_stats[name]['channel_attention'] = self._analyze_channel_attention(attention_data)
                    attention_stats[name]['feature_attention'] = self._analyze_feature_attention(attention_data)
                elif 'no_feature_attention' in name:
                    # 只分析spatial和channel attention
                    attention_stats[name]['spatial_attention'] = self._analyze_spatial_attention(attention_data)
                    attention_stats[name]['channel_attention'] = self._analyze_channel_attention(attention_data)
                
            except Exception as e:
                logging.error(f"Error processing attention data for {name}: {str(e)}")
                continue
        
            # 保存注意力统计数据
        try:
            np.save(os.path.join(self.results_manager.subdirs['logs'], 'attention_stats.npy'), 
                    attention_stats)
        except Exception as e:
            logging.error(f"Error saving attention stats: {str(e)}")
        
        return attention_stats

    
    def plot_channel_attention_comparison(self):
        """比较不同通道的注意力分布"""
        plt.figure(figsize=(15, 5))
        
        # 包含baseline在内的所有结果
        all_results = {}
        if hasattr(self, 'baseline_results'):
            all_results['baseline'] = self.baseline_results
        all_results.update(self.results)
        
        valid_plots = 0
        for idx, (name, result) in enumerate(all_results.items()):
            try:
                attention_data = result['attention_analysis']
                channel_weights = attention_data['raw_data']['channel_attention_weights']
                
                # 确保数据是numpy数组
                channel_weights = self._ensure_numpy_array(channel_weights)
            
                # 确保数据是2D的
                if channel_weights.ndim > 2:
                    channel_weights = channel_weights.reshape(channel_weights.shape[0], -1)
                
                # 创建DataFrame并处理可能的NaN值
                df = pd.DataFrame(channel_weights).melt()
                df = df.dropna()  # 删除NaN值
                if not df.empty:
                    valid_plots += 1
                    plt.subplot(1, len(all_results), idx+1)
                    sns.boxplot(data=pd.DataFrame(channel_weights).melt())
                    plt.title(f'{name}\nChannel Attention Distribution')
                    plt.xlabel('Channel')
                    plt.ylabel('Attention Weight')
                else:
                    logging.warning(f"No valid data for plotting {name}")
            except Exception as e:
                logging.error(f"Error plotting attention for {name}: {str(e)}")
                continue

        if valid_plots > 0:
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_manager.subdirs['plots']['attention'], 
                                    'channel_attention_comparison.png'))
        else:
            logging.warning("No valid attention data to plot")
        
        plt.close()
    

    def run_ablation_study(self, image_data, phenotype_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            # 准备数据
            train_loader, val_loader, test_loader = get_data_loaders(
                image_data, phenotype_data, 
                batch_size=self.config['batch_size']
            )

            # 首先加载baseline结果
            self.baseline_results = self.load_baseline_results(test_loader, device)
            
            # 运行所有实验配置
            for config_name, config in self.experiment_configs.items():
                try:
                    logging.info(f"\nStarting experiment: {config_name}")
                    self.results[config_name] = self.run_single_experiment(
                        config, train_loader, val_loader, test_loader, device
                    )
                    self.save_intermediate_results(config_name)
                    # 添加内存清理
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    logging.error(f"Error in {config_name}: {str(e)}")
                    continue

            # 最终分析
            if len(self.results) > 0:
                try:
                    self.analyze_training_dynamics()
                    comparison_table, relative_change = self.analyze_performance_metrics()
                    attention_stats = self.analyze_attention_patterns()
                    self.plot_channel_attention_comparison()
                    # 保存最终总结
                    self.save_final_summary(comparison_table, relative_change, attention_stats)
            
                except Exception as e:
                    logging.error(f"Error in final analysis: {str(e)}")
        finally:
            # 确保结束时清理内存
            torch.cuda.empty_cache()
            gc.collect()
        
        return self.results

    def save_intermediate_results(self, config_name):
        """保存中间结果"""
        result_path = os.path.join(
            self.results_manager.subdirs['logs'],
            f'intermediate_{config_name}_results.npy'
        )
        np.save(result_path, self.results[config_name])
        
    def save_final_summary(self, comparison_table, relative_change, attention_stats):
        """保存最终分析总结"""
        try:
            summary_path = os.path.join(self.results_manager.subdirs['logs'], 
                                    'ablation_summary.txt')
            
            with open(summary_path, 'w') as f:
                f.write("Ablation Study Summary\n")
                f.write("=====================\n\n")
                
                # 添加每个任务的结果
                for task in self.tasks:
                    if task in comparison_table and comparison_table[task] is not None:
                        f.write(f"\n{task.upper()} Task Results:\n")
                        f.write("-" * 40 + "\n")
                        f.write("\nAbsolute Metrics:\n")
                        f.write(comparison_table[task].to_string())
                        if task in relative_change and relative_change[task] is not None:
                            f.write("\n\nRelative Changes (%):\n")
                            f.write(relative_change[task].to_string())
                        f.write("\n")
                
                # 添加注意力分析结果
                if attention_stats:
                    f.write("\nAttention Pattern Analysis:\n")
                    f.write("-" * 40 + "\n")
                    f.write(json.dumps(attention_stats, indent=2))
        except Exception as e:
            logging.error(f"Error saving final summary: {str(e)}")
    


if __name__ == "__main__":
    # 设置实验配置
    config = {
        'batch_size': 8,
        'num_epochs': 200,
        'initial_lr': 1e-4,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 4,
        'num_phenotypes': 3,
        'patience': 20,
    }

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'ablation_study_lh_results_{timestamp}'

    
    # 指定baseline模型路径
    full_model_path = '/home/jouyang1/training_all_data_lh_mirror_attmap_20250205_031917/models/all_lh_best_model.pth'
    # '/home/jouyang1/training_all_data_lh_mirror20250128_154712/models/all_lh_best_model_improved.pth'
    # '/home/jouyang1/training_all_data_rh_mirror_attmap_20250205_041248/models/all_rh_best_model.pth'
    # '/home/jouyang1/training_all_data_lh_mirror_attmap_20250205_031917/models/all_lh_best_model.pth's
    
    # 数据路径
    image_path = '/projects/0/einf1049/scratch/jouyang/all_cnn_lh_brainimages.npy'
    # image_path = '/home/jouyang1/sample_cnn_lh_brainimages.npy'
    # image_path = '/projects/0/einf1049/scratch/jouyang/all_cnn_rh_brainimages.npy'
    phenotype_path = '/home/jouyang1/all_normalised_phenotypes_correct.npy'
    # phenotype_path = '/home/jouyang1/sample_normalised_phenotype.npy'
    
    # 加载数据
    image_data, loaded_phenotype_tensor = load_data(
        image_path, 
        phenotype_path,
        use_mmap=True
    )
    
    # 创建实验管理器
    ablation_manager = AblationStudyManager(config, output_dir, full_model_path)
    
    # 运行消融实验
    results = ablation_manager.run_ablation_study(image_data, loaded_phenotype_tensor)
    
    print("\nAblation study completed. Results saved in:", output_dir)