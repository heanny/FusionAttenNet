"""
training_output_all_lh_YYYYMMDD_HHMMSS/
├── attention/
│   ├── raw_data/
│   └── analysis/
│
├── cross_validation/
│   ├── fold_results/
│   ├── metrics/
│   └── analysis/
│
├── error_analysis/
│   ├── raw_data/
│   ├── statistics/
│   └── detailed_analysis/
│
├── logs/                      # 添加日志目录
│   └── training.log
│
├── models/                    # 添加模型目录
│   ├── best_model.pth
│   └── fold_*/
│
└── plots/                     # 添加可视化目录
    ├── attention_plots/
    ├── training_plots/
    └── error_plots/

Dropout率通常在0.2-0.5之间
选择原则：
越靠近输入层，dropout率越小（如0.2-0.3）
越靠近输出层，dropout率越大（如0.4-0.5）
如果层的神经元数量多，可以用更大的dropout率

/home/jouyang1/my_script.sh
"""
# 
import logging
import csv
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
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

# Training configuration
config = {
    'batch_size': 8,
    'num_epochs': 200,
    'initial_lr': 1e-4,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 4,
    'num_phenotypes': 3,
    'patience': 20,
    'counter': 0
    # 'warmup_epochs': 5
}

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


# Training configuration
# config = {
#     'batch_size': 16,
#     'num_epochs': 300,
#     'initial_lr': 1e-4,
#     'weight_decay': 0.01,
#     'gradient_accumulation_steps': 2,
#     'num_phenotypes': 3,
#     'patience': 20,
#     'counter': 0
#     # 'warmup_epochs': 5
# }
#     # best_val_loss = float('inf')
#     # train_losses = []
#     # val_losses = []
#     # counter = 0

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
def train_epoch(model, loader, optimizer, criterion, device, gradient_accumulation_steps):
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




# class ResultsManager:
#     """改进的实验结果管理器"""
#     def __init__(self, output_dir):
#         self.output_dir = output_dir
#         self.create_directories()
        
#     def create_directories(self):
#         """创建完整的目录结构"""
#         self.subdirs = {
#             'logs': os.path.join(self.output_dir, 'logs'),
#             'models': os.path.join(self.output_dir, 'models'),
#             'plots': os.path.join(self.output_dir, 'plots'),
#             'training': os.path.join(self.output_dir, 'training_history'),
#             'cv': os.path.join(self.output_dir, 'cross_validation'),
#             'ablation': os.path.join(self.output_dir, 'ablation_study'),
#             'attention': os.path.join(self.output_dir, 'attention_analysis'),
#             'error': os.path.join(self.output_dir, 'error_analysis'),
#             'baseline': os.path.join(self.output_dir, 'baseline')
#         }
        
#         for dir_path in self.subdirs.values():
#             os.makedirs(dir_path, exist_ok=True)
    
#     def save_model(self, model, name, fold=None):
#         """保存模型权重"""
#         if fold is not None:
#             name = f"{name}_fold_{fold}.pth"
#         else:
#             name = f"{name}.pth"
#         path = os.path.join(self.subdirs['models'], name)
#         torch.save(model.state_dict(), path)
        
#     def save_config(self, config):
#         """保存实验配置"""
#         path = os.path.join(self.output_dir, 'experiment_config.json')
#         with open(path, 'w') as f:
#             json.dump(config, f, indent=4)
            
#     def save_training_history(self, history, prefix=''):
#         """保存训练历史"""
#         # 保存数据
#         save_path = self.subdirs['training']
#         np.save(os.path.join(save_path, f'{prefix}training_metrics.npy'), history)
        
#         # 绘制训练历史图表
#         self.plot_training_history(history, save_path, prefix)
        
#     def plot_training_history(self, history, save_path, prefix=''):
#         """绘制详细的训练历史图表"""
#         plt.figure(figsize=(15, 10))
        
#         # 1. 总体损失
#         plt.subplot(2, 2, 1)
#         plt.plot(history['train_losses'], label='Train Loss')
#         plt.plot(history['val_losses'], label='Val Loss')
#         plt.title('Overall Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.grid(True)
        
#         # 2. 分任务损失
#         plt.subplot(2, 2, 2)
#         plt.plot(history['train_att_losses'], label='Attention Loss')
#         plt.plot(history['train_age_losses'], label='Age Loss')
#         plt.title('Task-specific Losses')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.grid(True)
        
#         # 3. 学习率变化
#         plt.subplot(2, 2, 3)
#         plt.plot(history['learning_rates'])
#         plt.title('Learning Rate Schedule')
#         plt.xlabel('Step')
#         plt.ylabel('Learning Rate')
#         plt.yscale('log')
#         plt.grid(True)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_path, f'{prefix}training_history.png'))
#         plt.close()
        
#     def save_attention_analysis(self, model, test_loader, device, fold=None):
#         """保存完整的注意力分析"""
#         save_path = self.subdirs['attention']
#         prefix = f'fold_{fold}_' if fold is not None else ''
        
#         model.eval()
#         with torch.no_grad():
#             # 收集多个batch的注意力图
#             attention_data = {
#                 'spatial_attention_maps': [],
#                 'channel_attention_weights': [],
#                 'feature_attention_weights': [],
#                 'targets': [],
#                 'predictions': []
#             }
            
#             for i, (brain_maps, extra_features, targets) in enumerate(test_loader):
#                 if i >= 5:  # 收集前5个batch
#                     break
                    
#                 brain_maps = brain_maps.to(device)
#                 extra_features = extra_features.to(device)
                
#                 # 获取所有注意力权重
#                 features = model.frontend(brain_maps)
#                 spatial_att = model.spatial_attention(features)
#                 channel_att = model.channel_attention(features)
                
#                 # 获取特征注意力
#                 x = model.avg_pool(features * channel_att)
#                 x = x.view(x.size(0), -1)
#                 phenotype_features = model.phenotype_encoder(extra_features)
#                 combined = torch.cat([x, phenotype_features], dim=1)
#                 shared_features = model.shared_features(combined)
#                 feature_att = model.feature_attention(shared_features)
                
#                 # 获取预测结果
#                 outputs = model(brain_maps, extra_features)
                
#                 # 保存数据
#                 attention_data['spatial_attention_maps'].extend(spatial_att.cpu().numpy())
#                 attention_data['channel_attention_weights'].extend(channel_att.cpu().numpy())
#                 attention_data['feature_attention_weights'].extend(feature_att.cpu().numpy())
#                 attention_data['targets'].extend(targets.numpy())
#                 attention_data['predictions'].extend(outputs.cpu().numpy())
        
#         # 保存原始数据
#         np.save(os.path.join(save_path, f'{prefix}attention_data.npy'), attention_data)
        
#         # 绘制注意力可视化
#         self.plot_attention_analysis(attention_data, save_path, prefix)
        
#     def save_cv_results(self, cv_results):
#         """保存交叉验证结果"""
#         save_path = self.subdirs['cv']
        
#         # 保存原始数据
#         np.save(os.path.join(save_path, 'cv_results.npy'), cv_results)
        
#         # 绘制交叉验证分析图
#         self.plot_cv_analysis(cv_results, save_path)
        
#     def plot_cv_analysis(self, cv_results, save_path):
#         """绘制交叉验证分析图"""
#         plt.figure(figsize=(15, 10))
        
#         # 1. 各折验证损失
#         plt.subplot(2, 2, 1)
#         for fold, history in enumerate(cv_results['fold_train_history']):
#             plt.plot(history['val_losses'], label=f'Fold {fold+1}')
#         plt.title('Validation Loss by Fold')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.grid(True)
        
#         # 2. R²分数分布
#         plt.subplot(2, 2, 2)
#         metrics = ['MSE', 'MAE', 'R^2']
#         tasks = ['sum_att', 'age']
        
#         for i, task in enumerate(tasks):
#             values = [fold[task]['R^2'] for fold in cv_results['fold_metrics']]
#             plt.boxplot(values, positions=[i], labels=[task])
#         plt.title('R² Score Distribution')
#         plt.grid(True)
        
#         # 3. 预测vs真实值（所有折）
#         plt.subplot(2, 2, 3)
#         all_preds = np.concatenate(cv_results['predictions'])
#         all_targets = np.concatenate(cv_results['targets'])
        
#         plt.scatter(all_targets[:, 0], all_preds[:, 0], alpha=0.5, label='Attention')
#         plt.scatter(all_targets[:, 1], all_preds[:, 1], alpha=0.5, label='Age')
#         plt.plot([all_targets.min(), all_targets.max()], 
#                 [all_targets.min(), all_targets.max()], 
#                 'r--', label='Ideal')
#         plt.title('Predictions vs Targets (All Folds)')
#         plt.xlabel('True Values')
#         plt.ylabel('Predictions')
#         plt.legend()
#         plt.grid(True)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_path, 'cv_analysis.png'))
#         plt.close()
        
#     def save_error_analysis(self, predictions, targets, metadata):
#         """保存错误分析结果"""
#         save_path = self.subdirs['error']
        
#         error_data = {
#             'predictions': predictions,
#             'targets': targets,
#             'metadata': metadata,
#             'abs_errors': np.abs(predictions - targets),
#             'relative_errors': np.abs((predictions - targets) / targets)
#         }
        
#         # 保存数据
#         np.save(os.path.join(save_path, 'error_analysis.npy'), error_data)
        
#         # 绘制错误分析图
#         self.plot_error_analysis(error_data, save_path)
    
#     def get_subdirs(self):
#         """返回所有子目录路径"""
#         return self.subdirs


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

    def save_attention_analysis(self, model, test_loader, device, fold=None):
        """增强的注意力分析与存储"""
        prefix = f'fold_{fold}_' if fold is not None else ''
        model.eval()
        
        attention_data = {
            'raw_data': {
                'spatial_attention_maps': [],
                'channel_attention_weights': [],
                'feature_attention_weights': [],
                'layer_attention_weights': {},
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
        
        # 收集所有数据
        with torch.no_grad():
            for brain_maps, extra_features, targets in test_loader:
                brain_maps = brain_maps.to(device)
                extra_features = extra_features.to(device)
                
                # 获取中间特征和attention
                features = model.frontend(brain_maps)
                spatial_att = model.spatial_attention(features)
                channel_att = model.channel_attention(features)
                
                # 获取feature attention
                x = model.avg_pool(features * channel_att)
                x = x.view(x.size(0), -1)
                phenotype_features = model.phenotype_encoder(extra_features)
                combined = torch.cat([x, phenotype_features], dim=1)
                shared_features = model.shared_features(combined)
                feature_att = model.feature_attention(shared_features)
                
                outputs = model(brain_maps, extra_features)
                
                # 保存数据
                attention_data['raw_data']['spatial_attention_maps'].append(spatial_att.cpu().numpy())
                attention_data['raw_data']['channel_attention_weights'].append(channel_att.cpu().numpy())
                attention_data['raw_data']['feature_attention_weights'].append(feature_att.cpu().numpy())
                attention_data['raw_data']['targets'].append(targets.numpy())
                attention_data['raw_data']['predictions'].append(outputs.cpu().numpy())
                attention_data['raw_data']['metadata'].append(extra_features.cpu().numpy())
        
        # 合并所有数据
        for key in attention_data['raw_data']:
            if isinstance(attention_data['raw_data'][key], list):
                attention_data['raw_data'][key] = np.concatenate(attention_data['raw_data'][key])
        
        # 计算统计信息
        attention_data['analysis']['attention_statistics'] = {
            'spatial_attention_mean': np.mean(attention_data['raw_data']['spatial_attention_maps'], axis=0),
            'spatial_attention_std': np.std(attention_data['raw_data']['spatial_attention_maps'], axis=0),
            'channel_attention_mean': np.mean(attention_data['raw_data']['channel_attention_weights'], axis=0),
            'feature_attention_mean': np.mean(attention_data['raw_data']['feature_attention_weights'], axis=0),
        }
        
        # 保存数据
        for data_type, data in attention_data['raw_data'].items():
            np.save(os.path.join(self.subdirs['attention']['raw_data'], f'{prefix}{data_type}.npy'), data)
        
        for analysis_type, analysis in attention_data['analysis'].items():
            np.save(os.path.join(self.subdirs['attention']['analysis'], f'{prefix}{analysis_type}.npy'), analysis)
        
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

    # def save_error_analysis(self, predictions, targets, metadata):
    #     """增强的错误分析与存储"""
    #     # 计算各种错误
    #     error_data = {
    #         'raw_data': {
    #             'predictions': predictions,
    #             'targets': targets,
    #             'metadata': metadata,
    #             'errors': predictions - targets,
    #             'abs_errors': np.abs(predictions - targets),
    #             'relative_errors': np.abs((predictions - targets) / targets)
    #         }
    #     }
        
    #     # 计算统计信息
    #     error_data['statistics'] = {
    #         'attention_score': {
    #             'mae': np.mean(np.abs(predictions[:, 0] - targets[:, 0])),
    #             'mse': np.mean((predictions[:, 0] - targets[:, 0])**2),
    #             'rmse': np.sqrt(np.mean((predictions[:, 0] - targets[:, 0])**2)),
    #             'error_distribution': {
    #                 'mean': np.mean(error_data['raw_data']['abs_errors'][:, 0]),
    #                 'std': np.std(error_data['raw_data']['abs_errors'][:, 0]),
    #                 'quartiles': np.percentile(error_data['raw_data']['abs_errors'][:, 0], [25, 50, 75])
    #             }
    #         },
    #         'age': {
    #             'mae': np.mean(np.abs(predictions[:, 1] - targets[:, 1])),
    #             'mse': np.mean((predictions[:, 1] - targets[:, 1])**2),
    #             'rmse': np.sqrt(np.mean((predictions[:, 1] - targets[:, 1])**2)),
    #             'error_distribution': {
    #                 'mean': np.mean(error_data['raw_data']['abs_errors'][:, 1]),
    #                 'std': np.std(error_data['raw_data']['abs_errors'][:, 1]),
    #                 'quartiles': np.percentile(error_data['raw_data']['abs_errors'][:, 1], [25, 50, 75])
    #             }
    #         }
    #     }
        
    #     # 识别和分析高错误样本
    #     for i, target_type in enumerate(['attention_score', 'age']):
    #         errors = error_data['raw_data']['errors'][:, i]
    #         error_mean = np.mean(errors)
    #         error_std = np.std(errors)
    #         high_error_mask = np.abs(errors - error_mean) > 2 * error_std
            
    #         error_data['detailed_analysis'] = {
    #             target_type: {
    #                 'high_error_samples': {
    #                     'indices': np.where(high_error_mask)[0],
    #                     'predictions': predictions[high_error_mask, i],
    #                     'targets': targets[high_error_mask, i],
    #                     'metadata': metadata[high_error_mask],
    #                     'errors': errors[high_error_mask]
    #                 },
    #                 'error_correlations': {
    #                     'aggressive_score': np.corrcoef(metadata[:, 0], np.abs(errors))[0, 1],
    #                     'sex': np.corrcoef(metadata[:, 1], np.abs(errors))[0, 1],
    #                     'edu_level': np.corrcoef(metadata[:, 2], np.abs(errors))[0, 1]
    #                 }
    #             }
    #         }
        
    #     # 保存所有结果
    #     for data_type, data in error_data.items():
    #         save_dir = self.subdirs['error_analysis'][data_type]
    #         np.save(os.path.join(save_dir, f'{data_type}.npy'), data)
        
    #     return error_data

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

def perform_error_analysis(model, test_loader, device):
    """执行详细的错误分析"""
    model.eval()
    predictions = []
    targets = []
    metadata = []  # 可以包含年龄、性别等信息
    
    with torch.no_grad():
        for brain_maps, extra_features, target in test_loader:
            brain_maps = brain_maps.to(device)
            extra_features = extra_features.to(device)
            output = model(brain_maps, extra_features)
            
            predictions.extend(output.cpu().numpy())
            targets.extend(target.numpy())
            metadata.extend(extra_features.cpu().numpy())
    
    return np.array(predictions), np.array(targets), np.array(metadata)

def create_fold_loaders(image_data, phenotype_data, train_idx, val_idx, batch_size=16, num_workers=2):
    """创建每个fold的数据加载器"""
    # 计算数据统计信息（使用训练集）
    mean = np.mean(image_data[train_idx], axis=(0, 2, 3))
    std = np.std(image_data[train_idx], axis=(0, 2, 3))
    
    # 创建训练集和验证集
    train_dataset = BrainMapDataset(
        image_data[train_idx],
        phenotype_data[train_idx],
        mean=mean,
        std=std,
        training=True
    )
    
    val_dataset = BrainMapDataset(
        image_data[val_idx],
        phenotype_data[val_idx],
        mean=mean,
        std=std,
        training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader

def get_predictions(model, loader, device):
    """获取模型在给定数据集上的预测结果"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for brain_maps, extra_features, target in loader:
            brain_maps = brain_maps.to(device)
            extra_features = extra_features.to(device)
            output = model(brain_maps, extra_features)
            
            predictions.extend(output.cpu().numpy())
            targets.extend(target.numpy())
    
    return np.array(predictions), np.array(targets)

def train_fold(model, train_loader, val_loader, optimizer, scheduler, criterion, device, config):
    """训练单个fold"""
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_att_losses': [],
        'train_age_losses': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    patience = 20
    counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training
        (train_loss, train_att_loss, train_age_loss), epoch_lrs = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config['gradient_accumulation_steps']
        )
        
        # Validation
        val_loss, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        # 记录历史
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_att_losses'].append(train_att_loss)
        history['train_age_losses'].append(train_age_loss)
        history['learning_rates'].extend(epoch_lrs)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return history

def perform_cross_validation(image_data, phenotype_data, model_template, device, config, k_folds=5):
    """执行k折交叉验证"""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    cv_results = {
        'fold_metrics': [],
        'predictions': [],
        'targets': [],
        'fold_train_history': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_data)):
        print(f"\nStarting fold {fold+1}/{k_folds}")
        
        # 创建该fold的数据加载器
        train_loader, val_loader = create_fold_loaders(
            image_data, phenotype_data, train_idx, val_idx,
            batch_size=config['batch_size']
        )
        
        # 创建新的模型实例
        model = copy.deepcopy(model_template).to(device)
        
        # 设置优化器和调度器
        optimizer, scheduler, criterion = get_training_components(
            model, config, steps_per_epoch=len(train_loader)
        )
        
        # 训练该fold
        fold_history = train_fold(
            model, train_loader, val_loader, optimizer,
            scheduler, criterion, device, config
        )
        
        # 在验证集上评估
        metrics = evaluate_model(model, val_loader, device, None)  # 不保存图表
        cv_results['fold_metrics'].append(metrics)
        
        # 获取预测结果
        predictions, targets = get_predictions(model, val_loader, device)
        cv_results['predictions'].append(predictions)
        cv_results['targets'].append(targets)
        cv_results['fold_train_history'].append(fold_history)
        
        # 清理内存
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()
    
    del train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return cv_results


# main
if __name__ == "__main__":
    # 创建输出文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'training_all_data_lh_mirror_attmap_{timestamp}'
    results_manager = ResultsManager(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    
    # 创建子文件夹
    # plots_dir = os.path.join(output_dir, 'plots')
    # logs_dir = os.path.join(output_dir, 'logs')
    # models_dir = os.path.join(output_dir, 'models')
    # os.makedirs(plots_dir, exist_ok=True)
    # os.makedirs(logs_dir, exist_ok=True)
    # os.makedirs(models_dir, exist_ok=True)

    # 修改日志配置
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(results_manager.subdirs['logs'], "all_lh_training.log")),
                        logging.StreamHandler()
                    ])
    
    # results_manager = ResultsManager(output_dir)
    # results_manager = ResultsManager('experiment_results')
    
    # training
    # num_epochs = 1
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    counter = 0
    
    # Set memory efficient device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # May help with speed?? not sure...

    # 数据路径
    image_path = '/projects/0/einf1049/scratch/jouyang/all_cnn_lh_brainimages.npy'
    # image_path = '/home/jouyang1/sample_cnn_lh_brainimages.npy'
    # phenotype_path = '/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes.npy'
    phenotype_path = '/home/jouyang1/all_normalised_phenotypes_correct.npy'
    # phenotype_path = '/home/jouyang1/sample_normalised_phenotype.npy'
    # order: sum_att  sum_agg	age	  sex(0/1)	edu_maternal(0/1/2) 
    
    torch.cuda.empty_cache()
    # load data
    image_data, loaded_phenotype_tensor = load_data(
        image_path, 
        phenotype_path,
        use_mmap=True  # 启用内存映射
    )

    # 保存实验配置
    full_config = {
        'model_config': config,
        'data_paths': {
            'image_path': image_path,
            'phenotype_path': phenotype_path
        },
        'training_params': {
            'num_epochs': config['num_epochs'],
            'patience': config['patience'],
            'num_phenotypes': config['num_phenotypes']
        },
        'timestamp': timestamp
    }
    results_manager.save_config(full_config)

    # print check for shape
    # print("\nData Shapes Debugging:")
    # print(f"Image data shape: {image_data.shape}")
    # print(f"Phenotype tensor shape: {loaded_phenotype_tensor.shape}")

    # Create smaller train/val/test splits
    # indices = np.arange(len(image_data))
    # train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    # train_idx, val_idx = train_test_split(train_val_idx, test_size=0.11111, random_state=42)

    # # Create datasets using indices
    # train_dataset = BrainMapDataset(image_data[train_idx], loaded_phenotype_tensor[train_idx])
    # val_dataset = BrainMapDataset(image_data[val_idx], loaded_phenotype_tensor[val_idx])
    # test_dataset = BrainMapDataset(image_data[test_idx], loaded_phenotype_tensor[test_idx])
    
    # Reduce batch size and use appropriate num_workers
    # batch_size = 32  # Reduced from 16
    # num_workers = 2  # Adjust based on your system

    train_loader, val_loader, test_loader = get_data_loaders(image_data, loaded_phenotype_tensor, batch_size=config['batch_size'], num_workers=2)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
    #                         num_workers=num_workers, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, 
    #                       num_workers=num_workers, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, 
    #                        num_workers=num_workers, pin_memory=True)
    

    # initialize model, optimizer, and loss functions
    # num_phenotypes = 3  # (12595,5) only for age, sex(0/1), edu_maternal(0/1/2)
    model = BrainADHDModel(config['num_phenotypes']).to(device)
    train_steps_per_epoch = len(train_loader) // config['gradient_accumulation_steps']

    optimizer, scheduler, criterion = get_training_components(model, config, steps_per_epoch=train_steps_per_epoch)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    # # optimizer, scheduler = get_optimizer_and_scheduler(model, num_epochs, train_steps_per_epoch)
    # criterion = nn.MSELoss()

    # # learning rate scheduler 
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, min_lr=1e-6, verbose=True)
    

    # save the training logs
    # csv_file = 'all_lh_training_loss_predictionM1_images_improved.csv'
    csv_file = os.path.join(results_manager.subdirs['logs'], 'all_lh_training_loss_predictionM1_images_improved.csv')

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

    # training starts
    start_time = time.time()
    learning_rates = []
    all_learning_rates = []

    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()

        # 记录每个epoch开始时的学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Training
        (train_loss, train_att_loss, train_age_loss), epoch_lrs = train_epoch(model, train_loader, optimizer, criterion, device, config['gradient_accumulation_steps'])
        all_learning_rates.extend(epoch_lrs)

        # Validation
        val_loss, _, _ = validate_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start_time
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                f"Train Loss: {train_loss:.4f} (Attention: {train_att_loss:.4f}, Age: {train_age_loss:.4f}), "
                f"Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Scheduler step
        # scheduler.step(val_loss)

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), 'all_lh_best_model_improved.pth')
            results_manager.save_model(model, 'all_lh_best_model')
            # model_path = os.path.join(models_dir, 'all_lh_best_model_improved.pth')
            # torch.save(model.state_dict(), model_path)
            counter = 0
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            counter += 1
        if counter >= config['patience']:
            logging.info("Early stopping")
            break

        # 每个epoch更新
        train_history['train_losses'].append(train_loss)
        train_history['val_losses'].append(val_loss)
        train_history['train_att_losses'].append(train_att_loss)
        train_history['train_age_losses'].append(train_age_loss)
        train_history['learning_rates'].extend(epoch_lrs)

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

    results_manager.save_training_history(train_history)
    # np.save(os.path.join(plots_dir, 'lh_training_history.npy'), train_history)

    # 执行交叉验证
    # print(f"\nStarting cross-validation...")

    # 创建模型模板
    # model_template = BrainADHDModel(config['num_phenotypes']).to(device)

    # 执行交叉验证
    # cv_results = perform_cross_validation(
    #     image_data, 
    #     loaded_phenotype_tensor,
    #     model_template,
    #     device,
    #     config,
    #     k_folds=5
    # )

    # 保存交叉验证结果
    # cv_results_path = os.path.join(plots_dir, 'cross_validation_results.npy')
    # np.save(cv_results_path, cv_results)
    # results_manager.save_cv_results(cv_results)
    
    # 注意力分析
    # results_manager.save_attention_analysis(model, test_loader, device)
    attention_data = results_manager.save_attention_analysis(model, test_loader, device)

    # 错误分析
    predictions, targets, metadata = perform_error_analysis(model, test_loader, device)
    # results_manager.save_error_analysis(predictions, targets, metadata)
    error_data = results_manager.save_error_analysis(predictions, targets, metadata)

    # 分析和可视化交叉验证结果
    # plt.figure(figsize=(15, 5))

    # 1. 损失曲线
    # plt.subplot(1, 3, 1)
    # for fold in range(len(cv_results['fold_train_history'])):
    #     plt.plot(cv_results['fold_train_history'][fold]['val_losses'], 
    #             label=f'Fold {fold+1}')
    # plt.title('Validation Loss Across Folds')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()

    # 2. R2分数分布
    # plt.subplot(1, 3, 2)
    # att_r2_scores = [fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]
    # age_r2_scores = [fold['age']['R^2'] for fold in cv_results['fold_metrics']]
    # plt.boxplot([att_r2_scores, age_r2_scores], labels=['Attention', 'Age'])
    # plt.title('R² Score Distribution Across Folds')

    # 3. MSE分布
    # plt.subplot(1, 3, 3)
    # att_mse_scores = [fold['sum_att']['MSE'] for fold in cv_results['fold_metrics']]
    # age_mse_scores = [fold['age']['MSE'] for fold in cv_results['fold_metrics']]
    # plt.boxplot([att_mse_scores, age_mse_scores], labels=['Attention', 'Age'])
    # plt.title('MSE Distribution Across Folds')

    # plt.tight_layout()
    # plt.savefig(os.path.join(results_manager.subdirs['cross_validation']['analysis'], 'cross_validation_analysis.png'))
    # plt.close()

    # 打印平均指标
    # logging.info(f"\nCross-validation Results:")
    # metrics_names = ['MSE', 'MAE', 'R^2']
    # tasks = ['sum_att', 'age']

    # for task in tasks:
    #     print(f"\n{task.upper()} Metrics:")
    #     for metric in metrics_names:
    #         values = [fold[task][metric] for fold in cv_results['fold_metrics']]
    #         mean_value = np.mean(values)
    #         std_value = np.std(values)
    #         print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")
    
    # logging.info(f"\nCross-validation finished.")
    
    # 清理内存
    del image_data, loaded_phenotype_tensor
    gc.collect()
    torch.cuda.empty_cache()


    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f} seconds")


    # 绘制更详细的学习率变化图
    plt.figure(figsize=(15, 5))
    plt.plot(all_learning_rates)
    plt.title('Learning Rate Changes During Training')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.yscale('log')  # 使用对数刻度更容易观察变化
    plt.grid(True)
    plt.savefig(os.path.join(results_manager.subdirs['plots']['training'], 'step_detailed_learning_rate_schedule.png'))
    plt.close()

    # 训练结束后绘制学习率曲线
    plt.figure(figsize=(10,5))
    plt.plot(learning_rates)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(results_manager.subdirs['plots']['training'], 'learning_rate_schedule.png'))
    plt.close()

    # plot learning rate curves
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # plt.savefig('all_lh_learning_curve_predict_brainimages_improved.png')
    plt.savefig(os.path.join(results_manager.subdirs['plots']['training'], 'all_lh_learning_curve_predict_brainimages_improved.png'))
    plt.close()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # 定义评估函数
    def evaluate_model(model, test_loader, device, plots_dir):
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for brain_maps, extra_features, targets in test_loader:
                brain_maps = brain_maps.to(device)
                extra_features = extra_features.to(device)
                outputs = model(brain_maps, extra_features)
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 计算每个目标变量的指标
        metrics = {}
        for i, target_name in enumerate(['sum_att', 'age']):
            mse = mean_squared_error(all_targets[:, i], all_predictions[:, i])
            mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
            r2 = r2_score(all_targets[:, i], all_predictions[:, i])
            
            metrics[target_name] = {
                'MSE': mse,
                'MAE': mae,
                'R^2': r2
            }
            
            # 绘制散点图
            plt.figure(figsize=(10, 5))
            plt.scatter(all_targets[:, i], all_predictions[:, i], alpha=0.5)
            plt.xlabel(f"True Values ({target_name})")
            plt.ylabel(f"Predictions ({target_name})")
            plt.title(f"Predictions vs True Values for {target_name}")
            
            # 添加对角线
            min_val = min(all_targets[:, i].min(), all_predictions[:, i].min())
            max_val = max(all_targets[:, i].max(), all_predictions[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
            
            # 添加统计信息
            plt.text(0.05, 0.95, 
                    f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', 
                    transform=plt.gca().transAxes,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            # plt.savefig(f'all_lh_test_predictions_{target_name}_brainimages_improved.png', dpi=300)
            plt.savefig(os.path.join(plots_dir, f'all_lh_test_predictions_{target_name}_brainimages_improved.png'), dpi=300)
            plt.close()

            # 保存预测结果和真实值
            results = {
                'predictions': all_predictions,
                'targets': all_targets,
                'metrics': metrics
            }
            np.save(os.path.join(plots_dir, 'lh_test_results_age_att.npy'), results)

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
        
        return metrics
    

    # 添加注意力图保存函数
    def save_attention_maps(model, test_loader, plots_dir):
        model.eval()
        with torch.no_grad():
            brain_maps, extra_features, targets = next(iter(test_loader))
            brain_maps = brain_maps.to(device)
            
            features = model.frontend(brain_maps)
            attention_weights = model.spatial_attention(features)
            attention_maps = attention_weights.cpu().numpy()
            
            plt.figure(figsize=(15, 5))
            for i in range(min(3, attention_maps.shape[0])):
                plt.subplot(1, 3, i+1)
                plt.imshow(attention_maps[i, 0], cmap='hot')
                plt.colorbar()
                plt.title(f"Subject {i+1} Attention Map\nAttention Score: {targets[i,0]:.3f}, Age: {targets[i,1]:.1f}")
            plt.suptitle("Spatial Attention Maps for Brain Regions", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'lh_attention_visualization.png'))
            plt.close()
            
            return attention_maps

    # # 最终评估
    # model.load_state_dict(torch.load('all_lh_best_model_improved.pth'))
    # model.eval()

    # 最终评估
    # model_path = os.path.join(results_manager.subdirs['models'], 'all_lh_best_model_improved.pth')
    model_path = os.path.join(results_manager.subdirs['models'], 'all_lh_best_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 1. 保存注意力图
    save_attention_maps(model, test_loader, results_manager.subdirs['attention']['analysis'])
    # 使用新的评估函数进行评估
    metrics = evaluate_model(model, test_loader, device, results_manager.subdirs['plots']['training'])

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # 打印评估结果
    print("\nFinal Evaluation Results:")
    for target_name, target_metrics in metrics.items():
        print(f"\nMetrics for {target_name}:")
        for metric_name, value in target_metrics.items():
            print(f"{metric_name}: {value:.4f}")

    # 打印最终测试损失
    test_loss, _, _ = validate_epoch(model, test_loader, criterion, device)
    print(f"\nFinal test loss: {test_loss:.4f}")

    # 3. 额外保存一些模型配置信息
    config_info = {
        'model_config': config,
        'timestamp': timestamp,
        'total_time': total_time,
        'best_val_loss': best_val_loss,
        'final_test_loss': test_loss
    }
    np.save(os.path.join(results_manager.subdirs['logs'], 'lh_training_config.npy'), config_info)

