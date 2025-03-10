"""
training_output_all_rh_YYYYMMDD_HHMMSS/
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
import re
import glob
import sys 
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import traceback
import signal
import copy
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

current_model = None
current_optimizer = None
current_fold = 0

def signal_handler(signum, frame):
    """处理中断信号"""
    if current_model is not None and current_optimizer is not None:
        logging.info(f"接收到中断信号，保存 fold {current_fold} 的检查点...")
        try:
            checkpoint_path = os.path.join(results_manager.subdirs['models'], f'checkpoint_fold_{current_fold}.pt')
            torch.save({
                'model_state_dict': current_model.state_dict(),
                'optimizer_state_dict': current_optimizer.state_dict(),
            }, checkpoint_path)
            logging.info("检查点保存成功")
        except Exception as e:
            logging.error(f"保存检查点时出错: {str(e)}")
    sys.exit(0)

# 只需要在文件开头注册一次信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Training configuration
config = {
    'batch_size': 4, #4
    'num_epochs': 200,
    'initial_lr': 1e-4,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 8, #8
    'num_phenotypes': 3,
    'patience': 20
    # 'warmup_epochs': 5
}

# BrainMapDataset class
class BrainMapDataset(Dataset):
    def __init__(self, image_data, features_and_targets, indices, mean=None, std=None, training=False):
        self.image_data = image_data
        self.features_and_targets = features_and_targets
        self.indices = indices  # 新增：存储索引
        self.mean = mean
        self.std = std
        self.training = training

    def __len__(self):
        # return len(self.features_and_targets)
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]  # 使用索引获取实际数据
        images = self.image_data[real_idx].astype(np.float32)

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
        features_and_target = self.features_and_targets[real_idx]
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
        if not hasattr(train_epoch, 'first_epoch_flag'):
            train_epoch.first_epoch_flag = True

        if train_epoch.first_epoch_flag and i == 0:  # 只为第一个批次打印形状信息
            print(f"\nBatch {i} shapes:")
            print(f"brain_images: {brain_images.shape}")
            print(f"phenotypes: {phenotypes.shape}")
            print(f"targets: {targets.shape}")
        
        if i % (num_batches // 10) == 0:
            logging.info(f"Processing batch {i}/{num_batches} ({(i/num_batches)*100:.1f}%)")
                

        # 只在必要时清理内存
        if i % 5 == 0:  # 每5个batch才清理一次
            torch.cuda.empty_cache()
            gc.collect()

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
        image_data, 
        phenotype_tensor,
        train_idx, 
        mean=mean,
        std=std,
        training=True  # 为训练集启用数据增强
    )
    val_dataset = BrainMapDataset(
        image_data, 
        phenotype_tensor,
        val_idx,
        mean=mean,
        std=std,
        training=False  # 验证集不需要启用数据增强
    )
    test_dataset = BrainMapDataset(
        image_data, 
        phenotype_tensor,
        test_idx,
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

def save_checkpoint(model, optimizer, epoch, train_history, fold_idx, results_manager):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history
    }
    checkpoint_path = os.path.join(results_manager.subdirs['models'], f'checkpoint_fold_{fold_idx}.pt')
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_history']

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
            # 修改这里，不要用嵌套字典
            'cross_validation': os.path.join(self.output_dir, 'cross_validation'),  # 改为直接路径
            'cross_validation_fold_results': os.path.join(self.output_dir, 'cross_validation/fold_results'),
            'cross_validation_metrics': os.path.join(self.output_dir, 'cross_validation/metrics'),
            'cross_validation_analysis': os.path.join(self.output_dir, 'cross_validation/analysis'),
            
            'error_analysis': {
                'raw_data': os.path.join(self.output_dir, 'error_analysis/raw_data'),
                'statistics': os.path.join(self.output_dir, 'error_analysis/statistics'),
                'detailed_analysis': os.path.join(self.output_dir, 'error_analysis/detailed_analysis'),
            },
            'logs': os.path.join(self.output_dir, 'logs'),
            'models': os.path.join(self.output_dir, 'models'),
            'plots': {
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
                # 'mean_r2': np.mean([fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]),
                # 'std_r2': np.std([fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]),
                # 'mean_mse': np.mean([fold['sum_att']['MSE'] for fold in cv_results['fold_metrics']]),
                # 'mean_mae': np.mean([fold['sum_att']['MAE'] for fold in cv_results['fold_metrics']])
                'mean_r2': np.mean([fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]),
                'std_r2': np.std([fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]),

                'mean_mse': np.mean([fold['sum_att']['MSE'] for fold in cv_results['fold_metrics']]),
                'std_mse': np.std([fold['sum_att']['MSE'] for fold in cv_results['fold_metrics']]),

                'mean_mae': np.mean([fold['sum_att']['MAE'] for fold in cv_results['fold_metrics']]),
                'std_mae': np.std([fold['sum_att']['MAE'] for fold in cv_results['fold_metrics']]),

                'mean_rmse': np.mean([fold['sum_att']['RMSE'] for fold in cv_results['fold_metrics']]),
                'std_rmse': np.std([fold['sum_att']['RMSE'] for fold in cv_results['fold_metrics']]),

                'mean_pearson': np.mean([fold['sum_att']['Pearson sum_att'] for fold in cv_results['fold_metrics']]),
                'std_pearson': np.std([fold['sum_att']['Pearson sum_att'] for fold in cv_results['fold_metrics']]),

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
        image_data,
        phenotype_data,
        train_idx, 
        mean=mean,
        std=std,
        training=True
    )
    
    val_dataset = BrainMapDataset(
        image_data,
        phenotype_data,
        val_idx, 
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
        prefetch_factor=1,  # 减小预取因子
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=1,  # 减小预取因子
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

def train_fold(model, train_loader, val_loader, optimizer, scheduler, criterion, device, config, fold_idx, results_manager):
    """训练单个fold"""
    global current_model, current_optimizer, current_fold
    current_model = model
    current_optimizer = optimizer
    current_fold = fold_idx

    # save_frequency = 10  # 每5个epoch保存一次
    best_model_path = os.path.join(results_manager.subdirs['models'], f'best_model_fold_{fold_idx}.pt')

    checkpoint_path = os.path.join(results_manager.subdirs['models'], f'checkpoint_fold_{fold_idx}.pt')
    if os.path.exists(checkpoint_path):
        start_epoch, history = load_checkpoint(model, optimizer, checkpoint_path)
    else:
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_att_losses': [],
            'train_age_losses': [],
            'learning_rates': []
        }
        
    best_val_loss = float('inf')
    patience = config['patience'] 
    counter = 0
    
    signal_handler.current_fold = fold_idx

    for epoch in range(config['num_epochs']):
        signal_handler.current_epoch = epoch
        epoch_start_time = time.time()
        torch.cuda.empty_cache() 

        if epoch % 2 == 0:  # 每两个epoch清理一次内存
            torch.cuda.empty_cache()
            gc.collect()

        # 保存检查点
        if epoch % 10 == 0:  # 每10个epoch保存一次
            results_manager.save_model(
                model, 
                f"checkpoint_fold_{fold_idx}_epoch_{epoch}",
                fold_idx
            )

        # Training
        (train_loss, train_att_loss, train_age_loss), epoch_lrs = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            config['gradient_accumulation_steps']
        )

        # 打印当前学习率
        # current_lr = optimizer.param_groups[0]['lr']
        # logging.info(f"Current learning rate: {current_lr:.2e}")

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
            torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                    f"Train Loss: {train_loss:.4f} (Attention: {train_att_loss:.4f}, Age: {train_age_loss:.4f}), "
                    f"Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return history

def get_slurm_time_limit():
    """获取SLURM作业的时间限制"""
    time_limit = os.getenv('SLURM_TIMELIMIT')
    if time_limit:
        # 转换SLURM时间格式为秒
        if '-' in time_limit:
            days, time = time_limit.split('-')
        else:
            days, time = '0', time_limit
        hours, mins, secs = time.split(':')
        return int(days) * 86400 + int(hours) * 3600 + int(mins) * 60 + int(secs)
    return None

def perform_cross_validation(image_data, phenotype_data, model_template, device, config, k_folds=5):
    """执行k折交叉验证"""
    start_time = time.time()
    time_limit = get_slurm_time_limit()
    if time_limit:
        safe_margin = 1800  # 30分钟安全边际
        time_limit -= safe_margin

    try:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_results = {
            'fold_metrics': [],
            'predictions': [],
            'targets': [],
            'fold_train_history': []
        }
        
        # 检查是否有未完成的fold
        completed_folds = []
        for fold in range(k_folds):
            signal_handler.current_fold = fold
            if os.path.exists(os.path.join(results_manager.subdirs['cross_validation_fold_results'], f'fold_{fold}_results.npy')):
                completed_folds.append(fold)
                # 加载已完成的fold结果
                fold_results = np.load(os.path.join(results_manager.subdirs['cross_validation_fold_results'], f'fold_{fold}_results.npy'), 
                                     allow_pickle=True).item()
                cv_results['fold_metrics'].append(fold_results['metrics'])
                cv_results['predictions'].append(fold_results['predictions'])
                cv_results['targets'].append(fold_results['targets'])
                cv_results['fold_train_history'].append(fold_results['train_history'])


        for fold, (train_idx, val_idx) in enumerate(kf.split(image_data)):
            if time_limit and (time.time() - start_time) > time_limit:
                logging.info("Approaching SLURM time limit, saving progress...")
                break

            if fold in completed_folds:
                logging.info(f"Fold {fold} already completed, skipping...")
                continue

            logging.info(f"\nStarting fold {fold+1}/{k_folds}")
            # logging.info(f"Train set size: {len(train_idx)}, Val set size: {len(val_idx)}")
            try:
                # 创建该fold的数据加载器
                train_loader, val_loader = create_fold_loaders(
                    image_data, phenotype_data, train_idx, val_idx,
                    batch_size=config['batch_size']
                )
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                # 创建新的模型实例
                model = copy.deepcopy(model_template).to(device)

                # 在这里计算steps_per_epoch
                steps_per_epoch = len(train_loader) // config['gradient_accumulation_steps']
                
                # 设置优化器和调度器
                optimizer, scheduler, criterion = get_training_components(
                    model, config, steps_per_epoch=steps_per_epoch)
                
                # 训练该fold
                fold_history = train_fold(
                    model, train_loader, val_loader, optimizer,
                    scheduler, criterion, device, config, fold, results_manager
                )
                
                results_manager.save_training_history(
                fold_history,
                prefix=f'fold_{fold}_'
            )
                
                # 在验证集上评估
                metrics = evaluate_model(model, val_loader, device, None)  # 不保存图表
                
                
                # 获取预测结果
                predictions, targets = get_predictions(model, val_loader, device)
                # 保存当前fold的结果
                fold_results = {
                    'metrics': metrics,
                    'predictions': predictions,
                    'targets': targets,
                    'train_history': fold_history
                }
                np.save(os.path.join(results_manager.subdirs['cross_validation_fold_results'], f'fold_{fold}_results.npy'), 
                    fold_results)
                
                cv_results['fold_metrics'].append(metrics)
                cv_results['predictions'].append(predictions)
                cv_results['targets'].append(targets)
                cv_results['fold_train_history'].append(fold_history)

                # 清理内存
                del model, optimizer, scheduler
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Error in fold {fold+1}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

            # 每个fold完成后记录内存使用情况
            logging.info(f"GPU memory after fold {fold+1}: "
                        f"{torch.cuda.memory_allocated()/1024**2:.1f}MB allocated, "
                        f"{torch.cuda.memory_reserved()/1024**2:.1f}MB reserved")
            
        # 清理内存
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Critical error in cross-validation: {str(e)}")
        logging.error(traceback.format_exc())
        raise

    return cv_results

def get_latest_fold_and_epoch(results_manager):
    """获取最新的fold和epoch"""
    models_dir = results_manager.subdirs['models']
    checkpoints = glob.glob(os.path.join(models_dir, 'checkpoint_fold_*.pt'))
    if not checkpoints:
        return 0, 0
        
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint)
    fold = int(re.search(r'fold_(\d+)', latest_checkpoint).group(1))
    epoch = checkpoint['epoch']
    return fold, epoch


# main
if __name__ == "__main__":

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
            rmse = np.sqrt(mse)  # 计算 RMSE
            mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
            r2 = r2_score(all_targets[:, i], all_predictions[:, i])

            # 计算 Pearson 相关系数（分别计算 Attention 和 Age 的相关性）
            pearson, p_value = pearsonr(all_targets[:, i], all_predictions[:, i])
            
            metrics[target_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R^2': r2,
                f'Pearson {target_name}': pearson,
                f'Pearson {target_name} p_value': p_value
            }
        
        return metrics
    

    # 创建输出文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'cross_validation_rh_{timestamp}'
    results_manager = ResultsManager(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
        
    # 检查是否需要恢复训练
    latest_fold, latest_epoch = get_latest_fold_and_epoch(results_manager)
    if latest_fold > 0 or latest_epoch > 0:
        logging.info(f"Resuming from fold {latest_fold}, epoch {latest_epoch}")

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
                        logging.FileHandler(os.path.join(results_manager.subdirs['logs'], "all_rh_training.log")),
                        logging.StreamHandler()
                    ])
    logging.info("Setting up signal handlers...")
    
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
    image_path = '/projects/0/einf1049/scratch/jouyang/all_cnn_rh_brainimages.npy'
    # image_path = '/home/jouyang1/sample_cnn_rh_brainimages.npy'
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

    logging.info(f"Loaded data shape: {image_data.shape}")
    # 在交叉验证开始前打印配置
    logging.info(f"Starting cross-validation with config: {config}")

    # 执行交叉验证
    logging.info(f"\nStarting cross-validation...")

    # 创建模型模板
    model_template = BrainADHDModel(config['num_phenotypes']).to(device)

    # optimizer, scheduler, criterion = get_training_components(model_template, config, steps_per_epoch=train_steps_per_epoch)
    torch.cuda.empty_cache()
    gc.collect()


    # 执行交叉验证
    cv_results = perform_cross_validation(
        image_data, 
        loaded_phenotype_tensor,
        model_template,
        device,
        config,
        k_folds=5
    )

    # 保存交叉验证结果
    results_manager.save_cv_results(cv_results)

    # 分析和可视化交叉验证结果
    plt.figure(figsize=(15, 5))

    # 1. 损失曲线
    plt.subplot(1, 3, 1)
    for fold in range(len(cv_results['fold_train_history'])):
        plt.plot(cv_results['fold_train_history'][fold]['val_losses'], 
                label=f'Fold {fold+1}')
    plt.title('Validation Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 2. R2分数分布
    plt.subplot(1, 3, 2)
    att_r2_scores = [fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]
    age_r2_scores = [fold['age']['R^2'] for fold in cv_results['fold_metrics']]
    plt.boxplot([att_r2_scores, age_r2_scores], labels=['Attention', 'Age'])
    plt.title('R² Score Distribution Across Folds')

    # 3. MSE分布
    plt.subplot(1, 3, 3)
    att_mse_scores = [fold['sum_att']['MSE'] for fold in cv_results['fold_metrics']]
    age_mse_scores = [fold['age']['MSE'] for fold in cv_results['fold_metrics']]
    plt.boxplot([att_mse_scores, age_mse_scores], labels=['Attention', 'Age'])
    plt.title('MSE Distribution Across Folds')

    plt.tight_layout()
    plt.savefig(os.path.join(results_manager.subdirs['cross_validation']['analysis'], 'cross_validation_analysis.png'))
    plt.close()

    # 打印平均指标
    logging.info(f"\nCross-validation Results:")
    metrics_names = ['MSE', 'MAE', 'R^2']
    tasks = ['sum_att', 'age']

    for task in tasks:
        logging.info(f"\n{task.upper()} Metrics:")
        for metric in metrics_names:
            values = [fold[task][metric] for fold in cv_results['fold_metrics']]
            mean_value = np.mean(values)
            std_value = np.std(values)
            logging.info(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")
    
    logging.info(f"\nCross-validation finished.")
    
    # 清理内存
    del image_data, loaded_phenotype_tensor
    gc.collect()
    torch.cuda.empty_cache()

