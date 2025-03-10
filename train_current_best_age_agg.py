"""
training_output_all_lh_YYYYMMDD_HHMMSS/
├── logs/
│   ├── all_lh_training_detailed_brainimages_improved.log
│   └── all_lh_training_loss_predictionM1_images_improved.csv
├── models/
│   └── all_lh_best_model_improved.pth
└── plots/
    ├── all_lh_learning_curve_predict_brainimages_improved.png
    ├── all_lh_test_predictions_sum_att_brainimages_improved.png
    └── all_lh_test_predictions_sum_agg_brainimages_improved.png

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
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# 
# setting the logging data
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("all_lh_training_detailed_brainimages_improved.log"),
                        logging.StreamHandler()
                    ])

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
        # targets: aggressive_behaviour_scores (index 1) 和 age (index 2)
        targets = np.array([
            features_and_target[1],  # aggressive_behaviour_scores
            features_and_target[2]   # age
        ]).astype(np.float32)

        # extra_features(as input): attention_scores (index 0) 和 sex (index 3) 和 maternal_edu_level (index 4)
        extra_features = np.array([
            features_and_target[0],  # attention_scores
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
        resnet = models.resnet50(pretrained=True)

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
config = {
    'batch_size': 16,
    'num_epochs': 300,
    'initial_lr': 1e-4,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 2
    # 'warmup_epochs': 5
}


# Loss function
# class CombinedLoss(nn.Module):
#     def __init__(self, alpha=0.6):
#         super().__init__()
#         self.alpha = alpha
#         self.mse = nn.MSELoss()
#         self.smooth_l1 = nn.SmoothL1Loss()

#     def forward(self, outputs, targets):
#         mse_loss = self.mse(outputs, targets)
#         smooth_l1_loss = self.smooth_l1(outputs, targets)
#         return self.alpha * mse_loss + (1 - self.alpha) * smooth_l1_loss
    

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
        del brain_images, phenotypes, targets, outputs
        torch.cuda.empty_cache()

        # 打印进度
        if i % 50 == 0:
            print(f'Batch [{i}/{num_batches}], Loss: {loss.item():.4f}')

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
            del brain_images, phenotypes, targets, outputs
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
def get_data_loaders(image_data, phenotype_tensor, batch_size, num_workers=4):
    # 计算均值和标准差
    mean = np.mean(image_data, axis=(0, 2, 3)) # image shape: (N, C, H, W)
    std = np.std(image_data, axis=(0, 2, 3))
    
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
        pin_memory=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# main
if __name__ == "__main__":

    # 创建输出文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'training_all_data_lh_mirror_aggre{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建子文件夹
    plots_dir = os.path.join(output_dir, 'plots')
    logs_dir = os.path.join(output_dir, 'logs')
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 修改日志配置
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(logs_dir, "all_lh_training_detailed_brainimages_improved.log")),
                        logging.StreamHandler()
                    ])
    
    # training
    num_epochs = 200
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 20
    counter = 0
    # gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches
    
    
    # Set memory efficient device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # May help with speed?? not sure...

    # 数据路径
    # image_path = '/projects/0/einf1049/scratch/jouyang/all_cnn_lh_brainimages.npy'
    image_path = '/home/jouyang1/sample_cnn_lh_brainimages.npy'
    # phenotype_path = '/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes.npy'
    # phenotype_path = '/home/jouyang1/all_normalised_phenotypes_correct.npy'
    phenotype_path = '/home/jouyang1/sample_normalised_phenotype.npy'
    # order: sum_att  sum_agg	age	  sex(0/1)	edu_maternal(0/1/2) 
    
    # load data
    image_data, loaded_phenotype_tensor = load_data(
        image_path, 
        phenotype_path,
        use_mmap=True  # 启用内存映射
    )

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

    train_loader, val_loader, test_loader = get_data_loaders(image_data, loaded_phenotype_tensor, batch_size=config['batch_size'], num_workers=4)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
    #                         num_workers=num_workers, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, 
    #                       num_workers=num_workers, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, 
    #                        num_workers=num_workers, pin_memory=True)
    

    # initialize model, optimizer, and loss functions
    num_phenotypes = 3  # (12595,5) only for age, sex(0/1), edu_maternal(0/1/2)
    model = BrainADHDModel(num_phenotypes).to(device)
    train_steps_per_epoch = len(train_loader)

    optimizer, scheduler, criterion = get_training_components(model, config, steps_per_epoch=train_steps_per_epoch)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    # # optimizer, scheduler = get_optimizer_and_scheduler(model, num_epochs, train_steps_per_epoch)
    # criterion = nn.MSELoss()

    # # learning rate scheduler 
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, min_lr=1e-6, verbose=True)
    

    # save the training logs
    # csv_file = 'all_lh_training_loss_predictionM1_images_improved.csv'
    csv_file = os.path.join(logs_dir, 'all_lh_training_loss_predictionM1_images_improved.csv')

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

    # training starts
    start_time = time.time()
    learning_rates = []
    all_learning_rates = []

    for epoch in range(num_epochs):
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
        
        logging.info(f"Epoch {epoch+1}/{num_epochs}, "
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
            model_path = os.path.join(models_dir, 'all_lh_best_model_improved.pth')
            torch.save(model.state_dict(), model_path)
            counter = 0
            logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            counter += 1
        if counter >= patience:
            logging.info("Early stopping")
            break

        # Clear memory
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
    plt.savefig(os.path.join(plots_dir, 'step_detailed_learning_rate_schedule.png'))
    plt.close()

    # 训练结束后绘制学习率曲线
    plt.figure(figsize=(10,5))
    plt.plot(learning_rates)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Training Steps')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'learning_rate_schedule.png'))
    plt.close()

    # plot learning rate curves
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs (Improved Model for left semi brain images)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # plt.savefig('all_lh_learning_curve_predict_brainimages_improved.png')
    plt.savefig(os.path.join(plots_dir, 'all_lh_learning_curve_predict_brainimages_improved.png'))
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
                'R2': r2
            }
            
            # 绘制散点图
            plt.figure(figsize=(10, 5))
            plt.scatter(all_targets[:, i], all_predictions[:, i], alpha=0.5)
            plt.xlabel(f"True Values ({target_name})")
            plt.ylabel(f"Predictions ({target_name})")
            plt.title(f"Predictions vs True Values for {target_name} (left semi brain)")
            
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

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
        
        return metrics

    # # 最终评估
    # model.load_state_dict(torch.load('all_lh_best_model_improved.pth'))
    # model.eval()
    # 最终评估
    model_path = os.path.join(models_dir, 'all_lh_best_model_improved.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 使用新的评估函数进行评估
    metrics = evaluate_model(model, test_loader, device, plots_dir)

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

