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
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# setting the logging data
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("sample_lh_training_predict_att_age.log"),
                        logging.StreamHandler()
                    ])

class BrainMapDataset(Dataset):
    def __init__(self, image_data, features_and_targets):
        self.image_data = image_data
        self.features_and_targets = features_and_targets

    def __len__(self):
        return len(self.features_and_targets)

    def __getitem__(self, idx):
        # 保持数据的原始处理方式
        images = self.image_data[idx].astype(np.float32)
        images = torch.from_numpy(images).float()
        
        features_and_target = self.features_and_targets[idx]
        
        # 确保目标变量的正确对应
        targets = np.array([
            features_and_target[0],  # attention_scores
            features_and_target[2]   # age
        ]).astype(np.float32)

        # 表型特征
        extra_features = np.array([
            features_and_target[1],  # aggressive_behaviour_scores 
            features_and_target[3],  # sex
            features_and_target[4]   # maternal_edu_level
        ]).astype(np.float32)

        return images, torch.tensor(extra_features).float(), torch.tensor(targets).float()

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(),
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, 3, padding=1),
                    nn.Dropout(0.2)
                )
            )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.mlp(self.max_pool(x).view(x.size(0), -1))
        out = torch.sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * out
    

class EnhancedBrainADHDModel(nn.Module):
    def __init__(self, num_phenotypes):
        super(EnhancedBrainADHDModel, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks with transition layers
        self.dense1 = DenseBlock(64, growth_rate=16, num_layers=4)
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(64 + 4 * 16),
            nn.Conv2d(64 + 4 * 16, 128, 1),
            nn.AvgPool2d(2)
        )
        
        self.dense2 = DenseBlock(128, growth_rate=16, num_layers=8)
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(128 + 8 * 16),
            nn.Conv2d(128 + 8 * 16, 256, 1),
            nn.AvgPool2d(2)
        )
        
        self.dense3 = DenseBlock(256, growth_rate=16, num_layers=16)
        
        # Channel attention
        self.channel_attention = ChannelAttention(256 + 16 * 16)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Phenotype attention
        self.phenotype_attention = nn.Sequential(
            nn.Linear(num_phenotypes, 64),
            nn.ReLU(),
            nn.Linear(64, num_phenotypes),
            nn.Sigmoid()
        )
        
        # Feature fusion with residual connection
        total_features = (256 + 16 * 16) + num_phenotypes
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Prediction heads with shared features
        self.shared_features = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.attention_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        self.behavior_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
    def forward(self, brain_images, phenotypes):
        # CNN pathway with dense connections
        x = self.conv1(brain_images)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        
        # Apply channel attention
        x = self.channel_attention(x)
        
        # Global pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Phenotype attention
        phenotype_weights = self.phenotype_attention(phenotypes)
        weighted_phenotypes = phenotypes * phenotype_weights
        
        # Feature fusion
        combined = torch.cat([x, weighted_phenotypes], dim=1)
        fused = self.fusion(combined)
        
        # Shared feature extraction
        shared = self.shared_features(fused)
        
        # Task-specific predictions
        attention = self.attention_head(shared)
        behavior = self.behavior_head(shared)
        
        return torch.cat([attention, behavior], dim=1)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# train and validate loop functions
def train_epoch(model, loader, optimizer, criterion, device, gradient_accumulation_steps):
    model.train()
    total_loss = 0
    total_att_loss = 0
    total_age_loss = 0
    step_learning_rates = []
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

        attention_loss = criterion(outputs[:, 0], targets[:, 0])  # attention score
        age_loss = criterion(outputs[:, 1], targets[:, 1])       # age
        # attention_weight = 1.0 - current_attention_r2
        # age_weight = 1.0 - current_age_r2
        # total_weight = attention_weight + age_weight
        # attention_weight = attention_weight / total_weight
        # age_weight = age_weight / total_weight
        # loss = attention_weight * attention_loss + age_weight * age_loss
        # loss = 0.5 * attention_loss + 0.5 * age_loss 
        # 动态权重
        # attention_var = torch.var(outputs[:, 0] - targets[:, 0])
        # age_var = torch.var(outputs[:, 1] - targets[:, 1])
        # att_weight = 1 / attention_var.item()
        # age_weight = 1 / age_var.item()
        att_weight = 0.7  # attention_score预测效果差，给更大权重
        age_weight = 0.3
        loss = (att_weight * attention_loss + age_weight * age_loss) / gradient_accumulation_steps


        # Normalize loss
        loss = loss / gradient_accumulation_steps

        # Backward pass with gradient accumulation
        loss.backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step() #OneCycleLR是按步更新的，所以在这里更新
            optimizer.zero_grad()
            step_learning_rates.append(optimizer.param_groups[0]['lr'])
        
        total_loss += loss.item() * gradient_accumulation_steps
        total_att_loss += attention_loss.item()
        total_age_loss += age_loss.item()

        # Clear some memory
        del brain_images, phenotypes, targets, outputs
        torch.cuda.empty_cache()

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

def get_optimizer_and_scheduler(model, num_epochs, train_steps_per_epoch):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=num_epochs,
        steps_per_epoch=train_steps_per_epoch,
        pct_start=0.2, # 更快进入降低学习率阶段
        anneal_strategy='cos',
        div_factor=10.0,  # 初始学习率将是max_lr的1/10
        final_div_factor=1000.0  # 最终学习率将是max_lr的1/1000
    )
    
    return optimizer, scheduler

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

# main
if __name__ == "__main__":

    # 创建输出文件夹
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'sample_training_lh_att_age{timestamp}'
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
    indices = np.arange(len(image_data))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.11111, random_state=42)

    # Create datasets using indices
    train_dataset = BrainMapDataset(image_data[train_idx], loaded_phenotype_tensor[train_idx])
    val_dataset = BrainMapDataset(image_data[val_idx], loaded_phenotype_tensor[val_idx])
    test_dataset = BrainMapDataset(image_data[test_idx], loaded_phenotype_tensor[test_idx])
    
    # Reduce batch size and use appropriate num_workers
    batch_size = 32  # Reduced from 16
    num_workers = 2  # Adjust based on your system
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                       num_workers=num_workers, pin_memory=True)
    

    # initialize model, optimizer, and loss functions
    num_phenotypes = 3  # (12595,5) only for age, sex(0/1), edu_maternal(0/1/2)
    model = EnhancedBrainADHDModel(num_phenotypes).to(device)
    model.apply(init_weights) 

    # Calculate steps per epoch
    train_steps_per_epoch = len(train_loader)

    # training
    num_epochs = 200
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    train_losses = []
    val_losses = []
    patience = 20
    counter = 0
    gradient_accumulation_steps = 2  # Accumulate gradients over 2 batches
    
    optimizer,scheduler = get_optimizer_and_scheduler(model, num_epochs, train_steps_per_epoch)
    criterion = nn.MSELoss()


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
        (train_loss, train_att_loss, train_age_loss), epoch_lrs = train_epoch(model, train_loader, optimizer, criterion, device, gradient_accumulation_steps)
        all_learning_rates.extend(epoch_lrs)

        # Validation
        val_loss, predictions, targets = validate_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start_time
        # 计算R²
        attention_r2 = r2_score(targets[:, 0], predictions[:, 0])
        age_r2 = r2_score(targets[:, 1], predictions[:, 1])
        current_r2 = (attention_r2 + age_r2) / 2
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # logging.info(f"(sample, lh) Epoch {epoch+1}/{num_epochs}, "
        #         f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        #         f"Time: {epoch_time:.2f}s")
        # 修改后的logging语句，使用正确的元组解包
        logging.info(f"Epoch {epoch+1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f} (Attention: {train_att_loss:.4f}, Age: {train_age_loss:.4f}), "
                f"Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Scheduler step，（OneCycleLR是按步更新的，所以这里不需要scheduler.step()）
        # scheduler.step(val_loss)

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss])
        
        if val_loss < best_val_loss and current_r2 > best_val_r2:
            best_val_loss = val_loss
            best_val_r2 = current_r2
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

