import logging
import csv
import torch
#print(torch.__version__)
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("training_detailed.log"),
                        logging.StreamHandler()
                    ])

#  process the dataset
class BrainMapDataset(Dataset):
    def __init__(self, brain_maps, features_and_targets):
        self.brain_maps = brain_maps
        self.features_and_targets = features_and_targets

    def __len__(self):
        return len(self.brain_maps)

    def __getitem__(self, idx):
        brain_map = self.brain_maps[idx]
        features_and_target = self.features_and_targets[idx]
        
        extra_features = features_and_target[2:5]
        targets = features_and_target[:2]

        return brain_map, extra_features, targets

# 假设brain_maps是形状为(1000, 769, 195, 4)的张量
# 假设features_and_targets是形状为(1000, 5)的张量


# load the data
loaded_tensor = np.load('sample_input_tensor_quantile_transform.npy')
loaded_phenotype_tensor = np.load('sample_phenotype_tensor_normalized.npy')
# order: sum_att  sum_agg	age	  sex(0/1)	edu_maternal(0/1/2)

brain_maps = loaded_tensor  # 形状为 (N, 769, 195, 4) 的张量
features_and_targets = loaded_phenotype_tensor


# define the model
class BrainMapModel(nn.Module):
    def __init__(self):
        super(BrainMapModel, self).__init__()
        
        # 3D convolutional layer
        self.conv3d = nn.Conv3d(1, 32, kernel_size=(3, 3, 4), padding=(1, 1, 0))
        
        # 2D convolutional layer
        self.conv2d = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # pooling 
        self.pool = nn.MaxPool2d(4)
        
        # Adaptive pooling to ensure fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((12, 12))  # Fixed output size
        
        # Calculate the flattened size
        self._to_linear = 128 * 12 * 12  # 18432
        
        # Fully connected layers with corrected dimensions - note the +3 for three extra features
        self.fc1 = nn.Linear(self._to_linear + 3, 256)  # +3 for age, sex, edu_maternal
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # Output 2 values for sum_att and sum_agg
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
     
    def forward_conv(self, x):
        # Add debugging print statements
        # print(f"Initial input shape: {x.shape}")
        
        x = x.unsqueeze(1)  # Add channel dimension
        # print(f"After unsqueeze shape: {x.shape}")
        
        x = F.relu(self.conv3d(x))
        # print(f"After conv3d shape: {x.shape}")
        
        x = x.squeeze(-1)  # Remove last dimension
        # print(f"After squeeze shape: {x.shape}")
        
        x = self.pool(F.relu(self.conv2d(x)))
        # print(f"After first conv2d and pool shape: {x.shape}")
        
        x = self.pool(F.relu(self.conv2d_2(x)))
        # print(f"After second conv2d and pool shape: {x.shape}")
        
        x = self.adaptive_pool(x)
        # print(f"After adaptive pool shape: {x.shape}")
        
        return x

    def forward(self, x, extra_features):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        # print(f"After flatten shape: {x.shape}")
        # print(f"Extra features shape: {extra_features.shape}")

        # 确保extra_features的维度正确
        assert extra_features.shape[1] == 3, f"Expected 3 extra features, got {extra_features.shape[1]}"
        
        x = torch.cat([x, extra_features], dim=1)  # Concatenate with all 3 extra features
        # print(f"After concatenation shape: {x.shape}")
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 分割数据：train : validate : test = 8 : 1 : 1
train_val_maps, test_maps, train_val_features, test_features = train_test_split(
    brain_maps, features_and_targets, test_size=0.1, random_state=42)

train_maps, val_maps, train_features, val_features = train_test_split(
    train_val_maps, train_val_features, test_size=0.15, random_state=42)  # 0.11111 of 90% is 10% of total

# 创建数据集
train_dataset = BrainMapDataset(train_maps, train_features)
val_dataset = BrainMapDataset(val_maps, val_features)
test_dataset = BrainMapDataset(test_maps, test_features)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 初始化模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainMapModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6) # weight_decay is L2 normalization
criterion = nn.MSELoss()

# 学习率调度器
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
# 训练函数
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for i, (brain_maps, extra_features, targets) in enumerate(loader):
        if i == 0:  # 只在第一个批次打印形状信息
            print(f"----This is the M1 model training----")
            print(f"Brain maps shape: {brain_maps.shape}")
            print(f"Extra features shape: {extra_features.shape}")
            print(f"Targets shape: {targets.shape}")
        
        brain_maps = brain_maps.to(device)
        extra_features = extra_features.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(brain_maps, extra_features)
        
        # 计算 L1 正则化项
        l1_lambda = 0.0001
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = criterion(outputs, targets) + l1_lambda * l1_norm
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 验证函数
def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for brain_maps, extra_features, targets in loader:
            brain_maps = brain_maps.to(device)
            extra_features = extra_features.to(device)
            targets = targets.to(device)
            
            outputs = model(brain_maps, extra_features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


# 首先定义评估函数
def evaluate_model(model, test_loader, device):
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
    for i, target_name in enumerate(['sum_att', 'sum_agg']):
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
        plt.savefig(f'test_predictions_{target_name}_M1.png', dpi=300)
        plt.close()
    
    return metrics


# 训练循环
num_epochs = 200
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 30
counter = 0

# 创建CSV文件
csv_file = 'training_loss_predictionM1.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

# 创建一个新的 logger 实例用于参数范数信息
norm_logger = logging.getLogger('norm_info')
norm_logger.setLevel(logging.INFO)

# 创建一个文件处理器，用于将参数范数信息写入单独的文件
norm_file_handler = logging.FileHandler("parameter_norms_prediction.log") # show L1 and L2 norm
norm_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
norm_file_handler.setFormatter(norm_formatter)

# 将文件处理器添加到 norm_logger
norm_logger.addHandler(norm_file_handler)

# 确保 norm_logger 不会将日志传播到根 logger!!!!
norm_logger.propagate = False

# 定义一个函数来记录参数范数
def log_norm_info(model, epoch):
    norm_logger.info(f"Epoch {epoch} - Parameter Norms:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            l1_norm = param.data.norm(1).item()
            l2_norm = param.data.norm(2).item()
            norm_logger.info(f"  {name}: L1 norm = {l1_norm:.4f}, L2 norm = {l2_norm:.4f}")



# training loop starts
start_time = time.time()

# 记录初始参数范数
norm_logger.info("Initial Parameter Norms:")
log_norm_info(model, 0)

# training loop
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate_epoch(model, val_loader, criterion)
    epoch_time = time.time() - epoch_start_time
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    logging.info(f"Epoch {epoch+1}/{num_epochs}, "
                 f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                 f"Time: {epoch_time:.2f}s")
    
    # 每隔一定数量的 epoch 记录参数范数
    if (epoch + 1) % 10 == 0:  # 例如，每 10 个 epoch
        log_norm_info(model, epoch + 1)
    
    # 更新学习率
    scheduler.step(val_loss)
    # 写入CSV文件
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, val_loss])
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
        logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
    else:
        counter += 1
        if counter >= patience:
            logging.info("Early stopping")
            break

total_time = time.time() - start_time
logging.info(f"Training completed in {total_time:.2f} seconds")

# 训练结束后记录最终的范数信息
norm_logger.info("Final Parameter Norms:")
log_norm_info(model, num_epochs)



# 训练结束后的评估流程

# 1. 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 2. 绘制学习曲线
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve_predict_M1.png')
plt.close()

# 3. 使用 evaluate_model 进行评估
metrics = evaluate_model(model, test_loader, device)

# 4. 打印评估结果
print("\nFinal Evaluation Results:")
for target_name, target_metrics in metrics.items():
    print(f"\nMetrics for {target_name}:")
    for metric_name, value in target_metrics.items():
        print(f"{metric_name}: {value:.4f}")

# 5. 打印最终测试损失
test_loss = validate_epoch(model, test_loader, criterion)
print(f"\nFinal test loss: {test_loss:.4f}")