import logging
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  


# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("training_detailed_brainimages.log"),
                        logging.StreamHandler()
                    ])

#  process the dataset
class BrainMapDataset(Dataset):
    def __init__(self, image_data, features_and_targets):
        self.image_data = image_data
        self.features_and_targets = features_and_targets

    def __len__(self):
        return len(self.features_and_targets)

    def __getitem__(self, idx):
        images = self.image_data[idx]
        images = [torch.from_numpy(img[None, :, :]).float() for img in images]  # 添加通道维度
        
        features_and_target = self.features_and_targets[idx]
        extra_features = features_and_target[2:5]
        targets = features_and_target[:2]

        return images, torch.tensor(extra_features).float(), torch.tensor(targets).float()


class BrainMapModel(nn.Module):
    def __init__(self):
        super(BrainMapModel, self).__init__()
        
        # 为每个输入图像创建一个卷积层序列
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 全连接层
        self._to_linear = None
        self._set_conv_output((1, 512, 512))
        
        self.fc1 = nn.Linear(self._to_linear * 4 + 3, 512)  # 4个图像通道的特征加上5个额外特征
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def _set_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output = self.conv_layers(input)
        n_size = output.data.view(bs, -1).size(1)
        self._to_linear = n_size

    def forward(self, brain_images, phenotypes):
        # brain_images 是一个列表，包含4个形状为 (batch_size, 1, 512, 512) 的张量
        batch_size = brain_images[0].size(0)
        processed_images = []
        
        for img in brain_images:
            x = self.conv_layers(img)  # 每个图像单独通过卷积层
            x = x.view(batch_size, -1)
            processed_images.append(x)
        
        # 合并处理后的图像特征
        x = torch.cat(processed_images, dim=1)
        
        # 合并 phenotype 数据
        x = torch.cat([x, phenotypes], dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# 假设brain_maps是形状为(1000, 769, 195, 4)的张量
# 假设features_and_targets是形状为(1000, 5)的张量

# 加载数据
loaded_phenotype_tensor = np.load('sample_phenotype_tensor_normalized.npy')

image_data = np.load('sample_cnn_lh_brainimages.npy')# shape: (1000, 4, 512, 512) 
# 分割数据
train_val_data, test_data, train_val_features, test_features = train_test_split(
    image_data, loaded_phenotype_tensor, test_size=0.1, random_state=42)

train_data, val_data, train_features, val_features = train_test_split(
    train_val_data, train_val_features, test_size=0.11111, random_state=42)

# 创建数据集
train_dataset = BrainMapDataset(train_data, train_features)
val_dataset = BrainMapDataset(val_data, val_features)
test_dataset = BrainMapDataset(test_data, test_features)

# 创建数据加载器
batch_size = 16  # 减小batch size以适应更大的图像
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 初始化模型、优化器和损失函数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainMapModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
criterion = nn.MSELoss()

# 学习率调度器
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

# 添加维度检查函数
def check_dimensions(train_loader):
    for brain_images, phenotypes, targets in train_loader:
        print("Brain images shapes:", [img.shape for img in brain_images])
        print("Phenotypes shape:", phenotypes.shape)
        print("Targets shape:", targets.shape)
        break

print("Checking initial data dimensions...")
check_dimensions(train_loader)

# 训练函数
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for idx, (brain_images, phenotypes, targets) in enumerate(loader):
        # 第一个batch的维度检查
        # if idx == 0:
        #     print("\nBatch dimensions:")
        #     print("Brain images:", [img.shape for img in brain_images])
        #     print("Phenotypes:", phenotypes.shape)
        #     print("Targets:", targets.shape)
        #     print("Model output shape:", model(
        #         [img.to(device) for img in brain_images], 
        #         phenotypes.to(device)
        #     ).shape)
            
        brain_images = [img.to(device) for img in brain_images]
        phenotypes = phenotypes.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(brain_images, phenotypes)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# 验证函数
def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for brain_images, phenotypes, targets in loader:
            brain_images = [img.to(device) for img in brain_images]
            phenotypes = phenotypes.to(device)
            targets = targets.to(device)
            
            outputs = model(brain_images, phenotypes)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    return total_loss / len(loader)


# 定义一个函数来记录参数范数
def log_norm_info(model, epoch):
    norm_logger.info(f"Epoch {epoch} - Parameter Norms:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            l1_norm = param.data.norm(1).item()
            l2_norm = param.data.norm(2).item()
            norm_logger.info(f"  {name}: L1 norm = {l1_norm:.4f}, L2 norm = {l2_norm:.4f}")


# 训练循环
num_epochs = 200
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 30
counter = 0

# 创建CSV文件
csv_file = 'training_loss_predictionM1_images.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

# 创建一个新的 logger 实例用于参数范数信息
norm_logger = logging.getLogger('norm_info')
norm_logger.setLevel(logging.INFO)

# 创建一个文件处理器，用于将参数范数信息写入单独的文件
norm_file_handler = logging.FileHandler("parameter_norms_prediction_images.log") # show L1 and L2 norm
norm_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
norm_file_handler.setFormatter(norm_formatter)

# 将文件处理器添加到 norm_logger
norm_logger.addHandler(norm_file_handler)

# 确保 norm_logger 不会将日志传播到根 logger!!!!
norm_logger.propagate = False

# training loop starts
start_time = time.time()

# 记录初始参数范数
norm_logger.info("Initial Parameter Norms:")
log_norm_info(model, 0)

# training loop
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate_epoch(model, val_loader, criterion, device)  # 修改这里
    epoch_time = time.time() - epoch_start_time
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    logging.info(f"Epoch {epoch+1}/{num_epochs}, "
                 f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                 f"Time: {epoch_time:.2f}s")
    
    if (epoch + 1) % 10 == 0:
        log_norm_info(model, epoch + 1)
    
    scheduler.step(val_loss)
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, val_loss])
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model_images.pth')
        counter = 0
        logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
    else:
        counter += 1
        if counter >= patience:
            logging.info("Early stopping")
            break

total_time = time.time() - start_time
logging.info(f"Training completed in {total_time:.2f} seconds")

norm_logger.info("Final Parameter Norms:")
log_norm_info(model, num_epochs)


# 定义评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for brain_maps, extra_features, targets in test_loader:
            brain_maps = [img.to(device) for img in brain_maps]
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
        plt.savefig(f'test_predictions_{target_name}_images.png', dpi=300)
        plt.close()
    
    return metrics


# 训练结束后的评估流程

# 1. 加载最佳模型
model.load_state_dict(torch.load('best_model_images.pth'))

# 2. 绘制学习曲线
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve_predict_M1_brainimages.png')
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
test_loss = validate_epoch(model, test_loader, criterion, device)
print(f"\nFinal test loss: {test_loss:.4f}")