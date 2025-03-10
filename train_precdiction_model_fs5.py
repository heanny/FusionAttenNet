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

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("training_detailed_fs5.log"),
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
        
        extra_features = features_and_target[:3]
        targets = features_and_target[3:]

        return brain_map, extra_features, targets

# 假设brain_maps是形状为(1000, 569, 18, 4)的张量
# 假设features_and_targets是形状为(1000, 5)的张量


# load the data
loaded_tensor = np.load('sample_input_tensor_fs5.npy')
loaded_phenotype_tensor = np.load('sample_phenotype_tensor.npy')
brain_maps = loaded_tensor  # 形状为 (N, 769, 195, 4) 的张量
features_and_targets = loaded_phenotype_tensor

# 分割数据
train_val_maps, test_maps, train_val_features, test_features = train_test_split(
    brain_maps, features_and_targets, test_size=0.1, random_state=42)

train_maps, val_maps, train_features, val_features = train_test_split(
    train_val_maps, train_val_features, test_size=0.11111, random_state=42)  # 0.11111 of 90% is 10% of total

# 创建数据集
train_dataset = BrainMapDataset(train_maps, train_features)
val_dataset = BrainMapDataset(val_maps, val_features)
test_dataset = BrainMapDataset(test_maps, test_features)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


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
        self.pool = nn.MaxPool2d(2)
        
        # calculating the number of input features of fully connected layer 计算全连接层的输入特征数
        self._to_linear = None
        self._initialize_size()
        
        # fully connected layer 
        self.fc1 = nn.Linear(self._to_linear + 3, 256)  # +3 is for the first three pheonotypes (age, sex, edu_level_parental) features
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # output 2 predicted values for attention scores and aggresive behavior scores
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def _initialize_size(self):
        # use an example input to calculate the number of flattened features
        sample_input = torch.zeros(1, 569, 18, 4)
        self.forward_conv(sample_input)
        
    def forward_conv(self, x):
        x = x.unsqueeze(1)  # add the channel dimension: (#batch, 1, 769, 195, 4)
        x = F.relu(self.conv3d(x))
        x = x.squeeze(-1)  # remove the last dimension: (#batch, 32, 767, 193)
        x = self.pool(F.relu(self.conv2d(x)))
        x = self.pool(F.relu(self.conv2d_2(x)))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x, extra_features):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.cat([x, extra_features], dim=1)  # connected to the 3 extra pheonotypes features
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
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6) # weight_decay is L2 normalization
criterion = nn.MSELoss()

# 学习率调度器
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
# 训练函数
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for brain_maps, extra_features, targets in loader:
        brain_maps, extra_features, targets = brain_maps.to(device), extra_features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(brain_maps, extra_features)
         # 计算 L1 正则化项
        l1_lambda = 0.0001  # 可以调整这个值
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        # 将 L1 正则化添加到损失中
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
            brain_maps, extra_features, targets = brain_maps.to(device), extra_features.to(device), targets.to(device)
            outputs = model(brain_maps, extra_features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

# 训练循环
num_epochs = 200
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 30
counter = 0

# 创建CSV文件
csv_file = 'training_loss_prediction_fs5.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

# 创建一个新的 logger 实例用于参数范数信息
norm_logger = logging.getLogger('norm_info')
norm_logger.setLevel(logging.INFO)

# 创建一个文件处理器，用于将参数范数信息写入单独的文件
norm_file_handler = logging.FileHandler("parameter_norms_prediction_fs5.log") # show L1 and L2 norm
norm_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
norm_file_handler.setFormatter(norm_formatter)

# 将文件处理器添加到 norm_logger
norm_logger.addHandler(norm_file_handler)

# 确保 norm_logger 不会将日志传播到根 logger!!!!
norm_logger.propagate = False

# 定义一个函数来记录参数范数
def log_norm_info(model, epoch):
    norm_logger.info(f"Epoch {epoch} - Parameter Norms for fs5 brain data:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            l1_norm = param.data.norm(1).item()
            l2_norm = param.data.norm(2).item()
            norm_logger.info(f"  {name}: L1 norm = {l1_norm:.4f}, L2 norm = {l2_norm:.4f}")



# training loop starts
start_time = time.time()

# 记录初始参数范数
norm_logger.info("Initial Parameter Norms for fs5 brain data:")
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
        torch.save(model.state_dict(), 'best_model_fs5.pth')
        counter = 0
        logging.info(f"New best model for fs5 brain data saved with validation loss: {best_val_loss:.4f}")
    else:
        counter += 1
        if counter >= patience:
            logging.info("Early stopping")
            break

total_time = time.time() - start_time
logging.info(f"Training completed in {total_time:.2f} seconds")

# 训练结束后记录最终的范数信息
norm_logger.info("Final Parameter Norms for fs5 brain data:")
log_norm_info(model, num_epochs)

# 绘制学习曲线
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('learning_curve_predict_fs5.png')
plt.show()

# 最终测试评估
model.load_state_dict(torch.load('best_model_fs5.pth'))
model.eval()

test_loss = validate_epoch(model, test_loader, criterion)
print(f"Final test loss for fs5 brain data: {test_loss:.4f}")

# 在测试集上进行预测并计算额外的指标
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

# 计算额外的评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(all_targets, all_predictions)
mae = mean_absolute_error(all_targets, all_predictions)
r2 = r2_score(all_targets, all_predictions)

print(f"Test MSE_Model1_fs5: {mse:.4f}")
print(f"Test MAE_Model1_fs5: {mae:.4f}")
print(f"Test R2 Score_Model1_fs5: {r2:.4f}")

# 可视化预测结果
# plt.figure(figsize=(10, 5))
# plt.scatter(all_targets[:, 0], all_predictions[:, 0], alpha=0.5)
# plt.xlabel("True Values_M1")
# plt.ylabel("Predictions_M1")
# plt.title("Predictions vs True Values (Prediction model 1)")
# plt.savefig('test_predictions_M1.png')
# plt.close()

# 可视化预测结果
plt.figure(figsize=(10, 5))
plt.scatter(all_targets[:, 0], all_predictions[:, 0], alpha=0.5)
plt.xlabel("True Values_fs5")
plt.ylabel("Predictions_fs5")
plt.title("Predictions vs True Values (Prediction model 1 with fs5 brain features)")

# 添加一条理想的对角线
min_val = min(all_targets[:, 0].min(), all_predictions[:, 0].min())
max_val = max(all_targets[:, 0].max(), all_predictions[:, 0].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

# 添加一些统计信息
plt.text(0.05, 0.95, f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('test_predictions_fs5.png', dpi=300)
plt.close()

