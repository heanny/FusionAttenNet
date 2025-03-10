import logging
import csv
import gc
import os
import torch
import torch.nn as nn
import torch.optim as optim
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
                        logging.FileHandler("all_rh_training_detailed_brainimages_improved.log"),
                        logging.StreamHandler()
                    ])

# BrainMapDataset class
class BrainMapDataset(Dataset):
    def __init__(self, image_data, features_and_targets):
        self.image_data = image_data
        self.features_and_targets = features_and_targets

    def __len__(self):
        return len(self.features_and_targets)

    def __getitem__(self, idx):
        images = self.image_data[idx].astype(np.float32)
        images = torch.from_numpy(images).float()  # convert to [4, 512, 512] 的张量
        
        features_and_target = self.features_and_targets[idx]
        # print(f"完整特征: {features_and_target}")  # 调试打印
        extra_features = features_and_target[2:5].astype(np.float32)  
        # print(f"extra_features: {extra_features}")  # 调试打印
        targets = features_and_target[:2].astype(np.float32)     # get sum_att and sum_agg for prediction
        # print(f"targets: {targets}")  # 调试打印

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

class BrainADHDModel(nn.Module):
    def __init__(self, num_phenotypes):
        super(BrainADHDModel, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers - make sure dimensions match
        total_features = 512 + num_phenotypes  # 512 from CNN + 3 phenotypes
        self.fc1 = nn.Linear(total_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # Output 2 scores
        
        self.dropout = nn.Dropout(0.5)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    

    def forward(self, brain_images, phenotypes):

        # Add shape debugging information
        # print(f"[Debug] Input brain_images shape: {brain_images.shape}")
        # print(f"[Debug] Input phenotypes shape: {phenotypes.shape}")


        x = self.conv1(brain_images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(f"[Debug] After ResNet layers shape: {x.shape}")

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # print(f"[Debug] After flatten shape: {x.shape}")

        # Combine with phenotype data
        combined = torch.cat([x, phenotypes], dim=1)
        # print(f"[Debug] After concatenation shape: {combined.shape}")
        
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# train and validate loop functions
# train and validate loop functions
def train_epoch(model, loader, optimizer, criterion, device, gradient_accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for i, (brain_images, phenotypes, targets) in enumerate(loader):
        # Move to device and convert to half precision if needed
        brain_images = brain_images.to(device)
        phenotypes = phenotypes.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(brain_images, phenotypes)
        loss = criterion(outputs, targets)
        loss = loss / gradient_accumulation_steps  # Normalize loss due to gradient_accumulation_steps

        # loss.backward()
        # optimizer.step()
        # total_loss += loss.item()

        # Backward pass with gradient accumulation
        loss.backward()
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps

        # Clear some memory
        del brain_images, phenotypes, targets, outputs
        torch.cuda.empty_cache()

    return total_loss / len(loader)

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


# main
# main
if __name__ == "__main__":

    # Set memory efficient device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # May help with speed?? not sure...

    # 数据路径
    image_path = '/projects/0/einf1049/scratch/jouyang/all_cnn_rh_brainimages.npy'
    phenotype_path = '/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes.npy'
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

    # Create smaller train/val/test splits: 8:1:1
    indices = np.arange(len(image_data))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.11111, random_state=42)

    # Create datasets using indices
    train_dataset = BrainMapDataset(image_data[train_idx], loaded_phenotype_tensor[train_idx])
    val_dataset = BrainMapDataset(image_data[val_idx], loaded_phenotype_tensor[val_idx])
    test_dataset = BrainMapDataset(image_data[test_idx], loaded_phenotype_tensor[test_idx])
    
    # Reduce batch size and use appropriate num_workers
    batch_size = 8  # Reduced from 16
    num_workers = 2  # Adjust based on your system
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           num_workers=num_workers, pin_memory=True)
    
    
    # # 在创建DataLoader后添加，调试打印
    # for brain_images, phenotypes, targets in train_loader:
    #     print("Phenotypes shape:", phenotypes.shape)      # 应该是 [batch_size, 3]
    #     print("Phenotypes sample:", phenotypes[0])        # 应该是 [age, sex, edu_maternal]
    #     print("Targets shape:", targets.shape)            # 应该是 [batch_size, 2]
    #     print("Targets sample:", targets[0])              # 应该是 [sum_att, sum_agg]
    #     break

    # initialize model, optimizer, and loss functions
    num_phenotypes = 3  # (12595,5) only for age, sex(0/1), edu_maternal(0/1/2)
    model = BrainADHDModel(num_phenotypes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    criterion = nn.MSELoss()

    # learning rate scheduler 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    # training
    num_epochs = 200
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 30
    counter = 0
    gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches
    

    # save the training logs
    csv_file = 'all_rh_training_loss_predictionM1_images_improved.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

    # training starts
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, gradient_accumulation_steps)
        # Validation
        val_loss, _, _ = validate_epoch(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start_time
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logging.info(f"(all, rh) Epoch {epoch+1}/{num_epochs}, "
                     f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                     f"Time: {epoch_time:.2f}s")
        
        # Scheduler step
        scheduler.step(val_loss)

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'all_rh_best_model_improved.pth')
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

    # plot learning rate curves
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs (Improved Model for right semi brain images)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_rh_learning_curve_predict_brainimages_improved.png')
    plt.close()

    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # 定义评估函数
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
            plt.title(f"Predictions vs True Values for {target_name} (right semi brain)")
            
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
            plt.savefig(f'all_rh_test_predictions_{target_name}_brainimages_improved.png', dpi=300)
            plt.close()

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
        
        return metrics

    # 最终评估
    model.load_state_dict(torch.load('all_rh_best_model_improved.pth'))
    model.eval()

    # 使用新的评估函数进行评估
    metrics = evaluate_model(model, test_loader, device)

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




# if __name__ == "__main__":
#     # load data
#     image_data = np.load('all_cnn_rh_brainimages.npy')
#     loaded_phenotype_tensor = np.load('all_normalised_phenotypes.npy')
#     # order: sum_att  sum_agg	age	  sex(0/1)	edu_maternal(0/1/2)

#     # print check for shape
#     # print("\nData Shapes Debugging:")
#     # print(f"Image data shape: {image_data.shape}")
#     # print(f"Phenotype tensor shape: {loaded_phenotype_tensor.shape}")


#     # split data
#     train_val_data, test_data, train_val_features, test_features = train_test_split(
#         image_data, loaded_phenotype_tensor, test_size=0.1, random_state=42)

#     train_data, val_data, train_features, val_features = train_test_split(
#         train_val_data, train_val_features, test_size=0.11111, random_state=42)

#     # create datasets
#     train_dataset = BrainMapDataset(train_data, train_features)
#     val_dataset = BrainMapDataset(val_data, val_features)
#     test_dataset = BrainMapDataset(test_data, test_features)

#     # create data loaders
#     batch_size = 16
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)


#     # check for the batch 1 output
#     # for brain_images, phenotypes, targets in train_loader:
#     #     print("\nFirst Batch Debugging:")
#     #     print(f"Brain images batch shape: {brain_images.shape}")     # [batch_size, 4, 512, 512]
#     #     print(f"Phenotypes batch shape: {phenotypes.shape}")         # [batch_size, 3]
#     #     print(f"Targets batch shape: {targets.shape}")               # [batch_size, 2]
#     #     print("\nPhenotypes first sample(age, sex(0/1), edu_maternal(0/1/2)):", phenotypes[0])           # 3 values
#     #     print("Targets first sample(sum_att, sum_agg):", targets[0])                   # 2 values
#     #     break

#     # initialize model, optimizer, and loss functions
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     num_phenotypes = 3  # (1000,5) only for age, sex(0/1), edu_maternal(0/1/2)
#     model = BrainADHDModel(num_phenotypes).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
#     criterion = nn.MSELoss()

#     # learning rate scheduler 
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

#     # training
#     num_epochs = 200
#     best_val_loss = float('inf')
#     train_losses = []
#     val_losses = []
#     patience = 30
#     counter = 0

#     # save the training logs
#     csv_file = 'all_rh_training_loss_predictionM1_images_improved.csv'
#     with open(csv_file, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

#     # training starts
#     start_time = time.time()

#     for epoch in range(num_epochs):
#         epoch_start_time = time.time()
#         train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
#         val_loss, _, _ = validate_epoch(model, val_loader, criterion, device)
#         epoch_time = time.time() - epoch_start_time
        
#         train_losses.append(train_loss)
#         val_losses.append(val_loss)
        
#         logging.info(f"(all, rh) Epoch {epoch+1}/{num_epochs}, "
#                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
#                      f"Time: {epoch_time:.2f}s")
        
#         scheduler.step(val_loss)
#         with open(csv_file, 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([epoch + 1, train_loss, val_loss])
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), 'all_rh_best_model_improved.pth')
#             counter = 0
#             logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
#         else:
#             counter += 1
#             if counter >= patience:
#                 logging.info("Early stopping")
#                 break

#     total_time = time.time() - start_time
#     logging.info(f"Training completed in {total_time:.2f} seconds")

#     # plot learning rate curves
#     plt.figure(figsize=(10,5))
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.title('Training and Validation Loss over Epochs (Improved Model for right semi brain images)')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('all_rh_learning_curve_predict_M1_brainimages_improved.png')
#     plt.close()

#     # 定义评估函数
#     def evaluate_model(model, test_loader, device):
#         model.eval()
#         all_predictions = []
#         all_targets = []
        
#         with torch.no_grad():
#             for brain_maps, extra_features, targets in test_loader:
#                 brain_maps = brain_maps.to(device)
#                 extra_features = extra_features.to(device)
#                 outputs = model(brain_maps, extra_features)
#                 all_predictions.extend(outputs.cpu().numpy())
#                 all_targets.extend(targets.numpy())
        
#         all_predictions = np.array(all_predictions)
#         all_targets = np.array(all_targets)
        
#         # 计算每个目标变量的指标
#         metrics = {}
#         for i, target_name in enumerate(['sum_att', 'sum_agg']):
#             mse = mean_squared_error(all_targets[:, i], all_predictions[:, i])
#             mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
#             r2 = r2_score(all_targets[:, i], all_predictions[:, i])
            
#             metrics[target_name] = {
#                 'MSE': mse,
#                 'MAE': mae,
#                 'R2': r2
#             }
            
#             # 绘制散点图
#             plt.figure(figsize=(10, 5))
#             plt.scatter(all_targets[:, i], all_predictions[:, i], alpha=0.5)
#             plt.xlabel(f"True Values ({target_name})")
#             plt.ylabel(f"Predictions ({target_name})")
#             plt.title(f"Predictions vs True Values for {target_name} (right semi brain)")
            
#             # 添加对角线
#             min_val = min(all_targets[:, i].min(), all_predictions[:, i].min())
#             max_val = max(all_targets[:, i].max(), all_predictions[:, i].max())
#             plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
            
#             # 添加统计信息
#             plt.text(0.05, 0.95, 
#                     f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', 
#                     transform=plt.gca().transAxes,
#                     verticalalignment='top', 
#                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
#             plt.legend()
#             plt.grid(True, linestyle='--', alpha=0.7)
#             plt.tight_layout()
#             plt.savefig(f'all_rh_test_predictions_{target_name}_brainimages_improved.png', dpi=300)
#             plt.close()
        
#         return metrics

#     # 最终评估
#     model.load_state_dict(torch.load('all_rh_best_model_improved.pth'))
#     model.eval()

#     # 使用新的评估函数进行评估
#     metrics = evaluate_model(model, test_loader, device)

#     # 打印评估结果
#     print("\nFinal Evaluation Results:")
#     for target_name, target_metrics in metrics.items():
#         print(f"\nMetrics for {target_name}:")
#         for metric_name, value in target_metrics.items():
#             print(f"{metric_name}: {value:.4f}")

#     # 打印最终测试损失
#     test_loss, _, _ = validate_epoch(model, test_loader, criterion, device)
#     print(f"\nFinal test loss: {test_loss:.4f}")


