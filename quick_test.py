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

image_data =  np.load('sample_cnn_lh_brainimages.npy')# shape: (1000, 4, 512, 512) 

# 选择第一个样本
first_sample = image_data[0]  # shape: (4, 512, 512)

# 创建一个2x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# 展平axes数组以便于遍历
axes = axes.flatten()

# 遍历4个通道并在每个子图中显示
for i in range(4):
    ax = axes[i]
    ax.imshow(first_sample[i], cmap='gray')
    ax.set_title(f'Channel {i+1}')
    ax.axis('off')  # 关闭坐标轴

plt.tight_layout()

# 保存图片
plt.savefig('sample_0_brain_4_channels.png', dpi=300, bbox_inches='tight')
plt.close()  # 关闭图形以释放内存
