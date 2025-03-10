# %% 加载数据
import logging
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from experiment_primary_updated import ModelExperiments
from train_precdiction_model_images_improved_lh import BrainADHDModel, BrainMapDataset
import os
import torch
import time
import numpy as np
from torch.utils.data import DataLoader

def load_data(image_path, phenotype_path, use_mmap=True, test_ratio=0.1):
    """
    Load data with mmap mode and optional test mode
    
    Args:
        image_path: 图像数据路径
        phenotype_path: 表型数据路径
        use_mmap: 是否使用内存映射
        test_mode: 是否使用测试模式（只加载部分数据）
        test_ratio: 测试模式下加载的数据比例
    """
    try:
        # 加载数据
        image_data = np.load(image_path, mmap_mode='r' if use_mmap else None)
        phenotype_data = np.load(phenotype_path, mmap_mode='r' if use_mmap else None)
    
        
        logging.info(f"Loaded data shape: Image {image_data.shape}, Phenotype {phenotype_data.shape}")
        return image_data, phenotype_data
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise


image_path = '/projects/0/einf1049/scratch/jouyang/all_cnn_lh_brainimages.npy'
phenotype_path = '/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes.npy'

logging.info("Loading data...")
image_data, loaded_phenotype_tensor = load_data(
    image_path, 
    phenotype_path, 
    use_mmap=True,
    test_ratio=0.1
)

# 创建数据分割
indices = np.arange(len(image_data))
print("please check:")
print(len(image_data))
# train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
# train_idx, val_idx = train_test_split(train_val_idx, test_size=0.11111, random_state=42)

# # 创建数据集
# train_dataset = BrainMapDataset(image_data[train_idx], loaded_phenotype_tensor[train_idx])
# val_dataset = BrainMapDataset(image_data[val_idx], loaded_phenotype_tensor[val_idx])
# test_dataset = BrainMapDataset(image_data[test_idx], loaded_phenotype_tensor[test_idx])