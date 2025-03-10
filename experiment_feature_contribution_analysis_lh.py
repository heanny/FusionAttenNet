"""
Experiment 2: Feature Experiment - Feature Contribution Analysis Framework

Experiment settings
- Based on the logical relationship of cortex features: volume = thickness × surface area, 
we can use the combination of thickness and surface area or volume in the experiments, I have trimmed down the experiments settings as follow:
Grouping based on feature type:
- Brain feature group (B):
    B1: complete brain feature group        B2: thickness + area               B3: volume           B4: white-to-gray ratio         
- Phenotypic feature group (D):
    D1: complete phenotype group (age + sex + edu)    D2: age + sex     D3: age + maternal educational level   D4: sex + maternal educational level
    D5: age                                                                      D6: sex               D7: maternal educational level

Experimental combinations
- Main combinations (1 in total): Full model (all features) - Only all brain features (B1) - Only all phenotypic features (D1)
- Brain feature analysis (5 in total): [B2 + D1],  [B3 + D1], [B4 + D1], [(B2 + B4) + D1], [(B3 + B4) + D1]
- Phenotypic feature analysis (6 in total): 
    For combined phenotypic features: B1 + D2/D3/D4
    For sole phenotypic features: B1 + D5/D6/D7
- Optimal combination verification:
    If [(B2 + B4) + D1] or [(B3 + B4) + D1] performs best, then test:
    - Optimal brain feature combination + D2/D3/D4/D5/D6/D7

"""

import logging
import tqdm
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import json
import gc
import os
import shap
import time
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from expt_prediction_model_images_improved_lh import BrainMapDataset, DynamicBrainADHDModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from torch.cuda.amp import GradScaler, autocast
import psutil
import GPUtil

class ExperimentManager:
    """实验管理器：负责实验目录创建和进度管理"""
    def __init__(self, test_mode=False):
        self.test_mode = test_mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self._create_experiment_directories()
        self.progress_file = self.experiment_dir / 'experiment_progress.json'
        
        # 设置日志
        self._setup_logging()
        
        # 记录实验配置
        self.config = {
            'test_mode': test_mode,
            'epochs': 5 if test_mode else 30,
            'patience': 10,
            'batch_size': 4 if test_mode else 8,
            'timestamp': self.timestamp
        }
        self._save_config()

    def _create_experiment_directories(self):
        """创建实验目录结构"""
        base_dir = Path('/home/jouyang1/test_experiment_results_lh' if self.test_mode else '/home/jouyang1/experiment_results_feat_contr_analys_lh')
        experiment_dir = base_dir / f'experiment_{self.timestamp}'
        
        # 定义目录结构
        directories = [
            'brain_features/1_1_complete_evaluation/metrics',
            'brain_features/1_1_complete_evaluation/plots',
            'brain_features/1_2_individual_analysis/metrics',
            'brain_features/1_2_individual_analysis/plots',
            'brain_features/importance_ranking',
            'phenotype_features/complete_evaluation',
            'phenotype_features/individual_analysis',
            'phenotype_features/importance_quantification',
            'interactions/manova_results/effect_sizes',
            'interactions/pca_results/variance_explained',
            'interactions/cca_results/canonical_correlations',
            'models/brain_features',
            'models/phenotype_features',
            'logs',
            'metrics',
            'visualizations',
            'data'
        ]
        
        # 创建目录
        for directory in directories:
            (experiment_dir / directory).mkdir(parents=True, exist_ok=True)
            
        return experiment_dir

    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.experiment_dir / 'logs' / 'experiment.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info(f"experiment directory is created: {self.experiment_dir}")

    def _convert_to_serializable(self, obj):
        """将对象转换为JSON可序列化的格式。

        Parameters
        ----------
        obj : Any
            需要转换的对象

        Returns
        -------
        Any
            转换后的可JSON序列化对象
        """
        
        # 处理numpy的基本数据类型
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (str, np.str_, np.bytes_)):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.ndarray, np.generic)):
            return self._convert_to_serializable(obj.tolist())
            
        # 处理PyTorch张量
        if torch.is_tensor(obj):
            return self._convert_to_serializable(obj.cpu().detach().numpy())
            
        # 处理Path对象
        if isinstance(obj, Path):
            return str(obj)
            
        # 处理字典
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
            
        # 处理列表和元组
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
            
        # 处理具有tolist方法的对象（如pandas.Series）
        if hasattr(obj, 'tolist'):
            try:
                return self._convert_to_serializable(obj.tolist())
            except:
                pass
                
        # 处理具有__dict__属性的对象
        if hasattr(obj, '__dict__'):
            try:
                return self._convert_to_serializable(obj.__dict__)
            except:
                pass
                
        # 尝试直接转换
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)

    def _save_config(self):
        """保存实验配置"""
        config_file = self.experiment_dir / 'experiment_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def save_progress(self, stage, results):
        """保存实验进度"""
        progress = self.load_progress()
        
        # 转换结果为可序列化的形式
        try:
            serializable_results = self._convert_to_serializable(results)
        except Exception as e:
            logging.error(f"Error converting results to serializable: {e}")
            logging.error(f"Results type: {type(results)}")
            logging.error(f"Sample of results: {results}")
            raise

        progress[stage] = {
            'completed': True,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': serializable_results
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
            
        logging.info(f"Saved the experiment process: {stage} finished")

        
    def load_progress(self):
        """加载实验进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}

class MemoryManager:
    """内存管理器：负责内存监控和清理"""
    @staticmethod
    def cleanup():
        """清理内存和GPU缓存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 额外的GPU内存整理
            torch.cuda.synchronize()

    @staticmethod
    def get_memory_status():
        """详细的内存状态报告"""
        memory_info = {
            'cpu': {
                'used': psutil.Process().memory_info().rss / (1024 * 1024),
                'percent': psutil.virtual_memory().percent,
                'available': psutil.virtual_memory().available / (1024 * 1024)
            }
        }
        
        if torch.cuda.is_available():
            gpu_info = GPUtil.getGPUs()[0]
            memory_info['gpu'] = {
                'used': gpu_info.memoryUsed,
                'total': gpu_info.memoryTotal,
                'percent': (gpu_info.memoryUsed / gpu_info.memoryTotal) * 100
            }
            
        return memory_info


    @staticmethod
    def log_memory_status(stage):
        """增强的内存状态日志"""
        memory_status = MemoryManager.get_memory_status()
        logging.info(f"\nMemory Status at {stage}:")
        logging.info(f"CPU Memory:")
        logging.info(f"  Used: {memory_status['cpu']['used']:.2f}MB")
        logging.info(f"  Available: {memory_status['cpu']['available']:.2f}MB")
        logging.info(f"  Usage: {memory_status['cpu']['percent']}%")
        
        if 'gpu' in memory_status:
            logging.info(f"GPU Memory:")
            logging.info(f"  Used: {memory_status['gpu']['used']}MB")
            logging.info(f"  Total: {memory_status['gpu']['total']}MB")
            logging.info(f"  Usage: {memory_status['gpu']['percent']:.2f}%")

class DataManager:
    """数据管理器：负责数据加载和处理"""
    def __init__(self, image_path, phenotype_path, test_mode=False):
        self.image_path = image_path
        self.phenotype_path = phenotype_path
        self.test_mode = test_mode
        self.batch_size = 4 if test_mode else 8
        self.num_workers = 2
        # add cache flag
        self._data_cache = None

    def load_data(self, use_mmap=True):
        """加载数据集,使用内存映射和分批处理"""
        try:
            # 记录数据加载前的内存状态
            MemoryManager.log_memory_status("Before data loading")

            if self._data_cache is not None:
                return self._data_cache

            # 使用内存映射模式加载
            image_data = np.load(self.image_path, mmap_mode='r' if use_mmap else None)
            phenotype_data = np.load(self.phenotype_path, mmap_mode='r' if use_mmap else None)

            # 记录数据加载后的内存状态
            MemoryManager.log_memory_status("After data loading")
            
            if self.test_mode:
                # 测试模式下只使用10%的数据
                total_samples = len(image_data)
                test_samples = int(total_samples * 0.1)
                indices = np.random.choice(total_samples, test_samples, replace=False)
                indices.sort()
                image_data = image_data[indices]
                phenotype_data = phenotype_data[indices]
            
            # 缓存数据
            self._data_cache = (image_data, phenotype_data)                
            logging.info(f"Data loading is finished: Image shape {image_data.shape}, Phenotype data shape {phenotype_data.shape}")
            # return image_data, phenotype_data
            return self._data_cache
            
        except Exception as e:
            logging.error(f"Data loading errors: {str(e)}")
            raise
    
    def verify_data_structure(self):
        """验证数据结构和特征分布"""
        # 加载数据
        image_data = np.load(self.image_path)
        phenotype_data = np.load(self.phenotype_path)
        
        print("数据形状:")
        print(f"Brain image data shape: {image_data.shape}")  # 应该是 (N, 4, 512, 512)
        print(f"Phenotype data shape: {phenotype_data.shape}")  # 应该是 (N, 3)
        
        print("\n脑部特征统计:")
        feature_names = ['Thickness', 'Volume', 'Area', 'White-Gray Ratio']
        for i, name in enumerate(feature_names):
            channel_data = image_data[:, i, :, :]
            print(f"\n{name}:")
            print(f"Mean: {np.mean(channel_data):.4f}")
            print(f"Std: {np.std(channel_data):.4f}")
            print(f"Range: [{np.min(channel_data):.4f}, {np.max(channel_data):.4f}]")
        
        print("\n表型特征统计:")
        pheno_names = ['Age', 'Sex(0/1)', 'Education(0/1/2)']
        for i, name in enumerate(pheno_names):
            print(f"\n{name}:")
            print(f"Mean: {np.mean(phenotype_data[:, i]):.4f}")
            print(f"Std: {np.std(phenotype_data[:, i]):.4f}")
            print(f"Unique values: {np.unique(phenotype_data[:, i])}")

    def create_data_splits(self, image_data, phenotype_data):
        """创建6/2/2的数据划分"""
        indices = np.arange(len(image_data))
        
        # 首先分出测试集（20%）
        train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        # 再将剩余数据分为训练集（75%）和验证集（25%），相当于整体的60%和20%
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)

        # 使用SubsetRandomSampler避免数据复制
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        
        # 创建单个数据集实例
        dataset = BrainMapDataset(image_data, phenotype_data)
        
        # # 创建数据集
        # train_dataset = BrainMapDataset(image_data[train_idx], phenotype_data[train_idx])
        # val_dataset = BrainMapDataset(image_data[val_idx], phenotype_data[val_idx])
        # test_dataset = BrainMapDataset(image_data[test_idx], phenotype_data[test_idx])
        
        # 创建数据加载器
        train_loader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True  # 保持worker进程活跃
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=test_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        logging.info(f"Data is divided: training set {len(train_idx)}, validation set {len(val_idx)}, testing set {len(test_idx)}")
        # return train_loader, val_loader, test_loader, (train_dataset, val_dataset, test_dataset)
        return train_loader, val_loader, test_loader, (dataset, dataset, dataset)


class ModelTrainer:
    """模型训练器：负责模型训练和评估"""
    def __init__(self, model, device, test_mode=False):
        self.model = model
        self.device = device
        self.test_mode = test_mode
        self.epochs = 5 if test_mode else 200
        self.patience = 30
        self.scaler = torch.amp.GradScaler('cuda')
        self.gradient_accumulation_steps = 4  # 添加梯度累积步数

        # 设置PyTorch内存分配器
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.85)  # 限制GPU内存使用
            torch.backends.cudnn.benchmark = True


    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0

        # 记录训练开始时的内存状态
        MemoryManager.log_memory_status("Start of training epoch")
        
        # 使用tqdm显示进度
        for i, (brain_images, phenotypes, targets) in enumerate(train_loader):
            try:
                # 将数据移到设备
                brain_images = brain_images.to(self.device, non_blocking=True)
                phenotypes = phenotypes.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # 添加维度打印
                print(f"Batch {i} shapes:")
                print(f"brain_images: {brain_images.shape}")
                print(f"phenotypes: {phenotypes.shape}")
                print(f"targets: {targets.shape}")
                
                optimizer.zero_grad(set_to_none=True)
                
                # 使用混合精度训练
                with torch.amp.autocast('cuda'):
                    outputs = self.model(brain_images, phenotypes)
                    loss = criterion(outputs, targets)
                     # 根据梯度累积步数归一化损失
                    loss = loss / self.gradient_accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
                # 根据梯度累积步数更新参数
                if (i + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                batch_count += 1
                
                # 清理内存, every 5 batches
                del brain_images, phenotypes, outputs
                if batch_count % 5 == 0: 
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    MemoryManager.cleanup()
                    logging.warning(f"Out of memory, please clear the memory to continue...")
                    continue
                else:
                    raise e

        # 记录训练结束时的内存状态
        MemoryManager.log_memory_status("End of training epoch")
        return total_loss / batch_count

    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for brain_images, phenotypes, targets in val_loader:
                brain_images = brain_images.to(self.device)
                phenotypes = phenotypes.to(self.device)
                targets = targets.to(self.device)

                with torch.amp.autocast('cuda'):
                    outputs = self.model(brain_images, phenotypes)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets_list.extend(targets.cpu().numpy())
                
                # 清理内存
                del brain_images, phenotypes, outputs
                MemoryManager.cleanup()
        
        return total_loss / len(val_loader), np.array(predictions), np.array(targets_list)


class FeatureAnalysis:
    """特征分析类：实现所有特征分析相关功能"""
    def __init__(self, experiment_manager, model_trainer, device):
        self.experiment_manager = experiment_manager
        self.model_trainer = model_trainer
        self.device = device
        self.results_dir = experiment_manager.experiment_dir
        
        # # 定义特征组
        # # brain data order: thickness, volume, surface area, white_gray_matter_ratio
        # self.brain_features = {
        #     'B1': {'name': 'complete_brain', 'indices': slice(4)}, # [thickness, volume, area, white_gray_ratio]
        #     'B2': {'name': 'thickness_area', 'indices': [0, 2]}, # thickness + surface area
        #     'B3': {'name': 'volume', 'indices': [1]}, # volume
        #     'B4': {'name': 'white_gray_ratio', 'indices': [3]} # white_gray_matter_ratio
        # }
        
        # # D1: complete phenotype group (age + sex + edu)    D2: age + sex     D3: age + maternal educational level   D4: sex + maternal educational level
        # # D5: age    D6: sex    D7: maternal educational level
        # # phenetype data order: age, sex(0/1), edu_maternal(0/1/2)
        # self.phenotype_features = {
        #     'D1': {'name': 'all_phenotype', 'indices': slice(None)}, 
        #     'D2': {'name': 'age_sex', 'indices': [0, 1]},
        #     'D3': {'name': 'age_edu', 'indices': [0, 2]},
        #     'D4': {'name': 'sex_edu', 'indices': [1, 2]},
        #     'D5': {'name': 'age', 'indices': [0]},
        #     'D6': {'name': 'sex', 'indices': [1]},
        #     'D7': {'name': 'edu', 'indices': [2]}
        # }
        # 定义特征组配置
        # brain data order: thickness, volume, surface area, white_gray_matter_ratio
        self.brain_features = {
            'B1': {
                'name': 'complete_brain',
                'indices': slice(4),
                'channels': 4,
                'description': 'Complete brain feature group'
            },
            'B2': {
                'name': 'thickness_area',
                'indices': [0, 2],
                'channels': 2,
                'description': 'Thickness + surface area'
            },
            'B3': {
                'name': 'volume',
                'indices': [1],
                'channels': 1,
                'description': 'Volume'
            },
            'B4': {
                'name': 'white_gray_ratio',
                'indices': [3],
                'channels': 1,
                'description': 'White-to-gray ratio'
            }
        }
        
        # phenetype data order: age, sex(0/1), edu_maternal(0/1/2)
        self.phenotype_features = {
            'D1': {
                'name': 'all_phenotype',
                'indices': slice(None),
                'channels': 3,
                'description': 'Complete phenotype group'
            },
            'D2': {
                'name': 'age_sex',
                'indices': [0, 1],
                'channels': 2,
                'description': 'Age + sex'
            },
            'D3': {
                'name': 'age_edu',
                'indices': [0, 2],
                'channels': 2,
                'description': 'Age + maternal educational level'
            },
            'D4': {
                'name': 'sex_edu',
                'indices': [1, 2],
                'channels': 2,
                'description': 'Sex + maternal educational level'
            },
            'D5': {
                'name': 'age',
                'indices': [0],
                'channels': 1,
                'description': 'Age only'
            },
            'D6': {
                'name': 'sex',
                'indices': [1],
                'channels': 1,
                'description': 'Sex only'
            },
            'D7': {
                'name': 'edu',
                'indices': [2],
                'channels': 1,
                'description': 'Maternal educational level only'
            }
        }

    def _get_indices_length(self, indices):
        """Helper method to get length of indices, whether they're slice objects or lists"""
        if isinstance(indices, slice):
            # For slice(None), this means all elements, so we need to return the full feature dimension
            if indices.start is None and indices.stop is None:
                # Return the full dimension size based on feature type
                return 3  # Assuming 3 features for phenotype (age, sex, education)
            else:
                # Calculate length from slice
                start = indices.start or 0
                stop = indices.stop
                step = indices.step or 1
                return len(range(start, stop, step))
        else:
            # For lists or arrays
            return len(indices)
    
    
    def create_model_for_experiment(self, brain_config, pheno_config):
        """创建实验用的模型实例
        参数:
        brain_config: str, 脑部特征配置 ('B1', 'B2', 'B3', 'B4', 'B2B4','B3B4')
        pheno_config: str, 表型特征配置 ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7')
        """
        
        model = DynamicBrainADHDModel(
            brain_feature_config=brain_config,  
            phenotype_feature_config=pheno_config
        )
        return model
    
    # def _get_brain_config_from_indices(self, indices):
    #     """将特征索引转换为脑部配置名称"""
    #     for config_name, config in self.brain_features.items():
    #         if isinstance(indices, slice) and config['indices'] == indices:
    #             return config_name
    #         elif isinstance(indices, list) and config['indices'] == indices:
    #             return config_name
    #     raise ValueError(f"Invalid brain feature indices: {indices}")

    def _get_brain_config_from_indices(self, indices):
        """将特征索引转换为脑部配置名称"""
        # 确保indices是列表类型
        if not isinstance(indices, (list, slice)):
            indices = list(indices)
        
        # 对于组合特征，创建一个新的配置名称
        if isinstance(indices, list) and len(indices) > 1:
            # 检查是否是已知的组合
            if sorted(indices) == sorted(list(self.brain_features['B2']['indices']) + [self.brain_features['B4']['indices'][0]]):
                return 'B2B4'
            elif sorted(indices) == sorted([self.brain_features['B3']['indices'][0], self.brain_features['B4']['indices'][0]]):
                return 'B3B4'
        
        # 对于单个特征组，检查已存在的配置
        for config_name, config in self.brain_features.items():
            if isinstance(indices, slice) and config['indices'] == indices:
                return config_name
            elif isinstance(indices, list):
                if isinstance(config['indices'], slice):
                    config_indices = list(range(config['indices'].start or 0, 
                                            config['indices'].stop, 
                                            config['indices'].step or 1))
                else:
                    config_indices = config['indices']
                
                if sorted(indices) == sorted(config_indices):
                    return config_name
                    
        raise ValueError(f"Invalid brain feature indices: {indices}")

    def _get_pheno_config_from_indices(self, indices):
        """将特征索引转换为表型配置名称"""
        for config_name, config in self.phenotype_features.items():
            if isinstance(indices, slice) and config['indices'] == indices:
                return config_name
            elif isinstance(indices, list) and config['indices'] == indices:
                return config_name
        raise ValueError(f"Invalid phenotype feature indices: {indices}")


    def _train_and_evaluate(self, train_loader, val_loader, brain_indices, phenotype_indices):
        """为每个特征组合训练和评估模型"""

        # Get feature dimensions using the helper method
        phenotype_dim = self._get_indices_length(phenotype_indices)

        # 将特征索引转换为配置名称
        brain_config = self._get_brain_config_from_indices(brain_indices)
        pheno_config = self._get_pheno_config_from_indices(phenotype_indices)

        # 获取特征数量
        def get_indices_length(indices):
            if isinstance(indices, slice):
                return indices.stop  # 现在直接返回 stop 值就可以了
            else:
                # 如果是列表或数组，直接返回长度
                return len(indices)
        
        experiment_name = f"B{get_indices_length(brain_indices)}_D{get_indices_length(phenotype_indices)}_{self.experiment_manager.timestamp}"
        model_save_path = self.results_dir / 'models' / f'{experiment_name}_best_model.pth'
        
        # 确保模型保存目录存在
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        num_epochs = 200 if not self.experiment_manager.test_mode else 5
        patience = 30
        best_val_loss = float('inf')
        counter = 0
        
        # 初始化模型
        # model = BrainADHDModel(phenotype_dim).to(self.device)
        model = self.create_model_for_experiment(brain_config, pheno_config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)
        criterion = nn.MSELoss()

        # 更新ModelTrainer的model
        self.model_trainer.model = model
        
        train_losses = []
        val_losses = []
        best_predictions = None
        best_targets = None
        
        logging.info(f"Starting training for feature combination: {experiment_name}")
        
        try:
            for epoch in range(num_epochs):
                # 训练
                train_loss = self.model_trainer.train_epoch(
                    train_loader, optimizer, criterion
                )
                
                # 验证
                val_loss, predictions, targets = self.model_trainer.validate(
                    val_loader, criterion
                )
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                logging.info(f"Epoch {epoch+1}/{num_epochs}, "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_predictions = predictions
                    best_targets = targets
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'brain_indices': brain_indices,
                        'phenotype_indices': phenotype_indices
                    }, model_save_path)
                    counter = 0
                    logging.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
                else:
                    counter += 1
                    if counter >= patience:
                        logging.info("Early stopping triggered")
                        break
                
                scheduler.step(val_loss)
                
                # 清理内存
                if epoch % 5 == 0:
                    MemoryManager.cleanup()
                    
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise
            
        # 加载最佳模型进行返回
        best_checkpoint = torch.load(model_save_path)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        return {
            'model': model,
            'predictions': best_predictions,
            'targets': best_targets,
            'history': {
                'train_loss': train_losses,
                'val_loss': val_losses
            },
            'model_path': model_save_path
        }


    def analyze_brain_features(self, train_loader, val_loader):
        """
        1. Brain Feature Importance Analysis
        包括完整脑部特征评估、单个特征组合分析和特征重要性排序
        """
        logging.info("Starting Experiment 1. Brain Feature Importance Analysis...")
        MemoryManager.log_memory_status("Before brain feature analysis")
        
        results = {
            'individual_features_analysis': {}, # 1.1
            'combined_feature_analysis': {}, # 1.2
            'importance_ranking': {} # 1.3
        }

        # 1.1 Complete brain feature group evaluation
        results['individual_features_analysis'] = self._analyze_individual_brain_features(
            train_loader, 
            val_loader
        )
        MemoryManager.log_memory_status("After individual features analysis")

        # 1.2 Individual feature combination analysis
        results['combined_feature_analysis'] = self._analyze_brain_feature_combinations(
            train_loader, 
            val_loader
        )
        MemoryManager.log_memory_status("After combined feature analysis")

        # 1.3 Feature importance ranking
        results['importance_ranking'] = self._rank_brain_features(
            train_loader,
            val_loader
        )
        # 记录每个主要步骤的内存状态
        MemoryManager.log_memory_status("After importance ranking")
        

        # 保存结果
        self._save_brain_analysis_results(results)
        return results
    
    def _calculate_feature_importance(self, model, feature_indices):
        """
        计算特征重要性
        
        Parameters:
        -----------
        model : BrainADHDModel
            训练好的模型
        feature_indices : list or slice
            特征的索引
        
        Returns:
        --------
        dict : 包含特征重要性分数的字典
        """
        importance_scores = {}
        
        try:
            # 1. 基于模型权重的重要性
            with torch.no_grad():
                # 获取第一个全连接层的权重
                fc1_weights = model.fc1.weight.cpu().numpy()
                
                # 计算每个特征的权重绝对值平均
                if isinstance(feature_indices, slice):
                    start = feature_indices.start or 0
                    stop = feature_indices.stop
                    indices = range(start, stop)
                else:
                    indices = feature_indices
                    
                for idx in indices:
                    feature_name = f'feature_{idx}'
                    # 为每个特征计算其对应权重的平均绝对值
                    importance = np.mean(np.abs(fc1_weights[:, idx]))
                    importance_scores[feature_name] = float(importance)  # convert to Python float for JSON serialization
            
            # 2. 归一化重要性分数
            max_score = max(importance_scores.values())
            if max_score > 0:
                importance_scores = {
                    k: float(v/max_score)  # 归一化并确保是Python float
                    for k, v in importance_scores.items()
                }
                
            logging.info(f"Feature importance calculation completed for {len(importance_scores)} features")
            
        except Exception as e:
            logging.error(f"Error in calculating feature importance: {str(e)}")
            # 在出错的情况下返回空的重要性分数
            importance_scores = {f'feature_{idx}': 0.0 for idx in indices}
        
        return {
            'importance_scores': importance_scores,
            'method': 'weight_based',
            'description': 'Feature importance based on average absolute weights from first FC layer'
        }

    def _calculate_shap_values(self, train_loader):
        """
        计算脑部特征的SHAP值，用于实验1.3 Brain Feature Importance Analysis
        
        Parameters:
        -----------
        train_loader : DataLoader
            训练数据加载器
        
        Returns:
        --------
        dict : 包含每个脑部特征SHAP值的字典，按{feature_name: shap_value}格式
        """
        try:
            # 收集样本数据
            brain_features = []  # 脑部特征 - 我们关心的特征
            pheno_features = []  # 表型特征 - 必要的模型输入
            sample_size = min(100, len(train_loader.dataset))
            
            # 随机选择背景数据
            indices = np.random.choice(len(train_loader.dataset), sample_size, replace=False)
            for idx in indices:
                brain_images, phenotypes, _ = train_loader.dataset[idx]
                brain_features.append(brain_images.clone())  # 脑部特征
                pheno_features.append(phenotypes.clone())    # 表型特征
            
            brain_features = torch.stack(brain_features).to(self.device)
            pheno_features = torch.stack(pheno_features).to(self.device)

            # 确保模型在评估模式
            self.model_trainer.model.eval()

            # 使用DeepExplainer
            with torch.no_grad():

                # 创建SHAP解释器 - 需要提供完整的模型输入
                explainer = shap.DeepExplainer(
                    self.model_trainer.model, 
                    [brain_features.clone(), pheno_features.clone()]  # 模型需要两个输入
                )
                
                # 计算SHAP值 - 同样需要完整输入
                shap_values = explainer.shap_values(
                    [brain_features.clone(), pheno_features.clone()]
                )

            # 只处理与脑部特征相关的SHAP值
            brain_shap = shap_values[0]  # 第一个输出对应brain_features
            brain_importance = np.abs(brain_shap).mean(0).mean(-1).mean(-1)  # 对样本和空间维度取平均
            
            # 创建脑部特征重要性字典
            feature_names = ['thickness', 'volume', 'surface_area', 'white_gray_ratio']
            feature_importance = {
                name: float(importance) 
                for name, importance in zip(feature_names, brain_importance)
            }
        
            logging.info("Brain feature SHAP value calculation completed successfully")
            return feature_importance
            
        except Exception as e:
            logging.error(f"Error in calculating brain feature SHAP values: {str(e)}")
            return {}

    def _calculate_phenotype_shap_values(self, train_loader):
        """
        计算表型特征的SHAP值
        
        Parameters:
        -----------
        train_loader : DataLoader
            训练数据加载器
        
        Returns:
        --------
        dict : 包含每个表型特征SHAP值的字典
        """
        
        try:
            # 收集表型数据
            phenotype_data = []
            sample_size = min(100, len(train_loader.dataset))
            indices = np.random.choice(len(train_loader.dataset), sample_size, replace=False)
            
            for idx in indices:
                _, phenotypes, _ = train_loader.dataset[idx]
                phenotype_data.append(phenotypes.numpy())
            
            phenotype_data = np.stack(phenotype_data)
            
            # 创建SHAP解释器
            background = torch.FloatTensor(phenotype_data).to(self.device)
            explainer = shap.DeepExplainer(self.model_trainer.model, background)
            
            # 计算SHAP值
            test_indices = np.random.choice(len(train_loader.dataset), min(50, len(train_loader.dataset)), replace=False)
            test_samples = []
            
            for idx in test_indices:
                _, phenotypes, _ = train_loader.dataset[idx]
                test_samples.append(phenotypes.numpy())
            
            test_samples = np.stack(test_samples)
            shap_values = explainer.shap_values(torch.FloatTensor(test_samples).to(self.device))
            
            # 处理SHAP值
            feature_importance = {}
            shap_values = np.abs(shap_values).mean(0)
            
            phenotype_names = ['age', 'sex', 'education']
            for i, importance in enumerate(shap_values):
                feature_importance[phenotype_names[i]] = float(importance)
            
            logging.info("Phenotype SHAP value calculation completed successfully")
            return feature_importance
            
        except Exception as e:
            logging.error(f"Error in phenotype SHAP value calculation: {str(e)}")
            return {}

    def _rank_features_by_shap(self, shap_values):
        """
        基于SHAP值对特征进行排序
        
        Parameters:
        -----------
        shap_values : dict
            包含特征SHAP值的字典
        
        Returns:
        --------
        dict : 排序后的特征重要性字典
        """
        try:
            # 按SHAP值大小排序
            sorted_features = sorted(shap_values.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features)
        except Exception as e:
            logging.error(f"Error in ranking features by SHAP: {str(e)}")
            return {}

    def _rank_by_shap(self, shap_values):
        """
        基于SHAP值对表型特征进行排序（与_rank_features_by_shap功能相同，但用于表型特征）
        
        Parameters:
        -----------
        shap_values : dict
            包含表型特征SHAP值的字典
        
        Returns:
        --------
        dict : 排序后的特征重要性字典
        """
        return self._rank_features_by_shap(shap_values)  # 直接复用现有方法

    def _calculate_permutation_importance(self, train_loader, val_loader, n_repeats=5):
        """
        计算特征的排列重要性
        
        Parameters:
        -----------
        train_loader : DataLoader
            训练数据加载器
        val_loader : DataLoader
            验证数据加载器
        n_repeats : int
            重复计算次数
        
        Returns:
        --------
        dict : 包含每个特征排列重要性的字典
        """
        try:
            model = self.model_trainer.model
            model.eval()
            feature_importance = {}
            
            # 收集基准性能
            baseline_loss = 0
            n_samples = 0
            criterion = nn.MSELoss()
            
            with torch.no_grad():
                for brain_images, phenotypes, targets in val_loader:
                    brain_images = brain_images.to(self.device)
                    phenotypes = phenotypes.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(brain_images, phenotypes)
                    loss = criterion(outputs, targets)
                    baseline_loss += loss.item() * len(targets)
                    n_samples += len(targets)
            
            baseline_loss /= n_samples
            
            # 计算每个特征的重要性
            for feature_idx in range(4):  # 4个脑部特征
                importance = []
                
                for _ in range(n_repeats):
                    permuted_loss = 0
                    n_samples = 0
                    
                    with torch.no_grad():
                        for brain_images, phenotypes, targets in val_loader:
                            brain_images = brain_images.to(self.device)
                            phenotypes = phenotypes.to(self.device)
                            targets = targets.to(self.device)
                            
                            # 打乱特定特征
                            permuted_images = brain_images.clone()
                            perm_idx = torch.randperm(len(brain_images))
                            permuted_images[:, feature_idx] = brain_images[perm_idx, feature_idx]
                            
                            outputs = model(permuted_images, phenotypes)
                            loss = criterion(outputs, targets)
                            permuted_loss += loss.item() * len(targets)
                            n_samples += len(targets)
                    
                    permuted_loss /= n_samples
                    importance.append(permuted_loss - baseline_loss)
                
                feature_importance[f'feature_{feature_idx}'] = float(np.mean(importance))
            
            logging.info("Permutation importance calculation completed successfully")
            return feature_importance
            
        except Exception as e:
            logging.error(f"Error in calculating permutation importance: {str(e)}")
            return {}

    def _calculate_phenotype_permutation_importance(self, train_loader, val_loader, n_repeats=5):
        """
        计算表型特征的排列重要性
        
        Parameters:
        -----------
        train_loader : DataLoader
            训练数据加载器
        val_loader : DataLoader
            验证数据加载器
        n_repeats : int
            重复计算次数
        
        Returns:
        --------
        dict : 包含每个表型特征排列重要性的字典
        """
        try:
            model = self.model_trainer.model
            model.eval()
            feature_importance = {}
            phenotype_names = ['age', 'sex', 'education']
            criterion = nn.MSELoss()
            
            # 收集基准性能
            baseline_loss = 0
            n_samples = 0
            
            with torch.no_grad():
                for brain_images, phenotypes, targets in val_loader:
                    brain_images = brain_images.to(self.device)
                    phenotypes = phenotypes.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(brain_images, phenotypes)
                    loss = criterion(outputs, targets)
                    baseline_loss += loss.item() * len(targets)
                    n_samples += len(targets)
            
            baseline_loss /= n_samples
            
            # 计算每个表型特征的重要性
            for feature_idx, feature_name in enumerate(phenotype_names):
                importance = []
                
                for _ in range(n_repeats):
                    permuted_loss = 0
                    n_samples = 0
                    
                    with torch.no_grad():
                        for brain_images, phenotypes, targets in val_loader:
                            brain_images = brain_images.to(self.device)
                            phenotypes = phenotypes.to(self.device)
                            targets = targets.to(self.device)
                            
                            # 打乱特定特征
                            permuted_phenotypes = phenotypes.clone()
                            perm_idx = torch.randperm(len(phenotypes))
                            permuted_phenotypes[:, feature_idx] = phenotypes[perm_idx, feature_idx]
                            
                            outputs = model(brain_images, permuted_phenotypes)
                            loss = criterion(outputs, targets)
                            permuted_loss += loss.item() * len(targets)
                            n_samples += len(targets)
                    
                    permuted_loss /= n_samples
                    importance.append(permuted_loss - baseline_loss)
                
                feature_importance[feature_name] = float(np.mean(importance))
            
            logging.info("Phenotype permutation importance calculation completed successfully")
            return feature_importance
            
        except Exception as e:
            logging.error(f"Error in calculating phenotype permutation importance: {str(e)}")
            return {}

    def _rank_features_by_permutation(self, perm_importance):
        """
        基于排列重要性对特征进行排序
        
        Parameters:
        -----------
        perm_importance : dict
            包含特征排列重要性的字典
        
        Returns:
        --------
        dict : 排序后的特征重要性字典
        """
        try:
            # 按重要性值大小排序
            sorted_features = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features)
        except Exception as e:
            logging.error(f"Error in ranking features by permutation: {str(e)}")
            return {}

    def _rank_by_permutation(self, perm_importance):
        """
        基于排列重要性对表型特征进行排序（与_rank_features_by_permutation功能相同，但用于表型特征）
        """
        return self._rank_features_by_permutation(perm_importance)  # 直接复用现有方法
    
    def _calculate_feature_correlations(self, train_loader):
        """
        计算特征之间的相关性矩阵
        
        Parameters:
        -----------
        train_loader : DataLoader
            训练数据加载器
        
        Returns:
        --------
        dict : 包含相关性矩阵和特征名称的字典
        """
        try:
            # 收集特征数据
            brain_features = []
            phenotype_features = []
            targets = []
            
            for brain_images, phenotypes, target in train_loader:
                # 提取脑部特征（对每个通道取平均）
                brain_feats = brain_images.numpy()
                brain_feats = brain_feats.reshape(brain_feats.shape[0], brain_feats.shape[1], -1)
                brain_feats = np.mean(brain_feats, axis=2)  # 对空间维度取平均
                brain_features.append(brain_feats)
                
                # 收集表型特征和目标值
                phenotype_features.append(phenotypes.numpy())
                targets.append(target.numpy())
            
            # 合并数据
            brain_features = np.vstack(brain_features)
            phenotype_features = np.vstack(phenotype_features)
            targets = np.vstack(targets)
            
            # 组合所有特征
            all_features = np.hstack([brain_features, phenotype_features, targets])
            
            # 特征名称
            feature_names = [
                'thickness', 'volume', 'surface_area', 'white_gray_ratio',  # 脑部特征
                'age', 'sex', 'education',  # 表型特征
                'attention', 'aggression'  # 目标变量
            ]
            
            # 计算相关性矩阵
            correlation_matrix = np.corrcoef(all_features.T)
            
            # 创建带有特征名称的结果字典
            result = {
                'correlation_matrix': correlation_matrix.tolist(),  # 转换为列表以便JSON序列化
                'feature_names': feature_names,
                'description': 'Pearson correlation matrix between all features and targets'
            }
            
            logging.info("Feature correlation calculation completed successfully")
            return result
        
        except Exception as e:
            logging.error(f"Error in calculating feature correlations: {str(e)}")
            return {
                'correlation_matrix': [],
                'feature_names': [],
                'description': f'Error occurred: {str(e)}'
            }

    def _rank_features_by_correlation(self, correlation_result):
        """
        基于特征与目标变量的相关性强度进行排序
        
        Parameters:
        -----------
        correlation_result : dict
            包含相关性矩阵和特征名称的字典
        
        Returns:
        --------
        dict : 排序后的特征重要性字典
        """
        try:
            matrix = np.array(correlation_result['correlation_matrix'])
            feature_names = correlation_result['feature_names']
            
            # 获取目标变量的索引
            attention_idx = feature_names.index('attention')
            aggression_idx = feature_names.index('aggression')
            
            # 计算每个特征与两个目标变量的平均绝对相关性
            feature_importance = {}
            n_features = len(feature_names) - 2  # 减去两个目标变量
            
            for i in range(n_features):
                # 计算与两个目标变量的平均绝对相关性
                mean_correlation = (abs(matrix[i, attention_idx]) + abs(matrix[i, aggression_idx])) / 2
                feature_importance[feature_names[i]] = float(mean_correlation)
            
            # 按相关性强度排序
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            result = {
                'feature_importance': dict(sorted_features),
                'description': 'Features ranked by average absolute correlation with targets'
            }
            
            logging.info("Feature correlation ranking completed successfully")
            return result
            
        except Exception as e:
            logging.error(f"Error in ranking features by correlation: {str(e)}")
            return {
                'feature_importance': {},
                'description': f'Error occurred: {str(e)}'
            }

    def _extract_brain_features(self, images):
        """
        从脑部图像中提取特征
        
        Parameters:
        -----------
        images : torch.Tensor
            脑部图像数据
            
        Returns:
        --------
        numpy.ndarray : 提取的特征矩阵
        """
        try:
            # 将图像数据转换为numpy数组
            img_data = images.cpu().numpy()
            
            # 对每个通道提取特征
            batch_size, n_channels, height, width = img_data.shape
            features = np.zeros((batch_size, n_channels))
            
            for i in range(batch_size):
                for c in range(n_channels):
                    channel_data = img_data[i, c]
                    
                    # 计算基本统计特征
                    features[i, c] = np.mean(channel_data)  # 可以添加其他统计量如std, max, min等
            
            return features
            
        except Exception as e:
            logging.error(f"Error in extracting brain features: {str(e)}")
            return np.array([])
    
    def _analyze_age_impact(self, model_output):
        """
        分析年龄对预测结果的影响
        
        Parameters:
        -----------
        model_output : dict
            模型输出结果，包含预测值和真实值
        
        Returns:
        --------
        dict : 年龄影响分析结果
        """
        try:
            predictions = np.array(model_output['predictions'])
            targets = np.array(model_output['targets'])
            
            # 从训练数据中获取年龄信息
            age_data = []
            for _, phenotypes, _ in self.model_trainer.model.train_loader:
                age_data.extend(phenotypes[:, 0].numpy())  # 年龄是第一个特征
            age_data = np.array(age_data)
            
            # 基本统计信息
            results = {
                'statistics': {
                    'mean_age': float(np.mean(age_data)),
                    'std_age': float(np.std(age_data)),
                    'median_age': float(np.median(age_data)),
                    'age_range': [float(np.min(age_data)), float(np.max(age_data))],
                    'quartiles': [float(np.percentile(age_data, q)) for q in [25, 50, 75]]
                }
            }
            
            # 使用四分位数进行分组分析
            # 基于四分位数进行分组，这样能更好地反映数据的实际分布：

            """ Youngest: 0-25%
                Young_Middle: 25-50%
                Middle_Upper: 50-75%
                Oldest: 75-100% """
            quartiles = np.percentile(age_data, [25, 50, 75])
            age_groups = [
                (float('-inf'), quartiles[0], 'Youngest'),
                (quartiles[0], quartiles[1], 'Young_Middle'),
                (quartiles[1], quartiles[2], 'Middle_Upper'),
                (quartiles[2], float('inf'), 'Oldest')
            ]
            
            results['age_group_performance'] = {}
            for min_age, max_age, group_name in age_groups:
                mask = (age_data >= min_age) & (age_data < max_age)
                if np.sum(mask) > 0:
                    group_predictions = predictions[mask]
                    group_targets = targets[mask]
                    group_ages = age_data[mask]
                    
                    group_results = {
                        'sample_count': int(np.sum(mask)),
                        'age_range': [float(np.min(group_ages)), float(np.max(group_ages))],
                        'mean_age': float(np.mean(group_ages)),
                        'performance': {
                            'mse': float(mean_squared_error(group_targets, group_predictions)),
                            'mae': float(mean_absolute_error(group_targets, group_predictions)),
                            'r2': float(r2_score(group_targets, group_predictions))
                        }
                    }
                    results['age_group_performance'][group_name] = group_results
            
            # 计算年龄与预测结果的相关性分析
            results['age_correlation'] = {
                'attention': {
                    'correlation': float(np.corrcoef(age_data, predictions[:, 0])[0, 1]),
                    'description': 'Correlation between age and attention prediction'
                },
                'aggression': {
                    'correlation': float(np.corrcoef(age_data, predictions[:, 1])[0, 1]),
                    'description': 'Correlation between age and aggression prediction'
                }
            }
            
            # 进行回归分析
            from scipy import stats
            
            # 对注意力的回归分析
            slope_att, intercept_att, r_att, p_att, std_err_att = stats.linregress(age_data, predictions[:, 0])
            results['age_regression_attention'] = {
                'slope': float(slope_att),
                'intercept': float(intercept_att),
                'r_value': float(r_att),
                'p_value': float(p_att),
                'std_error': float(std_err_att)
            }
            
            # 对攻击性的回归分析
            slope_agg, intercept_agg, r_agg, p_agg, std_err_agg = stats.linregress(age_data, predictions[:, 1])
            results['age_regression_aggression'] = {
                'slope': float(slope_agg),
                'intercept': float(intercept_agg),
                'r_value': float(r_agg),
                'p_value': float(p_agg),
                'std_error': float(std_err_agg)
            }
            
            # 检查非线性关系
            # 计算年龄的二次项与预测值的相关性
            age_squared = age_data ** 2
            results['nonlinear_analysis'] = {
                'attention': {
                    'quadratic_correlation': float(np.corrcoef(age_squared, predictions[:, 0])[0, 1])
                },
                'aggression': {
                    'quadratic_correlation': float(np.corrcoef(age_squared, predictions[:, 1])[0, 1])
                }
            }
            
            logging.info("Age impact analysis completed successfully")
            return results
            
        except Exception as e:
            logging.error(f"Error in analyzing age impact: {str(e)}")
            return {'error': str(e)}
            
    def _analyze_sex_impact(self, model_output):
        """
        分析性别对预测结果的影响
        
        Parameters:
        -----------
        model_output : dict
            模型输出结果，包含预测值和真实值
        
        Returns:
        --------
        dict : 性别影响分析结果
        """
        try:
            predictions = np.array(model_output['predictions'])
            targets = np.array(model_output['targets'])
            
            # 从训练数据中获取性别信息
            sex_data = []
            for _, phenotypes, _ in self.model_trainer.model.train_loader:
                sex_data.extend(phenotypes[:, 1].numpy())  # 性别是第二个特征
            sex_data = np.array(sex_data)
            
            results = {
                'gender_performance': {},
                'statistics': {
                    'gender_distribution': {
                        'female': int(np.sum(sex_data == 0)),
                        'male': int(np.sum(sex_data == 1))
                    }
                }
            }
            
            # 分别计算男性和女性的性能指标
            for sex_value, sex_name in [(0, 'female'), (1, 'male')]:
                mask = (sex_data == sex_value)
                if np.sum(mask) > 0:
                    sex_results = {
                        'sample_count': int(np.sum(mask)),
                        'mse': float(mean_squared_error(targets[mask], predictions[mask])),
                        'mae': float(mean_absolute_error(targets[mask], predictions[mask])),
                        'r2': float(r2_score(targets[mask], predictions[mask]))
                    }
                    results['gender_performance'][sex_name] = sex_results
            
            # 计算性别差异显著性
            female_mask = (sex_data == 0)
            male_mask = (sex_data == 1)
            
            if np.sum(male_mask) > 0 and np.sum(female_mask) > 0:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(
                    predictions[male_mask].mean(axis=1),
                    predictions[female_mask].mean(axis=1)
                )
                results['gender_difference'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value)
                }
            
            logging.info("Sex impact analysis completed successfully")
            return results
            
        except Exception as e:
            logging.error(f"Error in analyzing sex impact: {str(e)}")
            return {'error': str(e)}

    def _analyze_education_impact(self, model_output):
        """
        分析教育水平对预测结果的影响
        
        Parameters:
        -----------
        model_output : dict
            模型输出结果，包含预测值和真实值
        
        Returns:
        --------
        dict : 教育水平影响分析结果
        """
        try:
            predictions = np.array(model_output['predictions'])
            targets = np.array(model_output['targets'])
            
            # 从训练数据中获取教育水平信息
            edu_data = []
            for _, phenotypes, _ in self.model_trainer.model.train_loader:
                edu_data.extend(phenotypes[:, 2].numpy())  # 教育水平是第三个特征
            edu_data = np.array(edu_data)
            
            results = {
                'education_performance': {},
                'statistics': {
                    'education_distribution': {
                        'level_0': int(np.sum(edu_data == 0)),
                        'level_1': int(np.sum(edu_data == 1)),
                        'level_2': int(np.sum(edu_data == 2))
                    }
                }
            }
            
            # 计算各教育水平的性能指标
            for edu_value in [0, 1, 2]:
                mask = (edu_data == edu_value)
                if np.sum(mask) > 0:
                    edu_results = {
                        'sample_count': int(np.sum(mask)),
                        'mse': float(mean_squared_error(targets[mask], predictions[mask])),
                        'mae': float(mean_absolute_error(targets[mask], predictions[mask])),
                        'r2': float(r2_score(targets[mask], predictions[mask]))
                    }
                    results['education_performance'][f'level_{edu_value}'] = edu_results
            
            # 计算教育水平与预测结果的相关性
            results['education_correlation'] = {
                'attention': float(np.corrcoef(edu_data, predictions[:, 0])[0, 1]),
                'aggression': float(np.corrcoef(edu_data, predictions[:, 1])[0, 1])
            }
            
            # 进行方差分析
            from scipy import stats
            groups = [predictions[edu_data == i].mean(axis=1) for i in [0, 1, 2]]
            f_stat, p_value = stats.f_oneway(*groups)
            
            results['anova_test'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value)
            }
            
            logging.info("Education impact analysis completed successfully")
            return results
            
        except Exception as e:
            logging.error(f"Error in analyzing education impact: {str(e)}")
            return {'error': str(e)}

    def _calculate_effect_sizes(self, predictions, targets):
        """
        计算预测效应值
        
        Parameters:
        -----------
        predictions : np.ndarray
            模型预测值
        targets : np.ndarray
            真实目标值
            
        Returns:
        --------
        dict : 包含各种效应值度量的字典
        """
        try:
            from scipy import stats
            
            # 计算Cohen's d
            def cohen_d(group1, group2):
                n1, n2 = len(group1), len(group2)
                var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
                pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                return (np.mean(group1) - np.mean(group2)) / pooled_se if pooled_se != 0 else 0

            # 计算eta squared
            def eta_squared(predictions, targets):
                ss_total = np.sum((targets - np.mean(targets)) ** 2)
                ss_residual = np.sum((targets - predictions) ** 2)
                return 1 - (ss_residual / ss_total) if ss_total != 0 else 0

            results = {}
            
            # 对每个目标变量分别计算效应值
            for i, target_name in enumerate(['attention', 'aggression']):
                target_pred = predictions[:, i]
                target_true = targets[:, i]
                
                # Cohen's d
                d = cohen_d(target_pred, target_true)
                
                # Eta squared
                eta_sq = eta_squared(target_pred, target_true)
                
                # 相关系数
                correlation, p_value = stats.pearsonr(target_pred, target_true)
                
                # R squared (决定系数)
                r_squared = r2_score(target_true, target_pred)
                
                results[target_name] = {
                    'cohens_d': float(d),
                    'eta_squared': float(eta_sq),
                    'correlation': float(correlation),
                    'correlation_p_value': float(p_value),
                    'r_squared': float(r_squared)
                }
                
            # 计算整体效应
            overall_pred = predictions.mean(axis=1)
            overall_true = targets.mean(axis=1)
            
            results['overall'] = {
                'cohens_d': float(cohen_d(overall_pred, overall_true)),
                'eta_squared': float(eta_squared(overall_pred, overall_true)),
                'correlation': float(stats.pearsonr(overall_pred, overall_true)[0]),
                'r_squared': float(r2_score(overall_true, overall_pred))
            }
            
            logging.info("Effect size calculation completed successfully")
            return results
            
        except Exception as e:
            logging.error(f"Error in calculating effect sizes: {str(e)}")
            return {'error': str(e)}

    def _calculate_manova_effect_sizes(self, manova_result):
        """
        从MANOVA结果中计算效应值
        
        Parameters:
        -----------
        manova_result : MANOVA
            MANOVA分析结果对象
            
        Returns:
        --------
        dict : 包含MANOVA效应值的字典
        """
        try:
            results = {}
            test_results = manova_result.mv_test()
            
            for predictor, stats in test_results.items():
                if predictor != 'Intercept':
                    # Wilks' Lambda
                    wilks_lambda = stats['stat'].iloc[0]
                    F_stat = stats['stat'].iloc[1]
                    df = (stats['df'].iloc[0], stats['df'].iloc[1])
                    p_value = stats['P>F'].iloc[0]
                    
                    # Partial Eta Squared
                    partial_eta_sq = 1 - wilks_lambda
                    
                    # Effect size conversion (from Wilks' Lambda to Cohen's f)
                    cohens_f = np.sqrt(partial_eta_sq / (1 - partial_eta_sq))
                    
                    results[predictor] = {
                        'wilks_lambda': float(wilks_lambda),
                        'F_statistic': float(F_stat),
                        'degrees_of_freedom': [int(df[0]), int(df[1])],
                        'p_value': float(p_value),
                        'partial_eta_squared': float(partial_eta_sq),
                        'cohens_f': float(cohens_f)
                    }
            
            logging.info("MANOVA effect size calculation completed successfully")
            return results
            
        except Exception as e:
            logging.error(f"Error in calculating MANOVA effect sizes: {str(e)}")
            return {'error': str(e)}


    def _plot_phenotype_evaluation(self, results):
        """
        绘制表型特征评估的可视化图表
        
        Parameters:
        -----------
        results : dict
            表型特征评估结果
        """
        try:
            plot_dir = self.results_dir / 'phenotype_features/complete_evaluation/plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 性能对比条形图
            plt.figure(figsize=(12, 6))
            metrics = ['mse', 'mae', 'r2']
            combinations = list(results.keys())
            
            for i, metric in enumerate(metrics, 1):
                plt.subplot(1, 3, i)
                performance_values = [results[comb]['metrics'][metric] for comb in combinations]
                plt.bar(combinations, performance_values)
                plt.title(f'{metric.upper()} by Feature Combination')
                plt.xticks(rotation=45)
                plt.ylabel(metric.upper())
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 效应值热图
            plt.figure(figsize=(10, 8))
            effect_sizes = {}
            for comb in combinations:
                if 'effect_sizes' in results[comb]:
                    effect_sizes[comb] = results[comb]['effect_sizes']['overall']
            
            if effect_sizes:
                effect_df = pd.DataFrame(effect_sizes).T
                sns.heatmap(effect_df, annot=True, cmap='YlOrRd', fmt='.3f')
                plt.title('Effect Sizes Across Feature Combinations')
                plt.tight_layout()
                plt.savefig(plot_dir / 'effect_sizes_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 学习曲线
            plt.figure(figsize=(10, 5))
            for comb in combinations:
                if 'training_history' in results[comb]:
                    history = results[comb]['training_history']
                    plt.plot(history['train_loss'], label=f'{comb}_train')
                    plt.plot(history['val_loss'], label=f'{comb}_val')
            
            plt.title('Learning Curves Across Feature Combinations')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Phenotype evaluation plots generated successfully")
            
        except Exception as e:
            logging.error(f"Error in generating phenotype evaluation plots: {str(e)}")

    def _plot_individual_phenotype_analysis(self, results):
        """
        绘制单个表型特征分析的可视化图表
        
        Parameters:
        -----------
        results : dict
            单个表型特征分析结果
        """
        try:
            plot_dir = self.results_dir / 'phenotype_features/individual_analysis/plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 性能比较图
            plt.figure(figsize=(12, 6))
            features = list(results.keys())
            metrics = ['mse', 'mae', 'r2']
            
            for i, metric in enumerate(metrics, 1):
                plt.subplot(1, 3, i)
                values = [results[feat]['metrics'][metric] for feat in features]
                plt.bar(features, values)
                plt.title(f'{metric.upper()} by Feature')
                plt.xticks(rotation=45)
                plt.ylabel(metric.upper())
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'individual_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 特征特定分析图
            for feature in features:
                if 'feature_specific' in results[feature]:
                    spec_data = results[feature]['feature_specific']
                    
                    if feature == 'B1_plus_D5':  # 年龄分析
                        self._plot_age_specific_analysis(spec_data, plot_dir)
                    elif feature == 'B1_plus_D6':  # 性别分析
                        self._plot_sex_specific_analysis(spec_data, plot_dir)
                    elif feature == 'B1_plus_D7':  # 教育水平分析
                        self._plot_education_specific_analysis(spec_data, plot_dir)
            
            logging.info("Individual phenotype analysis plots generated successfully")
            
        except Exception as e:
            logging.error(f"Error in generating individual phenotype analysis plots: {str(e)}")

    def _plot_phenotype_importance(self, results):
        """
        绘制表型特征重要性的可视化图表
        
        Parameters:
        -----------
        results : dict
            表型特征重要性分析结果
        """
        try:
            plot_dir = self.results_dir / 'phenotype_features/importance_quantification/plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. SHAP值条形图
            plt.figure(figsize=(10, 6))
            if 'shap_analysis' in results and 'importance_ranking' in results['shap_analysis']:
                shap_values = results['shap_analysis']['importance_ranking']
                features = list(shap_values.keys())
                values = list(shap_values.values())
                
                plt.barh(features, values)
                plt.title('Feature Importance (SHAP Values)')
                plt.xlabel('SHAP Value')
                plt.tight_layout()
                plt.savefig(plot_dir / 'shap_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 相对重要性饼图
            plt.figure(figsize=(10, 10))
            if 'relative_importance' in results:
                rel_importance = results['relative_importance']
                plt.pie(rel_importance.values(), labels=rel_importance.keys(), autopct='%1.1f%%')
                plt.title('Relative Feature Importance')
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(plot_dir / 'relative_importance_pie.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 排列重要性箱线图
            plt.figure(figsize=(10, 6))
            if 'permutation_importance' in results and 'scores' in results['permutation_importance']:
                perm_importance = results['permutation_importance']['scores']
                
                if isinstance(perm_importance, dict):  # 如果是多次重复的结果
                    df_perm = pd.DataFrame(perm_importance)
                    sns.boxplot(data=df_perm)
                    plt.title('Permutation Importance Distribution')
                    plt.xticks(rotation=45)
                    plt.ylabel('Importance Score')
                    plt.tight_layout()
                    plt.savefig(plot_dir / 'permutation_importance_dist.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Phenotype importance plots generated successfully")
            
        except Exception as e:
            logging.error(f"Error in generating phenotype importance plots: {str(e)}")

    def _plot_phenotype_interactions(self, results):
        """
        绘制表型特征交互关系的可视化图表
        
        Parameters:
        -----------
        results : dict
            表型特征交互分析结果
        """
        try:
            plot_dir = self.results_dir / 'interactions/manova_results/plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. MANOVA效应值热图
            if 'effect_sizes' in results:
                plt.figure(figsize=(10, 8))
                effect_df = pd.DataFrame(results['effect_sizes'])
                sns.heatmap(effect_df, annot=True, cmap='YlOrRd', fmt='.3f')
                plt.title('MANOVA Effect Sizes')
                plt.tight_layout()
                plt.savefig(plot_dir / 'manova_effect_sizes.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. 描述性统计可视化
            if 'descriptive_stats' in results:
                stats_df = results['descriptive_stats']
                
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=stats_df)
                plt.title('Distribution of Features and Targets')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plot_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
                plt.close()

            # 3. Wilks Lambda可视化
            if 'test_results' in results:
                plt.figure(figsize=(10, 6))
                test_results = results['test_results']
                wilks_data = []
                for predictor, data in test_results.items():
                    if predictor != 'Intercept':
                        wilks_data.append({
                            'predictor': predictor,
                            'wilks': data['stat'].iloc[0],
                            'p_value': data['P>F'].iloc[0]
                        })
                
                df_wilks = pd.DataFrame(wilks_data)
                sns.barplot(data=df_wilks, x='predictor', y='wilks')
                plt.title("Wilks' Lambda by Predictor")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plot_dir / 'wilks_lambda.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logging.info("Phenotype interaction plots generated successfully")
            
        except Exception as e:
            logging.error(f"Error in generating phenotype interaction plots: {str(e)}")

    # 辅助函数：绘制年龄特定分析图
    def _plot_age_specific_analysis(self, spec_data, plot_dir):
        """绘制年龄特定分析图"""
        try:
            # 1. 年龄组性能比较
            plt.figure(figsize=(10, 6))
            age_groups = list(spec_data['age_group_performance'].keys())
            performance = [spec_data['age_group_performance'][g]['performance']['r2'] 
                        for g in age_groups]
            
            plt.bar(age_groups, performance)
            plt.title('Model Performance by Age Group')
            plt.ylabel('R² Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plot_dir / 'age_group_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 年龄相关性散点图
            plt.figure(figsize=(12, 5))
            
            # 注意力预测相关性
            plt.subplot(1, 2, 1)
            plt.scatter(spec_data['age_regression_attention']['x'], 
                    spec_data['age_regression_attention']['y'])
            plt.title('Age vs. Attention Prediction')
            plt.xlabel('Age')
            plt.ylabel('Predicted Attention')
            
            # 攻击性预测相关性
            plt.subplot(1, 2, 2)
            plt.scatter(spec_data['age_regression_aggression']['x'], 
                    spec_data['age_regression_aggression']['y'])
            plt.title('Age vs. Aggression Prediction')
            plt.xlabel('Age')
            plt.ylabel('Predicted Aggression')
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'age_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error in plotting age specific analysis: {str(e)}")

    # 辅助函数：绘制性别特定分析图
    def _plot_sex_specific_analysis(self, spec_data, plot_dir):
        """绘制性别特定分析图"""
        try:
            # 1. 性别组性能比较
            plt.figure(figsize=(10, 6))
            genders = list(spec_data['gender_performance'].keys())
            performance = [spec_data['gender_performance'][g]['mse'] for g in genders]
            
            plt.bar(genders, performance)
            plt.title('Model Performance by Gender')
            plt.ylabel('Mean Squared Error')
            plt.tight_layout()
            plt.savefig(plot_dir / 'gender_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 性别差异箱线图
            if 'predictions_by_gender' in spec_data:
                plt.figure(figsize=(10, 6))
                data = spec_data['predictions_by_gender']
                df = pd.DataFrame(data)
                
                sns.boxplot(x='gender', y='prediction', data=df)
                plt.title('Prediction Distribution by Gender')
                plt.tight_layout()
                plt.savefig(plot_dir / 'gender_predictions.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logging.error(f"Error in plotting sex specific analysis: {str(e)}")


    # 辅助函数：绘制教育水平特定分析图
    def _plot_education_specific_analysis(self, spec_data, plot_dir):
        """绘制教育水平特定分析图"""
        try:
            # 1. 教育水平组性能比较
            plt.figure(figsize=(10, 6))
            edu_levels = list(spec_data['education_performance'].keys())
            performance_metrics = ['mse', 'mae', 'r2']
            
            for i, metric in enumerate(performance_metrics):
                plt.subplot(1, 3, i+1)
                values = [spec_data['education_performance'][level][metric] 
                        for level in edu_levels]
                plt.bar(edu_levels, values)
                plt.title(f'{metric.upper()} by Education Level')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'education_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 教育水平相关性分析
            plt.figure(figsize=(12, 5))
            
            # 注意力预测相关性
            plt.subplot(1, 2, 1)
            if 'education_correlation' in spec_data:
                corr_attention = spec_data['education_correlation']['attention']
                plt.bar(['Attention'], [corr_attention])
                plt.title('Education-Attention Correlation')
                plt.ylabel('Correlation Coefficient')
            
            # 攻击性预测相关性
            plt.subplot(1, 2, 2)
            if 'education_correlation' in spec_data:
                corr_aggression = spec_data['education_correlation']['aggression']
                plt.bar(['Aggression'], [corr_aggression])
                plt.title('Education-Aggression Correlation')
                plt.ylabel('Correlation Coefficient')
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'education_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. ANOVA 结果可视化
            if 'anova_test' in spec_data:
                plt.figure(figsize=(10, 6))
                anova_results = spec_data['anova_test']
                
                # 创建条形图显示F统计量和p值
                metrics = ['f_statistic', 'p_value']
                values = [anova_results[metric] for metric in metrics]
                
                plt.bar(metrics, values)
                plt.title('ANOVA Test Results')
                plt.xticks(rotation=45)
                
                # 添加显著性水平参考线
                if 'p_value' in metrics:
                    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(plot_dir / 'education_anova.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            logging.info("Education specific analysis plots generated successfully")
            
        except Exception as e:
            logging.error(f"Error in plotting education specific analysis: {str(e)}")
            raise

    def _plot_cortical_interactions(self, results):
        """
        绘制皮层特征之间的交互关系可视化图表
        
        Parameters:
        -----------
        results : dict
            PCA分析结果
        """
        try:
            plot_dir = self.results_dir / 'interactions/pca_results/plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 解释方差比率条形图
            plt.figure(figsize=(10, 6))
            explained_var = results['explained_variance_ratio_']
            components = [f'PC{i+1}' for i in range(len(explained_var))]
            
            plt.bar(components, explained_var * 100)
            plt.title('Explained Variance Ratio by Principal Components')
            plt.xlabel('Principal Components')
            plt.ylabel('Explained Variance (%)')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # 添加累积解释方差率
            cum_explained_var = np.cumsum(explained_var) * 100
            plt.plot(components, cum_explained_var, 'r-o', label='Cumulative')
            plt.axhline(y=80, color='r', linestyle='--', alpha=0.5)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'explained_variance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 主成分载荷热图
            plt.figure(figsize=(10, 8))
            components_df = pd.DataFrame(
                results['components_'],
                columns=['thickness', 'volume', 'surface_area', 'white_gray_ratio'],
                index=[f'PC{i+1}' for i in range(results['components_'].shape[0])]
            )
            
            sns.heatmap(components_df, annot=True, cmap='RdBu_r', center=0)
            plt.title('Principal Component Loadings')
            plt.tight_layout()
            plt.savefig(plot_dir / 'component_loadings.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 双变量散点图（展示前两个主成分）
            if results['transformed_data'] is not None:
                plt.figure(figsize=(10, 8))
                transformed_data = results['transformed_data']
                plt.scatter(transformed_data[:, 0], transformed_data[:, 1], alpha=0.5)
                plt.xlabel('First Principal Component')
                plt.ylabel('Second Principal Component')
                plt.title('Data Distribution in PC1-PC2 Space')
                plt.grid(True, alpha=0.3)
                
                # 添加可解释方差百分比到轴标签
                var_exp = results['explained_variance_ratio_']
                plt.xlabel(f'PC1 ({var_exp[0]*100:.1f}%)')
                plt.ylabel(f'PC2 ({var_exp[1]*100:.1f}%)')
                
                plt.tight_layout()
                plt.savefig(plot_dir / 'pc_scatter.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            logging.info("Cortical interactions plots generated successfully")
            
        except Exception as e:
            logging.error(f"Error in generating cortical interactions plots: {str(e)}")

    def _plot_cross_domain_interactions(self, results):
        """
        绘制表型特征和皮层特征之间的交互关系可视化图表
        
        Parameters:
        -----------
        results : dict
            CCA分析结果
        """
        try:
            plot_dir = self.results_dir / 'interactions/cca_results/plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 典型相关系数条形图
            plt.figure(figsize=(10, 6))
            correlations = results['correlations']
            components = [f'CC{i+1}' for i in range(len(correlations))]
            
            plt.bar(components, correlations)
            plt.title('Canonical Correlations')
            plt.xlabel('Canonical Components')
            plt.ylabel('Correlation Coefficient')
            plt.grid(True, alpha=0.3)
            
            # 添加显著性水平参考线
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Medium correlation')
            plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Strong correlation')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'canonical_correlations.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 权重热图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Brain weights
            brain_weights_df = pd.DataFrame(
                results['brain_weights'],
                columns=['thickness', 'volume', 'surface_area', 'white_gray_ratio'],
                index=[f'CC{i+1}' for i in range(results['brain_weights'].shape[0])]
            )
            sns.heatmap(brain_weights_df, ax=ax1, annot=True, cmap='RdBu_r', center=0)
            ax1.set_title('Brain Feature Weights')
            
            # Phenotype weights
            pheno_weights_df = pd.DataFrame(
                results['phenotype_weights'],
                columns=['age', 'sex', 'education'],
                index=[f'CC{i+1}' for i in range(results['phenotype_weights'].shape[0])]
            )
            sns.heatmap(pheno_weights_df, ax=ax2, annot=True, cmap='RdBu_r', center=0)
            ax2.set_title('Phenotype Feature Weights')
            
            plt.tight_layout()
            plt.savefig(plot_dir / 'canonical_weights.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 典型变量散点图（第一对典型变量）
            if len(correlations) > 0:
                plt.figure(figsize=(10, 8))
                brain_scores = results['brain_components'][:, 0]
                pheno_scores = results['phenotype_components'][:, 0]
                
                plt.scatter(brain_scores, pheno_scores, alpha=0.5)
                plt.xlabel('Brain Canonical Variable 1')
                plt.ylabel('Phenotype Canonical Variable 1')
                plt.title(f'First Canonical Correlation (r = {correlations[0]:.3f})')
                
                # 添加趋势线
                z = np.polyfit(brain_scores, pheno_scores, 1)
                p = np.poly1d(z)
                plt.plot(brain_scores, p(brain_scores), "r--", alpha=0.8)
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir / 'canonical_scatter.png', dpi=300)
                plt.close()
                
            logging.info("Cross-domain interactions plots generated successfully")
        except Exception as e:
            logging.error(f"Error in generating cross-domain interactions plots: {str(e)}")


    def _save_phenotype_analysis_results(self, results):
        """
        保存表型特征分析结果
        
        Parameters:
        -----------
        results : dict
            表型特征分析结果
        """
        try:
            # 创建保存路径
            save_path = self.results_dir / 'phenotype_features'

            serializable_results = self._convert_to_serializable(results)
            
            # 1. 保存完整评估结果
            complete_eval_path = save_path / 'complete_evaluation'
            complete_eval_path.mkdir(parents=True, exist_ok=True)
            
            with open(complete_eval_path / 'results.json', 'w') as f:
                json.dump(serializable_results['complete_evaluation'], f, indent=4)
                
            # 2. 保存个体特征分析结果
            individual_path = save_path / 'individual_analysis'
            individual_path.mkdir(parents=True, exist_ok=True)
            
            with open(individual_path / 'results.json', 'w') as f:
                json.dump(serializable_results['individual_analysis'], f, indent=4)
                
            # 3. 保存重要性量化结果
            importance_path = save_path / 'importance_quantification'
            importance_path.mkdir(parents=True, exist_ok=True)
            
            with open(importance_path / 'results.json', 'w') as f:
                json.dump(serializable_results['importance_quantification'], f, indent=4)
                
            # 保存元数据
            metadata = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'analysis_version': '1.0',
                'feature_groups': list(self.phenotype_features.keys()),
                'metrics_used': ['mse', 'mae', 'r2', 'correlation', 'effect_size']
            }
            
            with open(save_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logging.info(f"Phenotype feature analysis results saved to: {save_path}")
            
        except Exception as e:
            logging.error(f"Error in saving phenotype analysis results: {str(e)}")

    def _save_interaction_analysis_results(self, results):
        """
        保存特征交互分析结果
        
        Parameters:
        -----------
        results : dict
            特征交互分析结果
        """
        try:
            # 创建保存路径
            save_path = self.results_dir / 'interactions'

            serializable_results = self._convert_to_serializable(results)
            
            # 1. 保存MANOVA结果
            manova_path = save_path / 'manova_results'
            manova_path.mkdir(parents=True, exist_ok=True)
            
            with open(manova_path / 'results.json', 'w') as f:
                json.dump(serializable_results['phenotypic_interactions'], f, indent=4)
                
            # 2. 保存PCA结果
            pca_path = save_path / 'pca_results'
            pca_path.mkdir(parents=True, exist_ok=True)
            
            # 转换numpy数组为列表以便JSON序列化
            pca_results = {
                'explained_variance_ratio': serializable_results['cortical_interactions']['explained_variance_ratio_'].tolist(),
                'components': serializable_results['cortical_interactions']['components_'].tolist()
            }
            
            with open(pca_path / 'results.json', 'w') as f:
                json.dump(pca_results, f, indent=4)
                
            # 3. 保存CCA结果
            cca_path = save_path / 'cca_results'
            cca_path.mkdir(parents=True, exist_ok=True)
            
            # 转换numpy数组为列表
            cca_results = {
                'correlations': serializable_results['cross_domain_interactions']['correlations'].tolist(),
                'brain_weights': serializable_results['cross_domain_interactions']['brain_weights'].tolist(),
                'phenotype_weights': serializable_results['cross_domain_interactions']['phenotype_weights'].tolist()
            }
            
            with open(cca_path / 'results.json', 'w') as f:
                json.dump(cca_results, f, indent=4)
                
            # 保存元数据
            metadata = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'analysis_version': '1.0',
                'methods_used': ['MANOVA', 'PCA', 'CCA'],
                'significance_level': 0.05
            }
            
            with open(save_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logging.info(f"Feature interaction analysis results saved to: {save_path}")
            
        except Exception as e:
            logging.error(f"Error in saving interaction analysis results: {str(e)}")


    def _write_brain_feature_summary(self, file, results):
        """
        写入脑部特征分析总结
        
        Parameters:
        -----------
        file : file object
            输出文件对象
        results : dict
            脑部特征分析结果
        """
        try:
            # 1.1 Complete evaluation
            file.write("\n1.1 Complete Brain Feature Evaluation\n")
            file.write("-" * 40 + "\n")
            
            eval_results = results.get('individual_features_analysis', {})
            for comb, result in eval_results.items():
                file.write(f"\nCombination: {comb}\n")
                metrics = result.get('metrics', {})
                file.write(f"MSE: {metrics.get('mse', 'N/A'):.4f}\n")
                file.write(f"MAE: {metrics.get('mae', 'N/A'):.4f}\n")
                file.write(f"R2 Score: {metrics.get('r2', 'N/A'):.4f}\n")
                if 'p_values' in metrics:
                    file.write(f"P-value: {metrics['p_values']:.4f}\n")
                
            # 1.2 Individual analysis
            file.write("\n1.2 Individual Feature Combination Analysis\n")
            file.write("-" * 40 + "\n")
            
            comb_results = results.get('combined_feature_analysis', {})
            for comb, result in comb_results.items():
                file.write(f"\nFeature Combination: {comb}\n")
                metrics = result.get('metrics', {})
                file.write(f"MSE: {metrics.get('mse', 'N/A'):.4f}\n")
                file.write(f"MAE: {metrics.get('mae', 'N/A'):.4f}\n")
                file.write(f"R2 Score: {metrics.get('r2', 'N/A'):.4f}\n")
                
                importance = result.get('feature_importance', {})
                if importance and 'importance_scores' in importance:
                    file.write("\nFeature Importance Scores:\n")
                    for feature, score in importance['importance_scores'].items():
                        file.write(f"{feature}: {score:.4f}\n")
            
            # 1.3 Feature importance ranking
            file.write("\n1.3 Feature Importance Ranking\n")
            file.write("-" * 40 + "\n")
            
            importance = results.get('importance_ranking', {})
            
            # SHAP values
            if 'shap_values' in importance:
                file.write("\nSHAP Value Rankings:\n")
                for feature, value in importance['shap_values'].get('ranking', {}).items():
                    file.write(f"{feature}: {value:.4f}\n")
            
            # Permutation importance
            if 'permutation_importance' in importance:
                file.write("\nPermutation Importance Rankings:\n")
                for feature, value in importance['permutation_importance'].get('ranking', {}).items():
                    file.write(f"{feature}: {value:.4f}\n")
            
            # Correlation analysis
            if 'correlation_analysis' in importance:
                file.write("\nCorrelation-based Rankings:\n")
                corr_results = importance['correlation_analysis'].get('ranking', {}).get('feature_importance', {})
                for feature, value in corr_results.items():
                    file.write(f"{feature}: {value:.4f}\n")
                    
        except Exception as e:
            logging.error(f"Error in writing brain feature summary: {str(e)}")
            file.write("\nError occurred while writing brain feature summary\n")

    def _write_phenotype_feature_summary(self, file, results):
        """
        写入表型特征分析总结
        
        Parameters:
        -----------
        file : file object
            输出文件对象
        results : dict
            表型特征分析结果
        """
        try:
            # 2.1 Complete evaluation
            file.write("\n2.1 Complete Phenotype Group Evaluation\n")
            file.write("-" * 40 + "\n")
            
            complete_eval = results.get('complete_evaluation', {})
            for group, result in complete_eval.items():
                file.write(f"\nGroup: {group}\n")
                metrics = result.get('metrics', {})
                file.write(f"MSE: {metrics.get('mse', 'N/A'):.4f}\n")
                file.write(f"MAE: {metrics.get('mae', 'N/A'):.4f}\n")
                file.write(f"R2 Score: {metrics.get('r2', 'N/A'):.4f}\n")
                
                if 'effect_sizes' in result:
                    file.write("\nEffect Sizes:\n")
                    for target, effects in result['effect_sizes'].items():
                        file.write(f"{target}:\n")
                        for metric, value in effects.items():
                            file.write(f"  {metric}: {value:.4f}\n")
            
            # 2.2 Individual feature analysis
            file.write("\n2.2 Individual Feature Analysis\n")
            file.write("-" * 40 + "\n")
            
            indiv_analysis = results.get('individual_analysis', {})
            for feature, result in indiv_analysis.items():
                file.write(f"\nFeature: {feature}\n")
                metrics = result.get('metrics', {})
                file.write(f"MSE: {metrics.get('mse', 'N/A'):.4f}\n")
                file.write(f"MAE: {metrics.get('mae', 'N/A'):.4f}\n")
                file.write(f"R2 Score: {metrics.get('r2', 'N/A'):.4f}\n")
                
                if 'feature_specific' in result:
                    spec_data = result['feature_specific']
                    file.write("\nFeature-specific Analysis:\n")
                    if isinstance(spec_data, dict):
                        for key, value in spec_data.items():
                            if isinstance(value, (int, float)):
                                file.write(f"{key}: {value:.4f}\n")
                            elif isinstance(value, dict):
                                file.write(f"{key}:\n")
                                for k, v in value.items():
                                    file.write(f"  {k}: {v}\n")
            
            # 2.3 Feature importance quantification
            file.write("\n2.3 Feature Importance Quantification\n")
            file.write("-" * 40 + "\n")
            
            importance = results.get('importance_quantification', {})
            
            # SHAP analysis
            if 'shap_analysis' in importance:
                file.write("\nSHAP Analysis Results:\n")
                for feature, value in importance['shap_analysis'].get('importance_ranking', {}).items():
                    file.write(f"{feature}: {value:.4f}\n")
            
            # Relative importance
            if 'relative_importance' in importance:
                file.write("\nRelative Feature Importance:\n")
                for feature, value in importance['relative_importance'].items():
                    file.write(f"{feature}: {value:.2f}%\n")
                    
        except Exception as e:
            logging.error(f"Error in writing phenotype feature summary: {str(e)}")
            file.write("\nError occurred while writing phenotype feature summary\n")

    def _write_interaction_summary(self, file, results):
        """
        写入特征交互分析总结
        
        Parameters:
        -----------
        file : file object
            输出文件对象
        results : dict
            特征交互分析结果
        """
        try:
            # 3.1 Phenotypic feature interactions
            file.write("\n3.1 Phenotypic Feature Interactions (MANOVA)\n")
            file.write("-" * 40 + "\n")
            
            pheno_inter = results.get('phenotypic_interactions', {})
            
            if 'test_results' in pheno_inter:
                file.write("\nMANOVA Test Results:\n")
                test_results = pheno_inter['test_results']
                for predictor, stats in test_results.items():
                    if predictor != 'Intercept':
                        file.write(f"\nPredictor: {predictor}\n")
                        file.write(f"Wilk's Lambda: {stats['stat'].iloc[0]:.4f}\n")
                        file.write(f"F-statistic: {stats['stat'].iloc[1]:.4f}\n")
                        file.write(f"p-value: {stats['P>F'].iloc[0]:.4f}\n")
            
            if 'effect_sizes' in pheno_inter:
                file.write("\nEffect Sizes:\n")
                for predictor, effects in pheno_inter['effect_sizes'].items():
                    file.write(f"\n{predictor}:\n")
                    for metric, value in effects.items():
                        file.write(f"{metric}: {value:.4f}\n")
            
            # 3.2 Cortical feature interactions
            file.write("\n3.2 Cortical Feature Interactions (PCA)\n")
            file.write("-" * 40 + "\n")
            
            cortical_inter = results.get('cortical_interactions', {})
            
            if 'explained_variance_ratio' in cortical_inter:
                file.write("\nExplained Variance Ratios:\n")
                var_ratio = cortical_inter['explained_variance_ratio_']
                for i, ratio in enumerate(var_ratio):
                    file.write(f"PC{i+1}: {ratio*100:.2f}%\n")
                file.write(f"\nCumulative Variance Explained: {np.sum(var_ratio)*100:.2f}%\n")
            
            if 'components' in cortical_inter:
                file.write("\nPrincipal Components:\n")
                components = cortical_inter['components_']
                feature_names = ['thickness', 'volume', 'surface_area', 'white_gray_ratio']
                for i, comp in enumerate(components):
                    file.write(f"\nPC{i+1} loadings:\n")
                    for j, feat in enumerate(feature_names):
                        file.write(f"{feat}: {comp[j]:.4f}\n")
            
            # 3.3 Cross-domain feature interactions
            file.write("\n3.3 Cross-domain Feature Interactions (CCA)\n")
            file.write("-" * 40 + "\n")
            
            cross_domain = results.get('cross_domain_interactions', {})
            
            if 'correlations' in cross_domain:
                file.write("\nCanonical Correlations:\n")
                for i, corr in enumerate(cross_domain['correlations']):
                    file.write(f"CC{i+1}: {corr:.4f}\n")
            
            if 'brain_weights' in cross_domain and 'phenotype_weights' in cross_domain:
                file.write("\nFeature Weights:\n")
                file.write("\nBrain Features:\n")
                brain_feat = ['thickness', 'volume', 'surface_area', 'white_gray_ratio']
                brain_weights = cross_domain['brain_weights']
                for i, weights in enumerate(brain_weights):
                    file.write(f"\nCC{i+1}:\n")
                    for feat, weight in zip(brain_feat, weights):
                        file.write(f"{feat}: {weight:.4f}\n")
                
                file.write("\nPhenotype Features:\n")
                pheno_feat = ['age', 'sex', 'education']
                pheno_weights = cross_domain['phenotype_weights']
                for i, weights in enumerate(pheno_weights):
                    file.write(f"\nCC{i+1}:\n")
                    for feat, weight in zip(pheno_feat, weights):
                        file.write(f"{feat}: {weight:.4f}\n")
                    
        except Exception as e:
            logging.error(f"Error in writing interaction summary: {str(e)}")
            file.write("\nError occurred while writing interaction summary\n")


    def _analyze_individual_brain_features(self, train_loader, val_loader):
        """
        1.1 brain individual features evaluation
        评估完整脑部特征组的表现（([B2 + D1], [B3 + D1], [B4 + D1])
        注意B1+D1直接看原本的训练结果
        """
        logging.info("1.1 Analysis complete brain feature group evaluation...")
        results = {}
        
        # 分析组合：[B2 + D1], [B3 + D1], [B4 + D1]
        feature_combinations = [
            ('B2_D1', self.brain_features['B2']['indices']),  # thickness_area + all phenotype
            ('B3_D1', self.brain_features['B3']['indices']),  # volume + all phenotype
            ('B4_D1', self.brain_features['B4']['indices'])   # white_gray_ratio + all phenotype
        ]
    
        
        for comb_name, indices in feature_combinations:
            # 评估该特征组合
            model_output = self._train_and_evaluate(
                train_loader,
                val_loader,
                indices,
                self.phenotype_features['D1']['indices']
            )
            
            # 计算和保存结果
            stats = self._calculate_statistics(
                model_output['predictions'],
                model_output['targets']
            )
            
            results[comb_name] = {
                'metrics': stats,
                'feature_importance': self._calculate_feature_importance(
                    model_output['model'],
                    indices
                ),
                'training_history': model_output['history'],
                'model_path': model_output['model_path']
            }

        # 生成比较可视化
        self._plot_feature_combination_comparison(results)
        
        # 清理内存
        MemoryManager.cleanup()
        
        return results



    def _analyze_brain_feature_combinations(self, train_loader, val_loader):
        """
        1.2 brain feature combination analysis
        分析组合特征的贡献 [(B2 + B4) + D1], [(B3 + B4) + D1]
        """
        logging.info("1.2 Analysing individual feature combination analysis (thickness + area, volume, white-to-gray ratio)...")
        results = {}
        
        # 分析组合：[(B2+B4), (B3+B4)]
        # 使用列表而不是numpy数组来组合索引
        feature_combinations = [
            ('B2B4_D1', list(self.brain_features['B2']['indices']) + [self.brain_features['B4']['indices']]),
            ('B3B4_D1', list(self.brain_features['B3']['indices']) + [self.brain_features['B4']['indices']])
        ]
        
        for comb_name, indices in feature_combinations:
            # 如果indices是嵌套列表，将其展平
            if any(isinstance(x, list) for x in indices):
                flat_indices = []
                for idx in indices:
                    if isinstance(idx, list):
                        flat_indices.extend(idx)
                    else:
                        flat_indices.append(idx)
                indices = flat_indices
                
            # 评估该特征组合
            model_output = self._train_and_evaluate(
                train_loader,
                val_loader,
                indices,
                self.phenotype_features['D1']['indices']
            )
            
            # 计算和保存结果
            stats = self._calculate_statistics(
                model_output['predictions'],
                model_output['targets']
            )
            
            results[comb_name] = {
                'metrics': stats,
                'feature_importance': self._calculate_feature_importance(
                    model_output['model'],
                    indices
                ),
                'training_history': model_output['history'],
                'model_path': model_output['model_path']
            }

        # 生成比较可视化
        self._plot_feature_combination_comparison(results)
        
        # 清理内存
        MemoryManager.cleanup()
        
        return results


    def _rank_brain_features(self, train_loader, val_loader):
        """
        1.3 Feature importance ranking
        使用多种方法对特征进行重要性排序
        """
        logging.info("1.3 Feature importance ranking...")
        results = {
            'shap_values': {},
            'permutation_importance': {},
            'correlation_analysis': {}
        }
        
        # 计算SHAP值
        shap_values = self._calculate_shap_values(train_loader)
        results['shap_values'] = {
            'values': shap_values,
            'ranking': self._rank_features_by_shap(shap_values)
        }
        
        # 计算排列重要性
        perm_importance = self._calculate_permutation_importance(
            train_loader,
            val_loader
        )
        results['permutation_importance'] = {
            'scores': perm_importance,
            'ranking': self._rank_features_by_permutation(perm_importance)
        }
        
        # 相关性分析
        correlation_matrix = self._calculate_feature_correlations(train_loader)
        results['correlation_analysis'] = {
            'matrix': correlation_matrix,
            'ranking': self._rank_features_by_correlation(correlation_matrix)
        }
        
        # 生成重要性排序可视化
        self._plot_feature_importance_rankings(results)
        # 清理内存
        MemoryManager.cleanup()

        return results

    def _calculate_statistics(self, predictions, targets):
        """计算统计指标"""
        stats = {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
            'correlation': pearsonr(targets.flatten(), predictions.flatten())[0],
            'p_values': self._calculate_significance(predictions, targets)
        }
        return stats

    def _calculate_significance(self, predictions, targets):
        """计算统计显著性"""
        from scipy import stats
        _, p_values = stats.ttest_ind(predictions, targets)
        return p_values

    def _apply_multiple_testing_correction(self, p_values):
        """应用多重比较校正"""
        _, corrected_p_values, _, _ = multipletests(
            p_values,
            alpha=0.05,
            method='bonferroni'
        )
        return corrected_p_values

    def _save_brain_analysis_results(self, results):
        """保存脑部特征分析结果"""
        save_path = self.results_dir / 'brain_features'
        # 创建一个深拷贝以避免修改原始数据
        serializable_results = self._convert_to_serializable(results)
        
        # 保存数值结果
        with open(save_path / 'analysis_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
        # 保存可视化结果已在各个plot函数中完成
        logging.info(f"1. Brain Feature Importance Analysis results are saved: {save_path}")

    def _convert_to_serializable(self, obj):
        """将对象转换为JSON可序列化的格式。

        Parameters
        ----------
        obj : Any
            需要转换的对象

        Returns
        -------
        Any
            转换后的可JSON序列化对象
        """
        
        # 处理numpy的基本数据类型
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.str_, np.bytes_)):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.ndarray, np.generic)):
            return self._convert_to_serializable(obj.tolist())
            
        # 处理PyTorch张量
        if torch.is_tensor(obj):
            return self._convert_to_serializable(obj.cpu().detach().numpy())
            
        # 处理Path对象
        if isinstance(obj, Path):
            return str(obj)
            
        # 处理字典
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
            
        # 处理列表和元组
        if isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
            
        # 处理具有tolist方法的对象（如pandas.Series）
        if hasattr(obj, 'tolist'):
            try:
                return self._convert_to_serializable(obj.tolist())
            except:
                pass
                
        # 处理具有__dict__属性的对象
        if hasattr(obj, '__dict__'):
            try:
                return self._convert_to_serializable(obj.__dict__)
            except:
                pass
                
        # 尝试直接转换
        try:
            json.dumps(obj)
            return obj
        except:
            return str(obj)

    def _plot_brain_feature_evaluation(self, results, analysis_type):
        """绘制脑部特征评估结果"""
        plot_dir = self.results_dir / 'brain_features' / analysis_type / 'plots'
        
        # 1. 性能指标图
        plt.figure(figsize=(12, 6))
        metrics = results['metrics']
        plt.bar(['MSE', 'MAE', 'R2', 'Correlation'], 
                [metrics['mse'], metrics['mae'], metrics['r2'], metrics['correlation']])
        plt.title('Brain Feature Performance Metrics')
        plt.savefig(plot_dir / 'performance_metrics.png')
        plt.close()
        
        # 2. 训练历史图
        history = results['training_history']
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(plot_dir / 'training_history.png')
        plt.close()

    def _plot_feature_combination_comparison(self, results):
        """绘制特征组合比较图"""
        plot_dir = self.results_dir / 'brain_features' / '1_2_individual_analysis' / 'plots'
        
        # 1. 性能对比图
        plt.figure(figsize=(15, 8))
        combinations = list(results.keys())
        metrics = ['mse', 'mae', 'r2', 'correlation']
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            values = [results[comb]['metrics'][metric] for comb in combinations]
            plt.bar(combinations, values)
            plt.title(f'{metric.upper()} Comparison')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'feature_combination_comparison.png')
        plt.close()

    def _plot_feature_importance_rankings(self, results):
        """绘制特征重要性排序图"""
        plot_dir = self.results_dir / 'brain_features/importance_ranking/plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. SHAP值可视化
        if 'shap_values' in results and 'ranking' in results['shap_values']:
            plt.figure(figsize=(10, 6))
            shap_ranking = results['shap_values']['ranking']
            if shap_ranking:  # 确保有数据
                plt.barh(list(shap_ranking.keys()), list(shap_ranking.values()))
                plt.title('Feature Importance (SHAP Values)')
                plt.savefig(plot_dir / 'shap_importance.png')
            plt.close()

        # 2. 相关性热图
        if ('correlation_analysis' in results and 
            'matrix' in results['correlation_analysis']):
            
            corr_data = results['correlation_analysis']['matrix']
            if isinstance(corr_data, dict) and 'correlation_matrix' in corr_data:
                matrix_data = np.array(corr_data['correlation_matrix'])
                feature_names = corr_data.get('feature_names', [])
                
                if matrix_data.size > 0 and len(matrix_data.shape) == 2:  # 确保是2D数组
                    plt.figure(figsize=(8, 8))
                    sns.heatmap(matrix_data,
                            annot=True,
                            cmap='coolwarm',
                            center=0,
                            xticklabels=feature_names,
                            yticklabels=feature_names)
                    plt.title('Feature Correlation Matrix')
                    plt.tight_layout()
                    plt.savefig(plot_dir / '1_3_correlation_heatmap.png')
                plt.close()

    def analyze_phenotype_features(self, train_loader, val_loader):
        """
        2. Phenotypic Feature Importance Analysis
        分析表型特征的重要性和影响
        """
        logging.info("Starting Experiment 2. Phenotypic Feature Importance Analysis...")
        MemoryManager.log_memory_status("Before phenotype feature analysis")

        results = {
            'complete_evaluation': {},
            'individual_analysis': {},
            'importance_quantification': {}
        }

        # 2.1 Complete phenotype group evaluation
        results['complete_evaluation'] = self._evaluate_complete_phenotype_features(
            train_loader,
            val_loader
        )

        # 2.2 Individual feature analysis
        results['individual_analysis'] = self._analyze_individual_phenotype_features(
            train_loader,
            val_loader
        )

        # 2.3 Feature importance quantification
        results['importance_quantification'] = self._quantify_phenotype_importance(
            train_loader,
            val_loader
        )

        # 保存结果
        self._save_phenotype_analysis_results(results)

        MemoryManager.log_memory_status("After phenotype feature analysis")
        return results

    def _evaluate_complete_phenotype_features(self, train_loader, val_loader):
        """
        2.1 Complete phenotype group evaluation (B1 + D2/D3/D4)
        评估完整表型特征组的性能
        """
        logging.info("2.1 Analysing complete phenotype group evaluation...")
        results = {}
        
        # 分析B1与D2/D3/D4的组合
        phenotype_groups = ['D2', 'D3', 'D4']
        
        for group in phenotype_groups:
            # 使用完整脑部特征(B1)和当前表型特征组
            model_output = self._train_and_evaluate(
                train_loader,
                val_loader,
                self.brain_features['B1']['indices'],
                self.phenotype_features[group]['indices']
            )
            
            # 计算统计指标
            stats = self._calculate_statistics(
                model_output['predictions'],
                model_output['targets']
            )
            
            # 添加效应值计算
            effect_sizes = self._calculate_effect_sizes(
                model_output['predictions'],
                model_output['targets']
            )
            
            results[f'B1_plus_{group}'] = {
                'metrics': stats,
                'effect_sizes': effect_sizes,
                'training_history': model_output['history'],
                'model_path': model_output['model_path']
            }

        # 生成可视化
        self._plot_phenotype_evaluation(results)

        # 清理内存
        MemoryManager.cleanup()
        
        return results

    def _analyze_individual_phenotype_features(self, train_loader, val_loader):
        """
        2.2 Individual feature analysis (age, sex, education level)
        分析单个表型特征的影响
        """
        logging.info("2.2 Analysing individual feature analysis (age, sex, education level)...")
        results = {}
        
        # 分析B1与D5/D6/D7的组合
        individual_features = ['D5', 'D6', 'D7']  # age, sex, education
        
        for feature in individual_features:
            # 使用完整脑部特征(B1)和单个表型特征
            model_output = self._train_and_evaluate(
                train_loader,
                val_loader,
                self.brain_features['B1']['indices'],
                self.phenotype_features[feature]['indices']
            )
            
            # 计算主要指标
            stats = self._calculate_statistics(
                model_output['predictions'],
                model_output['targets']
            )
            
            # 添加特征specific分析
            feature_specific = self._analyze_feature_specific_impact(
                feature,
                model_output
            )
            
            results[f'B1_plus_{feature}'] = {
                'metrics': stats,
                'feature_specific': feature_specific,
                'training_history': model_output['history']
            }

        # 生成可视化
        self._plot_individual_phenotype_analysis(results)

        # 清理内存
        MemoryManager.cleanup()
        
        return results

    def _quantify_phenotype_importance(self, train_loader, val_loader):
        """
        2.3 Feature importance quantification
        Analyzes overall importance ranking of phenotypic features
        """
        logging.info("2.3 phenotype feature importance quantification...")
        
        results = {
            'shap_analysis': {},
            'permutation_importance': {},
            'relative_importance': {}
        }
        
        # SHAP Analysis
        shap_values = self._calculate_phenotype_shap_values(train_loader)
        results['shap_analysis'] = {
            'values': shap_values,
            'importance_ranking': self._rank_by_shap(shap_values)
        }
        
        # Permutation Importance
        perm_importance = self._calculate_phenotype_permutation_importance(
            train_loader,
            val_loader
        )
        results['permutation_importance'] = {
            'scores': perm_importance,
            'ranking': self._rank_by_permutation(perm_importance)
        }
        
        # Calculate relative contribution percentages
        total_importance = sum(results['shap_analysis']['importance_ranking'].values())
        results['relative_importance'] = {
            feature: (importance / total_importance * 100)
            for feature, importance in results['shap_analysis']['importance_ranking'].items()
        }
        
        # Generate visualization
        self._plot_phenotype_importance(results)

        # 清理内存
        MemoryManager.cleanup()

        return results

    def _analyze_feature_specific_impact(self, feature, model_output):
        """分析特定表型特征的影响"""
        if feature == 'D5':  # age
            return self._analyze_age_impact(model_output)
        elif feature == 'D6':  # sex
            return self._analyze_sex_impact(model_output)
        else:  # education
            return self._analyze_education_impact(model_output)

    def analyze_feature_interactions(self, train_loader, val_loader):
        """
        3. Feature Interaction Analysis
        分析不同特征之间的交互关系
        """
        logging.info("Starting Experiment 3. Feature Interaction Analysis...")
        MemoryManager.log_memory_status("Before feature interaction analysis")
        
        results = {
            'phenotypic_interactions': {},
            'cortical_interactions': {},
            'cross_domain_interactions': {}
        }

        # 3.1 Phenotypic feature interactions (MANOVA)
        results['phenotypic_interactions'] = self._analyze_phenotype_interactions(
            train_loader
        )

        # 3.2 Cortical feature interactions (PCA)
        results['cortical_interactions'] = self._analyze_cortical_interactions(
            train_loader
        )

        # 3.3 Cross-domain feature interactions (CCA)
        results['cross_domain_interactions'] = self._analyze_cross_domain_interactions(
            train_loader
        )

        # 保存结果
        self._save_interaction_analysis_results(results)

        # 清理内存
        MemoryManager.log_memory_status("After feature interaction analysis")
        MemoryManager.cleanup()
        return results

    def _analyze_phenotype_interactions(self, train_loader):
        """
        3.1 使用MANOVA分析表型特征之间的交互作用
        """
        logging.info("3.1 Analysing phenotypic feature interactions (MANOVA)...")
        from statsmodels.multivariate.manova import MANOVA
        
        # 收集数据
        phenotype_data = []
        targets = []
        
        with torch.no_grad():
            for _, phenotypes, target in train_loader:
                phenotype_data.append(phenotypes.numpy())
                targets.append(target.numpy())
        
        phenotype_data = np.vstack(phenotype_data)
        targets = np.vstack(targets)
        
        # 创建数据框
        df = pd.DataFrame(np.hstack([phenotype_data, targets]))
        df.columns = ['age', 'sex', 'education', 'attention', 'aggression']
        # 或者更明确地验证一下
        print("Shape check:")
        print(f"Phenotype data shape: {phenotype_data.shape}")  # 应该是 (N, 3)
        print(f"Targets shape: {targets.shape}")        # 应该是 (N, 2)
        
        # 执行MANOVA分析
        manova = MANOVA.from_formula(
            'attention + aggression ~ age + sex + education', 
            data=df
        )
        
        # 计算效应值
        effect_sizes = self._calculate_manova_effect_sizes(manova)
        
        results = {
            'test_results': manova.mv_test(),
            'effect_sizes': effect_sizes,
            'descriptive_stats': df.describe()
        }
        
        # 生成可视化
        self._plot_phenotype_interactions(results)
        # 清理内存
        MemoryManager.cleanup()

        return results

    def _analyze_cortical_interactions(self, train_loader):
        """
        3.2 使用PCA分析皮层特征之间的交互关系
        """
        logging.info("3.2 Analysing Cortical feature interactions with PCA...")
        from sklearn.decomposition import PCA
        
        # 收集脑部特征数据
        brain_features = []
        for images, _, _ in train_loader:
            features = self._extract_brain_features(images)
            brain_features.append(features)
        
        brain_features = np.vstack(brain_features)
        
        # 执行PCA分析
        pca = PCA()
        pca_result = pca.fit_transform(brain_features)
        
        results = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_,
            'transformed_data': pca_result
        }
        
        # 生成可视化
        self._plot_cortical_interactions(results)

        # 清理内存
        MemoryManager.cleanup()

        return results

    def _analyze_cross_domain_interactions(self, train_loader):
        """
        3.3 使用CCA分析表型特征和皮层特征之间的交互关系
        """
        logging.info("3.3 Analysing Cross-domain feature interactions with CCA...")
        from sklearn.cross_decomposition import CCA
        
        # 收集数据
        brain_features = []
        phenotype_features = []
        
        for images, phenotypes, _ in train_loader:
            brain_feats = self._extract_brain_features(images)
            brain_features.append(brain_feats)
            phenotype_features.append(phenotypes.numpy())
        
        brain_features = np.vstack(brain_features)
        phenotype_features = np.vstack(phenotype_features)
        
        # 执行CCA分析
        n_components = min(brain_features.shape[1], phenotype_features.shape[1])
        cca = CCA(n_components = n_components)
        brain_cca, pheno_cca = cca.fit_transform(brain_features, phenotype_features)
        
        # 计算相关系数
        correlations = [np.corrcoef(brain_cca[:, i], pheno_cca[:, i])[0, 1]
                       for i in range(cca.n_components)]
        
        results = {
            'correlations': correlations,
            'brain_weights': cca.x_weights_,
            'phenotype_weights': cca.y_weights_,
            'brain_components': brain_cca,
            'phenotype_components': pheno_cca,
            'n_components': n_components
        }
        
        # 生成可视化
        self._plot_cross_domain_interactions(results)
        
        # 清理内存
        MemoryManager.cleanup()
        return results

    def run_complete_analysis(self, train_loader, val_loader):
        """
        运行完整的特征分析流程
        """
        try:
            logging.info("Starting the full feature contribution analysis...")
            MemoryManager.log_memory_status("Before complete analysis")
            
            # 1. Brain Feature Analysis
            brain_results = self.analyze_brain_features(train_loader, val_loader)
            
            # 2. Phenotypic Feature Analysis
            phenotype_results = self.analyze_phenotype_features(train_loader, val_loader)
            
            # 3. Feature Interaction Analysis
            interaction_results = self.analyze_feature_interactions(train_loader, val_loader)
            
            # 创建总结报告
            self._create_summary_report({
                'brain_features': brain_results,
                'phenotype_features': phenotype_results,
                'interactions': interaction_results
            })

            MemoryManager.log_memory_status("After complete analysis")
            logging.info("Feature contribution analysis is finished")
            return True
            
        except Exception as e:
            MemoryManager.log_memory_status("Error occurred during analysis")
            logging.error(f"Errors in feature contribution analysis: {str(e)}")
            raise

    def _create_summary_report(self, results):
        """
        创建分析总结报告
        """
        report_path = self.results_dir / 'analysis_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("Feature contribution analysis report\n")
            f.write("=" * 50 + "\n\n")
            
            # 1. 脑部特征分析结果
            f.write("1. Brain Feature Importance Analysis report\n")
            f.write("-" * 30 + "\n")
            self._write_brain_feature_summary(f, results['brain_features'])
            
            # 2. 表型特征分析结果
            f.write("\n2. Phenotypic Feature Importance Analysis report\n")
            f.write("-" * 30 + "\n")
            self._write_phenotype_feature_summary(f, results['phenotype_features'])
            
            # 3. 特征交互分析结果
            f.write("\n3. Feature Interaction Analysis report\n")
            f.write("-" * 30 + "\n")
            self._write_interaction_summary(f, results['interactions'])
            
        logging.info(f"All reports are saved to: {report_path}")
    def evaluate_on_test_set(self, test_loader):
        """
        在测试集上评估模型性能

        Parameters:
        -----------
        test_loader : DataLoader
            测试数据加载器

        Returns:
        --------
        dict : 包含评估结果的字典
        """
        logging.info("Starting final evaluation on test set...")
        results = {
            'metrics': {},
            'visualizations': {}
        }
        
        try:
            # 确保模型在评估模式
            self.model_trainer.model.eval()
            all_predictions = []
            all_targets = []
            
            # 收集预测结果
            with torch.no_grad():
                for brain_images, phenotypes, targets in test_loader:
                    brain_images = brain_images.to(self.device)
                    phenotypes = phenotypes.to(self.device)
                    
                    outputs = self.model_trainer.model(brain_images, phenotypes)
                    all_predictions.extend(outputs.cpu().numpy())
                    all_targets.extend(targets.numpy())
            
            # 转换为numpy数组
            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)
            
            # 计算每个目标变量的指标
            target_names = ['attention', 'aggression']
            for i, target_name in enumerate(target_names):
                # 计算指标
                mse = mean_squared_error(all_targets[:, i], all_predictions[:, i])
                mae = mean_absolute_error(all_targets[:, i], all_predictions[:, i])
                r2 = r2_score(all_targets[:, i], all_predictions[:, i])
                correlation, p_value = pearsonr(all_targets[:, i], all_predictions[:, i])
                
                results['metrics'][target_name] = {
                    'MSE': float(mse),
                    'MAE': float(mae),
                    'R2': float(r2),
                    'correlation': float(correlation),
                    'p_value': float(p_value)
                }
                
                # 生成预测散点图
                plt.figure(figsize=(10, 6))
                plt.scatter(all_targets[:, i], all_predictions[:, i], alpha=0.5)
                plt.xlabel(f'True {target_name.capitalize()}')
                plt.ylabel(f'Predicted {target_name.capitalize()}')
                plt.title(f'{target_name.capitalize()} Prediction Performance on Test Set')
                
                # 添加完美预测线
                min_val = min(all_targets[:, i].min(), all_predictions[:, i].min())
                max_val = max(all_targets[:, i].max(), all_predictions[:, i].max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                
                # 添加性能指标文本
                plt.text(0.05, 0.95, 
                        f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}\n'
                        f'Correlation: {correlation:.4f}',
                        transform=plt.gca().transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 保存图像
                plot_path = self.results_dir / 'test_evaluation' / 'plots'
                plot_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_path / f'{target_name}_prediction.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                results['visualizations'][target_name] = str(plot_path / f'{target_name}_prediction.png')
            
            # 保存完整评估结果
            results_path = self.results_dir / 'test_evaluation' / 'results.json'
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(results['metrics'], f, indent=4)
            
            logging.info("Test set evaluation completed successfully")
            logging.info("Results saved to: " + str(results_path))
            
            # 打印评估结果概要
            print("\nTest Set Evaluation Results:")
            for target_name, metrics in results['metrics'].items():
                print(f"\n{target_name.capitalize()} Predictions:")
                for metric_name, value in metrics.items():
                    print(f"{metric_name}: {value:.4f}")
            
            return results
            
        except Exception as e:
            logging.error(f"Error in test set evaluation: {str(e)}")
            raise


    




