"""
Experiment - Primary Model Evaluation Framework 
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import time
import psutil
import GPUtil
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind
import seaborn as sns
import pandas as pd
import pingouin as pg
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from pathlib import Path
import json
from datetime import datetime
import gc

class ModelExperiments:
    def __init__(self, model_class, dataset_class, device, num_phenotypes=3, 
                 base_epochs=30, patience=10, experiment_repeats=3, test_mode=False, experiment_dir=None):
        """
        初始化实验框架
        Args:
            model_class: 模型类
            dataset_class: 数据集类
            device: 计算设备
            num_phenotypes: 表型特征数量
            base_epochs: 基础训练轮数
            patience: early stopping的耐心值
            experiment_repeats: 实验重复次数
            test_mode: 是否为测试模式
        """
        self.model_class = model_class
        self.dataset_class = dataset_class
        self.device = device
        self.num_phenotypes = num_phenotypes
        self.scaler = GradScaler()
        self.base_epochs = base_epochs
        self.patience = patience
        self.experiment_repeats = experiment_repeats
        
        # # 设置结果目录
        # self.results_dir = Path('/home/jouyang1/test_experiment_results' if test_mode else '/home/jouyang1/experiment_results_primary')
        # self.results_dir.mkdir(exist_ok=True)
        # self.vis_dir = self.results_dir / 'visualizations'
        # self.vis_dir.mkdir(exist_ok=True)

        # 设置结果目录
        if experiment_dir is None:
            self.results_dir = Path('/home/jouyang1/test_experiment_results' if test_mode else '/home/jouyang1/experiment_results_primary')
        else:
            self.results_dir = Path(experiment_dir)
        
        # 使用子目录
        self.vis_dir = self.results_dir / 'visualizations'
        self.data_dir = self.results_dir / 'data'
        self.logs_dir = self.results_dir / 'logs'
        self.metrics_dir = self.results_dir / 'metrics'
        # 添加进度文件路径
        self.progress_file = self.results_dir / 'experiment_progress.json'
        
        # 创建所有目录
        for dir_path in [self.vis_dir, self.data_dir, self.logs_dir, self.metrics_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 记录实验参数
        self.experiment_params = {
            'test_mode': test_mode,
            'base_epochs': base_epochs,
            'patience': patience,
            'experiment_repeats': experiment_repeats,
            'device': str(device),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 设置目标变量名称
        self.target_names = ['sum_att', 'sum_agg']
    
    def _load_progress(self):
        """加载实验进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'sensitivity_test_completed': False,
            'sample_size_impact_completed': False,
            'stability_evaluation_completed': False,
            'computational_assessment_completed':False,
            'last_experiment': None
        }

    def _save_progress(self, progress):
        """保存实验进度"""
            # 添加路径检查
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f)

    def add_gaussian_noise(self, data, mean=0, std=0.1):
        """添加高斯噪声"""
        return data + torch.randn_like(data) * std + mean
    
    def random_masking(self, data, mask_ratio=0.1):
        """随机遮挡"""
        mask = torch.rand_like(data) > mask_ratio
        return data * mask
        
    def plot_scatter(self, predictions, targets, experiment_name, score_name, additional_info=None):
        """为单个目标变量绘制散点图"""
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', label='Perfect Prediction')
        
        # 计算指标
        correlation = pearsonr(targets, predictions)[0]
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        plt.xlabel(f'True {score_name}')
        plt.ylabel(f'Predicted {score_name}')
        title = f'{experiment_name} - {score_name}\n'
        title += f'MSE: {mse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}'
        plt.title(title)
        
        # 添加统计信息文本框
        stats_text = f'MSE: {mse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}\n'
        stats_text += f'Correlation: {correlation:.3f}'
        if additional_info:
            stats_text += f'\n{additional_info}'
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_path = self.vis_dir / f'scatter_{experiment_name}_{score_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'correlation': correlation,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'plot_path': str(save_path)
        }
    
    def calculate_icc(self, predictions_array, targets, ratings_name):
        """
        计算组内相关系数(ICC)
        Args:
            predictions_array: shape (n_trials, n_samples)
            targets: shape (n_samples,)
            ratings_name: 评分名称
        """
        # 准备数据
        predictions_data = []
        raters = []
        targets_repeated = []
        
        # 添加预测值
        for i in range(predictions_array.shape[0]):
            predictions_data.extend(predictions_array[i])
            raters.extend([f'pred_{i}'] * len(targets))
            targets_repeated.extend(range(len(targets)))
        
        # 添加真实值
        predictions_data.extend(targets)
        raters.extend(['target'] * len(targets))
        targets_repeated.extend(range(len(targets)))
        
        # 创建数据框
        data = pd.DataFrame({
            'scores': predictions_data,
            'raters': raters,
            'targets': targets_repeated
        })
        
        # 计算ICC
        icc_result = pg.intraclass_corr(data=data, targets='targets', 
                                      raters='raters', ratings='scores')
        
        # 提取ICC1值和p值
        icc = icc_result.loc[icc_result['Type'] == 'ICC1', 'ICC'].values[0]
        icc_p = icc_result.loc[icc_result['Type'] == 'ICC1', 'pval'].values[0]
        
        return icc, icc_p

    def _train_model(self, model, train_loader, val_loader, optimizer, criterion, experiment_name, early_stopping=True):
        """
        训练模型并返回训练指标
        """
        metrics = {
            'train_losses': [],
            'val_losses': [],
            'best_val_loss': float('inf'),
            'patience_counter': 0,
            'best_epoch': 0
        }
        
        for epoch in range(self.base_epochs):
            # 训练阶段
            model.train()
            train_loss = 0
            batch_count = 0
            
            for brain_images, phenotypes, targets in train_loader:
                # 将数据移到设备上
                brain_images = brain_images.to(self.device)  # [batch_size, 4, 512, 512]
                phenotypes = phenotypes.to(self.device)      # [batch_size, 3]
                targets = targets.to(self.device)            # [batch_size, 2]
                
                optimizer.zero_grad()
                
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(brain_images, phenotypes)  # [batch_size, 2]
                    loss = criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                batch_count += 1
                
                # 释放内存
                del brain_images, phenotypes, outputs
                torch.cuda.empty_cache()
            
            avg_train_loss = train_loss / batch_count
            metrics['train_losses'].append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for brain_images, phenotypes, targets in val_loader:
                    brain_images = brain_images.to(self.device)
                    phenotypes = phenotypes.to(self.device)
                    targets = targets.to(self.device)
                    
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(brain_images, phenotypes)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    val_batch_count += 1
                    
                    # 释放内存
                    del brain_images, phenotypes, outputs
                    torch.cuda.empty_cache()
            
            avg_val_loss = val_loss / val_batch_count
            metrics['val_losses'].append(avg_val_loss)

            # 添加边界条件检查
            if torch.isnan(torch.tensor(avg_val_loss)) or torch.isinf(torch.tensor(avg_val_loss)):
                logging.warning(f"{experiment_name} - Invalid validation loss detected at epoch {epoch}")
                break
            # 也可以检查训练损失
            if torch.isnan(torch.tensor(avg_train_loss)) or torch.isinf(torch.tensor(avg_train_loss)):
                logging.warning(f"{experiment_name} - Invalid training loss detected at epoch {epoch}")
                break
            
            # 早停检查
            if early_stopping:
                if avg_val_loss < metrics['best_val_loss']:
                    metrics['best_val_loss'] = avg_val_loss
                    metrics['best_epoch'] = epoch
                    metrics['patience_counter'] = 0
                else:
                    metrics['patience_counter'] += 1
                    
                if metrics['patience_counter'] >= self.patience:
                    logging.info(f"{experiment_name} - Early stopping triggered at epoch {epoch}")
                    break
            
            logging.info(f"{experiment_name} - Epoch {epoch+1}/{self.base_epochs}, "
                        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return metrics

    def _get_predictions(self, model, loader):
        """获取模型预测结果"""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for brain_images, phenotypes, target in loader:
                brain_images = brain_images.to(self.device)
                phenotypes = phenotypes.to(self.device)
                output = model(brain_images, phenotypes)
                predictions.append(output.cpu().numpy())
                targets.append(target.numpy())
        
        return np.concatenate(predictions), np.concatenate(targets)

    

    """
    1.1 Sensitivity test to different data perturbations
        > [!TIP] 
        > 数据扰动敏感性测试：使用高斯噪声、随机遮挡等
        > 评估指标：标准差、变异系数(CV)、ICCs(组内相关系数)
    """

    def sensitivity_test(self, model_path, test_loader, num_trials=5, noise_levels=[0.05, 0.1, 0.15]):
        """
        增强版敏感性测试，分别分析sum_att和sum_agg
        Args:
            model_path: 预训练模型路径
            test_loader: 测试数据加载器
            num_trials: 每个噪声级别的重复次数
            noise_levels: 噪声级别列表
        """
        logging.info("1.1 Starting enhanced sensitivity test...")
        
        model = self.model_class(self.num_phenotypes).to(self.device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        results = {
            'sum_att': {
                'gaussian_noise': [],
                'random_masking': []
            },
            'sum_agg': {
                'gaussian_noise': [],
                'random_masking': []
            }
        }
        
        # 记录原始预测结果
        original_predictions = []
        true_targets = []
        
        with torch.no_grad():
            for brain_images, phenotypes, targets in test_loader:
                brain_images = brain_images.to(self.device)
                phenotypes = phenotypes.to(self.device)
                outputs = model(brain_images, phenotypes)
                original_predictions.append(outputs.cpu().numpy())
                true_targets.append(targets.numpy())
        
        original_predictions = np.concatenate(original_predictions)  # shape: (N, 2)
        true_targets = np.concatenate(true_targets)                 # shape: (N, 2)
        
        # 为每个噪声级别进行测试
        for noise_level in noise_levels:
            gaussian_predictions = []
            masking_predictions = []
            
            for trial in range(num_trials):
                gaussian_trial_preds = []
                masking_trial_preds = []
                
                with torch.no_grad():
                    for brain_images, phenotypes, _ in test_loader:
                        brain_images = brain_images.to(self.device)
                        phenotypes = phenotypes.to(self.device)
                        
                        # 高斯噪声测试
                        noisy_images = self.add_gaussian_noise(brain_images, std=noise_level)
                        noisy_output = model(noisy_images, phenotypes)
                        gaussian_trial_preds.append(noisy_output.cpu().numpy())
                        
                        # 随机遮挡测试
                        masked_images = self.random_masking(brain_images, mask_ratio=noise_level)
                        masked_output = model(masked_images, phenotypes)
                        masking_trial_preds.append(masked_output.cpu().numpy())
                
                gaussian_predictions.append(np.concatenate(gaussian_trial_preds))
                masking_predictions.append(np.concatenate(masking_trial_preds))
            
            # 转换为numpy数组，shape: (num_trials, N, 2)
            gaussian_predictions = np.array(gaussian_predictions)
            masking_predictions = np.array(masking_predictions)
            
            # 分别分析两个预测目标
            for i, score_name in enumerate(self.target_names):
                # 计算高斯噪声的影响
                gaussian_changes = np.abs(gaussian_predictions[:, :, i] - original_predictions[:, i]) / np.abs(original_predictions[:, i])
                gaussian_icc, gaussian_icc_p = self.calculate_icc(
                    gaussian_predictions[:, :, i], 
                    true_targets[:, i],
                    f'{score_name}_gaussian'
                )
                gaussian_t_stat, gaussian_p_value = ttest_ind(
                    gaussian_predictions[:, :, i].flatten(),
                    original_predictions[:, i].flatten()
                )
                
                # 保存高斯噪声结果
                results[score_name]['gaussian_noise'].append({
                    'noise_level': noise_level,
                    'mean_change': float(np.mean(gaussian_changes)),
                    'std_dev': float(np.std(gaussian_changes)),
                    'cv': float(np.std(gaussian_changes) / np.mean(gaussian_changes)),
                    'icc': float(gaussian_icc),
                    'icc_p_value': float(gaussian_icc_p),
                    't_statistic': float(gaussian_t_stat),
                    'p_value': float(gaussian_p_value)
                })
                
                # 计算随机遮挡的影响
                masking_changes = np.abs(masking_predictions[:, :, i] - original_predictions[:, i]) / np.abs(original_predictions[:, i])
                masking_icc, masking_icc_p = self.calculate_icc(
                    masking_predictions[:, :, i], 
                    true_targets[:, i],
                    f'{score_name}_masking'
                )
                masking_t_stat, masking_p_value = ttest_ind(
                    masking_predictions[:, :, i].flatten(),
                    original_predictions[:, i].flatten()
                )
                
                # 保存随机遮挡结果
                results[score_name]['random_masking'].append({
                    'noise_level': noise_level,
                    'mean_change': float(np.mean(masking_changes)),
                    'std_dev': float(np.std(masking_changes)),
                    'cv': float(np.std(masking_changes) / np.mean(masking_changes)),
                    'icc': float(masking_icc),
                    'icc_p_value': float(masking_icc_p),
                    't_statistic': float(masking_t_stat),
                    'p_value': float(masking_p_value)
                })
                
                # 绘制散点图
                self.plot_scatter(
                    np.mean(gaussian_predictions[:, :, i], axis=0),
                    true_targets[:, i],
                    f'gaussian_noise_level_{noise_level}',
                    score_name,
                    f'Noise Level: {noise_level}\nICC: {gaussian_icc:.3f}\np-value: {gaussian_p_value:.3f}'
                )
                
                self.plot_scatter(
                    np.mean(masking_predictions[:, :, i], axis=0),
                    true_targets[:, i],
                    f'random_masking_level_{noise_level}',
                    score_name,
                    f'Mask Ratio: {noise_level}\nICC: {masking_icc:.3f}\np-value: {masking_p_value:.3f}'
                )
                
                # 绘制箱线图
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.boxplot(
                    [original_predictions[:, i]] + 
                    [pred[:, i] for pred in gaussian_predictions]
                )
                plt.title(f'{score_name} - Gaussian Noise Impact (level={noise_level})')
                plt.xticks(range(1, num_trials + 2), 
                          ['Original'] + [f'Trial {j+1}' for j in range(num_trials)])
                
                plt.subplot(1, 2, 2)
                plt.boxplot(
                    [original_predictions[:, i]] + 
                    [pred[:, i] for pred in masking_predictions]
                )
                plt.title(f'{score_name} - Random Masking Impact (level={noise_level})')
                plt.xticks(range(1, num_trials + 2), 
                          ['Original'] + [f'Trial {j+1}' for j in range(num_trials)])
                
                plt.tight_layout()
                plt.savefig(self.vis_dir / f'noise_impact_boxplot_{score_name}_level_{noise_level}.png')
                plt.close()
        
        # 保存结果到JSON文件
        with open(self.metrics_dir / '1_1_sensitivity_test_results.json', 'w') as f:
            json.dump({
                'test_parameters': {
                    'num_trials': num_trials,
                    'noise_levels': noise_levels,
                    'model_path': str(model_path)
                },
                'results': results
            }, f, indent=4)
        
        # 只在进度文件中保存关键信息
        progress_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics_file': str(self.metrics_dir / '1_1_sensitivity_test_results.json')
        }
        
        return progress_results
    
    """
    1.2 The impact of sample size changes on model performance
        > [!TIP] 
        > - MSE, MAE, RMSE, R² scores
        > - Correlation coefficient analysis
        > - Scatter plot analysis (predicted vs. true values)
    """
    def sample_size_impact(self, train_data, val_data, sample_sizes=[0.25, 0.5, 0.75, 1.0]):
        """
        增强的样本量影响分析，分别分析sum_att和sum_agg
        Args:
            train_data: 训练数据集
            val_data: 验证数据集
            sample_sizes: 样本比例列表
        """
        logging.info("1.2 Starting enhanced sample size impact analysis...")
        
        results = {
            'sum_att': [],
            'sum_agg': []
        }
        
        # performance_curves = {
        #     'sum_att': {size: {'mse': [], 'mae': [], 'r2': [], 'correlation': []} for size in sample_sizes},
        #     'sum_agg': {size: {'mse': [], 'mae': [], 'r2': [], 'correlation': []} for size in sample_sizes}
        # }
        # 获取总样本数
        total_available_samples = len(train_data)

        for size in sample_sizes:
            logging.info(f"Testing sample size: {size*100}%")
            n_samples = min(int(total_available_samples * size), total_available_samples)
            size_results = {
                'sum_att': [],
                'sum_agg': []
            }
            # n_samples = int(len(train_data) * size)
            
            for repeat in range(self.experiment_repeats):
                logging.info(f"Repeat {repeat + 1}/{self.experiment_repeats}")
                
                # 随机选择训练样本
                # indices = np.random.choice(len(train_data), n_samples, replace=False)
                indices = np.random.choice(total_available_samples, n_samples, replace=False)
                subset_train = torch.utils.data.Subset(train_data, indices)
                
                # 初始化模型和数据加载器
                model = self.model_class(self.num_phenotypes).to(self.device)
                train_loader = DataLoader(subset_train, batch_size=8, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=8)
                
                # 训练模型
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                metrics = self._train_model(
                    model, train_loader, val_loader, optimizer, criterion,
                    experiment_name=f"sample_size_{size}_repeat_{repeat}"
                )
                
                # 获取预测结果
                predictions, targets = self._get_predictions(model, val_loader)
                
                # 分别分析两个预测目标
                for i, score_name in enumerate(self.target_names):
                    # 计算指标
                    current_preds = predictions[:, i]
                    current_targets = targets[:, i]
                    
                    mse = mean_squared_error(current_targets, current_preds)
                    mae = mean_absolute_error(current_targets, current_preds)
                    r2 = r2_score(current_targets, current_preds)
                    correlation = pearsonr(current_targets, current_preds)[0]
                    
                    result_metrics = {
                        'sample_size': size,
                        'repeat': repeat,
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'correlation': correlation
                    }
                    
                    size_results[score_name].append(result_metrics)
                    
                    # 更新性能曲线数据
                    # performance_curves[score_name][size]['mse'].append(mse)
                    # performance_curves[score_name][size]['mae'].append(mae)
                    # performance_curves[score_name][size]['r2'].append(r2)
                    # performance_curves[score_name][size]['correlation'].append(correlation)
                    
                    # 绘制散点图
                    self.plot_scatter(
                        current_preds,
                        current_targets,
                        f'sample_size_{size}_repeat_{repeat}',
                        score_name,
                        f'Sample Size: {size*100}%\nRepeat: {repeat+1}/{self.experiment_repeats}'
                    )
                    
                # 清理内存
                del model
                torch.cuda.empty_cache()
            
            # 计算每个预测目标的平均指标
            for score_name in self.target_names:
                avg_metrics = {
                    'sample_size': size,
                    'mse_mean': np.mean([r['mse'] for r in size_results[score_name]]),
                    'mse_std': np.std([r['mse'] for r in size_results[score_name]]),
                    'mae_mean': np.mean([r['mae'] for r in size_results[score_name]]),
                    'mae_std': np.std([r['mae'] for r in size_results[score_name]]),
                    'r2_mean': np.mean([r['r2'] for r in size_results[score_name]]),
                    'r2_std': np.std([r['r2'] for r in size_results[score_name]]),
                    'correlation_mean': np.mean([r['correlation'] for r in size_results[score_name]]),
                    'correlation_std': np.std([r['correlation'] for r in size_results[score_name]])
                }
                results[score_name].append(avg_metrics)

        # 用于可视化模型性能随样本量的变化,展示不同样本量下模型的性能指标
        self._plot_learning_curves(results)
        
        # 保存结果
        with open(self.metrics_dir / '1_2_sample_size_impact_results.json', 'w') as f:
            json.dump({
                'test_parameters': {
                    'sample_sizes': sample_sizes,
                    'experiment_repeats': self.experiment_repeats
                },
                'results': results
            }, f, indent=4)

        # 只在进度文件中保存关键信息
        progress_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics_file': str(self.metrics_dir / '1_2_sample_size_impact_results.json'),
            'sample_sizes': sample_sizes
        }
        
        return progress_results
        

    def _plot_learning_curves(self, results):
        """
        用于可视化模型性能随样本量的变化,展示不同样本量下模型的性能指标
        """
        metrics = ['mse', 'mae', 'r2', 'correlation']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for target_name in self.target_names:  # ['sum_att', 'sum_agg']
                sizes = [r['sample_size'] for r in results[target_name]]
                means = [r[f'{metric}_mean'] for r in results[target_name]]
                stds = [r[f'{metric}_std'] for r in results[target_name]]
                
                ax.errorbar(sizes, means, yerr=stds, 
                        fmt='o-', capsize=5, label=target_name)
                
                # 添加数值标签
                for x, y, std in zip(sizes, means, stds):
                    ax.text(x, y, f'{y:.3f}\n±{std:.3f}', 
                        horizontalalignment='center', 
                        verticalalignment='bottom')
            
            ax.set_title(f'{metric.upper()} vs Sample Size')
            ax.set_xlabel('Sample Size Ratio')
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'learning_curves.png', dpi=300)
        plt.close()

    """
    2. Model Stability Evaluation
        1. Performance stability under different initialization conditions
            > [!TIP] 
            > - K-fold cross-validation statistics
            > - Performance metrics for each fold
            > - Mean and standard deviation analysis
            > - Training curve analysis
    """

    def stability_evaluation(self, train_data, n_splits=5):
        """
        进行K-fold交叉验证的稳定性评估，分别分析sum_att和sum_agg
        Args:
            train_data: 训练数据集
            n_splits: 折数
        """
        logging.info("2. Starting stability evaluation...")
        fold_histories = []
        
        results = {
            'sum_att': [],
            'sum_agg': []
        }
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # # 存储每个fold的训练曲线
        # training_curves = {
        #     'sum_att': {'train_losses': [], 'val_losses': []},
        #     'sum_agg': {'train_losses': [], 'val_losses': []}
        # }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
            logging.info(f"Processing fold {fold + 1}/{n_splits}")
            
            # 创建数据加载器
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(train_data, batch_size=8, sampler=train_sampler)
            val_loader = DataLoader(train_data, batch_size=8, sampler=val_sampler)
            
            # 初始化模型
            model = self.model_class(self.num_phenotypes).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # 训练模型
            train_metrics = self._train_model(
                model, 
                train_loader, 
                val_loader, 
                optimizer, 
                criterion,
                experiment_name=f"stability_fold_{fold}"
            )
            
            # 获取预测结果
            predictions, targets = self._get_predictions(model, val_loader)
            
            # 分别分析两个预测目标
            for i, score_name in enumerate(self.target_names):
                current_preds = predictions[:, i]
                current_targets = targets[:, i]
                
                # 计算各种指标
                mse = mean_squared_error(current_targets, current_preds)
                mae = mean_absolute_error(current_targets, current_preds)
                r2 = r2_score(current_targets, current_preds)
                correlation = pearsonr(current_targets, current_preds)[0]
                
                fold_metrics = {
                    'fold': fold,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'correlation': correlation,
                    'train_loss': train_metrics['train_losses'][-1],  # 最后一个epoch的损失
                    'val_loss': train_metrics['val_losses'][-1] if train_metrics['val_losses'] else None
                }
                
                results[score_name].append(fold_metrics)
                
                # 存储训练曲线数据
                # if fold == 0:  # 只存储第一个fold的完整训练曲线
                #     training_curves[score_name]['train_losses'] = train_metrics['train_losses']
                #     training_curves[score_name]['val_losses'] = train_metrics['val_losses']
                
                # 绘制散点图
                self.plot_scatter(
                    current_preds,
                    current_targets,
                    f'stability_fold_{fold}',
                    score_name,
                    f'Fold: {fold + 1}/{n_splits}'
                )
            
            history = {
                'train_losses': train_metrics['train_losses'],
                'val_losses': train_metrics['val_losses']
            }
            fold_histories.append(history)  # 存储每个fold的训练历史

            # 添加内存清理
            if torch.cuda.is_available():
                del model, optimizer
                torch.cuda.empty_cache()

            # 释放内存
            # del model
            torch.cuda.empty_cache()
        
        # 计算总体统计指标
        final_metrics = {}
        for score_name in self.target_names:
            final_metrics[score_name] = {
                'mse_mean': np.mean([r['mse'] for r in results[score_name]]),
                'mse_std': np.std([r['mse'] for r in results[score_name]]),
                'mae_mean': np.mean([r['mae'] for r in results[score_name]]),
                'mae_std': np.std([r['mae'] for r in results[score_name]]),
                'r2_mean': np.mean([r['r2'] for r in results[score_name]]),
                'r2_std': np.std([r['r2'] for r in results[score_name]]),
                'correlation_mean': np.mean([r['correlation'] for r in results[score_name]]),
                'correlation_std': np.std([r['correlation'] for r in results[score_name]])
            }
        
        # 在所有fold训练完成后调用
        # 在stability evaluation中显示每个fold的训练和验证loss随epoch的变化
        self._plot_performance_curves(fold_histories, 'stability_evaluation')  
        # 绘制稳定性分析箱线图
        self._plot_stability_boxplots(results)     
        
        # 保存结果
        with open(self.metrics_dir / '2_stability_evaluation_results.json', 'w') as f:
            json.dump({
                'test_parameters': {
                    'n_splits': n_splits,
                },
                'results': results,
                'final_metrics': final_metrics
            }, f, indent=4)

        progress_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics_file': str(self.metrics_dir / '2_stability_evaluation_results.json'),
            'n_splits': n_splits
        }
        
        return progress_results
        
    
    # def _plot_training_curves(self, training_curves):
    #     """绘制训练曲线，分别显示sum_att和sum_agg的结果"""
    #     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
    #     for i, score_name in enumerate(self.target_names):
    #         ax = axes[i]
    #         train_losses = training_curves[score_name]['train_losses']
    #         val_losses = training_curves[score_name]['val_losses']
            
    #         epochs = range(1, len(train_losses) + 1)
    #         ax.plot(epochs, train_losses, 'b-', label='Training Loss')
    #         if val_losses:
    #             ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
            
    #         ax.set_title(f'{score_name} Training Curves')
    #         ax.set_xlabel('Epoch')
    #         ax.set_ylabel('Loss')
    #         ax.grid(True, alpha=0.3)
    #         ax.legend()
        
    #     plt.tight_layout()
    #     plt.savefig(self.vis_dir / 'training_curves.png', dpi=300)
    #     plt.close()

    def _plot_performance_curves(self, curves_data, experiment_name):
        """
        绘制性能曲线（用于stability evaluation中展示训练过程）,展示模型在训练过程中loss的变化
        """
        plt.figure(figsize=(12, 6))
        
        # 找到最短的历史记录长度
        min_length = min(len(data['train_losses']) for data in curves_data)
        
        # 截断所有历史记录到相同长度
        truncated_data = []
        for data in curves_data:
            truncated_data.append({
                'train_losses': data['train_losses'][:min_length],
                'val_losses': data['val_losses'][:min_length]
            })
        
        # 绘制每个fold的曲线
        for fold_idx, data in enumerate(truncated_data):
            plt.plot(data['train_losses'], alpha=0.3, 
                    label=f'Fold {fold_idx+1} Train')
            plt.plot(data['val_losses'], alpha=0.3, linestyle='--',
                    label=f'Fold {fold_idx+1} Val')
        
        # 计算并绘制平均曲线
        train_means = np.mean([d['train_losses'] for d in truncated_data], axis=0)
        val_means = np.mean([d['val_losses'] for d in truncated_data], axis=0)
        train_stds = np.std([d['train_losses'] for d in truncated_data], axis=0)
        val_stds = np.std([d['val_losses'] for d in truncated_data], axis=0)
        
        epochs = range(len(train_means))
        plt.plot(epochs, train_means, 'b-', linewidth=2, label='Mean Train Loss')
        plt.plot(epochs, val_means, 'r-', linewidth=2, label='Mean Val Loss')
        
        # 添加标准差区域
        plt.fill_between(epochs, train_means - train_stds, train_means + train_stds,
                        alpha=0.1, color='blue')
        plt.fill_between(epochs, val_means - val_stds, val_means + val_stds,
                        alpha=0.1, color='red')
        
        plt.title('Model Training Stability')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(self.vis_dir / f'performance_curves_{experiment_name}.png', dpi=300)
        plt.close()


    def _plot_stability_boxplots(self, results):
        """绘制稳定性分析的箱线图"""
        metrics = ['mse', 'mae', 'r2', 'correlation']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            # 准备数据
            data = []
            labels = []
            for score_name in self.target_names:
                metric_values = [r[metric] for r in results[score_name]]
                data.append(metric_values)
                labels.extend([score_name] * len(metric_values))
            
            # 绘制箱线图
            sns.boxplot(data=data, ax=axes[i])
            axes[i].set_title(f'{metric.upper()} Distribution')
            axes[i].set_xticklabels(self.target_names)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'stability_boxplots.png', dpi=300)
        plt.close()



    """
    3. Computational Resource Assessment
         > [!TIP] 
         > - Training time measurement
         > - Inference time measurement
         > - GPU memory usage analysis
         > - Model parameter count
    """
    def computational_assessment(self, model_path, train_loader, test_loader):
        """
        评估计算资源使用情况，包括训练和推理时间、内存使用等
        Args:
            model_path: 预训练模型路径
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
        """
        logging.info("3. Starting computational assessment...")
        results = {
            'hardware_info': self._get_hardware_info(),
            'model_info': {},
            'training_metrics': {},
            'inference_metrics': {},
            'memory_metrics': {}
        }
        
        # 1. 获取模型信息
        model = self.model_class(self.num_phenotypes).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results['model_info'] = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'model_size_mb': sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        # 2. 测量训练时间和资源使用
        train_metrics = self._measure_training_performance(model, train_loader)
        results['training_metrics'] = train_metrics
        
        # 3. 测量推理性能
        # 加载预训练模型
        model.load_state_dict(torch.load(model_path, weights_only=True))
        inference_metrics = self._measure_inference_performance(model, test_loader)
        results['inference_metrics'] = inference_metrics
        
        # 4. 内存使用分析
        results['memory_metrics'] = self._measure_memory_usage(model)
        
        # 生成性能曲线图
        # self._plot_performance_metrics(results)
        self._plot_memory_metrics(results)  
        
        # 保存结果
        with open(self.metrics_dir / '3_computational_assessment_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        # 返回轻量级信息
        progress_results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics_file': str(self.metrics_dir / '3_computational_assessment_results.json'),
            'model_path': str(model_path)
        }
        
        return progress_results
    
    def _get_hardware_info(self):
        """获取硬件信息"""
        info = {
            'device': str(self.device),
            'cpu_info': {
                'physical_cores': psutil.cpu_count(logical=False),
                'total_cores': psutil.cpu_count(logical=True),
            },
            'memory_info': {
                'total_memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
        
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            info['gpu_info'] = {
                'name': gpu.name,
                'total_memory_mb': gpu.memoryTotal,
                'driver_version': torch.version.cuda
            }
        
        return info
    
    def _measure_training_performance(self, model, train_loader):
        """测量训练性能"""
        # 使用小批量数据进行测试
        subset_size = min(len(train_loader.dataset), 100)
        train_subset = torch.utils.data.Subset(train_loader.dataset, range(subset_size))
        train_loader_small = DataLoader(train_subset, batch_size=8)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 测量训练时间
        start_time = time.time()
        gpu_start = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # 训练几个epoch来获取平均性能
        for epoch in range(3):
            for batch_idx, (brain_images, phenotypes, targets) in enumerate(train_loader_small):
                brain_images = brain_images.to(self.device)
                phenotypes = phenotypes.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(brain_images, phenotypes)
                    loss = criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
        
        training_time = time.time() - start_time
        gpu_memory_used = torch.cuda.memory_allocated() - gpu_start if torch.cuda.is_available() else 0
        
        return {
            'total_training_time': training_time,
            'avg_time_per_epoch': training_time / 3,
            'avg_time_per_batch': training_time / (3 * len(train_loader_small)),
            'gpu_memory_used_training_mb': gpu_memory_used / (1024 * 1024) if gpu_memory_used > 0 else 0
        }
    
    def _measure_inference_performance(self, model, test_loader):
        """测量推理性能"""
        model.eval()
        inference_times = []
        batch_sizes = []
        memory_usage = []
        
        with torch.no_grad():
            for brain_images, phenotypes, _ in test_loader:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.time()
                brain_images = brain_images.to(self.device)
                phenotypes = phenotypes.to(self.device)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                _ = model(brain_images, phenotypes)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.time() - start_time
                mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
                inference_times.append(inference_time)
                batch_sizes.append(brain_images.shape[0])
                memory_usage.append((mem_after - mem_before) / (1024 * 1024))  # MB
        
        return {
            'avg_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'max_inference_time': np.max(inference_times),
            'min_inference_time': np.min(inference_times),
            'avg_memory_per_inference_mb': np.mean(memory_usage) if memory_usage[0] > 0 else 0,
            'throughput_samples_per_second': np.mean(batch_sizes) / np.mean(inference_times)
        }
    

    def _measure_memory_usage(self, model):
        """测量内存使用情况"""
        metrics = {
            'cpu_memory': {
                'total': psutil.virtual_memory().total / (1024**3),
                'available': psutil.virtual_memory().available / (1024**3),
                'used': psutil.virtual_memory().used / (1024**3),
                'percent': psutil.virtual_memory().percent
            },
            'gpu_memory': {
                'allocated': torch.cuda.memory_allocated() / (1024**3),
                'cached': torch.cuda.memory_reserved() / (1024**3),
                'peak': torch.cuda.max_memory_allocated() / (1024**3)
            }
        }
        return metrics


    def _plot_memory_metrics(self, results):
        """improved的性能指标可视化"""
        plt.figure(figsize=(15, 10))
        
        # 1. CPU和GPU内存使用
        plt.subplot(2, 2, 1)
        memory_data = [
            results['memory_metrics']['cpu_memory']['used'],
            results['memory_metrics']['gpu_memory']['allocated'],
            results['memory_metrics']['gpu_memory']['peak']
        ]
        labels = ['CPU Used', 'GPU Allocated', 'GPU Peak']
        plt.bar(labels, memory_data)
        plt.title('Memory Usage (GB)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. 时间性能
        plt.subplot(2, 2, 2)
        time_data = [
            results['training_metrics']['avg_time_per_epoch'],
            results['inference_metrics']['avg_inference_time'],
            results['inference_metrics']['max_inference_time']
        ]
        labels = ['Train/Epoch', 'Avg Inference', 'Max Inference']
        plt.bar(labels, time_data)
        plt.title('Time Performance (seconds)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. 内存使用百分比
        plt.subplot(2, 2, 3)
        plt.pie([
            results['memory_metrics']['cpu_memory']['percent'],
            100 - results['memory_metrics']['cpu_memory']['percent']
        ], labels=['Used', 'Available'], autopct='%1.1f%%')
        plt.title('CPU Memory Usage (%)')
        
        # 4. GPU内存使用趋势
        plt.subplot(2, 2, 4)
        gpu_data = [
            results['memory_metrics']['gpu_memory']['allocated'],
            results['memory_metrics']['gpu_memory']['cached'],
            results['memory_metrics']['gpu_memory']['peak']
        ]
        labels = ['Allocated', 'Cached', 'Peak']
        plt.bar(labels, gpu_data)
        plt.title('GPU Memory Usage Details (GB)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'performance_metrics.png')
        plt.close()


    """
    4. （optional) Comparison with Traditional Machine Learning Approaches
        1. Prediction Accuracy  (MSE, MAE, R²)
        2. Computational Resource Assessment
        3. Model complexity (Model parameter count)
        4. Feature Importance (参考section B)

    """


