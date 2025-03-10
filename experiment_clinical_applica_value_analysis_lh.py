"""
Experiment 4: Clinical Application Value Analysis
"""
import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import json
from datetime import datetime

class ClinicalAnalysis:
    def __init__(self, model, device, experiment_dir):
        """
        初始化临床分析类
        Args:
            model: 训练好的模型
            device: 计算设备
            experiment_dir: 实验结果保存目录
        """
        self.model = model
        self.device = device
        self.results_dir = Path(experiment_dir)
        self.metrics_dir = self.results_dir / 'metrics'
        self.vis_dir = self.results_dir / 'visualizations'
        
        # 确保目录存在
        self.metrics_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)
        
        # 设置目标变量名称
        self.target_names = ['sum_att', 'sum_agg']
        
        # 设置分组信息
        self.sex_groups = {0: 'Female', 1: 'Male'}
        self.edu_groups = {0: 'Low', 1: 'Medium', 2: 'High'}
    
    def _get_predictions(self, loader):
        """获取模型预测结果"""
        self.model.eval()
        predictions = []
        phenotypes = []
        targets = []
        
        with torch.no_grad():
            for brain_images, pheno, target in loader:
                brain_images = brain_images.to(self.device)
                pheno = pheno.to(self.device)
                output = self.model(brain_images, pheno)
                predictions.append(output.cpu().numpy())
                phenotypes.append(pheno.cpu().numpy())
                targets.append(target.numpy())
        
        return (np.concatenate(predictions), 
                np.concatenate(phenotypes), 
                np.concatenate(targets))
    
    def _calculate_metrics(self, predictions, targets):
        """计算性能指标"""
        return {
            'mse': float(mean_squared_error(targets, predictions)),
            'mae': float(mean_absolute_error(targets, predictions)),
            'r2': float(r2_score(targets, predictions)),
            'correlation': float(np.corrcoef(predictions, targets)[0, 1])
        }
    
    def group_wise_analysis(self, test_loader):
        """
        4.1 Group-wise Performance Analysis
        分析不同组别的预测性能
        """
        logging.info("4.1 Starting group-wise performance analysis...")
        
        # 获取预测结果和分组信息
        predictions, phenotypes, targets = self._get_predictions(test_loader)
        
        # 初始化结果字典
        results = {
            'sex_groups': {name: {} for name in self.sex_groups.values()},
            'education_groups': {name: {} for name in self.edu_groups.values()},
        }
        
        # 1. 性别分组分析
        sex_analysis = self._analyze_sex_groups(predictions, phenotypes, targets)
        results['sex_groups'] = sex_analysis
        
        # 2. 教育水平分组分析
        edu_analysis = self._analyze_education_groups(predictions, phenotypes, targets)
        results['education_groups'] = edu_analysis
        
        # 保存结果
        with open(self.metrics_dir / '4_1_group_wise_analysis.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'results': results
            }, f, indent=4)
        
        return results
    
    def _analyze_sex_groups(self, predictions, phenotypes, targets):
        """性别分组分析"""
        results = {}
        
        # 获取性别信息 (第3列是性别)
        sex_column = phenotypes[:, 1]  # 根据实际数据调整索引
        
        for sex_code, sex_name in self.sex_groups.items():
            # 获取当前性别的样本掩码
            mask = (sex_column == sex_code)
            
            group_results = {}
            # 分别分析两个目标变量
            for i, target_name in enumerate(self.target_names):
                current_predictions = predictions[mask, i]
                current_targets = targets[mask, i]
                
                # 计算基本指标
                metrics = self._calculate_metrics(current_predictions, current_targets)
                
                # 进行假设检验
                t_stat, p_value = stats.ttest_ind(current_predictions, current_targets)
                
                group_results[target_name] = {
                    'metrics': metrics,
                    'statistical_tests': {
                        't_test': {
                            'statistic': float(t_stat),
                            'p_value': float(p_value)
                        }
                    }
                }
                
                # 绘制该组的预测-真实值对比图
                self._plot_group_predictions(
                    current_predictions,
                    current_targets,
                    f'sex_group_{sex_name}_{target_name}',
                    f'{sex_name} Group - {target_name}'
                )
            
            results[sex_name] = group_results
        
        # 绘制组间比较图
        self._plot_group_comparisons('sex_groups', results)
        
        return results
    
    def _analyze_education_groups(self, predictions, phenotypes, targets):
        """教育水平分组分析"""
        results = {}
        
        # 获取教育水平信息 (第4列是母亲教育水平)
        edu_column = phenotypes[:, 2]  # 根据实际数据调整索引
        
        for edu_code, edu_name in self.edu_groups.items():
            # 获取当前教育水平的样本掩码
            mask = (edu_column == edu_code)
            
            group_results = {}
            # 分别分析两个目标变量
            for i, target_name in enumerate(self.target_names):
                current_predictions = predictions[mask, i]
                current_targets = targets[mask, i]
                
                # 计算基本指标
                metrics = self._calculate_metrics(current_predictions, current_targets)
                
                # 进行假设检验
                t_stat, p_value = stats.ttest_ind(current_predictions, current_targets)
                
                # 计算预测误差
                errors = current_predictions - current_targets
                error_stats = {
                    'mean_error': float(np.mean(errors)),
                    'std_error': float(np.std(errors)),
                    'error_distribution': {
                        'skewness': float(stats.skew(errors)),
                        'kurtosis': float(stats.kurtosis(errors))
                    }
                }
                
                group_results[target_name] = {
                    'metrics': metrics,
                    'error_analysis': error_stats,
                    'statistical_tests': {
                        't_test': {
                            'statistic': float(t_stat),
                            'p_value': float(p_value)
                        }
                    }
                }
                
                # 绘制该组的预测-真实值对比图
                self._plot_group_predictions(
                    current_predictions,
                    current_targets,
                    f'education_group_{edu_name}_{target_name}',
                    f'{edu_name} Education Group - {target_name}'
                )
                
                # 绘制误差分布图
                self._plot_error_distribution(
                    errors,
                    f'education_group_{edu_name}_{target_name}',
                    f'Error Distribution - {edu_name} Education Group - {target_name}'
                )
            
            results[edu_name] = group_results
        
        # 绘制组间比较图
        self._plot_group_comparisons('education_groups', results)
        
        return results
    
    def _plot_group_predictions(self, predictions, targets, name, title):
        """绘制单个组的预测-真实值对比散点图"""
        plt.figure(figsize=(10, 6))
        
        # 散点图
        plt.scatter(targets, predictions, alpha=0.5)
        
        # 添加对角线
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # 计算并添加回归线
        z = np.polyfit(targets, predictions, 1)
        p = np.poly1d(z)
        plt.plot(targets, p(targets), "b-", label='Regression Line')
        
        # 计算相关系数和其他指标
        correlation = np.corrcoef(targets, predictions)[0, 1]
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 添加文本框显示指标
        plt.text(0.05, 0.95, 
                f'Correlation: {correlation:.3f}\n'
                f'MSE: {mse:.3f}\n'
                f'R²: {r2:.3f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title(title)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.vis_dir / f'predictions_{name}.png')
        plt.close()
    
    def _plot_error_distribution(self, errors, name, title):
        """绘制误差分布图"""
        plt.figure(figsize=(12, 5))
        
        # 误差分布直方图
        plt.subplot(1, 2, 1)
        sns.histplot(errors, kde=True)
        plt.title(f'{title}\nHistogram')
        plt.xlabel('Error')
        plt.ylabel('Count')
        
        # Q-Q图
        plt.subplot(1, 2, 2)
        stats.probplot(errors, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / f'error_distribution_{name}.png')
        plt.close()
    
    def _plot_group_comparisons(self, group_type, results):
        """绘制组间比较图"""
        metrics = ['mse', 'mae', 'r2', 'correlation']
        
        for target_name in self.target_names:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{target_name} - {group_type} Comparison')
            
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                
                # 收集数据
                groups = []
                values = []
                for group_name, group_data in results.items():
                    groups.append(group_name)
                    values.append(group_data[target_name]['metrics'][metric])
                
                # 绘制条形图
                ax.bar(groups, values)
                ax.set_title(f'{metric.upper()}')
                ax.set_xticklabels(groups, rotation=45)
                
                # 添加数值标签
                for j, v in enumerate(values):
                    ax.text(j, v, f'{v:.3f}', ha='center', va='bottom')
                
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'{group_type}_comparison_{target_name}.png')
            plt.close()