"""
Experiment 3: Model Interpretability Analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, ttest_1samp, wilcoxon, norm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import scipy.stats as stats
from scipy.interpolate import griddata
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class PredictionPatternAnalysis:
    def __init__(self, model, device, results_dir):
        """
        1. 初始化预测模式分析类
        Args:
            model: 训练好的模型
            device: 计算设备
            results_dir: 结果保存目录
        """
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / 'metrics'
        self.vis_dir = self.results_dir / 'visualizations'
        self.metrics_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)
        
        # 设置目标变量名称
        self.target_names = ['sum_att', 'sum_agg']

    def _get_predictions(self, loader):
        """获取模型预测结果"""
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for brain_images, phenotypes, target in loader:
                brain_images = brain_images.to(self.device)
                phenotypes = phenotypes.to(self.device)
                output = self.model(brain_images, phenotypes)
                predictions.append(output.cpu().numpy())
                targets.append(target.numpy())
        
        return np.concatenate(predictions), np.concatenate(targets)

    def prediction_accuracy_distribution(self, test_loader):
        """
        1.1 预测精度分布分析
        - 使用Shapiro-Wilk检验评估预测误差的正态性
        """
        logging.info("1.1 Starting prediction accuracy distribution analysis...")
        
        # 获取预测结果
        predictions, targets = self._get_predictions(test_loader)
        results = {}
        
        for i, target_name in enumerate(self.target_names):
            # 计算预测误差
            errors = predictions[:, i] - targets[:, i]
            
            # Shapiro-Wilk正态性检验
            shapiro_stat, shapiro_p = shapiro(errors)
            
            # 基础统计量
            error_mean = np.mean(errors)
            error_std = np.std(errors)
            
            # 绘制Q-Q图
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            stats.probplot(errors, dist="norm", plot=plt)
            plt.title(f"{target_name} Error Q-Q Plot")
            
            # 绘制误差分布直方图
            plt.subplot(1, 2, 2)
            sns.histplot(errors, kde=True)
            plt.title(f"{target_name} Error Distribution")
            plt.axvline(error_mean, color='r', linestyle='--', label=f'Mean ({error_mean:.2f})')
            plt.axvline(error_mean + error_std, color='g', linestyle=':', label=f'+1 STD ({error_std:.2f})')
            plt.axvline(error_mean - error_std, color='g', linestyle=':', label=f'-1 STD')
            plt.legend()
            
            # 保存图片
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'1_1_error_distribution_{target_name}.png')
            plt.close()
            
            results[target_name] = {
                'shapiro_statistic': float(shapiro_stat),
                'shapiro_p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05,
                'error_mean': float(error_mean),
                'error_std': float(error_std),
                'error_skewness': float(stats.skew(errors)),
                'error_kurtosis': float(stats.kurtosis(errors))
            }
        
        # 保存结果
        with open(self.metrics_dir / '1_1_prediction_accuracy_distribution.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def prediction_confidence_analysis(self, test_loader, confidence_level=0.95):
        """
        1.2 预测置信度分析
        - 预测区间的置信度检验
        - 预测偏差的显著性检验
        """
        logging.info("1.2 Starting prediction confidence analysis...")
        
        predictions, targets = self._get_predictions(test_loader)
        results = {}
        
        for i, target_name in enumerate(self.target_names):
            pred = predictions[:, i]
            true = targets[:, i]
            errors = pred - true
            
            # 计算置信区间
            ci_lower = np.percentile(errors, ((1 - confidence_level) / 2) * 100)
            ci_upper = np.percentile(errors, (1 + confidence_level) / 2 * 100)
            
            # 进行显著性检验
            # 1. 对预测偏差进行t检验
            t_stat, t_p = ttest_1samp(errors, 0)
            
            # 2. Wilcoxon符号秩检验
            w_stat, w_p = wilcoxon(errors)
            
            # 绘制Bland-Altman图
            plt.figure(figsize=(10, 6))
            mean_values = (pred + true) / 2
            differences = pred - true
            
            plt.scatter(mean_values, differences, alpha=0.5)
            plt.axhline(y=np.mean(differences), color='r', linestyle='-', label='Mean difference')
            plt.axhline(y=np.mean(differences) + 1.96*np.std(differences), color='g', linestyle='--', 
                       label='+1.96 SD')
            plt.axhline(y=np.mean(differences) - 1.96*np.std(differences), color='g', linestyle='--',
                       label='-1.96 SD')
            
            plt.xlabel('Mean of predicted and true values')
            plt.ylabel('Difference (Predicted - True)')
            plt.title(f'Bland-Altman Plot for {target_name}')
            plt.legend()
            
            # 添加统计信息
            stats_text = (f'Mean diff: {np.mean(differences):.3f}\n'
                         f'SD: {np.std(differences):.3f}\n'
                         f't-test p: {t_p:.3e}\n'
                         f'Wilcoxon p: {w_p:.3e}')
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.savefig(self.vis_dir / f'1_2_bland_altman_{target_name}.png')
            plt.close()
            
            results[target_name] = {
                'confidence_interval': {
                    'lower': float(ci_lower),
                    'upper': float(ci_upper),
                    'level': confidence_level
                },
                'significance_tests': {
                    't_test': {
                        'statistic': float(t_stat),
                        'p_value': float(t_p),
                        'significant': t_p < 0.05
                    },
                    'wilcoxon_test': {
                        'statistic': float(w_stat),
                        'p_value': float(w_p),
                        'significant': w_p < 0.05
                    }
                },
                'prediction_bias': float(np.mean(errors)),
                'prediction_variance': float(np.var(errors))
            }
        
        # 保存结果
        with open(self.metrics_dir / '1_2_prediction_confidence_analysis.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def error_pattern_analysis(self, test_loader):
        """
        1.3 误差模式分析
        包括:
        - 误差分布分析
        - 误差-特征关系分析
        - 系统性误差模式检测
        - 误差影响分析
        """
        logging.info("1.3 Starting error pattern analysis...")
        
        predictions, targets = self._get_predictions(test_loader)
        results = {'error_distribution': {}, 'feature_relationship': {}, 
                  'systematic_patterns': {}, 'error_impact': {}}
        
        for i, target_name in enumerate(self.target_names):
            pred = predictions[:, i]
            true = targets[:, i]
            errors = pred - true
            
            # 1. Error Distribution Analysis
            dist_results = self._analyze_error_distribution(errors, target_name)
            results['error_distribution'][target_name] = dist_results
            
            # 2. Error-Feature Relationship Analysis
            feature_results = self._analyze_error_feature_relationship(errors, pred, true, target_name)
            results['feature_relationship'][target_name] = feature_results
            
            # 3. Systematic Error Pattern Detection
            systematic_results = self._detect_systematic_patterns(errors, pred, target_name)
            results['systematic_patterns'][target_name] = systematic_results
            
            # 4. Error Impact Analysis
            impact_results = self._analyze_error_impact(errors, true, target_name)
            results['error_impact'][target_name] = impact_results
        
        # 保存结果
        with open(self.metrics_dir / '1_3_error_pattern_analysis.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def _detect_systematic_patterns(self, errors, predictions, target_name):
        """
        系统性误差模式检测
        - 偏差分析
        - 时序依赖性分析
        - 组特异性误差
        - 误差自相关分析
        - 交叉验证一致性
        """
        # 计算预测范围的偏差
        pred_ranges = np.percentile(predictions, [0, 25, 50, 75, 100])
        range_biases = []
        
        for i in range(len(pred_ranges)-1):
            mask = (predictions >= pred_ranges[i]) & (predictions < pred_ranges[i+1])
            if sum(mask) > 0:
                range_biases.append({
                    'range': [float(pred_ranges[i]), float(pred_ranges[i+1])],
                    'mean_error': float(np.mean(errors[mask])),
                    'std_error': float(np.std(errors[mask])),
                    'sample_count': int(sum(mask))
                })
        
        # 误差自相关分析
        autocorr = np.correlate(errors, errors, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # 只保留正半部分
        autocorr = autocorr / autocorr[0]  # 归一化
        
        # 可视化
        plt.figure(figsize=(15, 5))
        
        # 预测范围的偏差图
        plt.subplot(1, 3, 1)
        ranges = [r['range'] for r in range_biases]
        means = [r['mean_error'] for r in range_biases]
        plt.plot([np.mean(r) for r in ranges], means, 'o-')
        plt.xlabel('Prediction Range')
        plt.ylabel('Mean Error')
        plt.title('Systematic Bias by Prediction Range')
        
        # 误差自相关图
        plt.subplot(1, 3, 2)
        lags = np.arange(len(autocorr))
        plt.plot(lags, autocorr)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Error Autocorrelation')
        
        # 累积误差图
        plt.subplot(1, 3, 3)
        cum_errors = np.cumsum(errors)
        plt.plot(np.arange(len(cum_errors)), cum_errors)
        plt.xlabel('Sample Index')
        plt.ylabel('Cumulative Error')
        plt.title('Cumulative Error Plot')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / f'1_3_systematic_patterns_{target_name}.png')
        plt.close()
        
        # 进行显著性检验
        # 对不同范围的偏差进行方差分析
        bias_values = [r['mean_error'] for r in range_biases]
        if len(bias_values) > 1:
            f_stat, anova_p = stats.f_oneway(*[np.array([v]) for v in bias_values])
            # Bonferroni校正
            bonferroni_p = min(anova_p * len(bias_values), 1.0)
        else:
            f_stat, anova_p, bonferroni_p = None, None, None
        
        # 检验自相关的显著性
        # 使用Ljung-Box检验
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_stat, lb_p = acorr_ljungbox(errors, lags=[5], return_df=False)
        
        return {
            'range_bias_analysis': {
                'ranges': range_biases,
                'anova_test': {
                    'f_statistic': float(f_stat) if f_stat is not None else None,
                    'p_value': float(anova_p) if anova_p is not None else None,
                    'bonferroni_corrected_p': float(bonferroni_p) if bonferroni_p is not None else None,
                    'significant': bonferroni_p < 0.05 if bonferroni_p is not None else None
                }
            },
            'autocorrelation_analysis': {
                'ljung_box_test': {
                    'statistic': float(lb_stat[0]),
                    'p_value': float(lb_p[0]),
                    'significant': lb_p[0] < 0.05
                },
                'autocorr_values': [float(x) for x in autocorr[:10]]  # 保存前10个lag的自相关值
            }
        }
        
    def _analyze_error_impact(self, errors, true_values, target_name):
        """
        误差影响分析
        - 临床显著性分析
        - 不同严重程度的误差分布
        - 高风险误差模式识别
        - 不同组别的误差一致性
        """
        # 定义严重程度阈值（基于真实值的分布）
        severity_thresholds = np.percentile(true_values, [25, 50, 75])
        severity_groups = pd.cut(true_values, 
                               bins=[-np.inf] + list(severity_thresholds) + [np.inf],
                               labels=['Mild', 'Moderate', 'Severe', 'Very Severe'])
        
        # 按严重程度分组分析误差
        severity_analysis = []
        for severity in ['Mild', 'Moderate', 'Severe', 'Very Severe']:
            mask = severity_groups == severity
            if sum(mask) > 0:
                group_errors = errors[mask]
                severity_analysis.append({
                    'severity': severity,
                    'mean_error': float(np.mean(group_errors)),
                    'std_error': float(np.std(group_errors)),
                    'max_error': float(np.max(np.abs(group_errors))),
                    'sample_count': int(sum(mask))
                })
        
        # 识别高风险误差（定义为超过2个标准差的误差）
        error_std = np.std(errors)
        high_risk_mask = np.abs(errors) > 2 * error_std
        high_risk_errors = errors[high_risk_mask]
        high_risk_values = true_values[high_risk_mask]
        
        # 可视化
        plt.figure(figsize=(15, 5))
        
        # 不同严重程度的误差箱线图
        plt.subplot(1, 3, 1)
        error_by_severity = [errors[severity_groups == sev] 
                           for sev in ['Mild', 'Moderate', 'Severe', 'Very Severe']]
        plt.boxplot(error_by_severity, labels=['Mild', 'Moderate', 'Severe', 'Very Severe'])
        plt.ylabel('Error')
        plt.title('Errors by Severity Level')
        plt.xticks(rotation=45)
        
        # 高风险误差散点图
        plt.subplot(1, 3, 2)
        plt.scatter(true_values, errors, alpha=0.5, c='blue', label='Normal')
        plt.scatter(high_risk_values, high_risk_errors, alpha=0.7, c='red', label='High Risk')
        plt.xlabel('True Values')
        plt.ylabel('Errors')
        plt.title('High Risk Error Pattern')
        plt.legend()
        
        # 误差密度图（按严重程度）
        plt.subplot(1, 3, 3)
        for i, severity in enumerate(['Mild', 'Moderate', 'Severe', 'Very Severe']):
            mask = severity_groups == severity
            if sum(mask) > 0:
                sns.kdeplot(errors[mask], label=severity)
        plt.xlabel('Error')
        plt.ylabel('Density')
        plt.title('Error Distribution by Severity')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / f'1_3_error_impact_{target_name}.png')
        plt.close()
        
        # 进行组间显著性检验
        # 使用Kruskal-Wallis H-test（非参数方法）
        severity_groups_for_test = [errors[severity_groups == sev] 
                                  for sev in ['Mild', 'Moderate', 'Severe', 'Very Severe']]
        severity_groups_for_test = [g for g in severity_groups_for_test if len(g) > 0]
        
        if len(severity_groups_for_test) > 1:
            h_stat, h_p = stats.kruskal(*severity_groups_for_test)
            # Bonferroni校正
            bonferroni_p = min(h_p * len(severity_groups_for_test), 1.0)
        else:
            h_stat, h_p, bonferroni_p = None, None, None
        
        return {
            'severity_analysis': {
                'groups': severity_analysis,
                'kruskal_test': {
                    'statistic': float(h_stat) if h_stat is not None else None,
                    'p_value': float(h_p) if h_p is not None else None,
                    'bonferroni_corrected_p': float(bonferroni_p) if bonferroni_p is not None else None,
                    'significant': bonferroni_p < 0.05 if bonferroni_p is not None else None
                }
            },
            'high_risk_patterns': {
                'count': int(sum(high_risk_mask)),
                'percentage': float(sum(high_risk_mask) / len(errors) * 100),
                'mean_magnitude': float(np.mean(np.abs(high_risk_errors))),
                'threshold': float(2 * error_std)
            }
        }

    def _analyze_error_distribution(self, errors, target_name):
        """误差分布分析"""
        # 基础统计量
        basic_stats = {
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'std': float(np.std(errors)),
            'skewness': float(stats.skew(errors)),
            'kurtosis': float(stats.kurtosis(errors))
        }
        
        # 正态性检验
        shapiro_stat, shapiro_p = shapiro(errors)
        
        # 分位数分析
        quantiles = np.percentile(errors, [25, 50, 75])
        
        # 异常值识别 (IQR方法)
        Q1, Q3 = np.percentile(errors, [25, 75])
        IQR = Q3 - Q1
        outlier_mask = (errors < Q1 - 1.5 * IQR) | (errors > Q3 + 1.5 * IQR)
        outliers = errors[outlier_mask]
        
        # 可视化
        plt.figure(figsize=(15, 5))
        
        # 直方图和核密度估计
        plt.subplot(1, 3, 1)
        sns.histplot(errors, kde=True)
        plt.title(f'{target_name} Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Count')
        
        # 箱线图
        plt.subplot(1, 3, 2)
        sns.boxplot(y=errors)
        plt.title('Error Boxplot')
        plt.ylabel('Error')
        
        # Q-Q图
        plt.subplot(1, 3, 3)
        stats.probplot(errors, dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / f'1_3_error_distribution_{target_name}.png')
        plt.close()
        
        return {
            'basic_statistics': basic_stats,
            'normality_test': {
                'shapiro_statistic': float(shapiro_stat),
                'shapiro_p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            },
            'quantile_analysis': {
                'q25': float(quantiles[0]),
                'q50': float(quantiles[1]),
                'q75': float(quantiles[2])
            },
            'outliers': {
                'count': int(sum(outlier_mask)),
                'percentage': float(sum(outlier_mask) / len(errors) * 100),
                'min': float(np.min(outliers)) if len(outliers) > 0 else None,
                'max': float(np.max(outliers)) if len(outliers) > 0 else None
            }
        }

    def _analyze_error_feature_relationship(self, errors, predictions, targets, target_name):
        """误差-特征关系分析"""
        # 计算误差与预测值的相关性
        corr_pred, p_pred = pearsonr(errors, predictions)
        
        # 计算误差与真实值的相关性
        corr_true, p_true = pearsonr(errors, targets)
        
        # 按预测值范围分组分析误差
        pred_bins = np.linspace(min(predictions), max(predictions), 10)
        pred_groups = np.digitize(predictions, pred_bins)
        
        error_by_pred_group = []
        for i in range(1, len(pred_bins)):
            mask = pred_groups == i
            if sum(mask) > 0:
                error_by_pred_group.append({
                    'bin_range': [float(pred_bins[i-1]), float(pred_bins[i])],
                    'mean_error': float(np.mean(errors[mask])),
                    'std_error': float(np.std(errors[mask])),
                    'sample_count': int(sum(mask))
                })
        
        # 可视化
        plt.figure(figsize=(15, 5))
        
        # 误差vs预测值散点图
        plt.subplot(1, 3, 1)
        plt.scatter(predictions, errors, alpha=0.5)
        plt.xlabel('Predicted Values')
        plt.ylabel('Errors')
        plt.title(f'{target_name} Errors vs Predictions')
        fit = np.polyfit(predictions, errors, 1)
        plt.plot(predictions, np.poly1d(fit)(predictions), 'r--', 
                label=f'Trend (r={corr_pred:.2f}, p={p_pred:.2e})')
        plt.legend()

        # 误差vs真实值散点图
        plt.subplot(1, 3, 2)
        plt.scatter(targets, errors, alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Errors')
        plt.title(f'{target_name} Errors vs True Values')
        fit = np.polyfit(targets, errors, 1)
        plt.plot(targets, np.poly1d(fit)(targets), 'r--', 
                label=f'Trend (r={corr_true:.2f}, p={p_true:.2e})')
        plt.legend()

        # 分组误差箱线图
        plt.subplot(1, 3, 3)
        error_means = [group['mean_error'] for group in error_by_pred_group]
        bin_centers = [(group['bin_range'][0] + group['bin_range'][1])/2 
                      for group in error_by_pred_group]
        plt.plot(bin_centers, error_means, 'o-')
        plt.xlabel('Prediction Range')
        plt.ylabel('Mean Error')
        plt.title('Mean Error by Prediction Range')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / f'1_3_error_feature_relationship_{target_name}.png')
        plt.close()

        # 执行多重比较校正
        # 对不同预测范围组的误差进行方差分析
        groups_for_anova = [errors[pred_groups == i] for i in range(1, len(pred_bins))]
        groups_for_anova = [g for g in groups_for_anova if len(g) > 0]  # 移除空组
        
        if len(groups_for_anova) > 1:  # 确保至少有两个组可以比较
            f_stat, anova_p = stats.f_oneway(*groups_for_anova)
            # Bonferroni校正
            bonferroni_p = min(anova_p * len(groups_for_anova), 1.0)
        else:
            f_stat, anova_p, bonferroni_p = None, None, None
        
        return {
            'correlations': {
                'with_predictions': {
                    'correlation': float(corr_pred),
                    'p_value': float(p_pred),
                    'significant': p_pred < 0.05
                },
                'with_true_values': {
                    'correlation': float(corr_true),
                    'p_value': float(p_true),
                    'significant': p_true < 0.05
                }
            },
            'group_analysis': {
                'groups': error_by_pred_group,
                'anova_results': {
                    'f_statistic': float(f_stat) if f_stat is not None else None,
                    'p_value': float(anova_p) if anova_p is not None else None,
                    'bonferroni_corrected_p': float(bonferroni_p) if bonferroni_p is not None else None,
                    'significant': bonferroni_p < 0.05 if bonferroni_p is not None else None
                }
            }
        }

class ProjectionFeatureAnalysis:
    def __init__(self, results_dir):
        """
        2. 初始化2D投影特征分析类
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / 'metrics'
        self.vis_dir = self.results_dir / 'visualizations'
        self.metrics_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)

    def feature_distribution_analysis(self, data, feature_names):
        """
        2.1 Feature Distribution Analysis (Core Analysis)
        包括:
        - Global feature distribution patterns
        - Local distribution characteristics
        - Cross-feature relationships
        """
        logging.info("2.1 Starting feature distribution analysis...")
        results = {
            'global_patterns': {},
            'local_characteristics': {},
            'cross_feature_relationships': {}
        }

        # 1. Global feature distribution patterns
        global_patterns = self._analyze_global_patterns(data, feature_names)
        results['global_patterns'] = global_patterns

        # 2. Local distribution characteristics
        local_chars = self._analyze_local_characteristics(data, feature_names)
        results['local_characteristics'] = local_chars

        # 3. Cross-feature relationships
        cross_features = self._analyze_cross_features(data, feature_names)
        results['cross_feature_relationships'] = cross_features

        # 保存结果
        with open(self.metrics_dir / '2_1_feature_distribution_analysis.json', 'w') as f:
            json.dump(results, f, indent=4)

        return results

    def _analyze_global_patterns(self, data, feature_names):
        """分析全局特征分布模式"""
        results = {}
        
        for i, feature in enumerate(feature_names):
            feature_data = data[:, i]
            
            # 基本统计量
            basic_stats = {
                'mean': float(np.mean(feature_data)),
                'median': float(np.median(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data))
            }
            
            # 正态性检验
            shapiro_stat, shapiro_p = shapiro(feature_data)
            
            # 空间分布特征
            spatial_gradient = np.gradient(feature_data)
            gradient_stats = {
                'mean_gradient': float(np.mean(spatial_gradient)),
                'std_gradient': float(np.std(spatial_gradient))
            }
            
            # 可视化
            plt.figure(figsize=(15, 5))
            
            # 分布直方图
            plt.subplot(1, 3, 1)
            sns.histplot(feature_data, kde=True)
            plt.title(f'{feature} Distribution')
            
            # Q-Q图
            plt.subplot(1, 3, 2)
            stats.probplot(feature_data, dist="norm", plot=plt)
            plt.title(f'{feature} Q-Q Plot')
            
            # 梯度分布
            plt.subplot(1, 3, 3)
            sns.histplot(spatial_gradient, kde=True)
            plt.title(f'{feature} Gradient Distribution')
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'2_1_global_patterns_{feature}.png')
            plt.close()
            
            results[feature] = {
                'basic_statistics': basic_stats,
                'normality_test': {
                    'shapiro_statistic': float(shapiro_stat),
                    'shapiro_p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05  # α = 0.05
                },
                'gradient_analysis': gradient_stats
            }
            
        return results

    def _analyze_local_characteristics(self, data, feature_names):
        """分析局部分布特征"""
        results = {}
        
        for i, feature in enumerate(feature_names):
            feature_data = data[:, i]
            
            # 局部稳定性分析
            window_size = min(50, len(feature_data)//10)  # 动态窗口大小
            local_stabilities = []
            
            for j in range(0, len(feature_data)-window_size, window_size):
                window = feature_data[j:j+window_size]
                local_stabilities.append(np.std(window))
            
            # 边缘区域特征分析
            edge_threshold = np.percentile(feature_data, [10, 90])
            edge_mask = (feature_data < edge_threshold[0]) | (feature_data > edge_threshold[1])
            edge_values = feature_data[edge_mask]
            
            # 连续性评估
            continuity_scores = np.diff(feature_data)
            
            # 可视化
            plt.figure(figsize=(15, 5))
            
            # 局部稳定性
            plt.subplot(1, 3, 1)
            plt.plot(local_stabilities)
            plt.title(f'{feature} Local Stability')
            plt.xlabel('Window Index')
            plt.ylabel('Standard Deviation')
            
            # 边缘区域分布
            plt.subplot(1, 3, 2)
            sns.histplot(edge_values, kde=True)
            plt.title('Edge Region Distribution')
            
            # 连续性分布
            plt.subplot(1, 3, 3)
            sns.histplot(continuity_scores, kde=True)
            plt.title('Continuity Distribution')
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'2_1_local_characteristics_{feature}.png')
            plt.close()
            
            # 进行显著性检验
            # 对局部稳定性进行检验
            _, local_stability_p = stats.kstest(local_stabilities, 'norm')
            
            # 对边缘值分布进行检验
            _, edge_p = stats.kstest(edge_values, 'norm')
            
            # 对连续性进行检验
            _, continuity_p = stats.kstest(continuity_scores, 'norm')
            
            # Bonferroni校正
            p_values = [local_stability_p, edge_p, continuity_p]
            _, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
            
            results[feature] = {
                'local_stability': {
                    'mean_stability': float(np.mean(local_stabilities)),
                    'std_stability': float(np.std(local_stabilities)),
                    'p_value': float(p_corrected[0]),
                    'significant': p_corrected[0] < 0.05
                },
                'edge_characteristics': {
                    'edge_proportion': float(np.mean(edge_mask)),
                    'edge_mean': float(np.mean(edge_values)),
                    'edge_std': float(np.std(edge_values)),
                    'p_value': float(p_corrected[1]),
                    'significant': p_corrected[1] < 0.05
                },
                'continuity_assessment': {
                    'mean_continuity': float(np.mean(continuity_scores)),
                    'std_continuity': float(np.std(continuity_scores)),
                    'p_value': float(p_corrected[2]),
                    'significant': p_corrected[2] < 0.05
                }
            }
            
        return results

    def _analyze_cross_features(self, data, feature_names):
        """分析特征间的关系"""
        results = {}
        
        # 计算相关矩阵
        corr_matrix = np.corrcoef(data.T)
        p_values = np.zeros((data.shape[1], data.shape[1]))
        
        # 计算相关性的p值
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if i != j:
                    corr, p = stats.pearsonr(data[:, i], data[:, j])
                    p_values[i, j] = p
        
        # Bonferroni校正
        mask = ~np.eye(p_values.shape[0], dtype=bool)
        p_values_flat = p_values[mask]
        _, p_corrected, _, _ = multipletests(p_values_flat, alpha=0.05, method='bonferroni')
        p_values_corrected = np.eye(p_values.shape[0])
        p_values_corrected[mask] = p_corrected
        
        # 可视化相关矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   xticklabels=feature_names,
                   yticklabels=feature_names,
                   annot=True, 
                   cmap='coolwarm')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.vis_dir / '2_1_cross_feature_correlations.png')
        plt.close()
        
        # 创建多特征区域模式分析
        # 使用PCA分析主要变异模式
        pca = PCA()
        pca_result = pca.fit_transform(StandardScaler().fit_transform(data))
        
        # 可视化PCA结果
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(pca.explained_variance_ratio_)+1), 
                pca.explained_variance_ratio_)
        plt.title('PCA Explained Variance Ratio')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        
        # 前两个主成分的散点图
        plt.subplot(1, 2, 2)
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
        plt.title('First Two Principal Components')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / '2_1_pca_analysis.png')
        plt.close()
        
        results = {
            'correlation_analysis': {
                'correlation_matrix': corr_matrix.tolist(),
                'p_values_corrected': p_values_corrected.tolist(),
                'significant_correlations': []
            },
            'pca_analysis': {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'n_components_95var': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1)
            }
        }
        
        # 添加显著相关性的详细信息
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if p_values_corrected[i, j] < 0.05:  # α = 0.05
                    results['correlation_analysis']['significant_correlations'].append({
                        'features': [feature_names[i], feature_names[j]],
                        'correlation': float(corr_matrix[i, j]),
                        'p_value': float(p_values_corrected[i, j])
                    })
        
        return results

    def quality_assessment(self, data, feature_names):
        """
        2.2 Quality Assessment (Method Validation)
        包括:
        - Sampling density metrics
        - Projection quality metrics
        - Resolution impact analysis
        """
        logging.info("2.2 Starting quality assessment analysis...")
        results = {
            'sampling_density': {},
            'projection_quality': {},
            'resolution_impact': {}
        }

        # 1. Sampling density metrics
        density_metrics = self._analyze_sampling_density(data, feature_names)
        results['sampling_density'] = density_metrics

        # 2. Projection quality metrics
        quality_metrics = self._analyze_projection_quality(data, feature_names)
        results['projection_quality'] = quality_metrics

        # 3. Resolution impact analysis
        resolution_metrics = self._analyze_resolution_impact(data, feature_names)
        results['resolution_impact'] = resolution_metrics

        # 保存结果
        with open(self.metrics_dir / '2_2_quality_assessment.json', 'w') as f:
            json.dump(results, f, indent=4)

        return results

    def _analyze_sampling_density(self, data, feature_names):
        """分析采样密度指标"""
        results = {}
        
        for i, feature in enumerate(feature_names):
            feature_data = data[:, i]
            
            # 计算密度变化系数
            density_variation = self._calculate_density_variation(feature_data)
            
            # 计算空区域比例
            empty_regions = self._calculate_empty_regions(feature_data)
            
            # 计算均匀性度量
            uniformity = self._calculate_uniformity(feature_data)
            
            # 可视化
            plt.figure(figsize=(15, 5))
            
            # 密度分布图
            plt.subplot(1, 3, 1)
            kde = stats.gaussian_kde(feature_data)
            x_range = np.linspace(min(feature_data), max(feature_data), 100)
            plt.plot(x_range, kde(x_range))
            plt.title(f'{feature} Density Distribution')
            
            # 空区域分布
            plt.subplot(1, 3, 2)
            plt.hist(np.diff(np.sort(feature_data)), bins=50)
            plt.title('Gap Distribution')
            
            # 均匀性可视化
            plt.subplot(1, 3, 3)
            plt.hist(feature_data, bins='auto', density=True)
            plt.plot(x_range, np.ones_like(x_range)/len(x_range), '--r', 
                    label='Uniform Distribution')
            plt.title('Uniformity Analysis')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'2_2_sampling_density_{feature}.png')
            plt.close()
            
            # 统计检验
            # 对密度分布进行Kolmogorov-Smirnov检验
            _, ks_p = stats.kstest(feature_data, 'uniform', 
                                 args=(min(feature_data), max(feature_data)))
            
            results[feature] = {
                'density_variation': {
                    'coefficient': float(density_variation['coefficient']),
                    'std': float(density_variation['std']),
                    'p_value': float(ks_p),
                    'significant': ks_p < 0.05  # α = 0.05
                },
                'empty_regions': {
                    'percentage': float(empty_regions['percentage']),
                    'count': int(empty_regions['count']),
                    'mean_gap': float(empty_regions['mean_gap'])
                },
                'uniformity': {
                    'score': float(uniformity['score']),
                    'deviation': float(uniformity['deviation'])
                }
            }
        
        return results

    def _calculate_density_variation(self, data):
        """计算密度变化系数"""
        # 使用KDE估计密度
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 100)
        density = kde(x_range)
        
        return {
            'coefficient': np.std(density) / np.mean(density),
            'std': np.std(density)
        }

    def _calculate_empty_regions(self, data):
        """计算空区域统计"""
        sorted_data = np.sort(data)
        gaps = np.diff(sorted_data)
        threshold = np.mean(gaps) + 2 * np.std(gaps)
        empty_gaps = gaps[gaps > threshold]
        
        return {
            'percentage': len(empty_gaps) / len(gaps) * 100,
            'count': len(empty_gaps),
            'mean_gap': np.mean(gaps)
        }

    def _calculate_uniformity(self, data):
        """计算均匀性度量"""
        hist, _ = np.histogram(data, bins='auto', density=True)
        ideal_height = 1.0 / len(hist)
        deviation = np.sum(np.abs(hist - ideal_height))
        
        return {
            'score': 1 - deviation / 2,  # 归一化到[0,1]
            'deviation': deviation
        }

    def _analyze_projection_quality(self, data, feature_names):
        """分析投影质量指标"""
        results = {}
        
        for i, feature in enumerate(feature_names):
            feature_data = data[:, i]
            
            # 特征保持评分
            preservation_score = self._calculate_preservation_score(feature_data)
            
            # 信息损失评估
            information_loss = self._calculate_information_loss(feature_data)
            
            # 投影失真度量
            distortion = self._calculate_distortion(feature_data)
            
            # 可视化
            plt.figure(figsize=(15, 5))
            
            # 特征保持可视化
            plt.subplot(1, 3, 1)
            plt.scatter(range(len(preservation_score['local_scores'])), 
                      preservation_score['local_scores'], alpha=0.5)
            plt.title('Feature Preservation Scores')
            
            # 信息损失可视化
            plt.subplot(1, 3, 2)
            plt.plot(information_loss['cumulative_loss'])
            plt.title('Cumulative Information Loss')
            
            # 失真分布
            plt.subplot(1, 3, 3)
            plt.hist(distortion['point_wise_distortion'], bins='auto')
            plt.title('Distortion Distribution')
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'2_2_projection_quality_{feature}.png')
            plt.close()
            
            # 统计检验
            # 对失真分布进行正态性检验
            _, sw_p = stats.shapiro(distortion['point_wise_distortion'])
            
            # 对信息损失进行显著性检验
            _, t_p = stats.ttest_1samp(information_loss['local_loss'], 0)
            
            # Bonferroni校正
            _, corrected_p = multipletests([sw_p, t_p], alpha=0.05, method='bonferroni')[1]
            
            results[feature] = {
                'preservation_score': {
                    'global_score': float(preservation_score['global_score']),
                    'mean_local_score': float(np.mean(preservation_score['local_scores'])),
                    'std_local_score': float(np.std(preservation_score['local_scores']))
                },
                'information_loss': {
                    'total_loss': float(information_loss['total_loss']),
                    'mean_local_loss': float(np.mean(information_loss['local_loss'])),
                    'p_value': float(corrected_p[1]),
                    'significant': corrected_p[1] < 0.05
                },
                'distortion': {
                    'mean_distortion': float(distortion['mean_distortion']),
                    'max_distortion': float(distortion['max_distortion']),
                    'shapiro_p_value': float(corrected_p[0]),
                    'distortion_normal': corrected_p[0] >= 0.05
                }
            }
        
        return results

    def _calculate_preservation_score(self, data):
        """计算特征保持评分"""
        # 使用局部线性嵌入的思想评估特征保持程度
        n_neighbors = min(10, len(data)-1)
        local_scores = []
        
        for i in range(len(data)):
            # 找到最近邻
            dists = np.abs(data - data[i])
            nearest = np.argsort(dists)[1:n_neighbors+1]
            
            # 计算局部保持评分
            original_dists = dists[nearest]
            projected_dists = np.abs(np.arange(n_neighbors) - n_neighbors/2)
            local_scores.append(np.corrcoef(original_dists, projected_dists)[0,1])
        
        return {
            'global_score': np.mean(local_scores),
            'local_scores': local_scores
        }

    def _calculate_information_loss(self, data):
        """计算信息损失"""
        # 使用PCA计算信息损失
        pca = PCA()
        pca.fit(data.reshape(-1, 1))
        
        return {
            'total_loss': 1 - sum(pca.explained_variance_ratio_),
            'local_loss': np.abs(data - np.mean(data)) / np.std(data),
            'cumulative_loss': 1 - np.cumsum(pca.explained_variance_ratio_)
        }

    def _calculate_distortion(self, data):
        """计算投影失真"""
        # 计算点对距离的保持程度
        original_dists = distance.pdist(data.reshape(-1, 1))
        projected_dists = distance.pdist(np.arange(len(data)).reshape(-1, 1))
        
        # 归一化距离
        original_dists /= np.max(original_dists)
        projected_dists /= np.max(projected_dists)
        
        # 计算失真
        point_wise_distortion = np.abs(original_dists - projected_dists)
        
        return {
            'mean_distortion': np.mean(point_wise_distortion),
            'max_distortion': np.max(point_wise_distortion),
            'point_wise_distortion': point_wise_distortion
        }

    def _analyze_resolution_impact(self, data, feature_names):
        """分析分辨率影响"""
        results = {}
        
        for i, feature in enumerate(feature_names):
            feature_data = data[:, i]
            
            # 分析不同binning分辨率的影响
            binning_effects = self._analyze_binning_effects(feature_data)
            
            # 分析网格大小敏感性
            grid_sensitivity = self._analyze_grid_sensitivity(feature_data)
            
            # 分析特征细节保持
            detail_preservation = self._analyze_detail_preservation(feature_data)
            
            # 可视化
            plt.figure(figsize=(15, 5))
            
            # Binning效果
            plt.subplot(1, 3, 1)
            plt.plot(binning_effects['n_bins'], binning_effects['information_retention'])
            plt.title('Information Retention vs Bins')
            plt.xlabel('Number of Bins')
            plt.ylabel('Information Retention')
            
            # 网格敏感性
            plt.subplot(1, 3, 2)
            plt.plot(grid_sensitivity['grid_sizes'], grid_sensitivity['stability_scores'])
            plt.title('Grid Size Sensitivity')
            plt.xlabel('Grid Size')
            plt.ylabel('Stability Score')
            
            # 细节保持
            plt.subplot(1, 3, 3)
            plt.plot(detail_preservation['scales'], detail_preservation['detail_scores'])
            plt.title('Detail Preservation Analysis')
            plt.xlabel('Scale')
            plt.ylabel('Detail Score')
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'2_2_resolution_impact_{feature}.png')
            plt.close()
            
            # 统计检验
            # 对不同分辨率的结果进行方差分析
            f_stat, anova_p = stats.f_oneway(
                binning_effects['information_retention'],
                grid_sensitivity['stability_scores'],
                detail_preservation['detail_scores']
            )
            
            # Bonferroni校正
            _, corrected_p = multipletests([anova_p], alpha=0.05, method='bonferroni')[1]
            
            results[feature] = {
                'binning_effects': {
                    'optimal_bins': int(binning_effects['optimal_bins']),
                    'mean_retention': float(np.mean(binning_effects['information_retention'])),
                    'std_retention': float(np.std(binning_effects['information_retention']))
                },
                'grid_sensitivity': {
                    'optimal_grid': int(grid_sensitivity['optimal_grid']),
                    'mean_stability': float(np.mean(grid_sensitivity['stability_scores'])),
                    'std_stability': float(np.std(grid_sensitivity['stability_scores']))
                },
                'detail_preservation': {
                    'overall_score': float(detail_preservation['overall_score']),
                    'scale_dependency': detail_preservation['scale_dependency']
                },
                'resolution_comparison': {
                    'f_statistic': float(f_stat),
                    'p_value': float(corrected_p[0]),
                    'significant': corrected_p[0] < 0.05
                }
            }
        
        return results

    def _analyze_binning_effects(self, data):
        """分析binning效果"""
        n_bins_range = np.arange(5, min(50, len(data)), 5)
        information_retention = []
        
        for n_bins in n_bins_range:
            hist, _ = np.histogram(data, bins=n_bins)
            # 计算信息保持度（使用熵）
            p = hist / len(data)
            p = p[p > 0]  # 避免log(0)
            entropy = -np.sum(p * np.log(p))
            information_retention.append(entropy)
        
        # 找到最优bin数
        optimal_bins = n_bins_range[np.argmax(information_retention)]
        
        return {
            'n_bins': n_bins_range.tolist(),
            'information_retention': information_retention,
            'optimal_bins': optimal_bins
        }


    def _analyze_grid_sensitivity(self, data):
        """分析网格大小敏感性"""
        # 测试不同的网格大小
        grid_sizes = np.arange(10, min(100, len(data)), 10)
        stability_scores = []
        
        for grid_size in grid_sizes:
            # 创建网格
            bins = np.linspace(min(data), max(data), grid_size)
            digitized = np.digitize(data, bins)
            
            # 计算网格稳定性得分
            counts = np.bincount(digitized)
            variance = np.var(counts) / np.mean(counts)  # 归一化方差
            stability = 1 / (1 + variance)  # 转换为稳定性得分
            stability_scores.append(stability)
        
        # 找到最优网格大小
        optimal_grid = grid_sizes[np.argmax(stability_scores)]
        
        return {
            'grid_sizes': grid_sizes.tolist(),
            'stability_scores': stability_scores,
            'optimal_grid': optimal_grid
        }

    def _analyze_detail_preservation(self, data):
        """分析特征细节保持程度"""
        # 在不同尺度下分析细节保持
        scales = np.arange(0.1, 1.1, 0.1)
        detail_scores = []
        scale_effects = {}
        
        for scale in scales:
            # 对数据进行降采样
            n_samples = int(len(data) * scale)
            if n_samples < 2:  # 确保至少有2个样本
                continue
                
            indices = np.linspace(0, len(data)-1, n_samples, dtype=int)
            sampled_data = data[indices]
            
            # 计算细节保持得分
            # 使用局部变异性作为细节度量
            local_var = np.var(np.diff(sampled_data))
            original_var = np.var(np.diff(data))
            detail_score = local_var / original_var if original_var != 0 else 0
            
            detail_scores.append(detail_score)
            scale_effects[float(scale)] = {
                'detail_score': float(detail_score),
                'n_samples': int(n_samples)
            }
        
        # 计算总体得分
        overall_score = np.mean(detail_scores)
        
        # 分析尺度依赖性
        scale_dependency = {
            'strong_preservation': float(scales[np.argmax(detail_scores)]),
            'weak_preservation': float(scales[np.argmin(detail_scores)]),
            'mean_preservation': float(np.mean(detail_scores))
        }
        
        return {
            'scales': scales.tolist(),
            'detail_scores': detail_scores,
            'overall_score': overall_score,
            'scale_dependency': scale_dependency,
            'scale_effects': scale_effects
        }
    

    def reliability_analysis(self, data, feature_names, n_splits=5):
        """
        2.3 Reliability Analysis (Optional)
        包括:
        - Cross-validation tests
        - Sampling stability analysis
        - Feature preservation verification
        - Regional consistency checks
        - Boundary analysis
        """
        logging.info("2.3 Starting reliability analysis...")
        results = {
            'cross_validation': {},
            'boundary_analysis': {}
        }
        
        # 1. Cross-validation tests
        cv_results = self._analyze_cross_validation(data, feature_names, n_splits)
        results['cross_validation'] = cv_results
        
        # 2. Boundary analysis
        boundary_results = self._analyze_boundary(data, feature_names)
        results['boundary_analysis'] = boundary_results
        
        # 保存结果
        with open(self.metrics_dir / '2_3_reliability_analysis.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        return results

    def _analyze_cross_validation(self, data, feature_names, n_splits):
        """执行交叉验证分析"""
        results = {}
        
        for i, feature in enumerate(feature_names):
            feature_data = data[:, i]
            
            # 准备交叉验证
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            stability_scores = []
            preservation_scores = []
            consistency_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(feature_data)):
                train_data = feature_data[train_idx]
                val_data = feature_data[val_idx]
                
                # 采样稳定性分析
                stability = self._calculate_sampling_stability(train_data, val_data)
                stability_scores.append(stability)
                
                # 特征保持验证
                preservation = self._calculate_feature_preservation(train_data, val_data)
                preservation_scores.append(preservation)
                
                # 区域一致性检查
                consistency = self._calculate_regional_consistency(train_data, val_data)
                consistency_scores.append(consistency)
            
            # 进行统计检验
            # 对稳定性得分进行检验
            t_stat_stability, p_stability = stats.ttest_1samp(stability_scores, 0.5)
            
            # 对保持度得分进行检验
            t_stat_preservation, p_preservation = stats.ttest_1samp(preservation_scores, 0.5)
            
            # 对一致性得分进行检验
            t_stat_consistency, p_consistency = stats.ttest_1samp(consistency_scores, 0.5)
            
            # Bonferroni校正
            p_values = [p_stability, p_preservation, p_consistency]
            _, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
            
            # 可视化结果
            plt.figure(figsize=(15, 5))
            
            # 稳定性分数分布
            plt.subplot(1, 3, 1)
            plt.boxplot(stability_scores, labels=['Stability'])
            plt.title(f'Stability Scores\np={p_corrected[0]:.3f}')
            
            # 特征保持分数分布
            plt.subplot(1, 3, 2)
            plt.boxplot(preservation_scores, labels=['Preservation'])
            plt.title(f'Preservation Scores\np={p_corrected[1]:.3f}')
            
            # 一致性分数分布
            plt.subplot(1, 3, 3)
            plt.boxplot(consistency_scores, labels=['Consistency'])
            plt.title(f'Consistency Scores\np={p_corrected[2]:.3f}')
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'2_3_cross_validation_{feature}.png')
            plt.close()
            
            # 记录结果
            results[feature] = {
                'sampling_stability': {
                    'mean': float(np.mean(stability_scores)),
                    'std': float(np.std(stability_scores)),
                    't_statistic': float(t_stat_stability),
                    'p_value': float(p_corrected[0]),
                    'significant': p_corrected[0] < 0.05
                },
                'feature_preservation': {
                    'mean': float(np.mean(preservation_scores)),
                    'std': float(np.std(preservation_scores)),
                    't_statistic': float(t_stat_preservation),
                    'p_value': float(p_corrected[1]),
                    'significant': p_corrected[1] < 0.05
                },
                'regional_consistency': {
                    'mean': float(np.mean(consistency_scores)),
                    'std': float(np.std(consistency_scores)),
                    't_statistic': float(t_stat_consistency),
                    'p_value': float(p_corrected[2]),
                    'significant': p_corrected[2] < 0.05
                }
            }
            
        return results

    def _calculate_sampling_stability(self, train_data, val_data):
        """计算采样稳定性"""
        # 使用KS检验比较分布
        _, p_value = stats.ks_2samp(train_data, val_data)
        return 1 - p_value  # 转换为稳定性得分
    
    def _calculate_feature_preservation(self, train_data, val_data):
        """计算特征保持度"""
        # 比较基本统计特征
        train_stats = [np.mean(train_data), np.std(train_data), stats.skew(train_data)]
        val_stats = [np.mean(val_data), np.std(val_data), stats.skew(val_data)]
        
        # 计算相对差异
        rel_diff = np.abs(np.array(train_stats) - np.array(val_stats)) / np.array(train_stats)
        return 1 - np.mean(rel_diff)  # 转换为保持度得分
    
    def _calculate_regional_consistency(self, train_data, val_data):
        """计算区域一致性"""
        # 将数据分成区域
        regions = 5
        train_regions = np.array_split(np.sort(train_data), regions)
        val_regions = np.array_split(np.sort(val_data), regions)
        
        # 比较每个区域的一致性
        consistencies = []
        for tr, vr in zip(train_regions, val_regions):
            tr_mean, vr_mean = np.mean(tr), np.mean(vr)
            rel_diff = abs(tr_mean - vr_mean) / tr_mean
            consistencies.append(1 - rel_diff)
            
        return np.mean(consistencies)

    def _analyze_boundary(self, data, feature_names):
        """执行边界分析"""
        results = {}
        
        for i, feature in enumerate(feature_names):
            feature_data = data[:, i]
            
            # 计算边界效应
            edge_effects = self._calculate_edge_effects(feature_data)
            
            # 计算边界稳定性
            boundary_stability = self._calculate_boundary_stability(feature_data)
            
            # 计算过渡平滑性
            transition_smoothness = self._calculate_transition_smoothness(feature_data)
            
            # 进行统计检验
            _, p_edge = stats.ttest_1samp(edge_effects['scores'], 0)
            _, p_stability = stats.ttest_1samp(boundary_stability['scores'], 0.5)
            _, p_smoothness = stats.ttest_1samp(transition_smoothness['scores'], 0.5)
            
            # Bonferroni校正
            p_values = [p_edge, p_stability, p_smoothness]
            _, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
            
            # 可视化结果
            plt.figure(figsize=(15, 5))
            
            # 边界效应
            plt.subplot(1, 3, 1)
            plt.plot(edge_effects['positions'], edge_effects['scores'])
            plt.title(f'Edge Effects\np={p_corrected[0]:.3f}')
            plt.xlabel('Position')
            plt.ylabel('Effect Score')
            
            # 边界稳定性
            plt.subplot(1, 3, 2)
            plt.plot(boundary_stability['positions'], boundary_stability['scores'])
            plt.title(f'Boundary Stability\np={p_corrected[1]:.3f}')
            plt.xlabel('Position')
            plt.ylabel('Stability Score')
            
            # 过渡平滑性
            plt.subplot(1, 3, 3)
            plt.plot(transition_smoothness['positions'], transition_smoothness['scores'])
            plt.title(f'Transition Smoothness\np={p_corrected[2]:.3f}')
            plt.xlabel('Position')
            plt.ylabel('Smoothness Score')
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'2_3_boundary_analysis_{feature}.png')
            plt.close()
            
            # 记录结果
            results[feature] = {
                'edge_effects': {
                    'mean': float(np.mean(edge_effects['scores'])),
                    'max': float(np.max(edge_effects['scores'])),
                    'p_value': float(p_corrected[0]),
                    'significant': p_corrected[0] < 0.05
                },
                'boundary_stability': {
                    'mean': float(np.mean(boundary_stability['scores'])),
                    'min': float(np.min(boundary_stability['scores'])),
                    'p_value': float(p_corrected[1]),
                    'significant': p_corrected[1] < 0.05
                },
                'transition_smoothness': {
                    'mean': float(np.mean(transition_smoothness['scores'])),
                    'min': float(np.min(transition_smoothness['scores'])),
                    'p_value': float(p_corrected[2]),
                    'significant': p_corrected[2] < 0.05
                }
            }
        
        return results
    
    def _calculate_edge_effects(self, data):
        """计算边界效应"""
        edge_width = len(data) // 10
        positions = np.arange(len(data))
        scores = []
        
        for i in positions:
            # 计算到最近边界的距离
            dist_to_edge = min(i, len(data)-1-i)
            # 计算边界效应得分
            if dist_to_edge < edge_width:
                score = 1 - (dist_to_edge / edge_width)
            else:
                score = 0
            scores.append(score)
        
        return {
            'positions': positions.tolist(),
            'scores': scores
        }
    
    def _calculate_boundary_stability(self, data):
        """计算边界稳定性"""
        window_size = len(data) // 20
        positions = np.arange(len(data))
        scores = []
        
        # 使用滑动窗口计算局部稳定性
        for i in range(len(data) - window_size + 1):
            window = data[i:i+window_size]
            score = 1 / (1 + np.std(window))
            scores.append(score)
        
        # 补充末尾的得分
        scores.extend([scores[-1]] * (window_size - 1))
        
        return {
            'positions': positions.tolist(),
            'scores': scores
        }
    
    def _calculate_transition_smoothness(self, data):
        """计算过渡平滑性"""
        # 计算一阶导数
        gradients = np.gradient(data)
        positions = np.arange(len(data))
        scores = []
        
        window_size = 5
        padded_gradients = np.pad(gradients, (window_size//2, window_size//2), mode='edge')
        
        # 计算局部平滑度
        for i in range(len(data)):
            window = padded_gradients[i:i+window_size]
            score = 1 / (1 + np.std(window))
            scores.append(score)
        
        return {
            'positions': positions.tolist(),
            'scores': scores
        }


class BrainComparisonAnalysis:
    def __init__(self, results_dir):
        """
        3. 初始化左右脑比较分析类
        Args:
            results_dir: 结果保存目录
        """
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / 'metrics'
        self.vis_dir = self.results_dir / 'visualizations'
        self.metrics_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)
        
        # 设置目标变量名称
        self.target_names = ['sum_att', 'sum_agg']
        
    def performance_comparison(self, left_preds, right_preds, left_targets, right_targets):
        """
        3.1 Performance Comparison
        对左右脑预测性能进行比较
        """
        logging.info("3.1 Starting performance comparison...")
        results = {}
        
        for i, target_name in enumerate(self.target_names):
            # 提取当前目标的预测结果
            left_pred = left_preds[:, i]
            right_pred = right_preds[:, i]
            left_target = left_targets[:, i]
            right_target = right_targets[:, i]
            
            # 计算性能指标
            left_metrics = self._calculate_metrics(left_pred, left_target)
            right_metrics = self._calculate_metrics(right_pred, right_target)
            
            # 执行统计检验
            # 比较MSE
            f_stat, p_value = self._compare_metrics(
                left_pred - left_target,
                right_pred - right_target
            )
            
            # Wilcoxon符号秩检验
            w_stat, w_p = wilcoxon(np.abs(left_pred - left_target),
                                 np.abs(right_pred - right_target))
            
            # Bonferroni校正
            _, corrected_p = multipletests([p_value, w_p], alpha=0.05, method='bonferroni')[1]
            
            # 可视化比较
            plt.figure(figsize=(15, 5))
            
            # MSE比较
            plt.subplot(1, 3, 1)
            plt.bar(['Left Brain', 'Right Brain'], 
                   [left_metrics['mse'], right_metrics['mse']])
            plt.title(f'MSE Comparison\np={corrected_p[0]:.3f}')
            plt.ylabel('Mean Squared Error')
            
            # R²比较
            plt.subplot(1, 3, 2)
            plt.bar(['Left Brain', 'Right Brain'], 
                   [left_metrics['r2'], right_metrics['r2']])
            plt.title('R² Score Comparison')
            plt.ylabel('R² Score')
            
            # 相关性比较
            plt.subplot(1, 3, 3)
            plt.bar(['Left Brain', 'Right Brain'], 
                   [left_metrics['correlation'], right_metrics['correlation']])
            plt.title('Correlation Comparison')
            plt.ylabel('Correlation Coefficient')
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'3_1_performance_comparison_{target_name}.png')
            plt.close()
            
            # 记录结果
            results[target_name] = {
                'left_brain': left_metrics,
                'right_brain': right_metrics,
                'statistical_tests': {
                    'f_test': {
                        'statistic': float(f_stat),
                        'p_value': float(corrected_p[0]),
                        'significant': corrected_p[0] < 0.05
                    },
                    'wilcoxon_test': {
                        'statistic': float(w_stat),
                        'p_value': float(corrected_p[1]),
                        'significant': corrected_p[1] < 0.05
                    }
                }
            }
        
        # 保存结果
        with open(self.metrics_dir / '3_1_performance_comparison.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def feature_importance_analysis(self, left_model, right_model, test_loader):
        """
        3.2 Feature Importance Analysis
        分析左右脑模型中特征的重要性差异
        """
        logging.info("3.2 Starting feature importance analysis...")
        results = {}
        
        # 获取特征重要性
        left_importance = self._calculate_feature_importance(left_model, test_loader)
        right_importance = self._calculate_feature_importance(right_model, test_loader)
        
        for target_name in self.target_names:
            # 比较特征重要性分布
            t_stat, p_value = stats.ttest_ind(
                left_importance[target_name],
                right_importance[target_name]
            )
            
            # Mann-Whitney U检验
            u_stat, u_p = stats.mannwhitneyu(
                left_importance[target_name],
                right_importance[target_name]
            )
            
            # Bonferroni校正
            _, corrected_p = multipletests([p_value, u_p], alpha=0.05, method='bonferroni')[1]
            
            # 可视化
            plt.figure(figsize=(15, 5))
            
            # 特征重要性分布
            plt.subplot(1, 3, 1)
            plt.boxplot([left_importance[target_name], right_importance[target_name]], 
                       labels=['Left Brain', 'Right Brain'])
            plt.title(f'Feature Importance Distribution\np={corrected_p[0]:.3f}')
            plt.ylabel('Importance Score')
            
            # 特征重要性相关性
            plt.subplot(1, 3, 2)
            plt.scatter(left_importance[target_name], right_importance[target_name], alpha=0.5)
            plt.plot([0, 1], [0, 1], 'r--')  # 对角线
            plt.xlabel('Left Brain Feature Importance')
            plt.ylabel('Right Brain Feature Importance')
            plt.title('Feature Importance Correlation')
            
            # 差异分布
            plt.subplot(1, 3, 3)
            differences = left_importance[target_name] - right_importance[target_name]
            plt.hist(differences, bins=20)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Importance Differences Distribution')
            plt.xlabel('Left - Right Importance')
            
            plt.tight_layout()
            plt.savefig(self.vis_dir / f'3_2_feature_importance_{target_name}.png')
            plt.close()
            
            # 记录结果
            results[target_name] = {
                'left_importance': {
                    'mean': float(np.mean(left_importance[target_name])),
                    'std': float(np.std(left_importance[target_name])),
                    'top_features': self._get_top_features(left_importance[target_name])
                },
                'right_importance': {
                    'mean': float(np.mean(right_importance[target_name])),
                    'std': float(np.std(right_importance[target_name])),
                    'top_features': self._get_top_features(right_importance[target_name])
                },
                'statistical_tests': {
                    't_test': {
                        'statistic': float(t_stat),
                        'p_value': float(corrected_p[0]),
                        'significant': corrected_p[0] < 0.05
                    },
                    'mann_whitney': {
                        'statistic': float(u_stat),
                        'p_value': float(corrected_p[1]),
                        'significant': corrected_p[1] < 0.05
                    }
                }
            }
        
        # 保存结果
        with open(self.metrics_dir / '3_2_feature_importance.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    # def clinical_implications(self, left_results, right_results):
    #     """
    #     3.3 Clinical Implications Analysis
    #     分析左右脑预测结果的临床意义
    #     """
    #     logging.info("3.3 Starting clinical implications analysis...")
    #     results = {}
        
    #     for target_name in self.target_names:
    #         # 分析预测差异的临床意义
    #         clinical_diff = self._analyze_clinical_differences(
    #             left_results[target_name],
    #             right_results[target_name]
    #         )
            
    #         # 分析预测准确度的临床相关性
    #         clinical_acc = self._analyze_clinical_accuracy(
    #             left_results[target_name],
    #             right_results[target_name]
    #         )
            
    #         # 进行统计检验
    #         # 对临床差异进行检验
    #         t_stat, p_value = stats.ttest_ind(
    #             clinical_diff['left_scores'],
    #             clinical_diff['right_scores']
    #         )
            
    #         # 对临床准确度进行检验
    #         chi2_stat, chi2_p = stats.chi2_contingency(
    #             clinical_acc['contingency_table']
    #         )[:2]
            
    #         # Bonferroni校正
    #         _, corrected_p = multipletests([p_value, chi2_p], alpha=0.05, method='bonferroni')[1]
            
    #         # 可视化
    #         plt.figure(figsize=(15, 5))
            
    #         # 临床差异分布
    #         plt.subplot(1, 3, 1)
    #         plt.boxplot([clinical_diff['left_scores'], clinical_diff['right_scores']], 
    #                    labels=['Left Brain', 'Right Brain'])
    #         plt.title(f'Clinical Score Distribution\np={corrected_p[0]:.3f}')
    #         plt.ylabel('Clinical Score')
            
    #         # 准确度比较
    #         plt.subplot(1, 3, 2)
    #         labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    #         left_acc = clinical_acc['left_confusion_matrix'].ravel()
    #         right_acc = clinical_acc['right_confusion_matrix'].ravel()
    #         x = np.arange(len(labels))
    #         width = 0.35
            
    #         plt.bar(x - width/2, left_acc, width, label='Left Brain')
    #         plt.bar(x + width/2, right_acc, width, label='Right Brain')
    #         plt.xticks(x, labels, rotation=45)
    #         plt.title('Confusion Matrix Comparison')
    #         plt.legend()
            
    #         # ROC曲线比较
    #         plt.subplot(1, 3, 3)
    #         plt.plot(clinical_acc['left_roc']['fpr'], clinical_acc['left_roc']['tpr'],
    #                 label=f'Left Brain (AUC = {clinical_acc["left_roc"]["auc"]:.2f})')
    #         plt.plot(clinical_acc['right_roc']['fpr'], clinical_acc['right_roc']['tpr'],
    #                 label=f'Right Brain (AUC = {clinical_acc["right_roc"]["auc"]:.2f})')
    #         plt.plot([0, 1], [0, 1], 'r--')
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.title('ROC Curve Comparison')
    #         plt.legend()
            
    #         plt.tight_layout()
    #         plt.savefig(self.vis_dir / f'3_3_clinical_implications_{target_name}.png')
    #         plt.close()
            
    #         # 记录结果
    #         results[target_name] = {
    #             'clinical_differences': clinical_diff['summary'],
    #             'clinical_accuracy': {
    #                 'left_metrics': clinical_acc['left_metrics'],
    #                 'right_metrics': clinical_acc['right_metrics']
    #             },
    #             'statistical_tests': {
    #                 'difference_test': {
    #                     'statistic': float(t_stat),
    #                     'p_value': float(corrected_p[0]),
    #                     'significant': corrected_p[0] < 0.05
    #                 },
    #                 'accuracy_test': {
    #                     'statistic': float(chi2_stat),
    #                     'p_value': float(corrected_p[1]),
    #                     'significant': corrected_p[1] < 0.05
    #                 }
    #             }
    #         }
        
    #     # 保存结果
    #     with open(self.metrics_dir / '3_3_clinical_implications.json', 'w') as f:
    #         json.dump(results, f, indent=4)
        
    #     return results

    # def _calculate_metrics(self, predictions, targets):
    #     """计算性能指标"""
    #     mse = mean_squared_error(targets, predictions)
    #     mae = mean_absolute_error(targets, predictions)
    #     r2 = r2_score(targets, predictions)
    #     correlation = np.corrcoef(predictions, targets)[0,1]
        
    #     return {
    #         'mse': float(mse),
    #         'mae': float(mae),
    #         'r2': float(r2),
    #         'correlation': float(correlation)
    #     }
    
    # def _compare_metrics(self, left_errors, right_errors):
    #     """比较性能指标"""
    #     return stats.f_oneway(left_errors, right_errors)
    
    # def _calculate_feature_importance(self, model, test_loader):
    #     """计算特征重要性分数"""
    #     importances = {}
    #     for target_name in self.target_names:
    #         importances[target_name] = []
            
    #         # 使用梯度方法计算特征重要性
    #         model.eval()
    #         with torch.no_grad():
    #             for data, phenotypes, _ in test_loader:
    #                 scores = []
    #                 for i in range(data.shape[1]):  # 遍历特征
    #                     perturbed = data.clone()
    #                     perturbed[:, i] = 0  # 特征置零
                        
    #                     original_output = model(data, phenotypes)
    #                     perturbed_output = model(perturbed, phenotypes)
                        
    #                     importance = torch.norm(original_output - perturbed_output)
    #                     scores.append(float(importance))
                    
    #                 importances[target_name].extend(scores)
        
    #     return importances
    
    # def _get_top_features(self, importance_scores, top_k=5):
    #     """获取最重要的特征"""
    #     top_indices = np.argsort(importance_scores)[-top_k:]
    #     return [{
    #         'index': int(idx),
    #         'importance': float(importance_scores[idx])
    #     } for idx in top_indices]

    # def _analyze_clinical_differences(self, left_results, right_results):
    #     """分析左右脑预测结果的临床差异"""
    #     # 计算左右脑预测的差异分数
    #     left_scores = []
    #     right_scores = []
        
    #     # 根据预测结果计算临床分数
    #     for l_res, r_res in zip(left_results, right_results):
    #         left_scores.append(self._calculate_clinical_score(l_res))
    #         right_scores.append(self._calculate_clinical_score(r_res))
            
    #     # 计算摘要统计量
    #     summary = {
    #         'left_brain': {
    #             'mean': float(np.mean(left_scores)),
    #             'std': float(np.std(left_scores)),
    #             'median': float(np.median(left_scores))
    #         },
    #         'right_brain': {
    #             'mean': float(np.mean(right_scores)),
    #             'std': float(np.std(right_scores)),
    #             'median': float(np.median(right_scores))
    #         },
    #         'difference': {
    #             'mean': float(np.mean(np.array(left_scores) - np.array(right_scores))),
    #             'std': float(np.std(np.array(left_scores) - np.array(right_scores)))
    #         }
    #     }
        
    #     return {
    #         'left_scores': left_scores,
    #         'right_scores': right_scores,
    #         'summary': summary
    #     }
    
    # def _analyze_clinical_accuracy(self, left_results, right_results, threshold=0.5):
    #     """分析预测准确度的临床相关性"""
    #     # 计算混淆矩阵
    #     left_cm = self._calculate_confusion_matrix(left_results, threshold)
    #     right_cm = self._calculate_confusion_matrix(right_results, threshold)
        
    #     # 计算ROC曲线
    #     left_roc = self._calculate_roc_curve(left_results)
    #     right_roc = self._calculate_roc_curve(right_results)
        
    #     # 计算临床指标
    #     left_metrics = {
    #         'sensitivity': float(left_cm[1, 1] / (left_cm[1, 1] + left_cm[1, 0])),
    #         'specificity': float(left_cm[0, 0] / (left_cm[0, 0] + left_cm[0, 1])),
    #         'ppv': float(left_cm[1, 1] / (left_cm[1, 1] + left_cm[0, 1])),
    #         'npv': float(left_cm[0, 0] / (left_cm[0, 0] + left_cm[1, 0]))
    #     }
        
    #     right_metrics = {
    #         'sensitivity': float(right_cm[1, 1] / (right_cm[1, 1] + right_cm[1, 0])),
    #         'specificity': float(right_cm[0, 0] / (right_cm[0, 0] + right_cm[0, 1])),
    #         'ppv': float(right_cm[1, 1] / (right_cm[1, 1] + right_cm[0, 1])),
    #         'npv': float(right_cm[0, 0] / (right_cm[0, 0] + right_cm[1, 0]))
    #     }
        
    #     return {
    #         'left_confusion_matrix': left_cm,
    #         'right_confusion_matrix': right_cm,
    #         'left_roc': left_roc,
    #         'right_roc': right_roc,
    #         'left_metrics': left_metrics,
    #         'right_metrics': right_metrics,
    #         'contingency_table': np.array([
    #             [left_cm[0, 0], right_cm[0, 0]],
    #             [left_cm[0, 1], right_cm[0, 1]],
    #             [left_cm[1, 0], right_cm[1, 0]],
    #             [left_cm[1, 1], right_cm[1, 1]]
    #         ])
    #     }
    
    # def _calculate_clinical_score(self, prediction):
    #     """计算单个预测的临床分数"""
    #     # 这里可以根据具体的临床标准进行调整
    #     return float(prediction)
    
    # def _calculate_confusion_matrix(self, predictions, threshold):
    #     """计算混淆矩阵"""
    #     binary_preds = (np.array(predictions) >= threshold).astype(int)
    #     binary_truth = (np.array(predictions) >= threshold).astype(int)
        
    #     cm = np.zeros((2, 2))
    #     for pred, true in zip(binary_preds, binary_truth):
    #         cm[pred, true] += 1
        
    #     return cm
    
    # def _calculate_roc_curve(self, predictions):
    #     """计算ROC曲线"""
    #     # 使用不同的阈值计算TPR和FPR
    #     thresholds = np.linspace(0, 1, 100)
    #     tpr = []
    #     fpr = []
        
    #     for threshold in thresholds:
    #         cm = self._calculate_confusion_matrix(predictions, threshold)
    #         tpr.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]))
    #         fpr.append(cm[0, 1] / (cm[0, 1] + cm[0, 0]))
        
    #     # 计算AUC
    #     auc = np.trapz(tpr, fpr)
        
    #     return {
    #         'tpr': tpr,
    #         'fpr': fpr,
    #         'auc': float(auc)
    #     }