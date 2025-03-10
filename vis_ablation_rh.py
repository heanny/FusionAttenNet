import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

class AblationAnalysis:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.setup_paths()
        
    def setup_paths(self):
        """Set up directory paths"""
        self.logs_dir = self.base_dir / 'logs'
        self.attention_dir = self.base_dir / 'attention'
        self.error_dir = self.base_dir / 'error_analysis'
        self.plots_dir = Path('vis_clau_ablation_study_rh')
        
        # Ensure directories exist
        self.plots_dir.mkdir(exist_ok=True)
        for subdir in ['attention', 'error']:
            (self.plots_dir / subdir).mkdir(exist_ok=True)

    def _plot_spatial_attention_comparison_full(self, baseline_maps, no_feature_maps):
        """Plot spatial attention heatmaps comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        # Calculate mean attention maps
        # Assuming the original shape is (N, 16, 16) where N is the number of samples 
        mean_baseline = np.mean(baseline_maps.reshape(-1, 16, 16), axis=0)
        mean_no_feature = np.mean(no_feature_maps.reshape(-1, 16, 16), axis=0)

        # Find global min and max for consistent colorbar
        vmin = min(mean_baseline.min(), mean_no_feature.min())
        vmax = max(mean_baseline.max(), mean_no_feature.max())

        # Plot baseline
        im1 = axes[0].imshow(mean_baseline, cmap='hot', vmin=vmin, vmax=vmax)
        axes[0].set_title('FusionAttenNet: Spatial Attention (Right-semi Brain)')
        
        # Plot no_feature_attention
        im2 = axes[1].imshow(mean_no_feature, cmap='hot', vmin=vmin, vmax=vmax)
        axes[1].set_title('No-Feature-Attention model: Spatial Attention (Right-semi Brain)')

        # 添加统一的colorbar
        # 调整位置使其位于两个子图之间
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
        fig.colorbar(im1, cax=cbar_ax)
        
        plt.savefig(self.plots_dir / 'attention' / 'spatial_attention_comparison.png')
        plt.close()

        
    def _plot_spatial_attention_comparison(self, baseline_maps, no_feature_maps):
        """Plot spatial attention heatmaps comparison for each feature channel"""
        features = ['Thickness', 'Volume', 'Surface Area', 'Gray-White Ratio']
        
        # 创建2x4的子图布局
        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1], left=0.1)
        
        # Reshape to (N, 4, 16, 16) to separate features
        baseline_reshaped = baseline_maps.reshape(-1, 4, 16, 16)
        no_feature_reshaped = no_feature_maps.reshape(-1, 4, 16, 16)
        
        # 计算每个特征的平均attention map
        mean_baseline = np.mean(baseline_reshaped, axis=0)  # (4, 16, 16)
        mean_no_feature = np.mean(no_feature_reshaped, axis=0)  # (4, 16, 16)
        
        # 找到所有maps的最大最小值，用于统一colorbar范围
        vmin = min(mean_baseline.min(), mean_no_feature.min())
        vmax = max(mean_baseline.max(), mean_no_feature.max())
        axes = []
        for i, feature in enumerate(features):
            ax_top = fig.add_subplot(gs[0, i])
            ax_bottom = fig.add_subplot(gs[1, i])
            axes.extend([ax_top, ax_bottom])
            # # Plot baseline
            # im1 = axes[0, i].imshow(mean_baseline[i], cmap='hot', 
            #                     interpolation='nearest',
            #                     vmin=vmin, vmax=vmax)
            # axes[0, i].set_title(f'FusionAttenNet\n{feature}')
            
            # # Plot no_feature_attention
            # im2 = axes[1, i].imshow(mean_no_feature[i], cmap='hot',
            #                     interpolation='nearest',
            #                     vmin=vmin, vmax=vmax)
            # axes[1, i].set_title(f'No-Feature-Attention\n{feature}')
            
            # # 移除坐标轴刻度
            # axes[0, i].set_xticks([])
            # axes[0, i].set_yticks([])
            # axes[1, i].set_xticks([])
            # axes[1, i].set_yticks([])
            im1 = ax_top.imshow(mean_baseline[i], cmap='hot', 
                           interpolation='nearest',
                           vmin=vmin, vmax=vmax)
            im2 = ax_bottom.imshow(mean_no_feature[i], cmap='hot',
                              interpolation='nearest',
                              vmin=vmin, vmax=vmax)
            ax_top.set_title(f'FusionAttenNet\n{features[i]} (Right-semi Brain)')
            ax_bottom.set_title(f'No-Feature-Attention\n{features[i]} (Right-semi Brain)')
            
            # 移除坐标轴刻度
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            ax_bottom.set_xticks([])
            ax_bottom.set_yticks([])
        
        # 在最左侧添加colorbar
        cbar_ax = fig.add_axes([0.02, 0.15, 0.02, 0.7])
        fig.colorbar(im1, cax=cbar_ax, orientation='vertical')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attention' / 'spatial_attention_comparison_by_feature.png')
        plt.close()


    def _plot_channel_attention_comparison(self):
        """Plot channel attention weights comparison for all models"""
        # Load channel attention weights for all models
        baseline_weights = np.load(self.attention_dir / 'raw_data' / 'baseline_channel_attention_weights.npy')
        no_feature_weights = np.load(self.attention_dir / 'raw_data' / 'no_feature_attention_channel_attention_weights.npy')
        no_spatial_weights = np.load(self.attention_dir / 'raw_data' / 'no_spatial_attention_channel_attention_weights.npy')
        
        # print("Baseline weights shape:", baseline_weights.shape)
        # print("No feature weights shape:", no_feature_weights.shape)
        # print("No spatial weights shape:", no_spatial_weights.shape)

        # Calculate mean weights
        mean_baseline = np.mean(baseline_weights, axis=0).reshape(4, -1).mean(axis=1)
        mean_no_feature = np.mean(no_feature_weights, axis=0).reshape(4, -1).mean(axis=1)
        mean_no_spatial = np.mean(no_spatial_weights, axis=0).reshape(4, -1).mean(axis=1)
        
        # Calculate standard deviations
        std_baseline = np.std(baseline_weights, axis=0).reshape(4, -1).mean(axis=1)
        std_no_feature = np.std(no_feature_weights, axis=0).reshape(4, -1).mean(axis=1)
        std_no_spatial = np.std(no_spatial_weights, axis=0).reshape(4, -1).mean(axis=1)
        

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        features = ['Thickness', 'Volume', 'Surface Area', 'Gray-White Ratio']
        x = np.arange(len(features))
        width = 0.25

        # Plot bars with error bars
        bars1 = ax.bar(x - width, mean_baseline, width, 
                    yerr=std_baseline,
                    label='FusionAttenNet',
                    capsize=5,
                    error_kw=dict(capthick=1, elinewidth=1))
        
        bars2 = ax.bar(x, mean_no_feature, width, 
                    yerr=std_no_feature,
                    label='No Feature Attention',
                    capsize=5,
                    error_kw=dict(capthick=1, elinewidth=1))
        
        bars3 = ax.bar(x + width, mean_no_spatial, width,
                    yerr=std_no_spatial,
                    label='No Spatial Attention',
                    capsize=5,
                    error_kw=dict(capthick=1, elinewidth=1))
        
            # Add value labels on top of each bar
        for i, v in enumerate(mean_baseline):
            ax.text(x[i] - width, v + std_baseline[i], 
                    f'{v:.3f}',
                    ha='center', va='bottom')
        
        for i, v in enumerate(mean_no_feature):
            ax.text(x[i], v + std_no_feature[i],
                    f'{v:.3f}',
                    ha='center', va='bottom')
        
        for i, v in enumerate(mean_no_spatial):
            ax.text(x[i] + width, v + std_no_spatial[i],
                    f'{v:.3f}',
                    ha='center', va='bottom')
            
        def autolabel(rects, std_values):
            for idx, rect in enumerate(rects):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + std_values[idx],
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Apply labels to both sets of bars
        autolabel(bars1, std_baseline)
        autolabel(bars2, std_no_feature)
        autolabel(bars3, std_no_spatial)
        
        ax.set_ylabel('Mean Attention Weight')
        ax.set_title('Channel Attention Weights Comparison (Right-semi Brain)')
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend(bbox_to_anchor=(1.02, 1.15), loc='upper right')
            # 增加右边和上边的边距以容纳图例
        plt.subplots_adjust(right=0.85, top=0.85)

        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attention' / 'channel_attention_comparison.png')
        plt.close()

    def _plot_feature_attention_comparison(self):
        """Plot feature attention weights comparison"""
        # Load feature attention weights
        baseline_weights = np.load(self.attention_dir / 'raw_data' / 'baseline_feature_attention_weights.npy')
        no_spatial_weights = np.load(self.attention_dir / 'raw_data' / 'no_spatial_attention_feature_attention_weights.npy')
        
        # print("Baseline weights shape:", baseline_weights.shape)
        # print("No spatial weights shape:", no_spatial_weights.shape)
        baseline_reshaped = np.mean(baseline_weights, axis=0).reshape(4, -1).mean(axis=1)
        no_spatial_reshaped = np.mean(no_spatial_weights, axis=0).reshape(4, -1).mean(axis=1)
    
        baseline_std = np.std(baseline_weights, axis=0).reshape(4, -1).mean(axis=1)
        no_spatial_std = np.std(no_spatial_weights, axis=0).reshape(4, -1).mean(axis=1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        features = ['Thickness', 'Volume', 'Surface Area', 'Gray-White Ratio']
        x = np.arange(len(features))
        width = 0.35

        # Plot bars with error bars
        bars1 = ax.bar(x - width/2, baseline_reshaped, width, 
                    yerr=baseline_std,
                    label='FusionAttenNet',
                    capsize=5,
                    error_kw=dict(capthick=1, elinewidth=1))
        
        bars2 = ax.bar(x + width/2, no_spatial_reshaped, width,
                    yerr=no_spatial_std,
                    label='No Spatial Attention',
                    capsize=5,
                    error_kw=dict(capthick=1, elinewidth=1))
        
        # Add value labels on top of each bar
        for i, v in enumerate(baseline_reshaped):
            ax.text(x[i] - width/2, v + baseline_std[i],
                    f'{v:.3f}',
                    ha='center', va='bottom')
        
        for i, v in enumerate(no_spatial_reshaped):
            ax.text(x[i] + width/2, v + no_spatial_std[i],
                    f'{v:.3f}',
                    ha='center', va='bottom')
        
        # Add value labels on top of each bar using the bar objects
        def autolabel(rects, std_values):
            for idx, rect in enumerate(rects):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + std_values[idx],
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        # Apply labels to both sets of bars
        autolabel(bars1, baseline_std)
        autolabel(bars2, no_spatial_std)

        ax.set_ylabel('Feature Attention Weight')
        ax.set_title('Feature Attention Weights Comparison (Right-semi Brain)')
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attention' / 'feature_attention_comparison.png')
        plt.close()

        # Calculate statistics
        # baseline_stats = {
        #     'mean': np.mean(baseline_reshaped, axis=0),
        #     'std': np.std(baseline_reshaped, axis=0)
        # }
        # no_spatial_stats = {
        #     'mean': np.mean(no_spatial_reshaped, axis=0),
        #     'std': np.std(no_spatial_reshaped, axis=0)
        # }
        
        # # Create plot
        # fig, ax = plt.subplots(figsize=(10, 6))
        # features = ['Thickness', 'Volume', 'Surface Area', 'Gray-White Ratio']
        # x = np.arange(len(features))
        # width = 0.35
        
        # ax.bar(x - width/2, baseline_stats['mean'], width, 
        #        yerr=baseline_stats['std'], label='FusionAttenNet',capsize=5,error_kw=dict(capthick=1, elinewidth=1))
        # ax.bar(x + width/2, no_spatial_stats['mean'], width,
        #        yerr=no_spatial_stats['std'], label='No Spatial Attention',capsize=5,error_kw=dict(capthick=1, elinewidth=1))
        
        # ax.set_ylabel('Feature Attention Weight')
        # ax.set_title('Feature Attention Weights Comparison')
        # ax.set_xticks(x)
        # ax.set_xticklabels(features)
        # ax.legend()
        
        # plt.tight_layout()
        # plt.savefig(self.plots_dir / 'attention' / 'feature_attention_comparison.png')
        # plt.close()

# %%
# import numpy as np
# baseline_stats = np.load('/home/jouyang1/ablation_study_rh_results_20250207_172815/ablation_study_20250207_172815/attention/analysis/baseline_attention_statistics.npy', 
#                                allow_pickle=True).item()
# no_feature_stats= np.load('/home/jouyang1/ablation_study_rh_results_20250207_172815/ablation_study_20250207_172815/attention/analysis/no_feature_attention_attention_statistics.npy',
#                                  allow_pickle=True).item()
# no_spatial_stats = np.load('/home/jouyang1/ablation_study_rh_results_20250207_172815/ablation_study_20250207_172815/attention/analysis/no_spatial_attention_attention_statistics.npy',
#                                  allow_pickle=True).item()
# %%

    def _plot_attention_statistics_comparison(self):
        """Plot attention mechanism statistics comparison"""
        # Load attention statistics
        baseline_stats = np.load(self.attention_dir / 'analysis' / 'baseline_attention_statistics.npy', 
                               allow_pickle=True).item()
        no_feature_stats= np.load(self.attention_dir / 'analysis' / 'no_feature_attention_attention_statistics.npy',
                                 allow_pickle=True).item()
        no_spatial_stats = np.load(self.attention_dir / 'analysis' / 'no_spatial_attention_attention_statistics.npy',
                                 allow_pickle=True).item()
        
        # Create multi-panel plot
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        features = ['Thickness', 'Volume', 'Surface Area', 'Gray-White Ratio']
        
        # 1. Spatial Attention Analysis
        ax = axes[0]
        # Reshape (2048, 16, 16) to (4, 512, 16, 16) and compute statistics
        baseline_spatial_mean = baseline_stats['spatial_attention']['mean'].reshape(4, 512, 16, 16)
        baseline_spatial_std = baseline_stats['spatial_attention']['std'].reshape(4, 512, 16, 16)
        no_feature_spatial_mean = no_feature_stats['spatial_attention']['mean'].reshape(4, 512, 16, 16)
        no_feature_spatial_std = no_feature_stats['spatial_attention']['std'].reshape(4, 512, 16, 16)
        
        # Compute per-feature statistics
        baseline_spatial_stats = {
            'mean': np.mean(baseline_spatial_mean, axis=(1, 2, 3)),  # Average over channels and spatial dims
            'std': np.mean(baseline_spatial_std, axis=(1, 2, 3))
        }
        no_feature_spatial_stats = {
            'mean': np.mean(no_feature_spatial_mean, axis=(1, 2, 3)),
            'std': np.mean(no_feature_spatial_std, axis=(1, 2, 3))
        }

        x = np.arange(len(features))
        width = 0.35
        # ax.bar(x - width/2, baseline_spatial_stats['mean'], width, 
        #     yerr=baseline_spatial_stats['std'], label='FusionAttenNet',capsize=5,error_kw=dict(capthick=1, elinewidth=1))
        # ax.bar(x + width/2, no_feature_spatial_stats['mean'], width,
        #     yerr=no_feature_spatial_stats['std'], label='No Feature',capsize=5,error_kw=dict(capthick=1, elinewidth=1))
        
        # 绘制柱状图并添加数值标注
        rects1 = ax.bar(x - width/2, baseline_spatial_stats['mean'], width, 
                        yerr=baseline_spatial_stats['std'],
                        label='FusionAttenNet',
                        capsize=5,
                        error_kw=dict(capthick=1, elinewidth=1))
        
        rects2 = ax.bar(x + width/2, no_feature_spatial_stats['mean'], width,
                        yerr=no_feature_spatial_stats['std'],
                        label='No Feature',
                        capsize=5,
                        error_kw=dict(capthick=1, elinewidth=1))
        
        # 添加标注
        def autolabel(rects, errors):
            for idx, rect in enumerate(rects):
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height + errors[idx],
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        autolabel(rects1, baseline_spatial_stats['std'])
        autolabel(rects2, no_feature_spatial_stats['std'])
        
        ax.set_title('Spatial Attention Statistics per Cortical Feature (Right-semi Brain)')
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()

    # 2. Channel Attention Analysis
        ax = axes[1]
        # Reshape (2048, 1, 1) to (4, 512) and compute statistics
        baseline_channel_mean = baseline_stats['channel_attention']['mean'].reshape(4, 512).squeeze()
        baseline_channel_std = baseline_stats['channel_attention']['std'].reshape(4, 512).squeeze()
        no_feature_channel_mean = no_feature_stats['channel_attention']['mean'].reshape(4, 512).squeeze()
        no_feature_channel_std = no_feature_stats['channel_attention']['std'].reshape(4, 512).squeeze()
        no_spatial_channel_mean = no_spatial_stats['channel_attention']['mean'].reshape(4, 512).squeeze()
        no_spatial_channel_std = no_spatial_stats['channel_attention']['std'].reshape(4, 512).squeeze()
        
        # Compute per-feature statistics
        width = 0.25
        # ax.bar(x - width, np.mean(baseline_channel_mean, axis=1), width,
        #     yerr=np.mean(baseline_channel_std, axis=1), label='FusionAttenNet',capsize=5,error_kw=dict(capthick=1, elinewidth=1))
        # ax.bar(x, np.mean(no_feature_channel_mean, axis=1), width,
        #     yerr=np.mean(no_feature_channel_std, axis=1), label='No Feature',capsize=5,error_kw=dict(capthick=1, elinewidth=1))
        # ax.bar(x + width, np.mean(no_spatial_channel_mean, axis=1), width,
        #     yerr=np.mean(no_spatial_channel_std, axis=1), label='No Spatial',capsize=5,error_kw=dict(capthick=1, elinewidth=1))
        
        rects1 = ax.bar(x - width, np.mean(baseline_channel_mean, axis=1), width,
                    yerr=np.mean(baseline_channel_std, axis=1),
                    label='FusionAttenNet',
                    capsize=5,
                    error_kw=dict(capthick=1, elinewidth=1))
    
        rects2 = ax.bar(x, np.mean(no_feature_channel_mean, axis=1), width,
                        yerr=np.mean(no_feature_channel_std, axis=1),
                        label='No Feature',
                        capsize=5,
                        error_kw=dict(capthick=1, elinewidth=1))
        
        rects3 = ax.bar(x + width, np.mean(no_spatial_channel_mean, axis=1), width,
                        yerr=np.mean(no_spatial_channel_std, axis=1),
                        label='No Spatial',
                        capsize=5,
                        error_kw=dict(capthick=1, elinewidth=1))
        
        # 添加标注
        autolabel(rects1, np.mean(baseline_channel_std, axis=1))
        autolabel(rects2, np.mean(no_feature_channel_std, axis=1))
        autolabel(rects3, np.mean(no_spatial_channel_std, axis=1))
        
        ax.set_title('Channel Attention Statistics per Cortical Feature (Right-semi Brain)')
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()


       # 3. Feature Attention Analysis
        ax = axes[2]
        # Reshape (128,) to (4, 32) and compute statistics
        baseline_feature_mean = baseline_stats['feature_attention']['mean'].reshape(4, 32)
        baseline_feature_std = baseline_stats['feature_attention']['std'].reshape(4, 32)
        no_spatial_feature_mean = no_spatial_stats['feature_attention']['mean'].reshape(4, 32)
        no_spatial_feature_std = no_spatial_stats['feature_attention']['std'].reshape(4, 32)
        
        # ax.bar(x - width/2, np.mean(baseline_feature_mean, axis=1), width,
        #     yerr=np.mean(baseline_feature_std, axis=1), label='FusionAttenNet',capsize=5,error_kw=dict(capthick=1, elinewidth=1))
        # ax.bar(x + width/2, np.mean(no_spatial_feature_mean, axis=1), width,
        #     yerr=np.mean(no_spatial_feature_std, axis=1), label='No Spatial',capsize=5,error_kw=dict(capthick=1, elinewidth=1))
        
        rects1 = ax.bar(x - width/2, np.mean(baseline_feature_mean, axis=1), width,
                    yerr=np.mean(baseline_feature_std, axis=1),
                    label='FusionAttenNet',
                    capsize=5,
                    error_kw=dict(capthick=1, elinewidth=1))
    
        rects2 = ax.bar(x + width/2, np.mean(no_spatial_feature_mean, axis=1), width,
                        yerr=np.mean(no_spatial_feature_std, axis=1),
                        label='No Spatial',
                        capsize=5,
                        error_kw=dict(capthick=1, elinewidth=1))
        
        # 添加标注
        autolabel(rects1, np.mean(baseline_feature_std, axis=1))
        autolabel(rects2, np.mean(no_spatial_feature_std, axis=1))
        
        ax.set_title('Feature Attention Statistics per Cortical Feature (Right-semi Brain)')
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attention' / 'attention_statistics_comparison.png')
        plt.close()

    
    def analyze_phenotype_feature_weights(self):
        """分析特征注意力对表型特征的权重分配"""
        # 加载相关数据
        baseline_weights = np.load(self.attention_dir / 'raw_data' / 'baseline_feature_attention_weights.npy')
        no_spatial_weights = np.load(self.attention_dir / 'raw_data' / 'no_spatial_attention_feature_attention_weights.npy')
        
        # 按表型特征分组重塑权重
        phenotype_features = ['Aggressive Behavior', 'Sex', 'Maternal Education']
        
        # 计算每个表型特征的平均权重和标准差
        baseline_weights_by_feature = baseline_weights.reshape(-1, len(phenotype_features))
        no_spatial_weights_by_feature = no_spatial_weights.reshape(-1, len(phenotype_features))

        baseline_means = np.mean(baseline_weights_by_feature, axis=0)
        baseline_stds = np.std(baseline_weights_by_feature, axis=0)
        
        no_spatial_means = np.mean(no_spatial_weights_by_feature, axis=0)
        no_spatial_stds = np.std(no_spatial_weights_by_feature, axis=0)
        colors = ['#2171b5', '#4292c6', '#6baed6']
        # 创建可视化
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(phenotype_features))
        width = 0.35
        
        # 绘制权重分布对比图
        rects1 = ax.bar(x - width/2, 
                        baseline_means,
                        width,
                        yerr=baseline_stds,
                        label='FusionAttenNet',
                        capsize=5, color='#2171b5')
        
        rects2 = ax.bar(x + width/2,
                        no_spatial_means,
                        width,
                        yerr=no_spatial_stds,
                        label='No Spatial Attention',
                        capsize=5, color='#4292c6')
       
        def autolabel(rects, errors):
                """在每个柱状图上添加标签"""
                for idx, rect in enumerate(rects):
                    height = rect.get_height()
                    offset = errors[idx] * 0.05  # 偏移量，避免遮挡
                    ax.text(rect.get_x() + rect.get_width()/2., height + errors[idx] + offset,
                            f'{height:.3f}',
                            ha='center', va='bottom', fontsize=10)

        
        autolabel(rects1,baseline_stds)
        autolabel(rects2, no_spatial_stds)

        ax.set_ylabel('Feature Attention Weight')
        ax.set_title('Phenotype Feature Attention Weights (Right-semi Brain)')
        ax.set_xticks(x)
        ax.set_xticklabels(phenotype_features)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attention' / 'phenotype_feature_weights.png')
        plt.close()

    def analyze_task_specific_weights(self):
        """分析不同预测任务的权重差异"""
        # 加载预测结果和权重
        baseline_preds = np.load(self.attention_dir / 'raw_data' / 'baseline_predictions.npy')
        baseline_weights = np.load(self.attention_dir / 'raw_data' / 'baseline_feature_attention_weights.npy')
        
        # 将样本按预测任务分类
        attention_task_mask = baseline_preds[:, 0] > np.median(baseline_preds[:, 0])
        age_task_mask = baseline_preds[:, 1] > np.median(baseline_preds[:, 1])
        
        # 按任务计算平均权重
        phenotype_features = ['Aggressive Behavior', 'Sex', 'Maternal Education']
        attention_weights = baseline_weights[attention_task_mask].reshape(-1, len(phenotype_features))
        age_weights = baseline_weights[age_task_mask].reshape(-1, len(phenotype_features))
        colors = ['#2171b5', '#4292c6', '#6baed6']
    
        # 可视化任务特定的权重分布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 注意力任务权重分布
        sns.boxplot(data=pd.DataFrame(attention_weights, columns=phenotype_features), ax=ax1, palette=colors, showfliers=False)
        ax1.set_title('Feature Weights in Attention Prediction (Right-semi Brain)')
        ax1.set_ylim(0, 1)  # 设置y轴范围
    # 在箱线图上添加中位数标签
        for i, feature in enumerate(phenotype_features):
            median = np.median(attention_weights[:, i])
            ax1.text(i, median + 0.02, f'{median:.2f}', ha='center', va='bottom', fontsize=10, color='black')

            # 年龄任务权重分布
        sns.boxplot(data=pd.DataFrame(age_weights, columns=phenotype_features), ax=ax2, palette=colors, showfliers=False)
        ax2.set_title('Feature Weights in Age Prediction (Right-semi Brain)')
        ax2.set_ylim(0, 1)  # 设置y轴范围

    # 在箱线图上添加中位数标签
        for i, feature in enumerate(phenotype_features):
            median = np.median(age_weights[:, i])
            ax2.text(i, median + 0.02, f'{median:.2f}', ha='center', va='bottom', fontsize=10, color='black')


        plt.tight_layout()
        plt.savefig(self.plots_dir / 'attention' / 'task_specific_weights.png')
        plt.close()

        # 打印分组统计信息
        print("\nAttention Task Weight Statistics (Right-semi Brain) with medians:")
        print(pd.DataFrame(attention_weights, columns=phenotype_features).describe())
        print("\nAge Task Weight Statistics (Right-semi Brain):")
        print(pd.DataFrame(age_weights, columns=phenotype_features).describe())



    # def analyze_temporal_stability(self):
        # """分析权重的时间稳定性"""
        # # 加载相关数据
        # baseline_weights = np.load(self.attention_dir / 'raw_data' / 'baseline_feature_attention_weights.npy')
        # targets = np.load(self.attention_dir / 'raw_data' / 'baseline_targets.npy')
        
        # print("Initial weights shape:", baseline_weights.shape)  # (1260, 128)
        
        # # 定义表型特征
        # phenotype_features = ['Aggressive Behavior', 'Sex', 'Maternal Education']
        # num_features = len(phenotype_features)
        
        # # 重塑权重数组：每个特征分配相同数量的权重
        # # 128 / 3 ≈ 42.67，所以我们取前 126 个权重 (42 * 3)
        # weights_per_feature = 42
        # total_weights = weights_per_feature * num_features
        
        # # 重塑权重，确保能够平均分配给每个特征
        # reshaped_weights = baseline_weights[:, :total_weights].reshape(baseline_weights.shape[0], num_features, -1)
        
        # # 对每个特征的权重取平均值
        # feature_weights = reshaped_weights.mean(axis=2)  # 现在形状是 (1260, 3)
        
        # # 按年龄对样本排序
        # age_sorted_indices = np.argsort(targets[:, 1])
        # sorted_weights = feature_weights[age_sorted_indices]
        # sorted_ages = targets[age_sorted_indices, 1]
        
        # # 创建年龄组 (5个等大小的组)
        # age_groups = pd.qcut(sorted_ages, q=5, labels=['G1: Youngest', 'G2', 'G3', 'G4', 'G5: Oldest'])
        
        # # 分析每个年龄组的权重分布
        # weights_by_group = []
        # group_labels = []
        
        # for group in age_groups.unique():
        #     group_mask = age_groups == group
        #     group_weights = sorted_weights[group_mask]
        #     weights_by_group.append(group_weights)
        #     group_labels.append(str(group))
        
        # # 可视化时间稳定性
        # fig, ax = plt.subplots(figsize=(15, 8))
        # positions = np.arange(num_features)
        
        # # 使用更好的颜色方案
        # colors = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1']
        
        # # 为每个年龄组创建箱线图
        # for i, (group_weights, color) in enumerate(zip(weights_by_group, colors)):
        #     offset = (i - 2) * 0.15
        #     bp = ax.boxplot(group_weights, 
        #                 positions=positions + offset,
        #                 widths=0.12,
        #                 patch_artist=True,
        #                 boxprops=dict(facecolor=color, alpha=0.6, color='black'),
        #                 medianprops=dict(color='black', linewidth=1.5),
        #                 flierprops=dict(marker='o', markerfacecolor=color, alpha=0.5),
        #                 whiskerprops=dict(color='black'),
        #                 capprops=dict(color='black'))
        
        # # 添加图例
        # legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.6) 
        #                 for color in colors]
        # ax.legend(legend_elements, group_labels, 
        #         title='Age Groups',
        #         loc='upper right',
        #         bbox_to_anchor=(1.15, 1))
        
        # # 美化图表
        # ax.set_xticks(positions)
        # ax.set_xticklabels(phenotype_features, rotation=15)
        # ax.set_title('Feature Weight Stability Across Age Groups (Right-semi Brain)', 
        #             pad=20, fontsize=12)
        # ax.set_ylabel('Feature Weight', fontsize=11)
        # ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # # 调整布局以确保图例可见
        # plt.tight_layout()
        # plt.savefig(self.plots_dir / 'attention' / 'temporal_stability.png',
        #             bbox_inches='tight', dpi=300)
        # plt.close()
        
        # # 打印一些统计信息
        # print("\nWeight statistics across age groups:")
        # for i, (feature, pos) in enumerate(zip(phenotype_features, positions)):
        #     print(f"\n{feature}:")
        #     for j, group in enumerate(group_labels):
        #         weights = weights_by_group[j][:, i]
        #         print(f"  {group}: mean={weights.mean():.3f}, std={weights.std():.3f}")
# %%
# import numpy as np
# baseline_stats = np.load('/home/jouyang1/ablation_study_rh_results_20250207_172815/ablation_study_20250207_172815/error_analysis/detailed_analysis/high_error_analysis.npy', 
#                                allow_pickle=True).item()
# %%

    def _plot_error_analysis(self):
        """Plot comprehensive error analysis with improved visualization"""
        statistics = np.load(self.error_dir / 'statistics' / 'statistics.npy', 
                            allow_pickle=True).item()
        high_error_data = np.load(self.error_dir / 'detailed_analysis' / 'high_error_analysis.npy',
                                allow_pickle=True).item()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Age Prediction Analysis (Top Row)
        stats_labels = ['MAE', 'MSE', 'RMSE', 'Std Error']
        age_stats = statistics['age']['basic_stats']
        stats_values = [age_stats['mae'], age_stats['mean_error'], 
                    age_stats['rmse'], age_stats['std_error']]
        
        # Left: Basic statistics with value annotations
        bars = axes[0, 0].bar(stats_labels, stats_values)
        axes[0, 0].set_title('Age Prediction - Error Statistics (Right-semi Brain)')
        axes[0, 0].set_xticklabels(stats_labels, rotation=45)
        
        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            # 如果是非常小的值，就把文字放在柱子上方
            if height < 0.01:
                y_pos = height + 0.0015  # 在柱子上方一点
                va = 'bottom'
            else:
                y_pos = height/2  # 在柱子中间
                va = 'center'
            
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., y_pos,
                            f'{height:.3f}',
                            ha='center', va=va)
        
        # Right: High error cases distribution
        sns.histplot(data=high_error_data['high_error_age']['errors'], 
                    ax=axes[0, 1], bins=30)
        axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        axes[0, 1].set_title('Age Prediction - High Error Distribution (Right-semi Brain)')
        
        # 2. Attention Prediction Analysis (Bottom Row)
        attention_stats = statistics['attention']['basic_stats']
        stats_values = [attention_stats['mae'], attention_stats['mean_error'],
                        attention_stats['mse'], attention_stats['std_error']]
        
        # Left: Basic statistics with value annotations
        bars = axes[1, 0].bar(stats_labels, stats_values)
        axes[1, 0].set_title('Attention Prediction - Error Statistics (Right-semi Brain)')
        axes[1, 0].set_xticklabels(stats_labels, rotation=45)
        
        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            # 如果是非常小的值，就把文字放在柱子上方
            if height < 0.01:
                y_pos = height + 0.01  # 在柱子上方一点
                va = 'bottom'
            else:
                y_pos = height/2  # 在柱子中间
                va = 'center'
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., y_pos,
                            f'{height:.3f}',
                            ha='center', va=va)
        
        # Right: High error cases distribution
        sns.histplot(data=high_error_data['high_error_attention']['errors'], 
                    ax=axes[1, 1], bins=30)
        axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
        axes[1, 1].set_title('Attention Prediction - High Error Distribution (Right-semi Brain)')
        
        # Add percentile annotations with improved colors
        for i, (ax, percs) in enumerate([
            (axes[0, 1], statistics['age']['percentiles']),
            (axes[1, 1], statistics['attention']['percentiles'])
        ]):
            colors = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']  # 更清晰的颜色
            labels = ['25th', '50th', '75th', '90th', '95th']
            for perc, color, label in zip(percs, colors, labels):
                ax.axvline(x=perc, color=color, linestyle=':', label=f'{label} ({perc:.2f})')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'error' / 'error_analysis.png', bbox_inches='tight')
        plt.close()

    def analyze_all(self):
        """Run all analyses and generate visualizations"""
        print("Starting comprehensive analysis...")
        
        # Load spatial attention maps
        baseline_spatial = np.load(self.attention_dir / 'raw_data' / 'baseline_spatial_attention_maps.npy')
        no_feature_spatial = np.load(self.attention_dir / 'raw_data' / 'no_feature_attention_spatial_attention_maps.npy')
        
        # Generate all visualizations
        # print("Plotting spatial attention comparison on each channel...")
        # self._plot_spatial_attention_comparison(baseline_spatial, no_feature_spatial)
        # print("Plotting spatial attention comparison...")
        # self._plot_spatial_attention_comparison_full(baseline_spatial, no_feature_spatial)
        
        # print("Plotting channel attention comparison...")
        # self._plot_channel_attention_comparison()
        
        # print("Plotting feature attention comparison...")
        # self._plot_feature_attention_comparison()
        
        # print("Plotting attention statistics comparison...")
        # self._plot_attention_statistics_comparison()
        
        # print("Plotting error analysis...")
        # self._plot_error_analysis()

            
        print("Analyzing phenotype feature weights...")
        self.analyze_phenotype_feature_weights()
        
        print("Analyzing task-specific weights...")
        self.analyze_task_specific_weights()
        
        print("Analyzing temporal stability...")
        # self.analyze_temporal_stability()
        
        print("Analysis complete. All visualizations have been saved to", self.plots_dir)

# Usage example
if __name__ == "__main__":
    base_dir = '/home/jouyang1/ablation_study_rh_results_20250207_172815/ablation_study_20250207_172815'
    analyzer = AblationAnalysis(base_dir)
    analyzer.analyze_all()
