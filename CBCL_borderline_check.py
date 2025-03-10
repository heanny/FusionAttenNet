"""
简化版的Borderline分析，直接计算整体数据的百分位值
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

class BorderlineAnalysis:
    def __init__(self, data_path, results_dir):
        """
        初始化BorderlineAnalysis类
        Args:
            data_path: .npy文件路径
            results_dir: 结果保存目录
        """
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / 'metrics'
        self.vis_dir = self.results_dir / 'visualizations'
        
        # 创建目录
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        self.vis_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载数据
        self.data = np.load(data_path)
        self.df = pd.DataFrame(self.data, columns=['sum_att', 'sum_agg', 'age', 'sex', 'edu_maternal'])
        logging.info(f"Loaded data shape: {self.data.shape}")

    def calculate_borderline_ranges(self):
        """
        计算80-93和93-97百分位的borderline值
        """
        results = {}
        percentile_ranges = [(80, 93), (93, 97)]
        behaviors = ['sum_att', 'sum_agg']
        behavior_names = {'sum_att': 'Attention Problems', 'sum_agg': 'Aggressive Behavior'}
        
        for behavior in behaviors:
            scores = self.df[behavior].values
            results[behavior] = {}
            
            # 计算不同百分位范围
            for lower, upper in percentile_ranges:
                range_key = f"{lower}-{upper}"
                
                # 计算百分位数值
                lower_bound = np.percentile(scores, lower)
                upper_bound = np.percentile(scores, upper)
                
                # 计算在此范围内的样本数
                mask = (scores >= lower_bound) & (scores <= upper_bound)
                range_samples = scores[mask]
                n_samples = sum(mask)
                
                # 计算统计量
                results[behavior][range_key] = {
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'n_samples': int(n_samples),
                    'percentage': float(n_samples / len(scores) * 100),
                    'mean': float(np.mean(range_samples)),
                    'std': float(np.std(range_samples))
                }
                
                # 额外信息
                results[behavior][range_key].update({
                    'total_samples': len(scores),
                    'below_range': int(sum(scores < lower_bound)),
                    'above_range': int(sum(scores > upper_bound))
                })
            
            # 绘制分布图
            self._plot_distribution(
                scores=scores,
                ranges=results[behavior],
                behavior_name=behavior_names[behavior]
            )
        
        # 保存结果
        with open(self.metrics_dir / 'borderline_ranges.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results

    def _plot_distribution(self, scores, ranges, behavior_name):
        """
        绘制分数分布和borderline范围
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制核密度估计
        sns.kdeplot(data=scores, fill=True)
        
        # 添加borderline范围
        max_height = plt.gca().get_ylim()[1]
        colors = ['#2ecc71', '#3498db']  # 绿色和蓝色
        
        for (range_name, range_info), color in zip(ranges.items(), colors):
            plt.axvline(range_info['lower_bound'], color=color, linestyle='--',
                       label=f'{range_name}%')
            plt.axvline(range_info['upper_bound'], color=color, linestyle='--')
            
            # 区间标注
            plt.fill_between(
                [range_info['lower_bound'], range_info['upper_bound']],
                [0, 0], [max_height, max_height],
                alpha=0.2, color=color
            )
        
        plt.title(f'Distribution of {behavior_name} Scores')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
        
        # 添加统计信息
        text = []
        for range_name, info in ranges.items():
            text.extend([
                f"{range_name}% range:",
                f"Range: {info['lower_bound']:.2f} - {info['upper_bound']:.2f}",
                f"N = {info['n_samples']} ({info['percentage']:.1f}%)",
                f"Mean ± SD: {info['mean']:.2f} ± {info['std']:.2f}",
                ""
            ])
        
        plt.text(0.02, 0.98, "\n".join(text),
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(self.vis_dir / f'borderline_distribution_{behavior_name.lower().replace(" ", "_")}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 设置路径
    data_path = '/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes.npy'
    results_dir = Path('/home/jouyang1/CBCL_borderline_checkup')
    
    # 创建分析器并运行
    analyzer = BorderlineAnalysis(data_path, results_dir)
    results = analyzer.calculate_borderline_ranges()
    
    # 打印结果摘要
    for behavior in results:
        print(f"\n{behavior} Results:")
        for range_name, range_info in results[behavior].items():
            print(f"\n{range_name} percentile range:")
            print(f"Range: {range_info['lower_bound']:.2f} - {range_info['upper_bound']:.2f}")
            print(f"Samples in range: {range_info['n_samples']} ({range_info['percentage']:.1f}%)")
            print(f"Mean ± SD: {range_info['mean']:.2f} ± {range_info['std']:.2f}")

if __name__ == "__main__":
    main()