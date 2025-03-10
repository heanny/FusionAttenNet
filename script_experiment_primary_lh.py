"""
Run the primary experiment for model evaluation
"""
from datetime import datetime
import logging
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from experiment_primary_updated_lh import ModelExperiments
from train_precdiction_model_images_improved_lh import BrainADHDModel, BrainMapDataset
import os
import torch
import time
import numpy as np
import json
from torch.utils.data import DataLoader

def load_data(image_path, phenotype_path, use_mmap=True, test_mode=False, test_ratio=0.1):
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
        
        if test_mode:
            # 只取部分数据用于测试
            total_samples = len(image_data)
            test_samples = int(total_samples * test_ratio)
            
            # 随机选择索引
            indices = np.random.choice(total_samples, test_samples, replace=False)
            indices.sort()  # 排序以保持数据连续性
            
            # 选择对应的数据
            image_data = image_data[indices]
            phenotype_data = phenotype_data[indices]
            
            logging.info(f"Test mode: Using {test_ratio*100}% of data")
            logging.info(f"Selected {test_samples} samples from {total_samples} total samples")
        
        logging.info(f"Loaded data shape: Image {image_data.shape}, Phenotype {phenotype_data.shape}")
        return image_data, phenotype_data
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def create_summary_report(results, results_dir):
    """Create a detailed summary report of all experiments"""
    report_path = results_dir / 'experiment_summary.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Experiment Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # 实验参数
        f.write("Experiment Parameters:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Mode: {results['test_mode']}\n")
        f.write(f"Total Time: {results['total_time']/3600:.2f} hours\n\n")
        
        # Sensitivity Analysis - 分别显示sum_att和sum_agg的结果
        f.write("1. Sensitivity Analysis\n")
        f.write("-" * 30 + "\n")
        for score_name in ['sum_att', 'sum_agg']:
            f.write(f"\n{score_name.upper()}:\n")
            for noise_type in ['gaussian_noise', 'random_masking']:
                f.write(f"\n{noise_type.upper()}:\n")
                for measurement in results['sensitivity'][score_name][noise_type]:
                    f.write(f"Noise level: {measurement['noise_level']}\n")
                    f.write(f"Mean change: {measurement['mean_change']:.4f}\n")
                    f.write(f"Standard deviation: {measurement['std_dev']:.4f}\n")
                    f.write(f"ICC: {measurement['icc']:.4f}\n")
                    f.write(f"ICC p-value: {measurement['icc_p_value']:.4f}\n")
        
        # Sample Size Impact
        f.write("\n2. Sample Size Impact\n")
        f.write("-" * 30 + "\n")
        for score_name in ['sum_att', 'sum_agg']:
            f.write(f"\n{score_name.upper()}:\n")
            for result in results['sample_size'][score_name]:
                f.write(f"\nSample size: {result['sample_size']*100}%\n")
                f.write(f"MSE: {result['mse_mean']:.4f} (±{result['mse_std']:.4f})\n")
                f.write(f"R2: {result['r2_mean']:.4f} (±{result['r2_std']:.4f})\n")
                f.write(f"Correlation: {result['correlation_mean']:.4f} (±{result['correlation_std']:.4f})\n")
        
        # Stability Analysis
        f.write("\n3. Stability Analysis\n")
        f.write("-" * 30 + "\n")
        for score_name in ['sum_att', 'sum_agg']:
            f.write(f"\n{score_name.upper()}:\n")
            metrics = results['stability'][1][score_name]  # Using the final metrics
            f.write(f"MSE: {metrics['mse_mean']:.4f} (±{metrics['mse_std']:.4f})\n")
            f.write(f"R2: {metrics['r2_mean']:.4f} (±{metrics['r2_std']:.4f})\n")
            f.write(f"Correlation: {metrics['correlation_mean']:.4f} (±{metrics['correlation_std']:.4f})\n")
        
        # Computational Assessment
        f.write("\n4. Computational Assessment\n")
        f.write("-" * 30 + "\n")
        comp = results['computational']
        f.write(f"Training time per epoch: {comp['training_metrics']['avg_time_per_epoch']:.2f}s\n")
        f.write(f"Average inference time: {comp['inference_metrics']['avg_inference_time']:.4f}s\n")
        f.write(f"GPU memory usage: {comp['memory_metrics']['gpu_memory']['allocated']:.2f} GB\n")
        f.write(f"Model parameters: {comp['model_info']['total_parameters']:,}\n")

def main(test_mode=False):

    print(f"Test mode:",test_mode)

    # 设置较小的batch size用于测试
    batch_size = 4 if test_mode else 8
    num_workers = 2

    # 在函数开始处添加时间戳和目录创建
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = '/home/jouyang1/test_experiment_results' if test_mode else '/home/jouyang1/experiment_results_primary'
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')
    
    # 创建目录结构
    # for subdir in ['data', 'visualizations', 'logs', 'metrics']: 
    #     os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)

    try:
    # 尝试创建所有必要的目录
        os.makedirs(experiment_dir, exist_ok=True)
        for subdir in ['data', 'visualizations', 'logs', 'metrics']:
            subdir_path = os.path.join(experiment_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            logging.info(f"Created subdirectory: {subdir_path}")
    except Exception as e:
        logging.error(f"Error creating directories: {e}")
        raise

    logs_dir = Path(os.path.join(experiment_dir, 'logs'))

    # verify the folders
    logging.info(f"Experiment directory exists: {os.path.exists(experiment_dir)}")
    logging.info(f"Experiment directory contents: {os.listdir(base_dir)}")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "experiments.log"), #has to be path objective to use / 
            logging.StreamHandler()
        ]
    )
    # 打印实验配置
    logging.info("Experiment Configuration:")
    logging.info(f"Test Mode: {test_mode}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"Number of Workers: {num_workers}")
    logging.info(f"Base Epochs: {5 if test_mode else 30}")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # 创建结果目录
    # results_dir = Path('/home/jouyang1/test_experiment_results' if test_mode else '/home/jouyang1/experiment_results_primary')
    results_dir = Path(experiment_dir)
    results_dir.mkdir(exist_ok=True)


    
    # 加载数据
    image_path = '/projects/0/einf1049/scratch/jouyang/all_cnn_lh_brainimages.npy'
    phenotype_path = '/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes.npy'
    
    logging.info("Loading data...")
    image_data, loaded_phenotype_tensor = load_data(
        image_path, 
        phenotype_path, 
        use_mmap=True,
        test_mode=test_mode,
        test_ratio=0.1 if test_mode else 1.0
    )
    
    # 创建数据分割
    indices = np.arange(len(image_data))
    # 实验评估用的数据划分（60/20/20）
    # 注：这与模型训练时的划分(80/20)不同，目的是进行更严格的性能评估
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42)

    # 创建数据集
    train_dataset = BrainMapDataset(image_data[train_idx], loaded_phenotype_tensor[train_idx])
    val_dataset = BrainMapDataset(image_data[val_idx], loaded_phenotype_tensor[val_idx])
    test_dataset = BrainMapDataset(image_data[test_idx], loaded_phenotype_tensor[test_idx])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           num_workers=num_workers, pin_memory=True)
    
    # 初始化实验
    experiments = ModelExperiments(
        BrainADHDModel, 
        BrainMapDataset, 
        device,
        base_epochs = 5 if test_mode else 30,  
        test_mode = test_mode,
        experiment_dir = results_dir
    )
    # 记录实验开始时间
    start_time = time.time()

    # 加载进度
    progress = experiments._load_progress()
    
    try: 
        if not progress.get('sensitivity_test_completed', False):
            # 1.1 Sensitivity Tests
            logging.info("1.1 Starting sensitivity tests...")
            model = BrainADHDModel(num_phenotypes=3).to(device)
            model.load_state_dict(torch.load('all_lh_best_model_improved.pth', weights_only=True)) 
            
            sensitivity_info = experiments.sensitivity_test(
                'all_lh_best_model_improved.pth',
                test_loader,
                num_trials=2 if test_mode else 5  # 测试模式使用更少的trials
            )
            progress['sensitivity_test_completed'] = True
            progress['sensitivity_info'] = sensitivity_info
            experiments._save_progress(progress)

            # 加载完整结果用于报告
            with open(sensitivity_info['metrics_file'], 'r') as f:
                sensitivity_results = json.load(f)['results']

        else:
            # 需要时加载完整结果
            with open(progress['sensitivity_info']['metrics_file'], 'r') as f:
                sensitivity_results = json.load(f)['results']
            logging.info("Skipping 1.1 sensitivity tests (already completed)")

        if not progress.get('sample_size_impact_completed', False):
            # 1.2 Sample Size Impact
            logging.info("1.2 Starting sample size impact analysis...")
            # sample_sizes = [0.25, 0.5] if test_mode else [0.25, 0.5, 0.75, 1.0]
            # sample_size_results = experiments.sample_size_impact(train_dataset, val_dataset, sample_sizes)
            # n_total = len(train_dataset)
            # 使用对数间隔的样本量
            if test_mode:
                sample_sizes = [0.2, 0.4]  # 测试模式用较少的样本量
            else:
                # 使用对数间隔
                base_size = 0.2  # 起始比例
                sample_sizes = [min(base_size * (2**i), 1.0) for i in range(4)]  # 生成[0.2, 0.4, 0.8, 1.0]
            logging.info(f"Using sample sizes: {[f'{size*100}%' for size in sample_sizes]}")
            sample_size_info = experiments.sample_size_impact(train_dataset, val_dataset, sample_sizes)  # 加上这行
            progress['sample_size_impact_completed'] = True
            progress['sample_size_info'] = sample_size_info
            experiments._save_progress(progress)

            with open(sample_size_info['metrics_file'], 'r') as f:
                sample_size_results = json.load(f)['results']

        else:
            with open(progress['sample_size_info']['metrics_file'], 'r') as f:
                sample_size_results = json.load(f)['results']
            logging.info("Skipping 1.2 sample size impact analysis (already completed)")
            
        
        if not progress.get('stability_evaluation_completed', False):            
            # 2. Stability Evaluation
            logging.info("2 Starting stability evaluation...")
            stability_info = experiments.stability_evaluation(
                train_dataset,
                n_splits=2 if test_mode else 5  # 测试模式使用更少的fold
            )
            progress['stability_evaluation_completed'] = True
            progress['stability_info'] = stability_info
            experiments._save_progress(progress)

            with open(stability_info['metrics_file'], 'r') as f:
                stability_data = json.load(f)
                stability_results = stability_data['results'], stability_data['final_metrics']

        else: 
            with open(progress['stability_info']['metrics_file'], 'r') as f:
                stability_data = json.load(f)
                stability_results = stability_data['results'], stability_data['final_metrics']
            logging.info("Skipping stability evaluation (already completed)")
            

        if not progress.get('computational_assessment_completed', False):
            # 3. Computational Assessment
            logging.info("3 Starting computational assessment...")
            comp_info = experiments.computational_assessment(
                'all_lh_best_model_improved.pth',
                train_loader, 
                test_loader
            )
            progress['computational_assessment_completed'] = True
            progress['computational_info'] = comp_info
            experiments._save_progress(progress)

            with open(comp_info['metrics_file'], 'r') as f:
                comp_results = json.load(f)

        else:
            with open(progress['computational_info']['metrics_file'], 'r') as f:
                comp_results = json.load(f)
            logging.info("Skipping computational assessment (already completed)")

    
        # 记录总运行时间
        total_time = time.time() - start_time
        logging.info(f"Total experiment time: {total_time/3600:.2f} hours")
        
        # 创建总结报告
        create_summary_report({
            'test_mode': test_mode,
            'total_time': total_time,
            'sensitivity': sensitivity_results,
            'sample_size': sample_size_results,
            'stability': stability_results,
            'computational': comp_results
        }, results_dir)
        
    except Exception as e:
        logging.error(f"Experiment interrupted: {str(e)}")
        logging.info(f"Progress saved. You can resume from the last completed experiment.")
        raise

if __name__ == "__main__":
    # 设置test_mode=True来运行测试版本
    main(test_mode=False)


