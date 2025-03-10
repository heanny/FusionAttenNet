"""
Script for Feature Contribution Analysis (Experiment Part Two)
"""

import os
import time
import torch
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
# from train_precdiction_model_images_improved_lh import BrainADHDModel, BrainMapDataset
from expt_prediction_model_images_improved_lh import BrainMapDataset, DynamicBrainADHDModel
from experiment_feature_contribution_analysis_lh import ExperimentManager, MemoryManager, DataManager, ModelTrainer, FeatureAnalysis

def main(test_mode=False):
    """
    特征分析实验的主函数
    
    Args:
        test_mode (bool): 是否在测试模式下运行
    """
    try:
        # 1. 初始化实验环境
        start_time = time.time()
        experiment_manager = ExperimentManager(test_mode=test_mode)
        logging.info(f"Starting Experiment Part Two: feature contribution analysis - ID: {experiment_manager.timestamp}")
        
        # 2. 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device: {device}")
        
        # 3. 加载数据
        data_manager = DataManager(
            image_path='/projects/0/einf1049/scratch/jouyang/all_cnn_lh_brainimages.npy',
            phenotype_path='/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes.npy',
            test_mode=test_mode
        )
        
        logging.info("Loading data...")
        image_data, phenotype_data = data_manager.load_data()

        data_manager.verify_data_structure()
        
        # 4. 创建数据加载器（使用6/2/2的划分）
        logging.info("Creating data loader...")
        train_loader, val_loader, test_loader, datasets = data_manager.create_data_splits(
            image_data, 
            phenotype_data
        )
        
        # 5. 初始化模型和训练器
        logging.info("Initialising model...")
        model = DynamicBrainADHDModel(
            brain_feature_config='B1',  # 初始时使用完整特征
            phenotype_feature_config='D1'  # 初始时使用完整特征
        ).to(device)
        model_trainer = ModelTrainer(
            model=model,
            device=device,
            test_mode=test_mode
        )
        
        # 6. 创建特征分析器
        feature_analyzer = FeatureAnalysis(
            experiment_manager=experiment_manager,
            model_trainer=model_trainer,
            device=device
        )
        
        # 7. 加载实验进度
        progress = experiment_manager.load_progress()
        
        # 8. 运行分析
        try:
            # 8.1 Brain Feature Analysis
            if not progress.get('brain_features_completed', False):
                logging.info("Starting Experiment 1. Brain Feature Importance Analysis...")
                logging.info("Analyzing brain feature combinations: B1, B2, B3, B4...")
                brain_results = feature_analyzer.analyze_brain_features(
                    train_loader,
                    val_loader
                )
                experiment_manager.save_progress('brain_features', brain_results)
                MemoryManager.cleanup()
            
            # 8.2 Phenotype Feature Analysis
            if not progress.get('phenotype_features_completed', False):
                logging.info("Starting Experiment 2. Phenotypic Feature Importance Analysis...")
                phenotype_results = feature_analyzer.analyze_phenotype_features(
                    train_loader,
                    val_loader
                )
                experiment_manager.save_progress('phenotype_features', phenotype_results)
                MemoryManager.cleanup()
            
            # 8.3 Feature Interaction Analysis
            if not progress.get('feature_interactions_completed', False):
                logging.info("Starting Experiment 3. Feature Interaction Analysis...")
                interaction_results = feature_analyzer.analyze_feature_interactions(
                    train_loader,
                    val_loader
                )
                experiment_manager.save_progress('feature_interactions', interaction_results)
                MemoryManager.cleanup()
            
            # # 9. 在测试集上进行最终评估
            # if not progress.get('final_evaluation_completed', False):
            #     logging.info("Final test sets evaluation...")
            #     final_results = feature_analyzer.evaluate_on_test_set(test_loader)
            #     experiment_manager.save_progress('final_evaluation', final_results)

            # 9. 在测试集上进行最终评估
            if not progress.get('final_evaluation_completed', False):
                try:
                    logging.info("Final test sets evaluation...")
                    final_results = feature_analyzer.evaluate_on_test_set(test_loader)
                    experiment_manager.save_progress('final_evaluation', final_results)
                    logging.info("Final evaluation completed successfully")
                except Exception as e:
                    logging.error(f"Error during final evaluation: {str(e)}")
                    # 可以选择继续执行或者raise
            
            # 10. 创建实验报告
            logging.info("Generating the experiment report...")
            create_final_report(experiment_manager.results_dir, {
                'test_mode': test_mode,
                'total_time': time.time() - start_time,
                'device': str(device),
                'data_split': {
                    'train': len(datasets[0]),
                    'val': len(datasets[1]),
                    'test': len(datasets[2])
                }
            })
            
            logging.info(f"Experiment finished，total time use: {(time.time() - start_time)/3600:.2f} hours")
            
        except Exception as e:
            logging.error(f"Errors in Experiments: {str(e)}")
            logging.info("Have saved the finished experiments, can be continued from the breaking point")
            raise
            
    except Exception as e:
        logging.error(f"Fail to initialising the model: {str(e)}")
        raise

def create_final_report(results_dir, experiment_info):
    """
    创建最终的实验报告
    
    Args:
        results_dir (Path): 结果目录
        experiment_info (dict): 实验信息
    """
    report_path = results_dir / 'final_report.md'
    
    with open(report_path, 'w') as f:
        # 1. 实验概述
        f.write("# Feature contribution report\n\n")
        f.write(f"Experiment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {experiment_info['total_time']/3600:.2f} hours\n")
        f.write(f"Device: {experiment_info['device']}\n\n")
        
        # 2. 数据集信息
        f.write("## Information of datasets\n\n")
        f.write("### Data dividing\n")
        f.write(f"- training set: {experiment_info['data_split']['train']} samples\n")
        f.write(f"- validation set: {experiment_info['data_split']['val']} samples\n")
        f.write(f"- testing set: {experiment_info['data_split']['test']} samples\n\n")
        
        # 3. 分析结果概述
        f.write("## Results summary\n\n")
        
        # 3.1 脑部特征分析
        f.write("### 1. Brain Feature Importance Analysis\n")
        f.write("Please check `brain_features/` directory\n\n")
        
        # 3.2 表型特征分析
        f.write("### 2. Phenotypic Feature Importance Analysis\n")
        f.write("Please check `phenotype_features/` directory\n\n")
        
        # 3.3 特征交互分析
        f.write("### 3. Feature Interaction Analysis\n")
        f.write("Please check `interactions/` directory\n\n")
        
        # 4. 可视化结果
        f.write("## Visualization results\n")
        f.write("All visualizations are saved in the seperate `plots/` sub-directory\n\n")
        
        # 5. 结论和建议
        # f.write("## Conclusion and suggestions\n")
        # f.write("Please check the \n")

if __name__ == "__main__":
    # 添加命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description='Run the Feature Contribution Analysis')
    parser.add_argument('--test', action='store_true', help='Running with test mode')
    args = parser.parse_args()
    
    # 运行主程序
    main(test_mode=args.test)