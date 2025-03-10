"""
Run model analysis experiments including prediction pattern analysis,
projection feature analysis, and brain comparison analysis.

(Experiment Part Three)
"""
from datetime import datetime
import logging
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from experiment_model_interpretability_analysis_lh import PredictionPatternAnalysis, ProjectionFeatureAnalysis, BrainComparisonAnalysis
from train_precdiction_model_images_improved_lh import BrainADHDModel, BrainMapDataset
import os
import json

def load_data(image_path, phenotype_path, side='both', use_mmap=True, test_mode=False):
    """
    Load data for left/right brain or both
    Args:
        image_path: Path to brain image data
        phenotype_path: Path to phenotype data
        side: 'left', 'right', or 'both'
        use_mmap: Whether to use memory mapping
        test_mode: Whether to use a small subset for testing
    """
    try:
        # Load data
        image_data = np.load(image_path, mmap_mode='r' if use_mmap else None)
        phenotype_data = np.load(phenotype_path, mmap_mode='r' if use_mmap else None)
        
        if test_mode:
            # Use only 10% of data for testing
            n_samples = int(len(image_data) * 0.1)
            indices = np.random.choice(len(image_data), n_samples, replace=False)
            image_data = image_data[indices]
            phenotype_data = phenotype_data[indices]
            
        if side == 'left':
            return image_data[:, :4], phenotype_data  # First 4 channels for left brain
        elif side == 'right':
            return image_data[:, 4:], phenotype_data  # Last 4 channels for right brain
        else:
            return image_data, phenotype_data
            
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def setup_experiment_dir(test_mode=False):
    """Setup experiment directory and logging"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = '/home/jouyang1/test_experiment_results_lh_3' if test_mode else '/home/jouyang1/experiment_results_three'
    experiment_dir = os.path.join(base_dir, f'experiment_{timestamp}')
    
    try:
        # Create base experiment directory
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['data', 'visualizations', 'logs', 'metrics']:
            subdir_path = os.path.join(experiment_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            logging.info(f"Created subdirectory: {subdir_path}")
            
        # Setup logging
        logs_dir = Path(os.path.join(experiment_dir, 'logs'))
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logs_dir / "experiments.log"),
                logging.StreamHandler()
            ]
        )
        
        # Verify directory creation
        logging.info(f"Experiment directory exists: {os.path.exists(experiment_dir)}")
        logging.info(f"Experiment directory contents: {os.listdir(base_dir)}")
        
        return Path(experiment_dir)
        
    except Exception as e:
        logging.error(f"Error creating directories: {e}")
        raise

def main(test_mode=False):
    """Run all analyses"""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = setup_experiment_dir(test_mode)
    logging.info(f"Using device: {device}")
    
    # Load data
    logging.info("Loading data...")
    image_path = '/projects/0/einf1049/scratch/jouyang/all_cnn_lh_brainimages.npy'
    phenotype_path = '/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes.npy'
    
    # Load left and right brain data separately
    left_images, left_phenotypes = load_data(image_path, phenotype_path, 'left', test_mode=test_mode)
    right_images, right_phenotypes = load_data(image_path, phenotype_path, 'right', test_mode=test_mode)
    
    # Create datasets and loaders
    train_idx, test_idx = train_test_split(np.arange(len(left_images)), test_size=0.2)
    
    # Create datasets
    left_train_dataset = BrainMapDataset(left_images[train_idx], left_phenotypes[train_idx])
    left_test_dataset = BrainMapDataset(left_images[test_idx], left_phenotypes[test_idx])
    right_train_dataset = BrainMapDataset(right_images[train_idx], right_phenotypes[train_idx])
    right_test_dataset = BrainMapDataset(right_images[test_idx], right_phenotypes[test_idx])
    
    # Create data loaders
    batch_size = 4 if test_mode else 8
    left_train_loader = DataLoader(left_train_dataset, batch_size=batch_size, shuffle=True)
    left_test_loader = DataLoader(left_test_dataset, batch_size=batch_size)
    right_train_loader = DataLoader(right_train_dataset, batch_size=batch_size, shuffle=True)
    right_test_loader = DataLoader(right_test_dataset, batch_size=batch_size)
    
    try:
        # Load pre-trained models
        left_model = BrainADHDModel(num_phenotypes=3).to(device)
        right_model = BrainADHDModel(num_phenotypes=3).to(device)
        left_model.load_state_dict(torch.load('all_lh_best_model_improved.pth'))
        right_model.load_state_dict(torch.load('all_rh_best_model_improved.pth'))
        
        # 1. Prediction Pattern Analysis
        logging.info("Running Prediction Pattern Analysis...")
        pred_analysis = PredictionPatternAnalysis(left_model, device, experiment_dir)
        prediction_results = {
            'accuracy_distribution': pred_analysis.prediction_accuracy_distribution(left_test_loader),
            'confidence_analysis': pred_analysis.prediction_confidence_analysis(left_test_loader),
            'error_pattern': pred_analysis.error_pattern_analysis(left_test_loader)
        }
        
        # 2. Projection Feature Analysis
        logging.info("Running Projection Feature Analysis...")
        proj_analysis = ProjectionFeatureAnalysis(experiment_dir)
        feature_names = [f'feature_{i}' for i in range(left_images.shape[1])]
        projection_results = {
            'feature_distribution': proj_analysis.feature_distribution_analysis(left_images, feature_names),
            'quality_assessment': proj_analysis.quality_assessment(left_images, feature_names),
            'reliability_analysis': proj_analysis.reliability_analysis(left_images, feature_names)
        }
        
        # 3. Brain Comparison Analysis
        logging.info("Running Brain Comparison Analysis...")
        brain_comp = BrainComparisonAnalysis(experiment_dir)
        
        # Get predictions from both models
        left_preds, left_targets = pred_analysis._get_predictions(left_test_loader)
        right_preds, right_targets = pred_analysis._get_predictions(right_test_loader)
        
        comparison_results = {
            'performance': brain_comp.performance_comparison(
                left_preds, right_preds, left_targets, right_targets
            ),
            'feature_importance': brain_comp.feature_importance_analysis(
                left_model, right_model, left_test_loader  # Use left test loader for feature importance
            ),
            'clinical_implications': brain_comp.clinical_implications(
                left_preds, right_preds
            )
        }
        
        # Save all results
        with open(experiment_dir / 'metrics' / 'all_results.json', 'w') as f:
            json.dump({
                'prediction_analysis': prediction_results,
                'projection_analysis': projection_results,
                'brain_comparison': comparison_results,
                'experiment_info': {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'test_mode': test_mode,
                    'device': str(device)
                }
            }, f, indent=4)
        
        logging.info("All analyses completed successfully")
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main(test_mode=False)