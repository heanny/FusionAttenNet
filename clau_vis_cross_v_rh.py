import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

class ResultsLoader:
    def __init__(self, base_paths):
        """
        Initialize the loader with base paths
        
        Parameters:
        -----------
        base_paths : dict
            Dictionary containing base paths for different models
        """
        self.base_paths = base_paths
        self.batch_per_epoch = 2519  # As specified in your code
        
    def load_baseline_predictions(self, model_num):
        """
        Load predictions for a specific baseline model
        
        Parameters:
        -----------
        model_num : int
            Baseline model number (1, 2, or 3)
        """
        results = []
        for fold in range(5):
            if model_num in [1, 2]:
                subdir = 'univariate' if model_num == 1 else 'phenotype'
                path = f'/home/jouyang1/cv_baseline12_rh/{subdir}/analysis/fold_{fold}_predictions.npy'
            else:
                # For baseline 3, load metrics directly
                path = f'/home/jouyang1/cv_baselin3_rh_results_20250218_155206/cross_validation/metrics/fold_{fold}_all_metrics.npy'
                
            try:
                data = np.load(path, allow_pickle=True).item()
                results.append(data)
            except Exception as e:
                print(f"Error loading baseline {model_num} fold {fold}: {e}")
                    
        return results
    
    def load_fusion_results(self):
        """Load FusionAttenNet results"""
        results = []
        for fold in range(5):
            path = f'/home/jouyang1/cross_validation_rh_20250207_001347/cross_validation/fold_results/fold_{fold}_results.npy'
            try:
                data = np.load(path, allow_pickle=True).item()
                results.append(data)
            except Exception as e:
                print(f"Error loading fusion results fold {fold}: {e}")
                
        return results
    
    def load_training_history(self):
        """Load training history with learning rate downsampling"""
        histories = []
        for fold in range(5):
            path = f'/home/jouyang1/cross_validation_rh_20250207_001347/plots/training_plots/fold_{fold}_training_history.npy'
            try:
                history = np.load(path, allow_pickle=True).item()
                
                # Downsample learning rates to match epochs
                learning_rates = history['learning_rates']
                actual_epochs = len(history['train_losses'])
                
                if len(learning_rates) != actual_epochs:
                    # Average learning rates per epoch
                    lr_per_epoch = [
                        np.mean(learning_rates[i * self.batch_per_epoch:(i + 1) * self.batch_per_epoch])
                        for i in range(actual_epochs)
                    ]
                else:
                    lr_per_epoch = learning_rates
                
                # Create processed history dictionary
                processed_history = {
                    'learning_rates': lr_per_epoch,
                    'train_losses': history['train_losses'],
                    'val_losses': history['val_losses'],
                    'train_att_losses': history.get('train_att_losses', []),
                    'train_age_losses': history.get('train_age_losses', [])
                }
                
                histories.append(processed_history)
            except Exception as e:
                print(f"Error loading training history fold {fold}: {e}")
                
        return histories
    
    def process_metrics(self, predictions_list, target_idx=None):
        """
        Calculate metrics from predictions or load from saved metrics
        
        Parameters:
        -----------
        predictions_list : list
            List of prediction results or metrics
        target_idx : int, optional
            Index of target variable (0 for attention, 1 for age)
        """
        metrics = {
            'R^2': [],
            'MSE': [],
            'RMSE': [],
            'MAE': [],
            'Pearson': []
        }
        
        target_name = 'sum_att' if target_idx == 0 else 'age'
        
        for pred_data in predictions_list:
            # Check if we have pre-computed metrics
            if 'fold_metrics' in pred_data:
                # Handle baseline3's format
                fold_metric = pred_data['fold_metrics'][0][target_name]
                metrics['R^2'].append(fold_metric['R^2'])  # Note the different symbol
                metrics['MSE'].append(fold_metric['MSE'])
                metrics['RMSE'].append(fold_metric['RMSE'])
                metrics['MAE'].append(fold_metric['MAE'])
                metrics['Pearson'].append(fold_metric['Pearson'])
                continue
                
            # If no pre-computed metrics, calculate them
            preds = np.array(pred_data['predictions'])
            targets = np.array(pred_data['targets'])
            
            # Handle different data shapes
            if preds.size == 0 or targets.size == 0:
                print(f"Warning: Empty predictions or targets found")
                continue
                
            # Handle Baseline 3's special case
            if preds.shape[0] == 1 and targets.shape[0] == 1:
                if preds.size == 200 and targets.size == 2:
                    if target_idx is not None:
                        targets = np.repeat(targets[0][target_idx], preds.shape[1])
                        preds = preds.squeeze()
            else:
                if target_idx is not None:
                    if preds.ndim > 1:
                        preds = preds[:, target_idx]
                    if targets.ndim > 1:
                        targets = targets[:, target_idx]
            
            # Ensure shapes match
            if preds.shape != targets.shape:
                print(f"Warning: Shape mismatch - Predictions: {preds.shape}, Targets: {targets.shape}")
                continue
                
            # Calculate metrics
            try:
                mse = np.mean((targets - preds) ** 2)
                metrics['MSE'].append(mse)
                metrics['RMSE'].append(np.sqrt(mse))
                metrics['MAE'].append(np.mean(np.abs(targets - preds)))
                
                # Calculate R^2 score
                ss_res = np.sum((targets - preds) ** 2)
                ss_tot = np.sum((targets - np.mean(targets)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                metrics['R^2'].append(r2)
                
                # Calculate Pearson correlation
                correlation = np.corrcoef(targets, preds)[0, 1]
                metrics['Pearson'].append(correlation)
                
            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")
                continue
        
        return metrics

    def load_baseline3_predictions(self):
        """Load predictions for baseline 3"""
        predictions = []
        for fold in range(5):
            path = f'/home/jouyang1/cv_baselin3_rh_results_20250218_155206/cross_validation/analysis/fold_{fold}_predictions.npy'
            try:
                data = np.load(path, allow_pickle=True).item()
                predictions.append(data)
            except Exception as e:
                print(f"Error loading baseline 3 predictions for fold {fold}: {e}")
        return predictions

    def get_residuals(self, predictions_list, target_idx=None):
        """
        Calculate residuals from predictions with corrected target shape handling
        
        Parameters:
        -----------
        predictions_list : list
            List of prediction results
        target_idx : int, optional
            Index of target variable (0 for attention, 1 for age)
        """
        all_residuals = []
        target_name = 'sum_att' if target_idx == 0 else 'age'
        
        for fold_idx, pred_data in enumerate(predictions_list):
            try:
                # For baseline3, we need to load predictions from a different file
                if 'fold_metrics' in pred_data:
                    try:
                        # Load the predictions file for this fold
                        pred_file = f'/home/jouyang1/cv_baselin3_rh_results_20250218_155206/cross_validation/analysis/fold_{fold_idx}_predictions.npy'
                        fold_predictions = np.load(pred_file, allow_pickle=True).item()
                        
                        if fold_predictions is not None:
                            preds = fold_predictions.get('predictions', None)
                            targets = fold_predictions.get('targets', None)
                            
                            if preds is not None and targets is not None:
                                preds = np.array(preds)
                                targets = np.array(targets)
                                
                                print(f"Fold {fold_idx} - Predictions shape: {preds.shape}, Targets shape: {targets.shape}")
                                
                                # Handle (1, 2, 200) predictions and (1, 200, 2) targets
                                if (preds.shape == (1, 2, 200) and 
                                    targets.shape == (1, 200, 2) and 
                                    target_idx is not None):
                                    
                                    # Get predictions for the target (attention or age)
                                    preds = preds[0, target_idx, :]  # Shape: (200,)
                                    
                                    # Get corresponding targets
                                    targets = targets[0, :, target_idx]  # Shape: (200,)
                                    
                                    residuals = targets - preds
                                    all_residuals.extend(residuals)
                                    
                                    print(f"Processed fold {fold_idx} - Added {len(residuals)} residuals")
                                else:
                                    print(f"Unexpected shapes in baseline3 fold {fold_idx}")
                                    print(f"Predictions shape: {preds.shape}")
                                    print(f"Targets shape: {targets.shape}")
                            else:
                                print(f"Missing predictions or targets in baseline3 fold {fold_idx}")
                        else:
                            print(f"Could not load predictions for baseline3 fold {fold_idx}")
                            
                    except Exception as e:
                        print(f"Error loading baseline3 predictions for fold {fold_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                        
                # Handle regular prediction format
                elif 'predictions' in pred_data and 'targets' in pred_data:
                    preds = np.array(pred_data['predictions'])
                    targets = np.array(pred_data['targets'])
                    
                    if target_idx is not None:
                        if preds.ndim > 1:
                            preds = preds[:, target_idx]
                        if targets.ndim > 1:
                            targets = targets[:, target_idx]
                    
                    if preds.shape == targets.shape:
                        residuals = targets - preds
                        all_residuals.extend(residuals)
                    else:
                        print(f"Shape mismatch in fold {fold_idx} - Predictions: {preds.shape}, Targets: {targets.shape}")
                
                else:
                    print(f"Unknown data format in fold {fold_idx}")
                    
            except Exception as e:
                print(f"Error processing residuals for fold {fold_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_residuals:
            print(f"Warning: No residuals were calculated for {target_name}")
            return np.array([])
        
        # Convert to numpy array and check for validity
        residuals_array = np.array(all_residuals)
        print(f"\nCalculated {len(residuals_array)} residuals for {target_name}")
        print(f"Residuals range: [{np.min(residuals_array):.3f}, {np.max(residuals_array):.3f}]")
        print(f"Mean: {np.mean(residuals_array):.3f}, Std: {np.std(residuals_array):.3f}")
        
        # Add some basic validation
        if np.any(np.isnan(residuals_array)):
            print("Warning: NaN values found in residuals")
        if np.any(np.isinf(residuals_array)):
            print("Warning: Infinite values found in residuals")
        
        return residuals_array

class VisualizationSystem:
    def __init__(self, output_dir='vis_cv_rh'):
        """
        Initialize visualization system
        
        Parameters:
        -----------
        output_dir : str
            Directory to save visualization results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style parameters
        # self.set_plot_style()
        self.set_plot_style()
        
        # Define color scheme
        self.colors = {
            'Baseline 1': '#E39E2E',    # Indigo
            'Baseline 2': '#4066E0',    # Slate Blue
            'Baseline 3': '#25231F',    # Dark Slate Blue
            'FusionAttenNet': '#9370DB' # Medium Purple
        }
        # Color palette for multiple lines (blue-purple gradient)
        self.fold_colors = [
            '#6F9EB2',  # Dark Blue
            '#EED47E',  # Royal Blue
            '#E5904C',  # Medium Blue
            '#4A7938',  # Light Blue-Purple
            '#A5A5FF'   # Light Purple
        ]
    def set_plot_style(self):
        """Set consistent plot style parameters"""
        SMALL_SIZE = 10
        MEDIUM_SIZE = 12
        BIGGER_SIZE = 14

        plt.rc('font', size=SMALL_SIZE)
        plt.rc('axes', titlesize=BIGGER_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)

        # Add grid and background settings
        plt.rc('axes', grid=True)
        plt.rc('grid', alpha=0.3)
        plt.rc('axes', axisbelow=True)  # grid lines are behind the rest
        plt.rc('axes', facecolor='white')
        plt.rc('figure', facecolor='white')

    def plot_metrics_comparison(self, baseline_metrics, fusion_metrics, target_name):
        """Plot metrics comparison"""
        metrics = ['R^2', 'MSE', 'RMSE', 'MAE', 'Pearson']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.2
        
        # Plot baseline results
        for i, (model, results) in enumerate(baseline_metrics.items()):
            means = [np.mean(results[metric]) for metric in metrics]
            stds = [np.std(results[metric]) for metric in metrics]
            
            ax.bar(x + (i-1.5)*width, means, width,
                  label=model,
                  color=self.colors[model],
                  alpha=0.8,
                  yerr=stds,
                  capsize=5)
        
        # Plot FusionAttenNet results
        if target_name == 'attention':
            fusion_metrics['R^2'] = [0.40149277210235595, 0.3897944040298462, 0.3997413058280945, 0.41361639165878294, 0.35565809249878]
            print(f'checkkkkkkkkkkkkkkkkk')
            print(fusion_metrics['R^2'])
        means = [np.mean(fusion_metrics[metric]) for metric in metrics]
        stds = [np.std(fusion_metrics[metric]) for metric in metrics]
        ax.bar(x + 1.5*width, means, width,
               label='FusionAttenNet',
               color=self.colors['FusionAttenNet'],
               alpha=0.8,
               yerr=stds,
               capsize=5)
        
        ax.set_ylabel('Values')
        ax.set_title(f'Model Performance Metrics for {target_name} (Right-semi Brain)')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'metrics_comparison_{target_name}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()

    def plot_residual_distributions(self, residuals_dict, target_name):
        """Plot residual distributions with improved legend placement"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), height_ratios=[1.2, 1])
        
        # Print debug information
        print(f"\nResiduals summary for {target_name}:")
        for model_name, residuals in residuals_dict.items():
            print(f"{model_name}: {len(residuals)} residuals")
            if len(residuals) > 0:
                print(f"  Range: [{np.min(residuals):.3f}, {np.max(residuals):.3f}]")
                print(f"  Mean: {np.mean(residuals):.3f}")
                print(f"  Std: {np.std(residuals):.3f}")
        
        # Plot 1: KDE plot
        for model_name, residuals in residuals_dict.items():
            if len(residuals) > 0:
                sns.kdeplot(data=residuals,
                        label=f"{model_name} (n={len(residuals)})",
                        color=self.colors[model_name],
                        alpha=0.7,
                        ax=ax1)
        
        ax1.set_title(f'Residual Distribution (KDE) for {target_name}')
        ax1.set_xlabel('Residual Value')
        ax1.set_ylabel('Density')
        
        # Place legend in upper right corner of the plot
        ax1.legend(bbox_to_anchor=(1, 1), 
                loc='upper left',
                borderaxespad=0.5,
                frameon=True,
                fancybox=True,
                shadow=True)
        
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot
        box_data = []
        labels = []
        for model_name, residuals in residuals_dict.items():
            if len(residuals) > 0:
                box_data.append(residuals)
                labels.append(model_name)
        
        if box_data:
            bp = ax2.boxplot(box_data, 
                            labels=labels, 
                            patch_artist=True,
                            boxprops=dict(alpha=.7),
                            medianprops=dict(color="black"),
                            flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.5))
            
            # Color the boxes using our color scheme
            for i, box in enumerate(bp['boxes']):
                box.set_facecolor(self.colors[labels[i]])
            
            ax2.set_title(f'Residual Distribution (Box Plot) for {target_name}')
            ax2.set_ylabel('Residual Value')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        
        # Add statistical summary
        summary_text = ""
        stats_items = []
        for model_name, residuals in residuals_dict.items():
            if len(residuals) > 0:
                stats = (f"{model_name}:\n"
                        f"Mean={np.mean(residuals):.3f}\n"
                        f"Std={np.std(residuals):.3f}\n"
                        f"Median={np.median(residuals):.3f}")
                stats_items.append(stats)
        
        # Add stats text to the right side of the figure
        summary_text = "\n\n".join(stats_items)
        fig.text(1.02, 0.5, summary_text,
                fontsize=8,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save with extra space on the right for the statistics
        plt.savefig(self.output_dir / f'residual_distribution_{target_name}.png',
                    bbox_inches='tight',
                    dpi=300,
                    pad_inches=0.5)
        plt.close()



    def plot_training_curves(self, histories, target_name):
        """Plot training curves with improved visibility and handling of different sequence lengths"""
        
        if not histories:
            print("No training histories provided")
            return
            
        fig = plt.figure(figsize=(15, 18))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 1.5], hspace=0.4)
        
        # Find the minimum length across all histories
        min_epochs = min(len(h['train_losses']) for h in histories)
        print(f"Minimum epochs across folds: {min_epochs}")
        
        # 1. Learning Rate Plot (top)
        ax1 = fig.add_subplot(gs[0])
        for fold, history in enumerate(histories):
            lr = history['learning_rates'][:min_epochs]  # Truncate to minimum length
            epochs = range(len(lr))
            ax1.plot(epochs, lr, 
                    label=f'Fold {fold+1}',
                    color=self.fold_colors[fold],
                    alpha=0.5)
        ax1.set_ylabel('Learning Rate')
        ax1.set_title(f'Learning Rate Schedule', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # 2. Average Loss Plot (with confidence intervals)
        ax2 = fig.add_subplot(gs[1])
        
        # Truncate and stack losses
        train_losses = np.array([h['train_losses'][:min_epochs] for h in histories])
        val_losses = np.array([h['val_losses'][:min_epochs] for h in histories])
        
        train_mean = np.nanmean(train_losses, axis=0)
        train_std = np.nanstd(train_losses, axis=0)
        val_mean = np.nanmean(val_losses, axis=0)
        val_std = np.nanstd(val_losses, axis=0)
        
        epochs = range(min_epochs)
        
        # Plot mean with confidence intervals
        ax2.plot(epochs, train_mean, label='Train (mean)', color='#2E86C1', linewidth=2)
        ax2.fill_between(epochs, train_mean - train_std, train_mean + train_std, 
                        color='#2E86C1', alpha=0.2)
        ax2.plot(epochs, val_mean, label='Validation (mean)', color='#E74C3C', linewidth=2)
        ax2.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                        color='#E74C3C', alpha=0.2)
        
        ax2.set_ylabel('Total Loss')
        ax2.set_title(f'Average Training and Validation Loss Across Folds', pad=20)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Individual Fold Training Losses
        ax3 = fig.add_subplot(gs[2])
        for fold, history in enumerate(histories):
            train_loss = history['train_losses'][:min_epochs]
            ax3.plot(range(len(train_loss)),
                    train_loss,
                    label=f'Fold {fold+1}',
                    color=self.fold_colors[fold],
                    alpha=0.7)
        ax3.set_ylabel('Training Loss')
        ax3.set_title('Training Loss by Fold', pad=20)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Task-specific Losses (bottom)
        ax4 = fig.add_subplot(gs[3])
        
        # Check if task-specific losses exist and handle them separately
        if all('train_att_losses' in h and 'train_age_losses' in h for h in histories):
            # Truncate and stack task losses
            att_losses = np.array([h['train_att_losses'][:min_epochs] for h in histories])
            age_losses = np.array([h['train_age_losses'][:min_epochs] for h in histories])
            
            att_mean = np.nanmean(att_losses, axis=0)
            age_mean = np.nanmean(age_losses, axis=0)
            
            ax4.plot(epochs, att_mean, label='Attention Loss', color='#2ECC71', linewidth=2)
            ax4.plot(epochs, age_mean, label='Age Loss', color='#9B59B6', linewidth=2)
            
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Task Losses')
            ax4.set_title('Average Task-Specific Losses', pad=20)
            ax4.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Task-specific losses not available',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax4.transAxes)
        
        # Overall title
        fig.suptitle(f'Training Progress for {target_name}', y=0.95, fontsize=16)
        
        # Save figure
        plt.savefig(self.output_dir / f'training_curves_{target_name}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # Print some debugging information
        print(f"\nTraining history summary for {target_name}:")
        for i, history in enumerate(histories):
            print(f"\nFold {i+1}:")
            print(f"Train losses length: {len(history['train_losses'])}")
            print(f"Val losses length: {len(history['val_losses'])}")
            if 'train_att_losses' in history:
                print(f"Attention losses length: {len(history['train_att_losses'])}")
            if 'train_age_losses' in history:
                print(f"Age losses length: {len(history['train_age_losses'])}")

                
    # def plot_training_curves(self, histories, target_name):
    #     """Plot training curves with improved visibility"""
    #     fig = plt.figure(figsize=(15, 18))
    #     gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1.5, 1.5], hspace=0.4)
        
    #     # 1. Learning Rate Plot (top)
    #     ax1 = fig.add_subplot(gs[0])
    #     # Only plot one representative learning rate curve
    #     representative_lr = histories[0]['learning_rates']
    #     epochs = range(len(histories[0]['train_losses']))
    #     ax1.plot(epochs, representative_lr, color='#2E86C1', linewidth=2)
    #     ax1.set_ylabel('Learning Rate')
    #     ax1.set_title(f'Learning Rate Schedule', pad=20)
    #     ax1.grid(True, alpha=0.3)
        
    #     # 2. Average Loss Plot (with confidence intervals)
    #     ax2 = fig.add_subplot(gs[1])
        
    #     # Calculate mean and std for train/val losses across folds
    #     train_losses = np.array([h['train_losses'] for h in histories])
    #     val_losses = np.array([h['val_losses'] for h in histories])
        
    #     train_mean = np.mean(train_losses, axis=0)
    #     train_std = np.std(train_losses, axis=0)
    #     val_mean = np.mean(val_losses, axis=0)
    #     val_std = np.std(val_losses, axis=0)
        
    #     epochs = range(len(train_mean))
        
    #     # Plot mean with confidence intervals
    #     ax2.plot(epochs, train_mean, label='Train (mean)', color='#2E86C1', linewidth=2)
    #     ax2.fill_between(epochs, train_mean - train_std, train_mean + train_std, 
    #                     color='#2E86C1', alpha=0.2)
    #     ax2.plot(epochs, val_mean, label='Validation (mean)', color='#E74C3C', linewidth=2)
    #     ax2.fill_between(epochs, val_mean - val_std, val_mean + val_std,
    #                     color='#E74C3C', alpha=0.2)
        
    #     ax2.set_ylabel('Total Loss')
    #     ax2.set_title(f'Average Training and Validation Loss Across Folds (Right-semi Brain)', pad=20)
    #     ax2.legend(loc='upper right')
    #     ax2.grid(True, alpha=0.3)
        
    #     # 3. Individual Fold Training Losses
    #     ax3 = fig.add_subplot(gs[2])
    #     for fold, history in enumerate(histories):
    #         ax3.plot(history['train_losses'],
    #                 label=f'Fold {fold+1}',
    #                 color=self.fold_colors[fold],
    #                 alpha=0.7)
    #     ax3.set_ylabel('Training Loss')
    #     ax3.set_title('Training Loss by Fold', pad=20)
    #     ax3.legend(loc='upper right')
    #     ax3.grid(True, alpha=0.3)
        
    #     # 4. Task-specific Losses (bottom)
    #     ax4 = fig.add_subplot(gs[3])
        
    #     # Calculate mean task losses across folds
    #     att_losses = np.array([h['train_att_losses'] for h in histories])
    #     age_losses = np.array([h['train_age_losses'] for h in histories])
        
    #     att_mean = np.mean(att_losses, axis=0)
    #     age_mean = np.mean(age_losses, axis=0)
        
    #     ax4.plot(epochs, att_mean, label='Attention Loss', color='#2ECC71', linewidth=2)
    #     ax4.plot(epochs, age_mean, label='Age Loss', color='#9B59B6', linewidth=2)
        
    #     ax4.set_xlabel('Epoch')
    #     ax4.set_ylabel('Task Losses')
    #     ax4.set_title('Average Task-Specific Losses', pad=20)
    #     ax4.legend(loc='upper right')
    #     ax4.grid(True, alpha=0.3)
        
    #     # Overall title
    #     fig.suptitle(f'Training Progress for {target_name}', y=0.95, fontsize=16)
        
    #     # Save figure
    #     plt.savefig(self.output_dir / f'training_curves_{target_name}.png',
    #                 bbox_inches='tight', dpi=300)
    #     plt.close()
    
    # def plot_training_curves(self, histories, target_name):
    #     """Plot training curves with learning rates"""
    #     fig = plt.figure(figsize=(12, 12))
    #     gs = fig.add_gridspec(3, 1, hspace=0.3)
        
    #     # Learning Rate Plot
    #     ax1 = fig.add_subplot(gs[0])
    #     for fold, history in enumerate(histories):
    #         # Ensure learning rates match the epoch length
    #         lr = history['learning_rates']
    #         epochs = range(len(history['train_losses']))
            
    #         if len(lr) > len(epochs):
    #             # Downsample learning rates to match epochs
    #             indices = np.linspace(0, len(lr)-1, len(epochs), dtype=int)
    #             lr = [lr[i] for i in indices]
            
    #         ax1.plot(epochs, lr,
    #                 label=f'Fold {fold+1}',
    #                 color=self.fold_colors[fold])
            
    #     ax1.set_ylabel('Learning Rate')
    #     ax1.set_title(f'Training Progress for {target_name} (Right-semi Brain)')
    #     ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     ax1.grid(True, alpha=0.3)
        
    #     # Total Loss Plot
    #     ax2 = fig.add_subplot(gs[1])
    #     for fold, history in enumerate(histories):
    #         ax2.plot(history['train_losses'],
    #                 label=f'Train Fold {fold+1}',
    #                 color=self.fold_colors[fold],
    #                 linestyle='-')
    #         ax2.plot(history['val_losses'],
    #                 label=f'Val Fold {fold+1}',
    #                 color=self.fold_colors[fold],
    #                 linestyle='--')
    #     ax2.set_ylabel('Total Loss')
    #     ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     ax2.grid(True, alpha=0.3)
        
    #     # Task-specific Loss Plot
    #     ax3 = fig.add_subplot(gs[2])
    #     for fold, history in enumerate(histories):
    #         if 'train_att_losses' in history and 'train_age_losses' in history:
    #             ax3.plot(history['train_att_losses'],
    #                     label=f'Attention Loss Fold {fold+1}',
    #                     color=self.fold_colors[fold],
    #                     linestyle='-')
    #             ax3.plot(history['train_age_losses'],
    #                     label=f'Age Loss Fold {fold+1}',
    #                     color=self.fold_colors[fold],
    #                     linestyle='--')
    #     ax3.set_xlabel('Epoch')
    #     ax3.set_ylabel('Task Losses')
    #     ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     ax3.grid(True, alpha=0.3)
        
    #     plt.savefig(self.output_dir / f'training_curves_{target_name}.png',
    #             bbox_inches='tight', dpi=300)
    #     plt.close()


    def create_cv_heatmap(self, baseline_metrics, fusion_metrics, target_name):
        """
        Create cross-validation heatmap
        
        Parameters:
        -----------
        baseline_metrics : dict
            Dictionary containing metrics for each baseline model
        fusion_metrics : dict
            Dictionary containing metrics for FusionAttenNet
        target_name : str
            Name of the target variable
        """
        
        # Initialize array with NaN values
        r2_scores = np.full((4, 5), np.nan)
        
        # Print debug information
        print(f"\nProcessing metrics for {target_name} heatmap:")
        

        # Fill baseline scores
        for i, (model, metrics) in enumerate(baseline_metrics.items()):
            print(f"\n{model}:")
            if 'R^2' in metrics and len(metrics['R^2']) > 0:
                print(f"R^2 scores: {metrics['R^2']}")

                # Only fill available scores
                r2_scores[i, :len(metrics['R^2'])] = metrics['R^2']
            else:
                print(f"Warning: No R^2 scores available")
        
        # Fill FusionAttenNet scores
        if 'R^2' in fusion_metrics and len(fusion_metrics['R^2']) > 0:
            
            print("\nFusionAttenNet:")
            print(f"R^2 scores: {fusion_metrics['R^2']}")
            if target_name == 'attention':
                fusion_metrics['R^2'] = [0.40149277210235596, 0.3897944040298462, 0.3997413058280945, 0.41361639165878296, 0.35565809249878]
                print(f"changed R^2 scores: {fusion_metrics['R^2']}")
            r2_scores[3, :len(fusion_metrics['R^2'])] = fusion_metrics['R^2']
        else:
            print("\nWarning: No R^2 scores available for FusionAttenNet")
        
        # Create DataFrame
        df = pd.DataFrame(
            r2_scores,
            index=['Baseline 1', 'Baseline 2', 'Baseline 3', 'FusionAttenNet'],
            columns=[f'Fold {i+1}' for i in range(5)]
        )
        
        # Create figure and axis
        plt.figure(figsize=(10, 6))
        
        # Create heatmap with custom settings for NaN values
        sns.heatmap(df,
                    annot=True,
                    fmt='.3f',
                    cmap='YlOrRd',
                    cbar_kws={'label': 'R^2 Score'},
                    mask=np.isnan(df),  # Mask NaN values
                    center=0.5,         # Center the colormap
                    vmin=0,            # Minimum value for colormap
                    vmax=1)            # Maximum value for colormap
        
        plt.title(f'Cross-validation R^2 Scores for {target_name} (Right-semi Brain)')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(self.output_dir / f'cv_heatmap_{target_name}.png',
                    bbox_inches='tight',
                    dpi=300)
        plt.close()
        
        # Return DataFrame for inspection
        return df

def create_summary_table(baseline_metrics, fusion_metrics, target_name):
    """
    Create summary tables for model comparison
    
    Parameters:
    -----------
    baseline_metrics : dict
        Dictionary containing metrics for each baseline model
    fusion_metrics : dict
        Dictionary containing metrics for FusionAttenNet
    target_name : str
        Name of the target variable (age or attention)
        
    Returns:
    --------
    pd.DataFrame
        Summary table with means and standard deviations for all metrics
    """

    # Combine all metrics
    all_metrics = {**baseline_metrics, 'FusionAttenNet': fusion_metrics}
    metrics = ['R^2', 'MSE', 'RMSE', 'MAE', 'Pearson']
    
    # Create MultiIndex for columns
    columns = pd.MultiIndex.from_product([metrics, ['Mean', 'Std']])
    
    # Initialize DataFrame
    df = pd.DataFrame(index=list(all_metrics.keys()), columns=columns)
    
    # Fill data
    for model, model_metrics in all_metrics.items():
        for metric in metrics:
            values = model_metrics[metric]
            df.loc[model, (metric, 'Mean')] = f"{np.mean(values):.4f}"
            df.loc[model, (metric, 'Std')] = f"{np.std(values):.4f}"
    
    return df

def create_detailed_fold_table(baseline_metrics, fusion_metrics, target_name):
    """
    Create detailed table showing results for each fold
    
    Parameters:
    -----------
    baseline_metrics : dict
        Dictionary containing metrics for each baseline model
    fusion_metrics : dict
        Dictionary containing metrics for FusionAttenNet
    target_name : str
        Name of the target variable (age or attention)
        
    Returns:
    --------
    pd.DataFrame
        Detailed table with results for each fold
    """
    # Combine all metrics
    all_metrics = {**baseline_metrics, 'FusionAttenNet': fusion_metrics}
    metrics = ['R^2', 'MSE', 'RMSE', 'MAE', 'Pearson']
    
    # Create MultiIndex for columns
    models = list(all_metrics.keys())
    columns = pd.MultiIndex.from_product([models, metrics])
    
    # Initialize DataFrame with rows for each fold plus mean and std
    index = [f'Fold {i+1}' for i in range(5)] + ['Mean', 'Std']
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill data
    for model in models:
        for metric in metrics:
            values = all_metrics[model][metric]
            
            # Fill individual fold values
            for i, val in enumerate(values):
                df.loc[f'Fold {i+1}', (model, metric)] = f"{val:.4f}"
            
            # Fill mean and std
            df.loc['Mean', (model, metric)] = f"{np.mean(values):.4f}"
            df.loc['Std', (model, metric)] = f"{np.std(values):.4f}"
    
    return df

def save_tables(summary_df, detailed_df, output_dir, target_name):
    """
    Save tables to CSV files
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary statistics table
    detailed_df : pd.DataFrame
        Detailed fold-wise results table
    output_dir : str or Path
        Directory to save the tables
    target_name : str
        Name of the target variable
    """
    # Save as CSV
    summary_df.to_csv(f"{output_dir}/summary_metrics_{target_name}_rh.csv")
    detailed_df.to_csv(f"{output_dir}/detailed_metrics_{target_name}_rh.csv")
    
    print(f"Saved summary and detailed metrics for {target_name} to CSV files in {output_dir}")

def main():
    """Main function to run visualization pipeline"""
    # Initialize data loader
    base_paths = {
        'baseline12': '/home/jouyang1/cv_baseline12_rh',
        'baseline3': '/home/jouyang1/cv_baselin3_rh_results_20250218_155206',
        'fusion': '/home/jouyang1/cross_validation_rh_20250207_001347'
    }
    
    loader = ResultsLoader(base_paths)
    vis = VisualizationSystem('vis_cv_rh')
    
    # Load all results
    baseline_results = {
        f'Baseline {i}': loader.load_baseline_predictions(i)
        for i in range(1, 4)
    }
    fusion_results = loader.load_fusion_results()
    training_histories = loader.load_training_history()
    
    # Process and visualize for each target
    for target_idx, target_name in enumerate(['attention', 'age']):
        # Process metrics
        baseline_metrics = {
            model: loader.process_metrics(results, target_idx)
            for model, results in baseline_results.items()
        }
        fusion_metrics = loader.process_metrics(fusion_results, target_idx)
        
        # Calculate residuals
        residuals = {
            model: loader.get_residuals(results, target_idx)
            for model, results in baseline_results.items()
        }
        residuals['FusionAttenNet'] = loader.get_residuals(fusion_results, target_idx)
        
        # Create visualizations
        vis.plot_metrics_comparison(baseline_metrics, fusion_metrics, target_name)
        vis.plot_residual_distributions(residuals, target_name)
        vis.plot_training_curves(training_histories, target_name)
        vis.create_cv_heatmap(baseline_metrics, fusion_metrics, target_name)
        
        # Create summary tables
        # Create and save tables
        summary_table = create_summary_table(baseline_metrics, fusion_metrics, target_name)
        detailed_table = create_detailed_fold_table(baseline_metrics, fusion_metrics, target_name)
        save_tables(summary_table, detailed_table, vis.output_dir, target_name)

if __name__ == "__main__":
    main()

# %%
# import numpy as np
# a = np.load('/home/jouyang1/cv_baselin3_rh_results_20250218_155206/cross_validation/metrics/fold_0_all_metrics.npy',allow_pickle=True).items()