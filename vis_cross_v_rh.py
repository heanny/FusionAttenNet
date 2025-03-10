import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define file paths
baseline1_analysis_path = '/home/jouyang1/cv_baseline12_rh/univariate/analysis/fold_0_predictions.npy'
baseline2_analysis_path = '/home/jouyang1/cv_baseline12_rh/phenotype/analysis/fold_0_predictions.npy'
baseline3_analysis_path = '/home/jouyang1/cv_baselin3_rh_results_20250218_155206/cross_validation/analysis/fold_0_predictions.npy'
fusion_results_path = '/home/jouyang1/cross_validation_rh_20250207_001347/cross_validation/fold_results/fold_0_results.npy'

fusion_train_history_path = '/home/jouyang1/cross_validation_rh_20250207_001347/plots/training_plots/fold_0_training_history.npy'

# Create directory for saving plots
output_dir = 'vis_cv_rh'
os.makedirs(output_dir, exist_ok=True)

# Model names
models = ['Baseline 1', 'Baseline 2', 'Baseline 3', 'FusionAttenNet']

# Initialize data containers
metrics = {model: {'sum_att': {'R^2': [], 'MAE': [], 'RMSE': [], 'MSE': []},
                   'age': {'R^2': [], 'MAE': [], 'RMSE': [], 'MSE': []}} for model in models}
residuals = {model: {'sum_att': [], 'age': []} for model in models}

# Load metrics and residuals
for fold in range(5):
    # Load Baseline predictions and residuals
    for i, (model, path) in enumerate(zip(['Baseline 1', 'Baseline 2', 'Baseline 3'],
                                          [baseline1_analysis_path, baseline2_analysis_path, baseline3_analysis_path])):
        data = np.load(path.format(fold), allow_pickle=True).item()
        for target in ['sum_att', 'age']:
            preds = np.array(data['predictions'])
            actuals = np.array(data['targets'])
            target_index = 0 if target == 'sum_att' else 1

            # Handle Baseline 3 shape mismatch
            if model == 'Baseline 3':
                if preds.shape[0] == 1:
                    preds = preds.squeeze()  # (1, 200) -> (200,)
                if actuals.shape[0] == 1:
                    actuals = actuals.squeeze()  # (1, 2) -> (2,)
                if preds.size == 200 and actuals.size == 2:
                    # Assume predictions for 200 samples, repeat actuals
                    actuals = np.tile(actuals[target_index], preds.shape[0])
                else:
                    actuals = actuals[target_index]

            else:
                preds = preds[:, target_index] if preds.ndim > 1 else preds
                actuals = actuals[:, target_index] if actuals.ndim > 1 else actuals

            # Debug prints
            print(f"[{model}] Fold {fold} - Target: {target}")
            print(f"Preds shape: {preds.shape}, Actuals shape: {actuals.shape}")

            # Ensure shape match before subtraction
            if preds.shape == actuals.shape:
                residual = actuals - preds
                residuals[model][target].extend(residual)
            else:
                print(f"Shape mismatch in {model}, fold {fold}, target {target}")

    # Load FusionAttenNet predictions and residuals
    data = np.load(fusion_results_path.format(fold), allow_pickle=True).item()
    for target in ['sum_att', 'age']:
        preds = np.array(data['predictions'])
        actuals = np.array(data['targets'])

        target_index = 0 if target == 'sum_att' else 1
        preds = preds[:, target_index] if preds.ndim > 1 else preds
        actuals = actuals[:, target_index] if actuals.ndim > 1 else actuals

        print(f"[FusionAttenNet] Fold {fold} - Target: {target}")
        print(f"Preds shape: {preds.shape}, Actuals shape: {actuals.shape}")

        if preds.shape == actuals.shape:
            residual = actuals - preds
            residuals['FusionAttenNet'][target].extend(residual)
        else:
            print(f"Shape mismatch in FusionAttenNet, fold {fold}, target {target}")

# 1️⃣ Residual Distribution
for target in ['sum_att', 'age']:
    plt.figure(figsize=(10, 6))
    for model in models:
        if residuals[model][target]:
            sns.histplot(residuals[model][target], bins=20, kde=True, label=model, alpha=0.5)

    plt.title(f'Residual Distribution for {target}')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'residual_distribution_{target}.png'))
    plt.close()

# 2️⃣ Performance Comparison for Both Targets
for target in ['sum_att', 'age']:
    plt.figure(figsize=(10, 6))
    mean_r2 = [np.mean([np.var(r) for r in residuals[model][target]]) for model in models]
    std_r2 = [np.std([np.var(r) for r in residuals[model][target]]) for model in models]

    plt.bar(models, mean_r2, yerr=std_r2, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title(f'Model Comparison (R²) for {target}')
    plt.ylabel('R² Score')
    plt.savefig(os.path.join(output_dir, f'model_comparison_r2_{target}.png'))
    plt.close()

# 3️⃣ Cross-Validation Boxplot for Multiple Metrics
for target in ['sum_att', 'age']:
    for metric in ['R^2', 'MAE', 'RMSE', 'MSE']:
        plt.figure(figsize=(12, 6))
        data = [metrics[model][target][metric] for model in models]
        sns.boxplot(data=data)
        plt.xticks(ticks=range(len(models)), labels=models)
        plt.title(f'{metric} Across Folds for {target}')
        plt.ylabel(metric)
        plt.savefig(os.path.join(output_dir, f'cv_boxplot_{metric}_{target}.png'))
        plt.close()

# 4️⃣ Learning Rate Fix (Downsampling)
batch_per_epoch = 2519
for fold in range(5):
    file_path = fusion_train_history_path.format(fold)
    history = np.load(file_path, allow_pickle=True).item()
    actual_epochs = range(1, len(history['train_losses']) + 1)

    # Downsample learning rates
    learning_rates = history['learning_rates']
    if len(learning_rates) != len(actual_epochs):
        lr_per_epoch = [np.mean(learning_rates[i * batch_per_epoch:(i + 1) * batch_per_epoch]) 
                        for i in range(len(actual_epochs))]
    else:
        lr_per_epoch = learning_rates

    # Plot
    plt.figure(figsize=(12, 8))

    # Total Loss
    plt.subplot(3, 1, 1)
    plt.plot(actual_epochs, history['train_losses'], label='Train Loss', color='purple')
    plt.plot(actual_epochs, history['val_losses'], label='Validation Loss', color='green')
    plt.title(f'Fold {fold + 1} - Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Subtask Losses
    plt.subplot(3, 1, 2)
    plt.plot(actual_epochs, history['train_att_losses'], label='Attention Task Loss', color='cyan')
    plt.plot(actual_epochs, history['train_age_losses'], label='Age Task Loss', color='magenta')
    plt.title(f'Fold {fold + 1} - Subtask Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Learning Rate (Downsampled)
    plt.subplot(3, 1, 3)
    plt.plot(actual_epochs, lr_per_epoch, label='Learning Rate', color='orange')
    plt.title(f'Fold {fold + 1} - Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fusionattennet_learning_curve_fold_{fold + 1}.png'))
    plt.close()

print(f"All plots saved in {output_dir}")
