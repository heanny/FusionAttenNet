# 
import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats
from scipy.stats import pearsonr
import logging
import os
import gc
from datetime import datetime

# Configuration
HEMISPHERE = 'lh'  # 可以改成 'lh' 用于左脑
FEATURE_PREFIX = f"{HEMISPHERE}."  # 自动生成正确的文件前缀

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'baseline_model_{HEMISPHERE}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def validate_data(data, data_name):
    """
    验证加载的数据，处理不同的数据类型
    """
    logging.info(f"\nValidating {data_name}:")
    logging.info(f"Shape: {data.shape if hasattr(data, 'shape') else len(data)}")
    logging.info(f"Type: {type(data)}")
    
    if isinstance(data, np.ndarray):
        logging.info(f"Data type: {data.dtype}")
        if data.size > 0:
            logging.info(f"Sample of data:\n{data[:2]}")
            
            # 只对数值类型的数组检查NaN和Inf
            if np.issubdtype(data.dtype, np.number):
                logging.info(f"Contains NaN: {np.any(np.isnan(data))}")
                logging.info(f"Contains Inf: {np.any(np.isinf(data))}")
            else:
                logging.info("Data is non-numeric, skipping NaN/Inf check")
            
            # 对于对象类型的数组，检查None值
            if data.dtype == object:
                none_count = sum(1 for x in data.flatten() if x is None)
                logging.info(f"Contains {none_count} None values")
    
    logging.info("Validation complete\n")

def load_data():
    """
    加载所需的所有数据文件
    """
    try:
        logging.info("\nLoading data files...")
        
        # 加载样本subjects数据
        sample_subjects = np.load('sample_ids_filename_updated.npy', allow_pickle=True)
        validate_data(sample_subjects, "sample_subjects")
        
        # 加载vertex IDs
        vertex_ids = np.load(f'ij_id_{HEMISPHERE}.npy', allow_pickle=True).astype(int)
        validate_data(vertex_ids, "vertex_ids")
        
        # 加载表型数据
        phenotype_data = np.load('sample_normalised_phenotype.npy', allow_pickle=True)
        validate_data(phenotype_data, "phenotype_data")
        
        # 验证数据一致性
        if len(sample_subjects) != len(phenotype_data):
            raise ValueError(f"Mismatch in data lengths: {len(sample_subjects)} subjects but {len(phenotype_data)} phenotype records")
        
        logging.info(f"Successfully loaded data: {len(sample_subjects)} subjects, {len(vertex_ids)} vertices")
        logging.info("\nData ranges:")
        logging.info(f"Attention Problem Scores: {phenotype_data[:, 0].min():.2f} to {phenotype_data[:, 0].max():.2f}")
        logging.info(f"Age: {phenotype_data[:, 2].min():.2f} to {phenotype_data[:, 2].max():.2f}")
        
        return sample_subjects, vertex_ids, phenotype_data
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def extract_vertex_features(subject_info, vertex_id):
    """
    提取单个vertex的所有特征
    """
    features = []
    try:
        # 提取4个特征文件的路径
        feature_files = {
            'thickness': f'{FEATURE_PREFIX}thickness.fwhm10.fsaverage.mgh',
            'volume': f'{FEATURE_PREFIX}volume.fwhm10.fsaverage.mgh',
            'area': f'{FEATURE_PREFIX}area.fwhm10.fsaverage.mgh',
            'wg_ratio': f'{FEATURE_PREFIX}w_g.pct.mgh.fwhm10.fsaverage.mgh'
        }
        
        for feature_name, file_name in feature_files.items():
            file_path = subject_info[2][file_name]
            feature_data = nib.load(file_path).get_fdata()
            value = feature_data[vertex_id][0][0]
            features.append(value)
            
        return np.array(features)
    except Exception as e:
        logging.error(f"Error extracting features for vertex {vertex_id}: {str(e)}")
        return None

def calculate_vertex_Scores(vertex_features, targets, phenotypes, n_vertices=1000):
    """
    计算vertex的综合得分
    """
    try:
        logging.info(f"\nCalculating Scores for vertex features shape: {vertex_features.shape}")
        
        # 1. 目标相关性得分
        n_tests = n_vertices * 2  # 两个目标变量
        bonferroni_alpha = 0.05 / n_tests   # Bonferroni校正后的显著性水平
        logging.info(f"Using Bonferroni corrected alpha: {bonferroni_alpha:.8f}")

        target_correlations = []
        for i in range(targets.shape[1]):
            corr, p_value = pearsonr(vertex_features[:, 0], targets[:, i])
            is_significant = p_value < bonferroni_alpha
            target_correlations.append(abs(corr) if is_significant else 0)
            logging.info(f"Correlation with target {i}: {corr:.4f} (p={p_value:.4f}, significant: {is_significant})")
        target_Scores = np.mean(target_correlations)
        
        # 2. 特征独立性得分
        independence_Scores = []
        for i in range(phenotypes.shape[1]):
            corr = stats.pearsonr(vertex_features[:, 0], phenotypes[:, i])[0]
            independence_Scores.append(1 - abs(corr))  # 相关性越低，独立性越高
        
        independence_Scores = np.mean(independence_Scores)
        
        # 3. 稳定性得分
        n_bootstrap = 100
        bootstrap_corrs = []
        n_samples = len(vertex_features)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            corr = stats.pearsonr(vertex_features[indices, 0], 
                                targets[indices, 0])[0]
            bootstrap_corrs.append(corr)
        
        stability_Scores = 1 / np.std(bootstrap_corrs)
        
        # 综合得分
        final_Scores = (0.4 * target_Scores + 
                      0.3 * independence_Scores + 
                      0.3 * stability_Scores)
        
        Scores_details = {
            'target_Scores': target_Scores,
            'independence_Scores': independence_Scores,
            'stability_Scores': stability_Scores,
            'bootstrap_std': np.std(bootstrap_corrs)
        }
        
        logging.info(f"Scores details: {Scores_details}")
        
        return final_Scores, Scores_details
    except Exception as e:
        logging.error(f"Error calculating vertex Scores: {str(e)}")
        return None, None

def select_best_vertex(sample_subjects, vertex_ids, phenotype_data):
    """
    选择最佳vertex
    """
    logging.info("\nStarting vertex selection process...")
    
    # 准备目标变量和表型数据
    targets = np.array([
        phenotype_data[:, 0],  # Attention Problem_Scores
        phenotype_data[:, 2]   # age
    ]).T
    
    extra_features = np.array([
        phenotype_data[:, 1],  # aggressive_behaviour_Scores
        phenotype_data[:, 3],  # sex
        phenotype_data[:, 4]   # maternal_edu_level
    ]).T
    
    best_Scores = -np.inf
    best_vertex = None
    best_features = None
    
    # 为了效率，我们可以只检查一部分vertices
    sample_vertices = np.random.choice(vertex_ids, size=1000, replace=False)
    
    total_vertices = len(sample_vertices)
    total_subjects = len(sample_subjects)
    
    for i, vertex_id in enumerate(sample_vertices):
        if i % 10 == 0:  # 每处理10个vertex输出一次进度
            progress = (i / total_vertices) * 100
            current_time = datetime.now().strftime("%H:%M:%S")
            logging.info(f"[{current_time}] Progress: {i}/{total_vertices} vertices ({progress:.2f}%)")
            
        logging.info(f"\nProcessing vertex {vertex_id}:")
        logging.info(f"Collecting features from {total_subjects} subjects...")
        
        # 收集所有subject的这个vertex的特征
        vertex_features = []
        for subject in sample_subjects:
            features = extract_vertex_features(subject, vertex_id)
            if features is not None:
                vertex_features.append(features)
        
        if len(vertex_features) == 0:
            continue
            
        vertex_features = np.array(vertex_features)
        
        # 计算该vertex的得分
        Scores, Scores_details = calculate_vertex_Scores(
            vertex_features, targets, extra_features, n_vertices=len(sample_vertices))
        
        if Scores is not None and Scores > best_Scores:
            best_Scores = Scores
            best_vertex = vertex_id
            best_features = vertex_features
            logging.info(f"\nFound new best vertex {vertex_id} with Scores {Scores:.4f}")
            logging.info(f"Scores details: {Scores_details}")
    
    return best_vertex, best_features

def train_ridge_model(vertex_features, phenotype_data):
    """
    训练岭回归模型
    """
    logging.info("\nStarting Ridge Regression model training...")
    
    # 准备特征和目标
    X = np.hstack([
        vertex_features,
        phenotype_data[:, [1, 3, 4]]  # extra features
    ])
    
    y = phenotype_data[:, [0, 2]]  # Attention Problem_Scores and age
    
    # 输出特征和目标的基本统计信息
    logging.info("\nFeature Statistics:")
    for i in range(X.shape[1]):
        logging.info(f"Feature {i}: mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}")
    
    logging.info("\nTarget Statistics:")
    logging.info(f"Attention Problem Scores: mean={y[:, 0].mean():.4f}, std={y[:, 0].std():.4f}")
    logging.info(f"Age: mean={y[:, 1].mean():.4f}, std={y[:, 1].std():.4f}")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    logging.info(f"\nTraining set size: {X_train.shape[0]}")
    logging.info(f"Test set size: {X_test.shape[0]}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    models = []
    predictions = []
    metrics = []
    
    target_names = ['Attention Problem Scores', 'Age']
    
    for i in range(y_train.shape[1]):
        logging.info(f"\nTraining model for {target_names[i]}...")
        
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train[:, i])
        models.append(model)
        
        # 输出特征重要性
        feature_importance = np.abs(model.coef_)
        for j, importance in enumerate(feature_importance):
            logging.info(f"Feature {j} importance: {importance:.4f}")
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        predictions.append(y_pred)
        
        # 计算评估指标
        metrics.append({
            'r2': r2_score(y_test[:, i], y_pred),
            'mse': mean_squared_error(y_test[:, i], y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test[:, i], y_pred)),
            'mae': mean_absolute_error(y_test[:, i], y_pred),
            'pearson': pearsonr(y_test[:, i], y_pred)[0]
        })
        
        logging.info(f"\nMetrics for {target_names[i]}:")
        for metric_name, value in metrics[-1].items():
            logging.info(f"{metric_name}: {value:.4f}")
    
    return models, predictions, metrics, (X_test, y_test)

def save_results(results_dict, hemisphere):
    """
    保存模型结果
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'baseline_results_{hemisphere}_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存结果为numpy文件
    np.save(os.path.join(save_dir, 'model_results.npy'), results_dict)
    
    # 保存详细结果为文本文件
    with open(os.path.join(save_dir, 'detailed_results.txt'), 'w') as f:
        f.write(f"Baseline Model Results for {hemisphere} hemisphere\n")
        f.write(f"Generated at: {timestamp}\n\n")
        
        f.write(f"Best Vertex ID: {results_dict['best_vertex']}\n\n")
        
        f.write("Model Performance:\n")
        for target, metrics in results_dict['metrics'].items():
            f.write(f"\n{target} Metrics:\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
        
        if 'feature_importance' in results_dict:
            f.write("\nFeature Importance:\n")
            for model_name, importances in results_dict['feature_importance'].items():
                f.write(f"\n{model_name}:\n")
                for i, importance in enumerate(importances):
                    f.write(f"Feature {i}: {importance:.4f}\n")
    
    logging.info(f"\nResults saved to directory: {save_dir}")

def main():
    try:
        logging.info(f"\nStarting baseline model analysis for {HEMISPHERE} hemisphere")
        
        # 1. 加载数据
        sample_subjects, vertex_ids, phenotype_data = load_data()
        
        # 2. 选择最佳vertex
        best_vertex, best_features = select_best_vertex(
            sample_subjects, vertex_ids, phenotype_data)
        
        if best_vertex is None:
            raise ValueError("No suitable vertex found")
        
        logging.info(f"\nSelected best vertex: {best_vertex}")
        logging.info(f"Best vertex features shape: {best_features.shape}")
        
        # 3. 训练岭回归模型
        models, predictions, metrics, test_data = train_ridge_model(
            best_features, phenotype_data)
        
        # 4. 输出最终结果
        logging.info("\nFinal Results Summary:")
        target_names = ['Attention Problem Scores', 'Age']
        for i, name in enumerate(target_names):
            logging.info(f"\nMetrics for {name}:")
            for metric_name, value in metrics[i].items():
                logging.info(f"{metric_name}: {value:.4f}")
        
        # 准备保存结果
        results_dict = {
            'best_vertex': best_vertex,
            'model_parameters': {i: model.get_params() for i, model in enumerate(models)},
            'metrics': {
                'Attention_Problem_Scores': metrics[0],
                'Age': metrics[1]
            },
            'feature_importance': {
                'Attention_Problem_Scores_Model': models[0].coef_.tolist(),
                'Age_Model': models[1].coef_.tolist()
            },
            'model_predictions': {
                'Attention_Problem_Scores': predictions[0].tolist(),
                'Age': predictions[1].tolist()
            },
            'test_data': {
                'X_test': test_data[0].tolist(),
                'y_test': test_data[1].tolist()
            }
        }
        
        # 保存结果
        save_results(results_dict, HEMISPHERE)
        
        logging.info("\nBaseline model training completed successfully!")
        return best_vertex, models, metrics
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
# %%
