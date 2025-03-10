import numpy as np
import nibabel as nib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats
from scipy.stats import pearsonr
import logging
from sklearn.model_selection import KFold
import os
import gc
from datetime import datetime

# Configuration
HEMISPHERE = 'rh'  # 可以改成 'lh' 用于左脑
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
class ResultsManager:
    """结果管理器，用于保存和组织实验结果"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.create_directories()
        
    def create_directories(self):
        """创建完整的目录结构"""
        self.subdirs = {
            'cross_validation': {
                'fold_results': os.path.join(self.output_dir, 'cross_validation/fold_results'),
                'metrics': os.path.join(self.output_dir, 'cross_validation/metrics'),
                'analysis': os.path.join(self.output_dir, 'cross_validation/analysis'),
            },
            'logs': os.path.join(self.output_dir, 'logs')
        }
        
        # 创建所有子目录
        for main_dir in self.subdirs.values():
            if isinstance(main_dir, dict):
                for sub_dir in main_dir.values():
                    os.makedirs(sub_dir, exist_ok=True)
            else:
                os.makedirs(main_dir, exist_ok=True)
    
    def save_cv_results(self, cv_results, fold=None):
        """保存交叉验证结果"""
        prefix = f'fold_{fold}_' if fold is not None else ''
        
        # 保存每个fold的详细结果
        for fold_idx, fold_data in enumerate(cv_results['fold_train_history']):
            fold_dir = os.path.join(self.subdirs['cross_validation']['fold_results'], 
                                  f'fold_{fold_idx}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # 保存训练历史
            np.save(os.path.join(fold_dir, f'{prefix}training_history.npy'), {
                'train_losses': fold_data['train_losses'],
                'val_losses': fold_data['val_losses'],
                'train_att_losses': fold_data['train_att_losses'],
                'train_age_losses': fold_data['train_age_losses'],
                'learning_rates': fold_data['learning_rates']
            })
        
        # 保存整体指标
        metrics_data = {
            'fold_metrics': cv_results['fold_metrics'],
            'overall_metrics': {
                # Attention Problem Scores 指标
                'mean_r2_sum_att': np.mean([fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]),
                'std_r2_sum_att': np.std([fold['sum_att']['R^2'] for fold in cv_results['fold_metrics']]),
                'mean_mse_sum_att': np.mean([fold['sum_att']['MSE'] for fold in cv_results['fold_metrics']]),
                'std_mse_sum_att': np.std([fold['sum_att']['MSE'] for fold in cv_results['fold_metrics']]),
                'mean_mae_sum_att': np.mean([fold['sum_att']['MAE'] for fold in cv_results['fold_metrics']]),
                'std_mae_sum_att': np.std([fold['sum_att']['MAE'] for fold in cv_results['fold_metrics']]),
                'mean_rmse_sum_att': np.mean([fold['sum_att']['RMSE'] for fold in cv_results['fold_metrics']]),
                'std_rmse_sum_att': np.std([fold['sum_att']['RMSE'] for fold in cv_results['fold_metrics']]),
                'mean_pearson_sum_att': np.mean([fold['sum_att']['Pearson'] for fold in cv_results['fold_metrics']]),
                'std_pearson_sum_att': np.std([fold['sum_att']['Pearson'] for fold in cv_results['fold_metrics']]),

                # Age 指标
                'mean_r2_age': np.mean([fold['age']['R^2'] for fold in cv_results['fold_metrics']]),
                'std_r2_age': np.std([fold['age']['R^2'] for fold in cv_results['fold_metrics']]),
                'mean_mse_age': np.mean([fold['age']['MSE'] for fold in cv_results['fold_metrics']]),
                'std_mse_age': np.std([fold['age']['MSE'] for fold in cv_results['fold_metrics']]),
                'mean_mae_age': np.mean([fold['age']['MAE'] for fold in cv_results['fold_metrics']]),
                'std_mae_age': np.std([fold['age']['MAE'] for fold in cv_results['fold_metrics']]),
                'mean_rmse_age': np.mean([fold['age']['RMSE'] for fold in cv_results['fold_metrics']]),
                'std_rmse_age': np.std([fold['age']['RMSE'] for fold in cv_results['fold_metrics']]),
                'mean_pearson_age': np.mean([fold['age']['Pearson'] for fold in cv_results['fold_metrics']]),
                'std_pearson_age': np.std([fold['age']['Pearson'] for fold in cv_results['fold_metrics']])
            }
            }

        np.save(os.path.join(self.subdirs['cross_validation']['metrics'], 
                            f'{prefix}all_metrics.npy'), metrics_data)
        
        # 保存预测结果
        predictions_data = {
            'predictions': cv_results['predictions'],
            'targets': cv_results['targets']
        }
        np.save(os.path.join(self.subdirs['cross_validation']['analysis'], 
                            f'{prefix}predictions.npy'), predictions_data)

    def save_log(self, message):
        """保存日志信息"""
        log_file = os.path.join(self.subdirs['logs'], 'training.log')
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'[{timestamp}] {message}\n')

    def get_subdirs(self):
        """返回所有子目录路径"""
        return self.subdirs
    
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

def extract_vertex_features(subjects, vertex_id, vertex_ids):
    """
    为给定的subjects提取指定vertex的特征
    
    Args:
        subjects: 样本数据
        vertex_id: 选定的vertex ID
        vertex_ids: 所有vertex IDs
    
    Returns:
        numpy.ndarray: 提取的特征数组
    """
    features = []
    for subject in subjects:
        vertex_features = []
        feature_files = {
            'thickness': f'{FEATURE_PREFIX}thickness.fwhm10.fsaverage.mgh',
            'volume': f'{FEATURE_PREFIX}volume.fwhm10.fsaverage.mgh',
            'area': f'{FEATURE_PREFIX}area.fwhm10.fsaverage.mgh',
            'wg_ratio': f'{FEATURE_PREFIX}w_g.pct.mgh.fwhm10.fsaverage.mgh'
        }
        
        for file_name in feature_files.values():
            file_path = subject[2][file_name]
            feature_data = nib.load(file_path).get_fdata()
            value = feature_data[vertex_id][0][0]
            vertex_features.append(value)
            
        features.append(vertex_features)
        
    return np.array(features)

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

def select_best_vertex(sample_subjects, vertex_ids, phenotype_data, n_vertices=1000):
    """
    选择最佳vertex并提取其特征
    
    Args:
        sample_subjects: 样本数据
        vertex_ids: 所有vertex IDs
        phenotype_data: 表型数据
        n_vertices: 随机采样的vertex数量
    
    Returns:
        tuple: (best_vertex_id, best_vertex_features)
    """
    best_vertex = None
    best_score = -np.inf
    best_features = None
    
    # 随机采样vertices
    sample_vertices = np.random.choice(vertex_ids, size=n_vertices, replace=False)
    
    # 目标变量
    targets = np.array([
        phenotype_data[:, 0],  # Attention Problem_Scores
        phenotype_data[:, 2]   # age
    ]).T
    
    # 额外特征
    extra_features = np.array([
        phenotype_data[:, 1],  # aggressive_behaviour_Scores
        phenotype_data[:, 3],  # sex
        phenotype_data[:, 4]   # maternal_edu_level
    ]).T
    
    for vertex_id in sample_vertices:
        # 提取该vertex的特征
        vertex_features = extract_vertex_features(sample_subjects, vertex_id, vertex_ids)
        
        if vertex_features is None:
            continue
            
        # 计算该vertex的得分、
        score, _ = calculate_vertex_Scores(
            vertex_features, 
            targets, 
            extra_features, 
            n_vertices=len(sample_vertices)
        )
        
        if score > best_score:
            best_score = score
            best_vertex = vertex_id
            best_features = vertex_features
            
    return best_vertex, best_features

# def train_ridge_model(vertex_features, phenotype_data):
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

def perform_cross_validation(sample_subjects, vertex_ids, phenotype_data, results_manager, config=None):
    """
    执行5折交叉验证
    
    Args:
        sample_subjects: 样本数据
        vertex_ids: 顶点ID
        phenotype_data: 表型数据
        results_manager: 结果管理器
        config: 配置参数
    """
    try:
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 初始化存储所有折结果的字典
        all_cv_results = {
            'fold_metrics': [],
            'fold_train_history': [],
            'predictions': [],
            'targets': [],
            'selected_vertices': []  # 记录每折选择的vertex
        }
        # 对每一折进行训练和评估
        for fold, (train_idx, val_idx) in enumerate(kf.split(sample_subjects)):
            logging.info(f"\nProcessing fold {fold + 1}/{n_splits}")
            
            # 准备当前折的数据
            train_subjects = sample_subjects[train_idx]
            val_subjects = sample_subjects[val_idx]
            train_phenotype = phenotype_data[train_idx]
            val_phenotype = phenotype_data[val_idx]
            
            # 选择最佳vertex（使用训练集）
            best_vertex, best_features = select_best_vertex(train_subjects, vertex_ids, train_phenotype)
            if best_vertex is None:
                raise ValueError(f"No suitable vertex found for fold {fold + 1}")
                
            logging.info(f"Selected best vertex {best_vertex} for fold {fold + 1}")
            
            # 为验证集提取特征
            val_features = extract_vertex_features(
                val_subjects, 
                best_vertex,
                vertex_ids
            )
            
            # 训练模型
            models, predictions, metrics, train_history = train_ridge_model(
                best_features, 
                train_phenotype,
                val_features,
                val_phenotype
            )
            
            # 创建当前折的结果字典
            fold_results = {
                'fold_metrics': [metrics],
                'fold_train_history': [train_history],
                'predictions': [predictions],
                'targets': [val_phenotype[:, [0, 2]]],
                'selected_vertex': best_vertex
            }
            
            # 保存当前折的结果
            results_manager.save_cv_results(fold_results, fold)
            
            # 将当前折结果添加到总结果中
            all_cv_results['fold_metrics'].append(metrics)
            all_cv_results['fold_train_history'].append(train_history)
            all_cv_results['predictions'].append(predictions)
            all_cv_results['targets'].append(val_phenotype[:, [0, 2]])
            all_cv_results['selected_vertices'].append(best_vertex)
            
        return all_cv_results
        
    except Exception as e:
        logging.error(f"Error in cross-validation: {str(e)}")
        raise


def train_ridge_model(train_features, train_phenotype, val_features, val_phenotype):
    """训练岭回归模型（单折）"""
    logging.info("\nStarting Ridge Regression model training...")
    
    # 准备特征和目标
    X_train = np.hstack([
        train_features,
        train_phenotype[:, [1, 3, 4]]  # extra features
    ])
    y_train = train_phenotype[:, [0, 2]]  # Attention Problem Scores and age
    
    X_val = np.hstack([
        val_features,
        val_phenotype[:, [1, 3, 4]]
    ])
    y_val = val_phenotype[:, [0, 2]]
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 训练历史记录
    train_history = {
        'train_losses': [],
        'val_losses': [],
        'train_att_losses': [],
        'train_age_losses': [],
        'learning_rates': []
    }
    
    models = []
    predictions = []
    metrics = {'sum_att': {},  # 修改这里，使用 'sum_att' 而不是 'attention_problem_scores'
        'age': {}}
    
    target_names = ['sum_att', 'age']
    
    for i in range(y_train.shape[1]):
        logging.info(f"\nTraining model for {target_names[i]}...")
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train[:, i])
        models.append(model)
        
        # 预测和评估
        y_pred = model.predict(X_val_scaled)
        predictions.append(y_pred)
        
        # 计算评估指标
        target_metrics = calculate_metrics(y_val[:, i], y_pred)
        metrics[target_names[i].lower().replace(' ', '_')] = target_metrics
        
        logging.info(f"\nMetrics for {target_names[i]}:")
        for metric_name, value in target_metrics.items():
            logging.info(f"{metric_name}: {value:.4f}")
    
    return models, predictions, metrics, train_history


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    
    return {
        'R^2': r2,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Pearson': pearson_corr
    }

def print_average_metrics(cv_results):
    """打印平均评估指标"""
    logging.info("\nCross-validation Results:")
    
    target_names = ['sum_att', 'age']
    metric_names = ['R^2', 'MSE', 'RMSE', 'MAE', 'Pearson']
    
    for target in target_names:
        print(f"\n{target.upper()} Metrics:")
        for metric in metric_names:
            values = [fold[target][metric] for fold in cv_results['fold_metrics']]
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")

def main():
    try:
        # 配置日志
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'cv_baselin3_{HEMISPHERE}_results_{timestamp}'
        # 先创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"\nStarting baseline model cross-validation analysis")
        
        # 创建结果管理器
        results_manager = ResultsManager(output_dir)
    
        
        # 加载数据
        logging.info("\nLoading data files...")
        try:
            # 加载样本subjects数据
            sample_subjects = np.load('sample_ids_filename_updated.npy', allow_pickle=True)
            vertex_ids = np.load(f'ij_id_{HEMISPHERE}.npy', allow_pickle=True).astype(int)
            phenotype_data = np.load('sample_normalised_phenotype.npy', allow_pickle=True)
            
            # 验证数据一致性
            if len(sample_subjects) != len(phenotype_data):
                raise ValueError(f"Mismatch in data lengths: {len(sample_subjects)} subjects but {len(phenotype_data)} phenotype records")
            
            logging.info(f"Successfully loaded data: {len(sample_subjects)} subjects, {len(vertex_ids)} vertices")
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
            
        # 配置参数
        config = {
            'n_vertices': 1000,  # 随机采样的vertex数量
            'n_splits': 5,       # 交叉验证折数
            'random_state': 42   # 随机种子
        }
        
        # 保存配置信息
        config_info = {
            'timestamp': timestamp,
            'config': config,
            'data_info': {
                'n_subjects': len(sample_subjects),
                'n_vertices': len(vertex_ids),
                'n_phenotype_features': phenotype_data.shape[1]
            }
        }
        np.save(os.path.join(output_dir, 'config_info.npy'), config_info)
        
        # 执行交叉验证
        cv_results = perform_cross_validation(
            sample_subjects=sample_subjects,
            vertex_ids=vertex_ids,
            phenotype_data=phenotype_data,
            results_manager=results_manager,
            config=config
        )
        
        # 打印最终结果
        logging.info("\nCross-validation completed. Final Results:")
        
        target_names = ['Attention Problems', 'Age']
        metric_names = ['R^2', 'MSE', 'RMSE', 'MAE', 'Pearson']
        
        for i, target in enumerate(target_names):
            logging.info(f"\nResults for {target}:")
            
            # 收集所有折的指标
            fold_metrics = {metric: [] for metric in metric_names}
            for fold_result in cv_results['fold_metrics']:
                target_key = 'sum_att' if i == 0 else 'age'
                for metric in metric_names:
                    fold_metrics[metric].append(fold_result[target_key][metric])
            
            # 计算并打印平均值和标准差
            for metric in metric_names:
                values = fold_metrics[metric]
                mean_value = np.mean(values)
                std_value = np.std(values)
                logging.info(f"{metric}: {mean_value:.4f} ± {std_value:.4f}")
        
        # 保存最终结果
        final_results = {
            'cv_results': cv_results,
            'config': config,
            'timestamp': timestamp
        }
        np.save(os.path.join(output_dir, 'final_results.npy'), final_results)
        
        logging.info("\nAll results have been saved successfully.")
        
        # 清理内存
        del cv_results, sample_subjects, vertex_ids, phenotype_data
        gc.collect()
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()