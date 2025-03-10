import numpy as np
import scipy.stats as stats
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import nibabel as nib
import logging
from datetime import datetime

# Configuration
HEMISPHERE = 'rh'
FEATURE_NAMES = ['thickness', 'volume', 'area', 'wg_ratio']
RANDOM_STATE = 42
# SAMPLE_SIZE = 1000  # 可调整的样本量

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'baselines_{HEMISPHERE}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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

def extract_all_vertex_features_batch(sample_subjects, vertex_ids):
    """使用批处理方式提取所有subjects的所有vertex特征"""
    n_subjects = len(sample_subjects)
    n_vertices = len(vertex_ids)
    all_features = np.zeros((n_subjects, n_vertices, 4))
    
    feature_files = {
        'thickness': f'{HEMISPHERE}.thickness.fwhm10.fsaverage.mgh',
        'volume': f'{HEMISPHERE}.volume.fwhm10.fsaverage.mgh',
        'area': f'{HEMISPHERE}.area.fwhm10.fsaverage.mgh',
        'wg_ratio': f'{HEMISPHERE}.w_g.pct.mgh.fwhm10.fsaverage.mgh'
    }

    logging.info(f"\n开始批量提取特征，共 {n_subjects} 个受试者，{n_vertices} 个顶点")
    
    # 对每个受试者
    for i, subject in enumerate(sample_subjects):

        logging.info(f"Processing subject {i+1}/{n_subjects}")
            
        # 一次性读取该受试者的所有特征文件
        feature_data = {}
        try:
            for feature_name, file_name in feature_files.items():
                file_path = subject[2][file_name]
                # 一次性读取整个文件的数据
                feature_data[feature_name] = nib.load(file_path).get_fdata()
                
            # 批量提取所有顶点的特征
            for j, vertex_id in enumerate(vertex_ids):
                # 直接从内存中的数据提取特征，不需要重复读取文件
                all_features[i, j, 0] = feature_data['thickness'][vertex_id][0][0]
                all_features[i, j, 1] = feature_data['volume'][vertex_id][0][0]
                all_features[i, j, 2] = feature_data['area'][vertex_id][0][0]
                all_features[i, j, 3] = feature_data['wg_ratio'][vertex_id][0][0]
                
        except Exception as e:
            logging.error(f"Error processing subject {i}: {str(e)}")
            continue

    return all_features

def evaluate_predictions(y_true, y_pred, target_name):
    """评估预测结果"""
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'corr': stats.pearsonr(y_true, y_pred)[0]
    }
    
    logging.info(f"\nResults for {target_name}:")
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")
    
    return metrics

class BaselineModels:
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    # def prepare_data_subset(self, sample_subjects, vertex_ids, phenotype_data, n_samples=SAMPLE_SIZE):
    #     """准备子数据集"""
    #     if len(sample_subjects) > n_samples:
    #         indices = np.random.choice(len(sample_subjects), n_samples, replace=False)
    #         return (sample_subjects[indices], vertex_ids, 
    #                phenotype_data[indices])
    #     return sample_subjects, vertex_ids, phenotype_data

    def univariate_baseline(self, vertex_features, phenotype_data, best_v, best_f):
        """Baseline 1: 单顶点Ridge回归"""
        logging.info("\nRunning Univariate Ridge Regression Baseline...")
        
        # 只使用选中的vertex特征
        X = vertex_features[:, best_v, best_f].reshape(-1, 1)
        y = phenotype_data[:, [0, 2]]  # ADHD评分 & 年龄
        
        results = {}
        for target_idx, target_name in enumerate(['ADHD', 'Age']):
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y[:, target_idx], test_size=0.2, random_state=self.random_state
            )
            
            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 训练模型
            model = Ridge(alpha=1.0, random_state=self.random_state)
            model.fit(X_train_scaled, y_train)
            
            # 预测和评估
            y_pred = model.predict(X_test_scaled)
            results[target_name] = evaluate_predictions(y_test, y_pred, target_name)
        
        return results

    def phenotype_baseline(self, phenotype_data):
        """Baseline 2: 使用所有表型数据的Random Forest"""
        logging.info("\nRunning Phenotype-based Random Forest Baseline...")
        
        # 使用其他表型特征预测ADHD评分和年龄
        # phenotype_data格式: [ADHD评分, 攻击行为分数, 年龄, 性别, 母亲教育水平]
        results = {}
        
        for target_idx, target_name in enumerate(['ADHD', 'Age']):
            # 准备特征：除了目标变量外的所有表型特征
            feature_indices = [i for i in range(phenotype_data.shape[1]) 
                             if i not in [0, 2]]  # 排除ADHD评分和年龄
            X = phenotype_data[:, feature_indices]
            y = phenotype_data[:, [0, 2][target_idx]]
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # 训练Random Forest模型
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state
            )
            model.fit(X_train_scaled, y_train)
            
            # 预测和评估
            y_pred = model.predict(X_test_scaled)
            results[target_name] = evaluate_predictions(y_test, y_pred, target_name)
            
            # 特征重要性
            feature_names = ['Aggressive_Behavior', 'Sex', 'Maternal_Education']
            importances = model.feature_importances_
            logging.info(f"\nFeature importances for {target_name}:")
            for name, imp in zip(feature_names, importances):
                logging.info(f"{name}: {imp:.4f}")
        
        return results
    def select_best_vertex(self, vertex_features, phenotype_data):
        """
        选择最佳vertex和feature
        vertex_features shape: (n_subjects, n_vertices, n_features)
        phenotype_data shape: (n_subjects, n_phenotypes)
        Returns: (best_vertex_idx, best_feature_idx)
        """
        logging.info("\nSelecting best vertex...")
        n_subjects, n_vertices, n_features = vertex_features.shape
        
        # 矩阵存储每个vertex和feature的相关系数
        best_corr = -np.inf
        best_v = 0
        best_f = 0
        
        targets = phenotype_data[:, [0, 2]]  # attention problems score and age
        n_tests = n_vertices * n_features * 2  # 总测试次数（每个顶点的每个特征对两个目标变量）
        bonferroni_alpha = 0.05 / n_tests  # Bonferroni校正
        
        for v in range(n_vertices):
            if v % 1000 == 0:  # 每处理1000个vertex输出一次进度
                logging.info(f"Processing vertex {v}/{n_vertices}")
                
            for f in range(n_features):
                feature_data = vertex_features[:, v, f]
                
                # 计算与两个目标变量的相关系数
                correlations = []
                for target_idx in range(2):
                    corr, p_value = stats.pearsonr(feature_data, targets[:, target_idx])
                    # 使用Bonferroni校正后的p值判断显著性
                    if p_value < bonferroni_alpha:
                        correlations.append(abs(corr))
                    else:
                        correlations.append(0)
                
                # 使用平均相关系数作为评价标准
                mean_corr = np.mean(correlations)
                if mean_corr > best_corr:
                    best_corr = mean_corr
                    best_v = v
                    best_f = f
                    logging.info(f"\nFound new best vertex {v} feature {f} with correlation {mean_corr:.4f}")
        
        logging.info(f"\nSelected best vertex {best_v} with feature {FEATURE_NAMES[best_f]}")
        logging.info(f"Best correlation: {best_corr:.4f}")
        
        return best_v, best_f

def main():
    try:
        # 1. 加载数据
        sample_subjects, vertex_ids, phenotype_data = load_data()
        
        # 2. 创建baseline模型实例
        baselines = BaselineModels()

        
        # 4. 提取特征（使用批处理方法）
        vertex_features = extract_all_vertex_features_batch(sample_subjects, vertex_ids)
        
        
        # 5. 为univariate baseline找到最佳vertex
        # 4. 为univariate baseline找到最佳vertex
        best_v, best_f = baselines.select_best_vertex(vertex_features, phenotype_data)
        
        # 6. 运行两个baseline
        univariate_results = baselines.univariate_baseline(
            vertex_features, phenotype_data, best_v, best_f
        )
        phenotype_results = baselines.phenotype_baseline(phenotype_data)
        
        # 7. 输出总结
        logging.info("\n=== Final Results Summary ===")
        logging.info("\nUnivariate Baseline Results:")
        for target, metrics in univariate_results.items():
            logging.info(f"\n{target}:")
            for metric_name, value in metrics.items():
                logging.info(f"{metric_name}: {value:.4f}")
        
        logging.info("\nPhenotype Baseline Results:")
        for target, metrics in phenotype_results.items():
            logging.info(f"\n{target}:")
            for metric_name, value in metrics.items():
                logging.info(f"{metric_name}: {value:.4f}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()