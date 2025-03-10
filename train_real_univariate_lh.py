import numpy as np
import scipy.stats as stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import nibabel as nib
import logging
from datetime import datetime

# ; # Configuration
# ; HEMISPHERE = 'lh'  # 可以改成 'lh' 用于左脑
# ; FEATURE_PREFIX = f"{HEMISPHERE}."  # 自动生成正确的文件前缀


# # 配置日志
# ; logging.basicConfig(
# ;     level=logging.INFO,
# ;     format='%(asctime)s - %(levelname)s - %(message)s',
# ;     handlers=[
# ;         logging.FileHandler(f'real_univariate_model_{HEMISPHERE}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
# ;         logging.StreamHandler()
# ;     ]
# ; )

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


# # === 1. 数据加载 ===
# def extract_vertex_features(subject_info, vertex_id):
#     """提取单个vertex的所有特征"""
#     features = []
#     try:
#         feature_files = {
#             'thickness': f'{HEMISPHERE}.thickness.fwhm10.fsaverage.mgh',
#             'volume': f'{HEMISPHERE}.volume.fwhm10.fsaverage.mgh',
#             'area': f'{HEMISPHERE}.area.fwhm10.fsaverage.mgh',
#             'wg_ratio': f'{HEMISPHERE}.w_g.pct.mgh.fwhm10.fsaverage.mgh'
#         }
        
#         for feature_name, file_name in feature_files.items():
#             file_path = subject_info[2][file_name]
#             feature_data = nib.load(file_path).get_fdata()
#             value = feature_data[vertex_id][0][0]  # 取出该vertex对应的值
#             features.append(value)
            
#         return np.array(features)
#     except Exception as e:
#         logging.error(f"Error extracting features for vertex {vertex_id}: {str(e)}")
#         return None

# # === 2. 遍历所有 vertex 并计算相关性 ===
# def find_best_vertex_univariate(sample_subjects, vertex_ids, phenotype_data):
#     """计算所有 vertex 的相关性，选出最相关的 vertex"""
#     targets = np.array([phenotype_data[:, 0], phenotype_data[:, 2]]).T  # ADHD评分 & 年龄
#     best_vertex, best_correlation, best_features = None, -1, None
    
#     p_values = []
#     correlations = []
#     count = 0
#     logging.info(f"开始遍历所有vertices...")
#     for vertex_id in vertex_ids:
#         count += 1
#         if count % 100 == 0:
#             logging.info(f"遍历进度: {count/len(vertex_ids)}")
#         vertex_features = []
#         for subject in sample_subjects:
#             features = extract_vertex_features(subject, vertex_id)
#             if features is not None:
#                 vertex_features.append(features)
                
#         if len(vertex_features) == 0:
#             continue
            
#         vertex_features = np.array(vertex_features)
        
#         # 计算相关性
#         r, p = stats.pearsonr(vertex_features[:, 0], targets[:, 0])  # ADHD 评分
#         correlations.append(abs(r))
#         p_values.append(p)

#     # Bonferroni 校正
#     bonferroni_alpha = 0.05 / len(vertex_ids)
#     significant_indices = np.where(np.array(p_values) < bonferroni_alpha)[0]

#     if len(significant_indices) == 0:
#         logging.info("没有显著相关的 vertex，选择相关性最大的")
#         best_vertex = vertex_ids[np.argmax(correlations)]
#     else:
#         best_vertex = vertex_ids[significant_indices[np.argmax(np.array(correlations)[significant_indices])]]
    
#     logging.info(f"选出的最佳 vertex: {best_vertex}")
    
#     # 提取最佳 vertex 的特征
#     best_features = []
#     for subject in sample_subjects:
#         features = extract_vertex_features(subject, best_vertex)
#         if features is not None:
#             best_features.append(features)
    
#     return best_vertex, np.array(best_features)

# # === 3. 使用最佳 vertex 训练 Ridge Regression ===
# def train_ridge_model(vertex_features, phenotype_data):
#     """使用 Ridge Regression 训练模型"""
#     logging.info("\nStarting Ridge Regression model training...")

#     X = np.hstack([
#         vertex_features,
#         phenotype_data[:, [1, 3, 4]]  # 额外的表型特征（行为分数、性别、母亲教育水平）
#     ])
#     y = phenotype_data[:, [0, 2]]  # ADHD 评分 & 年龄

#     # 训练 & 测试集划分
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # 数据标准化
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # 训练 Ridge Regression
#     ridge_model = Ridge(alpha=1.0)
#     ridge_model.fit(X_train_scaled, y_train)

#     # 预测
#     y_pred = ridge_model.predict(X_test_scaled)

#     # 计算评估指标
#     r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
#     mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
#     mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

#     logging.info("\nUnivariate Test Ridge Regression 结果:")
#     logging.info(f"R² (注意力分数): {r2_scores[0]:.4f}, R² (年龄): {r2_scores[1]:.4f}")
#     logging.info(f"MSE (注意力分数): {mse[0]:.4f}, MSE (年龄): {mse[1]:.4f}")
#     logging.info(f"MAE (注意力分数): {mae[0]:.4f}, MAE (年龄): {mae[1]:.4f}")

#     return ridge_model

# # === 4. 运行 Univariate Test Baseline ===
# def run_univariate_test(sample_subjects, vertex_ids, phenotype_data):
#     best_vertex, best_features = find_best_vertex_univariate(sample_subjects, vertex_ids, phenotype_data)
#     ridge_model = train_ridge_model(best_features, phenotype_data)
#     return ridge_model

# # 运行代码
# if __name__ == "__main__":
#     sample_subjects, vertex_ids, phenotype_data = load_data()  # 这里的 load_data 需要使用你的数据加载代码
#     run_univariate_test(sample_subjects, vertex_ids, phenotype_data)

# Configuration
HEMISPHERE = 'lh'
FEATURE_PREFIX = f"{HEMISPHERE}."
FEATURE_NAMES = ['thickness', 'volume', 'area', 'wg_ratio']

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'real_univariate_model_{HEMISPHERE}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# 保持你现有的 validate_data 和 load_data 函数不变

def extract_vertex_features(subject_info, vertex_id):
    """提取单个vertex的所有特征"""
    features = []
    try:
        feature_files = {
            'thickness': f'{HEMISPHERE}.thickness.fwhm10.fsaverage.mgh',
            'volume': f'{HEMISPHERE}.volume.fwhm10.fsaverage.mgh',
            'area': f'{HEMISPHERE}.area.fwhm10.fsaverage.mgh',
            'wg_ratio': f'{HEMISPHERE}.w_g.pct.mgh.fwhm10.fsaverage.mgh'
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

def extract_all_vertex_features(sample_subjects, vertex_ids):
    """遍历所有subjects，提取所有vertex的特征"""
    n_subjects = len(sample_subjects)
    n_vertices = len(vertex_ids)
    all_features = np.zeros((n_subjects, n_vertices, 4))

    logging.info(f"\n开始提取特征，共 {n_subjects} 个受试者，{n_vertices} 个顶点")
    for i, subject in enumerate(sample_subjects):
        logging.info(f"Processing subject {i+1}/{n_subjects}")
        for j, vertex_id in enumerate(vertex_ids):
            features = extract_vertex_features(subject, vertex_id)
            if features is not None:
                all_features[i, j, :] = features

    return all_features

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

def compute_correlations(vertex_features, y):
    """计算相关性"""
    n_subjects, n_vertices, n_features = vertex_features.shape
    correlations = np.zeros((n_vertices, n_features))
    p_values = np.zeros((n_vertices, n_features))

    logging.info(f"\n开始计算相关性，共 {n_vertices} 个顶点，{n_features} 个特征")
    
    for f in range(n_features):
        logging.info(f"\n处理特征 {FEATURE_NAMES[f]} ({f+1}/{n_features})")
        for v in range(n_vertices):
            if v % 1000 == 0:
                logging.info(f"Progress: {v}/{n_vertices} vertices processed ({(v/n_vertices*100):.1f}%)")
            r, p = stats.pearsonr(vertex_features[:, v, f], y[:, 0])
            correlations[v, f] = abs(r)
            p_values[v, f] = p
            
    return correlations, p_values

def train_model(vertex_features, phenotype_data, best_v, best_f):
    """训练Ridge回归模型"""
    X = vertex_features[:, best_v, best_f].reshape(-1, 1)
    X = np.hstack([X, phenotype_data[:, [1, 3, 4]]])
    y = phenotype_data[:, [0, 2]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)

    y_pred = ridge_model.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred)

def evaluate_model(y_test, y_pred):
    """评估模型性能"""
    r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

    logging.info("\n模型评估结果:")
    logging.info(f"R² (注意力分数): {r2_scores[0]:.4f}, R² (年龄): {r2_scores[1]:.4f}")
    logging.info(f"MSE (注意力分数): {mse[0]:.4f}, MSE (年龄): {mse[1]:.4f}")
    logging.info(f"MAE (注意力分数): {mae[0]:.4f}, MAE (年龄): {mae[1]:.4f}")

    return r2_scores, mse, mae

def main():
    try:
        # 1. 加载数据
        sample_subjects, vertex_ids, phenotype_data = load_data()
        
        # 2. 提取特征
        vertex_features = extract_all_vertex_features_batch(sample_subjects, vertex_ids)
        y = phenotype_data[:, [0, 2]]
        
        # 3. 计算相关性
        correlations, p_values = compute_correlations(vertex_features, y)
        
        # 4. 找出最佳组合
        bonferroni_alpha = 0.05 / (correlations.shape[0] * correlations.shape[1])
        significant_mask = p_values < bonferroni_alpha
        
        if np.any(significant_mask):
            best_v, best_f = np.unravel_index(np.argmax(correlations * significant_mask), correlations.shape)
        else:
            best_v, best_f = np.unravel_index(np.argmax(correlations), correlations.shape)
        
        logging.info(f"\n最佳组合:")
        logging.info(f"Vertex ID: {vertex_ids[best_v]}")
        logging.info(f"特征: {FEATURE_NAMES[best_f]}")
        logging.info(f"相关性系数: {correlations[best_v, best_f]:.4f}")
        
        # 5. 训练和评估模型
        train_model(vertex_features, phenotype_data, best_v, best_f)
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

