import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class ImprovedBaselineModel:
    def __init__(self):
        self.scaler_features = RobustScaler()
        self.scaler_targets = StandardScaler()
        self.pca = PCA(n_components=50)  # 降维到50个主成分
        self.model_att = XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )
        self.model_agg = XGBRegressor(
            objective='reg:squarederror',
            random_state=42
        )
        
    def extract_features(self, images):
        """增强的特征提取"""
        batch_size = images.shape[0]
        features_list = []
        
        # 1. 基本统计特征
        for i in range(4):  # 对每个通道
            channel = images[:, i, :, :]
            
            # 统计特征
            features = np.vstack([
                channel.mean(axis=(1, 2)),  # 平均值
                channel.std(axis=(1, 2)),   # 标准差
                channel.max(axis=(1, 2)),   # 最大值
                channel.min(axis=(1, 2)),   # 最小值
                np.median(channel, axis=(1, 2)),  # 中位数
                np.percentile(channel, 25, axis=(1, 2)),  # 第一四分位数
                np.percentile(channel, 75, axis=(1, 2)),  # 第三四分位数
                np.percentile(channel, 10, axis=(1, 2)),  # 10th percentile
                np.percentile(channel, 90, axis=(1, 2)),  # 90th percentile
                np.sum(channel > channel.mean(axis=(1, 2))[:, None, None], axis=(1, 2)) / (512*512),  # 高于均值的比例
            ]).T
            
            features_list.append(features)
            
            # 2. 区域统计
            # 将图像分成4x4的区块，计算每个区块的均值和标准差
            block_size = 128  # 512/4
            for row in range(4):
                for col in range(4):
                    block = channel[:, row*block_size:(row+1)*block_size, 
                                    col*block_size:(col+1)*block_size]
                    block_features = np.vstack([
                        block.mean(axis=(1, 2)),
                        block.std(axis=(1, 2))
                    ]).T
                    features_list.append(block_features)
        
        # 3. 合并所有特征
        combined_features = np.hstack(features_list)
        
        # 4. 应用PCA降维
        reduced_features = self.pca.fit_transform(combined_features)
        
        return reduced_features
    
    def fit(self, brain_images, phenotypes, targets):
        """训练模型"""
        # 特征提取和预处理
        brain_features = self.extract_features(brain_images)
        X = np.hstack([brain_features, phenotypes])
        X = self.scaler_features.fit_transform(X)
        y = self.scaler_targets.fit_transform(targets)
        
        # 使用网格搜索调优超参数
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6],
            'n_estimators': [50, 100, 150, 200],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
            'min_child_weight': [1, 2, 3]
        }
        
        grid_search = GridSearchCV(estimator=self.model_att, param_grid=param_grid, 
                                   scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X, y[:, 0])
        print(f"Best parameters for model_att: {grid_search.best_params_}")
        self.model_att = grid_search.best_estimator_
        
        grid_search.fit(X, y[:, 1])
        print(f"Best parameters for model_agg: {grid_search.best_params_}")
        self.model_agg = grid_search.best_estimator_

        return self
    
    def predict(self, brain_images, phenotypes):
        """预测"""
        brain_features = self.extract_features(brain_images)
        X = np.hstack([brain_features, phenotypes])
        X = self.scaler_features.transform(X)
        
        # 预测并转换回原始比例
        predictions = np.column_stack([
            self.model_att.predict(X),
            self.model_agg.predict(X)
        ])
        return self.scaler_targets.inverse_transform(predictions)
    
    def evaluate(self, brain_images, phenotypes, targets):
        """评估模型并返回特征重要性"""
        predictions = self.predict(brain_images, phenotypes)
        results = {}
        
        target_names = ['sum_att', 'sum_agg']
        for i, name in enumerate(target_names):
            mse = mean_squared_error(targets[:, i], predictions[:, i])
            r2 = r2_score(targets[:, i], predictions[:, i])
            results[name] = {'MSE': mse, 'R2': r2}
            
            # 绘制散点图
            plt.figure(figsize=(10, 5))
            plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
            plt.xlabel(f'True {name}')
            plt.ylabel(f'Predicted {name}')
            plt.title(f'Predictions vs True Values for {name}')
            
            # 添加对角线
            min_val = min(targets[:, i].min(), predictions[:, i].min())
            max_val = max(targets[:, i].max(), predictions[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
            
            # 添加统计信息
            plt.text(0.05, 0.95, 
                    f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            plt.legend()
            plt.grid(True)
            plt.savefig(f'test_baseline_{name}_predictions_samples.png')
            plt.close()
        
        return results

def main():
    # 加载数据
    image_data = np.load('sample_cnn_lh_brainimages.npy')
    phenotype_data = np.load('sample_phenotype_tensor_normalized.npy')
    
    # 分割数据
    train_images, test_images, train_phenotypes, test_phenotypes = train_test_split(
        image_data, phenotype_data, test_size=0.2, random_state=42)
    
    # 获取目标变量（前两列）和表型特征（后三列）
    train_targets = train_phenotypes[:, :2]
    test_targets = test_phenotypes[:, :2]
    train_phenotypes = train_phenotypes[:, 2:5]
    test_phenotypes = test_phenotypes[:, 2:5]
    
    # 训练和评估改进后的模型
    model = ImprovedBaselineModel()
    model.fit(train_images, train_phenotypes, train_targets)
    results = model.evaluate(test_images, test_phenotypes, test_targets)
    
    print("\nImproved Model Results:")
    for target, metrics in results.items():
        print(f"\n{target}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()

"""
Job ID: 8474328
"""

# import numpy as np
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # 
# class ImprovedBaselineModel:
#     def __init__(self):
#         self.scaler_features = RobustScaler()
#         self.scaler_targets = StandardScaler()
#         self.pca = PCA(n_components=50)  # 降维到50个主成分
#         self.model_att = XGBRegressor(
#             objective='reg:squarederror',
#             n_estimators=100,
#             learning_rate=0.1,
#             max_depth=5,
#             min_child_weight=1,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )
#         self.model_agg = XGBRegressor(
#             objective='reg:squarederror',
#             n_estimators=100,
#             learning_rate=0.1,
#             max_depth=5,
#             min_child_weight=1,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )
        
#     def extract_features(self, images):
#         """增强的特征提取"""
#         batch_size = images.shape[0]
#         features_list = []
        
#         # 1. 基本统计特征
#         for i in range(4):  # 对每个通道
#             channel = images[:, i, :, :]
            
#             # 统计特征
#             features = np.vstack([
#                 channel.mean(axis=(1, 2)),  # 平均值
#                 channel.std(axis=(1, 2)),   # 标准差
#                 channel.max(axis=(1, 2)),   # 最大值
#                 channel.min(axis=(1, 2)),   # 最小值
#                 np.median(channel, axis=(1, 2)),  # 中位数
#                 np.percentile(channel, 25, axis=(1, 2)),  # 第一四分位数
#                 np.percentile(channel, 75, axis=(1, 2)),  # 第三四分位数
#                 np.percentile(channel, 10, axis=(1, 2)),  # 10th percentile
#                 np.percentile(channel, 90, axis=(1, 2)),  # 90th percentile
#                 np.sum(channel > channel.mean(axis=(1, 2))[:, None, None], axis=(1, 2)) / (512*512),  # 高于均值的比例
#             ]).T
            
#             features_list.append(features)
            
#             # 2. 区域统计
#             # 将图像分成4x4的区块，计算每个区块的均值和标准差
#             block_size = 128  # 512/4
#             for row in range(4):
#                 for col in range(4):
#                     block = channel[:, row*block_size:(row+1)*block_size, 
#                                     col*block_size:(col+1)*block_size]
#                     block_features = np.vstack([
#                         block.mean(axis=(1, 2)),
#                         block.std(axis=(1, 2))
#                     ]).T
#                     features_list.append(block_features)
        
#         # 3. 合并所有特征
#         combined_features = np.hstack(features_list)
        
#         # 4. 应用PCA降维
#         reduced_features = self.pca.fit_transform(combined_features)
        
#         return reduced_features
    
#     def fit(self, brain_images, phenotypes, targets):
#         """训练模型"""
#         # 特征提取和预处理
#         brain_features = self.extract_features(brain_images)
#         X = np.hstack([brain_features, phenotypes])
#         X = self.scaler_features.fit_transform(X)
#         y = self.scaler_targets.fit_transform(targets)
        
#         # 分别训练两个目标的模型
#         self.model_att.fit(X, y[:, 0])
#         self.model_agg.fit(X, y[:, 1])
        
#         return self
    
#     def predict(self, brain_images, phenotypes):
#         """预测"""
#         brain_features = self.extract_features(brain_images)
#         X = np.hstack([brain_features, phenotypes])
#         X = self.scaler_features.transform(X)
        
#         # 预测并转换回原始比例
#         predictions = np.column_stack([
#             self.model_att.predict(X),
#             self.model_agg.predict(X)
#         ])
#         return self.scaler_targets.inverse_transform(predictions)
    
#     def evaluate(self, brain_images, phenotypes, targets):
#         """评估模型并返回特征重要性"""
#         predictions = self.predict(brain_images, phenotypes)
#         results = {}
        
#         target_names = ['sum_att', 'sum_agg']
#         for i, name in enumerate(target_names):
#             mse = mean_squared_error(targets[:, i], predictions[:, i])
#             r2 = r2_score(targets[:, i], predictions[:, i])
#             results[name] = {'MSE': mse, 'R2': r2}
            
#             # 绘制散点图
#             plt.figure(figsize=(10, 5))
#             plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
#             plt.xlabel(f'True {name}')
#             plt.ylabel(f'Predicted {name}')
#             plt.title(f'Predictions vs True Values for {name}')
            
#             # 添加对角线
#             min_val = min(targets[:, i].min(), predictions[:, i].min())
#             max_val = max(targets[:, i].max(), predictions[:, i].max())
#             plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
            
#             # 添加统计信息
#             plt.text(0.05, 0.95, 
#                     f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
#                     transform=plt.gca().transAxes,
#                     verticalalignment='top',
#                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(f'improved_baseline_{name}_predictions.png')
#             plt.close()
        
#         return results

# def main():
#     # 加载数据
#     image_data = np.load('sample_cnn_lh_brainimages.npy')
#     phenotype_data = np.load('sample_phenotype_tensor_normalized.npy')
    
#     # 分割数据
#     train_images, test_images, train_phenotypes, test_phenotypes = train_test_split(
#         image_data, phenotype_data, test_size=0.2, random_state=42)
    
#     # 获取目标变量（前两列）和表型特征（后三列）
#     train_targets = train_phenotypes[:, :2]
#     test_targets = test_phenotypes[:, :2]
#     train_phenotypes = train_phenotypes[:, 2:5]
#     test_phenotypes = test_phenotypes[:, 2:5]
    
#     # 训练和评估改进后的模型
#     model = ImprovedBaselineModel()
#     model.fit(train_images, train_phenotypes, train_targets)
#     results = model.evaluate(test_images, test_phenotypes, test_targets)
    
#     print("\nImproved Model Results:")
#     for target, metrics in results.items():
#         print(f"\n{target}:")
#         for metric_name, value in metrics.items():
#             print(f"{metric_name}: {value:.4f}")

# if __name__ == "__main__":
#     main()

"""job-ID:8470575"""

# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt

# class SimpleBaselineModel:
#     def __init__(self):
#         self.scaler = StandardScaler()
#         self.model = Ridge(alpha=1.0)
        
#     def preprocess_brain_images(self, images):
#         """将4D脑图像数据转换为2D特征矩阵"""
#         batch_size = images.shape[0]
#         # 对每个通道计算统计特征
#         features = []
        
#         for i in range(4):  # 对每个通道
#             channel = images[:, i, :, :]
#             # 计算每个图像的统计特征
#             mean = channel.mean(axis=(1, 2))
#             std = channel.std(axis=(1, 2))
#             max_val = channel.max(axis=(1, 2))
#             min_val = channel.min(axis=(1, 2))
#             # 计算四分位数
#             q1 = np.percentile(channel, 25, axis=(1, 2))
#             q3 = np.percentile(channel, 75, axis=(1, 2))
            
#             # 合并该通道的所有特征
#             channel_features = np.vstack([mean, std, max_val, min_val, q1, q3]).T
#             features.append(channel_features)
            
#         # 将所有通道的特征合并
#         return np.hstack(features)  # (batch_size, 24) - 每个通道6个特征，共4个通道
    
#     def fit(self, brain_images, phenotypes, targets):
#         """训练模型"""
#         # 预处理脑图像数据
#         brain_features = self.preprocess_brain_images(brain_images)
        
#         # 合并脑特征和表型数据
#         X = np.hstack([brain_features, phenotypes])
        
#         # 标准化特征
#         X = self.scaler.fit_transform(X)
        
#         # 训练模型
#         self.model.fit(X, targets)
        
#         return self
    
#     def predict(self, brain_images, phenotypes):
#         """预测"""
#         brain_features = self.preprocess_brain_images(brain_images)
#         X = np.hstack([brain_features, phenotypes])
#         X = self.scaler.transform(X)
#         return self.model.predict(X)
    
#     def evaluate(self, brain_images, phenotypes, targets):
#         """评估模型"""
#         predictions = self.predict(brain_images, phenotypes)
#         results = {}
        
#         target_names = ['sum_att', 'sum_agg']
#         for i, name in enumerate(target_names):
#             mse = mean_squared_error(targets[:, i], predictions[:, i])
#             r2 = r2_score(targets[:, i], predictions[:, i])
#             results[name] = {'MSE': mse, 'R2': r2}
            
#             # 绘制散点图
#             plt.figure(figsize=(10, 5))
#             plt.scatter(targets[:, i], predictions[:, i], alpha=0.5)
#             plt.xlabel(f'True {name}')
#             plt.ylabel(f'Predicted {name}')
#             plt.title(f'Predictions vs True Values for {name}')
            
#             # 添加对角线
#             min_val = min(targets[:, i].min(), predictions[:, i].min())
#             max_val = max(targets[:, i].max(), predictions[:, i].max())
#             plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
            
#             # 添加统计信息
#             plt.text(0.05, 0.95, 
#                     f'MSE: {mse:.4f}\nR²: {r2:.4f}', 
#                     transform=plt.gca().transAxes,
#                     verticalalignment='top',
#                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
#             plt.legend()
#             plt.grid(True)
#             plt.savefig(f'test_simplest_{name}_predictions_sample_lh.png')
#             plt.close()
            
#         return results

# # 使用示例
# def main():
#     # 加载数据
#     image_data = np.load('sample_cnn_lh_brainimages.npy')
#     phenotype_data = np.load('sample_phenotype_tensor_normalized.npy')
    
#     # 分割数据
#     train_images, test_images, train_phenotypes, test_phenotypes = train_test_split(
#         image_data, phenotype_data, test_size=0.2, random_state=42)
    
#     # 获取目标变量（前两列）
#     train_targets = train_phenotypes[:, :2]
#     test_targets = test_phenotypes[:, :2]
    
#     # 获取表型特征（后三列）
#     train_phenotypes = train_phenotypes[:, 2:5]
#     test_phenotypes = test_phenotypes[:, 2:5]
    
#     # 训练和评估模型
#     model = SimpleBaselineModel()
#     model.fit(train_images, train_phenotypes, train_targets)
#     results = model.evaluate(test_images, test_phenotypes, test_targets)
    
#     # 打印结果
#     print("\nBaseline Model Results:")
#     for target, metrics in results.items():
#         print(f"\n{target}:")
#         for metric_name, value in metrics.items():
#             print(f"{metric_name}: {value:.4f}")

# if __name__ == "__main__":
#     main()