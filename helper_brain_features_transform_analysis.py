
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# import os
# from sklearn.preprocessing import QuantileTransformer

# def load_and_analyze_data(file_path, output_dir):
#     """
#     Load data and perform comprehensive analysis.
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     data = np.load(file_path).reshape(-1)
#     print(f"Data shape: {data.shape}")
    
#     n_samples, n_features, n_values = data.shape
    
#     analyze_feature_distribution(data, output_dir)
#     compare_sample_distributions(data, output_dir)
#     analyze_feature_correlations(data, output_dir)
#     analyze_quantile_transform_effect(data, output_dir)
#     analyze_data_range(data, output_dir)
#     plot_qq(data, output_dir)
#     analyze_outliers(data, output_dir)

# def analyze_feature_distribution(data, output_dir):
#     n_samples, n_features, n_values = data.shape
    
#     for feature in range(n_features):
#         plt.figure(figsize=(12, 6))
        
#         plt.subplot(1, 2, 1)
#         sns.histplot(data[:, feature, :].flatten(), kde=True, bins=100)
#         plt.title(f"Feature {feature+1} - All Samples")
#         plt.xlabel("Value")
#         plt.ylabel("Frequency")
        
#         plt.subplot(1, 2, 2)
#         mean_values = np.mean(data[:, feature, :], axis=1)
#         sns.histplot(mean_values, kde=True, bins=50)
#         plt.title(f"Feature {feature+1} - Mean Across Samples")
#         plt.xlabel("Mean Value")
#         plt.ylabel("Frequency")
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f"feature_{feature+1}_distribution.png"))
#         plt.close()

# def compare_sample_distributions(data, output_dir, n_samples_to_plot=5):
#     n_samples, n_features, n_values = data.shape
    
#     for feature in range(n_features):
#         plt.figure(figsize=(12, 6))
        
#         for i in range(min(n_samples_to_plot, n_samples)):
#             sns.kdeplot(data[i, feature, :], label=f"Sample {i+1}")
        
#         plt.title(f"Feature {feature+1} - Distribution Across Samples")
#         plt.xlabel("Value")
#         plt.ylabel("Density")
#         plt.legend()
#         plt.savefig(os.path.join(output_dir, f"feature_{feature+1}_sample_comparison.png"))
#         plt.close()

# def analyze_feature_correlations(data, output_dir):
#     n_samples, n_features, n_values = data.shape
    
#     mean_values = np.mean(data, axis=(0, 2))
#     corr_matrix = np.corrcoef(mean_values)
    
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
#     plt.title("Feature Correlation Matrix (Based on Mean Values)")
#     plt.savefig(os.path.join(output_dir, "feature_correlation.png"))
#     plt.close()

# def analyze_quantile_transform_effect(data, output_dir):
#     n_samples, n_features, n_values = data.shape
    
#     qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
    
#     for feature in range(n_features):
#         original_data = data[:, feature, :].flatten()
#         transformed_data = qt.fit_transform(original_data.reshape(-1, 1)).flatten()
        
#         plt.figure(figsize=(12, 6))
        
#         plt.subplot(1, 2, 1)
#         sns.histplot(original_data, kde=True, bins=100)
#         plt.title(f"Feature {feature+1} - Original")
#         plt.xlabel("Value")
#         plt.ylabel("Frequency")
        
#         plt.subplot(1, 2, 2)
#         sns.histplot(transformed_data, kde=True, bins=100)
#         plt.title(f"Feature {feature+1} - After Quantile Transform")
#         plt.xlabel("Value")
#         plt.ylabel("Frequency")
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f"feature_{feature+1}_quantile_transform_effect.png"))
#         plt.close()

# def analyze_data_range(data, output_dir):
#     n_samples, n_features, n_values = data.shape
    
#     for feature in range(n_features):
#         feature_data = data[:, feature, :].flatten()
#         unique_values = np.unique(feature_data)
        
#         with open(os.path.join(output_dir, f"feature_{feature+1}_range_analysis.txt"), 'w') as f:
#             f.write(f"Feature {feature+1}:\n")
#             f.write(f"  Number of unique values: {len(unique_values)}\n")
#             f.write(f"  Range: {np.min(feature_data)} to {np.max(feature_data)}\n")
#             if len(unique_values) < 20:
#                 f.write(f"  Unique values: {unique_values}\n")
#             f.write("\n")

# def plot_qq(data, output_dir):
#     n_samples, n_features, n_values = data.shape
    
#     for feature in range(n_features):
#         feature_data = data[:, feature, :].flatten()
#         plt.figure(figsize=(10, 6))
#         stats.probplot(feature_data, dist="norm", plot=plt)
#         plt.title(f"Q-Q Plot for Feature {feature+1}")
#         plt.savefig(os.path.join(output_dir, f"qq_plot_feature_{feature+1}.png"))
#         plt.close()

# def analyze_outliers(data, output_dir):
#     n_samples, n_features, n_values = data.shape
    
#     for feature in range(n_features):
#         feature_data = data[:, feature, :].flatten()
#         q1, q3 = np.percentile(feature_data, [25, 75])
#         iqr = q3 - q1
#         lower_bound = q1 - (1.5 * iqr)
#         upper_bound = q3 + (1.5 * iqr)
#         outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
        
#         plt.figure(figsize=(10, 6))
#         plt.boxplot(feature_data)
#         plt.title(f"Boxplot with Outliers for Feature {feature+1}")
#         plt.savefig(os.path.join(output_dir, f"outliers_feature_{feature+1}.png"))
#         plt.close()
        
#         with open(os.path.join(output_dir, f"feature_{feature+1}_outlier_analysis.txt"), 'w') as f:
#             f.write(f"Feature {feature+1}:\n")
#             f.write(f"  Number of outliers: {len(outliers)}\n")
#             f.write(f"  Percentage of outliers: {len(outliers) / len(feature_data) * 100:.2f}%\n")
#             f.write("\n")

# def compare_left_right_brain(left_file, right_file, output_dir):
#     left_data = np.load(left_file)
#     right_data = np.load(right_file)
    
#     n_features = left_data.shape[1]  # Assuming second dimension is features
    
#     for feature in range(n_features):
#         plt.figure(figsize=(12, 6))
        
#         plt.subplot(1, 2, 1)
#         sns.histplot(left_data[:, feature, :].flatten(), kde=True, bins=100)
#         plt.title(f"Left Brain - Feature {feature+1}")
#         plt.xlabel("Value")
#         plt.ylabel("Frequency")
        
#         plt.subplot(1, 2, 2)
#         sns.histplot(right_data[:, feature, :].flatten(), kde=True, bins=100)
#         plt.title(f"Right Brain - Feature {feature+1}")
#         plt.xlabel("Value")
#         plt.ylabel("Frequency")
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(output_dir, f"left_right_comparison_feature_{feature+1}.png"))
#         plt.close()

# if __name__ == "__main__":
#     left_file_path = ' sample_input_tensor_quantile_transform.npy'
#     right_file_path = 'sample_input_tensor_quantile_transform_right.npy'
#     output_dir = 'output_comprehensive_analysis'
    
#     load_and_analyze_data(left_file_path, os.path.join(output_dir, 'left_brain'))
#     load_and_analyze_data(right_file_path, os.path.join(output_dir, 'right_brain'))
#     compare_left_right_brain(left_file_path, right_file_path, os.path.join(output_dir, 'left_right_comparison'))
    
#     print(f"Analysis complete. Results saved in '{output_dir}' directory.")






# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# import os

# def analyze_and_plot_data_in_batches(file_path, output_dir, batch_size=100):
#     """
#     Analyze brain data and save distribution plots in batches.
    
#     Args:
#     file_path: Path to the .npy file containing the brain data
#     output_dir: Directory to save the output plots
#     batch_size: Number of samples to process in each batch
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     data = np.load(file_path, mmap_mode='r')
#     print(f"Original data shape: {data.shape}")
    
#     n_samples, *dims, n_features = data.shape
#     total_points = np.prod(dims)
    
#     # Initialize statistics
#     stats_dict = {i: {"sum": 0, "sum_sq": 0, "min": np.inf, "max": -np.inf} for i in range(n_features)}
    
#     # Process data in batches
#     for start in range(0, n_samples, batch_size):
#         end = min(start + batch_size, n_samples)
#         batch = data[start:end].reshape((end - start) * total_points, n_features)
        
#         for i in range(n_features):
#             feature_data = batch[:, i]
#             stats_dict[i]["sum"] += np.sum(feature_data)
#             stats_dict[i]["sum_sq"] += np.sum(np.square(feature_data))
#             stats_dict[i]["min"] = min(stats_dict[i]["min"], np.min(feature_data))
#             stats_dict[i]["max"] = max(stats_dict[i]["max"], np.max(feature_data))
        
#         # Free up memory
#         del batch
    
#     # Calculate final statistics
#     total_count = n_samples * total_points
#     for i in stats_dict:
#         mean = stats_dict[i]["sum"] / total_count
#         var = (stats_dict[i]["sum_sq"] / total_count) - (mean ** 2)
#         stats_dict[i]["mean"] = mean
#         stats_dict[i]["std"] = np.sqrt(var)
    
#     # Plot and save distributions
#     for i in range(n_features):
#         plt.figure(figsize=(10, 6))
#         sns.histplot(data=data[:, :, :, i].flatten(), kde=True, bins=100)
#         plt.title(f"Feature {i+1} Distribution")
#         plt.xlabel("Feature Value")
#         plt.ylabel("Frequency")
#         plt.savefig(os.path.join(output_dir, f"feature_{i+1}_distribution.png"))
#         plt.close()
    
#     return stats_dict

# def suggest_normalization(stats_dict):
#     """
#     Suggest normalization methods based on the data distribution for each feature.
#     """
#     suggestions = {}
#     for feature, stats in stats_dict.items():
#         if abs(stats["max"] - stats["min"]) > 10 * stats["std"]:
#             suggestions[feature] = "Data range is large compared to its standard deviation. Consider using robust scaling or normalization."
#         elif stats["std"] < 0.01 * abs(stats["mean"]):
#             suggestions[feature] = "Data has low variance compared to its mean. Standard scaling might be appropriate."
#         else:
#             suggestions[feature] = "Data appears to have a reasonable spread. Standard scaling or min-max scaling could be appropriate."
#     return suggestions

# # Main execution
# # 
# if __name__ == "__main__":
#     file_path = 'sample_input_tensor_quantile_transform_pct01.npy'  # Update this to your file path
#     output_dir = 'output_plots_brain_features_quantile_trans_right'  # Directory to save plots
    
#     stats = analyze_and_plot_data_in_batches(file_path, output_dir)
    
#     print("Data Statistics:")
#     for feature, feature_stats in stats.items():
#         print(f"\nFeature {feature}:")
#         for stat_name, stat_value in feature_stats.items():
#             print(f"  {stat_name}: {stat_value}")
    
#     suggestions = suggest_normalization(stats)
#     print("\nNormalization Suggestions:")
#     for feature, suggestion in suggestions.items():
#         print(f"Feature {feature}: {suggestion}")

#     print(f"\nPlots have been saved in the '{output_dir}' directory.")







# load packages
import pandas as pd
import numpy as np
import pyreadr
import pyreadstat
# import random
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
import matplotlib.pyplot as plt
from longitude_transform import get_longitudinal_map_each, sphere_to_grid_each, color_map_DK, plot_original, plot_DK_map
from helper_func_prep import SOI_array_per_right,SOI_array_per_fs5, min_max_normalize, plot_SOI, robust_scale_normalize, quantile_normalize
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# import gc

"""
Generation R
We do on 5 files (core data has 2), and for different QC conditions, and we inspect the structure and range of the GenR phenotype data.
For GenR dataset, we have a concern about whether we should include follow-up data in out dataset, so I will include them into f13,
because of the data imbalance on f9 an f13!
The final usable dataframe is 'genr_merge_9' and 'genr_merge_13', and if including overlap only at focus 13: 'genr_merge_13_incl'
(idc 	age	    sex	  edu_maternal	  sum_att	  sum_agg)


count_for_nan = df['...'].isna().sum()
.unique() to find the unique values of each attributes (columns)
.nunique(): number of unique values

# compare two dataframe!!
# a = list(abcd_cbcl['subjectkey'])
# b = list(abcd_sex_age_edu['subjectkey'])
# diff = []
# for element in a:
#     if element not in b:
#         diff.append(element)
# print(len(diff))
# print(diff)
# merge = list(set(abcd_cbcl['subjectkey']) & set(abcd_qc['subjectkey']) & set(abcd_sex_age_edu['subjectkey']) & set(abcd_incfind['subjectkey']) & set(abcd_icv_mri['subjectkey']) & set(abcd_siblings['subjectkey']))


ls | grep 10604

"""

# QC for GenR core
# load the GenR core data
genr_core_f9_raw = pyreadr.read_r('./GenR_core/core_f9_qc_all.RDS') #3185, after QC, access with key None
genr_core_f13_raw = pyreadr.read_r('./GenR_core/core_f13_qc_all.RDS') #2165, after QC, access with key None, and they have 1229 duplicates

genr_core_f9_all = genr_core_f9_raw[None]
genr_core_f9_all['idc'] = genr_core_f9_all['idc'].astype('int').astype('str')

genr_core_f13_all = genr_core_f13_raw[None]
genr_core_f13_all['idc'] = genr_core_f13_all['idc'].astype('int').astype('str')

# this is the overlap at focus 9 and 13. 
genr_overlap_idc_list = list(genr_core_f9_all[genr_core_f9_all.idc.isin(genr_core_f13_all.idc)]['idc']) # 1229

# remove the overlap at focus 9 and 13
genr_core_f9 = genr_core_f9_all[genr_core_f9_all.idc.isin(genr_overlap_idc_list) == False].reset_index(drop=True) # 1956 rows × 75 columns
genr_core_f13 = genr_core_f13_all                                                    # we keep follow-up at focus 13, 2165 rows × 75 columns
# genr_core_f13 = genr_core_f13_all[genr_core_f13_all.idc.isin(genr_overlap_idc_list) == False].reset_index(drop=True) # 936 rows × 75 columns


#  GenR CBCL, and sex & edu_level data 
# load data for GenR phenotype data
genr_cbcl_f9_raw, _ = pyreadstat.read_sav('./GenR_core/genr/CHILDCBCL9_incl_Tscores_20201111.sav') # 9901 rows × 650 columns
genr_cbcl_f13_raw, _ =  pyreadstat.read_sav('./GenR_core/genr/GR1093-E1_CBCL_18062020.sav') # 9901 rows × 331 columns
genr_sex_n_edul_raw, _ = pyreadstat.read_sav('./GenR_core/genr/CHILD-ALLGENERALDATA_24102022.sav') # 9901 rows × 121 columns

# change to lower column name
genr_cbcl_f9_raw.columns = genr_cbcl_f9_raw.columns.str.lower()
genr_cbcl_f13_raw.columns = genr_cbcl_f13_raw.columns.str.lower()
genr_sex_n_edul_raw.columns = genr_sex_n_edul_raw.columns.str.lower()
# make idc as string without '.'
genr_cbcl_f9_raw['idc'] = genr_cbcl_f9_raw['idc'].astype('int').astype('str')
genr_cbcl_f13_raw['idc'] = genr_cbcl_f13_raw['idc'].astype('int').astype('str')
genr_sex_n_edul_raw['idc'] = genr_sex_n_edul_raw['idc'].astype('int').astype('str')

# GenR CBCL on focus 9 and 13 without overlap
genr_cbcl_f9 = genr_cbcl_f9_raw.loc[genr_cbcl_f9_raw['idc'].isin(list(genr_core_f9['idc'].astype('int').astype('str')))].reset_index(drop=True) # 1956 rows × 650 columns
genr_cbcl_f13 = genr_cbcl_f13_raw.loc[genr_cbcl_f13_raw['idc'].isin(list(genr_core_f13['idc'].astype('int').astype('str')))].reset_index(drop=True) # 2165 rows × 331 columns

# GenR CBCL on focus 13 with overlap
# genr_cbcl_f9_incl = genr_cbcl_f9_raw.loc[genr_cbcl_f9_raw['idc'].isin(list(genr_core_f9_all['idc'].astype('int').astype('str')))].reset_index(drop=True) # 3185 rows × 650 columns
# genr_cbcl_f13_incl = genr_cbcl_f13_raw.loc[genr_cbcl_f13_raw['idc'].isin(list(genr_core_f13_all['idc'].astype('int').astype('str')))].reset_index(drop=True) # 2165 rows × 331 columns

# GenR sex & edu_level on focus 9 and 13 without overlap
genr_sex_n_edul_f9 = genr_sex_n_edul_raw.loc[genr_sex_n_edul_raw["idc"].isin(list(genr_core_f9['idc'].astype('int').astype('str')))].reset_index(drop=True) # 1956 rows × 121 columns
genr_sex_n_edul_f13 = genr_sex_n_edul_raw.loc[genr_sex_n_edul_raw["idc"].isin(list(genr_core_f13['idc'].astype('int').astype('str')))].reset_index(drop=True) # 2165 rows × 121 columns


# random select one of the siblings at focus 9 and 13
print(genr_sex_n_edul_f9['mother'].duplicated().sum(), genr_sex_n_edul_f13['mother'].duplicated().sum())
genr_sex_n_edul_f9 = genr_sex_n_edul_f9.drop_duplicates(subset=['mother']) # 1893 rows × 121 columns, remove duplicates: 63
genr_sex_n_edul_f13 = genr_sex_n_edul_f13.drop_duplicates(subset=['mother']) # 2090 rows × 121 columns, remove duplicates: 75


#  select GenR phenotype data
# GenR: age diffent at MRI and CBCl, I choose the age at MRI assessment took place.
#       - Child age at assessment at @9 (“age_child_mri_f09”)
#       - Child age at assessment at @13 ("age_child_mri_f13")
#       - `genr_mri_core_data_20220311.rds`
#       - range at focus 9: 8.550308008213554 - 11.986310746064339
#       - range at focus 13: 12.591375770020534 - 16.67898699520876
genr_age_9 = genr_core_f9[['idc','age_child_mri_f09']] # 1956 rows × 2 columns
# genr_age_9 = genr_age_9[genr_age_9.idc.isin(list(age_9_general['idc']))== True].reset_index(drop=True) # for 6-month constrains, no need. 1290 rows × 2 columns, no nan
genr_age_13 = genr_core_f13[['idc','age_child_mri_f13']] # 2165 rows × 2 columns
# genr_age_13 = genr_age_13[genr_age_13.idc.isin(list(age_13_general['idc']))== True].reset_index(drop=True) # for 6-month constrains, no need. 1308 rows × 2 columns, no nan



# GenR: sex assigned at birth ("gender"), education$05 (check no missing data) or ("educm5")
# `CHILD-ALLGENERALDATA_24102022.sav`
#  gender: 1:..., 2:...(1 is the boy, 2 is the girl)
#  educm5: 0: no education finished, 1: primary, 2: secondary, phase 1, 3: secondary, phase 2, 4: higher, phase 1; 5: higher, phase 2
# categorise the edu_level: 0-1, 2-3, 4-5
genr_sex_n_edul_9 = genr_sex_n_edul_f9[['idc', 'gender','educm5']] #1893 rows × 3 columns, nan: educm5: 274
genr_sex_n_edul_13 = genr_sex_n_edul_f13[['idc', 'gender', 'educm5']] #2090 rows × 3 columns, nan: gender:1, educm5:276, the one missing gender, also missing educm5
# remove the nan
genr_sex_n_edul_9 = genr_sex_n_edul_9.dropna().reset_index(drop=True) #1619 rows × 3 columns
genr_sex_n_edul_13 = genr_sex_n_edul_13.dropna().reset_index(drop=True) #1814 rows × 3 columns
# GenR: CBCL: 
#       - @age 9: Attention problems ("sum_att_9m") and Aggression problems ("sum_agg_9m") from `CHILDCBCL9_incl_Tscores_20201111.sav`
#       - @age 13: Attention problems ("sum_att_14") and Aggression problems ("sum_agg_14") from `GR1093-E1_CBCL_18062020.sav`
#       - range: @9: attention: 0.0 - 16.0, aggresive: 0.0 - 23.0
#       - range: @13: attention: 0.0 - 18.0, aggresive: 0.0 - 34.0
genr_cbcl_9 = genr_cbcl_f9[['idc', 'sum_att_9m', 'sum_agg_9m']] #1956 rows × 3 columns, nan: sum_att_9m: 320, sum_agg_9m: 322
genr_cbcl_13 = genr_cbcl_f13[['idc', 'sum_att_14', 'sum_agg_14'] ]# 2165 rows × 3 columns, nan: sum_att_14: 259, sum_agg_14: 259
# remove the nan
genr_cbcl_9 = genr_cbcl_9.dropna().reset_index(drop=True) #1632 rows × 3 columns
genr_cbcl_13 = genr_cbcl_13.dropna().reset_index(drop=True) #1905 rows × 3 columns

# merge the GenR phenotype data
genr_merge_9 = genr_age_9.merge(genr_sex_n_edul_9, on='idc').merge(genr_cbcl_9, on='idc') #1429 rows × 6 columns
genr_merge_13 = genr_age_13.merge(genr_sex_n_edul_13, on='idc').merge(genr_cbcl_13, on='idc') #1659 rows × 6 columns

genr_merge_9_ = genr_merge_9.rename(columns = {'age_child_mri_f09':'age', 'sum_att_9m':'sum_att', 'sum_agg_9m':'sum_agg', 'educm5':'edu_maternal', 'gender':'sex'})
genr_merge_13_ = genr_merge_13.rename(columns = {'age_child_mri_f13':'age', 'sum_att_14':'sum_att', 'sum_agg_14':'sum_agg','educm5':'edu_maternal', 'gender':'sex'})
genr_merge_all = [genr_merge_9_, genr_merge_13_]
genr_merge_all = pd.concat(genr_merge_all).reset_index(drop=True) # 3088 rows × 6 columns

"""
ABCD phenotype data origanize

We do the QC on six .txt files for different QC conditions, and we inspect the structure and range of the ABCD phenotype data.
The final usable dataframe is abcd_phenotype_rename' with 9511 rows * 7 columns (as below). 
(subjectkey, cbcl_scr_syn_attention_r, cbcl_scr_syn_aggressive_r, interview_age, sex. demo_prnt_ed_v2_l) 

"""



#Read all ABCD_txt files
abcd_cbcl_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_cbcls01.txt", sep='\t')
abcd_qc_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_imgincl01.txt", sep='\t')
abcd_sex_age_edu_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_lpds01.txt", sep='\t')
abcd_incfind_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_mrfindings02.txt", sep='\t')
abcd_icv_mri_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_smrip10201.txt", sep='\t') 
abcd_siblings_txt = pd.read_csv("/home/jouyang/ABCD_txt/acspsw03.txt", sep='\t')

#  select the core data of interests
abcd_cbcl = abcd_cbcl_txt[['subjectkey', 'cbcl_scr_syn_attention_r', 'cbcl_scr_syn_aggressive_r']] # missing#: 2473, df.dropna()
abcd_qc = abcd_qc_txt[['subjectkey','imgincl_t1w_include', 'eventname']] # qc ==1 & !is.na(qc): qc==1 and df.dropna(), nomissing but qc=0 exist, we have #19097 in the end.
abcd_sex_age_edu = abcd_sex_age_edu_txt[['subjectkey','interview_age', 'sex', 'demo_prnt_ed_v2_l']] # NA check in sex, age, edu, df.dropna(), 28560 missing in edu, we have 11207 then.
abcd_incfind = abcd_incfind_txt[['subjectkey', 'mrif_score']] # exclude IF: set mrif_score = 1 or 2, we then have 18759 rows
abcd_icv_mri = abcd_icv_mri_txt[['subjectkey', 'smri_vol_scs_intracranialv']] # no missing 
abcd_siblings = abcd_siblings_txt[['subjectkey', 'rel_family_id']] # only one at one family: !duplicated(), we have 10700. 

print(len(abcd_cbcl), len(abcd_qc), len(abcd_sex_age_edu), len(abcd_incfind), len(abcd_icv_mri), len(abcd_siblings))
#  first do the NaN drop and QC control
#   - in the email (8 guesses):
#   - ABCD: siblings pick for ABCD
abcd_cbcl = abcd_cbcl.dropna().iloc[1:] # 39767 -> 37293
abcd_qc = abcd_qc.loc[abcd_qc['imgincl_t1w_include'] == '1'] # 19659 -> 19097
abcd_sex_age_edu = abcd_sex_age_edu.dropna().iloc[1:]  # 39767 -> 11206
abcd_sex_age_edu['demo_prnt_ed_v2_l'] = abcd_sex_age_edu['demo_prnt_ed_v2_l'].astype('int')
abcd_sex_age_edu = abcd_sex_age_edu.loc[abcd_sex_age_edu['demo_prnt_ed_v2_l'] != 777] # 11206 -> 11188 ('777': 18)
abcd_incfind = abcd_incfind.loc[abcd_incfind['mrif_score'].isin(['1','2'])] # 28542 -> 18759 
abcd_icv_mri = abcd_icv_mri.iloc[1:] # 19588 -> 19587
abcd_siblings = abcd_siblings.loc[abcd_siblings['rel_family_id'].notna()].iloc[1:]
abcd_siblings = abcd_siblings.drop_duplicates(subset=['rel_family_id']) # 23102 -> 10698
print(len(abcd_cbcl), len(abcd_qc), len(abcd_sex_age_edu), len(abcd_incfind), len(abcd_icv_mri), len(abcd_siblings))

#  then merge to a whole one, meaning simply find the common set for all these datasets above
abcd_merge = abcd_cbcl.merge(abcd_qc, on='subjectkey').merge(abcd_sex_age_edu, on='subjectkey').merge(abcd_incfind, on='subjectkey').merge(abcd_icv_mri, on='subjectkey').merge(abcd_siblings, on='subjectkey')
# print('merge1', len(abcd_merge))
abcd_merge = abcd_merge.dropna()
# print('merge2', len(abcd_merge))
abcd_merge = abcd_merge.drop_duplicates(subset=['subjectkey']) # merge: 9511
print('merge3', len(abcd_merge))

#  ABCD age (interview_age), `abcd_lpds01`
# what age do you want for ABCD? (116 to 149 months)
abcd_merge['interview_age'] = abcd_merge['interview_age'].astype('int')
abcd_age_max_mons = abcd_merge['interview_age'].max()/12 #12.42 yrs
abcd_age_min_mons = abcd_merge['interview_age'].min()/12 #9.67 yrs
# ABCD: BIO sex (sex), `abcd_lpds01`, M = Male; F = Female; 
print(abcd_age_max_mons, abcd_age_min_mons)

#  ABCD: education level of parents 
"""
1-12 are grade number, 
13 = High school graduate Preparatoria terminada ; 
14 = GED or equivalent Diploma General de Equivalencia (GED) o equivalente ; 
Genr == 1 (may include 15)

15 = Some college; 
16 = Associate degree: Occupational; 
17 = Associate degree: Academic Program Título de asociado: programa académico ; 
Genr == 2

18 = Bachelor's degree (ex. BA; 19 = Master's degree (ex. MA; 
20 = Professional School degree (ex. MD; 
21 = Doctoral degree (ex. PhD; 
Genr == 3

777 = Refused to answer Prefiero no responder (already exlude! in total 17!)
"""
# demo_prnt_ed_v2_l from `abcd_lpds01`
abcd_merge['demo_prnt_ed_v2_l'] = abcd_merge['demo_prnt_ed_v2_l'].astype('int')

abcd_edu_summary = np.sort(abcd_merge['demo_prnt_ed_v2_l'].unique())
print(abcd_edu_summary)
# output: array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        # 14,  15,  16,  17,  18,  19,  20,  21]), has removed 777
#  ABCD: CBCL
#       - Attention problems (cbcl_scr_syn_attention_r) and Aggressive problems (cbcl_scr_syn_aggressive_r) 
#       - from `abcd_cbcls01`
abcd_merge[['cbcl_scr_syn_attention_r', 'cbcl_scr_syn_aggressive_r']] = abcd_merge[['cbcl_scr_syn_attention_r', 'cbcl_scr_syn_aggressive_r']].astype('float')
print(abcd_merge['cbcl_scr_syn_attention_r'].max()) #19.0
print(abcd_merge['cbcl_scr_syn_attention_r'].min()) #0.0
print(abcd_merge['cbcl_scr_syn_aggressive_r'].max()) #33.0
print(abcd_merge['cbcl_scr_syn_aggressive_r'].min()) #0.0

#  ABCD dataset for phenotype (subjectkey, cbcl_scr_syn_attention_r, cbcl_scr_syn_aggressive_r)
# interview_age, sex, demo_prnt_ed_v2_l)
abcd_drop_list = ['imgincl_t1w_include', 'mrif_score', 'smri_vol_scs_intracranialv', 'rel_family_id']
abcd_phenotype = abcd_merge.copy()
abcd_phenotype = abcd_phenotype.drop(columns=abcd_drop_list, axis=1).reset_index(drop=True) # 9511 rows × 7 columns
abcd_phenotype_rename = abcd_phenotype.rename(columns = {'subjectkey':'idc','cbcl_scr_syn_attention_r':'sum_att','cbcl_scr_syn_aggressive_r':'sum_agg', 'interview_age':'age', 'demo_prnt_ed_v2_l':'edu_maternal'})
abcd_phenotype_12_below = abcd_phenotype_rename.loc[abcd_phenotype['interview_age']<=144] # 9366 rows × 7 columns
abcd_phenotype_12_up = abcd_phenotype_rename.loc[abcd_phenotype['interview_age']>144] # 145 rows × 7 columns


"""
Preprocess the phenotype_data, merge them as one dataset and re-organize the column names
"""
#  merge abcd and genr dataset
abcd_pheno_preprocess = abcd_phenotype_rename.copy()
abcd_pheno_preprocess['sex'].replace(['F','M'], [0, 1],inplace=True) # female = 0, male = 1
abcd_pheno_preprocess['age'] = np.around(abcd_pheno_preprocess['age']/12, decimals=2) # round the age to years with 2 decimals
# abcd maternal edu level : 0, 1, 2.
abcd_pheno_preprocess['edu_maternal'] = np.where(abcd_pheno_preprocess['edu_maternal'] < 15, 0, abcd_pheno_preprocess['edu_maternal'])
abcd_pheno_preprocess['edu_maternal'] = np.where((abcd_pheno_preprocess['edu_maternal'] >= 15) & (abcd_pheno_preprocess['edu_maternal'] < 18), 1, abcd_pheno_preprocess['edu_maternal'])
abcd_pheno_preprocess['edu_maternal'] = np.where(abcd_pheno_preprocess['edu_maternal'] >= 18, 2, abcd_pheno_preprocess['edu_maternal'])
abcd_pheno_preprocess['idc'] = abcd_pheno_preprocess['idc'].str.replace('_', '') # 9511 rows × 7 columns
abcd_pheno_preprocess_ = abcd_pheno_preprocess.drop(columns=['eventname']) #9511 rows × 6 columns
# 
genr_pheno_preprocess = genr_merge_all.copy()
genr_pheno_preprocess['sex'].replace([2.0, 1.0], [0, 1],inplace=True) # female = 0, male = 1
genr_pheno_preprocess['age'] = np.around(genr_pheno_preprocess['age'], decimals=2) # round the age to 2 decimals
# genr maternal edu level : 0, 1, 2.
genr_pheno_preprocess['edu_maternal'] = np.where((genr_pheno_preprocess['edu_maternal'] == 0.0) | (genr_pheno_preprocess['edu_maternal'] == 1.0), 0, genr_pheno_preprocess['edu_maternal'])
genr_pheno_preprocess['edu_maternal'] = np.where((genr_pheno_preprocess['edu_maternal'] == 2.0) | (genr_pheno_preprocess['edu_maternal'] == 3.0), 1, genr_pheno_preprocess['edu_maternal'])
genr_pheno_preprocess['edu_maternal'] = np.where((genr_pheno_preprocess['edu_maternal'] == 4.0) | (genr_pheno_preprocess['edu_maternal'] == 5.0), 2, genr_pheno_preprocess['edu_maternal'])

# genr_pheno_preprocess = genr_merge_all['sex'].replace(1.0, 1,inplace=True)
phenotype_data = [abcd_pheno_preprocess, genr_pheno_preprocess] # 12599 rows × 7 columns = abcd: 9511 rows × 7 columns + genr: 3088 rows × 6 columns
phenotype_data = pd.concat(phenotype_data).reset_index(drop=True)
phenotype_data = phenotype_data.astype({"sex": 'int', "edu_maternal": 'int'})
phenotype_data = phenotype_data.astype({"sex": 'category', "edu_maternal": 'category', "eventname": 'category'}) #12599 rows × 6 columns
# phenotype_data['idc'].to_csv('ids_list.txt', index=False, header=False)
"""
Copy the SOI files in my folder, this can be done one-time, do not run it every time!!
"""
#  save the ids as .txt files
abcd_pheno_idc = abcd_pheno_preprocess[['idc','eventname']].copy()
abcd_pheno_idc_baseline = abcd_pheno_idc.loc[abcd_pheno_idc['eventname']=='baseline_year_1_arm_1'] # 5781 rows × 2 columns
abcd_pheno_idc_baseline['filename'] = abcd_pheno_idc_baseline['idc'].apply(lambda x: f'sub-{x}_ses-baselineYear1Arm1')
abcd_pheno_idc_followup = abcd_pheno_idc.loc[abcd_pheno_idc['eventname']=='2_year_follow_up_y_arm_1'] # 3730 rows × 2 columns
abcd_pheno_idc_followup['filename'] = abcd_pheno_idc_followup['idc'].apply(lambda x: f'sub-{x}_ses-2YearFollowUpYArm1')
# for ind in abcd_pheno_preprocess.index:
#     if abcd_pheno_preprocess['eventname'][ind] == 'baseline_year_1_arm_1':
#         abcd_pheno_preprocess['filename'][ind] = abcd_pheno_preprocess['idc'][ind].apply(lambda x: f'sub-{x}_ses-baselineYear1Arm1')
#         print(f"[{ind}/9511]") 
#     else:
#         # abcd_pheno_idc[ind] = abcd_pheno_idc.apply(lambda x: f'sub-{x}_ses-2YearFollowUpYArm1')
#         abcd_pheno_preprocess['filename'][ind] = abcd_pheno_preprocess['idc'][ind].apply(lambda x: f'sub-{x}_ses-2YearFollowUpYArm1')
#         print(f"[{ind}/9511]")  
abcd_pheno_idc_baseline['filename'].to_csv('ids_list_abcd_base.txt', index=False, header=False)
abcd_pheno_idc_followup['filename'].to_csv('ids_list_abcd_folup.txt', index=False, header=False)
#  rename the filename for locating the MRi files
genr_f09_pheno_idc = genr_merge_9_[['idc']].copy()
genr_f09_pheno_idc['filename'] = genr_f09_pheno_idc['idc'].apply(lambda x: f'sub-{x}_ses-F09') # 1429: genr focus 9
genr_f09_pheno_idc['filename'].to_csv('ids_list_genr_f09.txt', index=False, header=False)

genr_f13_pheno_idc = genr_merge_13_[['idc']].copy()
genr_f13_pheno_idc['filename'] = genr_f13_pheno_idc['idc'].apply(lambda x: f'sub-{x}_ses-F13') # 1659: genr focus 13
genr_f13_pheno_idc['filename'].to_csv('ids_list_genr_f13.txt', index=False, header=False) 


#  remove the ids of phenotype data that does not have the corresponding MRI
# output_list = []
# # Open the output file and read its contents for abcd deleted ids
# with open("okay.txt", "r") as file:
#     for line in file:
#         # Check if "success!" is not in the line
#         if "success!" not in line:
#             # Extract the required string using split and slicing
#             start_index = line.find("sub-") + 4  # Find the start index of the pattern
#             end_index = line.find("_ses")       # Find the end index of the pattern
#             extracted_string = line[start_index:end_index]
#             output_list.append(extracted_string)

# # Write the extracted strings to a new file
# with open("processed_output.txt", "w") as out_file:
#     for item in output_list:
#         out_file.write(item + "\n")
# there two they don't have /surf

# sub-NDARINVE7869241_ses-2YearFollowUpYArm1
# sub-NDARINVR1927JG7_ses-2YearFollowUpYArm1
cleaned_list = ['2363', 'NDARMC003PZF', 'NDARINVE7869241', 'NDARINVR1927JG7'] # these are ids with no MRI (no baseline and followup year) in abcd
# remove the corresponding phenotype data using baseline data of ABCD
print(len(phenotype_data)) # original 12599 rows × 6 columns
# phenotype_data = phenotype_data.loc[phenotype_data['idc'] != ['2363', 'NDARMC003PZF']]
# print(len(phenotype_data))
phenotype_data = phenotype_data[~phenotype_data['idc'].isin(cleaned_list)].reset_index(drop=True) #12595 rows × 6 columns
print(len(phenotype_data))


#  Dataset split

"""
Dataset split

several thoughts:
1. if we merge ABCD and GenR phenotype datasets, and split then by age under 12.00 yrs, we get 149(abcd)+1105 = 1254 for testing (above 12 yrs),
    and 9366+1106 = 10472 for training (under 12 yrs). 
2. since we have to remove the phenotype data which has no MRI data, we get 143(abcd)+1104 = 1247 for testing (above 12 yrs),
    and 9145(abcd)+1106(genr) = 10251 for training (under 12 yrs). 
3. we decided to use the correspinding baseline and followup data, we get 145(abcd)+1104 = 1249 for testing (above 12 yrs),
    and 9364+1106 = 10470 for training (under 12 yrs). 
"""



# load the sample dataset index
# read the index .txt file
with open('sample_index.txt', 'r') as f:
    subset_phenotype_idx_ = [int(line.strip()) for line in f.readlines()]
# Remove newline characters
subset_phenotype = phenotype_data.iloc[subset_phenotype_idx_]
subset_phenotype_ids = subset_phenotype.iloc[:, :1] # no need to reset index, because we keep them for ID
subset_phenotype_ids = subset_phenotype_ids.reset_index(drop=True)
subset_phenotype_fname = []
for idx in subset_phenotype_ids.index:
    if subset_phenotype_ids.at[idx, 'idc'].startswith("NDAR"):
        if subset_phenotype_ids.at[idx, 'idc'] in abcd_pheno_idc_baseline['idc'].values:
            index = abcd_pheno_idc_baseline[abcd_pheno_idc_baseline['idc'] == subset_phenotype_ids.at[idx, 'idc']].index[0]
            subset_phenotype_fname.append(abcd_pheno_idc_baseline['filename'][index])
        elif subset_phenotype_ids.at[idx, 'idc'] in abcd_pheno_idc_followup['idc'].values:
            index = abcd_pheno_idc_followup[abcd_pheno_idc_followup['idc'] == subset_phenotype_ids.at[idx, 'idc']].index[0]
            subset_phenotype_fname.append(abcd_pheno_idc_followup['filename'][index])
    elif subset_phenotype_ids.at[idx, 'idc'] in genr_f09_pheno_idc['idc'].values:
        index = genr_f09_pheno_idc[genr_f09_pheno_idc['idc'] == subset_phenotype_ids.at[idx, 'idc']].index[0]
        subset_phenotype_fname.append(genr_f09_pheno_idc['filename'][index])
    elif subset_phenotype_ids.at[idx, 'idc'] in genr_f13_pheno_idc['idc'].values:
        index = genr_f13_pheno_idc[genr_f13_pheno_idc['idc'] == subset_phenotype_ids.at[idx, 'idc']].index[0]
        subset_phenotype_fname.append(genr_f13_pheno_idc['filename'][index])
subset_phenotype_ids['filename'] = subset_phenotype_fname


# Create a TensorDataset and DataLoader
#  function for brain MRI SOI dataset
def brain_SOI_matrix(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_half):
    # load the cortex label file, which is used for select the vertices.
    # left half brain: 149955, right half brain: 149926
    lh = open("/projects/0/einf1049/scratch/jouyang/GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
    # data is the raw data 
    data = lh.read().splitlines()
    # data_truc is the raw data without the header
    data_truc = data[2:]
    # This is the Longitude transform for one vertex of a person 
    longitude_mapping_per = get_longitudinal_map_each(data_truc)
    # This is the ij-2d grid for one vertex of a person, raduis = 100
    origin_ij_list, origin_ij_grid = sphere_to_grid_each(longitude_mapping_per,100)
    # maintain an indexing array ij_id
    ij_id = origin_ij_grid[:,0]
    ID_per_half = ij_id.astype('int')

    # load MRI files 
    thickness_array_re = SOI_array_per_right(ID_per_half, thick_path)  # thickness in [0, 4.37891531]
    volume_array_re = SOI_array_per_right(ID_per_half, volume_path)   # volume in [0, 5.9636817]
    SA_array_re = SOI_array_per_right(ID_per_half, SA_path)   # surface_area in [0, 1.40500367]
    w_g_array_re = SOI_array_per_right(ID_per_half, w_g_pct_path) # w/g ratio in [0, 48.43599319]

    # normalize the SOI data
    # FIXME: min-max normalization or preprocessing.normalize()?? Start with min-max first...
    # thickness_mx_norm = min_max_normalize(thickness_array_re)
    # volume_mx_norm = min_max_normalize(volume_array_re)
    # SA_mx_norm = min_max_normalize(SA_array_re)
    # w_g_ar_norm = w_g_array_re/100

    thickness_mx_norm = robust_scale_normalize(thickness_array_re)
    volume_mx_norm = robust_scale_normalize(volume_array_re)
    SA_mx_norm = robust_scale_normalize(SA_array_re)
    w_g_ar_norm = robust_scale_normalize(w_g_array_re)

    # quantile normalization the SOI data
    # thickness_mx_norm = quantile_normalize(thickness_array_re)
    # volume_mx_norm = quantile_normalize(volume_array_re)
    # SA_mx_norm = quantile_normalize(SA_array_re)
    # w_g_ar_norm = quantile_normalize(w_g_array_re/100)

    # stack them as a matrix
    SOI_mx_minmax = np.stack([thickness_mx_norm, volume_mx_norm, SA_mx_norm, w_g_ar_norm], axis=-1)

    return SOI_mx_minmax
# got the path of each person's brain MRI
# List of desired filenames
files_to_retrieve = [
    "lh.thickness.fwhm10.fsaverage.mgh",
    "lh.volume.fwhm10.fsaverage.mgh",
    "lh.area.fwhm10.fsaverage.mgh",
    "lh.w_g.pct.mgh.fwhm10.fsaverage.mgh",
    "rh.thickness.fwhm10.fsaverage.mgh",
    "rh.volume.fwhm10.fsaverage.mgh",
    "rh.area.fwhm10.fsaverage.mgh",
    "rh.w_g.pct.mgh.fwhm10.fsaverage.mgh"
]

# Function to get the full paths based on filename
def get_file_paths(filename):
    if filename.endswith("_ses-F09"):
        base_path = "/projects/0/einf1049/scratch/jouyang/GenR_mri/f09_mri"
    elif filename.endswith("_ses-F13"):
        base_path = "/projects/0/einf1049/scratch/jouyang/GenR_mri/f013_mri"
    else:
        base_path = "/projects/0/einf1049/scratch/jouyang/ABCD_mri"

    # Constructing the full paths for all desired files
    paths = {}
    for file in files_to_retrieve:
        paths[file] = os.path.join(base_path, filename, "surf", file)

    return paths
#
# Apply the function to each filename to get paths
subset_phenotype_ids['file_paths'] = subset_phenotype_ids['filename'].apply(get_file_paths)
# Display the results
print(subset_phenotype_ids)

# print(df['file_paths'][0]["lh.w-g.pct.mgh.fwhm10.fsaverage.mgh"])



"""
Data analysis for 4 brain related features (on fs7)
"""


# def brain_SOI_matrix_raw(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_half):
#     # load the cortex label file, which is used for select the vertices.
#     # left half brain: 149955, right half brain: 149926
#     rh = open("/home/jouyang/GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r")
#     # data is the raw data 
#     data = rh.read().splitlines()
#     # data_truc is the raw data without the header
#     data_truc = data[2:]
#     # This is the Longitude transform for one vertex of a person 
#     longitude_mapping_per = get_longitudinal_map_each(data_truc)
#     # This is the ij-2d grid for one vertex of a person, raduis = 100
#     _, origin_ij_grid = sphere_to_grid_each(longitude_mapping_per,100)
#     # maintain an indexing array ij_id
#     ij_id = origin_ij_grid[:,0]
#     ID_per_half = ij_id.astype('int')

#     # load MRI files 
#     thickness_array_re = SOI_array_per(ID_per_half, thick_path)  # thickness in [0, 4.37891531]
#     volume_array_re = SOI_array_per(ID_per_half, volume_path)   # volume in [0, 5.9636817]
#     SA_array_re = SOI_array_per(ID_per_half, SA_path)   # surface_area in [0, 1.40500367]
#     w_g_array_re = SOI_array_per(ID_per_half, w_g_pct_path) # w/g ratio in [0, 48.43599319]

#     # stack them as a matrix
#     SOI_mx = np.stack([thickness_array_re, volume_array_re, SA_array_re, w_g_array_re], axis=-1)

#     return SOI_mx

# 
lh = open("/home/jouyang/GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r")
# data is the raw data 
data = lh.read().splitlines()
# data_truc is the raw data without the header
data_truc = data[2:]
# This is the Longitude transform for one vertex of a person 
longitude_mapping_per = get_longitudinal_map_each(data_truc)
# This is the ij-2d grid for one vertex of a person, raduis = 100
origin_ij_list, origin_ij_grid = sphere_to_grid_each(longitude_mapping_per,100)
# maintain an indexing array ij_id
ij_id = origin_ij_grid[:,0]
# Deleting two unknown elements
indices_to_delete = [93576,93577]
ij_id = np.delete(ij_id, indices_to_delete)
ID_per_half = ij_id.astype('int')
input_data = []
for index, row in subset_phenotype_ids.iterrows():
    paths_dict = row['file_paths']

    # Extract paths based on the filename-parameter mapping
    thick_path = paths_dict["lh.thickness.fwhm10.fsaverage.mgh"]
    volume_path = paths_dict["lh.volume.fwhm10.fsaverage.mgh"]
    SA_path = paths_dict["lh.area.fwhm10.fsaverage.mgh"]
    w_g_pct_path = paths_dict["lh.w_g.pct.mgh.fwhm10.fsaverage.mgh"]

    # Call the function
    input_data.append(brain_SOI_matrix(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_half))
input_tensor = torch.tensor(input_data, dtype=torch.float32)
#  DONT RUN, DONE, make all sample left brain MRI SOI to tensor data
np.save('sample_input_tensor_quantile_transform_pct01.npy',input_tensor)

