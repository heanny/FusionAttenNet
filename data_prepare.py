"""
Data preparation is finished in this file, which including the GenR and ABCD data aggregation,
the QC step, and the training, validation, and testing datasets split. 

GenR:
* for twins/triplets-like data: we exclude the siblings
- brain imaging data (SOI): the volume, surface area, thickness, and W/G ratio.
- phenotype features data: age, sex, CBCL scores, ecnomics, ESA.
    - CBCL score: 

ABCD:
* for twins/triplets-like data: we randomly include one of the siblings
- brain imaging data (SOI): the volume, surface area, thickness, and W/G ratio.
- phenotype features data: age, sex, CBCL scores, ...?
"""
#%% load packages
import pandas as pd
import numpy as np
import pyreadr
import pyreadstat
import random
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
from helper_func_prep import SOI_array_per_left,SOI_array_per_fs5, min_max_normalize, plot_SOI, robust_scale_normalize, quantile_normalize, log_transform_and_scale

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
#%% load the GenR core data
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


# %% GenR CBCL, and sex & edu_level data 
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

# GenR sex & edu_level on focus 13 with overlap
# genr_sex_n_edul_f9_incl = genr_sex_n_edul_raw.loc[genr_sex_n_edul_raw["idc"].isin(list(genr_core_f9_all['idc'].astype('int').astype('str')))].reset_index(drop=True) # 3185 rows × 121 columns
# genr_sex_n_edul_f13_incl = genr_sex_n_edul_raw.loc[genr_sex_n_edul_raw["idc"].isin(list(genr_core_f13_all['idc'].astype('int').astype('str')))].reset_index(drop=True) # 2165 rows × 121 columns

# %% load the GenR data including sex, edu_level, CBCL syndrome scores (attention, aggresive behavior)
# FIXME: GenR: check the locale, the measurement formatting, (1.0000 vs 1,0000), english vs dutch

# %% QC for GenR other phenotype data
# another consent variable: FUPFASE3_9/FUPFASE4_13 should be "yes" or "1". Checked, all 1.0, even for data with overlap. 

# siblings / twins (duplicates "mother") (when $multiple =1, if multiple, would be siblings/twins...) simply keep one the duplicates.
#       `CHILD-ALLGENERALDATA_24102022.sav`
# random select one of the siblings at focus 9 and 13
print(genr_sex_n_edul_f9['mother'].duplicated().sum(), genr_sex_n_edul_f13['mother'].duplicated().sum())
genr_sex_n_edul_f9 = genr_sex_n_edul_f9.drop_duplicates(subset=['mother']) # 1893 rows × 121 columns, remove duplicates: 63
genr_sex_n_edul_f13 = genr_sex_n_edul_f13.drop_duplicates(subset=['mother']) # 2090 rows × 121 columns, remove duplicates: 75

# include the overlap: random select one of the siblings at focus 9 and 13 
# print(genr_sex_n_edul_f9_incl['mother'].duplicated().sum(), genr_sex_n_edul_f13_incl['mother'].duplicated().sum())
# genr_sex_n_edul_f9_incl = genr_sex_n_edul_f9_incl.drop_duplicates(subset=['mother']) # duplicates: 153, 3032 rows × 121 columns
# print(genr_sex_n_edul_f13_incl['mother'].duplicated().sum())
# genr_sex_n_edul_f13_incl = genr_sex_n_edul_f13_incl.drop_duplicates(subset=['mother']) # duplicates: 75, 2090 rows × 121 columns


# %% # don't run - GenR: age diffent at MRI and CBCl, I choose the age at MRI assessment took place.
#       - Child age at assessment at @9 (“age_child_mri_f09”)
#       - Child age at assessment at @13 ("age_child_mri_f13")
#       - `genr_mri_core_data_20220311.rds`
#
#       - Child age at assessment (“agechild_cbcl9m") `CHILDCBCL9_incl_Tscores_20201111.sav`
#       - Child age at assessment ("agechild_gr1093")`  GR1093-E1_CBCL_18062020.sav`
#       - what the diff between these ages? No, we still need to use the age of general data, because that is when 
#         the MRI took place, but the ADHD syndromes may or may not show at that age, even when they took CBCL test. 
# using Spearman test to check their coorelation, because they are not normally distributes.
"""
from scipy.stats import shapiro
stat, p = shapiro(age_9_CBCL['agechild_cbcl9m'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
 print('Sample looks Gaussian (fail to reject H0)')
else:
 print('Sample does not look Gaussian (reject H0)')
"""
import scipy.stats
age_9_CBCL = genr_cbcl_f9[['idc','agechild_cbcl9m']].dropna().reset_index(drop=True) 
age_9_general = genr_core_f9[['idc', 'age_child_mri_f09']]
age_9_general = age_9_general[age_9_general.idc.isin(list(age_9_CBCL['idc'])) == True].reset_index(drop=True)
# because these two datasets are not normally distributed (), so we chose spearman coorelation test. 
print(scipy.stats.spearmanr(age_9_CBCL['agechild_cbcl9m'], age_9_general['age_child_mri_f09']))
# (correlation=0.4275542306521193, pvalue=3.5526965993273696e-74) -> (correlation r, and p-value)
# r > 0 meaning it's a positive coorelation with p << 0.05, which means r is statistically significant.
# so they are modest or moderate correlations, and based on the reason 1 (have NAN in CBCL age), 
# and we then limit the difference of interview (MRI) age and CBCL test age within 6 months (0.5 year old)
age_9_general['diff'] = age_9_general['age_child_mri_f09'] - age_9_CBCL['agechild_cbcl9m']
age_9_general = age_9_general.loc[age_9_general['diff']<=0.5].reset_index(drop=True)
age_9_CBCL = age_9_CBCL[age_9_CBCL.idc.isin(list(age_9_general['idc']))== True].reset_index(drop=True)
print(scipy.stats.spearmanr(age_9_CBCL['agechild_cbcl9m'], age_9_general['age_child_mri_f09']))
# SpearmanrResult(correlation=0.7966206143743338, pvalue=7.328650298043946e-284), this is the very high correlation. (r>0.79)
# we remove 666 rows, -> 1290 rows
# below are for focus 13
age_13_CBCL = genr_cbcl_f13[['idc','agechild_gr1093']].dropna().reset_index(drop=True) 
age_13_general = genr_core_f13[['idc', 'age_child_mri_f13']]
age_13_general = age_13_general[age_13_general.idc.isin(list(age_13_CBCL['idc'])) == True].reset_index(drop=True)
print(scipy.stats.spearmanr(age_13_CBCL['agechild_gr1093'], age_13_general['age_child_mri_f13']))
# SpearmanrResult(correlation=0.5308281950034771, pvalue=1.766172206272165e-141), moderate correlation
age_13_general['diff'] = age_13_general['age_child_mri_f13'] - age_13_CBCL['agechild_gr1093']
age_13_general = age_13_general.loc[age_13_general['diff']<=0.5].reset_index(drop=True)
age_13_CBCL = age_13_CBCL[age_13_CBCL.idc.isin(list(age_13_general['idc']))== True].reset_index(drop=True)
print(scipy.stats.spearmanr(age_13_CBCL['agechild_gr1093'], age_13_general['age_child_mri_f13']))
# SpearmanrResult(correlation=0.8002431823647453, pvalue=2.507760234734031e-292), very high coorelation
# we remove 857 rows, -> 1308 rows


# %% select GenR phenotype data
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



#%%  GenR: sex assigned at birth ("gender"), education$05 (check no missing data) or ("educm5")
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
# %% nonono: including the overlap at Focus 13: select GenR phenotype data
# GenR: age diffent at MRI and CBCl, I choose the age at MRI assessment took place.
#       - Child age at assessment at @9 (“age_child_mri_f09”)
#       - Child age at assessment at @13 ("age_child_mri_f13")
#       - `genr_mri_core_data_20220311.rds`
#       - range at focus 9: 8.851471594798085 - 11.991786447638603
#       - range at focus 13: 12.591375770020534 - 16.61327857631759
# genr_age_9_incl = genr_core_f9_all[['idc','age_child_mri_f09']] #3185 rows × 2 columns, no nan
# genr_age_13_incl = genr_core_f13_all[['idc','age_child_mri_f13']] #2165 rows × 2 columns, no nan
#  GenR: sex assigned at birth ("gender"), education$05 (check no missing data) or ("educm5")
# `CHILD-ALLGENERALDATA_24102022.sav`
#  Gender: 1:..., 2:... (1 = boy, 2 = girl)
#  educm5: 0: no education finished, 1: primary, 2: secondary, phase 1, 3: secondary, phase 2, 4: higher, phase 1; 5: higher, phase 2
# genr_sex_n_edul_9_incl = genr_sex_n_edul_f9_incl[['idc', 'gender','educm5']] #3032 rows × 3 columns, nan: educm5: 391
# genr_sex_n_edul_13_incl = genr_sex_n_edul_f13_incl[['idc', 'gender', 'educm5']] #2090 rows × 3 columns, nan: educm5: 276
# remove the nan
# genr_sex_n_edul_9_incl = genr_sex_n_edul_9_incl.dropna() #2641 rows × 3 columns
# genr_sex_n_edul_13_incl = genr_sex_n_edul_13_incl.dropna() #1814 rows × 3 columns
# GenR: CBCL: 
#       - @age 9: Attention problems ("sum_att_9m") and Aggression problems ("sum_agg_9m") from `CHILDCBCL9_incl_Tscores_20201111.sav`
#       - @age 13: Attention problems ("sum_att_14") and Aggression problems ("sum_agg_14") from `GR1093-E1_CBCL_18062020.sav`
#       - range: @9: attention: 0.0 - 18.0, aggresive: 0.0 - 28.0
#       - range: @13: attention: 0.0 - 18.0, aggresive: 0.0 - 34.0
# genr_cbcl_9_incl = genr_cbcl_f9_incl[['idc', 'sum_att_9m', 'sum_agg_9m']] #3185 rows × 3 columns, nan: sum_att_9m: 479, sum_agg_9m: 481
# genr_cbcl_13_incl = genr_cbcl_f13_incl[['idc', 'sum_att_14', 'sum_agg_14'] ]# 2165 rows × 3 columns, nan: sum_att_14: 259, sum_agg_14: 259
# remove the nan
# genr_cbcl_9_incl = genr_cbcl_9_incl.dropna() #2700 rows × 3 columns
# genr_cbcl_13_incl = genr_cbcl_13_incl.dropna() #1905 rows × 3 columns

# merge the GenR phenotype data
# genr_merge_9_incl = genr_age_9_incl.merge(genr_sex_n_edul_9_incl, on='idc').merge(genr_cbcl_9_incl, on='idc') #2362 rows × 6 columns
# genr_merge_13_incl = genr_age_13_incl.merge(genr_sex_n_edul_13_incl, on='idc').merge(genr_cbcl_13_incl, on='idc') #1659 rows × 6 columns



"""
ABCD phenotype data origanize

We do the QC on six .txt files for different QC conditions, and we inspect the structure and range of the ABCD phenotype data.
The final usable dataframe is abcd_phenotype_rename' with 9511 rows * 7 columns (as below). 
(subjectkey, cbcl_scr_syn_attention_r, cbcl_scr_syn_aggressive_r, interview_age, sex. demo_prnt_ed_v2_l) 

"""



#%% Read all ABCD_txt files
abcd_cbcl_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_cbcls01.txt", sep='\t')
abcd_qc_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_imgincl01.txt", sep='\t')
abcd_sex_age_edu_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_lpds01.txt", sep='\t')
abcd_incfind_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_mrfindings02.txt", sep='\t')
abcd_icv_mri_txt = pd.read_csv("/home/jouyang/ABCD_txt/abcd_smrip10201.txt", sep='\t') 
abcd_siblings_txt = pd.read_csv("/home/jouyang/ABCD_txt/acspsw03.txt", sep='\t')

# %% select the core data of interests
abcd_cbcl = abcd_cbcl_txt[['subjectkey', 'cbcl_scr_syn_attention_r', 'cbcl_scr_syn_aggressive_r']] # missing#: 2473, df.dropna()
abcd_qc = abcd_qc_txt[['subjectkey','imgincl_t1w_include', 'eventname']] # qc ==1 & !is.na(qc): qc==1 and df.dropna(), nomissing but qc=0 exist, we have #19097 in the end.
abcd_sex_age_edu = abcd_sex_age_edu_txt[['subjectkey','interview_age', 'sex', 'demo_prnt_ed_v2_l']] # NA check in sex, age, edu, df.dropna(), 28560 missing in edu, we have 11207 then.
abcd_incfind = abcd_incfind_txt[['subjectkey', 'mrif_score']] # exclude IF: set mrif_score = 1 or 2, we then have 18759 rows
abcd_icv_mri = abcd_icv_mri_txt[['subjectkey', 'smri_vol_scs_intracranialv']] # no missing 
abcd_siblings = abcd_siblings_txt[['subjectkey', 'rel_family_id']] # only one at one family: !duplicated(), we have 10700. 

print(len(abcd_cbcl), len(abcd_qc), len(abcd_sex_age_edu), len(abcd_incfind), len(abcd_icv_mri), len(abcd_siblings))
# %% first do the NaN drop and QC control
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

# %% then merge to a whole one, meaning simply find the common set for all these datasets above
abcd_merge = abcd_cbcl.merge(abcd_qc, on='subjectkey').merge(abcd_sex_age_edu, on='subjectkey').merge(abcd_incfind, on='subjectkey').merge(abcd_icv_mri, on='subjectkey').merge(abcd_siblings, on='subjectkey')
# print('merge1', len(abcd_merge))
abcd_merge = abcd_merge.dropna()
# print('merge2', len(abcd_merge))
abcd_merge = abcd_merge.drop_duplicates(subset=['subjectkey']) # merge: 9511
print('merge3', len(abcd_merge))

# %% ABCD age (interview_age), `abcd_lpds01`
# what age do you want for ABCD? (116 to 149 months)
abcd_merge['interview_age'] = abcd_merge['interview_age'].astype('int')
abcd_age_max_mons = abcd_merge['interview_age'].max()/12 #12.42 yrs
abcd_age_min_mons = abcd_merge['interview_age'].min()/12 #9.67 yrs
# ABCD: BIO sex (sex), `abcd_lpds01`, M = Male; F = Female; 
print(abcd_age_max_mons, abcd_age_min_mons)

# %% ABCD: education level of parents 
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
# %% ABCD: CBCL
#       - Attention problems (cbcl_scr_syn_attention_r) and Aggressive problems (cbcl_scr_syn_aggressive_r) 
#       - from `abcd_cbcls01`
abcd_merge[['cbcl_scr_syn_attention_r', 'cbcl_scr_syn_aggressive_r']] = abcd_merge[['cbcl_scr_syn_attention_r', 'cbcl_scr_syn_aggressive_r']].astype('float')
print(abcd_merge['cbcl_scr_syn_attention_r'].max()) #19.0
print(abcd_merge['cbcl_scr_syn_attention_r'].min()) #0.0
print(abcd_merge['cbcl_scr_syn_aggressive_r'].max()) #33.0
print(abcd_merge['cbcl_scr_syn_aggressive_r'].min()) #0.0

# %% ABCD dataset for phenotype (subjectkey, cbcl_scr_syn_attention_r, cbcl_scr_syn_aggressive_r)
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
# %% merge abcd and genr dataset
abcd_pheno_preprocess = abcd_phenotype_rename.copy()
abcd_pheno_preprocess['sex'].replace(['F','M'], [0, 1],inplace=True) # female = 0, male = 1
abcd_pheno_preprocess['age'] = np.around(abcd_pheno_preprocess['age']/12, decimals=2) # round the age to years with 2 decimals
# abcd maternal edu level : 0, 1, 2.
abcd_pheno_preprocess['edu_maternal'] = np.where(abcd_pheno_preprocess['edu_maternal'] < 15, 0, abcd_pheno_preprocess['edu_maternal'])
abcd_pheno_preprocess['edu_maternal'] = np.where((abcd_pheno_preprocess['edu_maternal'] >= 15) & (abcd_pheno_preprocess['edu_maternal'] < 18), 1, abcd_pheno_preprocess['edu_maternal'])
abcd_pheno_preprocess['edu_maternal'] = np.where(abcd_pheno_preprocess['edu_maternal'] >= 18, 2, abcd_pheno_preprocess['edu_maternal'])
abcd_pheno_preprocess['idc'] = abcd_pheno_preprocess['idc'].str.replace('_', '') # 9511 rows × 7 columns
abcd_pheno_preprocess_ = abcd_pheno_preprocess.drop(columns=['eventname']) #9511 rows × 6 columns
# %%
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
# %% dont run!!! save the ids as .txt files
abcd_pheno_idc = abcd_pheno_preprocess[['idc','eventname']].copy()
abcd_pheno_idc_followup = abcd_pheno_idc.loc[abcd_pheno_idc['eventname']=='2_year_follow_up_y_arm_1'] # 3730 rows × 2 columns
abcd_pheno_idc_followup['filename'] = abcd_pheno_idc_followup['idc'].apply(lambda x: f'sub-{x}_ses-2YearFollowUpYArm1')
abcd_pheno_idc_baseline = abcd_pheno_idc.loc[abcd_pheno_idc['eventname']=='baseline_year_1_arm_1'] # 5781 rows × 2 columns
abcd_pheno_idc_baseline['filename'] = abcd_pheno_idc_baseline['idc'].apply(lambda x: f'sub-{x}_ses-baselineYear1Arm1')
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
# %% rename the filename for locating the MRi files
genr_f09_pheno_idc = genr_merge_9_[['idc']].copy()
genr_f09_pheno_idc['filename'] = genr_f09_pheno_idc['idc'].apply(lambda x: f'sub-{x}_ses-F09') # 1429: genr focus 9
genr_f09_pheno_idc['filename'].to_csv('ids_list_genr_f09.txt', index=False, header=False)

genr_f13_pheno_idc = genr_merge_13_[['idc']].copy()
genr_f13_pheno_idc['filename'] = genr_f13_pheno_idc['idc'].apply(lambda x: f'sub-{x}_ses-F13') # 1659: genr focus 13
genr_f13_pheno_idc['filename'].to_csv('ids_list_genr_f13.txt', index=False, header=False) 
# %% DOnt't !!only run it once! copy the SOR mri file in my directory
def copy_file_if_newer(src, dst):
    if not os.path.exists(dst) or os.stat(src).st_mtime - os.stat(dst).st_mtime > 1:
        shutil.copy2(src, dst)
        return True
    return False

def find_and_copy_files(src_dir, dest_dir, ids_list):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    total_dirs = len(ids_list)
    
    # Iterate over all IDs and check if a directory with that ID exists in the source directory
    for idx, directory_id in enumerate(ids_list, start=1):
        src_path = os.path.join(src_dir, directory_id)
        dest_path = os.path.join(dest_dir, directory_id)
        
        # If the directory exists in the source
        if os.path.isdir(src_path):
            if not os.path.exists(dest_path):
                # If destination doesn't exist, copy the whole directory
                shutil.copytree(src_path, dest_path)
                print(f"[{idx}/{total_dirs}] {directory_id} copied successfully!")
            else:
                # If destination exists, copy only new or updated files
                updated = False
                for root, dirs, files in os.walk(src_path):
                    rel_path = os.path.relpath(root, src_path)
                    dest_root = os.path.join(dest_path, rel_path)
                    if not os.path.exists(dest_root):
                        os.makedirs(dest_root)
                    for file in files:
                        src_file = os.path.join(root, file)
                        dest_file = os.path.join(dest_root, file)
                        if copy_file_if_newer(src_file, dest_file):
                            updated = True
                if updated:
                    print(f"[{idx}/{total_dirs}] {directory_id} updated!")
                else:
                    print(f"[{idx}/{total_dirs}] {directory_id} already up to date.")
        else:
            print(f"[{idx}/{total_dirs}] {directory_id} not found in {src_dir}.")

# def find_and_copy_files(src_dir, dest_dir, ids_list):
#     # Ensure the destination directory exists
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)

#     total_dirs = len(ids_list)
    
#     # Iterate over all IDs and check if a directory with that ID exists in the source directory
#     for idx, directory_id in enumerate(ids_list, start=1):
#         src_path = os.path.join(src_dir, directory_id)
        
#         # If the directory exists, copy it to the destination directory
#         if os.path.isdir(src_path):
#             dest_path = os.path.join(dest_dir, directory_id)
#             shutil.copytree(src_path, dest_path)
#             print(f"[{idx}/{total_dirs}] success!")
#         else:
#             print(f"[{idx}/{total_dirs}] {directory_id} not found in {src_dir}.")

# %% DON't RUN, already done
# copy abcd mri files - in total abcd mri 9509 files, 5781 for base
src_diry = "/projects/0/einf1049/data/abcd/rel4.0/bids/derivatives/freesurfer/6.0.0/untar/"
dest_diry = "/home/jouyang/ABCD_mri" 
with open('ids_list_abcd_base.txt', 'r') as f:
    ids = [line.strip() for line in f]
find_and_copy_files(src_diry, dest_diry, ids)
# %% DON't RUN, already done
# copy abcd followup files - 3730 files
src_diry = "/projects/0/einf1049/data/abcd/rel4.0/bids/derivatives/freesurferclear/6.0.0/untar/"
dest_diry = "/home/jouyang/ABCD_mri"
with open('ids_list_abcd_folup.txt', 'r') as f:
    ids = [line.strip() for line in f]
find_and_copy_files(src_diry, dest_diry, ids)
# %% DON'T RUN!! Already done
# copy genr f09 files - 1429 files
src_diry = "/projects/0/einf1049/data/GenR_MRI/bids/derivatives/freesurfer/6.0.0/qdecr/"
dest_diry = "/home/jouyang/GenR_mri/f09_mri"
with open('ids_list_genr_f09.txt', 'r') as f:
    ids = [line.strip() for line in f]
find_and_copy_files(src_diry, dest_diry, ids)
# %% DON'T RUN!! Already done
# copy genr f13 files - 1659 files
src_diry = "/projects/0/einf1049/data/GenR_MRI/bids/derivatives/freesurfer/6.0.0/qdecr/"
dest_diry = "/home/jouyang/GenR_mri/f013_mri"
with open('ids_list_genr_f13.txt', 'r') as f:
    ids = [line.strip() for line in f]
find_and_copy_files(src_diry, dest_diry, ids)
"""
copy MRI data result (missing two from 'phenotype_data' in total, originally is 12599 rows)
Genr: sub-2363_ses-F13 (genrf13) Not found!
abcd: based on 'eventname', we use the corresponding MRI files, idc=NDARMC003PZF (abcd_base) Not found!
Now in total: we have 1429(genrf9) + 1658(genrf13) + 5780(abcd_base) + 3730(abcd_followup) = 12597 MRI scans!!
"""


# %% remove the ids of phenotype data that does not have the corresponding MRI
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

# sub-NDARINVE7869241_ses-2YearFollowUpYArm1, does not have the corresponding MRI, abcd_followup
# sub-NDARINVR1927JG7_ses-2YearFollowUpYArm1, does not have the corresponding MRI, abcd_followup
# Genr: sub-2363_ses-F13 Not found!
# abcd: based on 'eventname', we use the corresponding MRI files, idc=NDARMC003PZF Not found!

cleaned_list = ['2363', 'NDARMC003PZF', 'NDARINVE7869241', 'NDARINVR1927JG7'] # these are ids with no MRI (no baseline and followup year) in abcd
# remove the corresponding phenotype data using baseline data of ABCD
print(len(phenotype_data)) # original 12599 rows × 6 columns
# phenotype_data = phenotype_data.loc[phenotype_data['idc'] != ['2363', 'NDARMC003PZF']]
# print(len(phenotype_data))
phenotype_data = phenotype_data[~phenotype_data['idc'].isin(cleaned_list)].reset_index(drop=True) #12595 rows × 6 columns
print(len(phenotype_data)) # 12595

#in total: we have 1429(genrf9) + 1658(genrf13) + 5780(abcd_base) + 3728(abcd_followup) = 12595 MRI scans!!

"""
 analysis the CBCL two scores and age statistics
 
"""

# %% analysis the CBCL two scores and age statistics
phenotype_data[["sum_agg","sum_att", "age"]].describe()

# %% plot the distribution of CBCL two scores and age
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
# %% dont run!!
output_dir = 'output_plots_all_phenotypes_analysis_raw'  # Directory to save plots
os.makedirs(output_dir, exist_ok=True)
# %% plot age distrubution
plt.figure(figsize=(10, 6))
sns.histplot(data=phenotype_data[["age"]], kde=True, bins=100)
plt.title(f"Age Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, f"age.png"))
# plt.show()
plt.close()
# %% plot sex distrubution
plt.figure(figsize=(10, 6))
sns.histplot(data=phenotype_data[["sex"]], kde=True, bins=100)
plt.title(f"Sex Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, f"sex.png"))
# plt.show()
plt.close()

# %% plot Maternal educational level distrubution
plt.figure(figsize=(10, 6))
sns.histplot(data=phenotype_data[["edu_maternal"]], kde=True, bins=100)
plt.title(f"Maternal educational level Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, f"maternal educational level.png"))
# plt.show()
plt.close()

"""

Phenotype data normalization:
- Min-max normalization on age 
- Log(x+1) transformation for processing 0 values and the min-max on two CBCL scores

"""
#%%
import numpy as np

# %% 
def log_transform_and_scale(x):
    # log transformation
    x_log = np.log1p(x)  # log1p is log(x + 1)
    # Min-Max 
    return min_max_normalize(x_log)

# %% load data
phenotype_data[["sum_agg","sum_att", "age"]]
age = phenotype_data["age"] # 你的年龄数据
cbcl_aggressive = phenotype_data["sum_agg"]# CBCL积极行为分数
cbcl_attention = phenotype_data["sum_att"] # CBCL注意力问题分数

# 对年龄进行 Min-Max 缩放
age_scaled = min_max_normalize(age)
# 对 CBCL 分数进行对数转换后的 Min-Max 缩放
cbcl_aggressive_transformed = log_transform_and_scale(cbcl_aggressive)
cbcl_attention_transformed = log_transform_and_scale(cbcl_attention)



phenotype_data['sum_agg'] = min_max_normalize(phenotype_data["age"])
phenotype_data['sum_att'] = log_transform_and_scale(phenotype_data["sum_agg"])
phenotype_data['age'] = log_transform_and_scale(phenotype_data["sum_att"])
# %% 打印一些统计信息以检查转换结果
phenotype_data[["sum_agg","sum_att", "age"]].describe()

# %%
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
# %% DONT RUN, already done. run once. make a subset for the small model
subset_phenotype_idx = random.sample(range(12595), 1000) # 1429(genrf9) + 1657(genrf13) + 5780(abcd_base) + 3730(abcd_followup) = 12596
# save the index of samples to .txt
with open('sample_index.txt', 'w') as f:
    f.write('\n'.join(map(str, subset_phenotype_idx)))



# %% load the sample dataset index
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
np.save('sample_filename.npy',subset_phenotype_ids['filename'])

# %% Don't RUN, Have done it ONCE!!
#FIXME: we remove the idc columns?? then have saved it 
subset_phenotype = subset_phenotype.drop(columns=['eventname','idc']).reset_index(drop=True)
# make the dataframes to tensors
subset_phenotype['sum_att'] = min_max_normalize(subset_phenotype['sum_att'])
subset_phenotype['sum_agg'] = min_max_normalize(subset_phenotype['sum_agg'])
subset_phenotype_tensor = torch.tensor(subset_phenotype.values, dtype=torch.float32)


# %% Convert pandas series to numpy arrays
# float_cols = ['sum_att','sum_agg','age']
# # Convert category columns to integers
# category_cols = ['sex', 'edu_maternal']
# # Convert pandas series to PyTorch tensors and store in lists
# float_tensors = [torch.tensor(subset_phenotype[col].values, dtype=torch.float32) for col in float_cols]
# category_tensors = [torch.tensor(subset_phenotype[col].values, dtype=torch.int64) for col in category_cols]
# # For operations involving only float columns:
# float_data = torch.stack(float_tensors, dim=-1)

# # For operations involving only categorical columns:
# category_data = torch.stack(category_tensors, dim=-1)
# # Concatenate tensors along the second dimension (columns)
# combined_tensor = torch.cat([torch.tensor(float_data), torch.tensor(category_data, dtype=torch.float32)], dim=-1)

# print(combined_tensor)

# Create a TensorDataset and DataLoader
# %% function for brain MRI SOI dataset
def brain_SOI_matrix(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_half):
    # load the cortex label file, which is used for select the vertices.
    # left half brain: 149955, right half brain: 149926
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
    ID_per_half = ij_id.astype('int')

    # load MRI files 
    thickness_array_re = SOI_array_per_left(ID_per_half, thick_path)  # thickness in [0, 4.37891531]
    volume_array_re = SOI_array_per_left(ID_per_half, volume_path)   # volume in [0, 5.9636817]
    SA_array_re = SOI_array_per_left(ID_per_half, SA_path)   # surface_area in [0, 1.40500367]
    w_g_array_re = SOI_array_per_left(ID_per_half, w_g_pct_path) # w/g ratio in [0, 48.43599319]

    # min-max normalize the SOI data
    # thickness_mx_norm = min_max_normalize(thickness_array_re)
    # volume_mx_norm = min_max_normalize(volume_array_re)
    # SA_mx_norm = min_max_normalize(SA_array_re)
    # w_g_ar_norm = w_g_array_re/100
    
    # robust scaler normalization the SOI data
    # thickness_mx_norm = robust_scale_normalize(thickness_array_re)
    # volume_mx_norm = robust_scale_normalize(volume_array_re)
    # SA_mx_norm = robust_scale_normalize(SA_array_re)
    # w_g_ar_norm = robust_scale_normalize(w_g_array_re)

    # quantile normalization the SOI data
    thickness_mx_norm = quantile_normalize(thickness_array_re)
    volume_mx_norm = quantile_normalize(volume_array_re)
    SA_mx_norm = quantile_normalize(SA_array_re)
    w_g_ar_norm = quantile_normalize(w_g_array_re)

    # stack them as a matrix
    SOI_mx_minmax = np.stack([thickness_mx_norm, volume_mx_norm, SA_mx_norm, w_g_ar_norm], axis=-1)

    return SOI_mx_minmax

# %% test brain_SOI_matrix function for on person
# thick_path = '/home/jouyang/ABCD_mri/sub-NDARINV7VL87WNF_ses-baselineYear1Arm1/surf/lh.thickness.fwhm10.fsaverage.mgh'
# volume_path = '/home/jouyang/ABCD_mri/sub-NDARINV7VL87WNF_ses-baselineYear1Arm1/surf/lh.volume.fwhm10.fsaverage.mgh'
# SA_path = '/home/jouyang/ABCD_mri/sub-NDARINV7VL87WNF_ses-baselineYear1Arm1/surf/lh.area.fwhm10.fsaverage.mgh'
# w_g_pct_path = '/home/jouyang/ABCD_mri/sub-NDARINV7VL87WNF_ses-baselineYear1Arm1/surf/lh.w_g.pct.mgh.fwhm10.fsaverage.mgh'

# # load the cortex label file, which is used for select the vertices.
# # left half brain: 149955, right half brain: 149926
# lh = open("/home/jouyang/GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r")
# # data is the raw data 
# data = lh.read().splitlines()
# # data_truc is the raw data without the header
# data_truc = data[2:]
# # This is the Longitude transform for one vertex of a person 
# longitude_mapping_per = get_longitudinal_map_each(data_truc)
# # This is the ij-2d grid for one vertex of a person, raduis = 100
# origin_ij_list, origin_ij_grid = sphere_to_grid_each(longitude_mapping_per,100)
# # maintain an indexing array ij_id
# ij_id = origin_ij_grid[:,0]
# ID_per_half = ij_id.astype('int')

# # test for per person
# SOI_mx_minmax = brain_SOI_matrix(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_half)
# print(SOI_mx_minmax.shape)
# plot_SOI(SOI_mx_minmax)

#%% got the path of each person's brain MRI
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
        base_path = "/home/jouyang/GenR_mri/f09_mri"
    elif filename.endswith("_ses-F13"):
        base_path = "/home/jouyang/GenR_mri/f013_mri"
    else:
        base_path = "/home/jouyang/ABCD_mri"

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
np.save('sample_ids_filename.npy',subset_phenotype_ids)

# print(df['file_paths'][0]["lh.w-g.pct.mgh.fwhm10.fsaverage.mgh"])

# %% DONT RUN, DONE, make all sample left brain MRI SOI to tensor data
# load the cortex label file, which is used for select the vertices.
# left half brain: 149955, right half brain: 149926
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
"""
Here we didn't do for left and right brain. 
Because 149953 is a prime number, and right brain doesn't have this problem!!
# Deleting two unknown elements, only for left brain!!!
indices_to_delete = [93576,93577]
ij_id = np.delete(ij_id, indices_to_delete) # 149953 for left, 149926 for the right

"""

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
# %% DONT RUN, DONE, make all sample left brain MRI SOI to tensor data
np.save('sample_input_tensor.npy',input_tensor)




"""
Data analysis for 4 brain related features (on fs7)
"""
# %% import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_and_plot_data_in_batches(file_path, output_dir, batch_size=100):
    """
    Analyze brain data and save distribution plots in batches.
    
    Args:
    file_path: Path to the .npy file containing the brain data
    output_dir: Directory to save the output plots
    batch_size: Number of samples to process in each batch
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data = np.load(file_path, mmap_mode='r')
    print(f"Original data shape: {data.shape}")
    
    n_samples, *dims, n_features = data.shape
    total_points = np.prod(dims)
    
    # Initialize statistics
    stats_dict = {i: {"sum": 0, "sum_sq": 0, "min": np.inf, "max": -np.inf} for i in range(n_features)}
    
    # Process data in batches
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = data[start:end].reshape((end - start) * total_points, n_features)
        
        for i in range(n_features):
            feature_data = batch[:, i]
            stats_dict[i]["sum"] += np.sum(feature_data)
            stats_dict[i]["sum_sq"] += np.sum(np.square(feature_data))
            stats_dict[i]["min"] = min(stats_dict[i]["min"], np.min(feature_data))
            stats_dict[i]["max"] = max(stats_dict[i]["max"], np.max(feature_data))
        
        # Free up memory
        del batch
    
    # Calculate final statistics
    total_count = n_samples * total_points
    for i in stats_dict:
        mean = stats_dict[i]["sum"] / total_count
        var = (stats_dict[i]["sum_sq"] / total_count) - (mean ** 2)
        stats_dict[i]["mean"] = mean
        stats_dict[i]["std"] = np.sqrt(var)
    
    # Plot and save distributions
    for i in range(n_features):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data[:, :, :, i].flatten(), kde=True, bins=100)
        plt.title(f"Feature {i+1} Distribution")
        plt.xlabel("Feature Value")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"feature_{i+1}_distribution.png"))
        plt.close()
    
    return stats_dict

def suggest_normalization(stats_dict):
    """
    Suggest normalization methods based on the data distribution for each feature.
    """
    suggestions = {}
    for feature, stats in stats_dict.items():
        if abs(stats["max"] - stats["min"]) > 10 * stats["std"]:
            suggestions[feature] = "Data range is large compared to its standard deviation. Consider using robust scaling or normalization."
        elif stats["std"] < 0.01 * abs(stats["mean"]):
            suggestions[feature] = "Data has low variance compared to its mean. Standard scaling might be appropriate."
        else:
            suggestions[feature] = "Data appears to have a reasonable spread. Standard scaling or min-max scaling could be appropriate."
    return suggestions

# Main execution
# %%
if __name__ == "__main__":
    file_path = 'all_age_raw.npy'  # Update this to your file path
    output_dir = 'output_plots_all_age_analysis_raw'  # Directory to save plots
    
    stats = analyze_and_plot_data_in_batches(file_path, output_dir)
    
    print("Data Statistics:")
    for feature, feature_stats in stats.items():
        print(f"\nFeature {feature}:")
        for stat_name, stat_value in feature_stats.items():
            print(f"  {stat_name}: {stat_value}")
    
    suggestions = suggest_normalization(stats)
    print("\nNormalization Suggestions:")
    for feature, suggestion in suggestions.items():
        print(f"Feature {feature}: {suggestion}")

    print(f"\nPlots have been saved in the '{output_dir}' directory.")


"""
Do normalization on four brain feature data

"""
# %%




"""
Let's do sample dataset for fsaverage5 on MRI

"""

# %% for fsaverage5 data
def brain_SOI_matrix_fs5(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_half):
    # load the cortex label file, which is used for select the vertices.
    # left half brain: 149955, right half brain: 149926
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
    ID_per_half = ij_id.astype('int')

    # load MRI files 
    thickness_array_re = SOI_array_per_fs5(ID_per_half, thick_path)  # thickness in [0, 4.37891531]
    volume_array_re = SOI_array_per_fs5(ID_per_half, volume_path)   # volume in [0, 5.9636817]
    SA_array_re = SOI_array_per_fs5(ID_per_half, SA_path)   # surface_area in [0, 1.40500367]
    w_g_array_re = SOI_array_per_fs5(ID_per_half, w_g_pct_path) # w/g ratio in [0, 48.43599319]

    # min-max normalize the SOI data
    # thickness_mx_norm = min_max_normalize(thickness_array_re)
    # volume_mx_norm = min_max_normalize(volume_array_re)
    # SA_mx_norm = min_max_normalize(SA_array_re)
    # w_g_ar_norm = w_g_array_re/100

    # thickness_mx_norm = robust_scale_normalize(thickness_array_re)
    # volume_mx_norm = robust_scale_normalize(volume_array_re)
    # SA_mx_norm = robust_scale_normalize(SA_array_re)
    # w_g_ar_norm = robust_scale_normalize(w_g_array_re)

    # quantile normalization the SOI data
    thickness_mx_norm = quantile_normalize(thickness_array_re)
    volume_mx_norm = quantile_normalize(volume_array_re)
    SA_mx_norm = quantile_normalize(SA_array_re)
    w_g_ar_norm = quantile_normalize(w_g_array_re)

    # stack them as a matrix
    SOI_mx_minmax = np.stack([thickness_mx_norm, volume_mx_norm, SA_mx_norm, w_g_ar_norm], axis=-1)

    return SOI_mx_minmax



# %% DONT RUN, DONE, make all sample left brain MRI SOI to tensor data
# load the cortex label file, which is used for select the vertices.
# left half brain: 149955, right half brain: 149926
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
# Deleting two unknown elements on left brain
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
    input_data.append(brain_SOI_matrix_fs5(thick_path, volume_path, SA_path, w_g_pct_path, ID_per_half))
input_tensor = torch.tensor(input_data, dtype=torch.float32)
# %% DONT RUN, DONE, make all sample left brain MRI SOI to tensor data
np.save('sample_input_tensor_fs5.npy',input_tensor)
# %% DONT RUN, save the above input tensor as a file to save time..
# Convert tensor to numpy array
# Save to .npy file
np.save('sample_input_tensor_fs5.npy',input_tensor)

# %% read this saved file to tensor
# Load data from .npy file
np.set_printoptions(precision=20) 
loaded_tensor_fs5 = np.load('sample_input_tensor_fs5.npy')
loaded_phenotype_tensor_fs5 = np.load('sample_phenotype_tensor.npy')
# Convert numpy array to PyTorch tensor
input_tensor_fs5_ = torch.tensor(loaded_tensor_fs5 , dtype=torch.float32)
subset_phenotype_tensor_fs5_ = torch.tensor(loaded_phenotype_tensor_fs5 , dtype=torch.float32)
torch.set_printoptions(precision=20)
print(input_tensor_fs5_, subset_phenotype_tensor_fs5_)


#%% CNN model



# %% new model for left half brain!
class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim):
        super(ConditionalVAE, self).__init__()
        
        self.fc1 = nn.Linear(input_dim + cond_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, latent_dim)
        self.fc32 = nn.Linear(256, latent_dim)
        self.fc4 = nn.Linear(latent_dim + cond_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, input_dim)
    # Initialize encoder weights using Xavier initialization
        # for layer in self.encoder:
        #     if isinstance(layer, nn.Linear):
        #         init.xavier_uniform_(layer.weight)
    def encode(self, x, c): # Q(z|x, c)
        x = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(x)) #hidden layer 1
        h2 = F.relu(self.fc2(h1))#hidden layer 2
        z_mu = self.fc31(h2)
        z_var = self.fc32(h2)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):  # P(x|z, c)
        z = torch.cat([z, c], dim=1)
        h3 = F.relu(self.fc4(z))#hidden layer 3
        h4 = F.relu(self.fc5(h3))#hidden layer 4
        return torch.sigmoid(self.fc6(h4))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 769*195*4), c)
        # mu, logvar = self.encode(x.view(x.size(0), -1), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = F.mse_loss(recon_x, x.view(-1, 769*195*4), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss, KLD

# %% Hyperparameters
num_phenotype_features = 5  # sum_att  sum_agg	age	sex	edu_maternal
input_dim = 769*195*4  # Adjusted for the input shape
latent_dim = 32

# %% Initialize CVAE model
cvae = ConditionalVAE(input_dim, num_phenotype_features, latent_dim)


# %% Training loop: Initialize empty lists to store loss values for each type of loss
reconstruction_loss_values = []
kl_loss_values = []
total_loss_values = []
optimizer = optim.Adam(cvae.parameters(), lr=0.001) #weight_decay=1e-5
# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    reconstruction_epoch_loss = 0.0
    kl_epoch_loss = 0.0
    total_epoch_loss = 0.0
    
    for batch_input, batch_phenotype in dataloader_sub:
        optimizer.zero_grad()
        recon_batch, z_mean, z_log_var = cvae(batch_input.float(), batch_phenotype.float())
        reconstruction_loss, kl_loss = loss_function(recon_batch, batch_input.float(), z_mean, z_log_var)
        loss = reconstruction_loss + kl_loss
        loss.backward()
        optimizer.step()
        
        reconstruction_epoch_loss += reconstruction_loss.item()
        kl_epoch_loss += kl_loss.item()
        total_epoch_loss += loss.item()
    
    reconstruction_epoch_loss /= len(dataloader_sub)
    kl_epoch_loss /= len(dataloader_sub)
    total_epoch_loss /= len(dataloader_sub)
    
    reconstruction_loss_values.append(reconstruction_epoch_loss)
    kl_loss_values.append(kl_epoch_loss)
    total_loss_values.append(total_epoch_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Reconstruction Loss: {reconstruction_epoch_loss:.4f}, KL Loss: {kl_epoch_loss:.4f}, Total Loss: {total_epoch_loss:.4f}")

# %%Plot the loss values
plt.plot(range(1, num_epochs+1), reconstruction_loss_values, label='Reconstruction Loss')
plt.plot(range(1, num_epochs+1), kl_loss_values, label='KL Loss')
plt.plot(range(1, num_epochs+1), total_loss_values, label='Total Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses")
plt.legend()
plt.grid(True)
plt.show()

# %% latent space inspection with PCA
from sklearn.decomposition import PCA
# Combine the input tensor and phenotype tensor
combined_input = torch.cat((input_tensor.view(input_tensor.size(0), -1), subset_phenotype_tensor), dim=1)
# Encode the combined input to get latent representations
with torch.no_grad():
    latent_representations = cvae.encode(combined_input)[0]
# visualize the latent representations
# use PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent_representations)
# Assuming you have a label or color for each sample

labels = [10,11,10,10,10] 
for i, label in enumerate(labels):
    plt.annotate(str(label), (latent_pca[i, 0], latent_pca[i, 1]), fontsize=10, color='black', alpha=0.7)

# plt.scatter(latent_pca[:, 0], latent_pca[:, 1], c=labels, cmap='viridis')
plt.scatter(latent_pca[:, 0], latent_pca[:, 1], cmap='viridis')
# plt.colorbar()
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent Space Visualization")
plt.show()

#%%