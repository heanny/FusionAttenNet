"""
Data preperation on the whole dataset including:

1. phenotype: @/home/jouyang/data_prepare.py (line 609 to line 637)
  > min-max on age
  > log transformation + min-max on two CBCl scores

-----------------------------------------------------------------------------------------------------------------------------------------------
【Notice】
For the age groupings of the CBCL/6-18 scale (6-11 years and 12-18 years), I recommend using rounding down rather than rounding. Here’s why:
The CBCL scale is divided according to complete age:
6-11 years old group: from 6 years old and 0 months to 11 years old and November
12-18 years old group: from 12 years old and 0 months to 18 years old and November

Example:
    If a child is 143 months old (11.916667 years old)
    Rounding up will make it 12 years old, resulting in being classified into the 12-18 year old group
    But in fact, this child is not yet 12 years old, and should belong to the 6-11 year old group


2. brain data
  > 3d-to-2d projection (512*512 images, robust scale, smoothed) @/home/jouyang/brain-3d-to-2d-projection.py

Data information:

1. phenotype + brain MRI: # 12595 
  > 1429(genrf9) + 1658(genrf13) + 5780(abcd_base) + 3728(abcd_followup) = 12595 MRI scans!!

"""


"""
1. phenotype: @/home/jouyang/data_prepare.py (line 609 to line 637)
  > min-max on age
  > log transformation + min-max on two CBCl scores
"""
# %%
#  load packages
import pandas as pd
import numpy as np
import pyreadr
import pyreadstat
import torch
import gc
import os
import matplotlib.pyplot as plt
from longitude_transform import get_longitudinal_map_each, sphere_to_grid_each, color_map_DK, plot_original, plot_DK_map
from helper_func_prep import SOI_array_per_left,SOI_array_per_right, min_max_normalize, plot_SOI, robust_scale_normalize, log_transform_and_scale
import seaborn as sns
from scipy.stats import binned_statistic_2d
from scipy.ndimage import distance_transform_edt, gaussian_filter
import shutil

"""
GenR Quality control:
* for twins/triplets-like data: we exclude the siblings
- brain imaging data (SOI): the volume, surface area, thickness, and W/G ratio.
- phenotype features data: age, sex, CBCL scores, ecnomics, ESA.
    - CBCL score
"""
#%% load the GenR core data
genr_core_f9_raw = pyreadr.read_r('/projects/0/einf1049/scratch/jouyang/GenR_core/core_f9_qc_all.RDS') #3185, after QC, access with key None
genr_core_f13_raw = pyreadr.read_r('/projects/0/einf1049/scratch/jouyang/GenR_core/core_f13_qc_all.RDS') #2165, after QC, access with key None, and they have 1229 duplicates

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


#%% GenR CBCL, and sex & edu_level data """
#load data for GenR phenotype data
genr_cbcl_f9_raw, _ = pyreadstat.read_sav('/projects/0/einf1049/scratch/jouyang/GenR_core/genr/CHILDCBCL9_incl_Tscores_20201111.sav') # 9901 rows × 650 columns
genr_cbcl_f13_raw, _ =  pyreadstat.read_sav('/projects/0/einf1049/scratch/jouyang/GenR_core/genr/GR1093-E1_CBCL_18062020.sav') # 9901 rows × 331 columns
genr_sex_n_edul_raw, _ = pyreadstat.read_sav('/projects/0/einf1049/scratch/jouyang/GenR_core/genr/CHILD-ALLGENERALDATA_24102022.sav') # 9901 rows × 121 columns

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

# GenR sex & edu_level on focus 9 and 13 without overlap
genr_sex_n_edul_f9 = genr_sex_n_edul_raw.loc[genr_sex_n_edul_raw["idc"].isin(list(genr_core_f9['idc'].astype('int').astype('str')))].reset_index(drop=True) # 1956 rows × 121 columns
genr_sex_n_edul_f13 = genr_sex_n_edul_raw.loc[genr_sex_n_edul_raw["idc"].isin(list(genr_core_f13['idc'].astype('int').astype('str')))].reset_index(drop=True) # 2165 rows × 121 columns

"""# %% load the GenR data including sex, edu_level, CBCL syndrome scores (attention, aggresive behavior)"""
# FIXME: GenR: check the locale, the measurement formatting, (1.0000 vs 1,0000), english vs dutch

# %% QC for GenR other phenotype data"""
# another consent variable: FUPFASE3_9/FUPFASE4_13 should be "yes" or "1". Checked, all 1.0, even for data with overlap. 

# siblings / twins (duplicates "mother") (when $multiple =1, if multiple, would be siblings/twins...) simply keep one the duplicates.
#       `CHILD-ALLGENERALDATA_24102022.sav`
# random select one of the siblings at focus 9 and 13
print(genr_sex_n_edul_f9['mother'].duplicated().sum(), genr_sex_n_edul_f13['mother'].duplicated().sum())
genr_sex_n_edul_f9 = genr_sex_n_edul_f9.drop_duplicates(subset=['mother']) # 1893 rows × 121 columns, remove duplicates: 63
genr_sex_n_edul_f13 = genr_sex_n_edul_f13.drop_duplicates(subset=['mother']) # 2090 rows × 121 columns, remove duplicates: 75

# %% select GenR phenotype data"""
# GenR: age diffent at MRI and CBCl, I choose the age at MRI assessment took place.
#       - Child age at assessment at @9 (“age_child_mri_f09”)
#       - Child age at assessment at @13 ("age_child_mri_f13")
#       - `genr_mri_core_data_20220311.rds`
#       - range at focus 9: 8.550308008213554 - 11.986310746064339
#       - range at focus 13: 12.591375770020534 - 16.67898699520876
genr_age_9 = genr_core_f9[['idc','age_child_mri_f09']] # 1956 rows × 2 columns
genr_age_13 = genr_core_f13[['idc','age_child_mri_f13']] # 2165 rows × 2 columns


# %%  GenR: sex assigned at birth ("gender"), education$05 (check no missing data) or ("educm5")"""
#       - see `CHILD-ALLGENERALDATA_24102022.sav`
#       - gender: 1:..., 2:...(1 is the boy, 2 is the girl)
#       - educm5: 0: no education finished, 1: primary, 2: secondary, phase 1, 3: secondary, phase 2, 4: higher, phase 1; 5: higher, phase 2
#       - categorise the edu_level: 0-1, 2-3, 4-5
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

#%% Read all ABCD_txt files"""
abcd_cbcl_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_cbcls01.txt", sep='\t')
abcd_qc_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_imgincl01.txt", sep='\t')
abcd_sex_age_edu_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_lpds01.txt", sep='\t')
abcd_incfind_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_mrfindings02.txt", sep='\t')
abcd_icv_mri_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_smrip10201.txt", sep='\t') 
abcd_siblings_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/acspsw03.txt", sep='\t')

# %% select the core data of interests"""
abcd_cbcl = abcd_cbcl_txt[['subjectkey', 'cbcl_scr_syn_attention_r', 'cbcl_scr_syn_aggressive_r']] # missing#: 2473, df.dropna()
abcd_qc = abcd_qc_txt[['subjectkey','imgincl_t1w_include', 'eventname']] # qc ==1 & !is.na(qc): qc==1 and df.dropna(), nomissing but qc=0 exist, we have #19097 in the end.
abcd_sex_age_edu = abcd_sex_age_edu_txt[['subjectkey','interview_age', 'sex', 'demo_prnt_ed_v2_l']] # NA check in sex, age, edu, df.dropna(), 28560 missing in edu, we have 11207 then.
abcd_incfind = abcd_incfind_txt[['subjectkey', 'mrif_score']] # exclude IF: set mrif_score = 1 or 2, we then have 18759 rows
abcd_icv_mri = abcd_icv_mri_txt[['subjectkey', 'smri_vol_scs_intracranialv']] # no missing 
abcd_siblings = abcd_siblings_txt[['subjectkey', 'rel_family_id']] # only one at one family: !duplicated(), we have 10700. 
print(len(abcd_cbcl), len(abcd_qc), len(abcd_sex_age_edu), len(abcd_incfind), len(abcd_icv_mri), len(abcd_siblings))

# %% first do the NaN drop and QC control"""
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

# %% then merge to a whole one, meaning simply find the common set for all these datasets above"""
abcd_merge = abcd_cbcl.merge(abcd_qc, on='subjectkey').merge(abcd_sex_age_edu, on='subjectkey').merge(abcd_incfind, on='subjectkey').merge(abcd_icv_mri, on='subjectkey').merge(abcd_siblings, on='subjectkey')
abcd_merge = abcd_merge.dropna()
abcd_merge = abcd_merge.drop_duplicates(subset=['subjectkey']) # merge: 9511
print('merge3', len(abcd_merge))

# %% ABCD age (interview_age), `abcd_lpds01`"""
abcd_merge['interview_age'] = abcd_merge['interview_age'].astype('int')
abcd_age_max_mons = abcd_merge['interview_age'].max()/12 #12.42 yrs
abcd_age_min_mons = abcd_merge['interview_age'].min()/12 #9.67 yrs
# ABCD: BIO sex (sex), `abcd_lpds01`, M = Male; F = Female; 
print(abcd_age_max_mons, abcd_age_min_mons)

# %% ABCD: education level of parents """
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
# %% ABCD: CBCL"""
#       - Attention problems (cbcl_scr_syn_attention_r) and Aggressive problems (cbcl_scr_syn_aggressive_r) 
#       - from `abcd_cbcls01`
abcd_merge[['cbcl_scr_syn_attention_r', 'cbcl_scr_syn_aggressive_r']] = abcd_merge[['cbcl_scr_syn_attention_r', 'cbcl_scr_syn_aggressive_r']].astype('float')
print(abcd_merge['cbcl_scr_syn_attention_r'].max()) #19.0
print(abcd_merge['cbcl_scr_syn_attention_r'].min()) #0.0
print(abcd_merge['cbcl_scr_syn_aggressive_r'].max()) #33.0
print(abcd_merge['cbcl_scr_syn_aggressive_r'].min()) #0.0

# %% ABCD dataset for phenotype (subjectkey, cbcl_scr_syn_attention_r, cbcl_scr_syn_aggressive_r)"""
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
# %% merge abcd and genr dataset"""
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
"""
Copy the SOI files in my folder, this can be done one-time, do not run it every time!!
"""
# seperate abcd_baseline and abcd_followup
abcd_pheno_idc = abcd_pheno_preprocess[['idc','eventname']].copy()
abcd_pheno_idc_followup = abcd_pheno_idc.loc[abcd_pheno_idc['eventname']=='2_year_follow_up_y_arm_1'] # 3730 rows × 2 columns
abcd_pheno_idc_followup['filename'] = abcd_pheno_idc_followup['idc'].apply(lambda x: f'sub-{x}_ses-2YearFollowUpYArm1')
abcd_pheno_idc_baseline = abcd_pheno_idc.loc[abcd_pheno_idc['eventname']=='baseline_year_1_arm_1'] # 5781 rows × 2 columns
abcd_pheno_idc_baseline['filename'] = abcd_pheno_idc_baseline['idc'].apply(lambda x: f'sub-{x}_ses-baselineYear1Arm1')

# seperate genr_f09 and genr_f013
genr_f09_pheno_idc = genr_merge_9_[['idc']].copy()
genr_f09_pheno_idc['filename'] = genr_f09_pheno_idc['idc'].apply(lambda x: f'sub-{x}_ses-F09') # 1429: genr focus 9
genr_f13_pheno_idc = genr_merge_13_[['idc']].copy()
genr_f13_pheno_idc['filename'] = genr_f13_pheno_idc['idc'].apply(lambda x: f'sub-{x}_ses-F13') # 1659: genr focus 13


# %% remove the ids of phenotype data that does not have the corresponding MRI and/or not found! """
# sub-NDARINVE7869241_ses-2YearFollowUpYArm1, does not have the corresponding MRI, abcd_followup
# sub-NDARINVR1927JG7_ses-2YearFollowUpYArm1, does not have the corresponding MRI, abcd_followup
# Genr: sub-2363_ses-F13 Not found!
# abcd: based on 'eventname', we use the corresponding MRI files, idc=NDARMC003PZF Not found!

cleaned_list = ['2363', 'NDARMC003PZF', 'NDARINVE7869241', 'NDARINVR1927JG7'] # these are ids with no MRI (no baseline and followup year) in abcd
# remove the corresponding phenotype data using baseline data of ABCD
print(len(phenotype_data)) # original 12599 rows × 6 columns
phenotype_data = phenotype_data[~phenotype_data['idc'].isin(cleaned_list)].reset_index(drop=True) #12595 rows × 6 columns
print(len(phenotype_data)) # after clean: 12595 rows × 6 columns

# So in total: we have 1429(genrf9) + 1658(genrf13) + 5780(abcd_base) + 3728(abcd_followup) = 12595 MRI scans!!

# %% don't run!! have done!!! 
# save the idc and filenames of all phenotypes for later
all_phenotype_ids = phenotype_data.iloc[:, :1]
all_phenotype_ids = all_phenotype_ids.reset_index(drop=True)
all_phenotype_fname = []
for idx in all_phenotype_ids.index:
    if all_phenotype_ids.at[idx, 'idc'].startswith("NDAR"):
        if all_phenotype_ids.at[idx, 'idc'] in abcd_pheno_idc_baseline['idc'].values:
            index = abcd_pheno_idc_baseline[abcd_pheno_idc_baseline['idc'] == all_phenotype_ids.at[idx, 'idc']].index[0]
            all_phenotype_fname.append(abcd_pheno_idc_baseline['filename'][index])
        elif all_phenotype_ids.at[idx, 'idc'] in abcd_pheno_idc_followup['idc'].values:
            index = abcd_pheno_idc_followup[abcd_pheno_idc_followup['idc'] == all_phenotype_ids.at[idx, 'idc']].index[0]
            all_phenotype_fname.append(abcd_pheno_idc_followup['filename'][index])
    elif all_phenotype_ids.at[idx, 'idc'] in genr_f09_pheno_idc['idc'].values:
        index = genr_f09_pheno_idc[genr_f09_pheno_idc['idc'] == all_phenotype_ids.at[idx, 'idc']].index[0]
        all_phenotype_fname.append(genr_f09_pheno_idc['filename'][index])
    elif all_phenotype_ids.at[idx, 'idc'] in genr_f13_pheno_idc['idc'].values:
        index = genr_f13_pheno_idc[genr_f13_pheno_idc['idc'] == all_phenotype_ids.at[idx, 'idc']].index[0]
        all_phenotype_fname.append(genr_f13_pheno_idc['filename'][index])
all_phenotype_ids['filename'] = all_phenotype_fname

# np.save('all_phenotypes_filename.npy', all_phenotype_ids['filename'])


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
all_phenotype_ids['file_paths'] = all_phenotype_ids['filename'].apply(get_file_paths)
# Display the results
print(all_phenotype_ids)
#%% don't run!!! DONE!!!
# np.save('/projects/0/einf1049/scratch/jouyang/all_phenotypes_ids_filename.npy',all_phenotype_ids)


"""
 Analysis of phenotype data
"""
""" # %% statistics of CBCL two scores and age
phenotype_data_check = phenotype_data.copy()
phenotype_data_check[["sum_agg","sum_att", "age"]].describe()
# %%  plot the distribution of CBCL two scores and age
# dont run!!
output_dir = 'output_plots_all_phenotypes_analysis_raw'  # Directory to save plots
os.makedirs(output_dir, exist_ok=True)
# %% dont run!! plot age distrubution
plt.figure(figsize=(10, 6))
sns.histplot(data=phenotype_data[["age"]], kde=True, bins=100)
plt.title(f"Age Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, f"age.png"))
plt.close()
# %% dont run!! plot sex distrubution
plt.figure(figsize=(10, 6))
sns.histplot(data=phenotype_data[["sex"]], kde=True, bins=100)
plt.title(f"Sex Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, f"sex.png"))
plt.close()

# %% dont run!! plot Maternal educational level distrubution
plt.figure(figsize=(10, 6))
sns.histplot(data=phenotype_data[["edu_maternal"]], kde=True, bins=100)
plt.title(f"Maternal educational level Distribution")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.close()
"""
"""
Phenotype data normalization:
- Min-max normalization on age 
- Log(x+1) transformation for processing 0 values and the min-max on two CBCL scores
"""

# %%
# making subset sample data
with open('sample_index.txt', 'r') as f:
    subset_phenotype_idx_ = [int(line.strip()) for line in f.readlines()]
# Remove newline characters
subset_phenotype = phenotype_data.iloc[subset_phenotype_idx_]
# %%
subset_phenotype = subset_phenotype.drop(columns=['eventname','idc']).reset_index(drop=True)
# make the dataframes to tensors
subset_phenotype_ = subset_phenotype.copy()
subset_phenotype_['sum_att'] = log_transform_and_scale(subset_phenotype_['sum_att'])
subset_phenotype_['sum_agg'] = log_transform_and_scale(subset_phenotype_['sum_agg'])
subset_phenotype_['age'] = min_max_normalize(subset_phenotype_['age'])
# %%
df_sub_pheno = subset_phenotype_[['sum_att', 'sum_agg', 'age', 'sex', 'edu_maternal']]
sub_tensor_phenotype = df_sub_pheno.to_numpy()
np.save('/home/jouyang1/sample_normalised_phenotype.npy',sub_tensor_phenotype) # (1000, 5), ['sum_att', 'sum_agg', 'age', 'sex', 'edu_maternal']

# %% Phenotype data normalization
# min-max on age
# log_transform and min-max for two CBCL scores
phenotype_data_correct = phenotype_data.copy()
phenotype_data_correct['sum_att'] = log_transform_and_scale(phenotype_data_correct["sum_att"])
phenotype_data_correct['sum_agg'] = log_transform_and_scale(phenotype_data_correct["sum_agg"])
phenotype_data_correct['age'] = min_max_normalize(phenotype_data_correct["age"])

# phenotype_data_correct_ = phenotype_data_correct.copy()
# phenotype_data_correct_['sum_att'] = phenotype_data_correct['sum_agg']
# phenotype_data_correct_['sum_agg'] = phenotype_data_correct['age']
# phenotype_data_correct_['age'] = phenotype_data_correct['sum_att']

# phenotype_data_correct['sum_agg'] = min_max_normalize(phenotype_data_correct["age"])
# phenotype_data_correct['sum_att'] = log_transform_and_scale(phenotype_data_correct["sum_agg"])
# phenotype_data_correct['age'] = log_transform_and_scale(phenotype_data_correct["sum_att"])
# # min-max on age
# # log_transform and min-max for two CBCL scores
# phenotype_data['sum_agg'] = min_max_normalize(phenotype_data["age"])
# phenotype_data['sum_att'] = log_transform_and_scale(phenotype_data["sum_agg"])
# phenotype_data['age'] = log_transform_and_scale(phenotype_data["sum_att"])

# %% print and check 
phenotype_data_check = phenotype_data_correct.copy()
phenotype_data_check[["sum_agg","sum_att", "age"]].describe()
# make it to tensor
df_phenotype = phenotype_data_correct[['sum_att', 'sum_agg', 'age', 'sex', 'edu_maternal']]
tensor_phenotype = df_phenotype.to_numpy()
print(tensor_phenotype.shape) # (12595, 5)

# %% don't run!! have done!!  save phenotype_data as the .npy tensor
# np.save('/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes_correct.npy',tensor_phenotype) # (12595, 5), ['sum_att', 'sum_agg', 'age', 'sex', 'edu_maternal']
# 保存文件
# output_path = '/home/jouyang1/all_normalised_phenotypes_correct.npy'
# np.save(output_path, tensor_phenotype)

# 修改文件权限为与其他.npy文件相同
# os.chmod(output_path, 0o664)  # -rw-rwx---

# %%
a = np.load('/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes.npy')

def compare_tensors(tensor1, tensor2):
    """
    比较两个tensor是否相同
    """
    # 检查形状
    if tensor1.shape != tensor2.shape:
        print("形状不同！")
        print(f"tensor1形状: {tensor1.shape}")
        print(f"tensor2形状: {tensor2.shape}")
        return
    
    # 检查是否完全相同
    if np.array_equal(tensor1, tensor2):
        print("两个tensor完全相同！")
        return
    
    # 如果不完全相同，进行详细比较
    diff = np.abs(tensor1 - tensor2)
    print("\n差异统计：")
    print(f"最大差异: {np.max(diff)}")
    print(f"平均差异: {np.mean(diff)}")
    print(f"标准差: {np.std(diff)}")
    
    # 找出差异最大的位置
    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"\n最大差异位置: {max_diff_idx}")
    print(f"tensor1值: {tensor1[max_diff_idx]}")
    print(f"tensor2值: {tensor2[max_diff_idx]}")
    
    # 检查每列的差异
    for i in range(tensor1.shape[1]):
        col_diff = np.abs(tensor1[:, i] - tensor2[:, i])
        print(f"\n第{i}列的差异统计：")
        print(f"最大差异: {np.max(col_diff)}")
        print(f"平均差异: {np.mean(col_diff)}")

# 3. 比较两个tensor
compare_tensors(a, tensor_phenotype)

# %%
# #
"""

2. brain data
  > 3d-to-2d projection (512*512 images, robust scale, smoothed) @/home/jouyang/brain-3d-to-2d-projection.py

"""
# 
def check_disk_space(required_space, directory):
    """检查指定目录是否有足够的磁盘空间"""
    try:
        stats = shutil.disk_usage(directory)
        available_space = stats.free
        if available_space < required_space:
            return False, f"需要 {required_space/(1024**3):.2f}GB, 但只有 {available_space/(1024**3):.2f}GB 可用"
        return True, "enough space"
    except Exception as e:
        return False, str(e)

def estimate_batch_size(image_shape, num_features=4, safety_factor=2):
    """估计单个批次所需的内存大小（以字节为单位）"""
    single_subject_size = np.prod(image_shape) * num_features * 4  # 4 bytes per float32
    return single_subject_size * safety_factor

def normalize_coordinates(coordinates):
    r = np.sqrt(np.sum(coordinates**2, axis=1))
    normalized = coordinates / r[:, np.newaxis]
    return normalized

def optimized_mercator_projection(coordinates, features, image_size=(512, 512)):
    if features.ndim == 3:
        features = features.reshape(-1, 4)
    
    if features.shape[0] != coordinates.shape[0]:
        raise ValueError(f"features numbers ({features.shape[0]}) does not match with coordinates numbers ({coordinates.shape[0]}).")

    valid_mask = ~np.isnan(coordinates).any(axis=1) & ~np.isinf(coordinates).any(axis=1) & \
                 ~np.isnan(features).any(axis=1) & ~np.isinf(features).any(axis=1)
    coordinates = coordinates[valid_mask]
    features = features[valid_mask]

    normalized_coords = normalize_coordinates(coordinates)
    x, y, z = normalized_coords[:, 0], normalized_coords[:, 1], normalized_coords[:, 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    
    u = phi
    v = np.log(np.tan(theta/2 + np.pi/4))
    
    # Handle potential NaN or inf values in v
    v = np.nan_to_num(v, nan=0.0, posinf=np.finfo(float).max, neginf=np.finfo(float).min)
    
    max_v = np.log(np.tan(np.pi/4 + 0.95*(np.pi/4)))
    v = np.clip(v, -max_v, max_v)
    
    valid_uv_mask = ~np.isnan(u) & ~np.isinf(u) & ~np.isnan(v) & ~np.isinf(v)
    u = u[valid_uv_mask]
    v = v[valid_uv_mask]
    features = features[valid_uv_mask]
    
    u_bins = np.linspace(u.min(), u.max(), image_size[0] + 1)
    v_bins = np.linspace(v.min(), v.max(), image_size[1] + 1)
    bins = [u_bins, v_bins]

    image = np.zeros((*image_size, 4))
    
    for i in range(4):
        feature = features[:, i]
        result = binned_statistic_2d(u, v, feature, 
                                     statistic='mean', 
                                     bins=bins)
        projection = result.statistic
        
        image[:, :, i] = projection.T
    
    image = np.nan_to_num(image)
    
    return image

def improved_fill_gaps(image, max_distance=10):
    mask = np.isnan(image)
    filled_image = np.copy(image)

    for c in range(image.shape[2]):
        channel = image[:, :, c]
        channel_mask = mask[:, :, c]
        
        if not np.any(channel_mask):
            continue  # Skip if there are no gaps to fill
        
        dist = distance_transform_edt(channel_mask)
        
        weights = np.exp(-dist / max_distance)
        weights[dist > max_distance] = 0
        
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights /= weight_sum
        
        filled = np.sum(channel[:, :, np.newaxis] * weights, axis=(0, 1))
        
        filled_image[channel_mask, c] = filled

    return filled_image

def smooth_image(image, kernel_size=3):
    smoothed = np.copy(image)
    for c in range(image.shape[2]):
        smoothed[:, :, c] = gaussian_filter(image[:, :, c], sigma=kernel_size/2)
    return smoothed

def process_subject(subject_info, coordinates, ID_per_half, left=True):
    feature_files = [
        f"{'lh' if left else 'rh'}.thickness.fwhm10.fsaverage.mgh",
        f"{'lh' if left else 'rh'}.volume.fwhm10.fsaverage.mgh",
        f"{'lh' if left else 'rh'}.area.fwhm10.fsaverage.mgh",
        f"{'lh' if left else 'rh'}.w_g.pct.mgh.fwhm10.fsaverage.mgh"
    ]
    
    features = []
    for feature_file in feature_files:
        file_path = subject_info[2][feature_file]
        feature_data = SOI_array_per_right(ID_per_half, file_path) #SOI_array_per_left if left = True, no worries, you used the correct function for lh.
        feature_norm = robust_scale_normalize(feature_data)
        features.append(feature_norm)
        
    SOI_mx_minmax = np.stack(features, axis=-1)
    
    image = optimized_mercator_projection(coordinates, SOI_mx_minmax)
    filled_image = improved_fill_gaps(image)
    smoothed_image = smooth_image(filled_image)
    
    # Clear some memory
    del image, filled_image, features, SOI_mx_minmax
    gc.collect()
    
    return smoothed_image.transpose(2, 0, 1)


def process_hemisphere(subjects_info, coordinates, ID_per_half, is_left):
    output_dir = '/projects/0/einf1049/scratch/jouyang'
    output_file = os.path.join(output_dir, f"all_cnn_{'lh' if is_left else 'rh'}_brainimages.npy")
    os.makedirs(output_dir, exist_ok=True)
    
    n_subjects = len(subjects_info)
    
    # 检查是否存在未完成的文件
    if os.path.exists(output_file):
        try:
            existing_data = np.load(output_file, mmap_mode='r')
            if existing_data.shape == (n_subjects, 4, 512, 512):
                print(f"Found existing file: {output_file}")
                # 查找最后处理的subject
                for i in range(n_subjects):
                    if np.all(np.isnan(existing_data[i])):
                        start_index = i
                        break
                else:
                    print("File seems complete, skipping processing")
                    return output_file
                print(f"Resuming from subject {start_index}")
            else:
                print("Existing file has wrong shape, starting over")
                start_index = 0
        except Exception as e:
            print(f"Error reading existing file: {e}")
            start_index = 0
    else:
        start_index = 0
    
    # 创建或打开内存映射文件
    mmap_array = np.lib.format.open_memmap(
        output_file,
        mode='w+' if start_index == 0 else 'r+',
        dtype=np.float32,
        shape=(n_subjects, 4, 512, 512)
    )
    
    # 如果是新文件，用NaN初始化
    if start_index == 0:
        mmap_array[:] = np.nan
        mmap_array.flush()
    
    for i in range(start_index, n_subjects):
        try:
            print(f"Processing {'left' if is_left else 'right'} brain: subject {i+1}/{n_subjects}")
            subject_data = process_subject(subjects_info[i], coordinates, ID_per_half, is_left)
            mmap_array[i] = subject_data
            
            # 每100个subject保存一次
            if (i + 1) % 100 == 0:
                print(f"Saving progress... ({i+1}/{n_subjects})")
                mmap_array.flush()
            
            # Clear memory
            del subject_data
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {'left' if is_left else 'right'} brain subject {i+1}: {str(e)}")
            mmap_array.flush()  # 保存已处理的数据
            continue
    
    # 最终保存
    mmap_array.flush()
    
    # 验证数据
    try:
        verify_data = np.load(output_file, mmap_mode='r')
        print(f"Verification - Data shape: {verify_data.shape}")
        print(f"Verification - Any NaN: {np.any(np.isnan(verify_data))}")
        del verify_data
    except Exception as e:
        print(f"Warning: Verification failed: {e}")
    
    print(f"Saved data to: {output_file}")
    return output_file

def main():
    required_space = 50 * (1024**3)
    output_dir = '/projects/0/einf1049/scratch/jouyang'
    space_ok, message = check_disk_space(required_space, output_dir)  # 检查输出目录的空间
    if not space_ok:
        print(f"Insufficient disk space: {message}")
        return

    print("Loading subject information...")
    subjects_info = np.load('/projects/0/einf1049/scratch/jouyang/all_phenotypes_ids_filename.npy', allow_pickle=True)
    
    # Process left brain
    # print("\nProcessing left brain...")
    # with open("/projects/0/einf1049/scratch/jouyang/GenR_mri/lh.fsaverage.sphere.cortex.mask.label", "r") as lh:
    #     data = lh.read().splitlines()[2:]
    # data_arr_lh = np.array([list(map(float, line.split())) for line in data])
    # coordinates_left = np.column_stack((data_arr_lh[:, 1], data_arr_lh[:, 2], data_arr_lh[:, 3]))
    # ID_per_half_left = np.load('ij_id_lh.npy', allow_pickle=True).astype(int)
    
    # left_file = process_hemisphere(subjects_info, coordinates_left, ID_per_half_left, True)

    # # check left brain shape
    # left_data = np.load(left_file, mmap_mode='r')
    # print(f"Left brain data shape: {left_data.shape}")
    # del left_data
    
    # # Clear memory
    # del coordinates_left, ID_per_half_left
    # gc.collect()
    
    # Process right brain
    print("\nProcessing right brain...")
    with open("/projects/0/einf1049/scratch/jouyang/GenR_mri/rh.fsaverage.sphere.cortex.mask.label", "r") as rh:
        data = rh.read().splitlines()[2:]
    data_arr_rh = np.array([list(map(float, line.split())) for line in data])
    coordinates_right = np.column_stack((data_arr_rh[:, 1], data_arr_rh[:, 2], data_arr_rh[:, 3]))
    ID_per_half_right = np.load('ij_id_rh.npy', allow_pickle=True).astype(int)
    
    right_file = process_hemisphere(subjects_info, coordinates_right, ID_per_half_right, False)
    
    # check right brain shape
    right_data = np.load(right_file, mmap_mode='r')
    print(f"Right brain data shape: {right_data.shape}")

    # Clear memory
    del coordinates_right, right_data, ID_per_half_right 
    gc.collect()

    print("\nProcessing completed!")
    # print(f"Left brain data saved to: {left_file}")
    print(f"Right brain data saved to: {right_file}")
# 
if __name__ == "__main__":
    main()

"""
验证 brain image是否正确，lh已经验证，rh已验证。
"""
# %% 
# import matplotlib.pyplot as plt
# import numpy as np

# a = np.load('/projects/0/einf1049/scratch/jouyang/all_cnn_rh_brainimages.npy', mmap_mode='r')
# subset = a[:100]

# # 假设 subset 已经加载好了
# images = subset[2]

# # 创建一个 1x4 的子图来显示每个图像
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# for i in range(4):
#     ax = axes[i]
#     ax.imshow(images[i], cmap='gray')  # 使用灰度图显示
#     ax.axis('off')  # 隐藏坐标轴
#     ax.set_title(f'Image {i + 1}')  # 设置标题

# plt.tight_layout()  # 调整布局以防止重叠

# # 保存图像而不显示
# plt.savefig('all_rh_test.png', dpi=300)  # 保存为PNG文件，分辨率设置为300 DPI

# plt.close()  # 关闭绘图以释放内存


# %%
