"""
1. phenotype: @/home/jouyang/data_prepare.py (line 609 to line 637)
  > min-max on age
  > log transformation + min-max on two CBCl scores
"""
# %% load packages
import pandas as pd
import numpy as np
import pyreadr
import pyreadstat
import gc
import os
import matplotlib.pyplot as plt
from longitude_transform import get_longitudinal_map_each, sphere_to_grid_each, color_map_DK, plot_original, plot_DK_map
from helper_func_prep import SOI_array_per_left,SOI_array_per_right, min_max_normalize, plot_SOI, robust_scale_normalize, log_transform_and_scale
import seaborn as sns
from scipy import stats
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


# %% GenR CBCL, and sex & edu_level data 
# load data for GenR phenotype data
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

# %% QC for GenR other phenotype data
# another consent variable: FUPFASE3_9/FUPFASE4_13 should be "yes" or "1". Checked, all 1.0, even for data with overlap. 

# siblings / twins (duplicates "mother") (when $multiple =1, if multiple, would be siblings/twins...) simply keep one the duplicates.
#       `CHILD-ALLGENERALDATA_24102022.sav`
# random select one of the siblings at focus 9 and 13
print(genr_sex_n_edul_f9['mother'].duplicated().sum(), genr_sex_n_edul_f13['mother'].duplicated().sum())
genr_sex_n_edul_f9 = genr_sex_n_edul_f9.drop_duplicates(subset=['mother']) # 1893 rows × 121 columns, remove duplicates: 63
genr_sex_n_edul_f13 = genr_sex_n_edul_f13.drop_duplicates(subset=['mother']) # 2090 rows × 121 columns, remove duplicates: 75


# %% select GenR phenotype data
# GenR: age diffent at MRI and CBCl, I choose the age at MRI assessment took place.
#       - Child age at assessment at @9 (“age_child_mri_f09”)
#       - Child age at assessment at @13 ("age_child_mri_f13")
#       - `genr_mri_core_data_20220311.rds`
#       - range at focus 9: 8.550308008213554 - 11.986310746064339
#       - range at focus 13: 12.591375770020534 - 16.67898699520876
genr_age_9 = genr_core_f9[['idc','age_child_mri_f09']] # 1956 rows × 2 columns
genr_age_13 = genr_core_f13[['idc','age_child_mri_f13']] # 2165 rows × 2 columns


# %%  GenR: sex assigned at birth ("gender"), education$05 (check no missing data) or ("educm5")
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

#%% Read all ABCD_txt files
abcd_cbcl_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_cbcls01.txt", sep='\t')
abcd_qc_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_imgincl01.txt", sep='\t')
abcd_sex_age_edu_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_lpds01.txt", sep='\t')
abcd_incfind_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_mrfindings02.txt", sep='\t')
abcd_icv_mri_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/abcd_smrip10201.txt", sep='\t') 
abcd_siblings_txt = pd.read_csv("/projects/0/einf1049/scratch/jouyang/ABCD_txt/acspsw03.txt", sep='\t')

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
abcd_merge = abcd_merge.dropna()
abcd_merge = abcd_merge.drop_duplicates(subset=['subjectkey']) # merge: 9511
print('merge3', len(abcd_merge))

# %% ABCD age (interview_age), `abcd_lpds01`
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


# %% remove the ids of phenotype data that does not have the corresponding MRI and/or not found! 
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

# %% don't run until you finish analyse the site
# save the idc and filenames of all phenotypes for later
"""all_phenotype_ids = phenotype_data.iloc[:, :1]
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
all_phenotype_ids['filename'] = all_phenotype_fname"""


"""#%% got the path of each person's brain MRI
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
# print(all_phenotype_ids)
# %% don't run!!! DONE!!!
# np.save('/projects/0/einf1049/scratch/jouyang/all_phenotypes_sites_ids_filename.npy',all_phenotype_ids)"""


"""
 Analysis of phenotype data
"""
# %% statistics of CBCL two scores and age
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
Phenotype data normalization:
- Min-max normalization on age 
- Log(x+1) transformation for processing 0 values and the min-max on two CBCL scores
"""

# %% Phenotype data normalization
# min-max on age
# log_transform and min-max for two CBCL scores
phenotype_data['sum_agg'] = min_max_normalize(phenotype_data["age"])
phenotype_data['sum_att'] = log_transform_and_scale(phenotype_data["sum_agg"])
phenotype_data['age'] = log_transform_and_scale(phenotype_data["sum_att"])

# %% print and check 
phenotype_data_check = phenotype_data.copy()
phenotype_data_check[["sum_agg","sum_att", "age"]].describe()
# make it to tensor
df_phenotype = phenotype_data[['sum_att', 'sum_agg', 'age', 'sex', 'edu_maternal']]
tensor_phenotype = df_phenotype.to_numpy()
print(tensor_phenotype.shape) # (12595, 5)

# don't run!! have done!!  save phenotype_data as the .npy tensor
# np.save('/projects/0/einf1049/scratch/jouyang/all_normalised_phenotypes_sites.npy',tensor_phenotype) # (12595, 5), ['sum_att', 'sum_agg', 'age', 'sex', 'edu_maternal']