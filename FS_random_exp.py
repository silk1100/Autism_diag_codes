"""
This script tests the features selected in the notebook Compare_FSALONE_VS_CORRALONE.ipynb and see
how stable they are.
The script runs as follow:
1. Get the features corresponds to highest CV score from Compare_FSALONE_VS_CORRALONE.ipynb notebook for
each technique FS (RF, SVM) alone, CORRANA alone (0.5 threshold), and FS+CorrANA
2. For each technique:
    2.1 Sample random features with the same size of the features selected using this technique.
    Randomly selected features shouldn't be overlapped with the selected features.
    2.2 Run repeatedkfold CV on the selected and random set
    2.3 Run statistical analysis on the results to see if there is any statistical significance
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeRegressor, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from joblib import load, dump
from prepare_data import get_csvfile_ready
import os
from constants import DATADIR_aparc, DATADIR_aparc_LEFT_NEW, DATADIR_aparc_RIGHT_NEW
from ML_model_selection import train_models

SELECTD_50_CORRANA_NO_L_R = ['thick_caudalanteriorcingulate_medMIQR',
 'thick_isthmuscingulate_medMIQR',
 'thick_medialorbitofrontal_medPIQR',
 'thick_parahippocampal_medPIQR',
 'thick_posteriorcingulate_medMIQR',
 'thick_rostralanteriorcingulate_medPIQR',
 'thick_frontalpole_medMIQR',
 'thick_temporalpole_medMIQR',
 'curv_bankssts_medPIQR',
 'curv_caudalanteriorcingulate_medMIQR',
 'curv_caudalanteriorcingulate_medPIQR',
 'curv_caudalmiddlefrontal_medMIQR',
 'curv_caudalmiddlefrontal_medPIQR',
 'curv_cuneus_medMIQR',
 'curv_cuneus_medPIQR',
 'curv_entorhinal_medMIQR',
 'curv_entorhinal_medPIQR',
 'curv_fusiform_medMIQR',
 'curv_fusiform_medPIQR',
 'curv_inferiorparietal_medMIQR',
 'curv_inferiorparietal_medPIQR',
 'curv_inferiortemporal_medMIQR',
 'curv_inferiortemporal_medPIQR',
 'curv_isthmuscingulate_medMIQR',
 'curv_lateraloccipital_medMIQR',
 'curv_lateraloccipital_medPIQR',
 'curv_lateralorbitofrontal_medPIQR',
 'curv_lingual_medMIQR',
 'curv_lingual_medPIQR',
 'curv_medialorbitofrontal_medPIQR',
 'curv_middletemporal_medMIQR',
 'curv_middletemporal_medPIQR',
 'curv_parahippocampal_medMIQR',
 'curv_paracentral_medMIQR',
 'curv_paracentral_medPIQR',
 'curv_parsopercularis_medMIQR',
 'curv_parsopercularis_medPIQR',
 'curv_parsorbitalis_medMIQR',
 'curv_parsorbitalis_medPIQR',
 'curv_parstriangularis_medMIQR',
 'curv_parstriangularis_medPIQR',
 'curv_postcentral_medMIQR',
 'curv_posteriorcingulate_medMIQR',
 'curv_posteriorcingulate_medPIQR',
 'curv_precentral_medMIQR',
 'curv_precentral_medPIQR',
 'curv_precuneus_medPIQR',
 'curv_rostralanteriorcingulate_medMIQR',
 'curv_rostralmiddlefrontal_medPIQR',
 'curv_superiorfrontal_medMIQR',
 'curv_superiorfrontal_medPIQR',
 'curv_superiorparietal_medMIQR',
 'curv_superiorparietal_medPIQR',
 'curv_superiortemporal_medMIQR',
 'curv_supramarginal_medMIQR',
 'curv_supramarginal_medPIQR',
 'curv_frontalpole_medMIQR',
 'curv_frontalpole_medPIQR',
 'curv_temporalpole_medMIQR',
 'curv_temporalpole_medPIQR',
 'curv_transversetemporal_medMIQR',
 'curv_transversetemporal_medPIQR',
 'area_bankssts_medMIQR',
 'area_bankssts_medPIQR',
 'area_caudalanteriorcingulate_medMIQR',
 'area_caudalanteriorcingulate_medPIQR',
 'area_entorhinal_medMIQR',
 'area_entorhinal_medPIQR',
 'area_fusiform_medMIQR',
 'area_isthmuscingulate_medMIQR',
 'area_isthmuscingulate_medPIQR',
 'area_lateralorbitofrontal_medMIQR',
 'area_medialorbitofrontal_medMIQR',
 'area_medialorbitofrontal_medPIQR',
 'area_parahippocampal_medMIQR',
 'area_parahippocampal_medPIQR',
 'area_parsopercularis_medMIQR',
 'area_parsorbitalis_medMIQR',
 'area_parstriangularis_medMIQR',
 'area_parstriangularis_medPIQR',
 'area_posteriorcingulate_medMIQR',
 'area_posteriorcingulate_medPIQR',
 'area_rostralanteriorcingulate_medMIQR',
 'area_rostralanteriorcingulate_medPIQR',
 'area_frontalpole_medMIQR',
 'area_temporalpole_medMIQR',
 'area_temporalpole_medPIQR',
 'area_transversetemporal_medPIQR',
 'vol_caudalanteriorcingulate_medPIQR',
 'vol_entorhinal_medPIQR',
 'vol_fusiform_medMIQR',
 'vol_isthmuscingulate_medMIQR',
 'vol_lateraloccipital_medPIQR',
 'vol_medialorbitofrontal_medMIQR',
 'vol_parahippocampal_medMIQR',
 'vol_superiortemporal_medMIQR',
 'vol_frontalpole_medMIQR',
 'vol_temporalpole_medMIQR',
 'vol_transversetemporal_medMIQR']
SELECTED_LEFT_RF = ['thick_llateraloccipital_medMIQR',
  'thick_lsuperiortemporal_medPIQR',
  'area_lbankssts_medMIQR',
  'area_lpericalcarine_medPIQR',
  'area_lsupramarginal_medPIQR',
  'vol_lbankssts_medMIQR']
SELECTED_LEFT_SVM = ['thick_linferiorparietal_medPIQR',
  'thick_lsuperiorparietal_medPIQR',
  'vol_linferiorparietal_medPIQR',
  'vol_llateraloccipital_medPIQR',
  'vol_lsuperiortemporal_medPIQR']

SELECTED_RIGHT_RF = ['thick_rsuperiorparietal_medPIQR',
  'thick_rsuperiortemporal_medPIQR',
  'area_rbankssts_medMIQR',
  'area_rpericalcarine_medPIQR',
  'area_rsupramarginal_medPIQR',
  'vol_rmedialorbitofrontal_medPIQR']
SELECTED_RIGHT_SVM = ['thick_rsuperiorparietal_medPIQR',
  'thick_rinferiorparietal_medPIQR',
  'vol_rinferiorparietal_medPIQR',
  'vol_rlateraloccipital_medPIQR',
  'vol_rsuperiortemporal_medPIQR']

TARGET = 'labels'


def LOAD_NEW_DATA():
    if (not os.path.isfile(DATADIR_aparc_RIGHT_NEW)) or (not os.path.isfile(DATADIR_aparc_LEFT_NEW)):
        df_train_left, df_train_right, df_test_left, df_test_right = \
            get_csvfile_ready(DATADIR_aparc, testratio=0.2, random_seed=11111111, updated_data=True)

        df_left = pd.concat([df_train_left, df_test_left], axis=0)
        df_right = pd.concat([df_train_right, df_test_right], axis=0)
        df_left.to_csv("D:\\PhD\\Data\\aparc\\df_left_newRepresentation.csv")
        df_right.to_csv("D:\\PhD\\Data\\aparc\\df_right_newRepresentation.csv")
    else:
        df_left = pd.read_csv(DATADIR_aparc_LEFT_NEW, index_col=0)
        df_right = pd.read_csv(DATADIR_aparc_LEFT_NEW, index_col=0)

    return df_left, df_right


def add_l_r(feats_list, prefix='l'):
    sided_feats_list = []
    for feat in feats_list:
        featL = feat.split('_')
        region = featL[1]
        region = prefix+region
        featL[1] = region
        sided_feats_list.append('_'.join(featL))
    return sided_feats_list


def compare_feats_performance(df_selected, df_random, cv,  rn):
    X_true, X_rand = df_selected.drop(TARGET, axis=1), df_random.drop(TARGET, axis=1)
    y_true, y_rand = df_selected[TARGET].values, df_random[TARGET].values

    X_true_norm, X_rand_norm = StandardScaler().fit_transform(X_true),\
                               StandardScaler().fit_transform(X_rand)

    results_dict_true = train_models(X_true_norm, y_true, cv, rn)
    results_dict_rand = train_models(X_rand_norm, y_rand, cv, rn)
    return results_dict_true, results_dict_rand


def main():
    # Add l and r to correlation analysis selected features
    SELECTED_50_CORRRANA_LEFT = add_l_r(SELECTD_50_CORRANA_NO_L_R, 'l')
    SELECTED_50_CORRRANA_RIGHT = add_l_r(SELECTD_50_CORRANA_NO_L_R, 'r')
    print(SELECTED_50_CORRRANA_LEFT, SELECTED_50_CORRRANA_RIGHT)

    # Create dictionary for each selected group to facilitate saving the results
    SELECTED_FEATS_DICT = {
        'RF-FS-LEFT': SELECTED_LEFT_RF,
        'RF-FS-RIGHT': SELECTED_RIGHT_RF,
        'SVM-FS-LEFT': SELECTED_LEFT_SVM,
        'SVM-FS-RIGHT': SELECTED_RIGHT_SVM,
    }

    # LOAD original data with modifying it to the median-IQR & median+IQR
    df_left, df_right = LOAD_NEW_DATA()
    age_left, age_right = df_left.pop('age'), df_right.pop('age')
    sex_left, sex_right = df_left.pop('sex'), df_right.pop('sex')
    # labels_left, labels_right = df_left.pop('labels'), df_right.pop('labels')
    all_left_feats = df_left.columns.drop('labels')
    all_right_feats = df_right.columns.drop('labels')

    cv = 5
    rn = 1254
    # Iterate over different representations
    feats_repr_results = defaultdict(list)
    for _ in range(5): # Each loop will sample different random features for analysis
        for repr in SELECTED_FEATS_DICT:
            selected_feats = SELECTED_FEATS_DICT[repr]
            if 'LEFT' in repr:
                df_selected = df_left[selected_feats+[TARGET]]
                unselected_feats = all_left_feats.drop(selected_feats)
                random_feats = np.random.choice(unselected_feats, len(selected_feats), replace=False)
                df_random = df_left[random_feats.tolist()+[TARGET]]
            elif 'RIGHT' in repr:
                df_selected = df_right[selected_feats+[TARGET]]
                unselected_feats = all_right_feats.drop(selected_feats)
                random_feats = np.random.choice(unselected_feats, len(selected_feats), replace=False)
                df_random = df_right[random_feats.tolist() + [TARGET]]

            else:
                raise ValueError('SELECTED_FEAT representation cant be found')

            results_true, results_rand = compare_feats_performance(df_selected, df_random, cv, rn+_)
            feats_repr_results[repr].append((results_true, results_rand))
            np.save('./Results/Rand/feats_repr_results.npy', feats_repr_results)


if __name__ == '__main__':
    main()
