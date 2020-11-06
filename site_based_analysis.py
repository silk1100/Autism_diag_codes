"""
Split subjects based on sites
For each site:
    1. Create data matrix from the selected features found in constant.py
    2. Perform 5 fold CV using these features and see how well they perform on each site separately

bonus: Perform a feature selection step on each site alone and see how mutual features we will get from
that site and the global features
"""

import numpy as np
import pandas as pd
from constants import SELECTED_FEATS_DICT, DATADIR_aparc_MODLEFT, DATADIR_aparc_MODRight, TARGET, TD, ASD,\
    OUTPUT_DIR_site_ana, DATADIR_aparc_ALLASDINC, SELECTED_FEATS_DICT_CORR
from ML_model_selection import train_models
from RFE_FS import select_features
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import os
from prepare_data import get_csvfile_ready


def LOAD_DATA(allASDINC=False):
    if not allASDINC:
        df_left, df_right = pd.read_csv(DATADIR_aparc_MODLEFT, index_col=0),\
                            pd.read_csv(DATADIR_aparc_MODRight, index_col=0)
    else:
        df_train_left, df_train_right, df_test_left, df_test_right = \
            get_csvfile_ready(DATADIR_aparc_ALLASDINC)
        df_left = pd.concat([df_train_left, df_test_left])
        df_right = pd.concat([df_train_right, df_test_right])
    return df_left, df_right


def adapt_new_feat_names(feats):
    new_feats = []
    for feat in feats:
        name_parts = feat.split('_')
        morph_name = name_parts[0]
        if 'vo' in morph_name:
            morph_name = 'volume'
        elif 'thic' in morph_name:
            morph_name = 'thickness'
        elif 'cur' in morph_name:
            morph_name = 'curv'
        elif 'are' in morph_name:
            morph_name = 'area'
        name_parts[0] = morph_name
        new_feats.append('_'.join(name_parts))
    return new_feats


def _extract_sites(df, combined_sites=True):
    subj_names = df.index.to_list()
    # check if the site is only one part or 2 parts
    site_subj_dict = defaultdict(list)
    for name in subj_names:
        name_parts = name.split('_')
        if len(name_parts)==2: # One part name
            site_subj_dict[name_parts[0]].append(name)
        elif len(name_parts)==3: # 2 parts name
            if combined_sites:
                site_subj_dict[name_parts[0]].append(name)
            else:
                site_subj_dict['_'.join(name_parts[:-1])].append(name)
        else:
            raise ValueError("Subject names doesnt follow the standard of {SITENAME_*SITEINDEX*_SUBJID}")

    return site_subj_dict


def extract_sites(df, combined_sites=True):
    if type(df) == type((0,1)):
        if len(df) != 2:
            raise ValueError("Tuple of 2 dataframes (df_left, df_right) can only be passed to "
                             "extract_sites()")
        df_left, df_right = df
        if type(df_right)!=type(pd.DataFrame(None)) or type(df_left)!=type(pd.DataFrame(None)):
            raise ValueError("Only a pandas dataframe or tuple of 2 pandas dataframe can be passed")
    elif type(df) != type(pd.DataFrame(None)):
        raise ValueError("Can extract sites only when the ")

    if 'df_left' in locals():
        sites_dict_left = _extract_sites(df_left, combined_sites)
        sites_dict_right = _extract_sites(df_right, combined_sites)
        if sites_dict_left!=sites_dict_right:
            print('WARNING: LEFT HEMISPHERE DATAFRAME SUBJECTS ARE DIFFERENT THAT THE RIGHT HEMISPHERE'
                  'DATAFRAME SUBJECTS')
            sites_dict = (sites_dict_left, sites_dict_right)
        else:
            sites_dict = sites_dict_left
    else:
        sites_dict = _extract_sites(df, combined_sites)

    return sites_dict


def printStats(df_left_site, site):
    asd_cnt = len(df_left_site[df_left_site[TARGET] == ASD])
    td_cnt = len(df_left_site[df_left_site[TARGET] == TD])

    print(f"Number of ASD in {site}: {asd_cnt}")
    print(f"Number of TD in {site}: {td_cnt}")
    if asd_cnt>td_cnt:
        ratio = float(asd_cnt)/float(td_cnt+asd_cnt)
    else:
        ratio = float(td_cnt)/float(td_cnt+asd_cnt)
    print(f"Baseline performance = {ratio*100}%")

    return ratio


def perform_FS(df_left, df_right, site_name, save_output=True):
    y_left, y_right = df_left[TARGET], df_right[TARGET]

    X_clean_left_norm = StandardScaler().fit_transform(df_left.drop(TARGET, axis=1))
    X_clean_right_norm = StandardScaler().fit_transform(df_right.drop(TARGET, axis=1))

    rf = RandomForestClassifier(n_estimators=500)
    svm = LinearSVC(max_iter=1e9)

    # FS for each site
    X_clean_left_rf, y_left, obj_left_rf = select_features(rf, X_clean_left_norm,
                                                           y_left,
                                                           scoring_metric='balanced_accuracy',
                                                           hemi='left')

    X_clean_left_svm, y_left, obj_left_svm = select_features(svm, X_clean_left_norm,
                                                             y_left,
                                                             scoring_metric='balanced_accuracy',
                                                             hemi='left')

    X_clean_right_rf, y_right, obj_right_rf = select_features(rf, X_clean_right_norm,
                                                              y_right,
                                                              scoring_metric='balanced_accuracy',
                                                              hemi='right')

    X_clean_right_svm, y_right, obj_right_svm = select_features(svm, X_clean_right_norm,
                                                                y_right,
                                                                scoring_metric='balanced_accuracy',
                                                                hemi='right')

    if save_output:
        dump(obj_left_rf, f'{os.path.join(OUTPUT_DIR_site_ana, "obj_left_rf")}_{site_name}.joblib')
        dump(obj_left_svm, f'{os.path.join(OUTPUT_DIR_site_ana, "obj_left_svm")}_{site_name}.joblib')
        dump(obj_right_rf, f'{os.path.join(OUTPUT_DIR_site_ana, "obj_right_rf")}_{site_name}.joblib')
        dump(obj_right_svm, f'{os.path.join(OUTPUT_DIR_site_ana, "obj_right_svm")}_{site_name}.joblib')

    return {
        'left-rf':(X_clean_left_rf, y_left, obj_left_rf),
        'right-rf':(X_clean_right_rf, y_right, obj_right_rf),
        'left-svm': (X_clean_left_svm, y_left, obj_left_svm),
        'right-svm': (X_clean_right_svm, y_right, obj_right_svm)
    }


def add_l_r_tofeatname(feat_names):
    left_feats = []
    right_feats = []
    for feat_name in feat_names:
        feat_name_parts = feat_name.split('_')
        b_region = feat_name_parts[1]
        left_b_region = 'l'+b_region
        right_b_region = 'r' + b_region
        feat_name_parts[1] = left_b_region
        left_feats.append('_'.join(feat_name_parts))
        feat_name_parts[1] = right_b_region
        right_feats.append('_'.join(feat_name_parts))

    return left_feats, right_feats


def main():
    # Load data
    df_left, df_right = LOAD_DATA(allASDINC=True)
    df_left.to_csv('D:\\PhD\\Data\\aparc\\df_left_aparc_ALLASDINC.csv')
    df_right.to_csv('D:\\PhD\\Data\\aparc\\df_right_aparc_ALLASDINC.csv')

    age_left, age_right = df_left.pop('age'), df_right.pop('age')
    sex_left, sex_right = df_left.pop('sex'), df_right.pop('sex')

    # Extract sites
    combined_sites = True
    save_FS_output = True
    sites_dict = extract_sites((df_left, df_right), combined_sites=combined_sites)
    sites_results = {}
    print(f"Number of size when we combine sites ({combined_sites}): {len(sites_dict)}")
    for site in sites_dict:
        print(f"site {site} contains {len(sites_dict[site])} subjects.")
        # if site in ['SDSU', 'CMU', 'Leuven', 'Pitt', 'MaxMun', 'OHSU', 'Olin', 'Trinity', 'UM']:
        #     print('site is already processed')
        #     continue
        df_left_site = df_left.loc[sites_dict[site], :]
        df_right_site = df_right.loc[sites_dict[site], :]

        ratio = printStats(df_left_site, site)
        if ratio > 0.65:
            print(f'Ignoring {site} with baseline {ratio*100}%.')
            continue
        # Perform FS
        # selected_dict = perform_FS(df_left_site, df_right_site, site, save_output=save_FS_output)

        # # Perform ML
        repr_dict = dict()
        # # for repr in SELECTED_FEATS_DICT: # FOR RFECV
        # for repr in SELECTED_FEATS_DICT_CORR: # For CORR feats (you need to add left and right for feats)
        #     # selected_feats = SELECTED_FEATS_DICT[repr]
        #     selected_feats = SELECTED_FEATS_DICT_CORR[repr]
        #     selected_feats = adapt_new_feat_names(selected_feats)
        #
        #     if 'LEFT' in repr: # used only with RFECV
        #         X_left = df_left_site[selected_feats]
        #         y_left = df_left_site[TARGET]
        #         X_left = StandardScaler().fit_transform(X_left)
        #         results = train_models(X_left, y_left, 5)
        #
        #     elif 'RIGHT' in repr: # used only with RFECV
        #         X_right = df_right_site[selected_feats]
        #         y_right = df_right_site[TARGET]
        #         X_right = StandardScaler().fit_transform(X_right)
        #         results = train_models(X_right, y_right, 5)
        #     elif repr == '50':
        #         left_selected_feats, right_selected_feats = add_l_r_tofeatname(selected_feats)
        #         X_left = df_left_site[left_selected_feats]
        #         y_left = df_left_site[TARGET]
        #         X_left = StandardScaler().fit_transform(X_left)
        #         results_left = train_models(X_left, y_left, 5)
        #         repr_dict['50_left'] = results_left
        #
        #         X_right = df_right_site[right_selected_feats]
        #         y_right = df_right_site[TARGET]
        #         X_right = StandardScaler().fit_transform(X_right)
        #         results_right = train_models(X_right, y_right, 5)
        #         repr_dict['50_right'] = results_right
        #     else:
        #         raise ValueError("Subject representation can only be either left or right")
        #     if repr!='50':
        #         repr_dict[repr] = results

        X_left = df_left_site.drop(TARGET, axis=1)
        y_left = df_left_site[TARGET]
        X_left = StandardScaler().fit_transform(X_left)
        results_left = train_models(X_left, y_left, 5)
        repr_dict['all_feats_left'] = results_left

        X_right = df_right_site.drop(TARGET, axis=1)
        y_right = df_right_site[TARGET]
        X_right = StandardScaler().fit_transform(X_right)
        results_right = train_models(X_right, y_right, 5)
        repr_dict['all_feats_right'] = results_right
        sites_results[site] = repr_dict
        dump(sites_results, './Results/sites_results_all.joblib')


if __name__ == '__main__':
    main()
