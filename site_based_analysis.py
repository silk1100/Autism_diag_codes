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
    OUTPUT_DIR_site_ana, DATADIR_aparc_ALLASDINC
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

    X_clean_left_norm = StandardScaler().fit_transform(df_left)
    X_clean_right_norm = StandardScaler().fit_transform(df_right)

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
        dump(obj_left_rf, f'{os.path.join(OUTPUT_DIR_site_ana, "obj_left_rf")}_{site_name}.npy')
        dump(obj_left_svm, f'{os.path.join(OUTPUT_DIR_site_ana, "obj_left_svm")}_{site_name}.npy')
        dump(obj_right_rf, f'{os.path.join(OUTPUT_DIR_site_ana, "obj_right_rf")}_{site_name}.npy')
        dump(obj_right_svm, f'{os.path.join(OUTPUT_DIR_site_ana, "obj_right_svm")}_{site_name}.npy')

    return {
        'left-rf':(X_clean_left_rf, y_left, obj_left_rf),
        'right-rf':(X_clean_right_rf, y_right, obj_right_rf),
        'left-svm': (X_clean_left_svm, y_left, obj_left_svm),
        'right-svm': (X_clean_right_svm, y_right, obj_right_svm)
    }


def main():
    # Load data
    df_left, df_right = LOAD_DATA(allASDINC=True)

    age_left, age_right = df_left.pop('age'), df_right.pop('age')
    sex_left, sex_right = df_left.pop('sex'), df_right.pop('sex')

    # Extract sites
    combined_sites = True
    save_FS_output = True
    sites_dict = extract_sites((df_left, df_right), combined_sites=combined_sites)
    print(f"Number of size when we combine sites ({combined_sites}): {len(sites_dict)}")
    for site in sites_dict:
        print(f"site {site} contains {len(sites_dict[site])} subjects.")
        df_left_site = df_left.loc[sites_dict[site], :]
        df_right_site = df_right.loc[sites_dict[site], :]

        ratio = printStats(df_left_site, site)
        if ratio > 0.65:
            print(f'Ignoring {site} with baseline {ratio*100}%.')
            continue
        # Perform FS
        selected_dict = perform_FS(df_left, df_right, site, save_output=save_FS_output)

        # # Perform ML
        # x=0


if __name__ == '__main__':
    main()
