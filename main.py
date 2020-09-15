"""
This script needs to implement the following functions:
    1. Read the data matrix
    2. Split the data matrix into 2 hemispheres
    3. For each hemisphere:
        3.1 Perform correlation analysis to eliminate multi-collinearity
        3.2 Feed the RF-RFE the updated data matrix to select features from it
        3.3 Evaluate the model
"""
import numpy as np
import pandas as pd
import constants
from prepare_data import get_csvfile_ready
from correlation_analysis import correlation_analysis
from joblib import dump, load
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from RFE_FS import select_features
from constants import MAX_ITR, OUTPUT_DIR_CORR, OUTPUT_DIR_FS, OUTPUT_DIR_ML, OUTPUT_DIR_SPLIT
from sklearn.preprocessing import StandardScaler
import os
from ML_model_selection import train_models


def createDirIfNotExist_max2levels(dir):
    """
    Creates directory with maximum new depth 2 folders. String is supposed to be separated via /
    :param dir: str: directory to be created
    :return: None
    """
    if not os.path.isdir(dir):
        if not os.path.isdir(''.join(dir.split('/')[:-1])):
            os.mkdir(''.join(dir.split('/')[:-1]))
        os.mkdir(dir)


def main():
    # # Create output folders
    # createDirIfNotExist_max2levels(OUTPUT_DIR_ML)
    # createDirIfNotExist_max2levels(OUTPUT_DIR_SPLIT)
    # createDirIfNotExist_max2levels(OUTPUT_DIR_FS)
    # createDirIfNotExist_max2levels(OUTPUT_DIR_CORR)
    #
    # # Get the data ready
    updated_data = True
    # df_leftHemi_train, df_rightHemi_train, df_test_left, df_test_right = \
    #     get_csvfile_ready(constants.DATADIR_aparc, testratio=0.2, random_seed=131417191,
    #                       updated_data=updated_data)
    #
    # # Check whether to save as new representation or old representation
    if updated_data:
        left_train_file = 'left_train_modifiedMedPIQR.csv'
        right_train_file = 'right_train_modifiedMedPIQR.csv'
        left_test_file = 'left_test_modifiedMedPIQR.csv'
        right_test_file = 'right_test_modifiedMedPIQR.csv'
        nocoll_left_file = 'left_train_noColliniarity_modifiedMedPIQR'
        nocoll_right_file = 'right_train_noColliniarity_modifiedMedPIQR'
        obj_svc_right_file = "right_svc_modifiedMedPIQR.joblib"
        obj_svc_left_file = "left_svc_modifiedMedPIQR.joblib"
        obj_rf_right_file = "right_rf_modifiedMedPIQR.joblib"
        obj_rf_left_file = "left_rf_modifiedMedPIQR.joblib"
        clf_svm_left_file = "clf_left_svm_modifiedMedPIQR.joblib"
        clf_svm_right_file = "clf_right_svm_modifiedMedPIQR.joblib"
        clf_rf_left_file = "clf_left_rf_modifiedMedPIQR.joblib"
        clf_rf_right_file = "clf_right_rf_modifiedMedPIQR.joblib"
    else:
        left_train_file = 'left_train.csv'
        right_train_file = 'right_train.csv'
        left_test_file = 'left_test.csv'
        right_test_file = 'right_test.csv'
        nocoll_left_file = 'left_train_noColliniarity.csv'
        nocoll_right_file = 'right_train_noColliniarity.csv'
        obj_svc_right_file = "right_svc.joblib"
        obj_svc_left_file = "left_svc.joblib"
        obj_rf_right_file = "right_rf.joblib"
        obj_rf_left_file = "left_rf.joblib"
        clf_svm_left_file = "clf_left_svm.joblib"
        clf_svm_right_file = "clf_right_svm.joblib"
        clf_rf_left_file = "clf_left_rf.joblib"
        clf_rf_right_file = "clf_right_rf.joblib"
    #
    # # Keep age and sex separate
    # age,_ = df_leftHemi_train.pop('age'), df_rightHemi_train.pop('age')
    # sex,_ = df_leftHemi_train.pop('sex'), df_rightHemi_train.pop('sex')
    #
    #
    # df_leftHemi_train.to_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,left_train_file)}')
    # df_rightHemi_train.to_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,right_train_file)}')
    # df_test_left.to_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,left_test_file)}')
    # df_test_right.to_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,right_test_file)}')

    df_leftHemi_train = pd.read_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,left_train_file)}', index_col=0)
    df_rightHemi_train = pd.read_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,right_train_file)}', index_col=0)
    # df_test_left = pd.read_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,left_test_file)}', index_col=0)
    # df_test_right = pd.read_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,right_test_file)}', index_col=0)
    # x=0
    # Correlation analysis
    corr_thresh = 50
    for corr_thresh in [60, 70, 85]:
        df_leftHemi_train_corr = correlation_analysis(df_leftHemi_train, corr_thresh=corr_thresh/100.0)
        df_rightHemi_train_corr = correlation_analysis(df_rightHemi_train, corr_thresh=corr_thresh/100.0)
        #
        df_leftHemi_train_corr.to_csv(os.path.join(OUTPUT_DIR_CORR, f'{nocoll_left_file}_{corr_thresh}.csv'))
        df_rightHemi_train_corr.to_csv(os.path.join(OUTPUT_DIR_CORR, f'{nocoll_right_file}_{corr_thresh}.csv'))
        # X_clean_right = df_rightHemi_train_corr.drop('labels', axis=1)
        # X_clean_left = df_leftHemi_train_corr.drop('labels', axis=1)
        # y_right = df_rightHemi_train_corr['labels']
        # y_left = df_leftHemi_train_corr['labels']
        # X_clean_right_norm = StandardScaler().fit_transform(X_clean_right)
        # X_clean_left_norm = StandardScaler().fit_transform(X_clean_left)
        #
        # # Feature selection (I added a hierarchy level without any testing or debugging--
        # # GO CHECK RFE_FS::select_feats BEFORE RUN)
        # rf = RandomForestClassifier(n_estimators=500, max_depth=5000)
        # svm = LinearSVC(max_iter=MAX_ITR)
        # X_clean_left_rf, y_left, obj_left_rf = select_features(rf,X_clean_left_norm,
        #                                                  y_left,
        #                                                  scoring_metric='balanced_accuracy',
        #                                                  hemi='left')
        #
        # X_clean_left_lsvc, y_left, obj_left_lsvm = select_features(svm, X_clean_left_norm,
        #                                                     y_left,
        #                                                     scoring_metric='balanced_accuracy',
        #                                                     hemi='left')
        #
        # X_clean_right_rf, y_right, obj_right_rf = select_features(rf, X_clean_right_norm,
        #                                                     y_right,
        #                                                     scoring_metric='balanced_accuracy',
        #                                                     hemi='right')
        #
        # X_clean_right_lsvc, y_right, obj_right_lsvm = select_features(svm, X_clean_right_norm,
        #                                                     y_right,
        #                                                     scoring_metric='balanced_accuracy',
        #                                                     hemi='right')

        # X_clean_left_lsvc.to_csv(os.path.join(OUTPUT_DIR_FS, "clean_left_svc.csv"))
        # X_clean_right_lsvc.to_csv(os.path.join(OUTPUT_DIR_FS, "clean_right_svc.csv"))
        # X_clean_left_rf.to_csv(os.path.join(OUTPUT_DIR_FS, "clean_left_rf.csv"))
        # X_clean_right_rf.to_csv(os.path.join(OUTPUT_DIR_FS, "clean_right_rf.csv"))
        # dump(obj_left_lsvm, os.path.join(OUTPUT_DIR_FS,obj_svc_left_file))
        # dump(obj_right_lsvm, os.path.join(OUTPUT_DIR_FS,obj_svc_right_file))
        # dump(obj_left_rf, os.path.join(OUTPUT_DIR_FS,obj_rf_left_file))
        # dump(obj_right_rf, os.path.join(OUTPUT_DIR_FS,obj_rf_right_file))



        # Load FS data
        # frames, objs = constants.LOAD_FS_RESULTS()
        # # X_clean_left_lsvc, X_clean_right_lsvc, X_clean_left_rf, X_clean_right_rf, y_left, y_right = frames
        # df_train_left = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT, 'left_train_modifiedMedPIQR.csv'), index_col=0)
        # df_train_right = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT, 'right_train_modifiedMedPIQR.csv'), index_col=0)
        # obj_left_lsvc, obj_right_lsvc, obj_left_rf, obj_right_rf = objs
        # feat_select_svc_left = np.where(obj_left_lsvc.ranking_==1)[0]
        # feat_select_svc_right = np.where(obj_right_lsvc.ranking_==1)[0]
        # feat_select_rf_left = np.where(obj_left_rf.ranking_==1)[0]
        # feat_select_rf_right = np.where(obj_right_rf.ranking_==1)[0]
        #
        # X_clean_right_rf = df_train_right[df_train_right.columns[feat_select_rf_right]]
        # X_clean_left_rf = df_train_left[df_train_left.columns[feat_select_rf_left]]
        # X_clean_right_lsvc = df_train_right[df_train_right.columns[feat_select_svc_right]]
        # X_clean_left_lsvc = df_train_left[df_train_left.columns[feat_select_svc_left]]
        # y_right = df_train_right['labels']
        # y_left = df_train_left['labels']
        # Normalization if required
        #     left_train_file = 'left_train_modifiedMedPIQR.csv'
        # #     right_train_file = 'right_train_modifiedMedPIQR.csv'
        # X_clean_left_rf_norm = StandardScaler().fit_transform(X_clean_left_rf)
        # X_clean_right_rf_norm = StandardScaler().fit_transform(X_clean_right_rf)
        #
        #
        #
        # X_clean_left_rf_norm = StandardScaler().fit_transform(X_clean_left_rf)
        # X_clean_right_rf_norm = StandardScaler().fit_transform(X_clean_right_rf)
        # # X_clean_left_lsvc_norm = StandardScaler().fit_transform(X_clean_left_lsvc)
        # # X_clean_right_lsvc_norm = StandardScaler().fit_transform(X_clean_right_lsvc)
        #
        # clf_left_rf = train_models(X_clean_left_rf_norm, y_left, 5)
        # clf_right_rf = train_models(X_clean_right_rf_norm, y_right, 5)
        # # clf_left_lsvc = train_models(X_clean_left_lsvc_norm, y_left, 5)
        # # clf_right_lsvc = train_models(X_clean_right_lsvc_norm, y_right, 5)
        #
        # # dump(clf_left_lsvc, os.path.join(OUTPUT_DIR_ML,clf_svm_left_file))
        # # dump(clf_right_lsvc, os.path.join(OUTPUT_DIR_ML,clf_svm_right_file))
        # dump(clf_left_rf, os.path.join(OUTPUT_DIR_ML,"clf_left_OnlyCorr_50.joblib"))
        # dump(clf_right_rf, os.path.join(OUTPUT_DIR_ML,"clf_right_OnlyCorr_50.joblib"))

        # This section is only to check correlation without FS
        df_leftHemi_train_corr.to_csv(os.path.join(OUTPUT_DIR_CORR, f'{nocoll_left_file}_{corr_thresh}.csv'))
        df_rightHemi_train_corr.to_csv(os.path.join(OUTPUT_DIR_CORR, f'{nocoll_right_file}_{corr_thresh}.csv'))
        y_left = df_leftHemi_train_corr.pop('labels')
        y_right = df_rightHemi_train_corr.pop('labels')

        X_clean_left_rf_norm = StandardScaler().fit_transform(df_leftHemi_train_corr)
        X_clean_right_rf_norm = StandardScaler().fit_transform(df_rightHemi_train_corr)

        clf_left_rf = train_models(X_clean_left_rf_norm, y_left, 5)
        clf_right_rf = train_models(X_clean_right_rf_norm, y_right, 5)

        #
        # # dump(clf_left_lsvc, os.path.join(OUTPUT_DIR_ML,clf_svm_left_file))
        # # dump(clf_right_lsvc, os.path.join(OUTPUT_DIR_ML,clf_svm_right_file))
        dump(clf_left_rf, os.path.join(OUTPUT_DIR_ML,f"clf_left_OnlyCorr_{corr_thresh}.joblib"))
        dump(clf_right_rf, os.path.join(OUTPUT_DIR_ML,f"clf_right_OnlyCorr_{corr_thresh}.joblib"))


if __name__ == '__main__':
    main()