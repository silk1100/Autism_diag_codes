from sklearn.feature_selection import RFECV
from joblib import dump
import numpy as np
from datetime import datetime
import os


def create_name_from_datetime(prefex, clcname, postfix):
    name = prefex
    name += '_'+clcname
    t = str(datetime.now())
    name += '_' + t[:t.index('.')].replace(':','-').replace(' ','_')
    name += postfix
    return name


def save_FSobject(obj, file_name, clc, hemi):
    try:
        if file_name is None:
            clc_name = str(clc)
            clc_name = clc_name[:clc_name.index('(')]
            if 'l' in hemi:
                clc_name += '_left_hier'
            else:
                clc_name += '_right_hier'
            name = create_name_from_datetime(prefex='RFECV', clcname=clc_name, postfix='.joblib')
            dump(obj, name)
        else:
            if '.joblib' in file_name:
                pos = file_name.index('.')
                name = file_name[:pos]+hemi+'.joblib'
                dump(obj, name)
            else:
                name = file_name+hemi+ '.joblib'
                dump(obj, name)
    except:
        return False

    return True, name


def select_features(clc,X, y, scoring_metric, hemi=None, save_file=True, file_name=None):
    # Feature selection using Random forest and 5fold Recursive feature elimination
    # rf = RandomForestClassifier(n_estimators=500, max_depth=5000)
    # obj_left = RFECV(rf, verbose=True, scoring='balanced_accuracy', n_jobs=-1)
    # obj_right = RFECV(rf, verbose=True, scoring='balanced_accuracy', n_jobs=-1)
    # Xupdated = X
    # for _ in range(3):
    obj = RFECV(clc, cv=10, verbose=3, scoring=scoring_metric, n_jobs=-1)
    obj.fit(X, y)
        # best_feats = Xupdated.columns[np.where(obj.ranking_==1)[0]]
        # Xupdated = Xupdated[best_feats]
    # obj_left.fit(df_leftHemi_train_corr.drop('labels', axis=1), df_leftHemi_train['labels'], )
    # dump(obj_left, "rfecv_lefthemisphere_RF.joblib")
    # obj_right.fit(df_rightHemi_train_corr.drop('labels', axis=1), df_rightHemi_train['labels'])
    # dump(obj_right, "rfecv_righthemisphere_RF.joblib")
    if save_file:
        if hemi is None:
            raise ValueError('hemi must be specified either right or left if you choose to save file!')
        response = save_FSobject(obj, file_name, clc, hemi)
        if not response:
            print(f"FS file couldn't be saved!")
        else:
            print(f'FS file is saved in {os.path.join(os.path._getfullpathname("."),response[1])}')

    # Train classifiers and evaluate using these features
    # selected_left_feats = df_leftHemi_train_corr.columns[np.where(obj_left.ranking_ == 1)[0]]
    # selected_right_feats = df_rightHemi_train_corr.columns[np.where(obj_right.ranking_ == 1)[0]]
    # X_left_clean = df_leftHemi_train_corr[selected_left_feats]
    # X_right_clean = df_rightHemi_train_corr[selected_right_feats]
    # y_left = df_leftHemi_train_corr['labels']
    # y_right = df_rightHemi_train_corr['labels']
    try:
        select_feats = X.columns[np.where(obj.ranking_==1)[0]]
        X_clean = X[select_feats]
    except:
        X_clean = X[np.where(obj.ranking_==1)[0]]
    return X_clean, y, obj
