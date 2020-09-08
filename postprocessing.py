from joblib import load
import pandas as pd
import numpy as np
import constants

from joblib import dump
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
from collections import defaultdict
from prepare_data import get_csvfile_ready
from correlation_analysis import correlation_analysis


def train_clc(clc, X, y):
    clx = clc.fit(X, y)
    if clx is None:
        return clc
    else:
        return clx


def evaluate_clc(clc, X, y):
    yhat = clc.predict(X)
    scoref1 = f1_score(y.values, yhat, average='weighted')
    scorebacc = balanced_accuracy_score(y, yhat)
    return scoref1, scorebacc


def get_score_dict(clc, Xtr, ytr, Xte, yte, scores_dict, key):
    rf_clc = train_clc(clc, Xtr, ytr)
    fscore, bscor = evaluate_clc(rf_clc, Xte, yte)
    scores_dict[key].append((fscore, bscor))
    return scores_dict


def main():
    obj_left = load('rfecv_lefthemisphere_RF.joblib')
    obj_right = load('rfecv_righthemisphere_RF.joblib')


    # Get the data ready
    df_leftHemi_train, df_rightHemi_train, df_test_left, df_test_right = \
        get_csvfile_ready(constants.DATADIR_aparc)

    # Correlation analysis
    df_leftHemi_train_corr = correlation_analysis(df_leftHemi_train)
    df_rightHemi_train_corr = correlation_analysis(df_rightHemi_train)


    selected_left_feats = df_leftHemi_train_corr.columns[np.where(obj_left.ranking_==1)[0]]
    selected_right_feats = df_rightHemi_train_corr.columns[np.where(obj_right.ranking_==1)[0]]
    X_left_clean = df_leftHemi_train_corr[selected_left_feats]
    X_right_clean = df_rightHemi_train_corr[selected_right_feats]
    y_left = df_leftHemi_train_corr['labels']
    y_right = df_rightHemi_train_corr['labels']

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=372957125)
    left_scores_dict = defaultdict(list)
    right_scores_dict = defaultdict(list)

    for train_index, test_index in rskf.split(X_left_clean, y_left):
        X_left_clean_train, X_left_clean_test = X_left_clean.iloc[train_index, :], X_left_clean.iloc[test_index,:]
        X_right_clean_train, X_right_clean_test = X_right_clean.iloc[train_index, :], X_right_clean.iloc[test_index,:]
        y_left_train, y_left_test = y_left.iloc[train_index], y_left.iloc[test_index]
        y_right_train, y_right_test = y_right.iloc[train_index], y_right.iloc[test_index]

        rf_clc_left, rf_clc_right= RandomForestClassifier(), RandomForestClassifier()
        left_scores_dict = get_score_dict(rf_clc_left, X_left_clean_train, y_left_train,
                                          X_left_clean_test, y_left_test, left_scores_dict, 'rf')
        right_scores_dict = get_score_dict(rf_clc_right, X_right_clean_train, y_right_train,
                                           X_right_clean_test, y_right_test, right_scores_dict, 'rf')

        rf_clc_left, rf_clc_right= LogisticRegression(), LogisticRegression()
        left_scores_dict = get_score_dict(rf_clc_left, X_left_clean_train, y_left_train,
                                          X_left_clean_test, y_left_test, left_scores_dict, 'lg')
        right_scores_dict = get_score_dict(rf_clc_right, X_right_clean_train, y_right_train,
                                           X_right_clean_test, y_right_test, right_scores_dict, 'lg')

        rf_clc_left, rf_clc_right= XGBClassifier(), XGBClassifier()
        left_scores_dict = get_score_dict(rf_clc_left, X_left_clean_train, y_left_train,
                                          X_left_clean_test, y_left_test, left_scores_dict, 'xgb')
        right_scores_dict = get_score_dict(rf_clc_right, X_right_clean_train, y_right_train,
                                           X_right_clean_test, y_right_test, right_scores_dict, 'xgb')

        rf_clc_left, rf_clc_right= XGBRFClassifier(), XGBRFClassifier()
        left_scores_dict = get_score_dict(rf_clc_left, X_left_clean_train, y_left_train,
                                          X_left_clean_test, y_left_test, left_scores_dict, 'xgbrf')
        right_scores_dict = get_score_dict(rf_clc_right, X_right_clean_train, y_right_train,
                                           X_right_clean_test, y_right_test, right_scores_dict, 'xgbrf')

        rf_clc_left, rf_clc_right= GaussianNB(), GaussianNB()
        left_scores_dict = get_score_dict(rf_clc_left, X_left_clean_train, y_left_train,
                                          X_left_clean_test, y_left_test, left_scores_dict, 'nb')
        right_scores_dict = get_score_dict(rf_clc_right, X_right_clean_train, y_right_train,
                                           X_right_clean_test, y_right_test, right_scores_dict, 'nb')

        rf_clc_left, rf_clc_right= SVC(), SVC()
        left_scores_dict = get_score_dict(rf_clc_left, X_left_clean_train, y_left_train,
                                          X_left_clean_test, y_left_test, left_scores_dict, 'rsvm')
        right_scores_dict = get_score_dict(rf_clc_right, X_right_clean_train, y_right_train,
                                           X_right_clean_test, y_right_test, right_scores_dict, 'rsvm')

        rf_clc_left, rf_clc_right= LinearSVC(), LinearSVC()
        left_scores_dict = get_score_dict(rf_clc_left, X_left_clean_train, y_left_train,
                                          X_left_clean_test, y_left_test, left_scores_dict, 'lsvm')
        right_scores_dict = get_score_dict(rf_clc_right, X_right_clean_train, y_right_train,
                                           X_right_clean_test, y_right_test, right_scores_dict, 'lsvm')


if __name__ == '__main__':
    main()