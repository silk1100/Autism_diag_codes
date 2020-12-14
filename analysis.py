from joblib import load
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
from collections import defaultdict


clf_dir = 'D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/ML/minmaxreg/clf_***_train.joblib'
clf_corr_dir = 'D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/ML/minmaxreg/clf_***_train_corr.joblib'
clf_corr_l_dir = 'D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/ML/minmaxreg/clf_***_train_corr_l.joblib'
clf_corr_r_dir = 'D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/ML/minmaxreg/clf_***_train_corr_r.joblib'

norm_obj_dir = 'D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/FS'
df_train_corr_dir = 'D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/CORR_ANA'
rfe_obj_dir = 'D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/FS/minmaxreg'

def load_analyze(clf_dir,clf_corr_dir,clf_corr_l_dir,clf_corr_r_dir):
    # clf = load('D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/ML/minmaxreg/clf__rf_train.joblib')
    # clf_corr = load('D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/ML/minmaxreg/clf__rf_train_corr.joblib')
    # clf_corr_l = load('D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/ML/minmaxreg/clf__rf_train_corr_l.joblib')
    # clf_corr_r = load('D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/ML/minmaxreg/clf__rf_train_corr_r.joblib')
    clf = load(clf_dir)
    clf_corr = load(clf_corr_dir)
    clf_corr_l = load(clf_corr_l_dir)
    clf_corr_r = load(clf_corr_r_dir)

    classifiers = ['lSVM', 'pagg','lg','XGB','GNB','SVC','Rf','nn']
    selected_clfs = {}
    selected_clfs_corr = {}
    selected_clfs_corr_l = {}
    selected_clfs_corr_r = {}

    for clc in classifiers:
        if clf[clc].best_score_ > 0.6:
            selected_clfs[clc] = clf[clc]

        if clf_corr[clc].best_score_>0.6:
            selected_clfs_corr[clc] = clf_corr[clc]

        if clf_corr_l[clc].best_score_>0.6:
            selected_clfs_corr_l[clc] = clf_corr_l[clc]

        if clf_corr_r[clc].best_score_>0.6:
            selected_clfs_corr_r[clc] = clf_corr_r[clc]

    return selected_clfs, selected_clfs_corr, selected_clfs_corr_l, selected_clfs_corr_r


def load_test_data(dir='D:/PhD/Codes/ScientificReport_codes/Final_Results_adjusted/INITIAL_SPLIT'):
    df_test = pd.read_csv(os.path.join(dir, 'test_fullbrain.csv'), index_col=0)
    df_test_l = pd.read_csv(os.path.join(dir, 'test_leftbrain.csv'), index_col=0)
    df_test_r = pd.read_csv(os.path.join(dir, 'test_rightbrain.csv'), index_col=0)

    return df_test, df_test_l, df_test_r


def prepare_test_data(df_test, df_test_l, df_test_r, norm=False, corr_ana=False, rfeobj=False):

    if corr_ana:
        df_train_corr = pd.read_csv(os.path.join(df_train_corr_dir, 'fullbrain_corr.csv'), index_col=0)
        df_train_corr_l = pd.read_csv(os.path.join(df_train_corr_dir, 'leftbrain_corr.csv'), index_col=0)
        df_train_corr_r = pd.read_csv(os.path.join(df_train_corr_dir, 'rightbrain_corr.csv'), index_col=0)
        corr_feats = df_train_corr.drop('labels', axis=1).columns
        corr_feats_l = df_train_corr_l.drop('labels', axis=1).columns
        corr_feats_r = df_train_corr_r.drop('labels', axis=1).columns

    if norm:
        norm_objs = [os.path.join(norm_obj_dir, x) for x in os.listdir(norm_obj_dir) if '.joblib' in x]
        corr_l = [x for x in norm_objs if 'corr_l_' in x][0]
        corr_r = [x for x in norm_objs if 'corr_r_' in x][0]
        corr = [x for x in norm_objs if ('corr_' in x)and('corr_l_' not in x)and('corr_r_' not in x)][0]
        nocorr = [x for x in norm_objs if 'corr_' not in x][0]
        minmax_corr_l = load(corr_l)
        minmax_corr_r = load(corr_r)
        minmax_corr = load(corr)
        minmax_nocorr = load(nocorr)
        Xtest = minmax_nocorr.transform(df_test.drop('labels', axis=1).values)
        if corr_ana:
            df_test_corr = df_test.loc[:, corr_feats]
            df_test_corr_l = df_test_l.loc[:, corr_feats_l]
            df_test_corr_r = df_test_r.loc[:, corr_feats_r]

            Xtest_corr = minmax_corr.transform(df_test_corr.values)
            Xtest_corr_l = minmax_corr_l.transform(df_test_corr_l.values)
            Xtest_corr_r = minmax_corr_r.transform(df_test_corr_r.values)

    nocorr_rfe = {}
    corr_rfe = {}
    corr_l_rfe = {}
    corr_r_rfe = {}

    if rfeobj:
        all_rfe_files = [os.path.join(rfe_obj_dir, x) for x in os.listdir(rfe_obj_dir) if ('rfetrain' in x)
                            and ('.joblib' in x)]

        corr_l_rfe = {x.split('\\')[-1].split('.')[0].split('_')[-1]:x for x in all_rfe_files if 'corr_l_' in x}
        corr_r_rfe = {x.split('\\')[-1].split('.')[0].split('_')[-1]: x for x in all_rfe_files if 'corr_r_' in x}
        corr_rfe = {x.split('\\')[-1].split('.')[0].split('_')[-1]: x for x in all_rfe_files if ('corr_' in x) and
                    ('corr_l_' not in x) and ('corr_r_' not in x)}
        nocorr_rfe = {x.split('\\')[-1].split('.')[0].split('_')[-1]: x for x in all_rfe_files if 'corr_' not in x}

    return (Xtest, Xtest_corr, Xtest_corr_l, Xtest_corr_r),df_test['labels'].values,(nocorr_rfe, corr_rfe, corr_l_rfe, corr_r_rfe)

def prepare_classifers_4_real_test(selected_clfs_tuple, Xtests, ytest, rfes, cl):
    selected_clfs, selected_clfs_corr, selected_clfs_corr_l, selected_clfs_corr_r = selected_clfs_tuple
    Xtest, Xtest_corr, Xtest_corr_l, Xtest_corr_r = Xtests
    nocorr_rfe, corr_rfe, corr_l_rfe, corr_r_rfe = rfes

    results_nocorr = defaultdict(dict)
    results_corr = defaultdict(dict)
    results_corr_l = defaultdict(dict)
    results_corr_r = defaultdict(dict)

    for clc in selected_clfs:
        best_clc = selected_clfs[clc].best_estimator_
        mean_std_cv = (selected_clfs[clc].cv_results_['mean_test_score'][selected_clfs[clc].best_index_],
                       selected_clfs[clc].cv_results_['std_test_score'][selected_clfs[clc].best_index_])
        y_hat = best_clc.predict(Xtest[:,np.where(load(nocorr_rfe[cl]).support_)[0]])
        report = classification_report(ytest, y_hat, output_dict=True)
        sensitivity, specificity, accuracy = report['1']['recall'], report['0']['recall'], report['accuracy']
        results_nocorr[cl][clc] = {
            'best_params': selected_clfs[clc].best_params_,
            'mean_cv_score':mean_std_cv[0],
            'std_cv_score': mean_std_cv[1],
            'sensitivity': sensitivity,
            'specificity': specificity,
            'acc': accuracy
        }

    for clc in selected_clfs_corr:
            best_clc = selected_clfs_corr[clc].best_estimator_
            mean_std_cv = (selected_clfs_corr[clc].cv_results_['mean_test_score'][selected_clfs_corr[clc].best_index_],
                           selected_clfs_corr[clc].cv_results_['std_test_score'][selected_clfs_corr[clc].best_index_])
            y_hat = best_clc.predict(Xtest_corr[:, np.where(load(corr_rfe[cl]).support_)[0]])
            report = classification_report(ytest, y_hat, output_dict=True)
            sensitivity, specificity, accuracy = report['1']['recall'], report['0']['recall'], report['accuracy']
            results_corr[cl][clc] = {
                'best_params': selected_clfs_corr[clc].best_params_,
                'mean_cv_score': mean_std_cv[0],
                'std_cv_score': mean_std_cv[1],
                'sensitivity': sensitivity,
                'specificity': specificity,
                'acc': accuracy
            }

    for clc in selected_clfs_corr_l:
            best_clc = selected_clfs_corr_l[clc].best_estimator_
            mean_std_cv = (selected_clfs_corr_l[clc].cv_results_['mean_test_score'][selected_clfs_corr_l[clc].best_index_],
                           selected_clfs_corr_l[clc].cv_results_['std_test_score'][selected_clfs_corr_l[clc].best_index_])
            y_hat = best_clc.predict(Xtest_corr_l[:, np.where(load(corr_l_rfe[cl]).support_)[0]])
            report = classification_report(ytest, y_hat, output_dict=True)
            sensitivity, specificity, accuracy = report['1']['recall'], report['0']['recall'], report['accuracy']
            results_corr_l[cl][clc] = {
                'best_params': selected_clfs_corr_l[clc].best_params_,
                'mean_cv_score': mean_std_cv[0],
                'std_cv_score': mean_std_cv[1],
                'sensitivity': sensitivity,
                'specificity': specificity,
                'acc': accuracy
            }

    for clc in selected_clfs_corr_r:
            best_clc = selected_clfs_corr_r[clc].best_estimator_
            mean_std_cv = (selected_clfs_corr_r[clc].cv_results_['mean_test_score'][selected_clfs_corr_r[clc].best_index_],
                           selected_clfs_corr_r[clc].cv_results_['std_test_score'][selected_clfs_corr_r[clc].best_index_])
            y_hat = best_clc.predict(Xtest_corr_r[:, np.where(load(corr_r_rfe[cl]).support_)[0]])
            report = classification_report(ytest, y_hat, output_dict=True)
            sensitivity, specificity, accuracy = report['1']['recall'], report['0']['recall'], report['accuracy']
            results_corr_r[cl][clc] = {
                'best_params': selected_clfs_corr_r[clc].best_params_,
                'mean_cv_score': mean_std_cv[0],
                'std_cv_score': mean_std_cv[1],
                'sensitivity': sensitivity,
                'specificity': specificity,
                'acc': accuracy
            }

    return results_nocorr, results_corr, results_corr_l, results_corr_r


def main():
    clcs = ['_rf', 'lg1','lg2','svm']
    df_test, df_test_l, df_test_r = load_test_data()
    best_acc = 0
    best_model = None
    for cl in clcs:
        clf_dir_up = clf_dir.replace('***',cl)
        clf_corr_dir_up = clf_corr_dir.replace('***',cl)
        clf_corr_l_dir_up = clf_corr_l_dir.replace('***',cl)
        clf_corr_r_dir_up = clf_corr_r_dir.replace('***',cl)
        if cl == '_rf':
            cl = 'rf'
        selected_clfs_tuple = load_analyze(clf_dir_up, clf_corr_dir_up, clf_corr_l_dir_up, clf_corr_r_dir_up)
        Xtests, ytest, rfes = prepare_test_data(df_test, df_test_l, df_test_r, norm=True, corr_ana=True, rfeobj=True)
        results_nocorr, results_corr, results_corr_l, results_corr_r = prepare_classifers_4_real_test(selected_clfs_tuple, Xtests, ytest, rfes, cl)

        for cls in results_nocorr[cl]:
            results_dict = results_nocorr[cl][cls]
            if results_dict['acc'] > best_acc:
                best_acc = results_dict['acc']
                best_model = (cl, cls,'nocorr', results_dict)

        for cls in results_corr[cl]:
            results_dict = results_corr[cl][cls]
            if results_dict['acc'] > best_acc:
                best_acc = results_dict['acc']
                best_model = (cl, cls,'corr', results_dict)

        for cls in results_corr_l[cl]:
            results_dict = results_corr_l[cl][cls]
            if results_dict['acc'] > best_acc:
                best_acc = results_dict['acc']
                best_model = (cl, cls,'corr_l', results_dict)

        for cls in results_corr_r[cl]:
            results_dict = results_corr_r[cl][cls]
            if results_dict['acc'] > best_acc:
                best_acc = results_dict['acc']
                best_model = (cl, cls, 'corr_r', results_dict)

    print(best_acc, best_model)
if __name__ == '__main__':
    main()