"""
What is the output of this analysis?
    - T1: (local analysis) For each site,
            -- what the maximum accuracy (CV, testing), &
            -- Features utilized to achieve this accuracy
    - T2: (Global analysis) Which site has the most common features with the top 16 selected features from the
       global analysis
    - T3: (Global analysis) What is the results if I used the 16 global selected features to train the best
       classifier for each site?
    - T4: (Local analysis) What is the difference between the sites with high accuracies and sites with low
       accuracies?
"""


import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import os
from sklearn.metrics import balanced_accuracy_score
from constants import OUTPUT_DIR_SPLIT, OUTPUT_DIR_CORR, OUTPUT_DIR_ML, OUTPUT_DIR_FS

# Constants
SITES = ['CMU', 'Leuven', 'MaxMun', 'Caltech', 'OHSU', 'Olin', 'Pitt', 'Trinity',
         'UCLA', 'UM', 'Yale', 'Stanford']


def read_test_files(site):
    fullbrain_path = os.path.join(OUTPUT_DIR_SPLIT, site, 'fullbrain_test.csv')
    leftbrain_path = os.path.join(OUTPUT_DIR_SPLIT, site, 'leftbrain_test.csv')
    rightbrain_path = os.path.join(OUTPUT_DIR_SPLIT, site, 'rightbrain_test.csv')

    return pd.read_csv(fullbrain_path, index_col=0),\
           pd.read_csv(leftbrain_path, index_col=0),\
           pd.read_csv(rightbrain_path, index_col=0)


def read_corr_files(site):
    fullbrain_path = os.path.join(OUTPUT_DIR_CORR, site, 'fullbrain_corr.csv')
    leftbrain_path = os.path.join(OUTPUT_DIR_CORR, site, 'leftbrain_corr.csv')
    rightbrain_path = os.path.join(OUTPUT_DIR_CORR, site, 'rightbrain_corr.csv')

    return pd.read_csv(fullbrain_path, index_col=0),\
           pd.read_csv(leftbrain_path, index_col=0),\
           pd.read_csv(rightbrain_path, index_col=0)


def read_fs_files(site):
    rfe_dict = defaultdict(dict)
    Xtrain_dict = defaultdict(dict)
    ytrain_dict = defaultdict(dict)
    normalizers_dict = defaultdict(dict)

    normalizers_path = os.path.join(OUTPUT_DIR_FS, site)
    fs_files_path = os.path.join(OUTPUT_DIR_FS, site, 'minmaxreg')

    all_files = [os.path.join(fs_files_path, x) for x in os.listdir(fs_files_path)
                 if ('.npy' in x) or ('.joblib' in x)]

    all_normalizers = [os.path.join(normalizers_path, x) for x in os.listdir(normalizers_path)
                 if '.joblib' in x]

    for normalizer in all_normalizers:
        file_name = normalizer.split('\\')[-1]
        if '_corr_l_' in file_name:
            normalizers_dict['lg1']['corr_l'] = normalizer
            normalizers_dict['lg2']['corr_l'] = normalizer
            normalizers_dict['rf']['corr_l'] = normalizer
            normalizers_dict['svm']['corr_l'] = normalizer
        elif '_corr_r_' in file_name:
            normalizers_dict['lg1']['corr_r'] = normalizer
            normalizers_dict['lg2']['corr_r'] = normalizer
            normalizers_dict['rf']['corr_r'] = normalizer
            normalizers_dict['svm']['corr_r'] = normalizer
        elif '_corr_' in file_name:
            normalizers_dict['lg1']['corr'] = normalizer
            normalizers_dict['lg2']['corr'] = normalizer
            normalizers_dict['rf']['corr'] = normalizer
            normalizers_dict['svm']['corr'] = normalizer
        else:
            normalizers_dict['lg1']['NA'] = normalizer
            normalizers_dict['lg2']['NA'] = normalizer
            normalizers_dict['rf']['NA'] = normalizer
            normalizers_dict['svm']['NA'] = normalizer

    for file in all_files:
        file_name = file.split('\\')[-1]
        if 'rfetrain' in file_name:
            if '_corr_l_' in file_name:
                clc_name = file_name.split('_')[3].split('.')[0]
                rfe_dict[clc_name]['corr_l'] = 'NA' if 'noneed' in file_name else file
            elif '_corr_r_' in file_name:
                clc_name = file_name.split('_')[3].split('.')[0]
                rfe_dict[clc_name]['corr_r'] = 'NA' if 'noneed' in file_name else file
            elif '_corr_' in file_name:
                clc_name = file_name.split('_')[2].split('.')[0]
                rfe_dict[clc_name]['corr'] = 'NA' if 'noneed' in file_name else file
            else:
                clc_name = file_name.split('_')[1].split('.')[0]
                rfe_dict[clc_name]['NA'] = 'NA' if 'noneed' in file_name else file

        elif 'Xtrain' in file_name:
            if '_corr_l_' in file_name:
                clc_name = file_name.split('_')[3].split('.')[0]
                Xtrain_dict[clc_name]['corr_l'] = file
            elif '_corr_r_' in file_name:
                clc_name = file_name.split('_')[3].split('.')[0]
                Xtrain_dict[clc_name]['corr_r'] = file
            elif '_corr_' in file_name:
                clc_name = file_name.split('_')[2].split('.')[0]
                Xtrain_dict[clc_name]['corr'] = file
            else:
                clc_name = file_name.split('_')[1].split('.')[0]
                Xtrain_dict[clc_name]['NA'] = file

        elif 'ytrain' in file_name:
            if '_corr_l' in file_name:
                ytrain_dict['lg1']['corr_l'] = file
                ytrain_dict['lg2']['corr_l'] = file
                ytrain_dict['rf']['corr_l'] = file
                ytrain_dict['svm']['corr_l'] = file

            elif '_corr_r' in file_name:
                ytrain_dict['lg1']['corr_r'] = file
                ytrain_dict['lg2']['corr_r'] = file
                ytrain_dict['rf']['corr_r'] = file
                ytrain_dict['svm']['corr_r'] = file
            elif '_corr' in file_name:
                ytrain_dict['lg1']['corr'] = file
                ytrain_dict['lg2']['corr'] = file
                ytrain_dict['rf']['corr'] = file
                ytrain_dict['svm']['corr'] = file
            else:
                ytrain_dict['lg1']['NA'] = file
                ytrain_dict['lg2']['NA'] = file
                ytrain_dict['rf']['NA'] = file
                ytrain_dict['svm']['NA'] = file

    return rfe_dict, Xtrain_dict, ytrain_dict, normalizers_dict


def plot_rfe(rfe_dict, pdf_file):
    pp = PdfPages(pdf_file)
    for clc in rfe_dict.keys():
        corr_type = rfe_dict[clc]
        for corr in corr_type.keys():
            file = corr_type[corr]
            if file != 'NA':
                rfe = load(file)
                f1 = plt.figure(1, figsize=(22, 18))
                plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_, label=f'{clc}-{corr}')
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.xlabel('Number of features', fontdict={'size': 14, 'weight': 'bold'})
                plt.ylabel('Accuracy', fontdict={'size': 14, 'weight': 'bold'})
                plt.legend(prop={'size': 12, 'weight': 'bold'})
                pp.savefig(f1)
                f1.clear()
    pp.close()


def get_selectedfeats_names(rfe_dict, df):
    selected_feats_names_scores = defaultdict(dict)
    for clc in rfe_dict.keys():
        corr_type = rfe_dict[clc]
        for corr in corr_type.keys():
            file = corr_type[corr]
            if file != 'NA':
                rfe = load(file)
                selected_feats_names_scores[clc][corr] =\
                    (df.columns[np.where(rfe.support_)[0]], np.max(rfe.grid_scores_))

    return selected_feats_names_scores


def read_ml_files(site):
    ml_files_dir = os.path.join(OUTPUT_DIR_ML, site)
    all_ml_files = [os.path.join(ml_files_dir, x) for x in os.listdir(ml_files_dir) if '.joblib' in x]
    ml_file_dict = defaultdict(dict)

    for ml_file in all_ml_files:
        file_name = ml_file.split('\\')[-1]
        if '_corr_l' in file_name:
            if '__rf_' in file_name:
                ml_file_dict['rf']['corr_l'] = ml_file
            else:
                ml_file_dict[file_name.split('_')[1]]['corr_l'] = ml_file
        elif '_corr_r' in file_name:
            if '__rf_' in file_name:
                ml_file_dict['rf']['corr_r'] = ml_file
            else:
                ml_file_dict[file_name.split('_')[1]]['corr_r'] = ml_file
        elif '_corr' in file_name:
            if '__rf_' in file_name:
                ml_file_dict['rf']['corr'] = ml_file
            else:
                ml_file_dict[file_name.split('_')[1]]['corr'] = ml_file
        else:
            if '__rf_' in file_name:
                ml_file_dict['rf']['NA'] = ml_file
            else:
                ml_file_dict[file_name.split('_')[1]]['NA'] = ml_file

    return ml_file_dict


def load_ml_models(ml_file_dict, site, rfe_dict, Xtrain_dict, ytrain_dict, dftest):
    results_dict = defaultdict(list)

    dftest_dict = dict()
    dftest_fb, dftest_corr, dftest_l, dftest_r = dftest
    dftest_dict['NA'] = dftest_fb
    dftest_dict['corr'] = dftest_corr
    dftest_dict['corr_r'] = dftest_r
    dftest_dict['corr_l'] = dftest_l

    for rfe_clc in ml_file_dict:
        corr_type = ml_file_dict[rfe_clc]
        for corr in corr_type:
            clf_dict = load(corr_type[corr])
            Xtest, ytest = dftest_dict[corr].drop('labels', axis=1).values, dftest_dict[corr]['labels'].values
            for clc_name in clf_dict.keys():
                clf = clf_dict[clc_name]
                rfe = load(rfe_dict[rfe_clc][corr]) if rfe_dict[rfe_clc][corr] != 'NA' else None
                Xtrain = np.load(Xtrain_dict[rfe_clc][corr])
                ytrain = np.load(ytrain_dict[rfe_clc][corr])
                n_feats = Xtrain.shape[1]
                baseline_acc = (len(np.where(ytrain==0)[0]))/(len(np.where(ytrain==1)[0])+len(np.where(ytrain==0)[0])) \
                    if len(np.where(ytrain==0)[0])>len(np.where(ytrain==1)[0])\
                    else (len(np.where(ytrain==1)[0]))/(len(np.where(ytrain==0)[0])+len(np.where(ytrain==1)[0]))

                best_clc = clf.best_estimator_
                if best_clc.fit(Xtrain, ytrain) is None:
                    best_clc.fit(Xtrain, ytrain)
                else:
                    best_clc = best_clc.fit(Xtrain, ytrain)
                ypred = best_clc.predict(Xtrain)
                train_acc = balanced_accuracy_score(ytrain, ypred)
                if rfe is not None:
                    try:
                        test_acc = balanced_accuracy_score(ytest, best_clc.predict(rfe.transform(Xtest)))
                    except:
                        print(f'feature_selection: {rfe.support_.sum()}')
                        print(f'Xtrain: {Xtrain.shape}')
                        print(f'rfe_clc: {rfe_clc}')
                        print(f'corr: {corr}')
                        raise ValueError(f'Error is in site: {site} with rfe object')

                else:
                    try:
                        test_acc = balanced_accuracy_score(ytest, best_clc.predict(Xtest))
                    except:
                        print(f'Xtrain: {Xtrain.shape}')
                        print(f'Xtest: {Xtest.shape}')
                        print(f'rfe_clc: {rfe_clc}')
                        print(f'corr: {corr}')
                        raise ValueError(f'Error is in site: {site} without rfe object')

                results_dict['Correlation_type'].append(corr)
                results_dict['RFE_clc'].append(rfe_clc)
                results_dict['Number_of_feats'].append(n_feats)
                results_dict['classifier'].append(clc_name)
                results_dict['Best_params'].append(clf.best_params_)
                results_dict['Baseline_acc'].append(baseline_acc)
                results_dict['Best_CV_acc'].append(clf.best_score_)
                results_dict['Train_acc'].append(train_acc)
                results_dict['Test_acc'].append(test_acc)

    df_results = pd.DataFrame(results_dict)
    df_results.to_csv(os.path.join(OUTPUT_DIR_ML, site, 'ML_results.csv'))
    return df_results


def T1():
    """

    """
    pass


def main():
    """
    1. Read the output folders and parse the results in a systematic way to have all what you need for
     each site
    """
    for site in SITES[3:]: # To [:3] to be removed
        dftest_site, dftest_l_site, dftest_r_site = read_test_files(site)
        dftrain_corr_site, dftrain_corr_l_site, dftrain_corr_r_site = read_corr_files(site)

        dftest_corr_site = dftest_site.loc[:,dftrain_corr_site.columns]
        dftest_corr_l_site = dftest_site.loc[:,dftrain_corr_l_site.columns]
        dftest_corr_r_site = dftest_site.loc[:,dftrain_corr_r_site.columns]

        rfe_dict, Xtrain_dict, ytrain_dict, norm_dict = read_fs_files(site)
        plot_rfe(rfe_dict, os.path.join(OUTPUT_DIR_FS, site, 'minmaxreg', 'figures.pdf'))
        ml_dict = read_ml_files(site)
        if site == 'Caltech':
            x=0
        df_results = load_ml_models(ml_dict,site,rfe_dict, Xtrain_dict, ytrain_dict,
                       (dftest_site, dftest_corr_site, dftest_corr_l_site, dftest_corr_r_site))
        print(ml_dict)
        # break


if __name__ == '__main__':
    main()