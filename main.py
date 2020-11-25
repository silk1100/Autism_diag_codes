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
import seaborn as sns
import constants
from prepare_data import get_csvfile_ready, select_sites
import matplotlib.pyplot as plt
from correlation_analysis import correlation_analysis
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from RFE_FS import select_features
from constants import MAX_ITR, OUTPUT_DIR_CORR, OUTPUT_DIR_FS, OUTPUT_DIR_ML, OUTPUT_DIR_SPLIT
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from ML_model_selection import train_models
from constants import  DROPBOX_API_TOKEN


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


def split_l_r(df):
    rcols = []
    lcols = []
    for col in df.columns:
        if col.split('_')[1][0] == 'l':
            lcols.append(col)
        elif col.split('_')[1][0] == 'r':
            rcols.append(col)
        else:
            raise ValueError(f'Undefined columns: {col}')
    return df.loc[:, lcols], df.loc[:, rcols]


def stat_analysis_MaterialsMethods_DataCollection_Age_Sex(df):
    from scipy.stats import ttest_ind, chi2_contingency
    # t-test for age
    age_asd = df[df['labels']==1]['age'].values
    age_td = df[df['labels']==0]['age'].values
    print(f't-test for age between asd and td group: {ttest_ind(age_asd, age_td)}')

    # Chi-square test for sex in TD group
    td_sex_count = df[df['labels']==0]['sex'].value_counts()
    males_td = td_sex_count[1]
    females_td = td_sex_count[2]

    td_table = [[males_td, females_td],[(males_td+females_td)//2,(males_td+females_td)//2]]
    print(f'chi-square test for gender in TD group: {chi2_contingency(td_table)}')

    # Chi-square test for sex in ASD group
    td_sex_count = df[df['labels'] == 1]['sex'].value_counts()
    males_asd = td_sex_count[1]
    females_asd = td_sex_count[2]

    td_table = [[males_asd, females_asd], [(males_asd + females_asd) // 2, (males_asd + females_asd) // 2]]
    print(f'chi-square test for gender in ASD group: {chi2_contingency(td_table)}')


class MyMinMax:
    def __init__(self, axis):
        self.sc = MinMaxScaler()
        self.axis = axis

    def fit(self, X):
        if self.axis==1:
            self.sc = self.sc.fit(X.transpose())
        elif self.axis==0:
            self.sc = self.sc.fit(X)
        return self.sc

    def transform(self, X):
        if self.axis==1:
            Xn = self.sc.transform(X.transpose()).transpose()
        elif self.axis==0:
            Xn = self.sc.transform(X)
        return Xn

    def fit_transform(self, X):
        if self.axis==1:
            self.sc = self.sc.fit(X.transpose())
            Xn = self.sc.transform(X.transpose()).transpose()
        elif self.axis==0:
            self.sc = self.sc.fit(X)
            Xn = self.sc.transform(X)
        return Xn


def loadRFEFilesFrom(morphBasedNormList):
    rfes = ['lg1','lg2','svm','_rf']
    data_dict = {}
    for rfe in rfes:
        ytrain_corr = [x for x in morphBasedNormList if ('ytrain_corr' in x)
                       and ("_corr_l" not in x) and ("_corr_r" not in x)]
        assert(len(ytrain_corr) == 1)
        ytrain_corr = ytrain_corr[0]

        ytrain = [x for x in morphBasedNormList if ('ytrain' in x)
                  and ("_corr" not in x) and ("_corr_l" not in x) and ("_corr_r" not in x)]

        assert(len(ytrain) == 1)
        ytrain = ytrain[0]

        ytrain_corr_l = [x for x in morphBasedNormList if ('ytrain_corr_l' in x)]
        assert(len(ytrain_corr_l) == 1)
        ytrain_corr_l = ytrain_corr_l[0]


        ytrain_corr_r = [x for x in morphBasedNormList if ('ytrain_corr_r' in x)]
        assert(len(ytrain_corr_r) == 1)
        ytrain_corr_r = ytrain_corr_r[0]

        rfe_files = [x for x in morphBasedNormList if rfe in x]
        Xtrain_corr_r, rfe_corr_r  = [x for x in rfe_files if ('_corr_r_' in x) and ('Xtrain' in x)], \
                                     [x for x in rfe_files if ('_corr_r_' in x) and ('rfe' in x) and (rfe in x)]

        assert(len(Xtrain_corr_r) == 1)
        Xtrain_corr_r = Xtrain_corr_r[0]
        assert(len(rfe_corr_r) == 1)
        rfe_corr_r = rfe_corr_r[0]


        Xtrain_corr_l, rfe_corr_l  = [x for x in rfe_files if ('_corr_l_' in x) and ('Xtrain' in x)], \
                                     [x for x in rfe_files if ('_corr_l_' in x) and ('rfe' in x) and (rfe in x)]
        assert(len(Xtrain_corr_l) == 1)
        Xtrain_corr_l = Xtrain_corr_l[0]
        assert(len(rfe_corr_l) == 1)
        rfe_corr_l = rfe_corr_l[0]

        Xtrain_corr, rfe_corr = [x for x in rfe_files
                          if ('_corr_' in x) and ('_corr_l_' not in x) and ('_corr_r_' not in x)
                                 and ('Xtrain' in x)], \
                                [x for x in rfe_files
                                 if ('_corr_' in x) and ('_corr_l_' not in x) and ('_corr_r_' not in x)
                                 and ('rfe' in x) and (rfe in x)]
        assert(len(Xtrain_corr) == 1)
        Xtrain_corr = Xtrain_corr[0]
        assert(len(rfe_corr) == 1)
        rfe_corr = rfe_corr[0]

        Xtrain, rfetrain = [x for x in rfe_files
                          if ('_corr_' not in x) and ('_corr_l_' not in x) and ('_corr_r_' not in x)
                            and ('Xtrain' in x)], \
                           [x for x in rfe_files
                            if ('_corr_' not in x) and ('_corr_l_' not in x) and ('_corr_r_' not in x)
                            and ('rfe' in x) and (rfe in x)]

        assert(len(Xtrain) == 1)
        Xtrain = Xtrain[0]
        assert(len(rfetrain) == 1)
        rfetrain = rfetrain[0]

        data_dict[rfe] = {
            'ytrain': ytrain,
            'ytrain_corr': ytrain_corr,
            'ytrain_corr_l':ytrain_corr_l,
            'ytrain_corr_r': ytrain_corr_r,
            'Xtrain': Xtrain,
            'Xtrain_corr': Xtrain_corr,
            'Xtrain_corr_l': Xtrain_corr_l,
            'Xtrain_corr_r': Xtrain_corr_r,
            'rfetrain': rfetrain,
            'rfe_corr': rfe_corr,
            'rfe_corr_l': rfe_corr_l,
            'rfe_corr_r': rfe_corr_r
        }
    return data_dict


def mynormalize(df, allfeats=False):
    """
    Normalize over the brain regions of each subject. Normalization is occured either over all morphological
    features by setting "allfeats=True". In this case, The maximum value of a morphological features within a
    subjects brain will be 1 i.e. volume, while other morphological features will take values from 0 to 1.
    While if "allfeats=False", then the normalization will occur within each morphological regions so let's
    say brain region with minimum surface area value will be set to 0 and maximum surface area values will be
    set to 1, same goes to volume, curv, ..etc. Thus, for each subject there will be zeros and ones with the
    same amount of the morphological features. The rationale behind this is to represent each subject by how his
    brain varies in morphological features w.r.t to himself. Thus, it is more like looking at correlation matrix
    or difference matrix of all subjects where we care more about the relationship of features across subjects
    rather than the absolute value of each feature.
    :param X: pandas.DataFrame [Data matrix without the labels]
    :param allfeats: bool
    :return Xn, scalersdict: numpy.array [Data normalized], dict [fitted MinMaxScalers]
    """
    scalersdict = {}
    if allfeats:
        sc = MyMinMax(axis=1)
        XN = sc.fit_transform(df.values)
        scalersdict['allfeat'] = sc
    else:
        morph_feats = ['area', 'curv', 'thickness', 'volume']
        XN = np.array([], dtype=np.double)
        for ind, morph_feat in enumerate(morph_feats):
            morph_cols = [col for col in df.columns if morph_feat in col]
            df_morph = df.loc[:, morph_cols]
            sc = MyMinMax(axis=1)
            Xn = sc.fit_transform(df_morph.values)
            if ind == 0:
                XN = np.append(XN, Xn).reshape(Xn.shape[0], -1)
            else:
                XN = np.concatenate([XN, Xn], axis=1)
            scalersdict[morph_feat] = sc
    return XN, scalersdict


def PIPELINE_PART1_DATA_PREPROCESSING(data_dir, write_on_disk=True):
    """
    data_dir is supposed to be the directory of the correct data representation (median-iqr, median+iqr) of
    the full dataset. The output will be a tuple of size 2: tuple[0]: 3 dataframes (full brain of subjects
    within selected unbiased sites, left hemisphere of subjects within selected unbiased sites, right hemisphere
    of subjects within selected unbiased sites), tuple[1]:(age series, gender series, label series)
    :param data_dir: str
    :return: list[df, df_left, df_right]
    """
    ## Read data
    # For every site:
    #   If number of ASD subjects/number of td subjects>0.6 (or other wise), then drop the site
    # Include only sites with balanced sites
    df = pd.read_csv(data_dir, index_col=0)
    df, info = select_sites(df)
    # Stats for the paper
    stat_analysis_MaterialsMethods_DataCollection_Age_Sex(df)

    # Separate unnecessary
    age = df.pop('age')
    sex = df.pop('sex')
    labels = df.pop('labels')

    # Split data into left and right hemisphere
    df_l, df_r = split_l_r(df)

    if write_on_disk:
        # Save files
        df.to_csv(os.path.join(OUTPUT_DIR_SPLIT, 'selected_sites_full_df.csv'))
        df_l.to_csv(os.path.join(OUTPUT_DIR_SPLIT, 'selected_sites_leftH_noAgeSexLabels_df.csv'))
        df_r.to_csv(os.path.join(OUTPUT_DIR_SPLIT, 'selected_sites_leftR_noAgeSexLabels_df.csv'))

    return (df, df_l, df_r), (age, sex, labels)


def PIPELINE_PART1_CORR_ANALYSIS(dftrain, labelstrain, corr_thresh, file_name=None, write_on_disk=True):

    dftrain_corr = correlation_analysis(pd.concat([dftrain, labelstrain], axis=1), corr_thresh=corr_thresh/100.0)

    if write_on_disk:
        if file_name is None:
            raise ValueError('To write a file on disk, you must specify file name')
        dftrain_corr.to_csv(os.path.join(OUTPUT_DIR_CORR, file_name))

    return dftrain_corr


def PIPELINE_PART2_FS_rf(Xtrain_norm_site, Xtrain_corr_norm_site, Xtrain_corr_l_norm_site, Xtrain_corr_r_norm_site,
                         ytrain_site,
                         site, normtype):
    rf = RandomForestClassifier(n_estimators=250, max_depth=5000)
    _PIPELINE_PART2_FS(rf, Xtrain_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '', 'rf')

    if Xtrain_corr_norm_site.shape[1] > 25:
        rf = RandomForestClassifier(n_estimators=250, max_depth=5000)
        _PIPELINE_PART2_FS(rf, Xtrain_corr_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr', 'rf')
    else:
        dump(Xtrain_corr_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_rf_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_rf.npy'), Xtrain_corr_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr.npy'), ytrain_site)

    if Xtrain_corr_l_norm_site.shape[1] > 25:
        rf = RandomForestClassifier(n_estimators=250, max_depth=5000)
        _PIPELINE_PART2_FS(rf, Xtrain_corr_l_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr_l', 'rf')
    else:
        dump(Xtrain_corr_l_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_l_rf_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_l_rf.npy'), Xtrain_corr_l_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr_l.npy'), ytrain_site)

    if Xtrain_corr_r_norm_site.shape[1] > 25:
        rf = RandomForestClassifier(n_estimators=250, max_depth=5000)
        _PIPELINE_PART2_FS(rf, Xtrain_corr_r_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr_r', 'rf')
    else:
        dump(Xtrain_corr_r_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_r_rf_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_r_rf.npy'), Xtrain_corr_r_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr_r.npy'), ytrain_site)


def PIPELINE_PART2_FS_svm(Xtrain_norm_site, Xtrain_corr_norm_site, Xtrain_corr_l_norm_site, Xtrain_corr_r_norm_site,
                     ytrain_site, site, normtype, MAX_ITR=10000000):

    svm = LinearSVC(max_iter=MAX_ITR)
    _PIPELINE_PART2_FS(svm, Xtrain_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '', 'svm')

    if Xtrain_corr_norm_site.shape[1] > 25:
        svm = LinearSVC(max_iter=MAX_ITR)
        _PIPELINE_PART2_FS(svm, Xtrain_corr_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr', 'svm')
    else:
        dump(Xtrain_corr_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_svm_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_svm.npy'), Xtrain_corr_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr.npy'), ytrain_site)

    if Xtrain_corr_l_norm_site.shape[1] > 25:
        svm = LinearSVC(max_iter=MAX_ITR)
        _PIPELINE_PART2_FS(svm, Xtrain_corr_l_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr_l', 'svm')
    else:
        dump(Xtrain_corr_l_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_l_svm_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_l_svm.npy'), Xtrain_corr_l_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr_l.npy'), ytrain_site)

    if Xtrain_corr_r_norm_site.shape[1] > 25:
        svm = LinearSVC(max_iter=MAX_ITR)
        _PIPELINE_PART2_FS(svm, Xtrain_corr_r_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr_r', 'svm')
    else:
        dump(Xtrain_corr_r_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_r_svm_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_r_svm.npy'), Xtrain_corr_r_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr_r.npy'), ytrain_site)


def PIPELINE_PART2_FS_lg2(Xtrain_norm_site, Xtrain_corr_norm_site, Xtrain_corr_l_norm_site, Xtrain_corr_r_norm_site,
                     ytrain_site, site, normtype, MAX_ITR=10000000):

    lg2 = LogisticRegression(penalty='l2', max_iter=MAX_ITR, solver='saga')
    _PIPELINE_PART2_FS(lg2, Xtrain_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '', 'lg2')

    if Xtrain_corr_norm_site.shape[1] > 25:
        lg2 = LogisticRegression(penalty='l2', max_iter=MAX_ITR, solver='saga')
        _PIPELINE_PART2_FS(lg2, Xtrain_corr_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr', 'lg2')
    else:
        dump(Xtrain_corr_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_lg2_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_lg2.npy'), Xtrain_corr_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr.npy'), ytrain_site)

    if Xtrain_corr_l_norm_site.shape[1] > 25:
        lg2 = LogisticRegression(penalty='l2', max_iter=MAX_ITR, solver='saga')
        _PIPELINE_PART2_FS(lg2, Xtrain_corr_l_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr_l', 'lg2')
    else:
        dump(Xtrain_corr_l_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_l_lg2_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_l_lg2.npy'), Xtrain_corr_l_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr_l.npy'), ytrain_site)

    if Xtrain_corr_r_norm_site.shape[1] > 25:
        lg2 = LogisticRegression(penalty='l2', max_iter=MAX_ITR, solver='saga')
        _PIPELINE_PART2_FS(lg2, Xtrain_corr_r_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr_r', 'lg2')
    else:
        dump(Xtrain_corr_r_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_r_lg2_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_r_lg2.npy'), Xtrain_corr_r_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr_r.npy'), ytrain_site)


def PIPELINE_PART2_FS_lg1(Xtrain_norm_site, Xtrain_corr_norm_site, Xtrain_corr_l_norm_site, Xtrain_corr_r_norm_site,
                          ytrain_site, site, normtype, MAX_ITR=10000000):

    lg1 = LogisticRegression(penalty='l1', max_iter=MAX_ITR, solver='saga')
    _PIPELINE_PART2_FS(lg1, Xtrain_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '', 'lg1')

    if Xtrain_corr_norm_site.shape[1] > 25:
        lg1 = LogisticRegression(penalty='l1', max_iter=MAX_ITR, solver='saga')
        _PIPELINE_PART2_FS(lg1, Xtrain_corr_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr', 'lg1')
    else:
        dump(Xtrain_corr_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_lg1_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_lg1.npy'), Xtrain_corr_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr.npy'), ytrain_site)

    if Xtrain_corr_l_norm_site.shape[1] > 25:
        lg1 = LogisticRegression(penalty='l1', max_iter=MAX_ITR, solver='saga')
        _PIPELINE_PART2_FS(lg1, Xtrain_corr_l_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr_l', 'lg1')
    else:
        dump(Xtrain_corr_l_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_l_lg1_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_l_lg1.npy'), Xtrain_corr_l_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr_l.npy'), ytrain_site)

    if Xtrain_corr_r_norm_site.shape[1] > 25:
        lg1 = LogisticRegression(penalty='l1', max_iter=MAX_ITR, solver='saga')
        _PIPELINE_PART2_FS(lg1, Xtrain_corr_r_norm_site, ytrain_site, 'balanced_accuracy', site, normtype, '_corr_r', 'lg1')
    else:
        dump(Xtrain_corr_r_norm_site, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain_corr_r_lg1_noneed.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain_corr_r_lg1.npy'), Xtrain_corr_r_norm_site)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain_corr_r.npy'), ytrain_site)


def _PIPELINE_PART2_FS(clc, Xtrain, ytrain, scoring_metric, site, normtype, datype, clc_name):
    # rf = RandomForestClassifier(n_estimators=250, max_depth=5000)
    if not os.path.isfile(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain{datype}_{clc_name}.npy')):
        Xtrain_clc, ytrain, rfetrain_clc = select_features(clc, Xtrain,
                                                         ytrain,
                                                         scoring_metric=scoring_metric,
                                                         save_file=False)

        if not os.path.isdir(os.path.join(OUTPUT_DIR_FS, site, normtype)):
            os.mkdir(os.path.join(OUTPUT_DIR_FS, site, normtype))

        dump(rfetrain_clc, os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/rfetrain{datype}_{clc_name}.joblib'))
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/Xtrain{datype}_{clc_name}.npy'), Xtrain_clc)
        np.save(os.path.join(OUTPUT_DIR_FS, site, f'{normtype}/ytrain{datype}.npy'), ytrain)


def PIPELINE_PART3_ML(morphBasedNormList, site):
    data_dict = loadRFEFilesFrom(morphBasedNormList)
    for rfe_clc in data_dict:
        print(rfe_clc)
        rfe_clc_data = data_dict[rfe_clc]
        rfe_clc = rfe_clc.split('.')[0]

        clf = train_models(np.load(rfe_clc_data['Xtrain'], allow_pickle=True),
                           np.load(rfe_clc_data['ytrain'], allow_pickle=True), 5)
        dump(clf, os.path.join(OUTPUT_DIR_ML,site, f"clf_{rfe_clc}_train.joblib"))

        clf = train_models(np.load(rfe_clc_data['Xtrain_corr'],allow_pickle=True),
                           np.load(rfe_clc_data['ytrain_corr'],allow_pickle=True), 5)
        dump(clf, os.path.join(OUTPUT_DIR_ML,site, f"clf_{rfe_clc}_train_corr.joblib"))

        clf = train_models(np.load(rfe_clc_data['Xtrain_corr_l'], allow_pickle=True),
                           np.load(rfe_clc_data['ytrain_corr_l'],allow_pickle=True), 5)
        dump(clf, os.path.join(OUTPUT_DIR_ML,site, f"clf_{rfe_clc}_train_corr_l.joblib"))

        clf = train_models(np.load(rfe_clc_data['Xtrain_corr_r'], allow_pickle=True),
                           np.load(rfe_clc_data['ytrain_corr_r'], allow_pickle=True), 5)
        dump(clf, os.path.join(OUTPUT_DIR_ML,site, f"clf_{rfe_clc}_train_corr_r.joblib"))


def PIPELINE_PART3_ML_ANALYSIS():
    pass


def main():
    # Create output folders
    createDirIfNotExist_max2levels(OUTPUT_DIR_ML)
    createDirIfNotExist_max2levels(OUTPUT_DIR_SPLIT)
    createDirIfNotExist_max2levels(OUTPUT_DIR_FS)
    createDirIfNotExist_max2levels(OUTPUT_DIR_CORR)


    ## Read data
    # For every site:
    #   If number of ASD subjects/number of td subjects>0.6 (or other wise), then drop the site
    # Include only sites with balanced sites
    data_dir = "D:\\PhD\\Data\\aparc\\DrEid_brain_sMRI_lr_TDASD.csv"
    dfs, series = PIPELINE_PART1_DATA_PREPROCESSING(data_dir, write_on_disk=False)
    df, df_l, df_r = dfs
    age, sex, labels = series

    # Calculate the difference between left and right hemisphere (using numpy not pandas)
    # print(np.sum(df_l.values-df_r.values)) # = 286.372303 (not zero as pandas produce)
    # for indx, col in enumerate(df_l.columns):
    #     print(f'Correlation coef of {col} = {np.corrcoef(df_r.iloc[:, indx].values, df_l.iloc[:, indx].values)}')

    # sns.heatmap(df.corr())
    # plt.legend()
    # plt.show()
    """
    """

    # Separate a testset before the pipline begins
    # from sklearn.model_selection import train_test_split
    # dftrain, dftest, labelstrain, labelstest = train_test_split(df, labels, test_size=0.1, random_state=42)
    # dftrain_l, dftrain_r = split_l_r(dftrain)
    # pd.concat([dftrain, labelstrain], axis=1).to_csv(os.path.join(OUTPUT_DIR_SPLIT, 'train_fullbrain.csv'))
    # pd.concat([dftest, labelstest], axis=1).to_csv(os.path.join(OUTPUT_DIR_SPLIT, 'test_fullbrain.csv'))

    """
    1. Split data set into smaller data sets for each site from the selected sites
    2. Run the whole pipeline for each site
    3. Save results w.r.t site names
    """
    subjs = df.index
    from collections import defaultdict
    sites_dict = defaultdict(list)

    for subj in subjs:
        sitename = subj.split('_')[0]
        sites_dict[sitename].append(subj)
    sites = ['Leuven',
        'MaxMun',
        'CMU']
    for site in sites:
        print('site: ', site)

        # Create output folders for this site
        for fldr in [OUTPUT_DIR_SPLIT, OUTPUT_DIR_CORR, OUTPUT_DIR_FS, OUTPUT_DIR_ML]:
            new_fldr = os.path.join(fldr, site)
            if not os.path.isdir(new_fldr):
                os.mkdir(new_fldr)

        df_site = df.loc[sites_dict[site], :]
        labels_site = labels[sites_dict[site]]

        if not os.path.isfile(os.path.join(OUTPUT_DIR_SPLIT, site, 'fullbrain_train.csv')):
            from sklearn.model_selection import train_test_split
            df_train_site, df_test_site, y_train_site, y_test_site = train_test_split(df_site, labels_site)

            dftrain_l_site, dftrain_r_site = split_l_r(df_train_site)
            dftest_l_site, dftest_r_site = split_l_r(df_test_site)

            pd.concat([df_train_site, y_train_site], axis=1).to_csv(os.path.join(OUTPUT_DIR_SPLIT, site, 'fullbrain_train.csv'))
            pd.concat([df_test_site, y_test_site], axis=1).to_csv(os.path.join(OUTPUT_DIR_SPLIT, site, 'fullbrain_test.csv'))

            pd.concat([dftrain_l_site, y_train_site], axis=1).to_csv(os.path.join(OUTPUT_DIR_SPLIT, site, 'leftbrain_train.csv'))
            pd.concat([dftest_l_site, y_test_site], axis=1).to_csv(os.path.join(OUTPUT_DIR_SPLIT, site, 'leftbrain_test.csv'))

            pd.concat([dftrain_r_site, y_train_site], axis=1).to_csv(os.path.join(OUTPUT_DIR_SPLIT, site, 'rightbrain_train.csv'))
            pd.concat([dftest_r_site, y_test_site], axis=1).to_csv(os.path.join(OUTPUT_DIR_SPLIT, site, 'rightbrain_test.csv'))
        else:
            df_train_site = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT, site, 'fullbrain_train.csv'), index_col=0)
            dftrain_l_site = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT, site, 'leftbrain_train.csv'), index_col=0)
            dftrain_r_site = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT, site, 'rightbrain_train.csv'), index_col=0)
            y_train_site = df_train_site.pop('labels')
            dftrain_l_site.drop('labels', axis=1, inplace=True)
            dftrain_r_site.drop('labels', axis=1, inplace=True)

        ## Pipeline Begins
        # 1. Perform correlation analysis
        corr_thresh = 50
        #1a. Correlation analysis over the whole brain
        if not os.path.isfile(os.path.join(OUTPUT_DIR_CORR, site, 'fullbrain_corr.csv')):
            dftrain_corr = PIPELINE_PART1_CORR_ANALYSIS(df_train_site, y_train_site, corr_thresh,
                                                    file_name=f'{site}/fullbrain_corr.csv', write_on_disk=True)
        else:
            dftrain_corr = pd.read_csv(os.path.join(OUTPUT_DIR_CORR, site, 'fullbrain_corr.csv'), index_col=0)

        # 1b. Correlation analysis over the left hemisphere
        if not os.path.isfile(os.path.join(OUTPUT_DIR_CORR, site, 'leftbrain_corr.csv')):
            dftrain_l_corr = PIPELINE_PART1_CORR_ANALYSIS(dftrain_l_site, y_train_site, corr_thresh,
                                                    file_name=f'{site}/leftbrain_corr.csv', write_on_disk=True)
        else:
            dftrain_l_corr = pd.read_csv(os.path.join(OUTPUT_DIR_CORR, site, 'leftbrain_corr.csv'), index_col=0)

        # 1c. Correlation analysis over the right hemisphere
        if not os.path.isfile(os.path.join(OUTPUT_DIR_CORR, site, 'rightbrain_corr.csv')):
            dftrain_r_corr = PIPELINE_PART1_CORR_ANALYSIS(dftrain_r_site, y_train_site, corr_thresh,
                                                    file_name=f'{site}/rightbrain_corr.csv', write_on_disk=True)
        else:
            dftrain_r_corr = pd.read_csv(os.path.join(OUTPUT_DIR_CORR, site, 'rightbrain_corr.csv'), index_col=0)

        ## Feature selection for (1a, 1b, 1c, full data)
        Xtrain_site = df_train_site.values
        Xtrain_corr_site = dftrain_corr.drop('labels', axis=1).values
        Xtrain_corr_l_site = dftrain_l_corr.drop('labels', axis=1).values
        Xtrain_corr_r_site = dftrain_r_corr.drop('labels', axis=1).values
        ytrain_site = y_train_site.values

        # Normalization
        sc = MinMaxScaler().fit(Xtrain_site)
        dump(sc, os.path.join(OUTPUT_DIR_FS,site,'sc_train_regminmax.joblib'))
        Xtrain_norm_site = sc.transform(Xtrain_site)

        sc = MinMaxScaler().fit(Xtrain_corr_site)
        dump(sc, os.path.join(OUTPUT_DIR_FS,site,'sc_train_corr_regminmax.joblib'))
        Xtrain_corr_norm_site = sc.transform(Xtrain_corr_site)

        sc = MinMaxScaler().fit(Xtrain_corr_l_site)
        dump(sc, os.path.join(OUTPUT_DIR_FS,site,'sc_train_corr_l_regminmax.joblib'))
        Xtrain_corr_l_norm_site = sc.transform(Xtrain_corr_l_site)

        sc = MinMaxScaler().fit(Xtrain_corr_r_site)
        dump(sc, os.path.join(OUTPUT_DIR_FS,site,'sc_train_corr_r_regminmax.joblib'))
        Xtrain_corr_r_norm_site = sc.transform(Xtrain_corr_r_site)

        normtype='minmaxreg'

        PIPELINE_PART2_FS_rf(Xtrain_norm_site, Xtrain_corr_norm_site, Xtrain_corr_l_norm_site, Xtrain_corr_r_norm_site,
                             ytrain_site, site, normtype)
        PIPELINE_PART2_FS_svm(Xtrain_norm_site, Xtrain_corr_norm_site, Xtrain_corr_l_norm_site, Xtrain_corr_r_norm_site,
                             ytrain_site, site, normtype)
        PIPELINE_PART2_FS_lg1(Xtrain_norm_site, Xtrain_corr_norm_site, Xtrain_corr_l_norm_site, Xtrain_corr_r_norm_site,
                             ytrain_site, site, normtype)
        PIPELINE_PART2_FS_lg2(Xtrain_norm_site, Xtrain_corr_norm_site, Xtrain_corr_l_norm_site, Xtrain_corr_r_norm_site,
                             ytrain_site, site, normtype)

        minMaxNormRegList = [os.path.join(OUTPUT_DIR_FS, site, 'minmaxreg',x)
                         for x in os.listdir(os.path.join(OUTPUT_DIR_FS, site, 'minmaxreg'))
                           if os.path.isfile(os.path.join(OUTPUT_DIR_FS, site, 'minmaxreg',x))]
        PIPELINE_PART3_ML(minMaxNormRegList, site)

    ## 3. Perform ML
    # Load data
    # IntraMorphFeatNormalization method
    rfes = ['lg1','lg2','svm','rf']
    # data = ['Xtrain','ytrain','rfetrain'] I will access each of them with every type and rfes so no need
    #                                       to make a list for them
    types = ['','corr','corr_r','corr_l']

    # morphBasedNormList = [os.path.join(OUTPUT_DIR_FS,x) for x in os.listdir(OUTPUT_DIR_FS)
    #                       if os.path.isfile(os.path.join(OUTPUT_DIR_FS,x))]
    # minMaxAxis1NormList = [os.path.join(OUTPUT_DIR_FS,'Normalize_allMorphFeats',x)
    #                        for x in os.listdir(os.path.join(OUTPUT_DIR_FS,'Normalize_allMorphFeats'))
    #                        if os.path.isfile(os.path.join(OUTPUT_DIR_FS,'Normalize_allMorphFeats',x))]
    minMaxNormRegList = [os.path.join(OUTPUT_DIR_FS, site, 'minmaxreg',x)
                         for x in os.listdir(os.path.join(OUTPUT_DIR_FS, site, 'minmaxreg'))
                           if os.path.isfile(os.path.join(OUTPUT_DIR_FS, site, 'minmaxreg',x))]
    #
    # PIPELINE_PART3_ML(morphBasedNormList)
    # PIPELINE_PART3_ML(minMaxAxis1NormList)
    PIPELINE_PART3_ML(minMaxNormRegList, site)

    # data_dict = loadRFEFilesFrom(morphBasedNormList)
    # for rfe_clc in data_dict:
    #     print(rfe_clc)
    #     rfe_clc_data = data_dict[rfe_clc]
    #     rfe_clc = rfe_clc.split('.')[0]
    #
    #     clf = train_models(rfe_clc_data['Xtrain'],rfe_clc_data['ytrain'], 5)
    #     dump(clf, os.path.join(OUTPUT_DIR_ML, f"clf_{rfe_clc}_train.joblib"))
    #
    #     clf = train_models(rfe_clc_data['Xtrain_corr'],rfe_clc_data['ytrain_corr'], 5)
    #     dump(clf, os.path.join(OUTPUT_DIR_ML, f"clf_{rfe_clc}_train_corr.joblib"))
    #
    #     clf = train_models(rfe_clc_data['Xtrain_corr_l'],rfe_clc_data['ytrain_corr_l'], 5)
    #     dump(clf, os.path.join(OUTPUT_DIR_ML, f"clf_{rfe_clc}_train_corr_l.joblib"))
    #
    #     clf = train_models(rfe_clc_data['Xtrain_corr_r'],rfe_clc_data['ytrain_corr_r'], 5)
    #     dump(clf, os.path.join(OUTPUT_DIR_ML, f"clf_{rfe_clc}_train_corr_r.joblib"))

        # clf_train_rf = train_models(Xtrain_norm, df_train['labels'].values, 5)
        # dump(clf_left_rf, os.path.join(OUTPUT_DIR_ML, f"clf_left_OnlyCorr_.joblib"))


if __name__ == '__main__':
    main()
