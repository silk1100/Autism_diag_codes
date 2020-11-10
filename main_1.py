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


def PIPELINE_PART2_FS():
    pass


def PIPELINE_PART3_ML():
    pass


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
    # data_dir = "D:\\PhD\\Data\\aparc\\DrEid_brain_sMRI_lr_TDASD.csv"
    # dfs, series = PIPELINE_PART1_DATA_PREPROCESSING(data_dir)
    # df, df_l, df_r = dfs
    # age, sex, labels = series

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

    ### Pipeline Begins
    ## 1. Perform correlation analysis
    # corr_thresh = 50

    # 1a. Correlation analysis over the whole brain
    # dftrain_corr = PIPELINE_PART1_CORR_ANALYSIS(dftrain, labelstrain, corr_thresh,
    #                                             file_name='fullbrain_corr.csv', write_on_disk=True)
    # # 1b. Correlation analysis over the left hemisphere
    # dftrain_l_corr = PIPELINE_PART1_CORR_ANALYSIS(dftrain_l, labelstrain, corr_thresh,
    #                                             file_name='leftbrain_corr.csv', write_on_disk=True)
    # # 1c. Correlation analysis over the right hemisphere
    # dftrain_r_corr = PIPELINE_PART1_CORR_ANALYSIS(dftrain_r, labelstrain, corr_thresh,
    #                                             file_name='rightbrain_corr.csv', write_on_disk=True)

    ## Feature selection for (1a, 1b, 1c, full data)
    df_train = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT,'train_fullbrain.csv'), index_col=0)
    df_train_corr = pd.read_csv(os.path.join(OUTPUT_DIR_CORR, 'fullbrain_corr.csv'), index_col=0)
    df_train_corr_l = pd.read_csv(os.path.join(OUTPUT_DIR_CORR, 'leftbrain_corr.csv'), index_col=0)
    df_train_corr_r = pd.read_csv(os.path.join(OUTPUT_DIR_CORR, 'rightbrain_corr.csv'), index_col=0)

    # print(df_train.columns)
    # print(df_train_corr.columns)
    # print(df_train_corr_l.columns)
    # print(df_train_corr_r.columns)
    # X = df_train.drop('labels', axis=1).values
    # y = df_train['labels'].values
    # Xcorr = df_train_corr.drop('labels', axis=1).values
    # ycorr = df_train_corr['labels'].values
    # Xcorr_l = df_train_corr_l.drop('labels', axis=1).values
    # ycorr_l = df_train_corr_l['labels'].values
    # Xcorr_r = df_train_corr_r.drop('labels', axis=1).values
    # ycorr_r = df_train_corr_r['labels'].values

    # Normalization
    # Trial 1
    # Xtrain_norm, scdict_train = mynormalize(df_train.drop('labels', axis=1))
    # Xtrain_corr_norm, scdict_traincorr = mynormalize(df_train_corr.drop('labels', axis=1))
    # Xtrain_corr_l_norm, scdict_traincorr_l = mynormalize(df_train_corr_l.drop('labels', axis=1))
    # Xtrain_corr_r_norm, scdict_traincorr_r = mynormalize(df_train_corr_r.drop('labels', axis=1))
    # Trial 2
    # Xtrain_norm, scdict_train = mynormalize(df_train.drop('labels', axis=1), allfeats=True)
    # Xtrain_corr_norm, scdict_traincorr = mynormalize(df_train_corr.drop('labels', axis=1), allfeats=True)
    # Xtrain_corr_l_norm, scdict_traincorr_l = mynormalize(df_train_corr_l.drop('labels', axis=1), allfeats=True)
    # Xtrain_corr_r_norm, scdict_traincorr_r = mynormalize(df_train_corr_r.drop('labels', axis=1), allfeats=True)
    # Trial 2
    Xtrain_norm = MinMaxScaler().fit_transform(df_train.drop('labels', axis=1))
    Xtrain_corr_norm = MinMaxScaler().fit_transform(df_train_corr.drop('labels', axis=1))
    Xtrain_corr_l_norm = MinMaxScaler().fit_transform(df_train_corr_l.drop('labels', axis=1))
    Xtrain_corr_r_norm = MinMaxScaler().fit_transform(df_train_corr_r.drop('labels', axis=1))

    ## 2. Perform Feature selection
    # Feature selection classifiers
    MAX_ITR=1000000
    # rf = RandomForestClassifier(n_estimators=250, max_depth=5000)
    # Xtrain_rf, ytrain, rfetrain_rf = select_features(rf,Xtrain_norm,
    #                                                  df_train['labels'].values,
    #                                                  scoring_metric='balanced_accuracy',
    #                                                  save_file=False)
    # dump(rfetrain_rf,os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/rfetrain_rf.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/Xtrain_rf.npy'), Xtrain_rf)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/ytrain.npy'), ytrain)
    #
    # rf = RandomForestClassifier(n_estimators=250, max_depth=5000)
    # Xtrain_corr_rf, ytrain_corr, rfetrain_corr_rf = select_features(rf,Xtrain_corr_norm,
    #                                                  df_train_corr['labels'].values,
    #                                                  scoring_metric='balanced_accuracy',
    #                                                  save_file=False)
    # dump(rfetrain_corr_rf,os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/rfetrain_corr_rf.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/Xtrain_corr_rf.npy'), Xtrain_corr_rf)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/ytrain_corr.npy'), ytrain_corr)
    #
    # rf = RandomForestClassifier(n_estimators=250, max_depth=5000)
    # Xtrain_corr_l_rf, ytrain_corr_l, rfetrain_corr_l_rf = select_features(rf,Xtrain_corr_l_norm,
    #                                                  df_train_corr_l['labels'].values,
    #                                                  scoring_metric='balanced_accuracy',
    #                                                  save_file=False)
    # dump(rfetrain_corr_l_rf,os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/rfetrain_corr_l_rf.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/Xtrain_corr_l_rf.npy'), Xtrain_corr_l_rf)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/ytrain_corr_l.npy'), ytrain_corr_l)
    #
    #
    # rf = RandomForestClassifier(n_estimators=250, max_depth=5000)
    # Xtrain_corr_r_rf, ytrain_corr_r, rfetrain_corr_r_rf = select_features(rf,Xtrain_corr_r_norm,
    #                                                  df_train_corr_r['labels'].values,
    #                                                  scoring_metric='balanced_accuracy',
    #                                                  save_file=False)
    # dump(rfetrain_corr_r_rf,os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/rfetrain_corr_r_rf.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/Xtrain_corr_r_rf.npy'), Xtrain_corr_r_rf)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/ytrain_corr_r.npy'), ytrain_corr_r)
    #
    #
    # svm = LinearSVC(max_iter=MAX_ITR)
    # Xtrain_svm, ytrain, rfetrain_svm = select_features(svm, Xtrain_norm,
    #                                                  df_train['labels'].values,
    #                                                  scoring_metric='balanced_accuracy',
    #                                                  save_file=False)
    # dump(rfetrain_svm, os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/rfetrain_svm.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/Xtrain_svm.npy'), Xtrain_svm)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/ytrain.npy'), ytrain)
    #
    # svm = LinearSVC(max_iter=MAX_ITR)
    # Xtrain_corr_svm, ytrain_corr, rfetrain_corr_svm = select_features(svm, Xtrain_corr_norm,
    #                                                                 df_train_corr['labels'].values,
    #                                                                 scoring_metric='balanced_accuracy',
    #                                                                 save_file=False)
    # dump(rfetrain_corr_svm, os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/rfetrain_corr_svm.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/Xtrain_corr_svm.npy'), Xtrain_corr_svm)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/ytrain_corr.npy'), ytrain_corr)
    #
    # svm = LinearSVC(max_iter=MAX_ITR)
    # Xtrain_corr_l_svm, ytrain_corr_l, rfetrain_corr_l_svm = select_features(svm, Xtrain_corr_l_norm,
    #                                                                       df_train_corr_l['labels'].values,
    #                                                                       scoring_metric='balanced_accuracy',
    #                                                                       save_file=False)
    # dump(rfetrain_corr_l_svm, os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/rfetrain_corr_l_svm.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/Xtrain_corr_l_svm.npy'), Xtrain_corr_l_svm)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/ytrain_corr_l.npy'), ytrain_corr_l)

    # svm = LinearSVC(max_iter=MAX_ITR)
    # Xtrain_corr_r_svm, ytrain_corr_r, rfetrain_corr_r_svm = select_features(svm, Xtrain_corr_r_norm,
    #                                                                       df_train_corr_r['labels'].values,
    #                                                                       scoring_metric='balanced_accuracy',
    #                                                                       save_file=False)
    # dump(rfetrain_corr_r_svm, os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/rfetrain_corr_r_svm.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/Xtrain_corr_r_svm.npy'), Xtrain_corr_r_svm)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Normalize_allMorphFeats/ytrain_corr_r.npy'), ytrain_corr_r)

    # Lg All data
    # lg2 = LogisticRegression(penalty='l2', max_iter=MAX_ITR, solver='saga')
    # Xtrain_lg2, ytrain, rfetrain_lg2 = select_features(lg2, Xtrain_norm, df_train['labels'].values,
    #                                                                       scoring_metric='balanced_accuracy',
    #                                                                       save_file=False)
    # dump(rfetrain_lg2, os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/rfetrain_lg2.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/Xtrain_lg2.npy'), Xtrain_lg2)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/ytrain.npy'), ytrain)
    #
    # lg1 = LogisticRegression(penalty='l1', max_iter=MAX_ITR, solver='saga')
    # Xtrain_lg1, ytrain, rfetrain_lg1 = select_features(lg1, Xtrain_norm, df_train['labels'].values,
    #                                                                       scoring_metric='balanced_accuracy',
    #                                                                       save_file=False)
    # dump(rfetrain_lg1, os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/rfetrain_lg1.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/Xtrain_lg1.npy'), Xtrain_lg1)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/ytrain.npy'), ytrain)
    #
    # # LG uncorrelated data
    # lg2 = LogisticRegression(penalty='l2', max_iter=MAX_ITR,solver='saga')
    # Xtrain_corr_lg2, ytrain_corr, rfetrain_corr_lg2 = select_features(lg2, Xtrain_corr_norm,
    #                                                                         df_train_corr['labels'].values,
    #                                                                         scoring_metric='balanced_accuracy',
    #                                                                         save_file=False)
    # dump(rfetrain_corr_lg2, os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/rfetrain_corr_lg2.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/Xtrain_corr_lg2.npy'), Xtrain_corr_lg2)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/ytrain_corr.npy'), ytrain_corr)
    #
    # lg1 = LogisticRegression(penalty='l1', max_iter=MAX_ITR,solver='saga')
    # Xtrain_corr_lg1, ytrain_corr, rfetrain_corr_lg1 = select_features(lg1, Xtrain_corr_norm,
    #                                                                         df_train_corr['labels'].values,
    #                                                                         scoring_metric='balanced_accuracy',
    #                                                                         save_file=False)
    # dump(rfetrain_corr_lg1, os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/rfetrain_corr_lg1.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/Xtrain_corr_lg1.npy'), Xtrain_corr_lg1)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/ytrain_corr.npy'), ytrain_corr)
    #
    # # LG left hemisphere data
    # lg2 = LogisticRegression(penalty='l2', max_iter=MAX_ITR,solver='saga')
    # Xtrain_corr_l_lg2, ytrain_corr_l, rfetrain_corr_l_lg2 = select_features(lg2, Xtrain_corr_l_norm,
    #                                                                         df_train_corr_l['labels'].values,
    #                                                                         scoring_metric='balanced_accuracy',
    #                                                                         save_file=False)
    # dump(rfetrain_corr_l_lg2, os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/rfetrain_corr_l_lg2.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/Xtrain_corr_l_lg2.npy'), Xtrain_corr_l_lg2)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/ytrain_corr_l.npy'), ytrain_corr_l)
    #
    # lg1 = LogisticRegression(penalty='l1', max_iter=MAX_ITR,solver='saga')
    # Xtrain_corr_l_lg1, ytrain_corr_l, rfetrain_corr_l_lg1 = select_features(lg1, Xtrain_corr_l_norm,
    #                                                                         df_train_corr_l['labels'].values,
    #                                                                         scoring_metric='balanced_accuracy',
    #                                                                         save_file=False)
    # dump(rfetrain_corr_l_lg1, os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/rfetrain_corr_l_lg1.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/Xtrain_corr_l_lg1.npy'), Xtrain_corr_l_lg1)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/ytrain_corr_l.npy'), ytrain_corr_l)
    #
    #
    # # LG Right hemisphere data
    # lg2 = LogisticRegression(penalty='l2', max_iter=MAX_ITR,solver='saga')
    # Xtrain_corr_r_lg2, ytrain_corr_r, rfetrain_corr_r_lg2 = select_features(lg2, Xtrain_corr_r_norm,
    #                                                                         df_train_corr_r['labels'].values,
    #                                                                         scoring_metric='balanced_accuracy',
    #                                                                         save_file=False)
    # dump(rfetrain_corr_r_lg2, os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/rfetrain_corr_r_lg2.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/Xtrain_corr_r_lg2.npy'), Xtrain_corr_r_lg2)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/ytrain_corr_r.npy'), ytrain_corr_r)
    #
    # lg1 = LogisticRegression(penalty='l1', max_iter=MAX_ITR, solver='saga')
    # Xtrain_corr_r_lg1, ytrain_corr_r, rfetrain_corr_r_lg1 = select_features(lg1, Xtrain_corr_r_norm,
    #                                                                         df_train_corr_l['labels'].values,
    #                                                                         scoring_metric='balanced_accuracy',
    #                                                                         save_file=False)
    # dump(rfetrain_corr_r_lg1, os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/rfetrain_corr_r_lg1.joblib'))
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/Xtrain_corr_r_lg1.npy'), Xtrain_corr_r_lg1)
    # np.save(os.path.join(OUTPUT_DIR_FS, 'Regular_minMaxNorm/ytrain_corr_r.npy'), ytrain_corr_r)

    ## 3. Perform ML
    # Load data
    # IntraMorphFeatNormalization method
    # Xtrain np.load('')
    #
    # clf_train_rf = train_models(Xtrain_norm, df_train['labels'].values, 5)
    # dump(clf_left_rf, os.path.join(OUTPUT_DIR_ML, f"clf_left_OnlyCorr_.joblib"))

    # # Get the data ready
    # updated_data = True
    # df_leftHemi_train, df_rightHemi_train, df_test_left, df_test_right = \
    #     get_csvfile_ready(constants.DATADIR_aparc, testratio=0.2, random_seed=131417191,
    #                       updated_data=updated_data)
    #
    # # Check whether to save as new representation or old representation
    # if updated_data:
    #     left_train_file = 'left_train_modifiedMedPIQR.csv'
    #     right_train_file = 'right_train_modifiedMedPIQR.csv'
    #     left_test_file = 'left_test_modifiedMedPIQR.csv'
    #     right_test_file = 'right_test_modifiedMedPIQR.csv'
    #     nocoll_left_file = 'left_train_noColliniarity_modifiedMedPIQR'
    #     nocoll_right_file = 'right_train_noColliniarity_modifiedMedPIQR'
    #     obj_svc_right_file = "right_svc_modifiedMedPIQR.joblib"
    #     obj_svc_left_file = "left_svc_modifiedMedPIQR.joblib"
    #     obj_rf_right_file = "right_rf_modifiedMedPIQR.joblib"
    #     obj_rf_left_file = "left_rf_modifiedMedPIQR.joblib"
    #     clf_svm_left_file = "clf_left_svm_modifiedMedPIQR.joblib"
    #     clf_svm_right_file = "clf_right_svm_modifiedMedPIQR.joblib"
    #     clf_rf_left_file = "clf_left_rf_modifiedMedPIQR.joblib"
    #     clf_rf_right_file = "clf_right_rf_modifiedMedPIQR.joblib"
    # else:
    #     left_train_file = 'left_train.csv'
    #     right_train_file = 'right_train.csv'
    #     left_test_file = 'left_test.csv'
    #     right_test_file = 'right_test.csv'
    #     nocoll_left_file = 'left_train_noColliniarity.csv'
    #     nocoll_right_file = 'right_train_noColliniarity.csv'
    #     obj_svc_right_file = "right_svc.joblib"
    #     obj_svc_left_file = "left_svc.joblib"
    #     obj_rf_right_file = "right_rf.joblib"
    #     obj_rf_left_file = "left_rf.joblib"
    #     clf_svm_left_file = "clf_left_svm.joblib"
    #     clf_svm_right_file = "clf_right_svm.joblib"
    #     clf_rf_left_file = "clf_left_rf.joblib"
    #     clf_rf_right_file = "clf_right_rf.joblib"
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
    #
    # df_leftHemi_train = pd.read_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,left_train_file)}', index_col=0)
    # df_rightHemi_train = pd.read_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,right_train_file)}', index_col=0)
    # # df_test_left = pd.read_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,left_test_file)}', index_col=0)
    # # df_test_right = pd.read_csv(f'{os.path.join(OUTPUT_DIR_SPLIT,right_test_file)}', index_col=0)
    # # x=0
    # # Correlation analysis
    # corr_thresh = 50
    # # for corr_thresh in [60, 70, 85]:
    # df_leftHemi_train_corr = correlation_analysis(df_leftHemi_train, corr_thresh=corr_thresh/100.0)
    # df_rightHemi_train_corr = correlation_analysis(df_rightHemi_train, corr_thresh=corr_thresh/100.0)
    # #
    # df_leftHemi_train_corr.to_csv(os.path.join(OUTPUT_DIR_CORR, f'{nocoll_left_file}_{corr_thresh}.csv'))
    # df_rightHemi_train_corr.to_csv(os.path.join(OUTPUT_DIR_CORR, f'{nocoll_right_file}_{corr_thresh}.csv'))
    # X_clean_right = df_rightHemi_train_corr.drop('labels', axis=1)
    # X_clean_left = df_leftHemi_train_corr.drop('labels', axis=1)
    # y_right = df_rightHemi_train_corr['labels']
    # y_left = df_leftHemi_train_corr['labels']
    # X_clean_right_norm = StandardScaler().fit_transform(X_clean_right)
    # X_clean_left_norm = StandardScaler().fit_transform(X_clean_left)
    # #
    # # # Feature selection (I added a hierarchy level without any testing or debugging--
    # # # GO CHECK RFE_FS::select_feats BEFORE RUN)
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
    #
    # # X_clean_left_lsvc.to_csv(os.path.join(OUTPUT_DIR_FS, "clean_left_svc_50.csv"))
    # # X_clean_right_lsvc.to_csv(os.path.join(OUTPUT_DIR_FS, "clean_right_svc_50.csv"))
    # # X_clean_left_rf.to_csv(os.path.join(OUTPUT_DIR_FS, "clean_left_rf_50.csv"))
    # # X_clean_right_rf.to_csv(os.path.join(OUTPUT_DIR_FS, "clean_right_rf_50.csv"))
    # dump(obj_left_lsvm, os.path.join(OUTPUT_DIR_FS,obj_svc_left_file))
    # dump(obj_right_lsvm, os.path.join(OUTPUT_DIR_FS,obj_svc_right_file))
    # dump(obj_left_rf, os.path.join(OUTPUT_DIR_FS,obj_rf_left_file))
    # dump(obj_right_rf, os.path.join(OUTPUT_DIR_FS,obj_rf_right_file))


    #
    # # Load FS data
    # frames, objs = constants.LOAD_FS_RESULTS()
    # # X_clean_left_lsvc, X_clean_right_lsvc, X_clean_left_rf, X_clean_right_rf, y_left, y_right = frames
    # df_train_left = pd.read_csv(os.path.join(OUTPUT_DIR_CORR, 'left_train_noColliniarity_modifiedMedPIQR_50.csv'), index_col=0)
    # df_train_right = pd.read_csv(os.path.join(OUTPUT_DIR_CORR, 'right_train_noColliniarity_modifiedMedPIQR_50.csv'), index_col=0)
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
    # # Normalization if required
    # #     left_train_file = 'left_train_modifiedMedPIQR.csv'
    # #     right_train_file = 'right_train_modifiedMedPIQR.csv'
    # # X_clean_left_rf_norm = StandardScaler().fit_transform(X_clean_left_rf)
    # # X_clean_right_rf_norm = StandardScaler().fit_transform(X_clean_right_rf)
    #
    #
    #
    # # X_clean_left_rf_norm = StandardScaler().fit_transform(X_clean_left_rf)
    # # X_clean_right_rf_norm = StandardScaler().fit_transform(X_clean_right_rf)
    # # X_clean_left_lsvc_norm = StandardScaler().fit_transform(X_clean_left_lsvc)
    # # X_clean_right_lsvc_norm = StandardScaler().fit_transform(X_clean_right_lsvc)
    #
    # # clf_left_rf = train_models(X_clean_left_rf_norm, y_left, 5)
    # # clf_right_rf = train_models(X_clean_right_rf_norm, y_right, 5)
    # # clf_left_lsvc = train_models(X_clean_left_lsvc_norm, y_left, 5)
    # # clf_right_lsvc = train_models(X_clean_right_lsvc_norm, y_right, 5)
    #
    # clf_left_rf = train_models(X_clean_left_rf, y_left, 5)
    # clf_right_rf = train_models(X_clean_right_rf, y_right, 5)
    # clf_left_lsvc = train_models(X_clean_left_lsvc, y_left, 5)
    # clf_right_lsvc = train_models(X_clean_right_lsvc, y_right, 5)
    #
    # dump(clf_left_lsvc, os.path.join(OUTPUT_DIR_ML,"clf_left_lsvm_CORR_50.joblib"))
    # dump(clf_right_lsvc, os.path.join(OUTPUT_DIR_ML,"clf_right_lsvm_CORR_50.joblib"))
    # dump(clf_left_rf, os.path.join(OUTPUT_DIR_ML,"clf_left_RF_CORR_50.joblib"))
    # dump(clf_right_rf, os.path.join(OUTPUT_DIR_ML,"clf_right_RF_CORR_50.joblib"))
    #
    # # This section is only to check correlation without FS
    # df_leftHemi_train_corr.to_csv(os.path.join(OUTPUT_DIR_CORR, f'{nocoll_left_file}_{corr_thresh}.csv'))
    # df_rightHemi_train_corr.to_csv(os.path.join(OUTPUT_DIR_CORR, f'{nocoll_right_file}_{corr_thresh}.csv'))
    # y_left = df_leftHemi_train_corr.pop('labels')
    # y_right = df_rightHemi_train_corr.pop('labels')
    #
    # X_clean_left_rf_norm = StandardScaler().fit_transform(df_leftHemi_train_corr)
    # X_clean_right_rf_norm = StandardScaler().fit_transform(df_rightHemi_train_corr)
    #
    # clf_left_rf = train_models(X_clean_left_rf_norm, y_left, 5)
    # clf_right_rf = train_models(X_clean_right_rf_norm, y_right, 5)
    #
    # #
    # # # dump(clf_left_lsvc, os.path.join(OUTPUT_DIR_ML,clf_svm_left_file))
    # # # dump(clf_right_lsvc, os.path.join(OUTPUT_DIR_ML,clf_svm_right_file))
    # dump(clf_left_rf, os.path.join(OUTPUT_DIR_ML,f"clf_left_OnlyCorr_{corr_thresh}.joblib"))
    # dump(clf_right_rf, os.path.join(OUTPUT_DIR_ML,f"clf_right_OnlyCorr_{corr_thresh}.joblib"))


if __name__ == '__main__':
    main()