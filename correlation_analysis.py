"""
** TODO
* Within each hemisphere
* Find the correlation between features at different thresholds 0.65, 0.85
* Select a feature from the correlated ones based on a supervised criteria (maximize variance
between groups and minimize variance within group)
* Create a report telling what features correlate with other features ??
"""
import pandas as pd
import numpy as np
import scipy
import seaborn as sns


def get_corr_dict(df_hemi, corr_thresh=0.65):
    """
    Returns a dictionary with keys as the features of df_hemi, and the value is a list of the
    correlated features with the key with a correlation value greater than or equal corr_thresh.
    :param df_hemi: pandas.DataFrame
    :param corr_thresh: float [0,1]
    :return: dict
    """
    corr_dict = dict()
    corr_mat_val = df_hemi.corr().abs()
    for ind, col in enumerate(df_hemi.columns):
        r= np.where(corr_mat_val[col]>=corr_thresh)[0]
        r = r[r!=ind]
        rnames = df_hemi.columns[r]
        corr_dict[col] = rnames

    return corr_dict


def default_criteria(df_hemi_correlated):
    """
    The default function which is used to select a feature among a correlated set of feature. This
    criteria is a supervised criteria at which it finds the feature that maximizes the between
    groups variance and minimize the within group variance among the correlated set. It assumes that
    the classes are defined in a column called "labels"
    :param df_hemi: pandas.DataFrame
    :return: str feature name
    """
    df_hemi_td = df_hemi_correlated[df_hemi_correlated['labels']==0]
    df_hemi_asd = df_hemi_correlated[df_hemi_correlated['labels']==1]
    scores = dict()
    for col in df_hemi_correlated.columns:
        if col == 'labels':
            continue
        sbb = np.square(np.mean(df_hemi_td[col])-np.mean(df_hemi_correlated[col])) + \
              np.square(np.mean(df_hemi_asd[col])-np.mean(df_hemi_correlated[col]))
        var_col = np.var(df_hemi_asd[col], ddof=1) + np.var(df_hemi_td[col], ddof=1)
        scores[col] = sbb/var_col
    s = pd.Series(scores)
    best_feat = s.sort_values(ascending=False).index[0]
    return best_feat


def remove_list_from_list(bigger_list, smaller_list_):
    """
    Remove all the elements in smaller_list from the bigger_list
    :param bigger_list: list
    :param smaller_list: list
    :return bigger_list: list
    """
    smaller_list = []
    for s in smaller_list_:
        if s in bigger_list:
            smaller_list.append(s)

    if len(bigger_list) < len(smaller_list):
        raise ValueError("The first argument should be the bigger list")

    for element in smaller_list:
        if element in bigger_list:
            bigger_list.remove(element)

    return bigger_list


def correlation_analysis(df_hemi, corr_thresh=0.65, criteria=None):
    """
    Perform correlation analysis of pandas.dataframe df_hemi. df_hemi is expected to be
    hemisphere of the brain containing all morphological features. After finding the correla-
    tion, one feature is selected from the correlation set based on the given criteria. The
    function returns a dataframe with features that have correlation values among each others
    less than the corr_thresh
    :param df_hemi: pandas.DataFrame
    :param corr_thresh: float [0,1]
    :param criteria: python.function
    :return df_hemi_nocollinearity: pandas.DataFrame
    """
    agePop=False
    sexPop=False
    if 'age' in df_hemi.columns:
        age = df_hemi.pop('age')
        agePop=True
    if 'sex' in df_hemi.columns:
        sex = df_hemi.pop('sex')
        sexPop=True
    if 'labels' in df_hemi.columns:
        corr_dict = get_corr_dict(df_hemi.drop('labels', axis=1), corr_thresh)
    else:
        corr_dict = get_corr_dict(df_hemi, corr_thresh)

    selected_feats = df_hemi.columns.to_list()
    if criteria is None:
        for key in corr_dict:
            primary_feat, list_of_corr_feats = key, corr_dict[key].tolist()
            list_of_corr_feats.append(primary_feat)
            selected_feat = default_criteria(df_hemi[list_of_corr_feats+['labels']])
            list_of_corr_feats.remove(selected_feat)
            selected_feats = remove_list_from_list(selected_feats, list_of_corr_feats)
    else:
        # TODO: implementation of various criteria
        pass
    if agePop:
        df_hemi['age'] = age
    if sexPop:
        df_hemi['sex'] = sex
    df_hemi_nocollinearity = df_hemi[selected_feats]
    return df_hemi_nocollinearity

