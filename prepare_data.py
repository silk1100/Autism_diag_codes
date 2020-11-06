import pandas as pd


def update_cols_suffix(old_feats, suffix1, suffix2):
    updated_feats = list()
    for feat in old_feats:
        featL = feat.split('_')
        morph = featL[-1]
        if 'med' in morph:
            morph = suffix1
        else:
            morph = suffix2
        featL[-1] = morph
        updated_feats.append('_'.join(featL))
    # Fix for age, sex, labels
    updated_feats[-1] = old_feats[-1]
    updated_feats[-2] = old_feats[-2]
    updated_feats[-3] = old_feats[-3]

    return updated_feats


def suffix1_fc(c1, c2):
    return c1-c2


def suffix2_fc(c1, c2):
    return c1+c2


def modifyMedRange2MedPlusMinusRange(df):
    """
    Convert the data representation from 1 column median, and 1 column IQR into
    1 column median-IQR, and 1 column median+IQR for each morphological feature for each brain region
    :param df: pandas.DataFrame
    :return updated_df: pandas.DataFrame
    """
    old_feats = df.columns.to_list()
    suffix1 = "medMIQR"
    suffix2 = "medPIQR"
    updated_feats = update_cols_suffix(old_feats, suffix1, suffix2)
    updated_df = pd.DataFrame(None, index=df.index, columns=updated_feats)
    for feat in updated_df.columns:
        if 'age' in feat or 'sex' in feat or 'labels' in feat:
            updated_df[feat] = df[feat]
            continue
        morph, brainreg, _ = feat.split('_')
        required_cols_from_old_df = ['_'.join([morph,brainreg,'med']),
                                     '_'.join([morph,brainreg,'20-80range'])]
        med_iqr_df = df[required_cols_from_old_df]
        if suffix1 in feat:
            updated_df[feat] = suffix1_fc(med_iqr_df.iloc[:, 0], med_iqr_df.iloc[:, 1])
        elif suffix2 in feat:
            updated_df[feat] = suffix2_fc(med_iqr_df.iloc[:, 0], med_iqr_df.iloc[:, 1])
        else:
            raise ValueError("Updated columns have an undefined problem!")

    return updated_df


def get_left_right_hemisphere(features):
    """
    Based on the nomenclature of the csv files (aparc, a2009s), this function returns the
    left and right hemispheres respectively
    :param features: list- feature names
    :return: tuple- (list_leftHemisphere_names, list_rightHemisphere_names)
    """
    left_cols, right_cols = list(), list()
    for feat in features:
        name_list = feat.split('_')
        if len(name_list) != 3:
            left_cols.append(feat)
            right_cols.append(feat)
        elif name_list[1][0] == 'l':
            left_cols.append(feat)
        elif name_list[1][0] == 'r':
            right_cols.append(feat)
        else:
            raise ValueError("Feature name should be in the form of *morph_l/rbrainreg_agg*"
                             "e.g curv_lbankssts_med which stands for the median value of the"
                             " curvature of the left bankssts")

    return left_cols, right_cols


def select_sites(df):
    sites = list(set([x.split('_')[0] for x in df.index]))
    rejected_list = []
    accepted_list = []
    acceptes_subjects = []
    acc_subj = 0
    rej_subj = 0
    for site in sites:
        site_subjects = [subj for subj in df.index if site in subj]
        subdf = df.loc[site_subjects,'labels']
        asd_cnt = len(subdf[subdf==1])
        td_cnt = len(subdf[subdf == 0])
        if asd_cnt < td_cnt:
            ratio = td_cnt/(asd_cnt+td_cnt)
        else:
            ratio = asd_cnt / (asd_cnt + td_cnt)
        if ratio > 0.6:
            rejected_list.append(site)
            rej_subj += td_cnt + asd_cnt
        else:
            accepted_list.append(site)
            acceptes_subjects += site_subjects
            acc_subj += td_cnt + asd_cnt

    stdf = df.loc[acceptes_subjects, :]
    return stdf, (accepted_list, rejected_list, acc_subj, rej_subj)


def get_csvfile_ready(fldr, testratio=0.2, random_seed=11111111, updated_data=True):
    """
    This function is supposed to read the csv file, hold out testratio of the dataset for
    model's testing, split the hemisphere into 2 halfs
    :param fldr: string - csv directory
    :return: tuple - (left_hemisphere_dataframe, right_hemisphere_dataframe, test_dataframe)
    """
    df = pd.read_csv(fldr, index_col=0)
    if updated_data:
        df = modifyMedRange2MedPlusMinusRange(df)

    df_asd, df_td = df[df['labels']==1], df[df['labels']==0]
    assert(df_asd.columns.to_list() == df_td.columns.to_list())
    df = select_sites(df)
    left_cols, right_cols = get_left_right_hemisphere(df.columns)

    if df_asd.shape[0] > df_td.shape[0]:
        test_size = int(df_td.shape[0]*testratio)+1
    else:
        test_size = int(df_asd.shape[0]*testratio)+1

    df_test_asd = df_asd.sample(test_size, random_state=random_seed)
    df_test_td = df_td.sample(test_size, random_state=random_seed)
    df_train_asd = df_asd.drop(df_test_asd.index)
    df_train_td = df_td.drop(df_test_td.index)
    df_test = pd.concat([df_test_td, df_test_asd], axis=0)
    df_train = pd.concat([df_train_td, df_train_asd], axis=0)
    df_train_left = df_train[left_cols]
    df_train_right = df_train[right_cols]
    df_test_left = df_test[left_cols]
    df_test_right = df_test[right_cols]

    return df_train_left, df_train_right, df_test_left, df_test_right


def main():
    df = pd.read_csv("D:\\PhD\\Data\\aparc\\DrEid_brain_sMRI_lr_TDASD.csv", index_col=0)
    # updated_df = modifyMedRange2MedPlusMinusRange(df)
    updated_df, info = select_sites(df)
    print(updated_df)
    # updated_df.to_csv('new_aparc.csv')


if __name__ == '__main__':
    main()