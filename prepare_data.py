import pandas as pd


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


def get_csvfile_ready(fldr, testratio=0.2, random_seed=11111111):
    """
    This function is supposed to read the csv file, hold out testratio of the dataset for
    model's testing, split the hemisphere into 2 halfs
    :param fldr: string - csv directory
    :return: tuple - (left_hemisphere_dataframe, right_hemisphere_dataframe, test_dataframe)
    """
    df = pd.read_csv(fldr, index_col=0)
    df_asd, df_td = df[df['labels']==1], df[df['labels']==0]
    assert(df_asd.columns.to_list() == df_td.columns.to_list())

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