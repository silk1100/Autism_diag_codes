"""
Split subjects based on sites
For each site:
    1. Create data matrix from the selected features found in constant.py
    2. Perform 5 fold CV using these features and see how well they perform on each site separately

bonus: Perform a feature selection step on each site alone and see how mutual features we will get from
that site and the global features
"""

import numpy as np
import pandas as pd
from constants import SELECTED_RIGHT_SVM, SELECTED_RIGHT_RF, SELECTD_50_CORRANA_NO_L_R,\
    SELECTED_LEFT_RF, SELECTED_LEFT_SVM, DATADIR_aparc_MODLEFT, DATADIR_aparc_MODRight
from ML_model_selection import train_models
from RFE_FS import select_features


def LOAD_DATA():
    df_left, df_right = pd.read_csv(DATADIR_aparc_MODLEFT, index_col=0),\
                        pd.read_csv(DATADIR_aparc_MODRight, index_col=0)
    return df_left, df_right


def main():
    # Load data
    df_left, df_right = LOAD_DATA()

    # Load features

if __name__ == '__main__':
    main()
