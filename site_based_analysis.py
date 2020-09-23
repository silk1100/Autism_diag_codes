"""
Split subjects into different sites
For each site
    1. Use the feature representations to create data matrix
    2. Perform ML on each site using the feature representation and save the performance

Use the features from Compare_FSALONE_VS_CORRALONE.ipynb
"""
import pandas as pd
import numpy as np


def LOAD_DATA():
    left_data_dir = "D:\\PhD\\Data\\aparc\\df_left_newRepresentation.csv"
    right_data_dir = "D:\\PhD\\Data\\aparc\\df_right_newRepresentation.csv"

    df_left = pd.read_csv(left_data_dir, index_col=0)
    df_right = pd.read_csv(right_data_dir, index_col=0)

    return df_left, df_right



def main():
    pass

if __name__ == '__main__':
    main()