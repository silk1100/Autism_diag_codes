"""
** Rationale behind this script
* Would PCA produce components that might represent like 95% of the variation using less features than those
  I get out of my whole system?
* The answer is NO (when I normalize the features).
* Would I use PCA and LDA to compare between the performance of feature extraction and my method?
"""


import pandas as pd
from sklearn.decomposition import PCA
from constants import OUTPUT_DIR_SPLIT, OUTPUT_DIR_CORR, OUTPUT_DIR_FS
import os
from sklearn.preprocessing import StandardScaler


def perform_pca(df):
    pca = PCA()
    transformeddf = pca.fit_transform(df)
    x=0


def main():
        df_right_train = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT,"right_train.csv"), index_col=0)
        df_left_train = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT,"left_train.csv"), index_col=0)
        age = df_left_train.pop('age')
        sex = df_left_train.pop('sex')
        labels = df_left_train.pop('labels')
        df_left_train_norm = StandardScaler().fit_transform(df_left_train)
        perform_pca(df_left_train_norm)


if __name__ == '__main__':
    main()