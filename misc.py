import numpy as np
import pandas as pd
from scipy.stats import anderson, kstest, shapiro


def isNormalData(X, violationThreshold=0.1):
    """
    X is a data matrix such that each row represents an observation and each column represents
    a feature. Features are tested using different normality statistical tests and we use
    majority voting to determine whether a column is normal or not. if violationThresold%
    of data is not normal, function return false.

    :param X: pandas.DataFrame
    :param violationThreshold: float32 (should be between 0->0.5)
    :return isnorm: bool
    """
    isnorm = True
    violated_feats = []
    if isinstance(X, pd.DataFrame):
        n_feats = len(X.columns)
        minAllowable_nFeats2Violate = int(float(n_feats)*violationThreshold)
        nFeats_violated = 0
        for col in X.columns:
            cnt = 0
            t, p = shapiro(X[col].values)
            if p < 0.05:
                cnt+=1

            result = kstest(X[col].values, 'norm', args=(X[col].mean(), X[col].std()))
            if result.pvalue < 0.05:
                cnt+=1

            result = anderson(X[col].values, 'norm')
            if result.critical_values[2] < 0.05:
                cnt += 1

            if cnt >=2:
                nFeats_violated += 1
                violated_feats.append(col)
                continue

        if nFeats_violated > minAllowable_nFeats2Violate:
            return False, violated_feats

    elif isinstance(X, np.array):
        pass
    else:
        raise ValueError("X is expected to be either pandas.DataFrame or numpy.array")
