import pandas as pd
import os
from joblib import load

DATADIR_aparc = "D:\\PhD\\Data\\aparc\\all_data.csv"
DATADIR_a2009s = "D:\\PhD\\Data\\a2009s\\all_data.csv"
MAX_ITR = 1000000
PARAM_GRID={
    'lSVM': {
        'C':[0.1,1,5, 10]
    },
    'pagg': {
        'C':[0.1,1,5,10], 'n_iter_no_change':[1,5,10]
    },
    'lg':{
        'penalty': ['l1','l2','elasticnet','none'],
        'C':[0.1,1,5, 10],
        'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    },
    'XGB': {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    },
    'GNB': {

    },
    'Rf': {
        'n_estimators':[50, 100, 200, 500, 1000],
        'criterion':['gini','entropy'],
        'max_features':['auto','sqrt'],
        'min_samples_split':[2,5,10],
        'min_samples_leaf':[0,0.1,0.2,0.3,0.4,0.5],
        'bootstrap':[True,False]

    },
    'SVC': {
        'C':[0.1,1,5, 10],
        'kernel':['poly','rbf','sigmoid'],
        'degree':[2,3,4,5,6],
        'gamma':['scale','auto'],
        'coef0':[0.0,0.01,0.1,0.5,1,5,10,50,100]
    },
    'nn':{
        'hidden_layer_sizes': [(150,100,50,), (100,50,25,), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001,0.001,0.01, 0.05, 0.1, 0.5],
        'learning_rate': ['constant','adaptive'],
        'beta_1':[0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.9],
        'beta_2':[0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.9],
    }
}

OUTPUT_DIR_ML = './Results/ML'
OUTPUT_DIR_FS = './Results/FS'
OUTPUT_DIR_CORR = './Results/CORR_ANA'
OUTPUT_DIR_SPLIT = './Results/INITIAL_SPLIT'


def LOAD_CORRANA_RESULTS():
    df_nocorr_left = pd.read_csv(os.path.join(OUTPUT_DIR_CORR,"left_train_noColliniarity.csv"))
    df_nocorr_right = pd.read_csv(os.path.join(OUTPUT_DIR_CORR,"right_train_noColliniarity.csv"))
    return df_nocorr_left, df_nocorr_right


def LOAD_FS_RESULTS():
    X_clean_right_rf = pd.read_csv(os.path.join(OUTPUT_DIR_FS, "clean_right_rf.csv"), index_col=0)
    X_clean_left_rf = pd.read_csv(os.path.join(OUTPUT_DIR_FS, "clean_left_rf.csv"), index_col=0)
    X_clean_right_lsvc = pd.read_csv(os.path.join(OUTPUT_DIR_FS, "clean_right_svc.csv"), index_col=0)
    X_clean_left_lsvc = pd.read_csv(os.path.join(OUTPUT_DIR_FS, "clean_left_svc.csv"), index_col=0)
    y_left = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT, "left_train.csv"), index_col=1)['labels']
    y_right = pd.read_csv(os.path.join(OUTPUT_DIR_SPLIT, "right_train.csv"), index_col=1)['labels']

    obj_left_svc = load(os.path.join(OUTPUT_DIR_FS,"left_svc.joblib"))
    obj_right_svc = load(os.path.join(OUTPUT_DIR_FS,"right_svc.joblib"))
    obj_left_rf = load(os.path.join(OUTPUT_DIR_FS,"left_rf.joblib"))
    obj_right_rf = load(os.path.join(OUTPUT_DIR_FS,"right_rf.joblib"))

    return (X_clean_left_lsvc, X_clean_right_lsvc, X_clean_left_rf, X_clean_right_rf, y_left, y_right),\
           (obj_left_svc, obj_right_svc, obj_left_rf, obj_right_rf)

