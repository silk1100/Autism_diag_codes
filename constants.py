import pandas as pd
import os
from joblib import load

DATADIR_aparc = "D:\\PhD\\Data\\aparc\\all_data.csv"
DATADIR_aparc_MODLEFT = "D:\\PhD\\Data\\aparc\\df_left_newRepresentation.csv"
DATADIR_aparc_MODRight = "D:\\PhD\\Data\\aparc\\df_right_newRepresentation.csv"
# DATADIR_aparc_ALLASDINC = "D:\\PhD\\Data\\aparc\\df_aparc_ALLASDINC.csv"
DATADIR_aparc_ALLASDINC = "D:\\PhD\\Data\\aparc\\DrEid_brain_sMRI_lr_TDASD.csv"
DATADIR_aparc_ALLASDINC_LEFT = 'D:\\PhD\\Data\\aparc\\df_left_aparc_ALLASDINC.csv'
DATADIR_aparc_ALLASDINC_RIGHT = 'D:\\PhD\\Data\\aparc\\df_right_aparc_ALLASDINC.csv'

DATADIR_a2009s = "D:\\PhD\\Data\\a2009s\\all_data.csv"
MAX_ITR = 1000000
PARAM_GRID={
    'lSVM': {
        'penalty':['l1','l2'],
        'loss':['hinge','squared_hinge'],
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
        'booster':['gbtree','gblinear','dart'],
        'learning_rate':[0.001, 0.01, 0.5, 1],
        'min_child_weight': [0.01, 0.5, 1, 10],
        'gamma': [0, 0.1, 1, 5, 50, 100],
        'reg_alpha':[0, 0.001, 0.5, 1, 10],
        'reg_lambda':[0, 0.001, 0.5, 1,  10],
        'colsample_bytree': [0.6, 0.8, 1.0],
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
        'coef0':[0.0,0.01,0.5,5,50,100]
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

OUTPUT_DIR_ML = './Final_Results_sites/ML'
OUTPUT_DIR_FS = './Final_Results_sites/FS'
OUTPUT_DIR_CORR = './Final_Results_sites/CORR_ANA'
OUTPUT_DIR_SPLIT = './Final_Results_sites/INITIAL_SPLIT'
OUTPUT_DIR_site_ana = './Final_Results_sites/site_analysis'

# OUTPUT_DIR_ML = './Final_Results/ML'
# OUTPUT_DIR_FS = './Final_Results/FS'
# OUTPUT_DIR_CORR = './Final_Results/CORR_ANA'
# OUTPUT_DIR_SPLIT = './Final_Results/INITIAL_SPLIT'
# OUTPUT_DIR_site_ana = './Final_Results/site_analysis'

#
# OUTPUT_DIR_ML = './Results/ML'
# OUTPUT_DIR_FS = './Results/FS'
# OUTPUT_DIR_CORR = './Results/CORR_ANA'
# OUTPUT_DIR_SPLIT = './Results/INITIAL_SPLIT'
# OUTPUT_DIR_site_ana = './Results/site_analysis'


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

    obj_left_svc = load(os.path.join(OUTPUT_DIR_FS,"left_svc_modifiedMedPIQR.joblib"))
    obj_right_svc = load(os.path.join(OUTPUT_DIR_FS,"right_svc_modifiedMedPIQR.joblib"))
    obj_left_rf = load(os.path.join(OUTPUT_DIR_FS,"left_rf_modifiedMedPIQR.joblib"))
    obj_right_rf = load(os.path.join(OUTPUT_DIR_FS,"right_rf_modifiedMedPIQR.joblib"))

    return (X_clean_left_lsvc, X_clean_right_lsvc, X_clean_left_rf, X_clean_right_rf, y_left, y_right),\
           (obj_left_svc, obj_right_svc, obj_left_rf, obj_right_rf)


# Features from Compare_FSALONE_VS_CORRALONE.ipynb
SELECTD_50_CORRANA_NO_L_R = ['thick_caudalanteriorcingulate_medMIQR',
 'thick_isthmuscingulate_medMIQR',
 'thick_medialorbitofrontal_medPIQR',
 'thick_parahippocampal_medPIQR',
 'thick_posteriorcingulate_medMIQR',
 'thick_rostralanteriorcingulate_medPIQR',
 'thick_frontalpole_medMIQR',
 'thick_temporalpole_medMIQR',
 'curv_bankssts_medPIQR',
 'curv_caudalanteriorcingulate_medMIQR',
 'curv_caudalanteriorcingulate_medPIQR',
 'curv_caudalmiddlefrontal_medMIQR',
 'curv_caudalmiddlefrontal_medPIQR',
 'curv_cuneus_medMIQR',
 'curv_cuneus_medPIQR',
 'curv_entorhinal_medMIQR',
 'curv_entorhinal_medPIQR',
 'curv_fusiform_medMIQR',
 'curv_fusiform_medPIQR',
 'curv_inferiorparietal_medMIQR',
 'curv_inferiorparietal_medPIQR',
 'curv_inferiortemporal_medMIQR',
 'curv_inferiortemporal_medPIQR',
 'curv_isthmuscingulate_medMIQR',
 'curv_lateraloccipital_medMIQR',
 'curv_lateraloccipital_medPIQR',
 'curv_lateralorbitofrontal_medPIQR',
 'curv_lingual_medMIQR',
 'curv_lingual_medPIQR',
 'curv_medialorbitofrontal_medPIQR',
 'curv_middletemporal_medMIQR',
 'curv_middletemporal_medPIQR',
 'curv_parahippocampal_medMIQR',
 'curv_paracentral_medMIQR',
 'curv_paracentral_medPIQR',
 'curv_parsopercularis_medMIQR',
 'curv_parsopercularis_medPIQR',
 'curv_parsorbitalis_medMIQR',
 'curv_parsorbitalis_medPIQR',
 'curv_parstriangularis_medMIQR',
 'curv_parstriangularis_medPIQR',
 'curv_postcentral_medMIQR',
 'curv_posteriorcingulate_medMIQR',
 'curv_posteriorcingulate_medPIQR',
 'curv_precentral_medMIQR',
 'curv_precentral_medPIQR',
 'curv_precuneus_medPIQR',
 'curv_rostralanteriorcingulate_medMIQR',
 'curv_rostralmiddlefrontal_medPIQR',
 'curv_superiorfrontal_medMIQR',
 'curv_superiorfrontal_medPIQR',
 'curv_superiorparietal_medMIQR',
 'curv_superiorparietal_medPIQR',
 'curv_superiortemporal_medMIQR',
 'curv_supramarginal_medMIQR',
 'curv_supramarginal_medPIQR',
 'curv_frontalpole_medMIQR',
 'curv_frontalpole_medPIQR',
 'curv_temporalpole_medMIQR',
 'curv_temporalpole_medPIQR',
 'curv_transversetemporal_medMIQR',
 'curv_transversetemporal_medPIQR',
 'area_bankssts_medMIQR',
 'area_bankssts_medPIQR',
 'area_caudalanteriorcingulate_medMIQR',
 'area_caudalanteriorcingulate_medPIQR',
 'area_entorhinal_medMIQR',
 'area_entorhinal_medPIQR',
 'area_fusiform_medMIQR',
 'area_isthmuscingulate_medMIQR',
 'area_isthmuscingulate_medPIQR',
 'area_lateralorbitofrontal_medMIQR',
 'area_medialorbitofrontal_medMIQR',
 'area_medialorbitofrontal_medPIQR',
 'area_parahippocampal_medMIQR',
 'area_parahippocampal_medPIQR',
 'area_parsopercularis_medMIQR',
 'area_parsorbitalis_medMIQR',
 'area_parstriangularis_medMIQR',
 'area_parstriangularis_medPIQR',
 'area_posteriorcingulate_medMIQR',
 'area_posteriorcingulate_medPIQR',
 'area_rostralanteriorcingulate_medMIQR',
 'area_rostralanteriorcingulate_medPIQR',
 'area_frontalpole_medMIQR',
 'area_temporalpole_medMIQR',
 'area_temporalpole_medPIQR',
 'area_transversetemporal_medPIQR',
 'vol_caudalanteriorcingulate_medPIQR',
 'vol_entorhinal_medPIQR',
 'vol_fusiform_medMIQR',
 'vol_isthmuscingulate_medMIQR',
 'vol_lateraloccipital_medPIQR',
 'vol_medialorbitofrontal_medMIQR',
 'vol_parahippocampal_medMIQR',
 'vol_superiortemporal_medMIQR',
 'vol_frontalpole_medMIQR',
 'vol_temporalpole_medMIQR',
 'vol_transversetemporal_medMIQR']
SELECTED_LEFT_RF = ['thick_llateraloccipital_medMIQR',
  'thick_lsuperiortemporal_medPIQR',
  'area_lbankssts_medMIQR',
  'area_lpericalcarine_medPIQR',
  'area_lsupramarginal_medPIQR',
  'vol_lbankssts_medMIQR']
SELECTED_LEFT_SVM = ['thick_linferiorparietal_medPIQR',
  'thick_lsuperiorparietal_medPIQR',
  'vol_linferiorparietal_medPIQR',
  'vol_llateraloccipital_medPIQR',
  'vol_lsuperiortemporal_medPIQR']

SELECTED_RIGHT_RF = ['thick_rsuperiorparietal_medPIQR',
  'thick_rsuperiortemporal_medPIQR',
  'area_rbankssts_medMIQR',
  'area_rpericalcarine_medPIQR',
  'area_rsupramarginal_medPIQR',
  'vol_rmedialorbitofrontal_medPIQR']
SELECTED_RIGHT_SVM = ['thick_rsuperiorparietal_medPIQR',
  'thick_rinferiorparietal_medPIQR',
  'vol_rinferiorparietal_medPIQR',
  'vol_rlateraloccipital_medPIQR',
  'vol_rsuperiortemporal_medPIQR']

TARGET = 'labels'
ASD = 1
TD = 0

SELECTED_FEATS_DICT_CORR={
    '50': SELECTD_50_CORRANA_NO_L_R
}

SELECTED_FEATS_DICT = {
    'RF-FS-LEFT': SELECTED_LEFT_RF,
    'RF-FS-RIGHT': SELECTED_RIGHT_RF,
    'SVM-FS-LEFT': SELECTED_LEFT_SVM,
    'SVM-FS-RIGHT': SELECTED_RIGHT_SVM,
}
DROPBOX_APP_KEY = "9gojgvexrdkoyc5"
DROPBOX_APP_SECRET = "4mbp9947p18esy9"
DROPBOX_API_TOKEN = "_4-XRPxQoNgAAAAAAAAAAbYReUH3nhcGqbelDP_eRS0jVmjAPLtFFIPDD_YrL2Mv"