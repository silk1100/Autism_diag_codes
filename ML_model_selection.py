from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import  XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score, make_scorer
from constants import MAX_ITR, PARAM_GRID


CLC_DIST={
    'lSVM': LinearSVC(max_iter=MAX_ITR),
    'pagg': PassiveAggressiveClassifier(n_jobs=-1),
    'lg': LogisticRegression(max_iter=MAX_ITR, n_jobs=-1),
    'XGB':XGBClassifier(validate_parameters=True, n_jobs=-1),
    'GNB': GaussianNB(),
    'Rf': RandomForestClassifier(n_jobs=-1),
    'SVC':SVC(max_iter=MAX_ITR),
    'nn':MLPClassifier(max_iter=MAX_ITR)
}


def _perform_randomizedSearch(clc, grid, cv, scorer):
    grid_size = 1
    MIN_GRID_SAMPLES = 1000
    for key in grid:
        grid_size *= len(grid[key])
    # sampled points should be 10% of the grid except if it is less that 1000. Then it is 1000 pts
    # except if the whole grid is less than 1000 pts then we perform exhaustive search
    # will be the whole grid
    npts = int(0.11*grid_size)
    if npts < MIN_GRID_SAMPLES:
        if grid_size < MIN_GRID_SAMPLES:
            npts = grid_size
        else:
            npts = MIN_GRID_SAMPLES

    clf = RandomizedSearchCV(clc, param_distributions=grid, n_iter=npts, n_jobs=-1, scoring=scorer,
                             refit=True, cv=cv, verbose=3, return_train_score=True)

    return clf


def optimize_clc(clc, grid, X, y, cv):
    scorer = make_scorer(balanced_accuracy_score, greater_is_better=True)

    clf = _perform_randomizedSearch(clc,grid,cv, scorer)
    # clf = GridSearchCV(clc, param_grid=grid, scoring=scorer, n_jobs=-1, cv=cv, verbose=3,
    #                    return_train_score=True, refit=True)

    clf_temp = clf.fit(X, y)
    if clf_temp is None:
        return clf
    else:
        return clf_temp


def train_clc(clc, X, y):
    clx = clc.fit(X, y)
    if clx is None:
        return clc
    else:
        return clx


def evaluate_clc(clc, X, y):
    yhat = clc.predict(X)
    scoref1 = f1_score(y.values, yhat, average='weighted')
    scorebacc = balanced_accuracy_score(y, yhat)
    return scoref1, scorebacc


def get_score_dict(clc, Xtr, ytr, Xte, yte, scores_dict, key):
    rf_clc = train_clc(clc, Xtr, ytr)
    fscore, bscor = evaluate_clc(rf_clc, Xte, yte)
    scores_dict[key].append((fscore, bscor, rf_clc))
    return scores_dict


def train_models(X, y, cv):
    stratcv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=49716)
    results_dict = dict()
    for clc in CLC_DIST:
        # Go check the function because I updated it to perform random search instead of exhaustive search
        clf = optimize_clc(CLC_DIST[clc], PARAM_GRID[clc], X, y, cv=stratcv)
        results_dict[clc] = clf
    return results_dict

from sklearn import datasets

def main():
    iris = datasets.load_iris()
    train_models(iris.data, iris.target, 5)


if __name__ == '__main__':
    main()