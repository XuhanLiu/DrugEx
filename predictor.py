import numpy as np
import pandas as pd
import torch as T
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader, TensorDataset
import model
import os
import util
from sklearn.externals import joblib


def TORCH(X, y, X_ind, y_ind, out, reg=False):
    if 'mtqsar' in out or reg:
        folds = KFold(5).split(X)
        NET = model.MTRegressor if reg else model.MTClassifier
    else:
        folds = StratifiedKFold(5).split(X, y[:, 0])
        NET = model.STRegressor if reg else model.STClassifier
    indep_set = TensorDataset(T.Tensor(X_ind), T.Tensor(y_ind))
    indep_loader = DataLoader(indep_set, batch_size=BATCH_SIZE)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        train_set = TensorDataset(T.Tensor(X[trained]), T.Tensor(y[trained]))
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        valid_set = TensorDataset(T.Tensor(X[valided]), T.Tensor(y[valided]))
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
        net = NET(X.shape[1], y.shape[1])
        net.fit(train_loader, valid_loader, out='%s_%d' % (out, i), epochs=N_EPOCH, lr=LR)
        cvs[valided] = net.predict(valid_loader)
        inds += net.predict(indep_loader)
    cv, ind = y == y, y_ind == y_ind
    return cvs[cv], inds[ind] / 5


def RF(X, y, X_ind, y_ind):
    folds = StratifiedKFold(5).split(X, y)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = RandomForestClassifier(n_estimators=500, n_jobs=1)
        model.fit(X[trained], y[trained])
        cvs[valided] = model.predict_proba(X[valided])[:, 1]
        inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def SVM(X, y, X_ind, y_ind):
    folds = StratifiedKFold(5).split(X, y)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    model = SVC(probability=True)
    gs = GridSearchCV(model, {'C': 2.0 ** np.array([-5, 15]), 'gamma': 2.0 ** np.array([-15, 5])}, n_jobs=5)
    gs.fit(X, y)
    params = gs.best_params_
    print(params)
    for i, (trained, valided) in enumerate(folds):
        model = SVC(probability=True, C=params['C'], gamma=params['gamma'])
        model.fit(X[trained], y[trained])
        cvs[valided] = model.predict_proba(X[valided])[:, 1]
        inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def KNN(X, y, X_ind, y_ind):
    folds = StratifiedKFold(5).split(X, y)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = KNeighborsClassifier(n_jobs=1)
        model.fit(X[trained], y[trained])
        cvs[valided] = model.predict_proba(X[valided])[:, 1]
        inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def NB(X, y, X_ind, y_ind):
    folds = StratifiedKFold(5).split(X, y)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = GaussianNB()
        model.fit(X[trained], y[trained])
        cvs[valided] = model.predict_proba(X[valided])[:, 1]
        inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def cross_validation(feat, alg, reg, subset=None):
    df = pd.read_table('data/CHEMBL251.txt')
    y = (df[['PCHEMBL_VALUE']].values >= 6.5).astype(float)
    X = util.Activity.ECFP_from_SMILES(df.CANONICAL_SMILES, 6, 4096)

    out = 'output/%s_%s_%s' % (alg, 'reg' if reg else 'cls', feat)
    if subset: out += '_' + subset

    data = pd.DataFrame()
    test = pd.DataFrame()
    data['CHEMBL_ID'], data['LABEL'] = df.CHEMBL_ID, y[:, 0]
    test['CHEMBL_ID'], test['LABEL'] = df.CHEMBL_ID, y[:, 0]

    if alg == 'RF':
        data['score'], test['score'] = RF(X, y[:, 0], X, y[:, 0])
    elif alg == 'SVM':
        data['score'], test['score'] = SVM(X, y[:, 0], X, y[:, 0])
    elif alg == 'KNN':
        data['score'], test['score'] = KNN(X, y[:, 0], X, y[:, 0])
    elif alg == 'NB':
        data['score'], test['score'] = NB(X, y[:, 0], X, y[:, 0])
    elif alg == 'DNN':
        data['score'], test['score'] = TORCH(X, y, X, y, out=out, reg=reg)
    data.to_csv(out + '.cv.txt', index=None)
    test.to_csv(out + '.ind.txt', index=None)


def Train_RF(X, y, out=''):

    model = RandomForestClassifier(n_estimators=1000, n_jobs=5)
    model.fit(X, y)
    joblib.dump(model, out)


if __name__ == '__main__':
    BATCH_SIZE = 1024
    N_EPOCH = 1000
    T.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    LR = 1e-5
    Train_RF('pred.pkg')