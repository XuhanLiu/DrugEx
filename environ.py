#!/usr/bin/env python
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader, TensorDataset
import models
import os
import utils
import joblib
from copy import deepcopy
from rdkit import Chem


def SVM(X, y, X_ind, y_ind, reg=False):
    """ Cross validation and Independent test for SVM classifion/regression model.

        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.

            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.

            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.

            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.

            reg (bool): it True, the training is for regression, otherwise for classification.


         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

    """
    if reg:
        folds = KFold(5).split(X)
        alg = SVR()
    else:
        folds = StratifiedKFold(5).split(X, y)
        alg = SVC(probability=True)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    gs = GridSearchCV(deepcopy(alg), {'C': 2.0 ** np.array([-15, 15]), 'gamma': 2.0 ** np.array([-15, 15])}, n_jobs=10)
    gs.fit(X, y)
    params = gs.best_params_
    print(params)
    for i, (trained, valided) in enumerate(folds):
        model = deepcopy(alg)
        model.C = params['C']
        model.gamma = params['gamma']
        if not reg:
            model.probability=True
        model.fit(X[trained], y[trained], sample_weight=[1 if v >= 4 else 0.1 for v in y[trained]])
        if reg:
            cvs[valided] = model.predict(X[valided])
            inds += model.predict(X_ind)
        else:
            cvs[valided] = model.predict_proba(X[valided])[:, 1]
            inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def RF(X, y, X_ind, y_ind, reg=False):
    """ Cross validation and Independent test for RF classifion/regression model.

        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.

            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.

            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.

            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.

            reg (bool): it True, the training is for regression, otherwise for classification.


         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

    """
    if reg:
        folds = KFold(5).split(X)
        alg = RandomForestRegressor
    else:
        folds = StratifiedKFold(5).split(X, y)
        alg = RandomForestClassifier
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = alg(n_estimators=1000, n_jobs=10)
        model.fit(X[trained], y[trained], sample_weight=[1 if v >= 4 else 0.1 for v in y[trained]])
        if reg:
            cvs[valided] = model.predict(X[valided])
            inds += model.predict(X_ind)
        else:
            cvs[valided] = model.predict_proba(X[valided])[:, 1]
            inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def KNN(X, y, X_ind, y_ind, reg=False):
    """ Cross validation and Independent test for KNN classifion/regression model.

        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.

            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.

            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.

            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.

            reg (bool): it True, the training is for regression, otherwise for classification.


         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

    """
    if reg:
        folds = KFold(5).split(X)
        alg = KNeighborsRegressor
    else:
        folds = StratifiedKFold(5).split(X, y)
        alg = KNeighborsClassifier
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = alg(n_jobs=10)
        model.fit(X[trained], y[trained])
        if reg:
            cvs[valided] = model.predict(X[valided])
            inds += model.predict(X_ind)
        else:
            cvs[valided] = model.predict_proba(X[valided])[:, 1]
            inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def NB(X, y, X_ind, y_ind):
    """ Cross validation and Independent test for Naive Bayes classifion model.

        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.

            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.

            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.

            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.

         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

    """
    folds = KFold(5).split(X)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = GaussianNB()
        model.fit(X[trained], y[trained], sample_weight=[1 if v >= 4 else 0.1 for v in y[trained]])
        cvs[valided] = model.predict_proba(X[valided])[:, 1]
        inds += model.predict_proba(X_ind)[:, 1]
    return cvs, inds / 5


def PLS(X, y, X_ind, y_ind):
    """ Cross validation and Independent test for PLS regression model.

        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.

            y (np.ndarray): m-d label array for cross validation, where m is the number of samples and
                equals to row of X.

            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.

            y_ind (np.ndarray): m-d label array for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.

            reg (bool): it True, the training is for regression, otherwise for classification.


         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

    """
    folds = KFold(5).split(X)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        model = PLSRegression()
        model.fit(X[trained], y[trained])
        cvs[valided] = model.predict(X[valided])[:, 0]
        inds += model.predict(X_ind)[:, 0]
    return cvs, inds / 5


def DNN(X, y, X_ind, y_ind, out, reg=False):
    """ Cross validation and Independent test for DNN classifion/regression model.

        Arguments:
            X (np.ndarray): m x n feature matrix for cross validation, where m is the number of samples
                and n is the number of features.

            y (np.ndarray): m x l label matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types.

            X_ind (np.ndarray): m x n Feature matrix for independent set, where m is the number of samples
                and n is the number of features.

            y_ind (np.ndarray): m-d label arrays for independent set, where m is the number of samples and
                equals to row of X_ind, and l is the number of types.

            reg (bool): it True, the training is for regression, otherwise for classification.


         Returns:
            cvs (np.ndarray): m x l result matrix for cross validation, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

            inds (np.ndarray): m x l result matrix for independent test, where m is the number of samples and
                equals to row of X, and l is the number of types and equals to row of X.

    """
    if y.shape[1] > 1 or reg:
        folds = KFold(5).split(X)
    else:
        folds = StratifiedKFold(5).split(X, y[:, 0])
    NET = models.STFullyConnected if y.shape[1] == 1 else models.MTFullyConnected
    indep_set = TensorDataset(torch.Tensor(X_ind), torch.Tensor(y_ind))
    indep_loader = DataLoader(indep_set, batch_size=BATCH_SIZE)
    cvs = np.zeros(y.shape)
    inds = np.zeros(y_ind.shape)
    for i, (trained, valided) in enumerate(folds):
        train_set = TensorDataset(torch.Tensor(X[trained]), torch.Tensor(y[trained]))
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
        valid_set = TensorDataset(torch.Tensor(X[valided]), torch.Tensor(y[valided]))
        valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
        net = NET(X.shape[1], y.shape[1], reg=reg)
        net.fit(train_loader, valid_loader, out='%s_%d' % (out, i), epochs=N_EPOCH, lr=LR)
        cvs[valided] = net.predict(valid_loader)
        inds += net.predict(indep_loader)
    return cvs, inds / 5


def Train_RF(X, y, out, reg=False):
    if reg:
        model = RandomForestRegressor(n_estimators=1000, n_jobs=10)
    else:
        model = RandomForestClassifier(n_estimators=1000, n_jobs=10)
    model.fit(X, y, sample_weight=[1 if v >= 4 else 0.1 for v in y])
    joblib.dump(model, out, compress=3)


def mt_task(fname, out, reg=False, is_extra=True, time_split=False):
    df = pd.read_table(fname)[pair].dropna(subset=pair[1:2])
    df = df[df.Target_ChEMBL_ID.isin(trgs)]
    year = df.groupby(pair[1])[pair[-1:]].min().dropna()
    year = year[year.Document_Year > 2015].index
    df = df[pair].set_index(pair[0:2])
    numery = df[pair[2]].groupby(pair[0:2]).mean().dropna()

    comments = df[(df.Comment.str.contains('Not Active') == True)]
    inhibits = df[(df.Standard_Type == 'Inhibition') & df.Standard_Relation.isin(['<', '<='])]
    relations = df[df.Standard_Type.isin(['EC50', 'IC50', 'Kd', 'Ki']) & df.Standard_Relation.isin(['>', '>='])]
    binary = pd.concat([comments, inhibits, relations], axis=0)
    binary = binary[~binary.index.isin(numery.index)]
    binary[pair[2]] = 3.99
    binary = binary[pair[2]].groupby(pair[0:2]).first()
    df = numery.append(binary) if is_extra else numery
    if not reg:
        df[pair[2]] = (df[pair[2]] > th).astype(float)
    df = df.unstack(pair[0])
    test_ix = set(df.index).intersection(year)

    df_test = df.loc[test_ix] if time_split else df.sample(len(test_ix))
    df_data = df.drop(df_test.index)
    df_data = df_data.sample(len(df_data))
    for alg in ['RF', 'MT_DNN', 'SVM', 'PLS', 'KNN', 'DNN']:
        if alg == 'MT_DNN':
            test_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in df_test.index])
            data_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in df_data.index])
            scaler = Scaler(); scaler.fit(data_x)
            test_x = scaler.transform(test_x)
            data_x = scaler.transform(data_x)

            data = df_data.stack().to_frame(name='Label')
            test = df_test.stack().to_frame(name='Label')
            data_p, test_p = DNN(data_x, df_data.values, test_x, df_test.values, out=out, reg=reg)
            data['Score'] = pd.DataFrame(data_p, index=df_data.index, columns=df_data.columns).stack()
            test['Score'] = pd.DataFrame(test_p, index=df_test.index, columns=df_test.columns).stack()
            data.to_csv(out + alg + '_LIGAND.cv.tsv', sep='\t')
            test.to_csv(out + alg + '_LIGAND.ind.tsv', sep='\t')
        else:
            for trg in trgs:
                test_y = df_test[trg].dropna()
                data_y = df_data[trg].dropna()
                test_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in test_y.index])
                data_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in data_y.index])
                if alg != 'RF':
                    scaler = Scaler(); scaler.fit(data_x)
                    test_x = scaler.transform(test_x)
                    data_x = scaler.transform(data_x)
                else:
                    X = np.concatenate([data_x, test_x], axis=0)
                    y = np.concatenate([data_y.values, test_y.values], axis=0)
                    Train_RF(X, y, out=out + '%s_%s.pkg' % (alg, trg), reg=reg)
                data, test = data_y.to_frame(name='Label'), test_y.to_frame(name='Label')
                a, b = cross_validation(data_x, data.values, test_x, test.values,
                                        alg, out + '%s_%s' % (alg, trg), reg=reg)
                data['Score'], test['Score'] = a, b
                data.to_csv(out + '%s_%s.cv.tsv' % (alg, trg), sep='\t')
                test.to_csv(out + '%s_%s.ind.tsv' % (alg, trg), sep='\t')


def single_task(feat, alg='RF', reg=False, is_extra=True):
    df = pd.read_table('data/LIGAND_RAW.tsv').dropna(subset=pair[1:2])
    df = df[df[pair[0]] == feat]
    df = df[pair].set_index(pair[1])
    year = df[pair[-1:]].groupby(pair[1]).min().dropna()
    test = year[year[pair[-1]] > 2015].index
    numery = df[pair[2]].groupby(pair[1]).mean().dropna()

    comments = df[(df.Comment.str.contains('Not Active') == True)]
    inhibits = df[(df.Standard_Type == 'Inhibition') & df.Standard_Relation.isin(['<', '<='])]
    relations = df[df.Standard_Type.isin(['EC50', 'IC50', 'Kd', 'Ki']) & df.Standard_Relation.isin(['>', '>='])]
    binary = pd.concat([comments, inhibits, relations], axis=0)
    binary = binary[~binary.index.isin(numery.index)]
    binary[pair[2]] = 3.99
    binary = binary[pair[2]].groupby(binary.index).first()
    df = numery.append(binary) if is_extra else numery
    if not reg:
        df = (df > th).astype(float)
    df = df.sample(len(df))
    print(feat, len(numery[numery >= th]), len(numery[numery < th]), len(binary))

    test_ix = set(df.index).intersection(test)
    test = df.loc[test_ix].dropna()
    data = df.drop(test.index)

    test_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in test.index])
    data_x = utils.Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in data.index])
    out = 'output/single/%s_%s_%s' % (alg, 'REG' if reg else 'CLS', feat)
    if alg != 'RF':
        scaler = Scaler(); scaler.fit(data_x)
        test_x = scaler.transform(test_x)
        data_x = scaler.transform(data_x)
    else:
        X = np.concatenate([data_x, test_x], axis=0)
        y = np.concatenate([data.values, test.values], axis=0)
        Train_RF(X, y[:, 0], out=out + '.pkg', reg=reg)
    data, test = data.to_frame(name='Label'), test.to_frame(name='Label')
    data['Score'], test['Score'] = cross_validation(data_x, data.values, test_x, test.values, alg, out, reg=reg)
    data.to_csv(out + '.cv.tsv', sep='\t')
    test.to_csv(out + '.ind.tsv', sep='\t')


def cross_validation(X, y, X_ind, y_ind, alg='DNN', out=None, reg=False):
    if alg == 'RF':
        cv, ind = RF(X, y[:, 0], X_ind, y_ind[:, 0], reg=reg)
    elif alg == 'SVM':
        cv, ind = SVM(X, y[:, 0], X_ind, y_ind[:, 0], reg=reg)
    elif alg == 'KNN':
        cv, ind = KNN(X, y[:, 0], X_ind, y_ind[:, 0], reg=reg)
    elif alg == 'NB':
        cv, ind = NB(X, y[:, 0], X_ind, y_ind[:, 0])
    elif alg == 'PLS':
        cv, ind = PLS(X, y[:, 0], X_ind, y_ind[:, 0])
    elif alg == 'DNN':
        cv, ind = DNN(X, y, X_ind, y_ind, out=out, reg=reg)
    return cv, ind


if __name__ == '__main__':
    pair = ['Target_ChEMBL_ID', 'Smiles', 'pChEMBL_Value', 'Comment',
            'Standard_Type', 'Standard_Relation', 'Document_Year']
    BATCH_SIZE = int(2 ** 11)
    N_EPOCH = 1000
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    th= 6.5
    trgs = ['CHEMBL226', 'CHEMBL251', 'CHEMBL240']

    for reg in [False, True]:
        LR = 1e-4 if reg else 1e-5
        for chembl in trgs:
            single_task(chembl, 'DNN', reg=reg)
            single_task(chembl, 'RF', reg=reg)
            single_task(chembl, 'SVM', reg=reg)
            if reg:
                single_task(chembl, 'PLS', reg=reg)
            else:
                single_task(chembl, 'NB', reg=reg)
            single_task(chembl, 'KNN', reg=reg)

    mt_task('data/LIGAND_RAW.tsv', 'output/random_split/', reg=reg, time_split=False)
    mt_task('data/LIGAND_RAW.tsv', 'output/time_split/', reg=reg, time_split=True)
