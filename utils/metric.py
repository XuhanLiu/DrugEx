from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import QED
import pandas as pd
from rdkit import DataStructs
import numpy as np
from rdkit import rdBase
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler as Scaler
from scipy import linalg
import torch
from torch.nn import functional as F
from .objective import Predictor

rdBase.DisableLog('rdApp.error')


def pad_mask(seq, pad_idx=0):
    return seq == pad_idx


def tri_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    masks = torch.triu(torch.ones((len_s, len_s)), diagonal=1)
    masks = masks.bool().to(seq.device)
    return masks


def unique(arr):
    # Finds unique rows in arr and return their indices
    if type(arr) == torch.Tensor:
        arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    idxs = np.sort(idxs)
    if type(arr) == torch.Tensor:
        idxs = torch.LongTensor(idxs).to(arr.get_device())
    return idxs


def kl_div(p_logit, q_logit, reduce=False):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.mean(p * (F.log_softmax(p_logit, dim=-1)
                         - F.log_softmax(q_logit, dim=-1)), 1, keepdim=True)
    return torch.mean(_kl) if reduce else _kl


def logP_mw(fnames, is_active=False):
    """ logP and molecular weight calculation for logP ~ MW chemical space visualization

    Arguments:
        fnames (list): List of file paths that contains CANONICAL_SMILES (, LOGP and MWT
            if it contains the logP and molecular weight for each molecule).
        is_active (bool, optional): selecting only active ligands (True) or all of the molecules (False)
            if it is true, the molecule with PCHEMBL_VALUE >= 6.5 or SCORE > 0.5 will be selected.
            (Default: False)

    Returns:
        df (DataFrame)： The table contains three columns;
            molecular weight, logP and index of file name in the fnames
    """
    df = pd.DataFrame()
    for i, fname in enumerate(fnames):
        print(fname)
        sub = pd.read_table(fname).dropna(subset=['Smiles'])
        sub['LABEL'] = i
        if 'Valid' in sub.columns:
            sub = sub[sub.Valid == 1]
        sub = sub.drop_duplicates(subset=['Smiles'])
        if len(sub) > 1e5:
            sub = sub.sample(int(1e5))
        if not ('LOGP' in sub.columns and 'MWT' in sub.columns):
            # If the the table does not contain LOGP and MWT
            # it will calculate these coefficients with RDKit.
            logp, mwt = [], []
            qed = []
            for i, row in sub.iterrows():
                try:
                    mol = Chem.MolFromSmiles(row.Smiles)
                    x, y = desc.MolWt(mol), desc.MolLogP(mol)
                    logp.append(y)
                    mwt.append(x)
                    qed.append(QED.qed(mol))
                except:
                    sub = sub.drop(i)
                    # print(row.Smiles)
            sub['LOGP'], sub['MWT'] = logp, mwt
            sub['QED'] = qed
        df = df.append(sub[['MWT', 'LOGP', 'LABEL', 'QED']])
    return df


def dimension(fnames, fp='ECFP', alg='PCA', maximum=int(1e5)):
    df = pd.DataFrame()
    for i, fname in enumerate(fnames):
        sub = pd.read_table(fname).dropna(subset=['Smiles'])
        sub = sub.drop_duplicates(subset=['Smiles'])
        if len(sub) > 1e5:
            sub = sub.sample(int(1e5))
        if 'Valid' in sub.columns:
            sub = sub[sub.Valid == True]
        if maximum is not None and len(sub) > maximum:
            sub = sub.sample(maximum)
        # if ref not in fname:
        #     sub = sub[sub.Valid == True]
        sub = sub.drop_duplicates(subset='Smiles')
        sub['LABEL'] = i
        df = df.append(sub)

    mols = [Chem.MolFromSmiles(s) for s in df.Smiles]
    df['QED'] = [QED.qed(m) for m in mols]
    if fp == 'similarity':
        ref = df[(df.LABEL == 0)]
        refs = [Chem.MolFromSmiles(s) for s in ref.Smiles]
        refs = Predictor.calc_ecfp_rd(refs)
        fps = Predictor.calc_ecfp_rd(mols)
        from rdkit.Chem import DataStructs
        fps = np.array([DataStructs.BulkTanimotoSimilarity(fp, refs) for fp in fps])
    else:
        fp_alg = Predictor.calc_ecfp if fp == 'ECFP' else Predictor.calc_physchem
        fps = fp_alg(mols)
    fps = Scaler().fit_transform(fps)
    pca = PCA(n_components=2) if alg == 'PCA' else TSNE(n_components=2, n_jobs=10, n_iter=10000)
    xy = pca.fit_transform(fps)
    df['X'], df['Y'] = xy[:, 0], xy[:, 1]
    if alg == 'PCA':
        ratio = pca.explained_variance_ratio_[:2]
        return df, ratio
    else:
        return df, None


def substructure(fname, sub, is_desired=False):
    sub = Chem.MolFromSmarts(sub)
    df = pd.read_table(fname).drop_duplicates(subset='Smiles')
    if is_desired:
        df = df[df.DESIRE == 1]
    else:
        df = df[df.VALID == 1]
    num = 0
    for smile in df.Smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol.HasSubstructMatch(sub):
            num += 1
            # print(smile)
    return num * 100 / len(df)


def diversity(fake_path, real_path=None):
    fake = pd.read_table(fake_path)
    fake = fake[fake.DESIRE == 1]
    fake = fake.drop_duplicates(subset='Smiles')
    fake_fps, real_fps = [], []
    for i, row in fake.iterrows():
        mol = Chem.MolFromSmiles(row.Smiles)
        fake_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048))
    if real_path:
        real = pd.read_table(real_path)
        real = real[real.DESIRE == True]
        for i, row in real.iterrows():
            mol = Chem.MolFromSmiles(row.Smiles)
            real_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048))
    else:
        real_fps = fake_fps
    method = np.max if real_path else np.mean
    score = 1 - np.array([method(DataStructs.BulkTanimotoSimilarity(f, real_fps)) for f in fake_fps])
    fake['DIST'] = score
    return fake


def Solow_Polasky_Diversity(path, is_cor=False):
    N_SAMPLE = 1000
    if is_cor:
        dist = np.loadtxt(path)
    else:
        df = pd.read_table(path)
        # df = df[df.DESIRE == 1]
        df = df.drop_duplicates(subset='Smiles').dropna()
        if len(df) < N_SAMPLE:
            return 0
        df = df.sample(N_SAMPLE)
        fps = []
        for i, row in df.iterrows():
            mol = Chem.MolFromSmiles(row.Smiles)
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 2048))
        dist = 1 - np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])
        np.savetxt(path[:-4] + '.div.tsv', dist, fmt='%.3f')
    ix = unique(dist)
    dist = dist[ix, :][:, ix]
    f_ = linalg.inv(np.e ** (-10 * dist))
    return np.sum(f_) / N_SAMPLE

