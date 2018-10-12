from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from rdkit import DataStructs
import numpy as np
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors as desc
from rdkit.Chem import Lipinski
import util
from tqdm import tqdm
from sklearn.decomposition import PCA


def measure(smiles, fps):
    best_score = []
    best_drugs = set()
    valid = 0
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        if mol is None: continue
        ecfp = AllChem.GetMorganFingerprint(mol, 6)
        scores = DataStructs.BulkTanimotoSimilarity(ecfp, fps)
        best_score.append(max(scores))
        if max(scores) > 0.4:
            valid += 1
            best_drugs.add(smile)
    return best_drugs, np.mean(best_score), valid


def rf_predict(rf, smiles):
    scores = []
    best_drugs = set()
    valid = 0
    fps = util.Activity.ECFP_from_SMILES(smiles, 6, 4096)
    rewards = rf.predict_proba(fps)[:, 1]
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            scores.append(0)
            continue
        else:
            scores.append(rewards[i])
        if max(scores) > 0.5:
            valid += 1
            best_drugs.add(smile)
    return best_drugs, np.mean(scores), valid


def converage(fnames):
    xy = []
    for i, fname in enumerate(fnames):
        lines = open(fname).readlines()
        for line in lines:
            if not line.startswith('Epoch'): continue
            ix, score = line.split(' ')[1], float(line.split(' ')[3])
            xy.append([i, score])
    return pd.DataFrame(xy, columns=['LABEL', 'SCORE'])


def logP_mw(fnames, active=False):
    df = pd.DataFrame()
    for i, fname in enumerate(fnames):
        sub = pd.read_table(fname)
        sub['LABEL'] = i
        if 'PCHEMBL_VALUE' in sub.columns:
            sub = sub[sub.PCHEMBL_VALUE >= (6.5 if active else 0)]
        elif 'SCORE' in sub.columns:
            sub = sub[sub.SCORE > (0.5 if active else 0)]
        sub = sub.drop_duplicates(subset='CANONICAL_SMILES')
        if not ('LOGP' in sub.columns and 'MWT' in sub.columns):
            logp, mwt = [], []
            for i, row in sub.iterrows():
                mol = Chem.MolFromSmiles(row.CANONICAL_SMILES)
                logp.append(Crippen.MolLogP(mol))
                mwt.append(desc.MolWt(mol))
            sub['LOGP'], sub['MWT'] = logp, mwt
        df = df.append(sub[['MWT', 'LOGP', 'LABEL']])
    return df


def pca(fnames, active=False):
    df = pd.DataFrame()
    if len(df) > int(1e5):
        df = df.sample(int(1e5))
    for i, fname in enumerate(fnames):
        sub = pd.read_table(fname)
        if 'PCHEMBL_VALUE' in sub.columns:
            sub = sub[sub.PCHEMBL_VALUE >= (6.5 if active else 0)]
            sub['SCORE'] = sub.PCHEMBL_VALUE
        elif 'SCORE' in sub.columns:
            sub = sub[sub.SCORE > (0.5 if active else 0)]
        sub = sub.drop_duplicates(subset='CANONICAL_SMILES')
        print(len(sub))
        sub['LABEL'] = i
        df = df.append(sub)

    fps = util.Activity.ECFP_from_SMILES(df.CANONICAL_SMILES)
    pca = PCA(n_components=2)
    xy = pca.fit_transform(fps)
    ratio = pca.explained_variance_ratio_[:2]
    df['X'], df['Y'] = xy[:, 0], xy[:, 1]
    return df, ratio


def count(fnames):
    sub = Chem.MolFromSmiles('c1ccco1')
    for fname in fnames:
        df = pd.read_table(fname)
        if 'SCORE' in df.columns:
            df = df[df.SCORE > 0.0]
        elif 'PCHEMBL_VALUE' in df.columns:
            df = df[df.PCHEMBL_VALUE >= 0.5]
        num = 0
        for smile in df.CANONICAL_SMILES:
            mol = Chem.MolFromSmiles(smile)
            if mol.HasSubstructMatch(sub):
                # , sif re.search(r'[a-zA-Z]\d\d', smile):
                num += 1
                # print(smile)
        print(num * 100 / len(df))


def diversity(fake_path, real_path=None):
    fake = pd.read_table(fake_path)
    fake = fake[fake.SCORE > 0.5]
    fake = fake.drop_duplicates(subset='CANONICAL_SMILES')
    fake_fps, real_fps = [], []
    for i, row in fake.iterrows():
        mol = Chem.MolFromSmiles(row.CANONICAL_SMILES)
        fake_fps.append(AllChem.GetMorganFingerprint(mol, 3))
    if real_path:
        real = pd.read_table(real_path)
        real = real[real.PCHEMBL_VALUE >= 6.5]
        for i, row in real.iterrows():
            mol = Chem.MolFromSmiles(row.CANONICAL_SMILES)
            real_fps.append(AllChem.GetMorganFingerprint(mol, 3))
    else:
        real_fps = fake_fps
    method = np.max if real_path else np.mean
    dist = 1 - np.array([method(DataStructs.BulkTanimotoSimilarity(f, real_fps)) for f in fake_fps])
    fake['DIST'] = dist
    return fake


def properties(fnames, labels, active=False):
    props = []
    for i, fname in enumerate(fnames):
        df = pd.read_table(fname)
        if 'SCORE' in df.columns:
            df = df[df.SCORE > (0.5 if active else 0)]
        elif 'PCHEMBL_VALUE' in df.columns:
            df = df[df.PCHEMBL_VALUE >= (6.5 if active else 0)]
        df = df.drop_duplicates(subset='CANONICAL_SMILES')
        if len(df) > int(1e5):
            df = df.sample(int(1e5))
        for smile in tqdm(df.CANONICAL_SMILES):
            mol = Chem.MolFromSmiles(smile)
            HA = Lipinski.NumHAcceptors(mol)
            props.append([labels[i], 'Hydrogen Bond\nAcceptor', HA])
            HD = Lipinski.NumHDonors(mol)
            props.append([labels[i], 'Hydrogen\nBond Donor', HD])
            RB = Lipinski.NumRotatableBonds(mol)
            props.append([labels[i], 'Rotatable\nBond', RB])
            RI = AllChem.CalcNumAliphaticRings(mol)
            props.append([labels[i], 'Aliphatic\nRing', RI])
            AR = Lipinski.NumAromaticRings(mol)
            props.append([labels[i], 'Aromatic\nRing', AR])
            HC = AllChem.CalcNumHeterocycles(mol)
            props.append([labels[i], 'Heterocycle', HC])
    df = pd.DataFrame(props, columns=['Set', 'Property', 'Number'])
    return df


def training_process(fname):
    log = open(fname)
    valid = []
    loss = []
    for line in log:
        if not line.startswith('Epoch:'): continue
        data = line.split(' ')
        valid.append(float(data[-1]))
        loss.append(float(data[-3]))
    return np.array(valid), np.array(loss)