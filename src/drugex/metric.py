#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script is used for measuring some coefficients of the molecules."""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors as desc, Lipinski, MolSurf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as Scaler
from tqdm import tqdm

from drugex import util


def converage(fnames):
    """This method parsed log files of reinforcement learning process
    Arguments:
        fnames (list): List of log file paths

    Returns:
        output (DataFrame): Table contains two columns, one is the index of
            file name in the list, the other is the average reward of batch SMILES
            given by the environment.
    """
    xy = []
    for i, fname in enumerate(fnames):
        lines = open(fname).readlines()
        for line in lines:
            if not line.startswith('Epoch'): continue
            # The average reward at current step in ith log file
            score = float(line.split(' ')[3])
            xy.append([i, score])
    output = pd.DataFrame(xy, columns=['LABEL', 'SCORE'])
    return output


def training_process(fname):
    """This method parsed log files of RNN training process
    Arguments:
        fname (str): log file paths of RNN training

    Returns:
        valid (ndarray): The validation rate at each epoch during the training process.
        loss (ndarray): The value of loss function at each epoch during the training process.
    """
    log = open(fname)
    valid = []
    loss = []
    for line in log:
        if not line.startswith('Epoch:'): continue
        data = line.split(' ')
        valid.append(float(data[-1]))
        loss.append(float(data[-3]))
    valid, loss = np.array(valid), np.array(loss)
    return valid, loss


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
        sub = pd.read_table(fname)
        sub['LABEL'] = i
        if 'PCHEMBL_VALUE' in sub.columns:
            sub = sub[sub.PCHEMBL_VALUE >= (6.5 if is_active else 0)]
        elif 'SCORE' in sub.columns:
            sub = sub[sub.SCORE > (0.5 if is_active else 0)]
        sub = sub.drop_duplicates(subset='CANONICAL_SMILES')
        if not ('LOGP' in sub.columns and 'MWT' in sub.columns):
            # If the the table does not contain LOGP and MWT
            # it will calculate these coefficients with RDKit.
            logp, mwt = [], []
            for i, row in sub.iterrows():
                try:
                    mol = Chem.MolFromSmiles(row.CANONICAL_SMILES)
                    x, y = desc.MolWt(mol), Crippen.MolLogP(mol)
                    logp.append(y)
                    mwt.append(x)
                except:
                    sub = sub.drop(i)
                    print(row.CANONICAL_SMILES)
            sub['LOGP'], sub['MWT'] = logp, mwt
        df = df.append(sub[['MWT', 'LOGP', 'LABEL']])
    return df


def dimension(fnames, fp='ECFP', is_active=False, alg='PCA', maximum=int(1e5)):
    """ Dimension reduction analysis it contains two algorithms: PCA and t-SNE,
    and two different descriptors: ECFP6 and PhysChem

    Arguments:
        fnames (list): List of file paths that contains CANONICAL_SMILES and SCORE (or PCHEMBL_VALUE).
        fp (str, optional): The descriptors for each molecule, either ECFP6 or PhysChem (Default: 'ECFP')
        is_active (bool, optional): selecting only active ligands (True) or all of the molecules (False)
            if it is true, the molecule with PCHEMBL_VALUE >= 6.5 or SCORE > 0.5 will be selected.
            (Default: False)
        alg (str, optional): Dimension reduction algorithms, either 'PCA' or 't-SNE' (Default: 'PCA')
        maximum (int, optional): Considering dimension reduction for the large dataset is extremely
            time- and resource consuming, if the size of dataset in one file larger than this threshold,
            maximum number of sample will be randomly selected (Default: 100,000)

    Returns:
        df (DataFrame): the table contains two columns, component 1 and 2.
    """
    df = pd.DataFrame()
    for i, fname in enumerate(fnames):
        sub = pd.read_table(fname)
        if maximum is not None and len(sub) > maximum:
            sub = sub.sample(maximum)
        if 'PCHEMBL_VALUE' in sub.columns:
            sub = sub[sub.PCHEMBL_VALUE >= (6.5 if is_active else 0)]
            sub['SCORE'] = sub.PCHEMBL_VALUE
        elif 'SCORE' in sub.columns:
            sub = sub[sub.SCORE > (0.5 if is_active else 0)]
        sub = sub.drop_duplicates(subset='CANONICAL_SMILES')
        print(len(sub))
        sub['LABEL'] = i
        df = df.append(sub)

    fp_alg = util.Environment.ECFP_from_SMILES if fp == 'ECFP' else PhyChem
    fps = fp_alg(df.CANONICAL_SMILES)
    pca = PCA(n_components=2) if alg == 'PCA' else TSNE(n_components=2)
    xy = pca.fit_transform(fps)
    df['X'], df['Y'] = xy[:, 0], xy[:, 1]
    if alg == 'PCA':
        ratio = pca.explained_variance_ratio_[:2]
        return df, ratio
    else:
        return df


def substructure(fname, sub, is_active=False):
    """ Calculating the percentage of molecules that contains the given substructure
    in the given dataset.

    Arguments:
        sub (str): molecular substructure with SMARTS representation.
        is_active (bool, optional): selecting only active ligands (True) or all of the molecules (False)
            if it is true, the molecule with PCHEMBL_VALUE >= 6.5 or SCORE > 0.5 will be selected.
            (Default: False)

    Returns:
        percentage (float): percentage of molecules (xx.xx%) that contains the given substructure
    """
    sub = Chem.MolFromSmarts(sub)
    df = pd.read_table(fname).drop_duplicates(subset='CANONICAL_SMILES')
    if 'SCORE' in df.columns:
        df = df[df.SCORE > (0.5 if is_active else 0.0)]
    elif 'PCHEMBL_VALUE' in df.columns:
        df = df[df.PCHEMBL_VALUE >= (6.5 if is_active else 0.0)]
    num = 0
    for smile in df.CANONICAL_SMILES:
        mol = Chem.MolFromSmiles(smile)
        if mol.HasSubstructMatch(sub):
            num += 1
            # print(smile)
    percentage = num * 100 / len(df)
    return percentage


def diversity(fake_path, real_path=None, is_active=False):
    """ Molecular diversity measurement based on Tanimoto-distance on ECFP6 fingerprints,
    including, intra-diversity and inter-diversity.

    Arguments:
        fake_path (str): the file path of molecules that need to measuring diversity

        real_path (str, optional): the file path of molecules as the reference, if it
            is provided, the inter-diversity will be calculated; otherwise, the intra-diversity
            will be calculated.
        is_active (bool, optional): selecting only active ligands (True) or all of the molecules (False)
            if it is true, the molecule with PCHEMBL_VALUE >= 6.5 or SCORE > 0.5 will be selected.
            (Default: False)

    Returns:
        df (DataFrame): the table that contains columns of CANONICAL_SMILES
            and diversity value for each molecules

    """
    fake = pd.read_table(fake_path)
    fake = fake[fake.SCORE > (0.5 if is_active else 0)]
    fake = fake.drop_duplicates(subset='CANONICAL_SMILES')
    fake_fps, real_fps = [], []
    for i, row in fake.iterrows():
        mol = Chem.MolFromSmiles(row.CANONICAL_SMILES)
        fake_fps.append(AllChem.GetMorganFingerprint(mol, 3))
    if real_path:
        real = pd.read_table(real_path)
        real = real[real.PCHEMBL_VALUE >= (6.5 if is_active else 0)]
        for i, row in real.iterrows():
            mol = Chem.MolFromSmiles(row.CANONICAL_SMILES)
            real_fps.append(AllChem.GetMorganFingerprint(mol, 3))
    else:
        real_fps = fake_fps
    method = np.min if real_path else np.mean
    dist = 1 - np.array([method(DataStructs.BulkTanimotoSimilarity(f, real_fps)) for f in fake_fps])
    fake['DIST'] = dist
    return fake


def properties(fnames, labels, is_active=False):
    """ Five structural properties calculation for each molecule in each given file.
    These properties contains No. of Hydrogen Bond Acceptor/Donor, Rotatable Bond,
    Aliphatic Ring, Aromatic Ring and Heterocycle.

    Arguments:
        fnames (list): the file path of molecules.
        labels (list): the label for each file in the fnames.
        is_active (bool, optional): selecting only active ligands (True) or all of the molecules (False)
            if it is true, the molecule with PCHEMBL_VALUE >= 6.5 or SCORE > 0.5 will be selected.
            (Default: False)

    Returns:
        df (DataFrame): the table contains three columns; 'Set' is the label
            of fname the molecule belongs to, 'Property' is the name of one
            of five properties, 'Number' is the property value.
    """

    props = []
    for i, fname in enumerate(fnames):
        df = pd.read_table(fname)
        if 'SCORE' in df.columns:
            df = df[df.SCORE > (0.5 if is_active else 0)]
        elif 'PCHEMBL_VALUE' in df.columns:
            df = df[df.PCHEMBL_VALUE >= (6.5 if is_active else 0)]
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


def clustering(fnames, scaffold=0, is_active=False):
    """ K-means clustering on ECFP6 of each molecule to divide given dataset into 20 clusters.

    Arguments:
        fnames (list): the file path of molecules for clustering
        scaffold (int, optional): the structure used for ECFP6 generation,
            0 == Full compound structure
            1 == Murcko scaffold
            2 == Murcko topological scaffold
        is_active (bool, optional): selecting only active ligands (True) or all of the molecules (False)
            if it is true, the molecule with PCHEMBL_VALUE >= 6.5 or SCORE > 0.5 will be selected.
            (Default: False)

    Returns:
        df (DataFrame): the table contains CANONICAL_SMMILES, index of cluster
            and index of file name in the fnames.
    """
    df = pd.DataFrame()
    if len(df) > int(1e5):
        df = df.sample(int(1e5))
    for i, fname in enumerate(fnames):
        sub = pd.read_table(fname)
        if 'PCHEMBL_VALUE' in sub.columns:
            sub = sub[sub.PCHEMBL_VALUE >= (6.5 if is_active else 0)]
            sub['SCORE'] = sub.PCHEMBL_VALUE
        elif 'SCORE' in sub.columns:
            sub = sub[sub.SCORE > (0.5 if is_active else 0)]
        sub = sub.drop_duplicates(subset='CANONICAL_SMILES')
        print(len(sub))
        sub['LABEL'] = i
        df = df.append(sub)
    fps = util.Environment.ECFP_from_SMILES(df.CANONICAL_SMILES, scaffold=scaffold)
    # cluster = AgglomerativeClustering(n_clusters=20).fit(fps)
    cluster = KMeans(n_clusters=20).fit(fps)
    df['CLUSTER'] = cluster.labels_
    return df


def PhyChem(smiles):
    """ Calculating the 19D physicochemical descriptors for each molecules,
    the value has been normalized with Gaussian distribution.

    Arguments:
        smiles (list): list of SMILES strings.
    Returns:
        props (ndarray): m X 19 matrix as nomalized PhysChem descriptors.
            m is the No. of samples
    """
    props = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        try:
            MW = desc.MolWt(mol)
            LOGP = Crippen.MolLogP(mol)
            HBA = Lipinski.NumHAcceptors(mol)
            HBD = Lipinski.NumHDonors(mol)
            rotable = Lipinski.NumRotatableBonds(mol)
            amide = AllChem.CalcNumAmideBonds(mol)
            bridge = AllChem.CalcNumBridgeheadAtoms(mol)
            heteroA = Lipinski.NumHeteroatoms(mol)
            heavy = Lipinski.HeavyAtomCount(mol)
            spiro = AllChem.CalcNumSpiroAtoms(mol)
            FCSP3 = AllChem.CalcFractionCSP3(mol)
            ring = Lipinski.RingCount(mol)
            Aliphatic = AllChem.CalcNumAliphaticRings(mol)
            aromatic = AllChem.CalcNumAromaticRings(mol)
            saturated = AllChem.CalcNumSaturatedRings(mol)
            heteroR = AllChem.CalcNumHeterocycles(mol)
            TPSA = MolSurf.TPSA(mol)
            valence = desc.NumValenceElectrons(mol)
            mr = Crippen.MolMR(mol)
            # charge = AllChem.ComputeGasteigerCharges(mol)
            prop = [MW, LOGP, HBA, HBD, rotable, amide, bridge, heteroA, heavy, spiro,
                    FCSP3, ring, Aliphatic, aromatic, saturated, heteroR, TPSA, valence, mr]
        except:
            print(smile)
            prop = [0] * 19
        props.append(prop)
    props = np.array(props)
    props = Scaler().fit_transform(props)
    return props
