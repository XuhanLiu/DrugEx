#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file is used for dataset construction.

It contains two dataset as follows:

1. ZINC set: it is used for pre-training model
2. A2AR set: it is used for fine-tuning model and training predictor
"""

import os
import re

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from drugex.util import Voc


def corpus(input, out):
    """Constructing the molecular corpus by splitting each SMILES into
    a range of tokens contained in vocabulary.

    Arguments:
        input (str): the path of tab-delimited data file that contains CANONICAL_SMILES.
        out (str): the path for vocabulary (containing all of tokens for SMILES construction)
            and output table (including CANONICAL_SMILES and whitespace delimited token sentence)
    """
    df = pd.read_table(input).CANONICAL_SMILES
    voc = Voc('data/voc.txt')
    words = set()
    canons = []
    tokens = []
    smiles = set()
    for smile in tqdm(df):
        # replacing the radioactive atom into nonradioactive atom
        smile = re.sub('\[\d+', '[', smile)
        # reserving the largest one if the molecule contains more than one fragments,
        # which are seperated by '.'.
        if '.' in smile:
            frags = smile.split('.')
            ix = np.argmax([len(frag) for frag in frags])
            smile = frags[ix]
        # if it doesn't contain carbon atom, it cannot be drug-like molecule, just remove
        if smile.count('C') + smile.count('c') < 2:
            continue
        if smile in smiles:
            print(smile)
        smiles.add(smile)
    # collecting all of the tokens in the sentences for vocabulary construction.
    for smile in tqdm(smiles):
        try:
            token = voc.tokenize(smile)
            if len(token) <= 100:
                words.update(token)
                canons.append(Chem.CanonSmiles(smile, 0))
                tokens.append(' '.join(token))
        except:
            print(smile)
    # persisting the vocabulary on the hard drive.
    log = open(out + '_voc.txt', 'w')
    log.write('\n'.join(sorted(words)))
    log.close()

    # saving the canonical smiles and token sentences as a table into hard drive.
    log = pd.DataFrame()
    log['CANONICAL_SMILES'] = canons
    log['SENT'] = tokens
    log.drop_duplicates(subset='CANONICAL_SMILES')
    log.to_csv(out + '_corpus.txt', sep='\t', index=None)


def ZINC(folder, out):
    """Uniformly random selecting molecule from ZINC database for Construction of ZINC set,
    which is used for pre-trained model training.

    Arguments:
        folder (str): the directory of the ZINC database, it contains all of the molecules
            that are seperated into different files based on the logP and molecular weight.
        out (str): the file path of output dataframe, it contains all of randomly selected molecules,
            also including its SMILES string, logP and molecular weight
    """
    files = os.listdir(folder)
    points = [(i, j) for i in range(200, 600, 25) for j in np.arange(-2, 6, 0.5)]
    select = pd.DataFrame()
    for symbol in tqdm([i+j for i in 'ABCDEFGHIJK' for j in 'ABCDEFGHIJK']):
        zinc = pd.DataFrame()
        for fname in files:
            if not fname.endswith('.txt'): continue
            if not fname.startswith(symbol): continue
            df = pd.read_table(folder+fname)[['mwt', 'logp', 'smiles']]
            df.columns = ['MWT', 'LOGP', 'CANONICAL_SMILES']
            zinc = zinc.append(df)
        for mwt, logp in points:
            df = zinc[(zinc.MWT > mwt) & (zinc.MWT <= (mwt + 25))]
            df = df[(df.LOGP > logp) & (df.LOGP <= (logp+0.5))]
            if len(df) > 2500:
                df = df.sample(2500)
            select = select.append(df)
    select.to_csv(out, sep='\t', index=None)


def A2AR(input, out):
    """Construction of A2AR set, which is used for fine-tuned model and predictor training.
    Arguments:
        input (str): the path of tab-delimited data file that contains CANONICAL_SMILES.
        out (str): the path saving the refined data after filtering the invalid data,
            including removing molecule contained metal atom, reserving the largest fragments,
            and replacing the nitrogen electrical group to nitrogen atom "N".
    """
    df = pd.read_table(input)
    df = df[['CMPD_CHEMBLID', 'CANONICAL_SMILES', 'PCHEMBL_VALUE']]
    df = df.dropna()
    for i, row in df.iterrows():
        # replacing the nitrogen electrical group to nitrogen atom "N"
        smile = row['CANONICAL_SMILES'].replace('[NH+]', 'N').replace('[NH2+]', 'N').replace('[NH3+]', 'N')
        # removing the radioactivity of each atom
        smile = re.sub('\[\d+', '[', smile)
        # reserving the largest fragments
        if '.' in smile:
            frags = smile.split('.')
            ix = np.argmax([len(frag) for frag in frags])
            smile = frags[ix]
        # Transforming into canonical SMILES based on the Rdkit built-in algorithm.
        df.loc[i, 'CANONICAL_SMILES'] = Chem.CanonSmiles(smile, 0)
        # removing molecule contained metal atom
        if '[Au]' in smile or '[As]' in smile or '[Hg]' in smile or '[Se]' in smile or smile.count('C') + smile.count('c') < 2:
            df = df.drop(i)
    # df = df.drop_duplicates(subset='CANONICAL_SMILES')
    df.to_csv(out, index=False, sep='\t')


def main():
    ZINC('zinc/', 'data/ZINC.txt')
    corpus('data/zinc.txt', 'data/zinc')
    A2AR('data/A2AR_raw.txt', 'data/CHEMBL251.txt')

if __name__ == '__main__':
    main()
