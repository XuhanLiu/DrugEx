import pandas as pd
from rdkit import Chem
from tqdm import tqdm
# from word2vec.dataset import VocM2V
from util import Voc
import numpy as np
import re
import os


def corpus(input, output):
    df = pd.read_table(input).CANONICAL_SMILES
    voc = Voc('output/Voc.txt')
    words = set()
    canons = []
    tokens = []
    smiles = set()
    for smile in tqdm(df):
        smile = re.sub('\[\d+', '[', smile)
        if '.' in smile:
            frags = smile.split('.')
            ix = np.argmax([len(frag) for frag in frags])
            smile = frags[ix]
        if {'C', 'c'}.isdisjoint(smile):
            continue
        if smile in smiles:
            print(smile)
        smiles.add(smile)
    for smile in tqdm(smiles):
        try:
            token = voc.tokenize(smile)
            if len(token) <= 100:
                words.update(token)
                canons.append(Chem.CanonSmiles(smile, 0))
                tokens.append(' '.join(token))
        except:
            print(smile)
    log = open(output + '_voc.txt', 'w')
    log.write('\n'.join(sorted(words)))
    log.close()

    log = pd.DataFrame()
    log['CANONICAL_SMILES'] = canons
    log['SENT'] = tokens
    log.drop_duplicates(subset='CANONICAL_SMILES')
    log.to_csv(output + '_corpus.txt', sep='\t', index=None)


def ZINC(folder):
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
    select.to_csv('data/ZINC_B.txt', sep='\t', index=None)


def A2AR():
    df = pd.read_table('data/CHEMBL251_raw.txt')
    df = df[['CMPD_CHEMBLID', 'CANONICAL_SMILES', 'PCHEMBL_VALUE']]
    df = df.dropna(subset='PCHEMBL_VALUE')
    for i, row in df.iterrows():
        smile = row['CANONICAL_SMILES'].replace('[NH+]', 'N').replace('[NH2+]', 'N').replace('[NH3+]', 'N')
        smile = re.sub('\[\d+', '[', smile)
        if '.' in smile:
            frags = smile.split('.')
            ix = np.argmax([len(frag) for frag in frags])
            smile = frags[ix]
        if '[Au]' in smile or '[As]' in smile or '[Hg]' in smile or '[Se]' in smile or smile.count('C') + smile.count('c') < 2:
            df = df.drop(i)
        df.loc[i, 'CANONICAL_SMILES'] = smile
    df = df.drop_duplicates(subset='CANONICAL_SMILES')
    df.to_csv('data/CHEMBL251.txt', index=False, sep='\t')


if __name__ == '__main__':
    ZINC('zinc/')
    corpus('data/CHEMBL251.txt', 'data/chembl')
    A2AR()