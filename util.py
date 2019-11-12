#!/usr/bin/env python
# This script provides the common methods and data structures that
# are generally used in the project
import torch
from torch.utils.data import Dataset
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import rdBase
import os
from sklearn.externals import joblib
from rdkit.Chem.Scaffolds import MurckoScaffold

torch.set_num_threads(1)
rdBase.DisableLog('rdApp.error')
# designate the device that the PyTorch is allowed to use.
dev = torch.device('cuda')


class Voc(object):
    """ Vocabulary Class for all of the tokens for SMILES string construction.
    It also provides the method to encode SMILES string into the index array of tokens
    and decode the index array into the SMILES string.

    Arguments:
        path (str): the path of vocabulary file that contains all of the tokens split by '\n'
    """
    def __init__(self, path, max_len=100):
        self.chars = ['EOS', 'GO']
        if path is not None and os.path.exists(path):
            f = open(path, 'r')
            chars = f.read().split()
            assert len(set(chars)) == len(chars)
            self.chars += chars
        self.size = len(self.chars)
        # dict -> {token: index} for encoding
        self.tk2ix = dict(zip(self.chars, range(len(self.chars))))
        # dict -> {index: token} for decoding
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}
        self.max_len = max_len

    def tokenize(self, smile):
        """Transform a SMILES string into a series of tokens
        Arguments:
            smile (str): SMILES string with correct grammar

        Returns:
             tokens (list): the list of tokens that are contained in the vocabulary
        """
        regex = '(\[[^\[\]]{1,6}\])'
        smile = re.sub('\[\d+', '[', smile)
        smile = Chem.CanonSmiles(smile, 0)
        smile = smile.replace('Cl', 'L').replace('Br', 'R')
        tokens = []
        for word in re.split(regex, smile):
            if word == '' or word is None: continue
            if word.startswith('['):
                tokens.append(word)
            else:
                for i, char in enumerate(word):
                    tokens.append(char)
        return tokens

    def encode(self, tokens):
        """ Encoding a series of tokens into a SMILES string

        Arguments：
            tokens (list): a series of tokens. Commonly, it is the output of
                the "tokenize" method.
        Returns:
            arr (LongTensor): a long tensor storing the indices of all tokens for one SMILES
s        """
        arr = torch.zeros(len(tokens)).long()
        for i, char in enumerate(tokens):
            arr[i] = self.tk2ix[char]
        return arr

    def decode(self, arr):
        """Takes an array of indices and returns the corresponding SMILES

        Arguments:
            arr (LongTensor): LongTensor stores the indices of all tokens for one SMILES

        Returns:
            smile (str): decoded SMILES string
        """
        chars = []
        for i in arr.cpu().numpy():
            if i == self.tk2ix['EOS']: break
            chars.append(self.ix2tk[i])
        smile = "".join(chars)
        smile = smile.replace('L', 'Cl').replace('R', 'Br')
        return smile


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing separated SMILES

    Arguments:
        df (str or DataFrame): it is file path of dataset if it is str;
            this data frame contains the column of CANONICAL_SMILES
        voc (Voc): the instance of Voc for SMILES token vocabulary
        token (str, optional): the column name in df for tokens;
            this is for time-saving if the SMILES can be transformed into a series of tokens
            and be saved into table, the "tokenize" step which is quite time-consuming can
            be ignored. (Default: None)
    """
    def __init__(self, df, voc, token=None):
        self.voc = voc
        if isinstance(df, str) and os.path.exists(df):
            df = pd.read_table(df)
        self.smiles = df.CANONICAL_SMILES.values
        self.tokens = []
        if token is None:
            for smile in self.smiles:
                token = self.voc.tokenize(smile)
                if len(token) > self.voc.max_len: continue
                self.tokens.append(token)
        else:
            for sent in df[token].values:
                token = sent.split(' ')
                self.tokens.append(token)

    def __getitem__(self, i):
        # mol = self.smiles[i]
        # tokenized = self.voc.tokenize(mol)
        encoded = self.voc.encode(self.tokens[i])
        return encoded

    def __len__(self):
        return len(self.tokens)

    @classmethod
    def collate_fn(cls, arr, max_len=100):
        """Function to take a list of encoded sequences and turn them into a batch"""
        collated_arr = torch.zeros(len(arr), max_len).long()
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


class QSARData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
    def __init__(self, voc, ligand):
        self.voc = voc
        self.smile = [voc.encode(voc.tokenize(i)) for i in ligand['CANONICAL_SMILES']]
        self.label = torch.Tensor((ligand['PCHEMBL_VALUE'] >= 6.5).values).float()

    def __getitem__(self, i):
        return self.smile[i], self.label[i]

    def __len__(self):
        return len(self.label)

    def collate_fn(self, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_len = max([item[0].size(0) for item in arr])
        smile_arr = torch.zeros(len(arr), max_len).long()
        label_arr = torch.zeros(len(arr), 1)
        for i, data in enumerate(arr):
            smile_arr[i, :data[0].size(0)] = data[0]
            label_arr[i, :] = data[1]
        return smile_arr, label_arr


class Environment:
    """Vitural environment that provided the reward for each molecule
    based on an ECFP predictor for activity.

    Arguments:
        env_path (str): the file path of predictor.
        radius (int): the radius parameter of ECFP
        bit_len (int): the the vector length of ECFP
        is_reg (bool, optional): regresstion (True) or classification (False) model (Default: False)
    """
    def __init__(self, env_path, radius=3, bit_len=4096, is_reg=False):
        self.clf_path = env_path
        self.clf = joblib.load(self.clf_path)
        self.radius = radius
        self.bit_len = bit_len
        self.is_reg = is_reg

    def __call__(self, smiles):
        fps = self.ECFP_from_SMILES(smiles)
        if self.is_reg:
            preds = self.clf.predict(fps)
        else:
            preds = self.clf.predict_proba(fps)[:, 1]
        return preds

    @classmethod
    def ECFP_from_SMILES(cls, smiles, radius=3, bit_len=4096, scaffold=0, index=None):
        fps = np.zeros((len(smiles), bit_len))
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            arr = np.zeros((1,))
            try:
                if scaffold == 1:
                    mol = MurckoScaffold.GetScaffoldForMol(mol)
                elif scaffold == 2:
                    mol = MurckoScaffold.MakeScaffoldGeneric(mol)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps[i, :] = arr
            except:
                print(smile)
                fps[i, :] = [0] * bit_len
        return pd.DataFrame(fps, index=(smiles if index is None else index))


def check_smiles(seqs, voc):
    """Decoding the indices LongTensor into a list of SMILES string
    and checking whether they can be correctly parsed into molecule by RDKit

    Arguments:
        seqs (LongTensor): m X n indices LongTensor, generally it is the output of RNN sampling.
            m is No. of samples; n is the value of max_len in voc
        voc (Voc): the instance of Voc for the SMILES token vocabulary.

    Returns:
        smiles (list): a list of decoded SMILES string.
        valids (ndarray): each value in this array is np.byte type and indicates
            whether the counterpart is grammar correct SMILES or not.
    """
    valids = []
    smiles = []
    for j, seq in enumerate(seqs.cpu()):
        smile = voc.decode(seq)
        valids.append(1 if Chem.MolFromSmiles(smile) else 0)
        smiles.append(smile)
    valids = np.array(valids, dtype=np.byte)
    return smiles, valids


def unique(arr):
    """Removing the duplicated row of indices and only reserving the unique rows for decoding

    Arguments:
        arr (LongTensor): m X n indices LongTensor. Generally it is the output of RNN sampling.
            m is No. of samples; n is the value of max_len in voc

    Returns:
        indices (LongTensor): l X n indices LongTensor without any repetitive rows.
            n is No. of samples; n is the value of max_len in voc
    """
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, indices = np.unique(arr_, return_index=True)
    indices = torch.LongTensor(np.sort(indices)).to(dev)
    return indices
