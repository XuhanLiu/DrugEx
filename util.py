import torch as T
from torch.utils.data import Dataset
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.preprocessing import MinMaxScaler
import os
from sklearn.externals import joblib


class Voc(object):
    def __init__(self, init_from_file=None, max_len=100):
        self.chars = ['EOS', 'GO']
        if init_from_file: self.init_from_file(init_from_file)
        self.size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len

    def tokenize(self, smile):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smile = re.sub('\[\d+', '[', smile)
        smile = replace_halogen(Chem.CanonSmiles(smile, 0))
        tokens = []
        for word in re.split(regex, smile):
            if word == '' or word is None: continue
            if word.startswith('['):
                tokens.append(word)
            else:
                for i, char in enumerate(word):
                    tokens.append(char)
        tokens.append('EOS')
        return tokens

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = T.zeros(len(char_list))
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace('L', 'Cl').replace('R', 'Br')
        return smiles

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
            assert len(set(chars)) == len(chars)
            self.chars += chars


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""
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
        # max_length = max([seq.size(0) for seq in arr])
        collated_arr = T.zeros(len(arr), max_len).long()
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


class QSARData(Dataset):
    """Custom PyTorch Dataset that takes a file containing \n separated SMILES"""

    def __init__(self, voc, ligand):
        self.voc = voc
        self.smile = [voc.encode(voc.tokenize(i)) for i in ligand['CANONICAL_SMILES']]
        self.label = T.Tensor((ligand['PCHEMBL_VALUE'] >= 6.5).astype(float).values)

    def __getitem__(self, i):
        return self.smile[i], self.label[i]

    def __len__(self):
        return len(self.label)

    def collate_fn(self, arr):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_len = max([item[0].size(0) for item in arr])
        smile_arr = T.zeros(len(arr), max_len).long()
        label_arr = T.zeros(len(arr), 1)
        for i, data in enumerate(arr):
            smile_arr[i, :data[0].size(0)] = data[0]
            label_arr[i, :] = data[1]
        return smile_arr, label_arr


def replace_halogen(smile):
    """Regex to replace Br and Cl with single letters"""
    smile = smile.replace('Br', 'R').replace('Cl', 'L')
    return smile


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = T.from_numpy(tensor)
    if isinstance(tensor, list):
        tensor = T.Tensor(tensor)
    return cuda(T.autograd.Variable(tensor))


def cuda(var):
    if T.cuda.is_available():
        return var.cuda()
    return var


class Activity:
    """Scores based on an ECFP classifier for activity."""
    def __init__(self, clf_path, radius=3, bit_len=4096):
        self.clf_path = clf_path
        self.clf = joblib.load(self.clf_path)
        self.radius = radius
        self.bit_len = bit_len

    def __call__(self, smiles):
        fps = self.ECFP_from_SMILES(smiles)
        preds = self.clf.predict_proba(fps)[:, 1]
        return preds

    @classmethod
    def ECFP_from_SMILES(cls, smiles, radius=3, bit_len=4096):
        fps = np.zeros((len(smiles), bit_len))
        for i, smile in enumerate(smiles):
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                fps[i, :] = [0] * bit_len
            else:
                arr = np.zeros((1,))
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps[i, :] = arr
        return fps


def check_smiles(seqs, voc):
    valids = []
    smiles = []
    for j, seq in enumerate(seqs.cpu()):
        smile = voc.decode(seq)
        valids.append(1 if Chem.MolFromSmiles(smile) else 0)
        smiles.append(smile)
    return smiles, np.array(valids, dtype=np.byte)


def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    return cuda(T.LongTensor(np.sort(idxs)))