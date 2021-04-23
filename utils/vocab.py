import torch
import re
from rdkit import Chem
from rdkit import rdBase
import numpy as np
from typing import List, Iterable, Optional

torch.set_num_threads(1)
rdBase.DisableLog('rdApp.error')
dev = torch.device('cuda')


def canonicalize(smiles: str, include_stereocenters=True) -> Optional[str]:
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return ''


def canonicalize_list(smiles_list: Iterable[str], include_stereocenters=True) -> List[str]:
    """
    Canonicalize a list of smiles. Filters out repetitions and removes corrupted molecules.

    Args:
        smiles_list: molecules as SMILES strings
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES strings

    Returns:
        The canonicalized and filtered input smiles.
    """

    canonicalized_smiles = [canonicalize(smiles, include_stereocenters) for smiles in smiles_list]

    return canonicalized_smiles


class Voc:
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    def __init__(self, init_from_file=None, max_len=100):
        """
        Args:
            init_from_file: the file path of vocabulary containing all of tokens split by '\n'
            max_len: the maximum number of tokens contained in one SMILES
        """
        self.control = ['EOS', 'GO']
        self.words = [] + self.control
        if init_from_file:
            self.words += self.init_from_file(init_from_file)
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}
        self.max_len = max_len

    def encode(self, smiles: List):
        """
        Takes a list of tokens (eg '[NH]') and encodes to array of indices
        Args:
            smiles: a list of SMILES squence represented as a series of tokens

        Returns:
            tokens (torch.LongTensor): a long tensor containing all of the indices of given tokens.
        """
        tokens = torch.zeros(len(smiles), self.max_len).long()
        for i, smile in enumerate(smiles):
            for j, char in enumerate(smile):
                tokens[i, j] = self.tk2ix[char]
        return tokens

    def decode(self, tensor):
        """Takes an array of indices and returns the corresponding SMILES
        Args:
            tensor(torch.LongTensor): a long tensor containing all of the indices of given tokens.

        Returns:
            smiles (str): a decoded smiles sequence.
        """
        tokens = []
        for i in tensor:
            token = self.ix2tk[i.item()]
            if token == 'EOS': break
            if token in self.control: continue
            tokens.append(token)
        smiles = "".join(tokens)
        smiles = smiles.replace('L', 'Cl').replace('R', 'Br')
        return smiles

    def tokenize(self, smile):
        """Takes a SMILES and return a list of characters/tokens
        Args:
            smiles (str): a decoded smiles sequence.

        Returns:
            tokens (List): a list of tokens decoded from the SMILES sequence.
        """
        regex = '(\[[^\[\]]{1,6}\])'
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

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        words = []
        with open(file, 'r') as f:
            chars = f.read().split()
            words += sorted(set(chars))
        return words

    def calc_voc_fp(self, smiles, prefix=None):
        fps = np.zeros((len(smiles), self.max_len), dtype=np.long)
        for i, smile in enumerate(smiles):
            token = self.tokenize(smile)
            if prefix is not None: token = [prefix] + token
            if len(token) > self.max_len: continue
            fps[i, :] = self.encode(token)
        return fps

    def check_smiles(self, seqs):
        """

        Args:
            seqs (Iterable): a batch of token indices.

        Returns:
            smiles (List): a list of decoded SMILES
            valids (List): if the decoded SMILES is valid or not
        """
        smiles = [self.decode(s) for s in seqs]
        valids = [1 if Chem.MolFromSmiles(smile) else 0 for smile in smiles]
        return smiles, valids

