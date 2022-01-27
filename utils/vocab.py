import torch
import re
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np
import pandas as pd
import utils


def clean_mol(smile, is_deep=True):
    smile = smile.replace('[O]', 'O').replace('[C]', 'C') \
        .replace('[N]', 'N').replace('[B]', 'B') \
        .replace('[2H]', '[H]').replace('[3H]', '[H]')
    try:
        mol = Chem.MolFromSmiles(smile)
        if is_deep:
            mol = rdMolStandardize.ChargeParent(mol)
        smileR = Chem.MolToSmiles(mol, 0)
        smile = Chem.CanonSmiles(smileR)
    except:
        print('Parsing Error:', smile)
        smile = None
    return smile


class Voc(object):
    def __init__(self, init_from_file=None, src_len=1000, trg_len=100):
        self.control = ('_', 'GO', 'EOS')
        self.words = list(self.control) + ['.']
        self.src_len = src_len
        self.trg_len = trg_len
        if init_from_file: self.init_from_file(init_from_file)
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}

    def split(self, seq, is_smiles=True):
        """Takes a SMILES and return a list of characters/tokens"""
        tokens = []
        if is_smiles:
            regex = '(\[[^\[\]]{1,6}\])'
            seq = re.sub('\[\d+', '[', seq)
            seq = seq.replace('Br', 'R').replace('Cl', 'L')
            for word in re.split(regex, seq):
                if word == '' or word is None: continue
                if word.startswith('['):
                    tokens.append(word)
                else:
                    for i, char in enumerate(word):
                        tokens.append(char)
        else:
            for token in seq:
                token.append('|' + token)
        return tokens

    def encode(self, input, is_smiles=True):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        seq_len = self.trg_len if is_smiles else self.src_len
        output = torch.zeros(len(input), seq_len).long()
        for i, seq in enumerate(input):
            # print(i, len(seq))
            for j, char in enumerate(seq):
                output[i, j] = self.tk2ix[char] if is_smiles else self.tk2ix['|' + char]
        return output

    def decode(self, matrix, is_smiles=True):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            token = self.ix2tk[i.item()]
            if token == 'EOS': break
            if token in self.control: continue
            chars.append(token)
        seqs = "".join(chars)
        if is_smiles:
            seqs = seqs.replace('L', 'Cl').replace('R', 'Br')
        else:
            seqs = seqs.replace('|', '')
        return seqs

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
            assert len(set(chars)) == len(chars)
            self.words += chars


class VocGraph:
    def __init__(self, init_from_file=None, max_len=80, n_frags=4):
        self.control = ('EOS', 'GO')
        self.words = list(self.control)
        self.max_len = max_len
        self.n_frags = n_frags
        self.tk2ix = {'EOS': 0, 'GO': 1}
        self.ix2nr = {0: 0, 1: 0}
        self.ix2ch = {0: 0, 1: 0}
        if init_from_file: self.init_from_file(init_from_file)
        self.size = len(self.words)
        self.E = {0: '', 1: '+', -1: '-'}

    def init_from_file(self, file):
        chars = []
        df = pd.read_table(file)
        self.masks = torch.zeros(len(df) + len(self.control)).long()
        for i, row in df.iterrows():
            self.masks[i + len(self.control)] = row.Val
            ix = i + len(self.control)
            self.tk2ix[row.Word] = ix
            self.ix2nr[ix] = row.Nr
            self.ix2ch[ix] = row.Ch
            chars.append(row.Word)
        assert len(set(chars)) == len(chars)
        self.words += chars

    def get_atom_tk(self, atom):
        sb = atom.GetSymbol() + self.E[atom.GetFormalCharge()]
        val = atom.GetExplicitValence() + atom.GetImplicitValence()
        tk = str(val) + sb
        return self.tk2ix[tk]

    def encode(self, smiles, subs):
        output = np.zeros([len(smiles), self.max_len-self.n_frags-1, 5], dtype=np.long)
        connect = np.zeros([len(smiles), self.n_frags+1, 5], dtype=np.long)
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            sub = Chem.MolFromSmiles(subs[i])
            # Chem.Kekulize(sub)
            sub_idxs = mol.GetSubstructMatches(sub)
            for sub_idx in sub_idxs:
                sub_bond = [mol.GetBondBetweenAtoms(
                    sub_idx[b.GetBeginAtomIdx()],
                    sub_idx[b.GetEndAtomIdx()]).GetIdx() for b in sub.GetBonds()]
                sub_atom = [mol.GetAtomWithIdx(ix) for ix in sub_idx]
                split_bond = {b.GetIdx() for a in sub_atom for b in a.GetBonds() if b.GetIdx() not in sub_bond}
                single = sum([int(mol.GetBondWithIdx(b).GetBondType()) for b in split_bond])
                if single == len(split_bond): break
            frags = Chem.FragmentOnBonds(mol, list(split_bond))

            Chem.MolToSmiles(frags)
            rank = eval(frags.GetProp('_smilesAtomOutputOrder'))
            mol_idx = list(sub_idx) + [idx for idx in rank if idx not in sub_idx and idx < mol.GetNumAtoms()]
            frg_idx = [i+1 for i, f in enumerate(Chem.GetMolFrags(sub)) for _ in f]

            Chem.Kekulize(mol)
            m, n, c = [(self.tk2ix['GO'], 0, 0, 0, 1)], [], [(self.tk2ix['GO'], 0, 0, 0, 0)]
            mol2sub = {ix: i for i, ix in enumerate(mol_idx)}
            for j, idx in enumerate(mol_idx):
                atom = mol.GetAtomWithIdx(idx)
                bonds = sorted(atom.GetBonds(), key=lambda x: mol2sub[x.GetOtherAtomIdx(idx)])
                bonds = [b for b in bonds if j > mol2sub[b.GetOtherAtomIdx(idx)]]
                n_split = sum([1 if b.GetIdx() in split_bond else 0 for b in bonds])
                tk = self.get_atom_tk(atom)
                for k, bond in enumerate(bonds):
                    ix2 = mol2sub[bond.GetOtherAtomIdx(idx)]
                    is_split = bond.GetIdx() in split_bond
                    if idx in sub_idx:
                        is_connect = is_split
                    elif len(bonds) == 1:
                        is_connect = False
                    elif n_split == len(bonds):
                        is_connect = is_split and k != 0
                    else:
                        is_connect = False
                    if bond.GetIdx() in sub_bond:
                        bin, f = m, frg_idx[j]
                    elif is_connect:
                        bin, f = c, 0
                    else:
                        bin, f = n, 0
                    if bond.GetIdx() in sub_bond or not is_connect:
                        tk2 = tk
                        tk = self.tk2ix['*']
                    else:
                        tk2 = self.tk2ix['*']
                    bin.append((tk2, j, ix2, int(bond.GetBondType()), f))
                if tk != self.tk2ix['*']:
                    bin, f = (m, frg_idx[j]) if idx in sub_idx else (n, f)
                    bin.append((tk, j, j, 0, f))
            output[i, :len(m+n), :] = m+n
            if len(c) > 0:
                connect[i, :len(c)] = c
        return np.concatenate([output, connect], axis=1)

    def decode(self, matrix):
        frags, smiles = [], []
        for m, adj in enumerate(matrix):
            # print('decode: ', m)
            emol = Chem.RWMol()
            esub = Chem.RWMol()
            try:
                for atom, curr, prev, bond, frag in adj:
                    atom, curr, prev, bond, frag = int(atom), int(curr), int(prev), int(bond), int(frag)
                    if atom == self.tk2ix['EOS']: continue
                    if atom == self.tk2ix['GO']: continue
                    if atom != self.tk2ix['*']:
                        a = Chem.Atom(self.ix2nr[atom])
                        a.SetFormalCharge(self.ix2ch[atom])
                        emol.AddAtom(a)
                        if frag != 0: esub.AddAtom(a)
                    if bond != 0:
                        b = Chem.BondType(bond)
                        emol.AddBond(curr, prev, b)
                        if frag != 0: esub.AddBond(curr, prev, b)
                Chem.SanitizeMol(emol)
                Chem.SanitizeMol(esub)
            except Exception as e:
                print(adj)
                # raise e
            frags.append(Chem.MolToSmiles(esub))
            smiles.append(Chem.MolToSmiles(emol))
        return frags, smiles


class VocSeq:
    def __init__(self, max_len=1000):
        self.chars = ['_'] + [r for r in utils.AA]
        self.size = len(self.chars)
        self.max_len = max_len
        self.tk2ix = dict(zip(self.chars, range(len(self.chars))))

    def encode(self, seqs):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        output = torch.zeros(len(seqs), self.max_len).long()
        for i, seq in enumerate(seqs):
            for j, res in enumerate(seq):
                if res not in self.chars:
                    res = '_'
                output[i, j] = self.tk2ix[res]
        return output


class VocSmiles:
    """A class for handling encoding/decoding from SMILES to an array of indices"""

    def __init__(self, init_from_file=None, max_len=100):
        self.control = ('_', 'GO', 'EOS')
        self.words = list(self.control) + ['.']
        if init_from_file:
            self.words += self.init_from_file(init_from_file)
        self.size = len(self.words)
        self.tk2ix = dict(zip(self.words, range(len(self.words))))
        self.ix2tk = {v: k for k, v in self.tk2ix.items()}
        self.max_len = max_len

    def encode(self, input):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        output = torch.zeros(len(input), self.max_len).long()
        for i, seq in enumerate(input):
            # print(i, len(seq))
            for j, char in enumerate(seq):
                output[i, j] = self.tk2ix[char]
        return output

    def decode(self, tensor, is_tk=True):
        """Takes an array of indices and returns the corresponding SMILES"""
        tokens = []
        for token in tensor:
            if not is_tk:
                token = self.ix2tk[int(token)]
            if token == 'EOS': break
            if token in self.control: continue
            tokens.append(token)
        smiles = "".join(tokens)
        smiles = smiles.replace('L', 'Cl').replace('R', 'Br')
        return smiles

    def split(self, smile):
        """Takes a SMILES and return a list of characters/tokens"""
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
        return tokens + ['EOS']

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
            smile = clean_mol(smile)
            token = self.split(smile)
            if prefix is not None: token = [prefix] + token
            if len(token) > self.max_len: continue
            if {'C', 'c'}.isdisjoint(token): continue
            if not {'[Na]', '[Zn]'}.isdisjoint(token): continue
            fps[i, :] = self.encode(token)
        return fps

