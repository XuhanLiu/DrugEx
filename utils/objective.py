import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors as desc
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit import DataStructs
from rdkit.Chem.QED import qed
from rdkit.Chem.GraphDescriptors import BertzCT
from utils import sascorer
from .nsgaii import similarity_sort, nsgaii_sort
from .fingerprints import get_fingerprint
from . import modifier
import joblib
import re
from tqdm import tqdm

rdBase.DisableLog('rdApp.error')


class Predictor:
    def __init__(self, path, type='CLS'):
        self.type = type
        self.model = joblib.load(path)

    def __call__(self, fps):
        if self.type == 'CLS':
            scores = self.model.predict_proba(fps)[:, 1]
        else:
            scores = self.model.predict(fps)
        return scores

    @classmethod
    def calc_fp(self, mols, radius=3, bit_len=2048):
        ecfp = self.calc_ecfp(mols, radius=radius, bit_len=bit_len)
        phch = self.calc_physchem(mols)
        fps = np.concatenate([ecfp, phch], axis=1)
        return fps

    @classmethod
    def calc_ecfp(cls, mols, radius=3, bit_len=2048):
        fps = np.zeros((len(mols), bit_len))
        for i, mol in enumerate(mols):
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, fps[i, :])
            except: pass
        return fps

    @classmethod
    def calc_ecfp_rd(cls, mols, radius=3):
        fps = []
        for i, mol in enumerate(mols):
            try:
                fp = AllChem.GetMorganFingerprint(mol, radius)
            except:
                fp = None
            fps.append(fp)
        return fps

    @classmethod
    def calc_physchem(cls, mols):
        prop_list = ['MW', 'logP', 'HBA', 'HBD', 'Rotable', 'Amide',
                     'Bridge', 'Hetero', 'Heavy', 'Spiro', 'FCSP3', 'Ring',
                     'Aliphatic', 'Aromatic', 'Saturated', 'HeteroR', 'TPSA', 'Valence', 'MR']
        fps = np.zeros((len(mols), 19))
        props = Property()
        for i, prop in enumerate(prop_list):
            props.prop = prop
            fps[:, i] = props(mols)
        return fps


class Similarity:
    def __init__(self, smile, fp_type):
        self.mol = Chem.MolFromSmiles(smile)
        self.fp_type = fp_type
        self.fp = get_fingerprint(self.mol, fp_type=fp_type)

    def __call__(self, mols):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(tqdm(mols)):
            try:
                fp = get_fingerprint(mol, fp_type=self.fp_type)
                scores[i] = DataStructs.TanimotoSimilarity(self.fp, fp)
            except: continue
        return scores


class Scaffold:
    def __init__(self, smart, is_match):
        self.frag = Chem.MolFromSmarts(smart)
        self.is_match = is_match

    def __call__(self, mols):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(tqdm(mols)):
            try:
                match = mol.HasSubstructMatch(self.frag)
                scores[i] = (match == self.is_match)
            except: continue
        return scores


class Property:
    def __init__(self, prop='MW'):
        self.prop = prop
        self.prop_dict = {'MW': desc.MolWt,
                          'logP': Crippen.MolLogP,
                          'HBA': AllChem.CalcNumLipinskiHBA,
                          'HBD': AllChem.CalcNumLipinskiHBD,
                          'Rotable': AllChem.CalcNumRotatableBonds,
                          'Amide': AllChem.CalcNumAmideBonds,
                          'Bridge': AllChem.CalcNumBridgeheadAtoms,
                          'Hetero': AllChem.CalcNumHeteroatoms,
                          'Heavy': Lipinski.HeavyAtomCount,
                          'Spiro': AllChem.CalcNumSpiroAtoms,
                          'FCSP3': AllChem.CalcFractionCSP3,
                          'Ring': Lipinski.RingCount,
                          'Aliphatic': AllChem.CalcNumAliphaticRings,
                          'Aromatic': AllChem.CalcNumAromaticRings,
                          'Saturated': AllChem.CalcNumSaturatedRings,
                          'HeteroR': AllChem.CalcNumHeterocycles,
                          'TPSA': AllChem.CalcTPSA,
                          'Valence': desc.NumValenceElectrons,
                          'MR': Crippen.MolMR,
                          'QED': qed,
                          'SA': sascorer.calculateScore,
                          'Bertz': BertzCT}

    def __call__(self, mols):
        scores = np.zeros(len(mols))
        for i, mol in enumerate(mols):
            try:
                scores[i] = self.prop_dict[self.prop](mol)
            except:
                continue
        return scores


class AtomCounter:

    def __init__(self, element: str) -> None:
        """
        Args:
            element: element to count within a molecule
        """
        self.element = element

    def __call__(self, mols):
        """
        Count the number of atoms of a given type.
        Args:
            mol: molecule
        Returns:
            The number of atoms of the given type.
        """
        # if the molecule contains H atoms, they may be implicit, so add them
        scores = np.zeros(len(mols))
        for i, mol in enumerate(mols):
            try:
                if self.element in ['', 'H']:
                    mol = Chem.AddHs(mol)
                if self.element == '':
                    scores[i] = len(mol.GetAtoms())
                else:
                    scores[i] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == self.element)
            except: continue
        return scores


class Isomer:
    """
    Scoring function for closeness to a molecular formula.
    The score penalizes deviations from the required number of atoms for each element type, and for the total
    number of atoms.
    F.i., if the target formula is C2H4, the scoring function is the average of three contributions:
    - number of C atoms with a Gaussian modifier with mu=2, sigma=1
    - number of H atoms with a Gaussian modifier with mu=4, sigma=1
    - total number of atoms with a Gaussian modifier with mu=6, sigma=2
    """

    def __init__(self, formula: str, mean_func='geometric') -> None:
        """
        Args:
            formula: target molecular formula
            mean_func: which function to use for averaging: 'arithmetic' or 'geometric'
        """
        self.objs, self.mods = self.scoring_functions(formula)
        self.mean_func = mean_func

    @staticmethod
    def parse_molecular_formula(formula: str):
        """
        Parse a molecular formulat to get the element types and counts.
        Args:
            formula: molecular formula, f.i. "C8H3F3Br"
        Returns:
            A list of tuples containing element types and number of occurrences.
        """
        matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

        # Convert matches to the required format
        results = []
        for match in matches:
            # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
            count = 1 if not match[1] else int(match[1])
            results.append((match[0], count))

        return results

    def scoring_functions(self, formula: str):
        element_occurrences = self.parse_molecular_formula(formula)

        total_n_atoms = sum(element_tuple[1] for element_tuple in element_occurrences)

        # scoring functions for each element
        objs = [AtomCounter(element) for element, n_atoms in element_occurrences]
        mods = [modifier.Gaussian(mu=n_atoms, sigma=1.0) for element, n_atoms in element_occurrences]
        # scoring functions for the total number of atoms
        objs.append(AtomCounter(''))
        mods.append(modifier.Gaussian(mu=total_n_atoms, sigma=2.0))

        return objs, mods

    def __call__(self, mols: list) -> np.array:
        # return the average of all scoring functions
        score = np.array([self.mods[i](obj(mols)) for i, obj in enumerate(self.objs)])
        scores = score.prod(axis=0) ** (1.0 / len(score)) if self.mean_func == 'geometric' else np.mean(score, axis=0)
        return scores


class Env:
    def __init__(self, objs, mods, keys, ths=None):
        """
        Initialized methods for the construction of environment.
        Args:
            objs (List[Ojective]): a list of objectives.
            mods (List[Modifier]): a list of modifiers, and its length
                equals the size of objs.
            keys (List[str]): a list of strings as the names of objectives,
                and its length equals the size of objs.
            ths (List): a list of float value, and ts length equals the size of objs.
        """
        self.objs = objs
        self.mods = mods if mods is not None else [None] * len(keys)
        self.ths = ths if ths is not None else [0.99] * len(keys)
        self.keys = keys

    def __call__(self, smiles, is_modified=True, frags=None):
        """
        Calculate the scores of all objectives for all of samples
        Args:
            mols (List): a list of molecules
            is_smiles (bool): if True, the type of element in mols should be SMILES sequence, otherwise
                it should be the Chem.Mol
            is_modified (bool): if True, the function of modifiers will work, otherwise
                the modifiers will ignore.

        Returns:
            preds (DataFrame): The scores of all objectives for all of samples which also includes validity
                and desirability for each SMILES.
        """
        preds = {}
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        for i, key in enumerate(self.keys):
            if type(self.objs[i]) == Predictor:
                fps = Predictor.calc_fp(mols)
                score = self.objs[i](fps)
            else:
                score = self.objs[i](mols)
            if is_modified and self.mods[i] is not None:
                score = self.mods[i](score)
            preds[key] = score
        preds = pd.DataFrame(preds)
        undesire = (preds < self.ths)  # ^ self.objs.on
        preds['DESIRE'] = (undesire.sum(axis=1) == 0).astype(int)
        preds['VALID'] = Env.check_smiles(smiles, frags=frags).all(axis=1).astype(int)

        preds[preds.VALID == 0] = 0
        return preds

    @classmethod
    def check_smiles(cls, smiles, frags=None):
        shape = (len(smiles), 1) if frags is None else (len(smiles), 2)
        valids = np.zeros(shape)
        for j, smile in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smile)
                valids[j, 0] = 0 if mol is None else 1
            except:
                valids[j, 0] = 0
            if frags is not None:
                try:
                    subs = frags[j].split('.')
                    subs = [Chem.MolFromSmiles(sub) for sub in subs]
                    valids[j, 1] = np.all([mol.HasSubstructMatch(sub) for sub in subs])
                except:
                    valids[j, 1] = 0
        return valids

    @classmethod
    def calc_fps(cls, mols, fp_type='ECFP6'):
        fps = []
        for i, mol in enumerate(mols):
            try:
                fps.append(get_fingerprint(mol, fp_type))
            except:
                fps.append(None)
        return fps

    def calc_reward(self, smiles, scheme='WS', frags=None):
        """
        Calculate the single value as the reward for each molecule used for reinforcement learning
        Args:
            smiles (List):  a list of SMILES-based molecules
            scheme (str): the label of different rewarding schemes, including
                'WS': weighted sum, 'PR': Pareto ranking with Tanimoto distance,
                and 'CD': Pareto ranking with crowding distance.

        Returns:
            rewards (np.ndarray): n-d array in which the element is the reward for each molecule, and
                n is the number of array which equals to the size of smiles.
        """
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        preds = self(smiles, frags=frags)
        valid = preds.VALID.values
        desire = preds.DESIRE.sum()
        undesire = len(preds) - desire
        preds = preds[self.keys].values

        if scheme == 'PR':
            fps = self.calc_fps(mols)
            rewards = np.zeros((len(smiles), 1))
            ranks = similarity_sort(preds, fps, is_gpu=True)
            score = (np.arange(undesire) / undesire / 2).tolist() + (np.arange(desire) / desire / 2 + 0.5).tolist()
            rewards[ranks, 0] = score
        elif scheme == 'CD':
            rewards = np.zeros((len(smiles), 1))
            ranks = nsgaii_sort(preds, is_gpu=True)
            rewards[ranks, 0] = np.arange(len(preds)) / len(preds)
        else:
            weight = ((preds < self.ths).mean(axis=0, keepdims=True) + 0.01) / \
                     ((preds >= self.ths).mean(axis=0, keepdims=True) + 0.01)
            weight = weight / weight.sum()
            rewards = preds.dot(weight.T)
        rewards[valid == 0] = 0
        return rewards
