import models
import utils
import pandas as pd
import torch
import os
from tqdm import tqdm
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sampling(netG_path, out, size=10000):
    """
    sampling a series of tokens squentially for molecule generation
    Args:
        netG_path (str): The file path of generator.
        out (str): The file path of genrated molecules including SMILES, and its scores for each sample
        size (int): The number of molecules required to be generated.
        env (utils.Environment): The environment to provide the scores of all objectives for each sample

    Returns:
        smiles (List): A list of generated SMILES-based molecules
    """
    batch_size = 250
    samples = []
    voc = utils.Voc(init_from_file="data/voc.txt")
    netG = models.Generator(voc)
    netG.load_state_dict(torch.load(netG_path))
    batch  = size // batch_size
    mod = size % batch_size
    for i in tqdm(range(batch + 1)):
        if i == 0:
            if mod == 0: continue
            tokens = netG.sample(batch)
        else:
            tokens = netG.sample(batch_size)
        smiles = [voc.decode(s) for s in tokens]
        samples.extend(smiles)
    return samples


if __name__ == '__main__':
    for z in ['REG']:
        # Construct the environment with three predictors and desirability functions
        keys = ['A1', 'A2A', 'ERG']
        A1 = utils.Predictor('output/env/RF_%s_CHEMBL226.pkg' % z, type=z)
        A2A = utils.Predictor('output/env/RF_%s_CHEMBL251.pkg' % z, type=z)
        ERG = utils.Predictor('output/env/RF_%s_CHEMBL240.pkg' % z, type=z)
        mod1 = utils.ClippedScore(lower_x=4, upper_x=6.5)
        mod2 = utils.ClippedScore(lower_x=9, upper_x=6.5)
        mod3 = utils.ClippedScore(lower_x=7.5, upper_x=5)
        objs = [A1, A2A, ERG]

        models = {'output/lstm_ligand.pkg': 'benchmark/FINE-TUNE_%s_%s.tsv' % (z, case),
                  'output/lstm_chembl.pkg': 'benchmark/PRE-TRAIN_%s_%s.tsv' % (z, case)}
        for case in ['OBJ1', 'OBJ3']:
            if case == 'OBJ3':
                mods = [mod1, mod1, mod3]
            else:
                mods = [mod2, mod1, mod3]

            env = utils.Env(objs=objs, mods=mods, keys=keys)

            for input, output in models.items():
                df = pd.DataFrame()
                df['Smiles'] = sampling(input, output)
                scores = env(df['Smiles'], is_smiles=True)
                df.to_csv(output, index=False, sep='\t')