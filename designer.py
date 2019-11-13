#!/usr/bin/env python
# This script is used for de novo designing the molecule with well-trained RNN model.
import model
import util
import pandas as pd
import torch
import os
import getopt
import sys


def generate(agent_path, out, num=10000, environ_path='output/RF_cls_ecfp6.pkg'):
    """ Generating novel molecules with SMILES representation and
    storing them into hard drive as a data frame.

    Arguments:
        agent_path (str): the neural states file paths for the RNN agent (generator).
        out (str): file path for the generated molecules (and scores given by environment).
        num (int, optional): the total No. of SMILES that need to be generated. (Default: 10000)
        environ_path (str): the file path of the predictor for environment construction.
    """
    batch_size = 500
    df = pd.DataFrame()
    voc = util.Voc("data/voc.txt")
    agent = model.Generator(voc)
    agent.load_state_dict(torch.load(agent_path))
    for i in range(num // batch_size + 1):
        if i == 0 and num % batch_size == 0: continue
        batch = pd.DataFrame()
        samples = agent.sample(batch_size if i != 0 else num % batch_size)
        smiles, valids = util.check_smiles(samples, agent.voc)
        if environ_path is not None:
            # calculating the reward of each SMILES based on the environment (predictor).
            environ = util.Environment(environ_path)
            scores = environ(smiles)
            scores[valids == 0] = 0
            valids = scores
            batch['SCORE'] = valids
        batch['CANONICAL_SMILES'] = smiles
        df = df.append(batch)
    df.to_csv(out, sep='\t', index=None)


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "i:b:g:n:")
    OPT = dict(opts)
    os.environ["CUDA_VISIBLE_DEVICES"] = OPT['-g'] if '-g' in OPT else "0"
    agent_path = OPT['-i'] if '-i' in OPT else 'data/agent.pkg' # FIXME: at this point this default file does not exist
    out_path = OPT['-o'] if '-o' in OPT else 'output/designer_mols.txt'
    pop_size = int(OPT['-n']) if '-n' in OPT else 10000
    batch_size = int(OPT['-b']) if '-b' in OPT else 500
    generate(agent_path, out_path, num=pop_size)
