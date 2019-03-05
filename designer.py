#!/usr/bin/env python
# This script is used for de novo designing the molecule with well-trained RNN model.
import model
import util
import pandas as pd
import torch
import os


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # main('v1/net_e_5_1_500x10.pkg', 'v1/mol_e_5_1_500x10.txt')
    # main('v1/net_e_10_1_500x10.pkg', 'v1/mol_e_10_1_500x10.txt')
    # main('v1/net_e_15_1_500x10.pkg', 'v1/mol_e_15_1_500x10.txt')
    # main('v1/net_e_20_1_500x10.pkg', 'v1/mol_e_20_1_500x10.txt')
    # main('v1/net_e_25_1_500x10.pkg', 'v1/mol_e_25_1_500x10.txt')
    #
    # main('v1/net_e_5_0_500x10.pkg', 'v1/mol_e_5_0_500x10.txt')
    # main('v1/net_e_10_0_500x10.pkg', 'v1/mol_e_10_0_500x10.txt')
    # main('v1/net_e_15_0_500x10.pkg', 'v1/mol_e_15_0_500x10.txt')
    # main('v1/net_e_20_0_500x10.pkg', 'v1/mol_e_20_0_500x10.txt')
    # main('v1/net_e_25_0_500x10.pkg', 'v1/mol_e_25_0_500x10.txt')
    #
    # main('v1/net_a_5_1_500x10.pkg', 'v1/mol_a_5_1_500x10.txt')
    # main('v1/net_a_10_1_500x10.pkg', 'v1/mol_a_10_1_500x10.txt')
    # main('v1/net_a_15_1_500x10.pkg', 'v1/mol_a_15_1_500x10.txt')
    # main('v1/net_a_20_1_500x10.pkg', 'v1/mol_a_20_1_500x10.txt')
    # main('v1/net_a_25_1_500x10.pkg', 'v1/mol_a_25_1_500x10.txt')
    #
    # main('v1/net_a_5_0_500x10.pkg', 'v1/mol_a_5_0_500x10.txt')
    # main('v1/net_a_10_0_500x10.pkg', 'v1/mol_a_10_0_500x10.txt')
    # main('v1/net_a_15_0_500x10.pkg', 'v1/mol_a_15_0_500x10.txt')
    # main('v1/net_a_20_0_500x10.pkg', 'v1/mol_a_20_0_500x10.txt')
    # main('v1/net_a_25_0_500x10.pkg', 'mol_a_25_0_500x10.txt')
    # main('v2/net_REINVENT_ex_ex.pkg', 'v2/mol_REINVENT_ex_ex.pkg')
    generate('v2/net_10_1_1_500x10.pkg', 'v2/mol_10_1_1_500x10.txt')
