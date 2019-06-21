#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script is used for *de novo* designing the molecule with well-trained RNN model."""

import os

import click
import pandas as pd
import torch

from drugex import model, util
from drugex.util import Voc


def generate(agent_path, out, *, voc: Voc, num=10000, batch_size=500, environment_path='output/RF_cls_ecfp6.pkg'):
    """Generate novel molecules with SMILES representation and store them into hard drive as a data frame.

    Arguments:
        agent_path (str): the neural states file paths for the RNN agent (generator).
        out (str): file path for the generated molecules (and scores given by environment).
        num (int, optional): the total No. of SMILES that need to be generated. (Default: 10000)
        environment_path (str, optional): the file path of the predictor for environment construction.
    """
    df = pd.DataFrame()
    agent = model.Generator(voc)
    agent.load_state_dict(torch.load(agent_path))
    for i in range(num // batch_size + 1):
        if i == 0 and num % batch_size == 0: continue
        batch = pd.DataFrame()
        samples = agent.sample(batch_size if i != 0 else num % batch_size)
        smiles, valids = util.check_smiles(samples, agent.voc)
        if environment_path is not None:
            # calculating the reward of each SMILES based on the environment (predictor).
            environ = util.Environment(environment_path)
            scores = environ(smiles)
            scores[valids == 0] = 0
            valids = scores
            batch['SCORE'] = valids
        batch['CANONICAL_SMILES'] = smiles
        df = df.append(batch)
    df.to_csv(out, sep='\t', index=None)


@click.command()
@click.option('--agent-path', required=True)
@click.option('--vocabulary-path', required=True)
@click.option('--output', type=click.File('w'), required=True)
@click.option('--environment-path')
@click.option('-n', '--number-smiles', type=int, default=10000, show_default=True)
@click.option('--batch-size', type=int, default=500, show_default=True)
@click.option('--cuda-visible-devices')
def main(agent_path, vocabulary_path, output, environment_path, number_smiles, batch_size, cuda_visible_devices):
    if cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    voc = Voc(vocabulary_path)
    generate(
        agent_path=agent_path,
        environment_path=environment_path,
        out=output,
        voc=voc,
        batch_size=batch_size,
        num=number_smiles,
    )

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


if __name__ == '__main__':
    main()
