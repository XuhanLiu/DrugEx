#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script is used for pre-training and fine-tuning the RNN network.

In this project, it is trained on ZINC set and A2AR set collected by
dataset.py. In the end, RNN model can generate molecule library.
"""

import os

import click
import pandas as pd
import torch as T
from torch.utils.data import DataLoader

from drugex import model, util


def _main_helper(*, input_directory, batch_size, output_directory, use_tqdm=False):
    # Construction of the vocabulary
    voc = util.Voc(os.path.join(input_directory, "voc.txt"))

    os.makedirs(output_directory, exist_ok=True)

    net_ex_pickle_path = os.path.join(output_directory, 'net_ex.pkg')
    net_ex_log_path = os.path.join(output_directory, 'net_ex.log')

    net_pr_pickle_path = os.path.join(output_directory, 'net_pr.pkg')
    net_pr_log_path = os.path.join(output_directory, 'net_pr.log')

    # Pre-training the RNN model with ZINC set
    prior = model.Generator(voc)
    if not os.path.exists(net_pr_pickle_path):
        print('Exploitation network begins to be trained...')
        zinc = util.MolData(os.path.join(input_directory, "zinc_corpus.txt"), voc, token='SENT')
        zinc = DataLoader(zinc, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=zinc.collate_fn)
        prior.fit(zinc, out_path=net_pr_pickle_path, log_path=net_pr_log_path, use_tqdm=use_tqdm)
        print('Exploitation network training is finished!')
    # TODO is this necessary if it just got trained?
    prior.load_state_dict(T.load(net_pr_pickle_path))

    # Fine-tuning the RNN model with A2AR set as exploration stragety
    explore = model.Generator(voc)
    df = pd.read_table(os.path.join(input_directory, 'chembl_corpus.txt')).drop_duplicates('CANONICAL_SMILES')
    valid = df.sample(batch_size)
    train = df.drop(valid.index)
    explore.load_state_dict(T.load(net_pr_pickle_path))

    # Training set and its data loader
    train = util.MolData(train, voc, token='SENT')
    train = DataLoader(train, batch_size=batch_size, collate_fn=train.collate_fn)

    # Validation set and its data loader
    valid = util.MolData(valid, voc, token='SENT')
    valid = DataLoader(valid, batch_size=batch_size, collate_fn=valid.collate_fn)

    print('Exploration network begins to be trained...')
    explore.fit(train, loader_valid=valid, out_path=net_ex_pickle_path, n_epochs=1000, log_path=net_ex_log_path)
    print('Exploration network training is finished!')


@click.command()
@click.option('-d', '--input-directory', type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.option('-o', '--output-directory', type=click.Path(dir_okay=True, file_okay=False), required=True)
@click.option('-b', '--batch-size', type=int, default=512, show_default=True)
@click.option('-g', '--cuda')
@click.option('--use-tqdm', is_flag=True)
def main(input_directory, output_directory, batch_size, cuda, use_tqdm):
    if cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    _main_helper(
        input_directory=input_directory,
        output_directory=output_directory,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
    )


if __name__ == "__main__":
    main()
