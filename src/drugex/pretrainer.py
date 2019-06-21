#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script is used for pre-training and fine-tuning the RNN network.

In this project, it is trained on ZINC set and A2AR set collected by
dataset.py. In the end, RNN model can generate molecule library.
"""

import getopt
import os
import sys

import pandas as pd
import torch as T
from torch.utils.data import DataLoader

from drugex import model, util


def main(batch_size):
    # Construction of the vocabulary
    voc = util.Voc("data/voc.txt")
    netP_path = 'output/net_pr'
    netE_path = 'output/net_ex'

    # Pre-training the RNN model with ZINC set
    prior = model.Generator(voc)
    if not os.path.exists(netP_path + '.pkg'):
        print('Exploitation network begins to be trained...')
        zinc = util.MolData("data/zinc_corpus.txt", voc, token='SENT')
        zinc = DataLoader(zinc, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=zinc.collate_fn)
        prior.fit(zinc, out=netP_path)
        print('Exploitation network training is finished!')
    prior.load_state_dict(T.load(netP_path + '.pkg'))

    # Fine-tuning the RNN model with A2AR set as exploration stragety
    explore = model.Generator(voc)
    df = pd.read_table('data/chembl_corpus.txt').drop_duplicates('CANONICAL_SMILES')
    valid = df.sample(batch_size)
    train = df.drop(valid.index)
    explore.load_state_dict(T.load(netP_path + '.pkg'))

    # Training set and its data loader
    train = util.MolData(train, voc, token='SENT')
    train = DataLoader(train, batch_size=batch_size, collate_fn=train.collate_fn)

    # Validation set and its data loader
    valid = util.MolData(valid, voc, token='SENT')
    valid = DataLoader(valid, batch_size=batch_size, collate_fn=valid.collate_fn)

    print('Exploration network begins to be trained...')
    explore.fit(train, loader_valid=valid, out=netE_path, epochs=1000)
    print('Exploration network training is finished!')


def more_main():
    opts, args = getopt.getopt(sys.argv[1:], "e:b:g:")
    OPT = dict(opts)
    batch_size = int(OPT['-b']) if '-b' in OPT else 512
    if '-g' in OPT:
        os.environ["CUDA_VISIBLE_DEVICES"] = OPT['-g']
    main(batch_size)


if __name__ == "__main__":
    more_main()
