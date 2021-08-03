#!/usr/bin/env python
import torch
import models
import utils
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import getopt
import sys


def training(is_lstm=True):
    voc = utils.Voc(init_from_file="data/voc.txt")
    if is_lstm:
        netP_path = 'output/lstm_chembl'
        netE_path = 'output/lstm_ligand'
    else:
        netP_path = 'output/gru_chembl'
        netE_path = 'output/gru_ligand'

    prior = models.Generator(voc, is_lstm=is_lstm)
    if not os.path.exists(netP_path + '.pkg'):
        chembl = pd.read_table("data/chembl_corpus.txt").Token
        chembl = torch.LongTensor(voc.encode([seq.split(' ') for seq in chembl]))
        chembl = DataLoader(chembl, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        prior.fit(chembl, out=netP_path, epochs=50)
    prior.load_state_dict(torch.load(netP_path + '.pkg'))

    # explore = model.Generator(voc)
    df = pd.read_table('data/ligand_corpus.txt').drop_duplicates('Smiles')
    valid = df.sample(len(df) // 10).Token
    train = df.drop(valid.index).Token
    # explore.load_state_dict(torch.load(netP_path + '.pkg'))

    train = torch.LongTensor(voc.encode([seq.split(' ') for seq in train]))
    train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    valid = torch.LongTensor(voc.encode([seq.split(' ') for seq in valid]))
    valid = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)
    print('Fine tuning progress begins to be trained...')

    prior.fit(train, loader_valid=valid, out=netE_path, epochs=1000, lr=lr)
    print('Fine tuning progress training is finished...')


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "g:m:")
    OPT = dict(opts)
    lr = 1e-4
    torch.set_num_threads(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if '-g' not in OPT else OPT['-g']
    BATCH_SIZE = 512
    is_lstm = opts['-m'] if '-m' in OPT else True
    training(is_lstm=is_lstm)
