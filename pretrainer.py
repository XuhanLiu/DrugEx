#!/usr/bin/env python
import torch as T
from rdkit import rdBase
import numpy as np
import model
import util
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.externals import joblib
from tqdm import tqdm

rdBase.DisableLog('rdApp.error')
T.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE = 500


def Pretrain(netG, loader, epochs=10, out=None):
    log = open(out + '.log', 'w')
    best_valid = 0.
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            netG.optim.zero_grad()
            loss, _ = netG.likelihood(batch)
            loss = -loss.mean()
            loss.backward()
            netG.optim.step()
            if i % 2 == 0 and i != 0:
                seqs = netG.sample(1000)
                ix = util.unique(seqs)
                seqs = seqs[ix]
                smiles, valids = util.check_smiles(seqs, netG.voc)
                valid = sum(valids) / 1000
                print("Epoch: %d step: %d loss: %.3f valid: %.3f" % (epoch, i, loss.data[0], valid), file=log)
                for i, smile in enumerate(smiles):
                    print('%d\t%s' % (valids[i], smile), file=log)
                if best_valid < valid:
                    T.save(netG.state_dict(), out + '.pkg')
                    best_valid = valid
    log.close()


def Fine_tune(netG, prior, df, epochs=100, out=None):
    dataset = util.MolData(netG.voc, df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)
    log = open(out + '.log', 'w')
    best_valid = 0.
    for epoch in range(epochs):
        for i, batch in enumerate(loader):
            netG.optim.zero_grad()
            loss_p, _ = prior.likelihood(batch)
            loss_a, _ = netG.likelihood(batch)
            loss = T.pow((loss_p + 6 - loss_a), 2).mean()
            loss -= 5 * 1e3 * (1 / loss_a).mean()
            loss.backward()
            netG.optim.step()
            if i % 2 == 0 and i != 0:
                seqs = netG.sample(BATCH_SIZE * 2)
                ix = util.unique(seqs)
                seqs = seqs[ix]
                smiles, valids = util.check_smiles(seqs, netG.voc)
                valid = sum(valids) / BATCH_SIZE / 2
                print("Epoch: %d step: %d loss: %.3f valid: %.3f" % (epoch, i, loss.data[0], valid), file=log)
                for i, smile in enumerate(smiles):
                    print('%d\t%s' % (valids[i], smile), file=log)
                if best_valid < valid:
                    T.save(netG.state_dict(), out + '.pkg')
                    best_valid = valid
    log.close()


def main():
    voc = util.Voc(init_from_file="data/voc_b.txt")
    netP_path = 'output/net_p'
    netE_path = 'output/net_ex'

    prior = model.Generator(voc)
    if not os.path.exists(netP_path + '.pkg'):
        zinc = util.MolData("data/zinc_b_corpus.txt", voc, token='SENT')
        zinc = DataLoader(zinc, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=zinc.collate_fn)
        Pretrain(prior, zinc, out=netP_path)
    prior.load_state_dict(T.load(netP_path + '.pkg'))
    prior.reset_optim()

    explore = model.Generator(voc)
    df = pd.read_table('data/CHEMBL251.txt')
    explore.load_state_dict(T.load(netP_path + '.pkg'))

    a2ar = util.MolData(df, voc)
    a2ar = DataLoader(a2ar, batch_size=BATCH_SIZE, drop_last=True, collate_fn=a2ar.collate_fn)
    print('Exploration stragety begins to be trained...')
    Pretrain(explore, a2ar, out=netP_path)
    Fine_tune(explore, prior, out=netE_path)
    print('Exploration stragety training is finished...')

if __name__ == "__main__":
    main()