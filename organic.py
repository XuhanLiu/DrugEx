#!/usr/bin/env python
import torch as T
from rdkit import rdBase
import numpy as np
import model
import util
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


rdBase.DisableLog('rdApp.error')
T.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE = 500
SIGMA = 0.5
MC = 1
BL = 0

VOCAB_SIZE = 58
EMBED_DIM = 128
FILTER_SIZE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
NUM_FILTER = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]


def Train_GAN(netG, netD, netR, sigma=SIGMA):
    seqs = []
    for _ in range(MC):
        seq = netG.sample(BATCH_SIZE)
        seqs.append(seq)
    seqs = T.cat(seqs, dim=0)
    ix = util.unique(seqs)
    seqs = seqs[ix]
    smiles, valids = util.check_smiles(seqs, netG.voc)
    preds = sigma * netR(smiles) + (1 - sigma) * netD(util.Variable(seqs)).data.cpu().numpy()[:, 0]
    preds[valids == False] = 0
    preds -= BL
    ds = TensorDataset(seqs, T.Tensor(preds.reshape(-1, 1)))
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    for seq, pred in loader:
        score, _ = netG.likelihood(seq)
        netG.optim.zero_grad()
        loss = netG.PGLoss(score, seq, pred)
        loss.backward()
        netG.optim.step()


def Train_dis_BCE(netD, netG, real_loader, epochs=1, out=None):
    best_loss = np.Inf
    for _ in range(epochs):
        for i, real in enumerate(real_loader):
            size = len(real)
            fake = netG.sample(size)
            data = util.Variable(T.cat([fake, util.cuda(real)], dim=0))
            label = util.Variable(T.cat([T.zeros(size, 1), T.ones(size, 1)]))
            netD.optim.zero_grad()
            loss = netD.BCELoss(data, label)
            loss.backward()
            netD.optim.step()
            if i % 10 == 0 and i != 0:
                for param_group in netD.optim.param_groups:
                    param_group['lr'] *= (1 - 0.03)
        if out and loss.data[0] < best_loss:
            T.save(netD.state_dict(), out + ".pkg")
            best_loss = loss.data[0]
    return loss.data[0]


def main():
    voc = util.Voc(init_from_file="data/voc_b.txt")
    netR_path = 'output/rf_dis.pkg'
    netG_path = 'output/net_p'
    netD_path = 'output/net_d'
    agent_path = 'output/net_gan_%d_%d_%dx%d' % (SIGMA * 10, BL * 10, BATCH_SIZE, MC)

    netR = util.Environment(netR_path)

    agent = model.Generator(voc)
    agent.load_state_dict(T.load(netG_path + '.pkg'))

    df = pd.read_table('data/CHEMBL251.txt')
    df = df[df['PCHEMBL_VALUE'] >= 6.5]
    data = util.MolData(df, voc)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=data.collate_fn)

    netD = model.Discriminator(VOCAB_SIZE, EMBED_DIM, FILTER_SIZE, NUM_FILTER)
    if not os.path.exists(netD_path + '.pkg'):
        Train_dis_BCE(netD, agent, loader, epochs=100, out=netD_path)
    netD.load_state_dict(T.load(netD_path + '.pkg'))

    best_score = 0
    log = open(agent_path + '.log', 'w')
    for epoch in range(1000):
        print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
        print('\nPolicy Gradient Training Generator : ')
        Train_GAN(agent, netD, netR)

        print('\nAdversarial Training Discriminator : ')
        Train_dis_BCE(netD, agent, loader, epochs=1)

        seqs = agent.sample(1000)
        ix = util.unique(seqs)
        smiles, valids = util.check_smiles(seqs[ix], agent.voc)
        scores = netR(smiles)
        scores[valids == False] = 0
        unique = (scores >= 0.5).sum() / 1000
        if best_score < unique:
            T.save(agent.state_dict(), agent_path + '.pkg')
            best_score = unique
        print("Epoch+: %d average: %.4f valid: %.4f unique: %.4f" % (epoch, scores.mean(), valids.mean(), unique), file=log)
        for i, smile in enumerate(smiles):
            print('%f\t%s' % (scores[i], smile), file=log)

        for param_group in agent.optim.param_groups:
            param_group['lr'] *= (1 - 0.01)

    log.close()


if __name__ == "__main__":
    main()
