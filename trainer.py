#!/usr/bin/env python
import torch as T
from rdkit import rdBase
import numpy as np
import model
import util
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

rdBase.DisableLog('rdApp.error')
T.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE = 500
MC = 10
BL = 0.1
EX = 0.1


def Policy_gradient(netG, netD, explore=None):
    seqs = []
    for _ in range(MC):
        seq = netG.sample(BATCH_SIZE, explore=explore, cutoff=EX)
        seqs.append(seq)
    seqs = T.cat(seqs, dim=0)
    ix = util.unique(seqs)
    seqs = seqs[ix]
    smiles, valids = util.check_smiles(seqs, netG.voc)
    preds = netD(smiles)
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


def Rollout_PG(netG, netD, mc=MC, explore=None):
    netG.optim.zero_grad()
    seqs = netG.sample(BATCH_SIZE, explore=explore, cutoff=EX)
    batch_size = seqs.size(0)
    seq_len = seqs.size(1)
    rewards = np.zeros((batch_size, seq_len))
    smiles, valids = util.check_smiles(seqs, netG.voc)
    preds = netD(smiles) - BL
    preds[valids == False] = -BL
    scores, hiddens = netG.likelihood(seqs)
    for _ in tqdm(range(mc)):
        for i in range(0, seq_len):
            if (seqs[:, i] != 0).any():
                h = hiddens[:, :, i, :]
                subseqs = netG.sample(batch_size, inits=(seqs[:, i], h, i+1, None))
                subseqs = T.cat([seqs[:, :i+1], subseqs], dim=1)
                subsmile, subvalid = util.check_smiles(subseqs, voc=netG.voc)
                subpred = netD(subsmile) - BL
                subpred[1 - subvalid] = -BL
            else:
                subpred = preds
            rewards[:, i] += subpred
    loss = netG.PGLoss(scores, seqs, T.FloatTensor(rewards / mc))
    loss.backward()
    netG.optim.step()
    return 0, valids.mean(), smiles, preds


def main():
    voc = util.Voc(init_from_file="data/voc_b.txt")
    netD_path = 'output/rf_dis.pkg'
    netG_path = 'output/net_p'
    agent_path = 'output/net_e_%d_%d_%dx%d' % (EX * 100, BL * 10, BATCH_SIZE, MC)
    netE_path = 'output/net_ex'
    netD = util.Activity(netD_path)
    agent = model.Generator(voc)

    agent.load_state_dict(T.load(netG_path + '.pkg'))

    explore = model.Generator(voc)

    explore.load_state_dict(T.load(netE_path + '.pkg'))

    best_score = 0
    log = open(agent_path + '.log', 'w')

    for epoch in range(1000):
        print('\n--------\nEPOCH %d\n--------' % (epoch + 1))
        print('\nForward Policy Gradient Training Generator : ')
        Policy_gradient(agent, netD, explore=explore)
        seqs = agent.sample(1000)
        ix = util.unique(seqs)
        smiles, valids = util.check_smiles(seqs[ix], agent.voc)
        scores = netD(smiles)
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
