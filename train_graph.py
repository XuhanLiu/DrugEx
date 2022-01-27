#!/usr/bin/env python
import torch
from rdkit import rdBase
from models.explorer import GraphExplorer
import utils
import pandas as pd
from models import GraphModel
from torch.utils.data import DataLoader
import getopt
import sys
import os
import numpy as np
import time
from shutil import copy2

np.random.seed(2)
torch.manual_seed(2)
rdBase.DisableLog('rdApp.error')
torch.set_num_threads(1)


def pretrain():
    out = 'output/%s_graph_%d' % (dataset, BATCH_SIZE)
    agent.fit(valid_loader, valid_loader, epochs=1000, out=out)


def train_ex():
    agent.load_state_dict(torch.load(params['pr_path'] + '.pkg', map_location=utils.dev))

    prior = GraphModel(voc)
    prior.load_state_dict(torch.load(params['ft_path'] + '.pkg', map_location=utils.dev))

    evolver = GraphExplorer(agent, mutate=prior)

    evolver.batch_size = BATCH_SIZE
    evolver.epsilon = float(OPT.get('-e', '1e-2'))
    evolver.sigma = float(OPT.get('-b', '0.00'))
    evolver.scheme = OPT.get('-s', 'WS')
    evolver.repeat = 1

    keys = ['A2A', 'QED']
    A2A = utils.Predictor('output/env/RF_%s_CHEMBL251.pkg' % z, type=z)
    QED = utils.Property('QED')

    # Chose the desirability function
    objs = [A2A, QED]

    if evolver.scheme == 'WS':
        mod1 = utils.ClippedScore(lower_x=3, upper_x=10)
        mod2 = utils.ClippedScore(lower_x=0, upper_x=1.0)
        ths = [0.5, 0]
    else:
        mod1 = utils.ClippedScore(lower_x=3, upper_x=6.5)
        mod2 = utils.ClippedScore(lower_x=0, upper_x=1.0)
        ths = [0.99, 0]
    mods = [mod1, mod2]
    evolver.env = utils.Env(objs=objs, mods=mods, keys=keys, ths=ths)

    # import evolve as agent
    evolver.out = root + '/%s_%s_%.0e' % (alg, evolver.scheme, evolver.epsilon)
    evolver.fit(train_loader, test_loader=valid_loader)


if __name__ == "__main__":
    params = {'pr_path': 'output/ligand_mf_brics_graph_256', 'ft_path': 'output/ligand_mf_brics_graph_256'}
    opts, args = getopt.getopt(sys.argv[1:], "a:e:b:d:g:s:")
    OPT = dict(opts)
    z = OPT.get('-z', 'REG')
    alg = OPT.get('-a', 'graph')
    devs = OPT.get('-g', "0")
    utils.devices = eval(devs) if ',' in devs else [eval(devs)]
    torch.cuda.set_device(utils.devices[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = devs

    BATCH_SIZE = int(OPT.get('-b', '128'))
    dataset = OPT.get('-d', 'ligand_mf_brics')

    voc = utils.VocGraph('data/voc_atom.txt', max_len=80, n_frags=4)
    data = pd.read_table('data/%s_train_code.txt' % dataset)
    data = torch.from_numpy(data.values).long().view(len(data), voc.max_len, -1)
    train_loader = DataLoader(data, batch_size=BATCH_SIZE * 4, drop_last=True, shuffle=True)

    test = pd.read_table('data/%s_test_code.txt' % dataset)
    # test = test.sample(int(1e4))
    test = torch.from_numpy(test.values).long().view(len(test), voc.max_len, -1)
    valid_loader = DataLoader(test, batch_size=BATCH_SIZE * 10, drop_last=True, shuffle=True)

    agent = GraphModel(voc).to(utils.dev)
    root = 'output/%s_%s' % (alg, time.strftime('%y%m%d_%H%M%S', time.localtime()))

    os.mkdir(root)
    copy2(alg + '_ex.py', root)
    copy2(alg + '.py', root)

    pretrain()
    train_ex()
