#!/usr/bin/env python
import os
import pandas as pd
from shutil import copy2
import utils
import getopt
import sys
import time
import torch
from models import GPT2Model
from models import generator
from models.explorer import SmilesExplorer
from torch.utils.data import DataLoader, TensorDataset


def pretrain(method='gpt'):
    if method == 'ved':
        agent = generator.EncDec(voc, voc).to(utils.dev)
    elif method == 'attn':
        agent = generator.Seq2Seq(voc, voc).to(utils.dev)
    else:
        agent = GPT2Model(voc, n_layer=12).to(utils.dev)

    out = 'output/%s_%s_%d' % (dataset, method, BATCH_SIZE)
    agent.fit(data_loader, test_loader, epochs=1000, out=out)


def rl_train():
    opts, args = getopt.getopt(sys.argv[1:], "a:e:b:g:c:s:z:")
    OPT = dict(opts)
    case = OPT['-c'] if '-c' in OPT else 'OBJ1'
    z = OPT['-z'] if '-z' in OPT else 'REG'
    alg = OPT['-a'] if '-a' in OPT else 'smile'
    os.environ["CUDA_VISIBLE_DEVICES"] = OPT['-g'] if '-g' in OPT else "0,1,2,3"

    voc = utils.VocSmiles(init_from_file="data/voc_smiles.txt", max_len=100)
    agent = GPT2Model(voc, n_layer=12)
    agent.load_state_dict(torch.load(params['pr_path'] + '.pkg', map_location=utils.dev))

    prior = GPT2Model(voc, n_layer=12)
    prior.load_state_dict(torch.load(params['ft_path'] + '.pkg', map_location=utils.dev))

    evolver = SmilesExplorer(agent, mutate=prior)

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
        mod2 = utils.ClippedScore(lower_x=0, upper_x=1)
        ths = [0.5, 0]
    else:
        mod1 = utils.ClippedScore(lower_x=3, upper_x=6.5)
        mod2 = utils.ClippedScore(lower_x=0, upper_x=0.5)
        ths = [0.99, 0]
    mods = [mod1, mod2] if case == 'OBJ3' else [mod1, mod2]
    evolver.env = utils.Env(objs=objs, mods=mods, keys=keys, ths=ths)

    root = 'output/%s_%s' % (alg, time.strftime('%y%m%d_%H%M%S', time.localtime()))

    os.mkdir(root)
    copy2(alg + '_ex.py', root)
    copy2(alg + '.py', root)

    # import evolve as agent
    evolver.out = root + '/%s_%s_%s_%s_%.0e' % (alg, evolver.scheme, z, case, evolver.epsilon)
    evolver.fit(data_loader, test_loader=test_loader)


if __name__ == "__main__":
    params = {'pr_path': 'output/ligand_mf_brics_gpt_256', 'ft_path': 'output/ligand_mf_brics_gpt_256'}
    opts, args = getopt.getopt(sys.argv[1:], "m:g:b:d:")
    OPT = dict(opts)
    torch.cuda.set_device(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = OPT.get('-g', "0,1,2,3")
    method = OPT.get('-m', 'gpt')
    step = OPT['-s']
    BATCH_SIZE = int(OPT.get('-b', '256'))
    dataset = OPT.get('-d', 'ligand_mf_brics')

    data = pd.read_table('data/%s_train_smi.txt' % dataset)
    test = pd.read_table('data/%s_test_smi.txt' % dataset)
    test = test.Input.drop_duplicates().sample(BATCH_SIZE * 10).values
    if method in ['gpt']:
        voc = utils.Voc('data/voc_smiles.txt', src_len=100, trg_len=100)
    else:
        voc = utils.VocSmiles('data/voc_smiles.txt', max_len=100)
    data_in = voc.encode([seq.split(' ') for seq in data.Input.values])
    data_out = voc.encode([seq.split(' ') for seq in data.Output.values])
    data_set = TensorDataset(data_in, data_out)
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)

    test_set = voc.encode([seq.split(' ') for seq in test])
    test_set = utils.TgtData(test_set, ix=[voc.decode(seq, is_tk=False) for seq in test_set])
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, collate_fn=test_set.collate_fn)

    pretrain(method=method)
    rl_train()
