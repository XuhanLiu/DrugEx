#!/usr/bin/env python
import os
import pandas as pd
from shutil import copy2
from models import rlearner
from models import generator
from models import classifier
from torch.utils.data import TensorDataset
import utils
import getopt
import sys
import time
import torch


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "a:e:b:g:c:s:z:")
    OPT = dict(opts)
    os.environ["CUDA_VISIBLE_DEVICES"] = OPT['-g'] if '-g' in OPT else "0"
    case = OPT['-c'] if '-c' in OPT else 'OBJ3'
    z = OPT['-z'] if '-z' in OPT else 'REG'
    alg = OPT['-a'] if '-a' in OPT else 'reinvent'
    scheme = OPT['-s'] if '-s' in OPT else 'WS'

    # construct the environment with three predictors
    keys = ['A1', 'A2A', 'ERG']
    A1 = utils.Predictor('output/env/RF_%s_CHEMBL226.pkg' % z, type=z)
    A2A = utils.Predictor('output/env/RF_%s_CHEMBL251.pkg' % z, type=z)
    ERG = utils.Predictor('output/env/RF_%s_CHEMBL240.pkg' % z, type=z)

    # Chose the desirability function
    objs = [A1, A2A, ERG]

    if scheme == 'WS':
        mod1 = utils.ClippedScore(lower_x=3, upper_x=10)
        mod2 = utils.ClippedScore(lower_x=10, upper_x=3)
        ths = [0.5] * 3
    else:
        mod1 = utils.ClippedScore(lower_x=3, upper_x=6.5)
        mod2 = utils.ClippedScore(lower_x=10, upper_x=6.5)
        ths = [0.99] * 3
    mods = [mod1, mod1, mod2] if case == 'OBJ3' else [mod2, mod1, mod2]
    env = utils.Env(objs=objs, mods=mods, keys=keys, ths=ths)

    root = 'output/%s_%s_%s_%s/'% (alg, case, scheme, time.strftime('%y%m%d_%H%M%S', time.localtime()))
    os.mkdir(root)
    copy2('models/rlearner.py', root)
    copy2('trainer.py', root)

    pr_path = 'output/lstm_chembl'
    ft_path = 'output/lstm_ligand'

    voc = utils.Voc(init_from_file="data/voc.txt")
    agent = generator.Generator(voc)
    agent.load_state_dict(torch.load(ft_path + '.pkg'))

    prior = generator.Generator(voc)
    prior.load_state_dict(torch.load(pr_path + '.pkg'))

    if alg == 'drugex':
        learner = rlearner.DrugEx(prior, env, agent)
    elif alg == 'organic':
        embed_dim = 128
        filter_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
        prior = classifier.Discriminator(agent.voc.size, embed_dim, filter_size, num_filters)
        df = pd.read_table('data/LIGAND_%s_%s.tsv' % (z, case))
        df = df[df.DESIRE == 1]
        data = voc.encode([voc.tokenize(s) for s in df.Smiles])
        data = TensorDataset(data)
        netD_path = root + 'netD'
        learner = rlearner.Organic(agent, env, prior, data)
        if not os.path.exists(netD_path + '.pkg'):
            learner.Train_dis_BCE(epochs=1000, out=netD_path)
        prior.load_state_dict(torch.load(netD_path + '.pkg'))
    elif alg == 'reinvent':
        learner = rlearner.Reinvent(agent, env, prior)
    else:
        crover = generator.Generator(voc)
        crover.load_state_dict(torch.load(ft_path + '.pkg'))
        learner = rlearner.Evolve(agent, env, prior, crover)

    learner.epsilon = learner.epsilon if '-e' not in OPT else float(OPT['-e'])
    learner.penalty = learner.penalty if '-b' not in OPT else float(OPT['-b'])
    learner.scheme = learner.scheme if scheme is None else scheme

    learner.out = root + '%s_%s_%s_%s' % (alg, learner.scheme, z, case)
    if alg in ['drugex', 'evolve']:
        learner.out += '_%.0e' % learner.epsilon
    learner.fit()
