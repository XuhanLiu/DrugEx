#!/usr/bin/env python
import torch
from rdkit import rdBase
from models import generator
import utils
import pandas as pd
from models import GPT2Model, GraphModel
from torch.utils.data import DataLoader
import getopt
import sys
import os


rdBase.DisableLog('rdApp.error')
torch.set_num_threads(1)
BATCH_SIZE = 1024


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "m:d:g:p:")
    OPT = dict(opts)
    # torch.cuda.set_device(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = OPT['-g'] if '-g' in OPT else "0, 1, 2, 3"
    method = OPT['-m'] if '-m' in OPT else 'atom'
    dataset = OPT['-d'] if '-d' in OPT else 'ligand_mf_brics'
    path = OPT['-p'] if '-p' in OPT else dataset
    utils.devices = [0]

    if method in ['gpt']:
        voc = utils.Voc('data/chembl_voc.txt', src_len=100, trg_len=100)
    else:
        voc = utils.VocSmiles('data/chembl_voc.txt', max_len=100)
    if method == 'ved':
        agent = generator.EncDec(voc, voc).to(utils.dev)
    elif method == 'attn':
        agent = generator.Seq2Seq(voc, voc).to(utils.dev)
    elif method == 'gpt':
        agent = GPT2Model(voc, n_layer=12).to(utils.dev)
    else:
        voc = utils.VocGraph('data/voc_atom.txt')
        agent = GraphModel(voc_trg=voc)

    for agent_path in ['benchmark/graph_PR_REG_OBJ1_0e+00.pkg', 'benchmark/graph_PR_REG_OBJ1_1e-01.pkg',
                       'benchmark/graph_PR_REG_OBJ1_1e-02.pkg', 'benchmark/graph_PR_REG_OBJ1_1e-03.pkg',
                       'benchmark/graph_PR_REG_OBJ1_1e-04.pkg', 'benchmark/graph_PR_REG_OBJ1_1e-05.pkg']:
        # agent_path = 'output/%s_%s_256.pkg' % (path, method)
        print(agent_path)
        agent.load_state_dict(torch.load(agent_path))

        z = 'REG'
        keys = ['A2A']
        A2A = utils.Predictor('output/env/RF_%s_CHEMBL251.pkg' % z, type=z)
        QED = utils.Property('QED')

        # Chose the desirability function
        objs = [A2A, QED]

        ths = [6.5, 0.0]

        env =  utils.Env(objs=objs, mods=None, keys=keys, ths=ths)
        if method in ['atom']:
            data = pd.read_table('data/ligand_mf_brics_test.txt')
            # data = data.sample(BATCH_SIZE * 10)
            data = torch.from_numpy(data.values).long().view(len(data), voc.max_len, -1)
            loader = DataLoader(data, batch_size=BATCH_SIZE)

            out = '%s.txt' % agent_path
        else:
            data = pd.read_table('data/%s_test.txt' % dataset).Input.drop_duplicates()
            # data = data.sample(BATCH_SIZE * 10)
            data = voc.encode([seq.split(' ')[:-1] for seq in data.values])
            loader = DataLoader(data, batch_size=BATCH_SIZE)

            out = agent_path + '.txt'
        frags, smiles, scores = agent.evaluate(loader, repeat=10, method=env)
        scores['Frags'] = frags
        scores['Smiles'] = smiles
        scores.to_csv(out, index=False, sep='\t')
