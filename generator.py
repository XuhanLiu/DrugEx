import model
import util
import pandas as pd
import torch


def main(netG_path, out, size=20, netD_path='output/rf_dis.pkg'):
    batch_size = 500
    df = pd.DataFrame()
    voc = util.Voc(init_from_file="data/voc_b.txt")
    netG = model.Generator(voc)
    netG.load_state_dict(torch.load(netG_path))
    for _ in range(size):
        batch = pd.DataFrame()
        samples = netG.sample(batch_size)
        smiles, valids = util.check_smiles(samples, netG.voc)
        if netD_path:
            netD = util.Activity(netD_path)
            scores = netD(smiles)
            scores[valids == 0] = 0
            valids = scores
        batch['SCORE'] = valids
        batch['CANONICAL_SMILES'] = smiles
        df = df.append(batch)
    df.to_csv(out, sep='\t', index=None)


if __name__ == '__main__':
    main()
