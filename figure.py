from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit import rdBase
from rdkit.Chem import Draw
from utils.metric import dimension
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt
from rdkit.Chem import Descriptors as desc

plt.switch_backend('Agg')
rdBase.DisableLog('rdApp.error')


def fig4(targets, out=None):
    plt.figure(figsize=(12, 4))
    for i, target in enumerate(targets):
        plt.subplot(131 + i)
        cate = ['cv', 'ind']
        for j, legend in enumerate(['Cross Validation', 'Independent Test']):
            df = pd.read_table('output/time_split/RF_REG_%s.%s.tsv' % (target, cate[j]))
            df = df[df.Label != 3.99]
            plt.scatter(df.Label, df.Score, s=5, label=legend)
        # plt.title(legends[i] + '(R^2=%.2f, RMSE=%.2f)' % (coef, rmse))
        plt.xlim([3.5, 10.5])
        plt.ylim([3.5, 10.5])
        plt.xlabel('Real pX')
        plt.ylabel('Predicted pX')
        plt.legend(loc='upper left')
    plt.tight_layout()
    if out is None:
        plt.show()
    else:
        plt.savefig(out, dpi=300)


def fig5(paths, out):
    fig = plt.figure(figsize=(12, 6))
    objs = ['SA', 'QED']
    z = 'REG'
    ix = 0
    keys, labels = [], []
    for i, case in enumerate(['OBJ3', 'OBJ1']):
        for j, s in enumerate(['PR', 'WS']):
            for k, obj in enumerate(objs):
                ix += 1
                ax = plt.subplot(240 + ix)
                plt.text(0.05, 0.9, chr(ord('A') + ix - 1), fontweight="bold", transform=ax.transAxes)
                for l, (key, dir) in enumerate(paths.items()):
                    if key == 'LIGAND':
                        fname = 'benchmark/old/%s_%s_%s.tsv' % (dir, z, case)
                    else:
                        fname = 'benchmark/old/%s_%s_%s_%s.tsv' % (dir, s, z, case)
                    df = pd.read_table(fname)
                    df = df[df.DESIRE == 1].drop_duplicates(subset='Smiles')
                    if obj in ['SA']:
                        data = df['SA'].values * 10
                        xx = np.linspace(0, 10, 1000)
                    else:
                        xx = np.linspace(0, 1, 1000)
                        data = df[obj].values
                    density = stats.gaussian_kde(data)(xx)
                    label = plt.plot(xx, density, c=colors[l])[0]
                    if i == 0 and j == 0 and k == 0:
                        keys.append(key)
                        labels.append(label)
        # ax.title(obj + ' Score')
    fig.legend(labels, keys, loc="lower center", ncol=len(keys))
    fig.savefig(out, dpi=300, bbox_inches = "tight", pil_kwargs={"compression": "tiff_lzw"})


def fig6(paths):
    fig = plt.figure(figsize=(12, 12))
    ix = 0
    keys, labels = [], []
    for i, case in enumerate(['OBJ3', 'OBJ1']):
        z = 'REG'
        out = 'benchmark/old/GPCR_%s_%s.tsv' % (z, case)
        df = pd.read_table(out)
        df = df[df.VALID == 1]
        df['LABEL'] = 'LIGAND'
        for j, s in enumerate(['PR', 'WS']):
            for k, (key, dir) in enumerate(paths.items()):
                out = 'benchmark/old/%s_%s_%s_%s.tsv' % (dir, s, z, case)
                sub = pd.read_table(out)
                sub = sub[sub.DESIRE == 1]
                sub['LABEL'] = key
                sub['SCHEME'] = s
                df = df.append(sub)
        df = dimension(df, alg='TSNE')
        # df.to_csv('t-SNE-%s.tsv' % case, sep='\t', index=False)
        df = pd.read_table('figure/t-SNE-%s.tsv' % case)
        # df, ratio = dimension(df, fp='ECFP')
        # df, ratio = dimension(df, fp='similarity', ref='LIGAND')
        group0 = df[df.LABEL == 'LIGAND']
        groupn = group0[group0.DESIRE == 1]
        for j, s in enumerate(['PR', 'WS']):
            for k, key in enumerate(['DrugEx v1', 'DrugEx v2', 'ORGANIC', 'REINVENT']):
                ix += 1
                ax = fig.add_subplot(4, 4, ix)
                plt.text(0.05, 0.9, chr(ord('A') + ix -1), fontweight="bold", transform=ax.transAxes)
                group1 = df[(df.LABEL == key) & (df.DESIRE == 1) & (df.SCHEME == s)]
                ax.scatter(group0.X, group0.Y, s=10, marker='o', c='', edgecolor=colors[0], label='Known Ligands')
                ax.scatter(group1.X, group1.Y, s=10, marker='o', c='', edgecolor=colors[k+1], label=key)
                ax.scatter(groupn.X, groupn.Y, s=5, marker='o', c='', edgecolor=colors[-1], label='Desired Ligands')
                # ax.set(ylabel='Principal Component 2 (%.2f%%)' % (ratio[1] * 100),
                #        xlabel='Principal Component 1 (%.2f%%)' % (ratio[0] * 100), xlim=[-5, 6], ylim=[-5, 6])
                ax.set(ylabel='Component 2', xlabel='Component 1', xlim=[-75, 75], ylim=[-75, 75])
                if i == 0 and j == 0:
                    handle, label = ax.get_legend_handles_labels()
                    if k == 0:
                        keys.extend(['Known Ligands', 'Desired Ligands'])
                        labels.extend([handle[0], handle[2]])
                    keys.append(key)
                    labels.append(handle[1])
    # fig.tight_layout()
    fig.legend(labels, keys, loc="lower center", ncol=len(keys), bbox_to_anchor=(0.45, 0.02))
    fig.subplots_adjust(wspace=0.35, hspace=0.35)
    fig.savefig('figure/fig_8.tif', dpi=300, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})


def fig9():
    for obj in ['OBJ1', 'OBJ3']:
        df = pd.read_table('benchmark/old/evolve_PR_REG_%s.tsv' % obj)
        df = df[df.DESIRE == 1].drop_duplicates(subset='Smiles')
        sub, mw = [], []
        ribose = Chem.MolFromSmiles('OC1COCC1O')
        for i, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row.Smiles)
                x, y = desc.MolWt(mol), mol.HasSubstructMatch(ribose)
                sub.append(y)
                mw.append(x)
            except:
                df = df.drop(i)
        df['Sub'], df['Mw'] = sub, mw
        df = df[(df.Sub == 0) & (df.Mw < 400)]
        for key in ['A1', 'A2A', 'ERG']:
            if key == 'ERG' or (key == 'A1' and obj == 'OBJ1'):
                df.loc[:, key] = 7 * (1 - df[key]) + 3
            else:
                df.loc[:, key] = 7 * df[key] + 3
        if obj == 'OBJ3':
            df.loc[:, 'x'] = df['A2A'] - df['ERG']
            df.loc[:, 'y'] = df['A1'] - df['ERG']
        else:
            df.loc[:, 'x'] = df['A2A'] - df['ERG']
            df.loc[:, 'y'] = df['A2A'] - df['A1']
        print(obj, min(df.x), max(df.x), min(df.y), max(df.y))
        mols = []
        si = [1, 1.5, 2, 2.5, 3, 3.5]
        sj = [1, 1.5, 2, 2.5, 3, 3.5]
        for i in si:
            for j in sj:
                sub = df[(df.x > i) & (df.x <= i+0.5) & (df.y > j) & (df.y <= j+0.5)]
                # sub = sub[sub.Sim == sub.Sim.min()]
                if len(sub) > 1:
                    sub = sub.sample(1)
                # print(sub.Sim.values[0])
                smile = sub.Smiles.values[0] if len(sub) ==1 else 'H'
                mols.append(Chem.MolFromSmiles(smile))
        img = Draw.MolsToGridImage(mols, molsPerRow=len(sj), subImgSize=(400, 300))
        img.save('figure/figure_9_%s.tif' % obj)


def figS1(paths, out):
    fig = plt.figure(figsize=(12, 6))
    objs = ['SA', 'QED']
    z = 'REG'
    ix = 0
    keys, labels = [], []
    for i, case in enumerate(['OBJ3', 'OBJ1']):
        for j, s in enumerate(['PR', 'WS']):
            for k, obj in enumerate(objs):
                ix += 1
                ax = plt.subplot(240 + ix)
                plt.text(0.05, 0.9, chr(ord('A') + ix - 1), fontweight="bold", transform=ax.transAxes)
                for l, (key, e) in enumerate(paths.items()):
                    if key == 'LIGAND':
                        fname = 'benchmark/old/%s_%s_%s.tsv' % (e, z, case)
                    else:
                        fname = 'benchmark/old/evolve_%s_%s_%s_%s.tsv' % (s, z, case, e)
                    df = pd.read_table(fname)
                    df = df[df.DESIRE == 1].drop_duplicates(subset='Smiles')
                    if obj in ['SA']:
                        data = df['SA'].values * 10
                        xx = np.linspace(0, 10, 1000)
                    else:
                        xx = np.linspace(0, 1, 1000)
                        data = df[obj].values
                    density = stats.gaussian_kde(data)(xx)
                    label = plt.plot(xx, density, c=colors[l])[0]
                    if i == 0 and j == 0 and k == 0:
                        keys.append(key)
                        labels.append(label)
                # ax.title(obj + ' Score')
    fig.legend(labels, keys, loc="lower center", ncol=len(keys))
    fig.savefig(out, dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})


def figS2(paths):
    plt.figure(figsize=(10, 10))
    z = 'REG'
    for i, case in enumerate(['OBJ3', 'OBJ1']):
        path = 'benchmark/old/GPCR_%s_%s.tsv' % (z, case)
        for j, s in enumerate(['PR', 'WS']):
            df = pd.read_table(path)
            df = df[df.DESIRE == 1]
            df.loc[:, 'Dataset'] = 'LIGAND'
            plt.subplot(221 + i * 2 + j)
            for key, dir in paths.items():
                out = 'benchmark/old/%s_%s_%s_%s.tsv' % (dir, s, z, case)
                sub = pd.read_table(out)
                sub = sub[sub.DESIRE == 1]
                sub.loc[:, 'Dataset'] = key
                df = df.append(sub)
            data = pd.DataFrame()
            for key, value in {'A1': 'A1AR', 'A2A': 'A2AAR', 'ERG': 'hERG'}.items():
                sub = df[['Dataset']]
                if key == 'ERG' or (key =='A1' and case == 'OBJ1'):
                    sub.loc[:, 'Score'] = 7 * (1 - df[key]) + 3
                else:
                    sub.loc[:, 'Score'] = 7 * df[key] + 3
                sub.loc[:, 'Prop'] = value
                data = data.append(sub)
            sns.set(style="white", palette="pastel", color_codes=True)
            sns.violinplot(x='Prop', y='Score', hue='Dataset', data=data, linewidth=1, bw=1)
            sns.despine(left=True)
            plt.ylim([3.0, 11.0])
            plt.xlabel('Distribution of pX for Each Target')
            plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('figure/Figure_S2.tif', dpi=300)


if __name__ == '__main__':
    colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#101010']  # orange, blue, green, red, purple
    paths = {'DrugEx v1': 'drugex', 'DrugEx v2': 'evolve',
             'ORGANIC': 'organic', 'REINVENT': 'reinvent'}
    fig4(['CHEMBL226', 'CHEMBL251', 'CHEMBL240'], 'figure/fig_4.tif')
    fig5({'LIGAND': 'GPCR', 'DrugEx v1': 'drugex', 'DrugEx v2': 'evolve',
          'ORGANIC': 'organic', 'REINVENT': 'reinvent'}, out='figure/figure_5.tif')
    fig6(paths)
    # Figure S1
    figS1({'LIGAND': 'GPCR', 'ε = 0e+00': '0e+00', 'ε = 1e-02': '1e-02',
          'ε = 1e-03': '1e-03', 'ε = 1e-04': '1e-04'}, out='figure/fig_S1.tif')
    fig9()
