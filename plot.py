from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit.Chem import Draw
from utils.metric import logP_mw, dimension
import seaborn as sns
from matplotlib_venn import venn3
from scipy import stats
from matplotlib import pyplot as plt


def figure3(out='Figure_3.tif'):
    fig = plt.figure(figsize=(8, 8))
    dataset = ['LIGAND+', 'LIGAND-', 'LIGAND0', 'ChEMBL']
    ax1 = fig.add_subplot(221)
    num = pd.DataFrame(columns=['Num', 'Set'])
    for ds in dataset:
        sub = pd.read_table('figures/%s_num.txt' % ds, dtype=float)
        sub['Set'] = ds
        num = num.append(sub)
    num = num.dropna()
    sns.set(style="white", palette="pastel", color_codes=True)
    sns.violinplot(x='Set', y='Num', data=num, order=dataset, linewidth=1.5, bw=0.8)
    plt.text(0.02, 0.95, chr(ord('A')), fontweight="bold", transform=ax1.transAxes)
    ax1.set(ylim=[0.0, 15.0], xlabel='Dataset', ylabel='Number of Fragments per Molecule')

    frags = []
    ax2 = fig.add_subplot(222)
    for ds in dataset:
        sub = pd.read_table('figures/%s_frag.txt' % ds)
        frag = set(sub['Frags'])
        frags.append(frag)
        sns.kdeplot(sub['MW'], shade=True, linewidth=1.5, label=ds)
        plt.text(0.02, 0.95, chr(ord('B')), fontweight="bold", transform=ax2.transAxes)
    ax2.set(xlabel='Molecular Weight', ylabel='Value')

    ax3 = fig.add_subplot(223)
    for ds in dataset:
        sub = np.loadtxt('figures/%s_div.txt' % ds)
        np.fill_diagonal(sub, np.NaN)
        sub = sub[sub == sub]
        sns.kdeplot(sub, shade=True, linewidth=1.5, label=ds)
    plt.text(0.02, 0.95, chr(ord('C')), fontweight="bold", transform=ax3.transAxes)
    ax3.set(xlim=[0.0, 0.5], xlabel='Tanimoto Similarity', ylabel='Value')

    ax4 = fig.add_subplot(224)
    venn3(frags[:-1], set_labels=dataset)
    plt.text(0.02, 0.95, chr(ord('D')), fontweight="bold", transform=ax4.transAxes)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.tight_layout()
    if out is None:
        plt.show()
    else:
        plt.savefig(out, dpi=600, bbox_inches = "tight", pil_kwargs={"compression": "tiff_lzw"})


def figure4():
    fnames = ['data/chembl_mf_brics_test.txt', 'benchmark/chembl_mix.txt']
    labels, keys = [], []
    fig = plt.figure(figsize=(12, 8))
    lab = ['ChEMBL Set', 'Pre-trained Model']

    ax1 = fig.add_subplot(231)
    df = logP_mw(fnames)
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    plt.text(0.05, 0.9, chr(ord('A')), fontweight="bold", transform=ax1.transAxes)
    ax1.scatter(group0.MWT, group0.LOGP, s=1, marker='o', label=lab[0], c='', edgecolor=colors[0])
    ax1.scatter(group1.MWT, group1.LOGP, s=10, marker='o', label=lab[1], c='', edgecolor=colors[1])
    ax1.set(ylabel='LogP', xlabel='Molecular Weight', xlim=[0, 1000], ylim=[-5, 10])
    handle, label = ax1.get_legend_handles_labels()
    labels.extend(handle)
    keys.extend(label)

    ax2 = fig.add_subplot(232)
    df, ratio = dimension(fnames, fp='physchem')
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    plt.text(0.05, 0.9, chr(ord('C')), fontweight="bold", transform=ax2.transAxes)
    ax2.scatter(group0.X, group0.Y, s=1, marker='o', label=lab[0], c='', edgecolor=colors[0])
    ax2.scatter(group1.X, group1.Y, s=10, marker='o', label=lab[1], c='', edgecolor=colors[1])
    ax2.set(ylabel='Principal Component 2 (%.2f%%)' % (ratio[1] * 100),
            xlabel='Principal Component 1 (%.2f%%)' % (ratio[0] * 100))

    ax3 = fig.add_subplot(233)
    # df, ratio = dimension(fnames, alg='TSNE')
    df = pd.read_table('t-SNE_pr.txt')
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    plt.text(0.05, 0.9, chr(ord('E')), fontweight="bold", transform=ax3.transAxes)
    ax3.scatter(group0.X, group0.Y, s=1, marker='o', label=lab[0], c='', edgecolor=colors[0])
    ax3.scatter(group1.X, group1.Y, s=10, marker='o', label=lab[1], c='', edgecolor=colors[1])
    ax3.set(ylabel='Component 2', xlabel='Component 1')

    fnames = ['data/ligand_mf_brics_test.txt', 'benchmark/ligand_mix.txt']
    lab = ['LIGAND Set', 'Fine-tuned Model']
    ax4 = fig.add_subplot(234)
    df = logP_mw(fnames)
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    plt.text(0.05, 0.9, chr(ord('B')), fontweight="bold", transform=ax4.transAxes)
    ax4.scatter(group0.MWT, group0.LOGP, s=10, marker='o', label=lab[0], c='', edgecolor=colors[2])
    ax4.scatter(group1.MWT, group1.LOGP, s=1, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax4.set(ylabel='LogP', xlabel='Molecular Weight', xlim=[0, 1000], ylim=[-5, 10])
    handle, label = ax4.get_legend_handles_labels()
    labels.extend(handle)
    keys.extend(label)

    ax5 = fig.add_subplot(235)
    df, ratio = dimension(fnames, fp='physchem')
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    plt.text(0.05, 0.9, chr(ord('D')), fontweight="bold", transform=ax5.transAxes)
    ax5.scatter(group0.X, group0.Y, s=10, marker='o', label=lab[0], c='', edgecolor=colors[2])
    ax5.scatter(group1.X, group1.Y, s=1, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax5.set(ylabel='Principal Component 2 (%.2f%%)' % (ratio[1] * 100),
            xlabel='Principal Component 1 (%.2f%%)' % (ratio[0] * 100))

    ax6 = fig.add_subplot(236)
    # df, ratio = dimension(fnames, alg='TSNE')
    df = pd.read_table('t-SNE_ft.txt')
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    plt.text(0.05, 0.9, chr(ord('F')), fontweight="bold", transform=ax6.transAxes)
    ax6.scatter(group0.X, group0.Y, s=10, marker='o', label=lab[0], c='', edgecolor=colors[2])
    ax6.scatter(group1.X, group1.Y, s=1, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax6.set(ylabel='Component 2', xlabel='Component 1')

    fig.legend(labels, keys, loc="lower center", ncol=len(keys), bbox_to_anchor=(0.45, 0.00))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    # plt.tight_layout()
    plt.savefig('Figure_4.tif', dpi=600, bbox_inches = "tight", pil_kwargs={"compression": "tiff_lzw"})


def figure5():
    fig = plt.figure(figsize=(8, 8))
    objs = ['QED', 'SA']
    ix = 0
    keys, labels = [], []
    methods = ['ved', 'attn', 'gpt', 'graph']
    for i, d in enumerate(['chembl', 'ligand']):
        labs = ['ChEMBL Set' if d == 'chembl' else 'LIGAND Set',
                  'LSTM-BASE', 'LSTM+ATTN', 'Sequence Transformer', 'Graph Transformer']
        dfs = {}
        fnames = ['benchmark/%s_set_qed_sa.txt' % d] + ['benchmark/%s_%s_qed_sa.txt' % (d, m) for m in methods]
        for j, fname in enumerate(fnames):
            dfs[labs[j]] = pd.read_table(fname)
        for k, obj in enumerate(objs):
            ix += 1
            ax = plt.subplot(220 + ix)
            plt.text(0.02, 0.9, chr(ord('A') + ix - 1), fontweight="bold", transform=ax.transAxes)
            for l, (key, df) in enumerate(dfs.items()):
                if obj in ['SA']:
                    xx = np.linspace(0, 10, 1000)
                else:
                    xx = np.linspace(0, 1, 1000)
                data = df[obj].values
                density = stats.gaussian_kde(data)(xx)
                if key in ['ChEMBL Set']:
                    color = colors[0]
                elif key in ['LIGAND Set']:
                    color = colors[1]
                else:
                    color = colors[l+1]
                label = plt.plot(xx, density, c=color)[0]
                if (i == 0 and k == 0) or (key in ['LIGAND Set'] and k == 0):
                    keys.append(key)
                    labels.append(label)
            # ax.title(obj + ' Score')
    fig.legend(labels, keys, loc="upper center", ncol=3, bbox_to_anchor=(0.45, 0.08))
    fig.subplots_adjust(wspace=0.35, hspace=0.35)
    fig.savefig('figure_5.tif', dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})


def figure6():
    fig = plt.figure(figsize=(12, 8))
    er = ['0e+00', '1e-01', '2e-01', '3e-01', '4e-01', '5e-01']
    ers = {'0e+00': '0.0', '1e-01': '0.1', '2e-01': '0.2', '3e-01': '0.3', '4e-01': '0.4', '5e-01': '0.5'}
    # df = dimension(['benchmark/ligand_rl_%s.txt' % e for e in ers], alg='TSNE')
    # df.to_csv('t-SNE.txt', index=False, sep='\t')
    df = pd.read_table('t-SNE.txt')
    for i, e in enumerate(er):
        group0 = df[df.LABEL == 0]
        # group0 = group0[group0['QED'] > 0.4]
        group1 = df[df.LABEL == i + 1]
        ax = fig.add_subplot(231 + i)
        plt.text(0.02, 0.9, chr(ord('A') + i), fontweight="bold", transform=ax.transAxes)
        ax.scatter(group1.X, group1.Y, s=10, marker='.', label='ε = %s' % ers[e], c='', edgecolor=colors[2])
        ax.scatter(group0.X, group0.Y, s=10, marker='o', label='LIGAND set', c='', edgecolor=colors[1])
        ax.set(ylabel='Component 2', xlabel='Component 1')
        ax.legend(loc='upper right')
    plt.savefig('Figure_6.tif', dpi=600, bbox_inches = "tight", pil_kwargs={"compression": "tiff_lzw"})


def figure7():
    df = pd.read_table('benchmark/ligand_rl_2e-01.txt')
    df = df[df.DESIRE == 1]
    subs = ['c1cocc1.n1cncnc1.n1c[nH]nc1',
            'c1cocc1.O=c1[nH]c(=O)c2nc[nH]c2[nH]1', 'c1cocc1.Nc1ncc2[nH]nnc2n1',
            'c1cocc1.n1cncnc1', 'c1cocc1.n1c[nH]nc1','n1cncnc1.n1c[nH]nc1']

    subset = {sub: [] for sub in subs}
    submol = {sub: Chem.MolFromSmiles(sub) for sub in subs}
    for smile in df.Smiles:
        if smile != smile: continue
        mol = Chem.MolFromSmiles(smile)
        for sub in subs:
            s = submol[sub]
            match = mol.HasSubstructMatch(s)
            if match:
                subset[sub].append(smile)
                break
    for i, sub in enumerate(subs[::-1]):
        if len(subset[sub]) > 120:
            mol = list(subset[sub])[:60]
        else:
            mol = list(subset[sub])
        mols = [Chem.MolFromSmiles(m) for m in mol]
        img = Draw.MolsToGridImage(mols, molsPerRow=6, subImgSize=(400, 300))
        img.save('figures/figure_6_%d.tif' % i)
        print(mol)


if __name__ == '__main__':
    colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c', '#9467bd', 'cyan']  # orange, blue, green, red, purple
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()