from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit import rdBase
from rdkit.Chem import Draw
from metric import logP_mw, pca, properties, converage, diversity, training_process
import seaborn as sns
from sklearn import metrics
from matplotlib import pyplot as plt

plt.switch_backend('Agg')
rdBase.DisableLog('rdApp.error')


def fig4():
    pair = ['label', 'score']
    legends = ['NB', 'RF', 'KNN', 'SVM', 'DNN']
    fnames = ['output/NB_cls_qsar.cv.txt', 'output/RF_cls_qsar.cv.txt',
              'output/KNN_cls_qsar.cv.txt', 'output/SVM_cls_qsar.cv.txt',
              'output/DNN_cls_qsar.cv.txt']
    preds = []
    for fname in fnames:
        df = pd.read_csv(fname)
        preds.append(df[pair].value)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.subplots(121)
    lw = 1.5
    for i, pred in enumerate(preds):
        fpr, tpr, ths = metrics.roc_curve(pred[:, 0], pred[:, 1])
        auc = metrics.auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=lw, label=legends[i] + '(AUC=%.3f)' % auc)
    for i in range(1, 10):
        plt.plot([i * 0.1, i * 0.1], [0, 1], color='gray', lw=lw, linestyle='--')
        plt.plot([0, 1], [i * 0.1, i * 0.1], color='gray', lw=lw, linestyle='--')
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    ax1.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0], xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(loc="lower right")

    th = 0.5
    ax2 = fig.subplots(122)
    for j, pred in enumerate(preds):
        label, score = pred[:, 0], pred[:, 1]
        square = np.zeros((2, 2), dtype=int)
        for i, value in enumerate(score):
            row, col = int(label[i]), int(value > th)
            square[row, col] += 1
        mcc = metrics.matthews_corrcoef(label, score > th)
        sn = square[1, 1] / (square[1, 0] + square[1, 1])
        sp = square[0, 0] / (square[0, 0] + square[0, 1])
        acc = metrics.accuracy_score(label, score > th)
        ax2.bar(np.arange(4) + 0.17 * j, (mcc, sn, sp, acc), 0.17, alpha=0.8, label=legends[j])
    ax2.xticks(np.arange(4) + 0.34, ('MCC', 'Sensitivity', 'Specificity', 'Accuracy'))
    ax2.xlabel('Metric')
    ax2.ylim([0.0, 1.0])
    ax2.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig('Figure_5.tif', dpi=300)


def fig5():
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    valid, loss = training_process('output/net_p.log')
    ax1.plot(valid, label='SMILES validation rate')
    ax1.plot(loss, label='Value of Loss function')
    ax1.set_xlabel('Training Epochs')
    ax1.legend(loc='center right')
    ax1.set(ylim=(0, 1.0), xlim=(0, 1000))

    ax2 = fig.add_subplot(122)
    valid, loss = training_process('net_ex.log')
    ax2.plot([value for i, value in enumerate(valid) if i % 2 == 0], label='SMILES validation rate')
    ax2.plot([value for i, value in enumerate(loss / 100) if i % 2 == 0], label='Value of Loss function')
    ax2.set_xlabel('Training Epochs')
    ax2.legend(loc='center right')
    ax2.set(ylim=(0, 1.0), xlim=(0, 1000))
    fig.tight_layout()
    fig.savefig('Figure_5.tif', dpi=300)


def fig6():
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    sns.set(style="white", palette="pastel", color_codes=True)
    df = properties(['data/ZINC_B.txt', 'mol_p.txt'], ['ZINC Dataset', 'Pre-trained Model'])
    sns.violinplot(x='Property', y='Number', hue='Set', data=df, linewidth=1, split=True, bw=1)
    sns.despine(left=True)
    plt.ylim([0.0, 18.0])
    plt.xlabel('Structural Properties')

    plt.subplot(122)
    df = properties(['data/CHEMBL251.txt', 'mol_ex.txt'], ['A2AR Dataset', 'Fine-tuned Model'])
    sns.set(style="white", palette="pastel", color_codes=True)
    sns.violinplot(x='Property', y='Number', hue='Set', data=df, linewidth=1, split=True, bw=1)
    sns.despine(left=True)
    plt.ylim([0.0, 18.0])
    plt.xlabel('Structural Properties')
    plt.tight_layout()
    plt.savefig('Figure_6.tif', dpi=300)


def fig7():
    fig = plt.figure(figsize=(12, 12))
    lab = ['ZINC Dataset', 'Pre-trained Model']
    ax1 = fig.add_subplot(221)
    df = logP_mw(['data/ZINC_B.txt', 'mol_p.txt'])
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax1.scatter(group0.MWT, group0.LOGP, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax1.scatter(group1.MWT, group1.LOGP, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax1.set(ylabel='LogP', xlabel='Molecular Weight')
    ax1.legend(loc='lower right')

    ax2 = fig.add_subplot(222)
    df, ratio = pca(['data/ZINC_B.txt', 'mol_p.txt'])
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax2.scatter(group0.X, group0.Y, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax2.scatter(group1.X, group1.Y, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax2.set(ylabel='Principal Component 2 (%.2f%%)' % (ratio[1] * 100),
            xlabel='Principal Component 1 (%.2f%%)' % (ratio[0] * 100))
    ax2.legend(loc='lower right')

    lab = ['A2AR Dataset', 'Fine-tuned Model']
    ax3 = fig.add_subplot(223)
    df = logP_mw(['data/CHEMBL251.txt', 'mol_ex.txt'])
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax3.scatter(group0.MWT, group0.LOGP, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax3.scatter(group1.MWT, group1.LOGP, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax3.set(ylabel='LogP', xlabel='Molecular Weight', xlim=[0, 1000], ylim=[-5, 10])
    ax3.legend(loc='lower right')


    ax4 = fig.add_subplot(224)
    df, ratio = pca(['data/CHEMBL251.txt', 'mol_ex.txt'])
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax4.scatter(group0.X, group0.Y, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax4.scatter(group1.X, group1.Y, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax4.set(ylabel='Principal Component 2 (%.2f%%)' % (ratio[1] * 100),
            xlabel='Principal Component 1 (%.2f%%)' % (ratio[0] * 100),
            xlim=[-4, 5], ylim=[-4, 5])
    ax4.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('Figure_7.tif', dpi=300)


def fig8():
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    df = converage(log_paths)
    for i, label in enumerate(labels):
        ax1.plot(df[df.LABEL == i].SCORE.values, label=label)
    ax1.set(ylabel='Average Score', xlabel='Training Epochs')

    ax2 = fig.add_subplot(122)
    df = converage(log_paths1)
    for i, label in enumerate(labels):
        ax2.plot(df[df.LABEL == i].SCORE.values, label=label)
    ax2.set(ylabel='Average Score', xlabel='Training Epochs')
    fig.tight_layout()
    fig.savefig('Figure_8.tif', dpi=300)


def fig9():
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(211)
    sns.set(style="white", palette="pastel", color_codes=True)
    df = properties(mol_paths + real_path, labels + real_label, active=True)
    sns.violinplot(x='Property', y='Number', hue='Set', data=df, linewidth=1, bw=0.8)
    sns.despine(left=True)
    ax1.set(ylim=[0.0, 15.0], xlabel='Structural Properties')

    ax2 = fig.add_subplot(212)
    df = properties(mol_paths1 + real_path, labels + real_label, active=True)
    sns.set(style="white", palette="pastel", color_codes=True)
    sns.violinplot(x='Property', y='Number', hue='Set', data=df, linewidth=1, bw=0.8)
    sns.despine(left=True)
    ax2.set(ylim=[0.0, 15.0], xlabel='Structural Properties')
    fig.tight_layout()
    fig.savefig('Figure_9.tif', dpi=300)


def fig10():
    fig = plt.figure(figsize=(20, 10))
    legends = ['Active Ligands', 'DrugEx(Fine-tuned)', 'DrugEx(Pre-trained)', 'REINVENT', 'ORGANIC']
    df = logP_mw(real_path + ['mol_e_10_1_500x10.txt', 'mol_a_10_1_500x10.txt',
                              'output/sample_REINVENT.txt', 'mol_gan_5_0_500x10.txt'], active=True)
    group0 = df[df.LABEL == 0]
    for i in range(1, len(legends)):
        ax = fig.add_subplot(240 + i)
        group1 = df[df.LABEL == i]
        ax.scatter(group1.MWT, group1.LOGP, s=10, marker='o', label=legends[i], c='', edgecolor=colors[i])
        ax.scatter(group0.MWT, group0.LOGP, s=10, marker='o', label=legends[0], c='', edgecolor=colors[0])
        ax.set(ylabel='LogP', xlabel='Molecular Weight', xlim=[0, 1000], ylim=[-5, 10])
        ax.legend(loc='lower right')
    df, ratio = pca(real_path + ['mol_e_10_1_500x10.txt', 'mol_a_10_1_500x10.txt',
                                 'output/sample_REINVENT.txt', 'mol_gan_5_0_500x10.txt'], active=True)
    group0 = df[df.LABEL == 0]
    for i in range(1, len(legends)):
        ax = fig.add_subplot(244 + i)
        group1 = df[df.LABEL == i]
        ax.scatter(group1.X, group1.Y, s=10, marker='o', label=legends[i], c='', edgecolor=colors[i])
        ax.scatter(group0.X, group0.Y, s=10, marker='o', label=legends[0], c='', edgecolor=colors[0])
        ax.set(ylabel='Principal Component 2 (%.2f%%)' % (ratio[1] * 100),
               xlabel='Principal Component 1 (%.2f%%)' % (ratio[0] * 100),
               xlim=[-3, 6], ylim=[-3, 5])
        ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig('Figure_10.tif', dpi=300)


def fig11():
    dist = diversity('mol_e_10_1_500x10.txt', 'data/CHEMBL251.txt')
    dist.to_csv('distance.txt', index=None, sep='\t')

    df = pd.read_table('distance.txt')
    dists = [0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    scores = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i, dist in enumerate(dists):
        if i == len(dists) - 1: continue
        samples = df[(df.DIST > dist) & (df.DIST < dists[i + 1])].sort_values("SCORE")
        mols = []
        for j, score in enumerate(scores):
            if j == len(scores) - 1: continue
            sample = samples[(samples.SCORE > score) & (samples.SCORE < scores[j+1])]
            if len(sample) > 0:
                sample = sample.sample(1)
                print(sample.values)
                mols += [Chem.MolFromSmiles(smile) for smile in sample.CANONICAL_SMILES]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 300))
    img.save('Figure_11.tiff' % (dist))


def figS1():
    df = logP_mw(['mol_p.txt', 'output/sample_agent_without_ex.txt', 'data/CHEMBL251.txt'])
    labs = ['Pre-trained model', 'Reinforced model', 'Active Ligands']
    plt.figure(figsize=(6, 6))
    groups = df.groupby('LABEL')
    for i, group in groups:
        plt.scatter(group.MWT, group.LOGP, s=10, marker='o', label=labs[i], c='', edgecolor=colors[i])
    plt.ylabel('LogP')
    plt.xlabel('Molecular Weight')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('Figure_S1.tif', dpi=300)


def main():
    fig4()
    fig5()
    fig7()
    fig8()
    fig10()
    fig11()

    fig6()
    fig9()
    # Table 2
    # count(['mol_p.txt', 'data/CHEMBL251.txt'] +
    #       ['mol_a_10_1_500x10.txt', 'mol_e_10_1_500x10.txt', 'output/sample_REINVENT.txt', 'mol_gan_5_0_500x10.txt'])


if __name__ == '__main__':
    colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c', '#9467bd']  # orange, blue, green, red, purple
    pkg_paths = ['net_a_1_0_500x10.pkg', 'net_a_1_1_500x10.pkg',
                 'net_a_10_0_500x10.pkg', 'net_a_10_1_500x10.pkg']
    log_paths = ['net_a_1_0_500x10.log', 'net_a_1_1_500x10.log',
                 'net_a_10_0_500x10.log', 'net_a_10_1_500x10.log', ]
    mol_paths = ['mol_a_1_0_500x10.txt', 'mol_a_1_1_500x10.txt',
                 'mol_a_10_0_500x10.txt', 'mol_a_10_1_500x10.txt', ]
    pkg_paths1 = ['net_e_1_0_500x10.pkg', 'net_e_1_1_500x10.pkg',
                  'net_e_10_0_500x10.pkg', 'net_e_10_1_500x10.pkg']
    log_paths1 = ['net_e_1_0_500x10.log', 'net_e_1_1_500x10.log',
                  'net_e_10_0_500x10.log', 'net_e_10_1_500x10.log', ]
    mol_paths1 = ['mol_e_1_0_500x10.txt', 'mol_e_1_1_500x10.txt',
                  'mol_e_10_0_500x10.txt', 'mol_e_10_1_500x10.txt', ]
    labels = ["ε = 0.01, β = 0.0", "ε = 0.01, β = 0.1",
              "ε = 0.1, β = 0.0", "ε = 0.1, β = 0.1"]
    real_path = ['data/CHEMBL251.txt']
    real_label = ['Active Ligands']
    main()