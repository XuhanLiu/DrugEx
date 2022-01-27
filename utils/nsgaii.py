import numpy as np
from rdkit import DataStructs
import torch
import utils


def dominate(ind1, ind2):
    all = np.all(ind1 <= ind2)
    any = np.any(ind1 < ind2)
    return all & any


def gpu_non_dominated_sort(swarm: torch.Tensor):
    domina = (swarm.unsqueeze(1) <= swarm.unsqueeze(0)).all(-1)
    domina_any = (swarm.unsqueeze(1) < swarm.unsqueeze(0)).any(-1)
    domina = (domina & domina_any).half()
    fronts = []
    while (domina.diag() == 0).any():
        count = domina.sum(dim=0)
        front = torch.where(count == 0)[0]
        fronts.append(front)
        domina[front, :] = 0
        domina[front, front] = -1
    return fronts


# Function to carry out NSGA-II's fast non dominated sort
def cpu_non_dominated_sort(population):
    domina = [[] for _ in range(len(population))]
    front = []
    count = np.zeros(len(population), dtype=int)
    ranks = np.zeros(len(population), dtype=int)
    for p, ind1 in enumerate(population):
        for q in range(p + 1, len(population)):
            ind2 = population[q]
            if dominate(ind1, ind2):
                    domina[p].append(q)
                    count[q] += 1
            elif dominate(ind2, ind1):
                domina[q].append(p)
                count[p] += 1
        if count[p] == 0:
            ranks[p] = 0
            front.append(p)

    fronts = [np.sort(front)]
    i = 0
    while len(fronts[i]) > 0:
        temp = []
        for f in fronts[i]:
            for d in domina[f]:
                count[d] -= 1
                if count[d] == 0:
                    ranks[d] = i + 1
                    temp.append(d)
        i = i + 1
        fronts.append(np.sort(temp))
    del fronts[len(fronts) - 1]
    return fronts


# Function to calculate crowding distance
def crowding_distance(population, front):
    distance = np.zeros(len(front))
    for i in range(population.shape[1]):
        rank = population[front, i].argsort()
        front = front[rank]
        distance[rank[0]] = 10 ** 4
        distance[rank[-1]] = 10 ** 4
        m_values = [population[j, i] for j in front]
        scale = max(m_values) - min(m_values)
        if scale == 0: scale = 1
        for j in range(1, len(front) - 1):
            distance[rank[j]] += (population[front[j+1], i] - population[front[j-1], i]) / scale
    return distance


def nsgaii_sort(array, is_gpu=False):
    if is_gpu:
        array = torch.Tensor(array).to(utils.dev)
        fronts = gpu_non_dominated_sort(array)
    else:
        fronts = cpu_non_dominated_sort(array)
    rank = []
    for i, front in enumerate(fronts):
        dist = crowding_distance(array, front)
        fronts[i] = front[np.argsort(dist)]
        rank.extend(fronts[i].tolist())
    return rank


def similarity_sort(array, fps, is_gpu=False):
    if is_gpu:
        array = torch.Tensor(array).to(utils.dev)
        fronts = gpu_non_dominated_sort(array)
    else:
        fronts = cpu_non_dominated_sort(array)
    rank = []
    for i, front in enumerate(fronts):
        fp = [fps[f] for f in front]
        if len(front) > 2 and None not in fp:
            dist = np.zeros(len(front))
            for j in range(len(front)):
                tanimoto = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp[j], fp))
                order = tanimoto.argsort()
                dist[order[0]] += 0
                dist[order[-1]] += 10 ** 4
                for k in range(1, len(order)-1):
                    dist[order[k]] += tanimoto[order[k+1]] - tanimoto[order[k-1]]
            fronts[i] = front[dist.argsort()]
        rank.extend(fronts[i].tolist())
    return rank