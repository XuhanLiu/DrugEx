#!/usr/bin/env python
import torch
import models
import utils
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import time
import numpy as np




class PGLearner(object):
    """ Reinforcement learning framework with policy gradient. This class is the base structure for all
        policy gradient-based  deep reinforcement learning models.

    Arguments:

        agent (models.Generator): The agent which generates the desired molecules

        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.

        prior: The auxiliary model which is defined differently in each methods.
    """
    def __init__(self, agent, env, prior=None, memory=None, mean_func='geometric'):
        self.replay = 10
        self.agent = agent
        self.prior = prior
        self.batch_size = 64  # * 4
        self.n_samples = 128  # * 8
        self.env = env
        self.epsilon = 1e-3
        self.penalty = 0
        self.scheme = 'PR'
        self.out = None
        self.memory = memory
        # mean_func: which function to use for averaging: 'arithmetic' or 'geometric'
        self.mean_func = mean_func

    def policy_gradient(self):
        pass

    def fit(self):
        best = 0
        last_save = 0
        log = open(self.out + '.log', 'w')
        for epoch in range(1000):
            print('\n----------\nEPOCH %d\n----------' % epoch)
            self.policy_gradient()
            seqs = self.agent.sample(self.n_samples)
            ix = utils.unique(seqs)
            smiles = [self.agent.voc.decode(s) for s in seqs[ix]]
            scores = self.env(smiles, is_smiles=True)

            desire = (scores.DESIRE).sum() / self.n_samples
            score = scores[self.env.keys].values.mean()
            valid = scores.VALID.mean()

            if best <= score:
                torch.save(self.agent.state_dict(), self.out + '.pkg')
                best = score
                last_save = epoch

            print("Epoch: %d average: %.4f valid: %.4f unique: %.4f" %
                  (epoch, score, valid, desire), file=log)
            for i, smile in enumerate(smiles):
                score = "\t".join(['%0.3f' % s for s in scores.values[i]])
                print('%s\t%s' % (score, smile), file=log)
            if epoch - last_save > 100:
                break
        for param_group in self.agent.optim.param_groups:
            param_group['lr'] *= (1 - 0.01)
        log.close()


class Reinvent(PGLearner):
    """ REINVENT algorithm

    Reference: Olivecrona, M., Blaschke, T., Engkvist, O. et al. Molecular de-novo design
               through deep reinforcement learning. J Cheminform 9, 48 (2017).
               https://doi.org/10.1186/s13321-017-0235-x

    Arguments:

        agent (models.Generator): The agent network which is constructed by deep learning model
                                   and generates the desired molecules.

        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.

        prior (models.Generator): The prior network which is constructed by deep learning model
                                   and ensure the agent to generate molecules with correct grammar.
    """
    def __init__(self, agent, env, prior, epsilon=60, beta=0.5):
        super(Reinvent, self).__init__(agent, env, prior)
        for param in self.prior.parameters():
            param.requires_grad = False
        # self.agent.optim.lr = 0.0005
        self.epsilon = epsilon
        self.beta = beta

    def policy_gradient(self):
        seqs = []
        for _ in range(self.replay):
            seq = self.agent.sample(self.batch_size)
            seqs.append(seq)
        seqs = torch.cat(seqs, dim=0)
        ix = utils.unique(seqs)
        seqs = seqs[ix]
        smiles = [self.agent.voc.decode(s) for s in seqs]

        scores = self.env.calc_reward(smiles, self.scheme)[:, 0]
        ds = TensorDataset(seqs, torch.Tensor(scores-self.beta).to(utils.dev))
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)

        for seq, score in loader:
            # Calculate gradients and make an update to the network weights
            self.agent.optim.zero_grad()
            prior_likelihood = self.prior.likelihood(seq).sum(dim=1)
            agent_likelihood = self.agent.likelihood(seq).sum(dim=1)
            augmented_likelihood = prior_likelihood + self.epsilon * score
            loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # Calculate loss
            loss = loss.mean()

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = - (1 / agent_likelihood).mean()
            loss += 5 * 1e3 * loss_p
            loss.backward()
            self.agent.optim.step()


class DrugEx(PGLearner):
    """ DrugEx algorithm (version 1.0)

    Reference: Liu, X., Ye, K., van Vlijmen, H.W.T. et al. An exploration strategy improves the diversity
               of de novo ligands using deep reinforcement learning: a case for the adenosine A2A receptor.
               J Cheminform 11, 35 (2019).
               https://doi.org/10.1186/s13321-019-0355-6

    Arguments:

        agent (models.Generator): The agent network which is constructed by deep learning model
                                   and generates the desired molecules.

        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.

        prior (models.Generator): The pre-trained network which is constructed by deep learning model
                                   and ensure the agent to explore the approriate chemical space.
    """
    def __init__(self, agent, env, prior=None, memory=None):
        super(DrugEx, self).__init__(agent, env, prior, memory=memory)

    def policy_gradient(self):
        seqs = []
        for _ in range(self.replay):
            seq = self.agent.evolve1(self.batch_size, epsilon=self.epsilon, mutate=self.prior)
            seqs.append(seq)
        seqs = torch.cat(seqs, dim=0)
        ix = utils.unique(seqs)
        seqs = seqs[ix]
        smiles = [self.agent.voc.decode(s) for s in seqs]

        scores = self.env.calc_reward(smiles, self.scheme)
        ds = TensorDataset(seqs, torch.Tensor(scores).to(utils.dev))
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)
        self.agent.PGLoss(loader)


class Organic(PGLearner):
    """ ORGANIC algorithm

    Reference: Sanchez-Lengeling B, Outeiral C, Guimaraes GL, Aspuru-Guzik A (2017)
               Optimizing distributions over molecular space. An Objective-Reinforced
               Generative Adversarial Network for Inverse-design Chemistry (ORGANIC)
               https://doi.org/10.26434/chemrxiv.5309668.v3

    Arguments:

        agent (models.Generator): The agent network which is constructed by deep learning model
                                   and generates the desired molecules.

        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.

        prior (models.Generator): The discriminator which is constrcuted by deep learning model and
                                   judge if the generated molecule is similar to the real molecule.
    """
    def __init__(self, agent, env, prior, real=None, memory=None):
        super(Organic, self).__init__(agent, env, prior, memory=memory)
        self.epsilon = 0.5
        self.loader = DataLoader(real, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                 collate_fn=real.collate_fn)
        for param_group in self.prior.optim.param_groups:
            param_group['lr'] = 1e-3

    def policy_gradient(self):
        seqs = []
        for _ in range(self.replay):
            seq = self.agent.sample(self.batch_size)
            seqs.append(seq)
        seqs = torch.cat(seqs, dim=0)
        ix = utils.unique(seqs)
        seqs = seqs[ix]
        smiles = [self.agent.voc.decode(s) for s in seqs]

        scores = self.env.calc_reward(smiles, self.scheme)
        scores = torch.Tensor(scores).to(utils.dev)
        scores = self.epsilon * scores + (1 - self.epsilon) * self.prior(seqs).data

        ds = TensorDataset(seqs, scores)
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)
        self.agent.PGLoss(loader)
        self.Train_dis_BCE(epochs=1)

    def Train_dis_BCE(self, epochs=1, out=None):
        best_loss = np.Inf
        criterion = nn.BCELoss()
        log = open(out + '.log', 'w') if out is not None else None
        last_save = 0
        for epoch in range(epochs):
            t0 = time.time()
            for i, real in enumerate(self.loader):
                size = len(real)
                fake = self.agent.sample(size)
                data = torch.cat([fake, real.to(utils.dev)], dim=0)
                label = torch.cat([torch.zeros(size, 1), torch.ones(size, 1)]).to(utils.dev)
                self.prior.optim.zero_grad()
                label_ = self.prior(data)
                loss = criterion(label_, label)
                loss.backward()
                self.prior.optim.step()
                if i % 10 == 0 and i != 0:
                    for param_group in self.prior.optim.param_groups:
                        param_group['lr'] *= (1 - 0.03)
            if out is None: continue
            print('[Epoch: %d/%d] %.1fs loss: %f' % (
                epoch, epochs, time.time() - t0, loss.item()), file=log)
            if loss.item() < best_loss:
                print('[Performance] loss is improved from %f to %f, Save model to %s' %
                      (best_loss, loss, out + '.pkg'), file=log)
                torch.save(self.prior.state_dict(), out + ".pkg")
                best_loss = loss.item()
                last_save = epoch
            else:
                print('[Performance] loss  is not improved.', file=log)
            if epoch - last_save > 100:
                break
        return loss.item()


class Evolve(PGLearner):
    """ DrugEx algorithm (version 2.0)

    Reference: Liu, X., Ye, K., van Vlijmen, H.W.T. et al. DrugEx v2: De Novo Design of Drug Molecule by
               Pareto-based Multi-Objective Reinforcement Learning in Polypharmacology.
               J Cheminform (2021). https://doi.org/10.1186/s13321-019-0355-6

    Arguments:

        agent (models.Generator): The agent network which is constructed by deep learning model
                                   and generates the desired molecules.

        env (utils.Env): The environment which provides the reward and judge
                                 if the genrated molecule is valid and desired.

        prior (models.Generator): The pre-trained network which is constructed by deep learning model
                                   and ensure the agent to explore the approriate chemical space.
    """
    def __init__(self, agent, env, prior=None, crover=None, mean_func='geometric', memory=None):
        super(Evolve, self).__init__(agent, env, prior, mean_func=mean_func, memory=memory)
        self.crover = crover

    def policy_gradient(self, crover=None, memory=None, epsilon=None):
        seqs = []
        start = time.time()
        for _ in range(self.replay):
            seq = self.agent.evolve1(self.batch_size, epsilon=epsilon, crover=crover, mutate=self.prior)
            seqs.append(seq)
        t1 = time.time()
        seqs = torch.cat(seqs, dim=0)
        if memory is not None:
            mems = [memory, seqs]
            seqs = torch.cat(mems)
        smiles = np.array([self.agent.voc.decode(s) for s in seqs])
        # smiles = np.array(utils.canonicalize_list(smiles))
        ix = utils.unique(np.array([[s] for s in smiles]))
        smiles = smiles[ix]
        seqs = seqs[torch.LongTensor(ix).to(utils.dev)]

        scores = self.env.calc_reward(smiles, self.scheme)
        if memory is not None:
            scores[:len(memory), 0] = 1
            ix = scores[:, 0].argsort()[-self.batch_size * 4:]
            seqs, scores = seqs[ix, :], scores[ix, :]
        t2 = time.time()
        ds = TensorDataset(seqs, torch.Tensor(scores).to(utils.dev))
        loader = DataLoader(ds, batch_size=self.n_samples, shuffle=True)

        self.agent.PGLoss(loader)
        t3 = time.time()
        print(t1 - start, t2-t1, t3-t2)

    def fit(self):
        best = 0
        log = open(self.out + '.log', 'w')
        last_smiles = []
        last_scores = []
        interval = 250
        last_save = -1

        for epoch in range(10000):
            print('\n----------\nEPOCH %d\n----------' % epoch)
            if epoch < interval and self.memory is not None:
                self.policy_gradient(crover=None, memory=self.memory, epsilon=1e-1)
            else:
                self.policy_gradient(crover=self.crover, epsilon=self.epsilon)
            seqs = self.agent.sample(self.n_samples)
            smiles = [self.agent.voc.decode(s) for s in seqs]
            smiles = np.array(utils.canonicalize_list(smiles))
            ix = utils.unique(np.array([[s] for s in smiles]))
            smiles = smiles[ix]
            scores = self.env(smiles, is_smiles=True)

            desire = (scores.DESIRE).sum() / self.n_samples
            if self.mean_func == 'arithmetic':
                score = scores[self.env.keys].values.sum() / self.n_samples / len(self.env.keys)
            else:
                score = scores[self.env.keys].values.prod(axis=1) ** (1.0 / len(self.env.keys))
                score = score.sum() / self.n_samples
            valid = scores.VALID.sum() / self.n_samples

            print("Epoch: %d average: %.4f valid: %.4f unique: %.4f" %
                  (epoch, score, valid, desire), file=log)
            if best < score:
                torch.save(self.agent.state_dict(), self.out + '.pkg')
                best = score
                last_smiles = smiles
                last_scores = scores
                last_save = epoch

            if epoch % interval == 0 and epoch != 0:
                for i, smile in enumerate(last_smiles):
                    score = "\t".join(['%.3f' % s for s in last_scores.values[i]])
                    print('%s\t%s' % (score, smile), file=log)
                self.agent.load_state_dict(torch.load(self.out + '.pkg'))
                self.crover.load_state_dict(torch.load(self.out + '.pkg'))
            if epoch - last_save > interval: break
        log.close()