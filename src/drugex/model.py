#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file defined all of the architectures of deep neural networks (DNN)
that are used in this project similar to Scikit-learn style.

All of the DNN models are implemented by Pytorch ( >= version 1.0).
"""

import time

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from drugex import util


class Base(nn.Module):
    """ This class is the base structure for all of classification/regression DNN models.
    Mainly, it provides the general methods for training, evaluating model and predcting the given data.
    """

    def fit(self, train_loader, valid_loader, out, epochs=100, lr=1e-4):
        """Training the DNN model, similar to the scikit-learn or Keras style.
        In the end, the optimal value of parameters will also be persisted on the hard drive.
        
        Arguments:
            train_loader (DataLoader): Data loader for training set,
                including m X n target FloatTensor and m X l label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the No. of classes or tasks)
            valid_loader (DataLoader): Data loader for validation set.
                The data structure is as same as loader_train.
            out (str): the file path for the model file (suffix with '.pkg')
                and log file (suffix with '.log').
            epochs(int, optional): The maximum of training epochs (default: 100)
            lr (float, optional): learning rate (default: 1e-4)
        """

        if 'optim' in self.__dict__:
            optimizer = self.optim
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr)
        # record the minimum loss value based on the calculation of loss function by the current epoch
        best_loss = np.inf
        # record the epoch when optimal model is saved.
        last_save = 0
        log = open(out + '.log', 'w')
        for epoch in range(epochs):
            t0 = time.time()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (1 - 1 / epochs) ** (epoch * 10)
            for i, (Xb, yb) in enumerate(train_loader):
                # Batch of target tenor and label tensor
                Xb, yb = Xb.to(util.dev), yb.to(util.dev)
                optimizer.zero_grad()
                # predicted probability tensor
                y_ = self.forward(Xb, istrain=True)

                # ignore all of the NaN values
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                # loss function calculation based on predicted tensor and label tensor
                loss = self.criterion(y_, yb)
                loss.backward()
                optimizer.step()
            # loss value on validation set based on which optimal model is saved.
            loss_valid = self.evaluate(valid_loader)
            print('[Epoch: %d/%d] %.1fs loss_train: %f loss_valid: %f' % (
                    epoch, epochs, time.time() - t0, loss.item(), loss_valid), file=log)
            if loss_valid < best_loss:
                torch.save(self.state_dict(), out + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' %
                      (best_loss, loss_valid, out + '.pkg'), file=log)
                best_loss = loss_valid
                last_save = epoch
            else:
                print('[Performance] loss_valid is not improved.', file=log)
                # early stopping, if the performance on validation is not improved in 100 epochs.
                # The model training will stop in order to save time.
                if epoch - last_save > 100: break
        log.close()
        self.load_state_dict(torch.load(out + '.pkg'))

    def evaluate(self, loader):
        """Evaluating the performance of the DNN model.

        Arguments:
            loader (torch.utils.data.DataLoader): data loader for test set,
                including m X n target FloatTensor and l X n label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the No. of classes or tasks)

        Return:
            loss (float): the average loss value based on the calculation of loss function with given test set.
        """
        loss = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(util.dev), yb.to(util.dev)
            y_ = self.forward(Xb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += self.criterion(y_, yb).data[0]
        loss = loss / len(loader)
        return loss

    def predict(self, loader):
        """Predicting the probability of each sample in the given dataset.

        Arguments:
            loader (torch.utils.data.DataLoader): data loader for test set,
                only including m X n target FloatTensor
                (m is the No. of sample, n is the No. of features)

        Return:
            score (ndarray): probability of each sample in the given dataset,
                it is a m X l FloatTensor (m is the No. of sample, l is the No. of classes or tasks.)
        """
        score = []
        for Xb, yb in loader:
            Xb = Xb.to(util.dev)
            y_ = self.forward(Xb)
            score.append(y_.detach().cpu())
        score = torch.cat(score, dim=0).numpy()
        return score


class STFullyConnected(Base):
    """Single task DNN classification/regression model. It contains four fully connected layers between which
        are dropout layer for robustness.

    Arguments:
        n_dim (int): the No. of columns (features) for input tensor
        n_class (int): the No. of columns (classes) for output tensor.
        is_reg (bool, optional): Regression model (True) or Classification model (False)
    """
    def __init__(self, n_dim, n_class, is_reg=False):
        super(STFullyConnected, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, n_class)
        self.is_reg = is_reg
        if is_reg:
            # loss function for regression
            self.criterion = nn.MSELoss()
        elif n_class == 1:
            # loss function and activation function of output layer for binary classification
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        else:
            # loss function and activation function of output layer for multiple classification
            self.criterion = nn.CrossEntropyLoss()
            self.activation = nn.Softmax()
        self.to(util.dev)

    def forward(self, X, istrain=False):
        """Invoke the class directly as a function

        Arguments:
            X (FloatTensor): m X n FloatTensor, m is the No. of samples, n is the No. of features.
            istrain (bool, optional): is it invoked during training process (True) or just for prediction (False)

        Return:
            y (FloatTensor): m X l FloatTensor, m is the No. of samples, n is the No. of classes
        """
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        if self.is_reg:
            y = self.fc3(y)
        else:
            y = self.activation(self.fc3(y))
        return y


class MTFullyConnected(Base):
    """Multi-task DNN classification/regression model. It contains four fully connected layers
    between which are dropout layer for robustness.

    Arguments:
        n_dim (int): the No. of columns (features) for input tensor
        n_task (int): the No. of columns (tasks) for output tensor.
        is_reg (bool, optional): Regression model (True) or Classification model (False)
    """
    def __init__(self, n_dim, n_task, is_reg=False):
        super(MTFullyConnected, self).__init__()
        self.n_task = n_task
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.output = nn.Linear(2000, n_task)
        self.is_reg = is_reg
        if is_reg:
            # loss function for regression
            self.criterion = nn.MSELoss()
        else:
            # loss function and activation function of output layer for multiple classification
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        self.to(util.dev)

    def forward(self, X, istrain=False):
        """Invoke the class directly as a function

        Arguments:
            X (FloatTensor): m X n FloatTensor, m is the No. of samples, n is the No. of features.
            istrain (bool, optional): is it invoked during training process (True)
                or just for prediction (False)

        Return:
            y (FloatTensor): m X l FloatTensor, m is the No. of samples, n is the No. of tasks
        """
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        if self.is_reg:
            y = self.output(y)
        else:
            y = self.activation(self.output(y))
        return y


class Generator(nn.Module):
    """Recurrent neuroal networks based SMILES format molecule generator,
    this RNN model is used both for exploration and exploitation network
    that are involved in DrugEx training process. In the end, only
    exploitation network is used as agent for molecule design.

    input layer >> embedding layer >> recurrent layer >> output layer

    Arguments:
        voc (util.Vocabulay): pre-defined data structure of tokens used for
            SMILES construction.
        embed_size (int): the neuron units of embeding layers.
        hidden_size (int): the neron units of RNN hidden layers.
        is_lstm (bool): is LSTM (True) or GRU (False) used for implementation of RNN architecture
    """
    def __init__(self, voc, embed_size=128, hidden_size=512, is_lstm=True):
        super(Generator, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size
        self.is_lstm = is_lstm
        
        self.embed = nn.Embedding(voc.size, embed_size)
        if is_lstm:
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=3, batch_first=True)
        else:
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hidden_size, voc.size)
        self.optim = torch.optim.Adam(self.parameters())
        self.to(util.dev)

    def forward(self, input, h):
        """Invoke the instance of this class as an function directly.

        Arguments:
            input (LongTensor): The index of tokens at current position for each samples.
                It is (m, ) LongTensor, m is the No. of samples
            h (Tensor or tuple of Tensor): the output of hidden states from the previous step of RNN.
                if LSTM is used, it is a tuple and contains two FloatTensors, else if GRU is used
                it is just a FloatTensor.

        Returns:
            output: The probability of next tokens in vocabulary. It is (m, n) FloatTensor,
                m is the No. of sample, n is the size of vocabulary.
            h_out (Tensor or tuple of Tensor): the hidden states from the current step of RNN.
                if LSTM is used, it is a tuple and contains two FloatTensors, else if GRU is used.
                it is just a FloatTensor.

        """
        output = self.embed(input.unsqueeze(-1))
        output, h_out = self.rnn(output, h)
        output = self.linear(output).squeeze(1)
        return output, h_out

    def init_h(self, batch_size):
        """Initialize the hidden states for the Recurrent layers.

        Arguments:
            batch_size (int): the No. of sample for each batch iterated by data loader.

        Results:
            h (FloatTensor): the hidden states for the recurrent layers initialization.
            c (FloatTensor): the LSTM cell states for the recurrent layers initialization,
                this tensor is only for LSTM based recurrent layers.
        """
        h = torch.zeros(3, batch_size, self.hidden_size).to(util.dev)
        c = torch.zeros(3, batch_size, self.hidden_size).to(util.dev)
        return (h, c) if self.is_lstm else h

    def likelihood(self, target):
        """Calculating the probability for generating each token in each SMILES string

        Arguments:
            target (LongTensor): m X n LongTensor of SMILES string representation.
                m is the No. of samples, n is the maximum size of the tokens for
                the whole SMILES strings.

        Returns:
            scores (FloatTensor): m X n LongTensor of the probability for for generating
                each token in each SMILES string. m is the No. of samples, n is n is
                the maximum size of the tokens for the whole SMILES strings
        """
        batch_size, seq_len = target.size()
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(util.dev)
        h = self.init_h(batch_size)
        scores = torch.zeros(batch_size, seq_len).to(util.dev)
        for step in range(seq_len):
            logits, h = self(x, h)
            score = logits.log_softmax(dim=-1)
            score = score.gather(1, target[:, step:step+1]).squeeze()
            scores[:, step] = score
            x = target[:, step]
        return scores

    def PGLoss(self, score, reward):
        """Policy gradicent method for loss function construction under reinforcement learning

        Arguments:
            score (FloatTensor): m X n matrix of the probability for for generating
                each token in each SMILES string. m is the No. of samples,
                n is n is the maximum size of the tokens for the whole SMILES strings.
                In general, it is the output of likelihood methods. It requies gradient.
            reward (FloatTensor): if it is the final reward given by environment (predictor),
                it will be m X 1 matrix, the m is the No. of samples.
                if it is the step rewards obtained by Monte Carlo Tree Search based on environment,
                it will b m X n matrix, the m is the No. of samples, n is the No. of sequence length.
        Returns:
            loss (FloatTensor): The loss value calculated with REINFORCE loss function
                It requies gradient for parameter update.
        """
        loss = score * reward
        loss = -loss.mean()
        return loss

    def sample(self, batch_size, epsilon=0.01, explore=None):
        """Token selection based on the probability distribution for molecule generation

        Arguments:
            batch_size (int): the No. of sample to be generated for each time.
            epsilon (float, optional): the exploration rate, it determines the percentage of
                contribution the exploration network make.
            explore (Generator optional): exploration network, it has the same neural architecture with
                the exploitation network.

        Returns:
            sequences (LongTensor): m X n matrix that contains the index of tokens from vocaburary
                for SMILES sequence construction. m is the No. of samples, n is the maximum sequence
                length.
        """
        # Start tokens
        x = torch.LongTensor([self.voc.tk2ix['GO']] * batch_size).to(util.dev)
        # Hidden states initialization for exploitation network
        h = self.init_h(batch_size)
        # Hidden states initialization for exploration network
        h1 = self.init_h(batch_size)
        # Initialization of output matrix
        sequences = torch.zeros(batch_size, self.voc.max_len).long().to(util.dev)
        # labels to judge and record which sample is ended
        is_end = torch.zeros(batch_size).byte().to(util.dev)

        for step in range(self.voc.max_len):
            logit, h = self(x, h)
            if explore:
                logit1, h1 = explore(x, h1)
                loc = (torch.rand(batch_size, 1) < epsilon).expand(logit.size()).to(util.dev)
                logit[loc] = logit1[loc]
            proba = logit.softmax(dim=-1)
            # sampling based on output probability distribution
            x = torch.multinomial(proba, 1).view(-1)

            x[is_end] = self.voc.tk2ix['EOS']
            sequences[:, step] = x

            # Judging whether samples are end or not.
            end_token = (x == self.voc.tk2ix['EOS'])
            is_end = torch.ge(is_end + end_token, 1)
            #  If all of the samples generation being end, stop the sampling process
            if (is_end == 1).all(): break
        return sequences

    def fit(self, loader_train, out, loader_valid=None, epochs=100, lr=1e-3):
        """Training the RNN generative model, similar to the scikit-learn or Keras style.
        In the end, the optimal value of parameters will also be persisted on the hard drive.

        Arguments:
            loader_train (DataLoader): Data loader for training set, it contains
            Dataset with util.MolData; for each iteration, the output batch is
            m X n LongTensor, m is the No. of samples, n is the maximum length
            of sequences.
        out (str): the file path for the model file (suffix with '.pkg')
        valid_loader (DataLoader, optional): Data loader for validation set.
            The data structure is as same as loader_train.
            and log file (suffix with '.log').
        epochs(int, optional): The maximum of training epochs (default: 100)
        lr (float, optional): learning rate (default: 1e-4)
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        log = open(out + '.log', 'w')
        best_error = np.inf
        for epoch in range(epochs):
            for i, batch in enumerate(loader_train):
                optimizer.zero_grad()
                loss_train = self.likelihood(batch.to(util.dev))
                loss_train = -loss_train.mean()
                loss_train.backward()
                optimizer.step()
                # Performance Evaluation
                if i % 10 == 0 or loader_valid is not None:
                    # 1000 SMILES is sampled
                    seqs = self.sample(1000)
                    # ix = util.unique(seqs)
                    # seqs = seqs[ix]
                    # Checking the validation of each SMILES
                    smiles, valids = util.check_smiles(seqs, self.voc)
                    error = 1 - sum(valids) / len(seqs)
                    info = "Epoch: %d step: %d error_rate: %.3f loss_train: %.3f" % (epoch, i, error, loss_train.item())
                    # Saving the optimal parameter of the model with minimum loss value.
                    if loader_valid is not None:
                        # If the validation set is given, the loss function will be
                        # calculated on the validation set.
                        loss_valid, size = 0, 0
                        for j, batch in enumerate(loader_valid):
                            size += batch.size(0)
                            loss_valid += -self.likelihood(batch.to(util.dev)).sum()
                        print(size)
                        loss_valid = loss_valid / size / self.voc.max_len
                        if loss_valid.item() < best_error:
                            torch.save(self.state_dict(), out + '.pkg')
                            best_error = loss_valid.item()
                        info += ' loss_valid: %.3f' % loss_valid.item()
                    elif error < best_error:
                        # If the validation is not given, the loss function will be
                        # just based on the training set.
                        torch.save(self.state_dict(), out + '.pkg')
                        best_error = error
                    print(info, file=log)
                    for i, smile in enumerate(smiles):
                        print('%d\t%s' % (valids[i], smile), file=log)
        log.close()
        self.load_state_dict(torch.load(out + '.pkg'))


class Discriminator(Base):
    """A highway version of CNN for text classification
        architecture: Embedding >> Convolution >> Max-pooling >> Softmax
        refered to Pytorch version of SeqGAN in https://github.com/suragnair/seqGAN
    """

    def __init__(self, vocab_size, emb_dim, filter_sizes, num_filters, dropout=0.25):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, emb_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(sum(num_filters), sum(num_filters))
        self.dropout = nn.Dropout(p=dropout)
        self.lin = nn.Linear(sum(num_filters), 1)
        self.sigmoid = nn.Sigmoid()
        self.init_parameters()

        self.to(util.dev)
        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) * F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.sigmoid(self.lin(self.dropout(pred)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)
