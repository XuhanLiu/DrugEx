import numpy as np
import torch as T
from torch import nn
from torch import optim
from torch.nn import functional as F
import time
import util
import configparser
import json


class Base(nn.Module):
    def fit(self, loader_train, loader_valid, out, epochs=100, lr=1e-3):
        if 'optim' in self.__dict__:
            optimizer = self.optim
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = np.inf
        last_save = 0
        log = open(out + '.log', 'w')
        for epoch in range(epochs):
            t0 = time.time()
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (1 - 1 / epochs) ** (epoch * 10)
            for i, (Xb, yb) in enumerate(loader_train):
                Xb, yb = util.Variable(Xb), util.Variable(yb)
                optimizer.zero_grad()
                y_ = self.forward(Xb, istrain=True)
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                loss = self.criterion(y_, yb)
                loss.backward()
                optimizer.step()
            loss_valid = self.evaluate(loader_valid)
            print('[Epoch: %d/%d] %.1fs loss_train: %f loss_valid: %f' % (
                    epoch, epochs, time.time() - t0, loss.data[0], loss_valid), file=log)
            if loss_valid < best_loss:
                T.save(self.state_dict(), out + '.pkg')
                print('[Performance] loss_valid is improved from %f to %f, Save model to %s' %
                      (best_loss, loss_valid, out + '.pkg'), file=log)
                best_loss = loss_valid
                last_save = epoch
            else:
                print('[Performance] loss_valid is not improved.', file=log)
                if epoch - last_save > 100: break
        log.close()
        self.load_state_dict(T.load(out + '.pkg'))

    def evaluate(self, loader):
        loss = 0
        for Xb, yb in loader:
            Xb, yb = util.Variable(Xb), util.Variable(yb)
            y_ = self.forward(Xb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            loss += self.criterion(y_, yb).data[0]
        return loss / len(loader)

    def predict(self, loader):
        score = []
        for Xb, yb in loader:
            Xb = util.Variable(Xb)
            y_ = self.forward(Xb)
            score.append(y_.cpu().data)
        return T.cat(score, dim=0).numpy()

    def BCELoss(self, inp, target):
        criterion = nn.BCELoss()
        out = self(inp)
        return criterion(out, target)


class STClassifier(Base):
    def __init__(self, n_dim, n_class):
        super(STClassifier, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, n_class)
        if n_class == 1:
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.activation = nn.Softmax()
        util.cuda(self)

    def forward(self, X, istrain=False):
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        y = self.activation(self.fc3(y))
        return y


class STRegressor(Base):
    def __init__(self, n_dim, n_class):
        super(STRegressor, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, n_class)
        self.criterion = nn.MSELoss()
        util.cuda(self)

    def forward(self, X, istrain=False):
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        y = self.fc3(y)
        return y


class MTClassifier(Base):
    def __init__(self, n_dim, n_task):
        super(MTClassifier, self).__init__()
        self.n_task = n_task
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.fc1 = nn.Linear(8000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.output = nn.Linear(2000, n_task)
        self.criterion = nn.BCELoss()
        self.activation = nn.Sigmoid()
        util.cuda(self)

    def forward(self, X, istrain=False):
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc2(y))
        if istrain:
            y = self.dropout(y)
        y = self.activation(self.output(y))
        return y


class MTRegressor(Base):
    def __init__(self, n_dim, n_task):
        super(MTRegressor, self).__init__()
        self.n_task = n_task
        self.dropout = nn.Dropout(0.25)
        self.fc0 = nn.Linear(n_dim, 8000)
        self.bn0 = nn.BatchNorm1d(8000)
        self.fc1 = nn.Linear(8000, 4000)
        # self.fc2 = nn.Linear(4000, 2000)
        self.output = nn.Linear(4000, n_task)
        self.criterion = nn.MSELoss()
        util.cuda(self)

    def forward(self, X, istrain=False):
        y = F.relu(self.fc0(X))
        if istrain:
            y = self.dropout(y)
        y = F.relu(self.fc1(y))
        if istrain:
            y = self.dropout(y)
        # y = F.sigmoid(self.fc2(y))
        # if istrain: y = self.dropout(y)
        y = self.output(y)
        return y


class Generator(nn.Module):
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc, lr=0.001):
        self.lr = lr
        super(Generator, self).__init__()
        self.voc = voc
        self.embedding = nn.Embedding(voc.size, 128)
        self.gru_1 = nn.GRUCell(128, 512)
        self.gru_2 = nn.GRUCell(512, 512)
        self.gru_3 = nn.GRUCell(512, 512)
        self.linear = nn.Linear(512, voc.size)
        self.proba = nn.Softmax()
        self.score = nn.LogSoftmax()
        self.criterion = nn.NLLLoss(size_average=False)
        util.cuda(self)
        self.optim = T.optim.Adam(self.parameters())

    def reset_optim(self):
        self.optim = T.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x, h):
        x = self.embedding(x)
        h_out = util.Variable(T.zeros(h.size()))
        x = h_out[0] = self.gru_1(x, h[0])
        x = h_out[1] = self.gru_2(x, h[1])
        x = h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, batch_size):
        return util.Variable(T.zeros(3, batch_size, 512))

    def likelihood(self, target):
        batch_size, seq_len = target.size()
        start_token = T.LongTensor(batch_size, 1).fill_(self.voc.vocab['GO'])
        end_token = T.LongTensor(batch_size, 1).fill_(self.voc.vocab['EOS'])
        x = T.cat([util.Variable(start_token), util.Variable(target[:, :-1]), util.Variable(end_token)], dim=1)
        h = self.init_h(batch_size)
        scores = util.Variable(T.zeros(batch_size, seq_len))
        hiddens = util.Variable(T.zeros(3, batch_size, seq_len, 512))
        for step in range(seq_len):
            logits, h = self(x[:, step], h)
            hiddens[:, :, step, :] = h
            score = self.score(logits)
            score = T.gather(score, 1, x[:, step+1:step+2]).squeeze()
            scores[:, step] = score
        return scores, hiddens

    def PGLoss(self, score, seq, reward):
        seq = util.Variable(seq.unsqueeze(-1))
        seq.require_grad = False
        loss = score * util.Variable(reward)
        return -loss.mean()

    def sample(self, batch_size, cutoff=0.01, inits=None, explore=None):
        if inits is None:
            start_token = T.LongTensor(batch_size).fill_(self.voc.vocab['GO'])
            h = self.init_h(batch_size)
            h1 = self.init_h(batch_size)
            x = start_token
            start = 0
        else:
            x, h, start, h1 = inits
        x = util.Variable(x)
        sequences = util.cuda(T.zeros(batch_size, self.voc.max_len-start).long())
        isEnd = util.cuda(T.zeros(batch_size).byte())

        for step in range(self.voc.max_len-start):
            logit, h = self(x, h)
            if explore:
                logit1, h1 = explore(x, h1)
                loc = util.Variable(T.rand(batch_size, 1) < cutoff).expand(logit.size())
                logit[loc] = logit1[loc]
            proba = self.proba(logit)
            x = T.multinomial(proba).view(-1)
            x[isEnd] = self.voc.vocab['EOS']
            sequences[:, step] = x.data

            end_token = (x == self.voc.vocab['EOS']).data
            isEnd = T.ge(isEnd + end_token, 1)
            if (isEnd == 1).all(): break
        return sequences


class Discriminator(Base):
    """A CNN for text classification
    architecture: Embedding >> Convolution >> Max-pooling >> Softmax
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

        util.cuda(self)
        self.optim = T.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        """
        Args:
            x: (batch_size * seq_len)
        """
        emb = self.emb(x).unsqueeze(1)  # batch_size * 1 * seq_len * emb_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = T.cat(pools, 1)  # batch_size * num_filters_sum
        highway = self.highway(pred)
        pred = F.sigmoid(highway) * F.relu(highway) + (1. - F.sigmoid(highway)) * pred
        pred = self.sigmoid(self.lin(self.dropout(pred)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)