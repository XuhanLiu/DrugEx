import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import time
import utils


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
                Xb, yb = Xb.to(utils.dev), yb.to(utils.dev)
                optimizer.zero_grad()
                # predicted probability tensor
                y_ = self(Xb, istrain=True)
                # ignore all of the NaN values
                ix = yb == yb
                yb, y_ = yb[ix], y_[ix]
                wb = torch.Tensor(yb.size()).to(utils.dev)
                wb[yb == 3.99] = 0.1
                wb[yb != 3.99] = 1
                # loss function calculation based on predicted tensor and label tensor
                loss = self.criterion(y_ * wb, yb * wb)
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
            loader (torch.util.data.DataLoader): data loader for test set,
                including m X n target FloatTensor and l X n label FloatTensor
                (m is the No. of sample, n is the No. of features, l is the No. of classes or tasks)
        Return:
            loss (float): the average loss value based on the calculation of loss function with given test set.
        """
        loss = 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(utils.dev), yb.to(utils.dev)
            y_ = self.forward(Xb)
            ix = yb == yb
            yb, y_ = yb[ix], y_[ix]
            wb = torch.Tensor(yb.size()).to(utils.dev)
            wb[yb == 3.99] = 0.1
            wb[yb != 3.99] = 1
            loss += self.criterion(y_ * wb, yb * wb).item()
        loss = loss / len(loader)
        return loss

    def predict(self, loader):
        """Predicting the probability of each sample in the given dataset.
        Arguments:
            loader (torch.util.data.DataLoader): data loader for test set,
                only including m X n target FloatTensor
                (m is the No. of sample, n is the No. of features)
        Return:
            score (ndarray): probability of each sample in the given dataset,
                it is a m X l FloatTensor (m is the No. of sample, l is the No. of classes or tasks.)
        """
        score = []
        for Xb, yb in loader:
            Xb = Xb.to(utils.dev)
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
        self.fc0 = nn.Linear(n_dim, 4000)
        self.fc1 = nn.Linear(4000, 1000)
        # self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, n_class)
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
        self.to(utils.dev)

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
        # if istrain:
        #     y = self.dropout(y)
        # y = F.relu(self.fc2(y))
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
        self.fc0 = nn.Linear(n_dim, 4000)
        self.fc1 = nn.Linear(4000, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.output = nn.Linear(1000, n_task)
        self.is_reg = is_reg
        if is_reg:
            # loss function for regression
            self.criterion = nn.MSELoss()
        else:
            # loss function and activation function of output layer for multiple classification
            self.criterion = nn.BCELoss()
            self.activation = nn.Sigmoid()
        self.to(utils.dev)

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

        self.to(utils.dev)
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
        pred = highway.sigmoid() * F.relu(highway) + (1. - highway.sigmoid()) * pred
        pred = self.sigmoid(self.lin(self.dropout(pred)))
        return pred

    def init_parameters(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)