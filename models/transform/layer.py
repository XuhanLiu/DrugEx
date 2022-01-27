''' Define the Layers '''
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import math


def pad_mask(seq, pad_idx=0):
    return seq == pad_idx


def tri_mask(seq, diag=1):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    masks = torch.ones((len_s, len_s)).triu(diagonal=diag)
    masks = masks.bool().to(seq.device)
    return masks


def sparse_to_dense(idx, val):
    edge = []
    for i, ix in enumerate(idx):
        adj = torch.sparse.Tensor(ix, val[i]).to_dense()
        edge.append(adj)
    edge = torch.stack(edge)
    edge = edge.to(val.device)
    return edge


def lapacian(src):
    src_edge = torch.ones(len(src), 64, 64).to(src.device)
    for i in range(len(src)):
        edge = torch.sparse_coo_tensor(src[i, 1:-1, :], src[i, -1, :], size=(64, 64))
        src_edge[i, 1:, 1:] = edge.to_dense()[:-1, :-1]

    D = src_edge.sum(dim=-1).pow(-0.5).diag_embed()
    L = src_edge.sum(dim=-1).diag_embed() - src_edge
    _, _, v = (D @ L.float() @ D).svd()
    return v


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise

    def forward(self, x):
        y = self.w_1(x).relu()
        y = self.w_2(y)
        return y


class SublayerConnection(nn.Module):
    """
    a residual connection followed by a layer norm
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ Apply residual connection to any sublayer with the same size"""
        y = sublayer(x)
        y = self.dropout(y)
        y = self.norm(x + y)
        return y


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len=100, batch_first=False):
        super(PositionalEmbedding, self).__init__()

        self.batch_first = batch_first
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            y = self.pe[:x.size(1), :].unsqueeze(0).detach()
        else:
            y = self.pe[:x.size(0), :].unsqueeze(1).detach()
        return y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len=100, batch_first=False):
        super(PositionalEncoding, self).__init__()

        self.batch_first = batch_first
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        bsize, sqlen = x.size()
        y = x.reshape(bsize * sqlen)
        code = self.pe[y, :].view(bsize, sqlen, -1)
        # if sqlen > 10:
        #     assert code[10, 5, 8] == self.pe[x[10, 5], 8]
        return code


class EncoderLayer(nn.Module):
    """ Compose with two layers """
    def __init__(self, d_model, n_head, d_inner, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.pffn = PositionwiseFeedForward(d_model, d_inner)
        self.connector = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.connector[0](x, lambda x: self.attn(x, x, x, pad_mask, attn_mask=attn_mask)[0])
        x = self.connector[1](x, self.pffn)
        return x


class DecoderLayer(nn.Module):
    """ Compose with three layers """
    def __init__(self, d_model, n_head, d_inner, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.enc_attn = MultiheadAttention(d_model, n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner)
        self.connector = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, mem, trg_mask=None, src_mask=None, atn_mask=None):
        x = self.connector[0](x, lambda x: self.slf_attn(x, x, x, trg_mask, attn_mask=atn_mask)[0])
        x = self.connector[1](x, lambda x: self.enc_attn(x, mem, mem, src_mask)[0])
        x = self.connector[2](x, self.pos_ffn)
        return x