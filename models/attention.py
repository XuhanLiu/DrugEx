import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn = self.main(encoder_outputs.contiguous().view(-1, self.h_dim))  # (b, s, h) -> (b * s, 1)
        attn = F.softmax(attn.contiguous().view(b_size, -1), dim=1).unsqueeze(2)  # (b*s, 1) -> (b, s, 1)
        return attn


class Attention(nn.Module):
    def __init__(self, hidden_size, method='general'):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size, bias=False)

        elif self.method in 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size, bias=False)

        elif self.method == 'percept':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, memory):
        seq_len = memory.size(1)
        if self.method == 'dot':
            energy = memory
        elif self.method == 'general':
            energy = self.attn(memory)
        elif self.method == 'concat':
            h = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            energy = self.attn(torch.cat([h, memory], 2))
            hidden = torch.eye(self.hidden_size)
        elif self.method == 'percept':
            h = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            energy = self.attn(torch.cat([h, memory], 2)).tanh()
            hidden = self.v.repeat(memory.size(0), 1)  # [B*1*H]
        energy = torch.bmm(energy, hidden.unsqueeze(2))  # [B*T]
        attn_w = F.softmax(energy.transpose(1, 2), dim=-1)
        return attn_w


class DecoderAttn(nn.Module):
    def __init__(self, voc, embed_size, hidden_size, n_layers=3, is_lstm=True):
        super(DecoderAttn, self).__init__()
        self.voc = voc
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = voc.size
        self.n_layers = n_layers
        self.is_lstm = is_lstm
        rnn_layer = nn.LSTM if is_lstm else nn.GRU
        self.embed = nn.Embedding(voc.size, embed_size)
        self.attention = Attention(hidden_size)
        self.rnn = rnn_layer(embed_size + hidden_size, hidden_size, num_layers=self.n_layers)
        self.out = nn.Linear(hidden_size, voc.size)

    def init_h(self, batch_size):
        h = torch.zeros(3, batch_size, self.hidden_size).to(utils.dev)
        if self.is_lstm:
            c = torch.zeros(3, batch_size, self.hidden_size).to(utils.dev)
        return (h, c) if self.is_lstm else h

    def forward(self, input, hc, memory):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input.unsqueeze(0)) # (1,B,N)
        # Calculate attention weights and apply to encoder outputs
        attn_w = self.attention(hc[0][0], memory)
        context = attn_w.bmm(memory)  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        output = torch.cat([embedded, context], 2)
        output, (h_out, c_out) = self.rnn(output, hc)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        # context = context.squeeze(0)
        # output = self.out(torch.cat([output, context], dim=1))
        output = self.out(output)
        return output, (h_out, c_out)
