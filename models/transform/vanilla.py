''' Define the Transformer model '''
import torch
import torch.nn as nn
from torch import optim
from .layer import EncoderLayer, DecoderLayer, PositionalEmbedding
from .layer import pad_mask, tri_mask
import utils
from models.generator import Base


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, voc, dropout=0., d_emb=512, n_layers=6, n_head=8, d_inner=1024,
                 d_model=512, pad_idx=0, has_seg=False):

        super().__init__()
        self.voc = voc
        self.d_emb = d_emb
        self.n_layers = 6
        self.n_head = n_head
        self.d_inner = d_inner
        self.d_model = d_model
        self.token_emb = nn.Embedding(voc.size, d_emb, padding_idx=pad_idx)
        self.posit_emb = PositionalEmbedding(d_emb, max_len=voc.max_len)
        self.has_seg = has_seg
        if has_seg:
            self.segmt_emb = nn.Embedding(3, d_emb, padding_idx=pad_idx)
        self.dropout = dropout
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_inner=d_inner, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, x, src_pad=None, src_atn=None, seg_ids=None):
        # -- Forward
        x = self.posit_emb(x) + self.token_emb(x)
        if self.has_seg:
            x += self.segmt_emb(seg_ids)

        for enc_layer in self.layer_stack:
            x = enc_layer(x, pad_mask=src_pad, attn_mask=src_atn)
        return x


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, voc, dropout=0., d_emb=512, n_layers=6, n_head=8, d_inner=1024,
                 d_model=512, pad_idx=0, ):

        super().__init__()

        self.token_emb = nn.Embedding(voc.size, d_emb, padding_idx=pad_idx)
        self.posit_emb = PositionalEmbedding(d_emb, max_len=voc.max_len)
        self.dropout = dropout
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_inner, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, x, mem, trg_mask=None, src_mask=None, atn_mask=None):

        x = self.posit_emb(x) + self.token_emb(x)

        for dec_layer in self.layer_stack:
            x = dec_layer(x, mem, trg_mask=trg_mask, src_mask=src_mask, atn_mask=atn_mask)
        return x


class Transformer(Base):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, voc_src, voc_trg, src_pad=0, trg_pad=0, d_inner=1024,
                 d_emb=512, d_model=512, n_layers=6, n_head=8, dropout=0.1,
                 emb_weight_sharing=True, prj_weight_sharing=False):

        super().__init__()
        self.src_pad, self.trg_pad = src_pad, trg_pad
        self.voc_src, self.voc_trg = voc_src, voc_trg
        self.encoder = Encoder(voc=voc_src, d_emb=d_emb, d_model=d_model,
                               n_layers=n_layers, n_head=n_head, d_inner=d_inner,
                               pad_idx=src_pad, dropout=dropout)

        self.decoder = Decoder(voc=voc_trg, d_emb=d_emb, d_model=d_model,
                               n_layers=n_layers, n_head=n_head, d_inner=d_inner,
                               pad_idx=trg_pad, dropout=dropout)
        self.prj_word = nn.Linear(d_model, voc_trg.size, bias=False)
        self.init_states()

        self.x_logit_scale = 1.
        if prj_weight_sharing:
            self.prj_word.weight = self.decoder.token_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_weight_sharing:
            self.encoder.token_emb.weight = self.decoder.token_emb.weight

        self.optim = utils.ScheduledOptim(
            optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 2.0, d_model)

    def forward(self, src, trg=None):
        src_pad = pad_mask(src, self.src_pad)
        src_atn = tri_mask(src)
        bsz = len(src)
        src = src.transpose(0, 1)
        mem = self.encoder(src, src_pad, src_atn=src_atn)
        x = torch.LongTensor([[self.voc_trg.tk2ix['GO']]] * bsz).to(utils.dev)
        if trg is not None:
            input = torch.cat([x, trg[:, :-1]], dim=1)
            trg_pad = pad_mask(input, self.trg_pad)
            trg_atn = tri_mask(input)
            input = input.transpose(0, 1)
            dec = self.decoder(input, mem, trg_pad, src_pad, trg_atn)
            dec = self.prj_word(dec) * self.x_logit_scale
            dec = dec.transpose(0, 1).log_softmax(dim=-1)
            out = dec.gather(2, trg.unsqueeze(2)).squeeze(2)
        else:
            out = torch.zeros(bsz, self.voc_trg.max_len).long().to(utils.dev)
            out = torch.cat([x, out], dim=1)
            isEnd = torch.zeros(bsz).bool().to(utils.dev)
            for step in range(1, self.voc_trg.max_len+1):  # decode up to max length
                trg_pad = pad_mask(out[:, :step])
                trg_atn = tri_mask(out[:, :step])
                dec = out[:, :step].transpose(0, 1)
                dec = self.decoder(dec, mem, trg_pad, src_pad, trg_atn)
                dec = self.prj_word(dec) * self.x_logit_scale
                dec = dec.softmax(dim=-1)[-1, :, :]
                dec = dec.multinomial(1).view(-1)
                isEnd |= dec == self.voc_trg.tk2ix['EOS']
                dec[isEnd] = self.voc_trg.tk2ix['_']
                out[:, step] = dec.code
                if isEnd.all(): break
        return out
