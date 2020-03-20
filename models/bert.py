# -*- coding:utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bert(nn.Module):
    def __init__(self, d_hidden, n_heads, n_layers, n_groups, vocab_size, max_num_tokens):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_groups = n_groups

        self.embedding = Embedding(vocab_size=vocab_size, dims=d_hidden, max_num_tokens=max_num_tokens, dropout=0.1)

        assert n_layers % n_groups == 0 and n_layers >= n_groups
        self.transformers = nn.ModuleList(
            [Transformer(d_hidden, n_heads, d_hidden * 4, dropout=0.1) for _ in range(n_groups)])

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x)
        #x = self.expand_fc(x)  # 차원 확장

        for i in range(self.n_layers):  # 각 그룹에 해당하는 Bert 적용
            x = self.transformers[i // (self.n_layers // self.n_groups)].forward(x, mask)
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size, dims, max_num_tokens, dropout):
        super().__init__()
        self.dims = dims
        self.pe = self.position_embedding(max_num_tokens, dims)  # position embedding
        self.te = nn.Embedding(vocab_size, dims, padding_idx=0)  # token embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # pe = self.position_embedding(x.shape[-1], self.dims).to(x.device)
        x = self.te(x) + self.pe.to(x.device)
        return self.dropout(x)

    def position_embedding(self, max_num_tokens, dims):
        pe = torch.zeros(max_num_tokens, dims)
        pe.require_grad = False

        position = torch.arange(0, max_num_tokens).float().unsqueeze(1)
        div_term = (torch.arange(0, dims, 2).float() * -(math.log(10000.0) / dims)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        return pe


class Transformer(nn.Module):
    def __init__(self, d_hidden, n_heads, d_fc, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(n_heads, d_hidden)
        self.feed_forward = nn.Sequential(
                                nn.Linear(d_hidden, d_fc),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_fc, d_hidden)
                            )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.layer_norms[0](x)
        att = self.attention(x, x, x, mask=mask)
        x = x + self.dropout(att)

        x = self.layer_norms[1](x)
        att = self.feed_forward(x)
        x = x + self.dropout(att)

        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_hidden, dropout=0.1):
        super().__init__()
        assert d_hidden % n_heads == 0

        self.d_k = d_hidden // n_heads
        self.h = n_heads

        self.projection = nn.ModuleList([nn.Linear(d_hidden, d_hidden) for _ in range(3)])
        self.output_linear = nn.Linear(d_hidden, d_hidden)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.projection, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

    def attention(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
