# -*- coding:utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BertLM(nn.Module):
    def __init__(self, bert, n_hidden, vocab_size):
        super().__init__()
        self.bert = bert
        self.mask_lm = nn.Linear(n_hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x)
        out = self.mask_lm(x)

        return out


class BertClassifier(nn.Module):
    def __init__(self, bert, n_hidden, n_class):
        super().__init__()
        self.bert = bert
        self.rnn = nn.LSTM(n_hidden, n_hidden//2, 2, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Dropout(0.3),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_class),
        )

    def forward(self, xs):
        lengths = [len(x) for x in xs]
        bert_output = self.bert(torch.cat(xs, 0))[:, 0]

        features = []
        base_idx = 0
        for length in lengths:
            sample = bert_output[base_idx:base_idx+length]
            base_idx += length

            sample = sample.unsqueeze(0)
            out, _ = self.rnn(sample)
            out = sample + out

            out = self.classifier(out)
            # features.append(out.mean(dim=1))
            features.append(out.max(dim=1)[0])

            # out, _ = sample.max(dim=0)
            # features.append(out.unsqueeze(0))

        out = torch.cat(features, 0)
        #out = self.classifier(features)

        return out
