# -*- coding:utf-8 -*-

from preprocessing import Tokenizer

import random
import csv
import json

import numpy as np
import sentencepiece as spm
from konlpy.tag import Okt
import torch
from torch.utils.data import Dataset, DataLoader


class BertLMDataset(Dataset):
    def __init__(self, dataset, tokenizer: Tokenizer, vocab_size=5000):
        self.tokenizer = tokenizer

        # 데이터 로딩
        with open(dataset, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 데이터 전처리 (str to int)
        for i, d in enumerate(self.data):
            self.data[i]['content'] = tokenizer.tokens_to_ids(d['content'])

        # masking을 위한 토큰 클래스 로딩
        self.total_tokens = tokenizer.get_tokens(vocab_prefix=f'vocab_{vocab_size}', for_masking=True)

    def __getitem__(self, item):
        tokens = self.data[item]['content']
        masked_tokens, candi_index, answers = self._masking(tokens)
        masked_tokens = torch.LongTensor(masked_tokens)

        mask = np.zeros_like(masked_tokens)
        mask[candi_index] = 1  # ex) [0, 1, 1, 0, 0, 1, ...]
        mask = torch.from_numpy(mask).long()

        sparse_answers = np.zeros_like(masked_tokens)
        sparse_answers[candi_index] = answers  # ex) [0, 32, 5, 0, 0, 12, ...]
        sparse_answers = torch.from_numpy(sparse_answers).long()

        return masked_tokens, mask, sparse_answers

    def _masking(self, tokens):
        sep_idx = tokens.index(self.tokenizer.token_to_id('[SEP]'))
        t_tokens = tokens[1:sep_idx]

        k = int(len(t_tokens) * 0.15)
        candi_index = list(range(1, len(t_tokens)+1))  # CLS를 제외했기 때문에 +1
        random.shuffle(candi_index)
        candi_index = candi_index[:k]

        random_token_index = candi_index[:int(k * 0.1)]  # 랜덤 마스킹
        # correct_token_index = candi_index[int(k * 0.1):int(k * 0.2)]  # 정답 마스킹
        mask_token_index = candi_index[int(k * 0.2):]  # 마스크토큰 마스킹

        masked_tokens = np.array(tokens)
        answers = masked_tokens[candi_index]  # MASK에 해당하는 라벨 토큰
        for idx in random_token_index:
            masked_tokens[idx] = self.tokenizer.token_to_id(random.choice(self.total_tokens))
        masked_tokens[mask_token_index] = self.tokenizer.token_to_id('[MASK]')

        return masked_tokens, candi_index, answers

    def __len__(self):
        return len(self.data)


class BertClsDataset(Dataset):
    def __init__(self, dataset, tokenizer: Tokenizer, max_num_seq=20, inference=False, vocab_size=5000, is_train=True):
        self.max_num_seq = max_num_seq
        self.inference = inference
        self.is_train = is_train
        self.tokenizer = tokenizer
        self.total_tokens = tokenizer.get_tokens(vocab_prefix=f'vocab_{vocab_size}', for_masking=True)

        # 데이터 로딩
        with open(dataset, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 데이터 전처리 (str to int)
        for i, d in enumerate(self.data):
            doc = d['content']
            n_doc = []
            for sub_doc in doc:
                n_doc.append(self.tokenizer.tokens_to_ids(sub_doc))
                # n_doc.append(list(map(self.tokenizer.PieceToId, sub_doc.split())))
            self.data[i]['content'] = n_doc

    def __getitem__(self, item):
        doc = self.data[item]['content']
        if not self.inference and len(doc) > self.max_num_seq:  # 문장 수가 많으면 일부 문장만 선택
            sp = random.choice(list(range(len(doc) - self.max_num_seq)))
            doc = doc[sp:sp + self.max_num_seq]
        if self.is_train:
            for i, sub_doc in enumerate(doc):  ##
                doc[i] = self._masking(sub_doc, mask_rate=0.3)
        doc = torch.LongTensor(doc)
        label = self.data[item]['label']

        return doc, label

    def _masking(self, tokens, mask_rate=0.1):
        sep_idx = list(tokens).index(self.tokenizer.token_to_id('[SEP]'))
        t_tokens = tokens[1:sep_idx]

        k = int(len(t_tokens) * mask_rate)
        candi_index = list(range(1, len(t_tokens)+1))  # CLS를 제외했기 때문에 +1
        random.shuffle(candi_index)
        candi_index = candi_index[:k]

        random_token_index = candi_index[:int(k * 0.2)]  # 랜덤 마스킹
        mask_token_index = candi_index[int(k * 0.8):]  # UNK 마스킹

        masked_tokens = np.array(tokens)
        for idx in random_token_index:
            masked_tokens[idx] = self.tokenizer.token_to_id(random.choice(self.total_tokens))
        masked_tokens[mask_token_index] = self.tokenizer.token_to_id('[UNK]')

        return masked_tokens

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = BertClsDataset('bertcls_val_v5000_t128.json')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (doc, label) in enumerate(data_loader):
        print(doc.shape)
        print(doc)
        print(label)
        if i > 0:
            break